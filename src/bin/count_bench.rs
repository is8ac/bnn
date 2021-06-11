use bnn::bits::{b32, BitMap, BitPack, FromBool, IncrementCounters, PackedMap, BMA};
use bnn::count::ElementwiseAdd;
use bnn::ecc;
use bnn::shape::{LongDefault, Map, Pack, Shape, ZipMap};
use rand::distributions::{Distribution, Standard};
use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::any::type_name;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::ops::{Add, AddAssign};
use std::path::Path;
use std::sync::atomic::{compiler_fence, Ordering};
use std::time::Instant;

fn count_batch<S, T: FromBool>(strings: &[<S as BitPack<bool>>::T]) -> <S as Pack<T>>::T
where
    S: Pack<T> + BitPack<bool> + IncrementCounters<T>,
    <S as Pack<T>>::T: LongDefault,
{
    //assert!(strings.len() < T::MAX);
    strings
        .iter()
        .fold(<S as Pack<T>>::T::long_default(), |mut counts, bits| {
            <S as IncrementCounters<T>>::increment_in_place(&bits, &mut counts);
            counts
        })
}

fn split<I, O, T>(
    examples: &Vec<(<I as BitPack<bool>>::T, <O as BitPack<bool>>::T)>,
) -> <O as Pack<<I as BitPack<bool>>::T>>::T
where
    T: Copy + PartialOrd + Add<Output = T>,
    I: BitPack<bool>
        + Pack<T>
        + PackedMap<bool, bool>
        + ZipMap<T, T, bool>
        + IncrementCounters<T>
        + ZipMap<T, T, T>,
    O: BitPack<bool>
        + Pack<<I as BitPack<bool>>::T>
        + Pack<[(usize, <I as Pack<T>>::T); 2]>
        + Map<[(usize, <I as Pack<T>>::T); 2], <I as BitPack<bool>>::T>
        + ZipMap<
            [(usize, <I as Pack<T>>::T); 2],
            [(usize, <I as Pack<T>>::T); 2],
            [(usize, <I as Pack<T>>::T); 2],
        > + BitMap<bool, [(usize, <I as Pack<T>>::T); 2]>,
    <O as Pack<[(usize, <I as Pack<T>>::T); 2]>>::T: LongDefault + Sync + Send,
    <I as BitPack<bool>>::T: Sync,
    <O as BitPack<bool>>::T: Sync,
    [(); 2]: ZipMap<(usize, <I as Pack<T>>::T), (usize, <I as Pack<T>>::T), (usize, <I as Pack<T>>::T)>
        + Pack<(usize, <I as Pack<T>>::T), T = [(usize, <I as Pack<T>>::T); 2]>,
{
    let counts = examples
        .par_iter()
        .fold(
            || <O as Pack<[(usize, <I as Pack<T>>::T); 2]>>::T::long_default(),
            |mut acc, (input, target)| {
                <O as BitMap<bool, [(usize, <I as Pack<T>>::T); 2]>>::map_mut(
                    target,
                    &mut acc,
                    |sign, counters| {
                        <I as IncrementCounters<T>>::counted_increment_in_place(
                            input,
                            &mut counters[sign as usize],
                        );
                    },
                );
                acc
            },
        )
        .reduce_with(|a, b| {
            <O as ZipMap<
                [(usize, <I as Pack<T>>::T); 2],
                [(usize, <I as Pack<T>>::T); 2],
                [(usize, <I as Pack<T>>::T); 2],
            >>::zip_map(&a, &b, |a, b| {
                <[(); 2] as ZipMap<
                    (usize, <I as Pack<T>>::T),
                    (usize, <I as Pack<T>>::T),
                    (usize, <I as Pack<T>>::T),
                >>::zip_map(a, b, |a, b| {
                    (
                        a.0 + b.0,
                        <I as ZipMap<T, T, T>>::zip_map(&a.1, &b.1, |&a, &b| a + b),
                    )
                })
            })
        })
        .unwrap();

    <O as Map<[(usize, <I as Pack<T>>::T); 2], <I as BitPack<bool>>::T>>::map(&counts, |counts| {
        let bools = <I as ZipMap<T, T, bool>>::zip_map(&counts[0].1, &counts[1].1, |&a, &b| a < b);
        <I as PackedMap<bool, bool>>::map(&bools, |&sign| sign)
    })
}

fn lloyds<S, T: FromBool + Copy>(
    strings: &Vec<<S as BitPack<bool>>::T>,
    centers: Vec<<S as BitPack<bool>>::T>,
) -> Vec<<S as BitPack<bool>>::T>
where
    S: BitPack<bool> + Pack<T> + BMA<bool> + IncrementCounters<T> + PackedMap<T, bool>,
    <S as Pack<T>>::T: LongDefault + Send + Sync + ElementwiseAdd + std::fmt::Debug,
    <S as BitPack<bool>>::T: Send + Sync,
{
    //assert!(strings.len() < T::MAX);
    let counts: Vec<(usize, <S as Pack<T>>::T)> = strings
        .par_iter()
        .fold(
            || {
                centers
                    .iter()
                    .map(|_| (0, <S as Pack<T>>::T::long_default()))
                    .collect::<Vec<_>>()
            },
            |mut acc, string| {
                let closest_center = centers
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, center)| <S as BMA<bool>>::bma(center, string))
                    .unwrap()
                    .0;
                <S as IncrementCounters<T>>::counted_increment_in_place(
                    string,
                    &mut acc[closest_center],
                );
                acc
            },
        )
        .reduce_with(|mut a, b| {
            a.elementwise_add(&b);
            a
        })
        .unwrap();
    dbg!(&counts[0]);
    counts
        .iter()
        .map(|(n, counts)| {
            let threshold = (n / 2) as u64;
            <S as PackedMap<T, bool>>::map(counts, |&count| count.to_u64() > threshold)
        })
        .collect()
}

fn bench_count<T: FromBool, S, const V: usize>(n: usize, t: usize)
where
    <S as BitPack<bool>>::T: Sync,
    <S as Pack<T>>::T: Send + ElementwiseAdd + std::fmt::Debug + Eq,
    S: BitPack<bool> + Pack<T> + IncrementCounters<T> + Shape,
    <S as Pack<T>>::T: LongDefault,
    Standard: Distribution<<S as BitPack<bool>>::T>,
{
    compiler_fence(Ordering::SeqCst);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(t)
        .build()
        .unwrap();
    //rayon::ThreadPoolBuilder::new().num_threads(t).build_global().unwrap();
    let mut rng = Hc128Rng::seed_from_u64(0);
    let bits: Vec<_> = (0..n)
        .map(|_| {
            let bits: <S as BitPack<bool>>::T = rng.gen();
            bits
        })
        .collect();
    //let counts = <S as Pack<T>>::T::long_default();
    compiler_fence(Ordering::SeqCst);
    let start = Instant::now();
    compiler_fence(Ordering::SeqCst);
    let foo = count_batch::<S, T>(&bits);
    compiler_fence(Ordering::SeqCst);
    let elapsed_a = start.elapsed();
    compiler_fence(Ordering::SeqCst);

    compiler_fence(Ordering::SeqCst);
    let start = Instant::now();
    compiler_fence(Ordering::SeqCst);
    let results = pool.install(|| {
        bits.chunks(n / t)
            .par_bridge()
            .map(|strings_chunk| {
                //dbg!(strings_chunk.len());
                count_batch::<S, T>(strings_chunk)
            })
            .collect::<Vec<_>>()
    });
    let bar = results
        .iter()
        .fold(<S as Pack<T>>::T::long_default(), |mut a, b| {
            a.elementwise_add(&b);
            a
        });

    compiler_fence(Ordering::SeqCst);
    let elapsed_b = start.elapsed();
    compiler_fence(Ordering::SeqCst);

    assert_eq!(foo, bar);
    println!(
        "| {} | {} | {} | {} | {:?} | {:?} | {} | {} | {} |",
        V,
        S::N,
        type_name::<T>(),
        type_name::<S>(),
        elapsed_a,
        elapsed_a.as_nanos() as f64 / n as f64,
        (elapsed_a.as_nanos() as f64 / n as f64) / S::N as f64,
        (elapsed_b.as_nanos() as f64 / n as f64) / S::N as f64,
        (elapsed_b.as_nanos() as f64 / (n as f64 / t as f64)) / S::N as f64,
    );
    compiler_fence(Ordering::SeqCst);
}

type InputShape = [[[(); 32]; 8]; 2];
type OutputShape = [[(); 32]; 8];

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .stack_size(2usize.pow(31))
        .build_global()
        .unwrap();

    let file = File::open("tiny-shakespeare.txt").unwrap();
    //let file = File::open("target/release/count_bench").unwrap();
    let mut buf_reader = BufReader::new(file);
    let bytes: Vec<[b32; 8]> = buf_reader
        .bytes()
        .take(2usize.pow(20))
        .map(|x| {
            let x = x.unwrap();
            ecc::encode_byte(x)
        })
        .collect();

    /*
    let bytes: Vec<[b32; 8]> = (0..2000_u32)
        .map(|x| (x % 210) as u8)
        .chain((0..255))
        .map(|x| ecc::encode_byte(x))
        .collect();
    */

    let examples: Vec<_> = bytes.windows(3).map(|w| ([w[0], w[1]], w[2])).collect();

    let start = Instant::now();
    let weights: [[<InputShape as BitPack<bool>>::T; 32]; 8] =
        split::<InputShape, OutputShape, u32>(&examples);
    dbg!(&weights);
    dbg!(start.elapsed());

    let n_correct: u64 = examples
        .iter()
        .map(|(input, target)| {
            let output = <OutputShape as PackedMap<<InputShape as BitPack<bool>>::T, bool>>::map(
                &weights,
                |weights| {
                    <InputShape as BMA<bool>>::bma(weights, input) < (InputShape::N / 2) as u32
                },
            );
            //dbg!(&output);
            let byte = ecc::decode_byte(&output);
            let target_byte = ecc::decode_byte(&target);
            //dbg!(byte);
            //dbg!(target);
            (byte == target_byte) as u64
        })
        .sum();
    dbg!(n_correct);
    println!("{}", n_correct as f64 / bytes.len() as f64);
    /*
    let mut rng = Hc128Rng::seed_from_u64(0);
    let centers: Vec<<StringShape as BitPack<bool>>::T> = (0..256).map(|_| rng.gen()).collect();
    let strings: Vec<_> = (0..2usize.pow(20))
        .map(|_| {
            let bits: <StringShape as BitPack<bool>>::T = rng.gen();
            bits
        })
        .collect();
    dbg!();
    //let centers = lloyds::<StringShape>(&strings, centers);
    //dbg!(&centers[0]);
    let start = Instant::now();
    let centers = lloyds::<StringShape, u32>(&strings, centers);
    dbg!(start.elapsed());
    //dbg!(&centers[0]);
    //let centers = lloyds::<StringShape, u64>(&strings, centers);
    //dbg!(&centers[0]);

    println!("| version | n | counter | shape | real time | time per string | ns per bit st | ns per bit mt | per core |");
    println!("| - | - | - | - | - | - | - | - | - |");

    bench_count::<u32, [[[(); 32]; 32]; 32], 0>(2usize.pow(17), 16);
    bench_count::<u32, [[[(); 32]; 32]; 16], 0>(2usize.pow(18), 16);
    bench_count::<u32, [[[(); 32]; 32]; 8], 0>(2usize.pow(19), 16);
    bench_count::<u32, [[[(); 32]; 32]; 4], 0>(2usize.pow(20), 16);
    bench_count::<u32, [[[(); 32]; 32]; 2], 0>(2usize.pow(21), 16);
    bench_count::<u32, [[(); 32]; 32], 0>(2usize.pow(22), 16);
    bench_count::<u32, [[(); 32]; 16], 0>(2usize.pow(23), 16);
    bench_count::<u32, [[(); 32]; 8], 0>(2usize.pow(24), 16);
    bench_count::<u32, [[(); 32]; 4], 0>(2usize.pow(25), 16);
    bench_count::<u32, [[(); 32]; 2], 0>(2usize.pow(26), 16);
    bench_count::<u32, [[(); 32]; 1], 0>(2usize.pow(27), 16);
    bench_count::<u16, [[(); 32]; 1], 0>(2usize.pow(27), 16);
    bench_count::<u64, [[(); 32]; 1], 0>(2usize.pow(27), 16);
    bench_count::<u8, [[(); 32]; 1], 0>(2usize.pow(27), 16);
    */
}
