use bnn::bits::{
    b32, t32, BitMap, BitPack, FromBool, IncrementCounters, PackedMap, SIMDincrementCounters, BMA,
};
use bnn::count::ElementwiseAdd;
use bnn::ecc;
use bnn::shape::{IndexGet, IndexMap, LongDefault, Map, Pack, Shape, ZipMap};
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

fn word_transpose_weights_matrix<IWS, OS, IW>(
    input: &<IWS as Pack<<OS as Pack<[[u32; 32]; 2]>>::T>>::T,
    counts: &<OS as Pack<[u64; 2]>>::T,
) -> <OS as Pack<[(u64, <IWS as Pack<[u32; 32]>>::T); 2]>>::T
where
    OS: IndexGet<u64>
        + Pack<[u32; 2]>
        + Pack<[u64; 2]>
        + IndexGet<[u64; 2]>
        + IndexGet<[[u32; 32]; 2]>
        + Pack<[[u32; 32]; 2]>
        + Pack<[(u64, <IWS as Pack<[u32; 32]>>::T); 2]>
        + IndexMap<[(u64, <IWS as Pack<[u32; 32]>>::T); 2], ()>,
    IWS: Pack<<OS as Pack<[[u32; 32]; 2]>>::T>
        + Pack<[u32; 32]>
        + IndexMap<[u32; 32], ()>
        + IndexGet<<OS as Pack<[[u32; 32]; 2]>>::T>,
{
    <OS as IndexMap<[(u64, <IWS as Pack<[u32; 32]>>::T); 2], ()>>::index_map((), |o| {
        let counts = <OS as IndexGet<[u64; 2]>>::index_get(counts, o);
        [
            (
                counts[0],
                <IWS as IndexMap<[u32; 32], ()>>::index_map((), |iw| {
                    let slice =
                        <IWS as IndexGet<<OS as Pack<[[u32; 32]; 2]>>::T>>::index_get(input, iw);
                    <OS as IndexGet<[[u32; 32]; 2]>>::index_get(&slice, o)[0]
                }),
            ),
            (
                counts[1],
                <IWS as IndexMap<[u32; 32], ()>>::index_map((), |iw| {
                    let slice =
                        <IWS as IndexGet<<OS as Pack<[[u32; 32]; 2]>>::T>>::index_get(input, iw);
                    <OS as IndexGet<[[u32; 32]; 2]>>::index_get(&slice, o)[1]
                }),
            ),
        ]
    })
}

fn count_matrix_par<I, O>(
    examples: &[(<I as BitPack<bool>>::T, <O as BitPack<bool>>::T)],
) -> <O as Pack<[(u64, <I as Pack<u32>>::T); 2]>>::T
where
    I: BitPack<bool>
        + Pack<u32>
        + IncrementCounters<u32>
        + ZipMap<u32, u32, u32>
        + SIMDincrementCounters,
    O: BitPack<bool>
        + Pack<[(u64, <I as Pack<u32>>::T); 2]>
        + Map<(), [(u64, I::SIMDbyts); 2]>
        + Map<[(u64, I::SIMDbyts); 2], [(u64, <I as Pack<u32>>::T); 2]>
        + ZipMap<
            [(u64, <I as Pack<u32>>::T); 2],
            [(u64, <I as Pack<u32>>::T); 2],
            [(u64, <I as Pack<u32>>::T); 2],
        > + BitMap<bool, [(u64, I::SIMDbyts); 2]>,
    <O as Pack<[(u64, <I as Pack<u32>>::T); 2]>>::T: LongDefault + Sync + Send,
    <O as Pack<[(u64, I::SIMDbyts); 2]>>::T: LongDefault,
    <I as BitPack<bool>>::T: Sync,
    <O as Pack<()>>::T: Default,
    <O as BitPack<bool>>::T: Sync,
    [(); 2]: ZipMap<(u64, <I as Pack<u32>>::T), (u64, <I as Pack<u32>>::T), (u64, <I as Pack<u32>>::T)>
        + Pack<(u64, <I as Pack<u32>>::T), T = [(u64, <I as Pack<u32>>::T); 2]>,
{
    examples
        .par_chunks(255)
        .fold(
            || <O as Pack<[(u64, <I as Pack<u32>>::T); 2]>>::T::long_default(),
            |mut acc, chunk| {
                count_matrix_batch::<I, O>(chunk, &mut acc);
                acc
            },
        )
        .reduce_with(|a, b| {
            <O as ZipMap<
                [(u64, <I as Pack<u32>>::T); 2],
                [(u64, <I as Pack<u32>>::T); 2],
                [(u64, <I as Pack<u32>>::T); 2],
            >>::zip_map(&a, &b, |a, b| {
                <[(); 2] as ZipMap<
                    (u64, <I as Pack<u32>>::T),
                    (u64, <I as Pack<u32>>::T),
                    (u64, <I as Pack<u32>>::T),
                >>::zip_map(a, b, |a, b| {
                    (
                        a.0 + b.0,
                        <I as ZipMap<u32, u32, u32>>::zip_map(&a.1, &b.1, |&a, &b| a + b),
                    )
                })
            })
        })
        .unwrap()
}

fn count_matrix_batch<I, O>(
    examples: &[(<I as BitPack<bool>>::T, <O as BitPack<bool>>::T)],
    u32_acc: &mut <O as Pack<[(u64, <I as Pack<u32>>::T); 2]>>::T,
) where
    I: BitPack<bool>
        + Pack<u32>
        + IncrementCounters<u32>
        + ZipMap<u32, u32, u32>
        + SIMDincrementCounters,
    O: BitPack<bool>
        + Map<(), [(u64, I::SIMDbyts); 2]>
        + Map<[(u64, I::SIMDbyts); 2], [(u64, <I as Pack<u32>>::T); 2]>
        + Pack<[(u64, <I as Pack<u32>>::T); 2]>
        + ZipMap<
            [(u64, <I as Pack<u32>>::T); 2],
            [(u64, <I as Pack<u32>>::T); 2],
            [(u64, <I as Pack<u32>>::T); 2],
        > + BitMap<bool, [(u64, I::SIMDbyts); 2]>,
    <I as BitPack<bool>>::T: Sync,
    <O as Pack<()>>::T: Default,
    <O as BitPack<bool>>::T: Sync,
    <O as Pack<[(u64, I::SIMDbyts); 2]>>::T: LongDefault,
    [(); 2]: ZipMap<(u64, <I as Pack<u32>>::T), (u64, <I as Pack<u32>>::T), (u64, <I as Pack<u32>>::T)>
        + Pack<(u64, <I as Pack<u32>>::T), T = [(u64, <I as Pack<u32>>::T); 2]>,
{
    assert!(examples.len() < 256);
    //dbg!(std::mem::size_of::<<O as Pack<[(u64, <I as Pack<T>>::T); 2]>>::T>());
    //let expanded_input = I::expand_bits(&examples[0].0);

    let simd_counts = examples.iter().fold(
        <O as Pack<[(u64, I::SIMDbyts); 2]>>::T::long_default(),
        |mut acc, (input, target)| {
            let expanded_input = I::expand_bits(&input);
            <O as BitMap<bool, [(u64, I::SIMDbyts); 2]>>::map_mut(
                target,
                &mut acc,
                |sign, counters| {
                    counters[sign as usize].0 += 1;
                    I::simd_increment_in_place(&expanded_input, &mut counters[sign as usize].1);
                },
            );
            acc
        },
    );
    <O as Map<[(u64, I::SIMDbyts); 2], [(u64, <I as Pack<u32>>::T); 2]>>::map_mut(
        &simd_counts,
        u32_acc,
        |simd_counts, u32_counts| {
            u32_counts[0].0 += simd_counts[0].0;
            u32_counts[1].0 += simd_counts[1].0;
            I::add_to_u32s(&simd_counts[0].1, &mut u32_counts[0].1);
            I::add_to_u32s(&simd_counts[1].1, &mut u32_counts[1].1);
        },
    );
}

type InputShape = [[[(); 32]; 8]; 4];
type OutputShape = [[(); 32]; 8];

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
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
    //let bytes: Vec<[b32; 8]> = (0..5000_u32).map(|x| (x % 60) as u8).chain((0..255)).map(|x| ecc::encode_byte(x)).collect();

    let examples: Vec<_> = bytes
        .windows(5)
        .map(|w| ([w[0], w[1], w[2], w[3]], w[4]))
        .collect();
    dbg!();
    let start = Instant::now();
    let counts = count_matrix_par::<InputShape, OutputShape>(&examples);
    dbg!(start.elapsed());
    //dbg!(&counts[3][5][0]);

    /*
    let n_correct: u64 = examples
        .iter()
        .map(|(input, target)| {
            let output = <OutputShape as PackedMap<<InputShape as BitPack<WeightType>>::T, bool>>::map(&weights, |weights| {
                <InputShape as BMA<WeightType>>::bma(weights, input) < <InputShape as BMA<WeightType>>::THRESHOLD
            });
            //dbg!(&output);
            let byte = ecc::decode_byte(&output);
            let target_byte = ecc::decode_byte(&target);
            //dbg!(byte);
            //dbg!(target_byte);
            (byte == target_byte) as u64
        })
        .sum();
    dbg!(n_correct);
    println!("{}", n_correct as f64 / bytes.len() as f64);
    */
}
