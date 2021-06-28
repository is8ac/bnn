use bnn::bits::{
    b32, t32, BitMap, BitPack, FromBool, IncrementCounters, PackedMap, SIMDincrementCounters,
    SIMDword32, BMA,
};
use bnn::count::ElementwiseAdd;
use bnn::ecc;
use bnn::matrix::CacheLocalMatrixBatch;
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

fn safe_count_matrix_par<I, O>(
    examples: &[(<I as BitPack<bool>>::T, <O as BitPack<bool>>::T)],
) -> Box<<O as Pack<[(u32, <I as Pack<u32>>::T); 2]>>::T>
where
    I: BitPack<bool> + Pack<u32> + Pack<[u32; 32]> + SIMDincrementCounters + Map<u32, u32>,
    O: BitPack<bool>
        + Pack<[(u32, <I as Pack<u32>>::T); 2]>
        + CacheLocalMatrixBatch<I>
        + Map<[(u32, <I as Pack<u32>>::T); 2], [(u32, <I as Pack<u32>>::T); 2]>,
    <I as SIMDincrementCounters>::WordShape: Pack<[u32; 32]> + Pack<<O as Pack<[[u32; 32]; 2]>>::T>,
    <I as BitPack<bool>>::T: Sync,
    <O as BitPack<bool>>::T: Sync,
    <O as Pack<[(u32, <I as Pack<u32>>::T); 2]>>::T: LongDefault + Sync + Send,
{
    examples
        .par_chunks(255)
        .fold(
            || Box::<<O as Pack<[(u32, <I as Pack<u32>>::T); 2]>>::T>::long_default(),
            |mut acc, examples| {
                <O as CacheLocalMatrixBatch<I>>::safe_increment_counters_batch(examples, &mut acc);
                acc
            },
        )
        .reduce_with(|a, mut b| {
            <O as Map<[(u32, <I as Pack<u32>>::T); 2], [(u32, <I as Pack<u32>>::T); 2]>>::map_mut(
                &a,
                &mut b,
                |a, b| {
                    b[0].0 += a[0].0;
                    b[1].0 += a[1].0;
                    <I as Map<u32, u32>>::map_mut(&a[0].1, &mut b[0].1, |&a, b| *b += a);
                    <I as Map<u32, u32>>::map_mut(&a[1].1, &mut b[1].1, |&a, b| *b += a);
                },
            );
            b
        })
        .unwrap()
}

fn count_matrix_par<I, O>(
    examples: &[(<I as BitPack<bool>>::T, <O as BitPack<bool>>::T)],
) -> <O as Pack<[(u32, <I as Pack<u32>>::T); 2]>>::T
where
    I: BitPack<bool>
        + Pack<u32>
        + IncrementCounters<u32>
        + ZipMap<u32, u32, u32>
        + SIMDincrementCounters
        + Pack<[u32; 32]>,
    O: CacheLocalMatrixBatch<I>
        + BitPack<bool>
        + Pack<u32>
        + Pack<[u32; 2]>
        + Pack<[[u32; 32]; 2]>
        + Pack<[(u64, <I as Pack<u32>>::T); 2]>
        + IndexGet<u32>
        + IndexGet<[u32; 2]>
        + Map<(), [(u64, I::SIMDbyts); 2]>
        + Map<[(u64, I::SIMDbyts); 2], [(u64, <I as Pack<u32>>::T); 2]>
        + Map<[(u32, <I as Pack<u32>>::T); 2], [(u32, <I as Pack<u32>>::T); 2]>
        + Pack<[(u32, <I as Pack<u32>>::T); 2]>
        + ZipMap<
            [(u64, <I as Pack<u32>>::T); 2],
            [(u64, <I as Pack<u32>>::T); 2],
            [(u64, <I as Pack<u32>>::T); 2],
        > + BitMap<bool, [(u64, I::SIMDbyts); 2]>,
    <O as Pack<[(u64, <I as Pack<u32>>::T); 2]>>::T: LongDefault + Sync + Send,
    <O as Pack<[(u64, I::SIMDbyts); 2]>>::T: LongDefault,
    <I as SIMDincrementCounters>::WordShape: Pack<[u32; 32]>,
    <O as Pack<u32>>::T: Sync + Send,
    <<I as SIMDincrementCounters>::WordShape as Pack<<O as Pack<[[u32; 32]; 2]>>::T>>::T:
        Send + Sync,
    <I as Pack<u32>>::T: LongDefault,
    <I as SIMDincrementCounters>::WordShape: Pack<<O as Pack<[[u32; 32]; 2]>>::T>
        + IndexGet<<O as Pack<[[u32; 32]; 2]>>::T>
        + IndexGet<b32>,
    Box<(
        <<I as SIMDincrementCounters>::WordShape as Pack<<O as Pack<[[u32; 32]; 2]>>::T>>::T,
        <O as Pack<u32>>::T,
        u32,
    )>: LongDefault,
    <I as BitPack<bool>>::T: Sync,
    <O as Pack<()>>::T: Default,
    <O as BitPack<bool>>::T: Sync,
    <I as SIMDincrementCounters>::WordShape: Pack<[u32; 32], T = <I as Pack<u32>>::T>,
    [(); 2]: ZipMap<(u64, <I as Pack<u32>>::T), (u64, <I as Pack<u32>>::T), (u64, <I as Pack<u32>>::T)>
        + Pack<(u64, <I as Pack<u32>>::T), T = [(u64, <I as Pack<u32>>::T); 2]>,
{
    let acc: Box<(
        <<I as SIMDincrementCounters>::WordShape as Pack<<O as Pack<[[u32; 32]; 2]>>::T>>::T,
        <O as Pack<u32>>::T,
        u32,
    )> = examples
        .par_chunks(255)
        .fold(
            || <O as CacheLocalMatrixBatch<I>>::allocate_acc(),
            |mut acc, chunk| {
                <O as CacheLocalMatrixBatch<I>>::cache_local_count_matrix_batch(chunk, &mut acc);
                acc
            },
        )
        .reduce_with(|mut a, b| {
            <O as CacheLocalMatrixBatch<I>>::merge(&mut a, &b);
            a
        })
        .unwrap();

    <O as CacheLocalMatrixBatch<I>>::word_transpose_weights_matrix(&*acc)
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
    /*
    let bytes: Vec<[b32; 8]> = buf_reader
        .bytes()
        .take(2usize.pow(16))
        .map(|x| {
            let x = x.unwrap();
            ecc::encode_byte(x)
        })
        .collect();
    */
    let bytes: Vec<[b32; 8]> = (0..2u32.pow(28))
        .map(|x| (x % 253) as u8)
        .map(|x| ecc::encode_byte(x))
        .collect();

    let examples: Vec<_> = bytes.windows(3).map(|w| ([w[0], w[1]], w[2])).collect();
    dbg!();
    let fast_start = Instant::now();
    let fast_counts = count_matrix_par::<InputShape, OutputShape>(&examples);
    dbg!(fast_start.elapsed());

    let safe_start = Instant::now();
    let safe_counts = safe_count_matrix_par::<InputShape, OutputShape>(&examples);
    dbg!(safe_start.elapsed());
    //dbg!(&counts[3][5][0]);
    assert_eq!(fast_counts, *safe_counts);

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
