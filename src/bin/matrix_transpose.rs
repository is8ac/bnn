use bnn::bits::{b64, t64};
use bnn::count::ElementwiseAdd;
use bnn::ecc::{decode_byte_64, encode_byte_u64};
use bnn::matrix::{transpose, Counters};
use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::mem;
use std::path::Path;
use std::time::Instant;

//const N_EXAMPLES_EXP: u32 = 20;
//const N_EXAMPLES: usize = 2usize.pow(N_EXAMPLES_EXP);
const CHUNK_EXP: u32 = 12;
const CHUNK_SIZE: usize = 2usize.pow(CHUNK_EXP);
const INPUT_LEN: usize = 16;
const TARGET_LEN: usize = 4;

const N_WORKERS: usize = 16;

fn bma(trits: t64, bits: b64) -> u32 {
    ((trits.0 ^ bits.0) & trits.1).count_ones() * 2
}

fn thd<const L: usize>(trits: &[t64; L], bits: &[b64; L]) -> u32 {
    let mut target = 0u32;
    for i in 0..L {
        target += bma(trits[i], bits[i]);
    }
    target
}

fn tvmm<const I: usize, const T: usize>(
    weights: &[[([t64; I], u32); 64]; T],
    input: &[b64; I],
) -> [b64; T] {
    let mut target = [b64::ZEROS; T];
    for t in 0..T {
        for b in 0..64 {
            let act = thd(&weights[t][b].0, input);
            let bit = act > weights[t][b].1;
            target[t].0 |= (bit as u64) << b;
        }
    }
    target
}

fn main() {
    // ulimit -S -s 1073741824
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_WORKERS)
        .stack_size(2usize.pow(28))
        .build_global()
        .unwrap();

    let mut rng = Hc128Rng::seed_from_u64(0);

    let load_start = Instant::now();
    //let bytes = std::fs::read("tiny-shakespeare.txt").unwrap();
    let bytes = std::fs::read("/big/temp/books_txt/Delphi Complete Works of Charles Dickens (Illustrated) - Charles Dickens.txt").unwrap();
    //let bytes: Vec<u8> = (0..2usize.pow(18)).map(|i| (i % 3) as u8).collect();
    dbg!();
    let expanded: Vec<[b64; 4]> = bytes[0..2usize.pow(18)]
        .par_iter()
        .map(|&b| encode_byte_u64(b))
        .collect();
    dbg!();
    let ngrams: Vec<([b64; 16], [b64; 4])> = expanded
        .par_windows(5)
        .map(|w| {
            let mut input = <[b64; 16]>::default();
            for i in 0..16 {
                input[i] = w[i / 4][i % 4];
            }
            (input, w[4])
        })
        .collect();
    dbg!(load_start.elapsed());

    let start = Instant::now();
    let count = ngrams
        .par_chunks_exact(CHUNK_SIZE)
        .fold(
            || Counters::<INPUT_LEN, TARGET_LEN>::new_box(),
            |mut acc, chunk| {
                let (input_chunk, target_chunk): (Vec<[b64; 16]>, Vec<[b64; 4]>) =
                    chunk.iter().cloned().unzip();

                let input: &[[b64; 16]; CHUNK_SIZE] =
                    <&[[b64; 16]; CHUNK_SIZE]>::try_from(&input_chunk[0..]).unwrap();
                let target: &[[b64; 4]; CHUNK_SIZE] =
                    <&[[b64; 4]; CHUNK_SIZE]>::try_from(&target_chunk[0..]).unwrap();
                let t_input = transpose::<INPUT_LEN, { CHUNK_SIZE / 64 }>(input);
                let t_target = transpose::<TARGET_LEN, { CHUNK_SIZE / 64 }>(target);
                acc.increment::<{ CHUNK_SIZE / 64 }>(&t_input, &t_target);
                acc
            },
        )
        .reduce_with(|mut a, b| {
            Counters::merge(&mut a, b);
            a
        })
        .unwrap();

    dbg!(count.n);
    let duration = start.elapsed();
    dbg!(duration);
    println!(
        "Count: {:.3} ns/example",
        (duration.as_nanos() as f64 * N_WORKERS as f64) / count.n as f64
    );
    //dbg!(&count);
    dbg!(mem::size_of::<Counters::<INPUT_LEN, TARGET_LEN>>());

    let trits_start = Instant::now();
    let trits = count.to_trits();
    dbg!(trits_start.elapsed());
    dbg!(trits[0][0]);

    let n_correct: u64 = expanded
        .par_windows(5)
        .map(|w| {
            let mut input = <[b64; 16]>::default();
            for i in 0..16 {
                input[i] = w[i / 4][i % 4];
            }
            let target_byte = decode_byte_64(&w[4]);
            let predicted_target = tvmm(&trits, &input);
            //dbg!(&predicted_target);
            let predicted_target_byte = decode_byte_64(&predicted_target);
            (target_byte == predicted_target_byte) as u64
        })
        .sum();
    dbg!(n_correct as f64 / expanded.len() as f64);

    /*
    assert_eq!(t_input[0][0].len(), N_EXAMPLES / 64);

    println!("popcnt: {:.3} ns/example", popcnt_start.elapsed().as_nanos() as f64 / N_EXAMPLES as f64);
    dbg!(counts[0][0][0][0]);
    dbg!(o_counts[0][0]);
    dbg!(i_counts[0][0]);
    dbg!(N_EXAMPLES / 4);
    //dbg!(cv_counts);
    dbg!(mem::size_of::<[[[[u32; 64]; INPUT_LEN]; 64]; INPUT_LEN]>());
    dbg!(mem::size_of::<[[u64; INPUT_LEN]; N_EXAMPLES]>());
    */
}
