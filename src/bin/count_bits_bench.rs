use bnn::bits::{b64, t64};
use bnn::count_bits::count_bits;
use bnn::ecc::ExpandByte;
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::mem;
use std::path::Path;
use std::time::Instant;

const N_WORKERS: usize = 16;

fn main() {
    // ulimit -S -s 1073741824
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_WORKERS)
        .stack_size(2usize.pow(28))
        .build_global()
        .unwrap();

    //let mut rng = Hc128Rng::seed_from_u64(0);
    let load_start = Instant::now();
    //let bytes = std::fs::read("tiny-shakespeare.txt").unwrap();
    //let bytes = std::fs::read("rand_bytes").unwrap();
    //let bytes = std::fs::read("/bin/hugo").unwrap();
    let bytes = std::fs::read("/big/temp/books_txt/Delphi Complete Works of Charles Dickens (Illustrated) - Charles Dickens.txt").unwrap();

    let expanded: Vec<b64> = bytes
        .par_iter()
        .map(|&b| ExpandByte::encode_byte(b))
        .collect();

    let ngrams: Vec<([b64; 4], [b64; 1])> = expanded
        .par_windows(5)
        .map(|w| {
            let mut input = <[b64; 4]>::default();
            for i in 0..4 {
                //input[i] = w[i / 4][i % 4];
                input[i] = w[i];
            }
            (input, [w[4]])
        })
        .collect();

    let start = Instant::now();
    let counts = count_bits::<4, 1, { 2usize.pow(8) }>(&ngrams[0..2usize.pow(24)]);
    dbg!(&counts.n);
    let weights = counts.to_mask(0.47);
    //let weights = counts.to_bits();
    dbg!(weights);
    dbg!(start.elapsed());
    dbg!(weights.len());
    dbg!(weights[0].len());

    //dbg!(weights);
}
