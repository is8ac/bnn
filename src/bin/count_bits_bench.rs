#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
use bnn::bits::{b64, t64};
use bnn::count_bits::{count_act_dist, count_act_dist_cache_local, count_target_bits, HammingDist};
use bnn::ecc::{decode_byte, encode_byte};
use bnn::search::{SearchManager, UnitSearch, Weights};
use bnn::shape::flatten_2d;
use rayon::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::fmt::Debug;
use std::fs::File;
use std::hash::Hash;
use std::io::prelude::*;
use std::io::BufReader;
use std::mem;
use std::path::Path;
use std::time::Instant;

fn predict_byte<W: HammingDist<Bits = [b64; I]>, const I: usize>(
    input: &[b64; I],
    weights: &[[(W, u32); 64]; 4],
) -> u8 {
    let mut target = [b64(0); 4];
    for w in 0..4 {
        for b in 0..64 {
            let bit = weights[w][b].0.hamming_dist(input) > weights[w][b].1;
            target[w].0 |= (bit as u64) << b;
        }
    }
    decode_byte(&target)
}

fn train<W, M, F: Fn(W) -> M, const I: usize>(
    ngrams: &[([b64; I], [b64; 4])],
    n: usize,
    m: usize,
    manager_init_fn: F,
) -> ([[(W, u32); 64]; 4], f64)
where
    M: SearchManager<W> + Debug,
    W: Weights + Copy + Debug + Send + Sync + HammingDist<Bits = [b64; I]>,
    W::Index: Eq + Hash + Copy + Debug,
    W::Mutation: Eq + Hash + Copy + Debug,
    [(); I * 64]: ,
{
    let weights = [[W::init(); 64]; 4];

    let mut managers: [[M; 64]; 4] = weights
        .iter()
        .map(|x| {
            x.iter()
                .map(|&bits| manager_init_fn(bits))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let full_weights = [[(W::init(), 0u32); 64]; 4];

    (0..n)
        .fold(
            (managers, (full_weights, 0f64)),
            |(mut managers, (full_weights, _)), i| {
                let mutations: [[Vec<W>; 64]; 4] = managers
                    .iter()
                    .map(|x| {
                        x.iter()
                            .map(|x| x.mutation_candidates(m))
                            .collect::<Vec<_>>()
                            .try_into()
                            .unwrap()
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                let mutations_start = Instant::now();
                let dists =
                    count_act_dist_cache_local::<W, I, 4, { 2usize.pow(13) }>(&ngrams, &mutations);
                //dbg!(mutations_start.elapsed());

                let thresholds_start = Instant::now();
                let thresholds = dists.find_thresholds2();
                //dbg!(thresholds_start.elapsed());

                let sum_correct: u64 = managers
                    .iter_mut()
                    .zip(thresholds.iter())
                    .zip(mutations.iter())
                    .map(|((m, t), b)| {
                        m.iter_mut()
                            .zip(t.iter())
                            .zip(b.iter())
                            .map(|((m, n), b)| {
                                let updates: Vec<_> =
                                    b.iter().cloned().zip(n.iter().map(|(_, n)| *n)).collect();
                                m.update(&updates)
                            })
                            .sum::<u64>()
                    })
                    .sum();

                let mut full_weights = [[(W::init(), 032); 64]; 4];
                for w in 0..4 {
                    for b in 0..64 {
                        full_weights[w][b].0 = managers[w][b].weights();
                        full_weights[w][b].1 = thresholds[w][b][0].0;
                    }
                }
                let avg_correct = (sum_correct as f64 / ngrams.len() as f64) / (64 * 4) as f64;
                println!("{}: {}", i, avg_correct);
                (managers, (full_weights, avg_correct))
            },
        )
        .1
}

fn layer2<W, M>(
    expanded: &[[b64; 4]],
    bytes: &[u8],
    n: usize,
    m: usize,
) -> ([[(W, u32); 64]; 4], f64, f64)
where
    W: Weights + Copy + Debug + Send + Sync + HammingDist<Bits = [b64; 8]>,
    W::Index: Eq + Hash + Copy + Debug,
    W::Mutation: Eq + Hash + Copy + Debug,
    //[(); { N * 4 } * 64]: ,
{
    let ngrams: Vec<([b64; { 8 }], [b64; 4])> = expanded
        .par_windows(N + 1)
        .map(|w| {
            (
                *flatten_2d::<b64, 2, 4>(<&[[b64; 4]; N]>::try_from(&w[0..2]).unwrap()),
                w[2],
            )
        })
        .collect();

    let (full_weights, bit_acc) =
        train::<W, _, _, 8>(&ngrams[0..2usize.pow(SIZE)], 50, 27, |bits| {
            UnitSearch::<([b64; 8], [b64; 8])>::init(bits)
        });

    let n_correct: u64 = expanded
        .par_windows(N)
        .zip(bytes[N..].par_iter())
        .map(|(input, target)| {
            //let x = w[0..N].iter().map(|&b| encode_byte(b)).collect::<Vec<[b64; 4]>>();
            let input = flatten_2d::<b64, N, 4>(<&[[b64; 4]; N]>::try_from(&input[0..]).unwrap());
            let predicted = predict_byte(&input, &full_weights);
            (predicted == *target) as u64
        })
        .sum();

    println!(
        "bit acc: {:.6}, byte acc: {:.6}",
        bit_acc,
        n_correct as f64 / ngrams.len() as f64
    );
    (
        full_weights,
        bit_acc,
        n_correct as f64 / ngrams.len() as f64,
    )
}

const N_WORKERS: usize = 16;
const SIZE: u32 = 16;
const N: usize = 2;

fn main() {
    // ulimit -S -s 1073741824
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_WORKERS)
        .stack_size(2usize.pow(28))
        .build_global()
        .unwrap();

    let full_start = Instant::now();

    //let mut rng = Hc128Rng::seed_from_u64(0);
    let load_start = Instant::now();
    //let bytes = std::fs::read("tiny-shakespeare.txt").unwrap();
    //let bytes = std::fs::read("rand_bytes").unwrap();
    //let bytes = std::fs::read("/bin/hugo").unwrap();
    let bytes = std::fs::read("/big/temp/books_txt/Delphi Complete Works of Charles Dickens (Illustrated) - Charles Dickens.txt").unwrap();
    //let bytes: Vec<u8> = (0..(2usize.pow(25) + 5)).map(|i| (i % 50) as u8).collect();

    let expanded: Vec<[b64; 4]> = bytes.par_iter().map(|&b| encode_byte(b)).collect();

    let ngrams: Vec<([b64; { N * 4 }], [b64; 4])> = expanded
        .par_windows(N + 1)
        .map(|w| {
            (
                *flatten_2d::<b64, N, 4>(<&[[b64; 4]; N]>::try_from(&w[0..N]).unwrap()),
                w[N],
            )
        })
        .collect();

    let (full_weights, bit_acc) = train::<([b64; { N * 4 }], [b64; { N * 4 }]), _, _, { N * 4 }>(
        &ngrams[0..2usize.pow(SIZE)],
        50,
        27,
        |bits| UnitSearch::<([b64; 4 * N], [b64; 4 * N])>::init(bits),
    );

    let n_correct: u64 = expanded
        .par_windows(N)
        .zip(bytes[N..].par_iter())
        .take(2usize.pow(SIZE))
        .map(|(input, target)| {
            //let x = w[0..N].iter().map(|&b| encode_byte(b)).collect::<Vec<[b64; 4]>>();
            let input = flatten_2d::<b64, N, 4>(<&[[b64; 4]; N]>::try_from(&input[0..]).unwrap());
            let predicted = predict_byte(&input, &full_weights);
            (predicted == *target) as u64
        })
        .sum();

    println!(
        "bit acc: {:.6}, byte acc: {:.6}",
        bit_acc,
        n_correct as f64 / 2usize.pow(SIZE) as f64
    );
    dbg!(full_start.elapsed());
}
