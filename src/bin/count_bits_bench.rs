use bnn::bits::{b64, t64};
use bnn::count_bits::{count_act_dist, count_act_dist_cache_local, count_bits, HammingDist};
use bnn::ecc::ExpandByte;
use bnn::search::{SearchManager, UnitSearch};
use bnn::shape::flatten_2d;
use rayon::prelude::*;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::fs::File;
use std::hash::Hash;
use std::io::prelude::*;
use std::io::BufReader;
use std::mem;
use std::path::Path;
use std::time::Instant;

fn predict_byte<W: HammingDist<Bits = [b64; I]>, const I: usize, const T: usize>(
    input: &[b64; I],
    weights: &[[(W, u32); 64]; T],
) -> u8
where
    [b64; T]: ExpandByte,
{
    let mut target = [b64(0); T];
    for w in 0..T {
        for b in 0..64 {
            let bit = weights[w][b].0.hamming_dist(input) > weights[w][b].1;
            target[w].0 |= (bit as u64) << b;
        }
    }
    //dbg!(&target);
    target.decode_byte()
}

const N_WORKERS: usize = 16;
const SIZE: u32 = 20;

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
    //let bytes: Vec<u8> = (0..(2usize.pow(25) + 5)).map(|i| (i % 50) as u8).collect();

    let expanded: Vec<[b64; 4]> = bytes
        .par_iter()
        .map(|&b| ExpandByte::encode_byte(b))
        .collect();

    let ngrams: Vec<([b64; 16], [b64; 4])> = expanded
        .par_windows(5)
        .map(|w| {
            (
                *flatten_2d::<b64, 4, 4>(<&[[b64; 4]; 4]>::try_from(&w[0..4]).unwrap()),
                w[4],
            )
        })
        .collect();

    let counts = count_bits::<16, 4, { 2usize.pow(8) }>(&ngrams[0..2usize.pow(SIZE)]);

    let start = Instant::now();

    let weights = [[[t64(0, 0); 16]; 64]; 4];

    let mut managers: [[UnitSearch<[t64; 16]>; 64]; 4] = weights
        .iter()
        .map(|x| {
            x.iter()
                .map(|&bits| UnitSearch::<[t64; 16]>::init(bits))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    for i in 0..1000 {
        //dbg!(i);
        let mutations: [[Vec<[_; 16]>; 64]; 4] = managers
            .iter()
            .map(|x| {
                x.iter()
                    .map(|x| x.mutation_candidates(30))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mutations_start = Instant::now();
        let dists = count_act_dist_cache_local::<[_; 16], 16, 4, { 2usize.pow(13) }>(
            &ngrams[0..2usize.pow(SIZE)],
            &mutations,
        );
        //dbg!(mutations_start.elapsed());

        let thresholds = dists.find_thresholds(&counts.target);

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
        println!(
            "{} avg bit acc: {}",
            i,
            sum_correct as f64 / counts.n as f64 / (64 * 4) as f64
        );

        let mut full_weights = [[([t64(0, 0); 16], 032); 64]; 4];
        for w in 0..4 {
            for b in 0..64 {
                full_weights[w][b].0 = managers[w][b].weights();
                full_weights[w][b].1 = thresholds[w][b][0].0;
            }
        }
        let start = Instant::now();

        let n_correct: u64 = bytes
            .par_windows(5)
            .take(2usize.pow(SIZE))
            .map(|w| {
                let x = w[0..4]
                    .iter()
                    .map(|&b| ExpandByte::encode_byte(b))
                    .collect::<Vec<[b64; 4]>>();
                let input = flatten_2d::<b64, 4, 4>(<&[[b64; 4]; 4]>::try_from(&x[0..]).unwrap());
                let predicted = predict_byte(&input, &full_weights);
                (predicted == w[4]) as u64
            })
            .sum();
        //dbg!(start.elapsed());

        println!("byte acc: {}", n_correct as f64 / 2usize.pow(SIZE) as f64);
    }

    dbg!(start.elapsed());
    //dbg!(&thresholds);
    /*
    for w in 0..4 {
        for b in 0..64 {
            thresholds[w][b].iter().for_each(|&(_, n_correct)| {
                println!("{}", n_correct as f64 / counts.n as f64);
            });
        }
    }
    */
}
