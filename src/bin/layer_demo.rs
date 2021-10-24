#![feature(generic_const_exprs)]

use bnn::bits::b64;
use bnn::bitslice::BitArray64;
use bnn::count_bits::{BitSliceBitCounter, ExpCountBits, PopCountBitCounter, UnitCountBits};
use bnn::ecc::{decode_byte, encode_byte};
use bnn::layer;
use bnn::search::{compute_exp_candidates, update_weights, weights_to_dense, weights_to_sparse};
use bnn::shape::flatten_2d;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::time::Instant;

type CounterType = BitSliceBitCounter<BitArray64<8>, 5>;
const EXP_SIZE: u32 = 8;
const EXP_THRESHOLDS: usize = 3;

fn stackable_layer<'a>(input: &'a [[b64; 4]], target: &'a [[b64; 4]], target_bytes: &'a [u8]) -> (Vec<[b64; 4]>, &'a [[b64; 4]], &'a [u8]) {
    let l: Vec<[b64; 8]> = input
        .par_windows(2)
        .map(|slice| flatten_2d::<b64, 2, 4>(<&[[b64; 4]; 2]>::try_from(&slice[0..2]).unwrap()))
        .collect();

    let target = &target[1..];

    let weights: [([Option<bool>; 512], u32); 256] = (0..256).map(|_| ([None; 512], 0u32)).collect::<Vec<_>>().try_into().unwrap();
    let bit_counter = CounterType {
        slice_type: PhantomData::default(),
    };
    let weights = (0..5).fold(weights, |mut weights, i| {
        let sparse: [(Vec<(usize, bool)>, [u32; 2]); 256] = weights.iter().map(|w| weights_to_sparse(w)).collect::<Vec<_>>().try_into().unwrap();
        let unit_counts = <CounterType as UnitCountBits<8, 4, 2>>::unit_count_bits(&bit_counter, &l[0..TRAIN_LEN], &target[0..TRAIN_LEN], &sparse, 4096);
        let exp_candidates: [(Vec<(usize, bool)>, [(usize, bool); EXP_SIZE as usize], [u32; EXP_THRESHOLDS]); 256] = weights
            .iter()
            .zip(unit_counts.iter())
            .map(|(w, counts)| compute_exp_candidates::<512, 2, EXP_THRESHOLDS, EXP_SIZE>(w, counts))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let exp_counts =
            <CounterType as ExpCountBits<8, 4, EXP_THRESHOLDS, EXP_SIZE>>::exp_count_bits(&bit_counter, &l[0..TRAIN_LEN], &target[0..TRAIN_LEN], &exp_candidates, 4096);
        let _sum_acc: u64 = weights
            .iter_mut()
            .zip(exp_candidates.iter())
            .zip(exp_counts.iter())
            .map(|((weights, exp_candidates), exp_counts)| update_weights::<512, EXP_THRESHOLDS, EXP_SIZE>(weights, &exp_candidates.1, exp_counts))
            .sum();
        weights
    });
    let sparsity: usize = weights.iter().map(|(w, _)| w.iter().filter(|x| x.is_none()).count()).sum();
    let weights = weights
        .chunks(64)
        .map(|chunk| chunk.iter().map(|w| weights_to_dense(w)).collect::<Vec<_>>().try_into().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let expanded: Vec<[b64; 4]> = l.par_iter().map(|input| layer::apply(&weights, &input)).collect();

    let n_correct: u64 = expanded
        .par_iter()
        .zip(target_bytes.par_iter())
        .map(|(predicted, target)| {
            let decoded = decode_byte(predicted);
            (decoded == *target) as u64
        })
        .sum();
    println!(
        "accuracy {:.3}%, sparsity: {:.3}%",
        (n_correct as f64 / expanded.len() as f64) * 100f64,
        (sparsity as f64 / (256 * 512) as f64) * 100f64
    );
    (expanded, target, &target_bytes[1..])
}

const TRAIN_LEN: usize = 2usize.pow(18);

fn main() {
    rayon::ThreadPoolBuilder::new().stack_size(2usize.pow(30)).build_global().unwrap();

    let bytes = std::fs::read("/big/temp/books_txt/Delphi Complete Works of Charles Dickens (Illustrated) - Charles Dickens.txt").unwrap();

    let expanded: Vec<[b64; 4]> = bytes.par_iter().map(|&b| encode_byte(b)).collect();

    let (input, target, target_bytes) = stackable_layer(&expanded, &expanded[1..], &bytes[2..]);

    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);
    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);
    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);
    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);

    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);
    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);
    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);
    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);
    let (input, target, target_bytes) = stackable_layer(&input, target, target_bytes);
}
