use crate::bits::b64;
use crate::bits::GetBit;
use crate::count_bits::TritArray;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;

fn compute_thresholds<const N: usize>(base: u32) -> [u32; N] {
    (0..N)
        .map(|x| base + x as u32)
        .collect::<Vec<u32>>()
        .try_into()
        .unwrap()
}

pub fn compute_exp_candidates<const I: usize, const UN: usize, const EN: usize, const E: u32>(
    weights: &([Option<bool>; I], u32),
    unit_counts: &[[[u64; UN]; 2]; I],
) -> (Vec<(usize, bool)>, [(usize, bool); E as usize], [u32; EN]) {
    let mut sorted_candidates: Vec<_> = weights
        .0
        .iter()
        .zip(unit_counts.iter())
        .enumerate()
        .filter_map(|(i, (weight, counts))| {
            if let None = weight {
                Some(
                    counts
                        .iter()
                        .enumerate()
                        .map(|(s, counts)| counts.iter().max().map(|&a| (a, (i, s == 1))).unwrap())
                        .max()
                        .unwrap(),
                )
            } else {
                // skip
                None
            }
        })
        .collect();
    sorted_candidates.sort_by_key(|&(a, _)| Reverse(a));

    let candidates: [(usize, bool); E as usize] = sorted_candidates
        .iter()
        .map(|&(_, (i, s))| (i, s))
        .take(E as usize)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let base = weights
        .0
        .iter()
        .enumerate()
        .filter_map(|(i, &w)| w.map(|sign| (i, sign)))
        .collect();

    (base, candidates, compute_thresholds(weights.1))
}

pub fn update_weights<const I: usize, const N: usize, const E: u32>(
    weights: &mut ([Option<bool>; I], u32),
    exp_candidates: &[(usize, bool); E as usize],
    exp_counts: &[[u64; N]; 2usize.pow(E)],
) -> u64 {
    let thresholds = compute_thresholds::<N>(weights.1);
    let (mask, (accuracy, threshold)): (usize, (u64, u32)) = exp_counts
        .iter()
        .enumerate()
        .map(|(mask, &counts)| {
            let (accuracy, threshold) = counts
                .iter()
                .zip(thresholds.iter())
                .max_by_key(|(c, t)| (*c, Reverse(*t)))
                .unwrap();
            (mask, (*accuracy, *threshold))
        })
        .max_by_key(|(mask, (a, _))| (*a, mask.count_zeros()))
        .unwrap();
    if mask == 0 {
        //println!("stuck",);
    }
    weights.1 = threshold;

    exp_candidates.iter().enumerate().for_each(|(b, &(i, s))| {
        weights.0[i] = Some(s).filter(|_| mask.bit(b));
    });
    accuracy
}

pub fn weights_to_sparse<const I: usize, const N: usize>(
    weights: &([Option<bool>; I], u32),
) -> (Vec<(usize, bool)>, [u32; N]) {
    (
        weights
            .0
            .iter()
            .enumerate()
            .filter_map(|(i, t)| t.map(|s| (i, s)))
            .collect(),
        compute_thresholds(weights.1),
    )
}

pub fn weights_to_dense<const I: usize>(
    weights: &([Option<bool>; 64 * I], u32),
) -> (([b64; I], [b64; I]), u32) {
    let base = weights
        .0
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.map(|s| (i, s)))
        .fold(([b64(0); I], [b64(0); I]), |w, (i, s)| w.set_trit(i, s));
    (base, weights.1)
}
