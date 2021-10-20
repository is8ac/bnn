use crate::bits::b64;
use crate::count_bits::{
    masked_hamming_dist, BitSliceBitCounter, ExpCountBits, PopCountBitCounter, UnitCountBits,
};
use crate::search::{compute_exp_candidates, update_weights, weights_to_dense, weights_to_sparse};
use rayon::prelude::*;
#[cfg(target_feature = "avx2")]
use std::arch::x86_64::{__m256i, __m512i, _mm256_setzero_si256};
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::time::Instant;

const N_THRESHOLDS: usize = 3;
const N_EXP: u32 = 8;

const UNIT_THRESHOLDS: usize = 2;

/*
type CounterType = BitSliceBitCounter<__m256i, 5>;

pub fn train_layer(input: &[[b64; 8]], target: &[[b64; 4]]) -> [[(([b64; 8], [b64; 8]), u32); 64]; 4] {
    assert_eq!(input.len(), target.len());
    let weights: [([Option<bool>; 512], u32); 256] = (0..256).map(|_| ([None; 512], 0u32)).collect::<Vec<_>>().try_into().unwrap();

    let bit_counter = BitSliceBitCounter::<__m256i, 5> {
        slice_type: PhantomData::default(),
    };

    let full_start = Instant::now();
    let weights = (0..7).fold(weights, |mut weights, i| {
        let sparse: [(Vec<(usize, bool)>, [u32; UNIT_THRESHOLDS]); 256] = weights.iter().map(|w| weights_to_sparse(w)).collect::<Vec<_>>().try_into().unwrap();

        let count_start = Instant::now();
        let unit_counts = <CounterType as UnitCountBits<8, 4, UNIT_THRESHOLDS>>::unit_count_bits(&bit_counter, &input, &target, &sparse, 16, 8);
        dbg!(&count_start.elapsed());

        let exp_candidates: [(Vec<(usize, bool)>, [(usize, bool); N_EXP as usize], [u32; N_THRESHOLDS]); 256] = weights
            .iter()
            .zip(unit_counts.iter())
            .map(|(w, counts)| compute_exp_candidates::<512, UNIT_THRESHOLDS, N_THRESHOLDS, N_EXP>(w, counts))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let count_start = Instant::now();
        let exp_counts = <CounterType as ExpCountBits<8, 4, N_THRESHOLDS, N_EXP>>::exp_count_bits(&bit_counter, &input, &target, &exp_candidates, 16, 8);
        dbg!(&count_start.elapsed());

        let sum_acc: u64 = weights
            .iter_mut()
            .zip(exp_candidates.iter())
            .zip(exp_counts.iter())
            .map(|((weights, exp_candidates), exp_counts)| update_weights::<512, N_THRESHOLDS, N_EXP>(weights, &exp_candidates.1, exp_counts))
            .sum();
        //println!("avg bit acc: {:.7}", (sum_acc as f64 / input.len() as f64) / weights.len() as f64);
        weights
    });
    //dbg!(&full_start.elapsed());
    weights
        .chunks(64)
        .map(|chunk| chunk.iter().map(|w| weights_to_dense(w)).collect::<Vec<_>>().try_into().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
*/

pub fn apply(weights: &[[(([b64; 8], [b64; 8]), u32); 64]; 4], input: &[b64; 8]) -> [b64; 4] {
    let mut target = [b64(0); 4];
    for w in 0..4 {
        for b in 0..64 {
            let sign = masked_hamming_dist(input, &weights[w][b].0) > weights[w][b].1;
            target[w].set_bit_in_place(b, sign);
        }
    }
    target
}
