use crate::bits::{BitArray, BitMul, BitWord, IndexedFlipBit};
use crate::float::FloatLoss;
use crate::shape::{Element, Shape};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::ops::AddAssign;

pub fn decend<
    RNG: Rng,
    I: BitArray + Element<O::BitShape> + Sync + BitWord,
    O: BitArray + Sync + BitWord + Send,
    const C: usize,
>(
    rng: &mut RNG,
    layer_weights: &mut <I as Element<O::BitShape>>::Array,
    aux_weights: &[<f32 as Element<O::BitShape>>::Array; C],
    counts: &[(I, [u32; C])],
    window_size: usize,
    window_thresh: usize,
) where
    //[O; C]: Objective<O, C>,
    <I as Element<O::BitShape>>::Array: BitMul<I, O> + Sync + IndexedFlipBit<I, O>,
    f32: Element<O::BitShape>,
    [<f32 as Element<O::BitShape>>::Array; C]: FloatLoss<O, [(); C]> + Sync,
{
    if window_size > window_thresh {
        decend(
            rng,
            layer_weights,
            aux_weights,
            counts,
            (window_size * 4) / 5,
            window_thresh,
        );
    }
    let indices = {
        let mut indices: Vec<usize> = (0..I::BIT_LEN)
            .map(|i| (i * (counts.len() - window_size)) / I::BIT_LEN)
            .collect();
        indices.shuffle(rng);
        indices
    };
    for (ib, &index) in indices.iter().enumerate() {
        let minibatch = &counts[index..index + window_size];
        let mut cur_sum_loss = minibatch
            .par_iter()
            .map(|(input, counts)| aux_weights.counts_loss(&layer_weights.bit_mul(input), counts))
            .sum();
        //dbg!(cur_sum_loss);
        for ob in 0..O::BIT_LEN {
            layer_weights.indexed_flip_bit(ob, ib);
            let new_loss: f32 = minibatch
                .par_iter()
                .map(|(input, counts)| {
                    aux_weights.counts_loss(&layer_weights.bit_mul(input), counts)
                })
                .sum();
            if new_loss < cur_sum_loss {
                cur_sum_loss = new_loss;
            } else {
                layer_weights.indexed_flip_bit(ob, ib);
            }
        }
    }
    let cur_sum_loss: f32 = counts
        .par_iter()
        .map(|(input, counts)| aux_weights.counts_loss(&layer_weights.bit_mul(input), counts))
        .sum();
    let n_examples: u64 = counts
        .iter()
        .map(|(_, c)| c.iter().sum::<u32>() as u64)
        .sum();
    println!("w:{}: l:{}", window_size, cur_sum_loss / n_examples as f32);
}
