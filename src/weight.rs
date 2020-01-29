use crate::bits::{BitArray, BitMul, BitWord, IndexedFlipBit, BFMA};

use crate::shape::{Element, Shape};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::ops::AddAssign;

pub trait Sum<T> {
    fn sum(&self) -> T;
}

impl Sum<f32> for f32 {
    fn sum(&self) -> f32 {
        *self
    }
}
impl Sum<u32> for u32 {
    fn sum(&self) -> u32 {
        *self
    }
}

impl<T: Sum<E>, E: AddAssign + Default, const L: usize> Sum<E> for [T; L] {
    fn sum(&self) -> E {
        let mut sum = E::default();
        for i in 0..L {
            sum += self[i].sum();
        }
        sum
    }
}

pub trait Min
where
    Self: Shape + Sized,
    f32: Element<Self>,
{
    fn min(values: &<f32 as Element<Self>>::Array) -> Option<(Self::Index, f32)>;
}

impl Min for () {
    fn min(&values: &f32) -> Option<((), f32)> {
        Some(((), values))
    }
}

impl<T: Min, const L: usize> Min for [T; L]
where
    f32: Element<T>,
    Self: Shape<Index = (usize, T::Index)>,
    T::Index: Copy,
{
    fn min(values: &[<f32 as Element<T>>::Array; L]) -> Option<((usize, T::Index), f32)> {
        let mut cur_min: Option<((usize, T::Index), f32)> = None;
        for i in 0..L {
            if let Some((sub_index, sub_min)) = T::min(&values[i]) {
                if let Some((_, min)) = cur_min {
                    if !(sub_min >= min) {
                        cur_min = Some(((i, sub_index), sub_min));
                    }
                } else {
                    cur_min = Some(((i, sub_index), sub_min));
                }
            }
        }
        cur_min
    }
}

pub trait FlipBool
where
    bool: Element<Self>,
    Self: Shape + Sized,
{
    fn flip_bool(bools: &mut <bool as Element<Self>>::Array, index: Self::Index);
}

impl FlipBool for () {
    fn flip_bool(bools: &mut bool, _: ()) {
        *bools = !*bools;
    }
}

impl<T: FlipBool + Shape, const L: usize> FlipBool for [T; L]
where
    bool: Element<T>,
{
    fn flip_bool(
        bools: &mut [<bool as Element<T>>::Array; L],
        (index, sub_index): (usize, T::Index),
    ) {
        T::flip_bool(&mut bools[index], sub_index);
    }
}

pub trait GenWeights<I: Element<O::BitShape>, O: BitArray + Element<C>, C: Shape> {
    type Accumulator;
    fn gen_weights(
        acc: &Self::Accumulator,
    ) -> (<I as Element<O::BitShape>>::Array, <O as Element<C>>::Array);
    fn string_name() -> String;
}

pub struct SupervisedWeightsGen<I, O, const C: usize> {
    input: PhantomData<I>,
    target: PhantomData<O>,
}

pub struct DecendOneHiddenLayer<
    const K: usize,
    const SEED: u64,
    const THRESHOLD: u32,
    const E: usize,
    const C: usize,
>();

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
    [<f32 as Element<O::BitShape>>::Array; C]: FloatObj<O, [(); C]> + Sync,
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

pub trait FloatObj<I: BitArray, C: Shape>
where
    u32: Element<C>,
{
    fn loss(&self, input: &I, true_class: usize) -> f32;
    ///// Given a list of the number of times that this input is of each class, compute the loss.
    fn counts_loss(&self, input: &I, classes_counts: &<u32 as Element<C>>::Array) -> f32;
    //fn counts_generate(inputs: &[(I, [u32; C])]) -> Self;
    //fn count_decend(&mut self, inputs: &Vec<(I, [u32; C])>, cur_sum_loss: &mut u64, iters: usize);
    //fn new<R: RngCore>(rng: &mut R) -> Self;
}

impl<I: BitArray + BFMA, const C: usize> FloatObj<I, [(); C]>
    for [<f32 as Element<I::BitShape>>::Array; C]
where
    f32: Element<I::BitShape>,
    [f32; C]: Default,
    [[f32; 2]; C]: Default,
    rand::distributions::Standard:
        rand::distributions::Distribution<<f32 as Element<<I as BitArray>::BitShape>>::Array>,
    I::BitShape: Default,
    [<f32 as Element<I::BitShape>>::Array; C]: Noise,
{
    fn loss(&self, input: &I, true_class: usize) -> f32 {
        let mut exp = <[f32; C]>::default();
        let mut sum_exp = 0f32;
        for c in 0..C {
            exp[c] = input.bfma(&self[c]).exp();
            sum_exp += exp[c];
        }
        let mut sum_loss = 0f32;
        for c in 0..C {
            let scaled = exp[c] / sum_exp;
            sum_loss += (scaled - (c == true_class) as u8 as f32).powi(2);
        }
        sum_loss
    }
    /// `weights.counts_loss(&input, &counts)`` is equivalent to
    ///
    /// ```
    ///counts
    ///    .iter()
    ///    .enumerate()
    ///    .map(|(class, count)| weights.loss(&input, class) * *count as f32)
    ///    .sum()
    ///```
    /// except that floating point is imprecise so it often won't actualy be the same.
    fn counts_loss(&self, input: &I, counts: &[u32; C]) -> f32 {
        let (total_sum_loss, losses_diffs) = {
            let (exp, sum_exp) = {
                let mut exp = <[f32; C]>::default();
                let mut sum_exp = 0f32;
                for c in 0..C {
                    exp[c] = input.bfma(&self[c]).exp();
                    sum_exp += exp[c];
                }
                (exp, sum_exp)
            };
            let mut losses_diffs = <[f32; C]>::default();
            let mut total_sum_loss = 0f32;
            for c in 0..C {
                let scaled = exp[c] / sum_exp;
                let loss_0 = scaled.powi(2);
                let loss_1 = (scaled - 1f32).powi(2);
                losses_diffs[c] = loss_1 - loss_0;
                total_sum_loss += loss_0;
            }
            (total_sum_loss, losses_diffs)
        };
        let mut sum_loss = 0f32;
        for c in 0..C {
            sum_loss += (total_sum_loss + losses_diffs[c]) * counts[c] as f32;
        }
        sum_loss
    }
    //fn count_decend(&mut self, inputs: &Vec<(I, [u32; C])>, cur_sum_loss: &mut u64, iters: usize) {
    //    let mut rng = rand::thread_rng();
    //    let noise: Vec<<f32 as Element<I::BitShape>>::Array> = (0..C + iters)
    //        .map(|_| <f32 as Element<I::BitShape>>::Array::noise(&mut rng, -0.1, 0.1))
    //        .collect();
    //}
    //fn new<R: RngCore>(rng: &mut R) -> Self {
    //    <[<f32 as Element<I::BitShape>>::Array; C]>::noise(rng)
    //}
}

pub trait Noise {
    fn noise<R: RngCore>(rng: &mut R, sdev: f32) -> Self;
}

impl Noise for f32 {
    fn noise<R: RngCore>(rng: &mut R, sdev: f32) -> f32 {
        let normal = Normal::new(0f32, sdev).unwrap();
        normal.sample(rng)
    }
}

impl<T: Noise, const L: usize> Noise for [T; L]
where
    [T; L]: Default,
{
    fn noise<RNG: RngCore>(rng: &mut RNG, sdev: f32) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::noise(rng, sdev);
        }
        target
    }
}
