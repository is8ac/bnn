use crate::bits::{BitArray, BitMul, BitWord, Distance, IncrementFracCounters, IndexedFlipBit};
use crate::block::BlockCode;
use crate::count::{CounterArray, ElementwiseAdd};
use crate::shape::{Element, Shape};
use rand::distributions;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::ops::AddAssign;

// f32: 3m50.574s
// f64: 7m21.906s
// f32 values
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

pub trait Objective<I, const C: usize> {
    fn loss(&self, input: &I, class: usize) -> u32;
    fn max_class_index(&self, input: &I) -> usize;
    fn generate(inputs: &Vec<(u32, I, usize)>) -> Self;
    fn decend(&mut self, inputs: &Vec<(u32, I, usize)>, cur_sum_loss: &mut u64);
}

impl<
        I: Copy + Distance + BitArray + BitWord + IncrementFracCounters + Send + Sync,
        const C: usize,
    > Objective<I, { C }> for [I; C]
where
    [(I, u32); C]: Default + std::fmt::Debug,
    [I; C]: Default + std::fmt::Debug,
    [u64; C]: Default + std::fmt::Debug,
    u32: Element<I::BitShape>,
    [(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C]:
        Default + Send + Sync + ElementwiseAdd,
    (usize, <u32 as Element<<I as BitArray>::BitShape>>::Array): Default + ElementwiseAdd,
{
    fn loss(&self, input: &I, class: usize) -> u32 {
        let target_act = input.distance(&self[class]);
        let mut n_gre = 0u32;
        for c in 0..C {
            let other_act = input.distance(&self[c]);
            n_gre += (target_act <= other_act) as u32;
        }
        n_gre - 1 // remove target from the count
    }
    fn max_class_index(&self, input: &I) -> usize {
        let mut max_act = 0_u32;
        let mut max_class = 0_usize;
        for c in 0..C {
            let act = self[c].distance(input);
            if act >= max_act {
                max_act = act;
                max_class = c;
            }
        }
        max_class
    }
    fn generate(inputs: &Vec<(u32, I, usize)>) -> Self {
        let activation_counts: [(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C] =
            inputs
                .par_iter()
                .fold(
                    || {
                        <[(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C]>::default(
                        )
                    },
                    |mut acc, (count, input, class)| {
                        input.weighted_increment_frac_counters(*count, &mut acc[*class]);
                        acc
                    },
                )
                .reduce(
                    || {
                        <[(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C]>::default(
                        )
                    },
                    |mut a, b| {
                        a.elementwise_add(&b);
                        a
                    },
                );
        let mut weights = <[I; C]>::default();
        weights.iter_mut().enumerate().for_each(|(c, target)| {
            let other_counts = activation_counts
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != c)
                .fold(
                    <(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array)>::default(),
                    |mut sum, (_, val)| {
                        sum.elementwise_add(val);
                        sum
                    },
                );
            *target = I::bitpack_fracs(&other_counts, &activation_counts[c]);
        });
        weights
    }
    fn decend(&mut self, inputs: &Vec<(u32, I, usize)>, cur_sum_loss: &mut u64) {
        dbg!(&cur_sum_loss);
        for b in 0..I::BIT_LEN {
            for c in 0..C {
                self[c].flip_bit(b);
                let new_loss: u64 = inputs
                    .par_iter()
                    .map(|(count, input, class)| self.loss(input, *class) as u64 * *count as u64)
                    .sum();
                //dbg!((new_loss, *cur_sum_loss));
                if new_loss < *cur_sum_loss {
                    *cur_sum_loss = new_loss;
                    dbg!(new_loss);
                } else {
                    self[c].flip_bit(b);
                }
            }
        }
    }
}

pub struct DecendOneHiddenLayer<
    const K: usize,
    const SEED: u64,
    const THRESHOLD: u32,
    const E: usize,
    const C: usize,
>();

impl<
        I: BitArray
            + Element<O::BitShape>
            + Element<[(); K]>
            + BitWord
            + Sync
            + Send
            + BlockCode<{ K }>,
        O: BitArray + BitWord + Sync + Send,
        const K: usize,
        const SEED: u64,
        const THRESHOLD: u32,
        const E: usize,
        const C: usize,
    > GenWeights<I, O, [(); C]>
    for DecendOneHiddenLayer<{ K }, { SEED }, { THRESHOLD }, { E }, { C }>
where
    <I as Element<O::BitShape>>::Array: IndexedFlipBit<I, O> + BitMul<I, O> + Sync,
    <I as Element<[(); K]>>::Array: Sync,
    [O; C]: Objective<O, C>,
    distributions::Standard: distributions::Distribution<<I as Element<O::BitShape>>::Array>
        + distributions::Distribution<[O; C]>,
{
    type Accumulator = CounterArray<I, { K }, { C }>;
    fn gen_weights(
        accumulator: &Self::Accumulator,
    ) -> (<I as Element<O::BitShape>>::Array, [O; C]) {
        let mut rng = Hc128Rng::seed_from_u64(SEED);

        let inputs: Vec<(u32, I, usize)> = accumulator
            .counters
            .par_iter()
            .enumerate()
            .map(|(class, inputs)| {
                inputs
                    .par_iter()
                    .enumerate()
                    .filter(|(_, count)| **count > THRESHOLD)
                    .map(move |(index, count)| (*count, index, class))
                    .map(|(count, index, class)| {
                        (
                            count,
                            <I>::reverse_block(&accumulator.bit_matrix, index),
                            class,
                        )
                    })
            })
            .flatten()
            .collect();
        dbg!(inputs.len());

        let total: u64 = inputs.par_iter().map(|(count, _, _)| *count as u64).sum();
        dbg!(total);

        let mut layer: <I as Element<<O as BitArray>::BitShape>>::Array = rng.gen();
        //let mut aux_weights = <[O; C]>::generate(
        //    &inputs
        //        .par_iter()
        //        .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
        //        .collect(),
        //);
        let mut aux_weights: [O; C] = rng.gen();

        let mut cur_loss: u64 = inputs
            .par_iter()
            .map(|(count, input, class)| {
                aux_weights.loss(&layer.bit_mul(input), *class) as u64 * *count as u64
            })
            .sum();
        let hidden_inputs: Vec<(u32, O, usize)> = inputs
            .par_iter()
            .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
            .collect();
        // s0: 0.14042
        // s1: 0.12978
        // s2: 0.14628
        // s3: 0.1264
        // s4: 0.13594
        // e3:  0.12856
        // e7:  0.1264
        // e8:  0.1289
        // e9:  0.13224
        // e11: 0.1341
        // e13: 0.1379
        for e in 0..E {
            dbg!(e);
            aux_weights.decend(&hidden_inputs, &mut cur_loss);
            for ib in 0..I::BIT_LEN {
                for ob in 0..O::BIT_LEN {
                    layer.indexed_flip_bit(ob, ib);
                    let new_loss: u64 = inputs
                        .par_iter()
                        .map(|(count, input, class)| {
                            aux_weights.loss(&layer.bit_mul(input), *class) as u64 * *count as u64
                        })
                        .sum();
                    if new_loss < cur_loss {
                        cur_loss = new_loss;
                        dbg!(cur_loss as f64 / total as f64);
                    } else {
                        layer.indexed_flip_bit(ob, ib);
                    }
                }
            }
        }
        (layer, aux_weights)
    }
    fn string_name() -> String {
        format!(
            "DecendOneHiddenLayer<I:{}, O:{}, K:{}, SEED:{}, THRESHOLD:{}, E:{}, C:{}>",
            std::any::type_name::<I>(),
            std::any::type_name::<O>(),
            K,
            SEED,
            THRESHOLD,
            E,
            C
        )
    }
}
