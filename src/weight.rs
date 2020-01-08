use crate::bits::{BitArray, BitWord, Distance, IncrementFracCounters};
use crate::count::ElementwiseAdd;
use crate::shape::{Element, Shape};
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

pub trait GenWeights<I: BitArray, O: BitArray>
where
    (I::WordType, I::WordType): Element<I::WordShape>,
    (
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ): Element<O::BitShape>,
{
    type Accumulator;
    fn gen_weights(
        acc: &Self::Accumulator,
    ) -> <(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ) as Element<O::BitShape>>::Array;
}

pub struct SupervisedWeightsGen<I, O, const C: usize> {
    input: PhantomData<I>,
    target: PhantomData<O>,
}

pub trait GenFilterSupervised
where
    Self: BitArray + Sized,
    u32: Element<Self::BitShape>,
    <u32 as Element<<Self as BitArray>::BitShape>>::Array: Element<<Self as BitArray>::BitShape>,
    (Self::WordType, Self::WordType): Element<Self::WordShape>,
{
    /// Generate sign and mask bits from counters for a binary classification.
    ///
    /// dist_matrix_counters is a 2d square symetrical matrix.
    /// The `x`th by `y`th entry is the number of examples in which the xth and yth bit were the same.
    ///
    /// value_counters is the number of time that each bit was set in each of the two classes.
    fn gen_filter(
        value_counters: &[(usize, <u32 as Element<Self::BitShape>>::Array); 2],
        dist_matrix_counters: &<<u32 as Element<Self::BitShape>>::Array as Element<
            Self::BitShape,
        >>::Array,
        n: usize,
    ) -> (
        <(Self::WordType, Self::WordType) as Element<Self::WordShape>>::Array,
        u32,
    );
}

pub trait UnsupervisedCluster<I: BitArray, O>
where
    u32: Element<I::BitShape>,
    [(usize, <u32 as Element<I::BitShape>>::Array); 2]: Element<I::BitShape>,
{
    fn unsupervised_cluster(
        counts: &<[(usize, <u32 as Element<I::BitShape>>::Array); 2] as Element<I::BitShape>>::Array,
        n_examples: usize,
    ) -> Self;
}

pub trait GenClassify<I: BitArray, C: Shape>
where
    (I::WordType, I::WordType): Element<I::WordShape>,
    (
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ): Element<C>,
{
    type Accumulator;
    fn gen_classify(
        acc: &Self::Accumulator,
    ) -> <(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ) as Element<C>>::Array;
}

pub struct SimpleClassify<I, const C: usize> {
    input: PhantomData<I>,
}

pub struct UnsupervisedClusterWeightsGen<I, O, const C: usize> {
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
