use crate::bits::{BitArray, BitArrayOPs, Distance};
use crate::shape::{Element, Fold, Map, Shape, ZipMap};
use std::boxed::Box;
use std::collections::HashSet;
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

pub trait Mse
where
    Self: Shape + Sized,
    bool: Element<Self>,
    f32: Element<Self>,
    <f32 as Element<Self>>::Array: Element<Self>,
{
    fn bit_flip_mses(
        edges: &Box<<<f32 as Element<Self>>::Array as Element<Self>>::Array>,
        local_avgs: &Box<<f32 as Element<Self>>::Array>,
        mask: &<bool as Element<Self>>::Array,
    ) -> <f32 as Element<Self>>::Array;
}

impl<
        S: Shape
            + Map<f32, f32>
            + ZipMap<f32, f32, f32>
            + ZipMap<bool, f32, f32>
            + ZipMap<f32, bool, f32>
            + ZipMap<f32, <f32 as Element<S>>::Array, <f32 as Element<S>>::Array>
            + Fold<<f32 as Element<S>>::Array, <f32 as Element<S>>::Array>
            + Fold<Box<<f32 as Element<S>>::Array>, <f32 as Element<S>>::Array>,
    > Mse for S
where
    bool: Element<S>,
    f32: Element<S>,
    <f32 as Element<S>>::Array: Element<S> + Sum<f32> + Default + std::fmt::Debug,
    Box<S>: Map<<f32 as Element<S>>::Array, <f32 as Element<S>>::Array>
        + ZipMap<<f32 as Element<S>>::Array, f32, <f32 as Element<S>>::Array>,
    Box<<<f32 as Element<S>>::Array as Element<S>>::Array>: std::fmt::Debug,
{
    fn bit_flip_mses(
        edges: &Box<<<f32 as Element<S>>::Array as Element<S>>::Array>,
        values: &Box<<f32 as Element<S>>::Array>,
        mask: &<bool as Element<S>>::Array,
    ) -> <f32 as Element<S>>::Array {
        let n: f32 = S::N as f32;
        let values_sum = values.sum();
        let bit_flip_local_counts = <Box<S> as Map<
            <f32 as Element<S>>::Array,
            <f32 as Element<S>>::Array,
        >>::map(&edges, |edge_set| {
            let sum = masked_sum::<S>(edge_set, mask);
            <S as ZipMap<bool, f32, f32>>::zip_map(&mask, &edge_set, |&mask_bit, &edge| {
                if mask_bit {
                    sum - edge
                } else {
                    sum + edge
                }
            })
        });
        //dbg!(&bit_flip_local_counts);
        let bit_flip_scales = {
            let bit_flip_sums =
                <S as Fold<<f32 as Element<S>>::Array, <f32 as Element<S>>::Array>>::fold(
                    &bit_flip_local_counts,
                    <f32 as Element<S>>::Array::default(),
                    |a, b| <S as ZipMap<f32, f32, f32>>::zip_map(&a, b, |x, y| x + y),
                );
            //dbg!(&bit_flip_local_counts);
            <S as Map<f32, f32>>::map(&bit_flip_sums, |sum| values_sum / sum)
        };
        let bit_flip_local_mses = <Box<S> as ZipMap<
            <f32 as Element<S>>::Array,
            f32,
            <f32 as Element<S>>::Array,
        >>::zip_map(
            &bit_flip_local_counts,
            values,
            |local_counts, local_avg| {
                <S as ZipMap<f32, f32, f32>>::zip_map(
                    local_counts,
                    &bit_flip_scales,
                    |count, scale| (local_avg - (count * scale)).powi(2),
                )
            },
        );
        //dbg!(&bit_flip_local_mses);
        let smes = <S as Fold<<f32 as Element<S>>::Array, <f32 as Element<S>>::Array>>::fold(
            &bit_flip_local_mses,
            <f32 as Element<S>>::Array::default(),
            |a, b| <S as ZipMap<f32, f32, f32>>::zip_map(&a, b, |x, y| x + y),
        );
        <S as Map<f32, f32>>::map(&smes, |sum| sum / n)
    }
}

fn masked_sum<S: Shape + ZipMap<f32, bool, f32>>(
    edge_set: &<f32 as Element<S>>::Array,
    mask: &<bool as Element<S>>::Array,
) -> f32
where
    bool: Element<S>,
    f32: Element<S>,
    <f32 as Element<S>>::Array: Sum<f32>,
{
    <S as ZipMap<f32, bool, f32>>::zip_map(
        &edge_set,
        &mask,
        |&edge, &mask_bit| if mask_bit { edge } else { 0_f32 },
    )
    .sum()
}

pub trait GenWeights
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
    fn gen_weights(
        value_counters: &Box<[(usize, <u32 as Element<Self::BitShape>>::Array); 2]>,
        dist_matrix_counters: &Box<
            <<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array,
        >,
        n: usize,
    ) -> (
        <(Self::WordType, Self::WordType) as Element<Self::WordShape>>::Array,
        u32,
    );
}

impl<B: BitArray + BitArrayOPs> GenWeights for B
where
    B::BitShape: Mse
        + Min
        + FlipBool
        + Map<(), bool>
        + Map<u32, f32>
        + ZipMap<bool, u32, f32>
        + ZipMap<u32, u32, f32>
        + ZipMap<u32, u32, bool>
        + Element<<B as BitArray>::BitShape>
        + Default,
    B::WordType: Copy + Element<B::WordShape, Array = B>,
    B::WordShape: ZipMap<B::WordType, B::WordType, (B::WordType, B::WordType)>,
    (Self::WordType, Self::WordType): Element<Self::WordShape>,
    Box<B::BitShape>: Map<<u32 as Element<B::BitShape>>::Array, <f32 as Element<B::BitShape>>::Array>
        + ZipMap<bool, <u32 as Element<B::BitShape>>::Array, <f32 as Element<B::BitShape>>::Array>,
    <B::BitShape as Element<<B as BitArray>::BitShape>>::Array: Shape,
    u32: Element<B::BitShape> + Element<<B::BitShape as Element<B::BitShape>>::Array>,
    f32: Element<B::BitShape> + Element<<B::BitShape as Element<B::BitShape>>::Array>,
    <u32 as Element<B::BitShape>>::Array: Element<<B as BitArray>::BitShape>,
    <f32 as Element<B::BitShape>>::Array:
        Element<<B as BitArray>::BitShape> + Sum<f32> + std::fmt::Debug,
    <<f32 as Element<B::BitShape>>::Array as Element<B::BitShape>>::Array:
        Default + std::fmt::Debug,
    bool: Element<B::BitShape>,
    (): Element<B::BitShape, Array = B::BitShape>,
    <B::BitShape as Shape>::Index: std::fmt::Debug,
    <bool as Element<B::BitShape>>::Array: std::fmt::Debug,
    <(B::WordType, B::WordType) as Element<B::WordShape>>::Array: Distance<Rhs = B>,
{
    fn gen_weights(
        value_counters: &Box<[(usize, <u32 as Element<<B as BitArray>::BitShape>>::Array); 2]>,
        dist_matrix_counters: &Box<
            <<u32 as Element<B::BitShape>>::Array as Element<B::BitShape>>::Array,
        >,
        n_examples: usize,
    ) -> (
        <(Self::WordType, Self::WordType) as Element<Self::WordShape>>::Array,
        u32,
    ) {
        let na = (value_counters[0].0 + 1) as f32;
        let nb = (value_counters[1].0 + 1) as f32;
        let sign_bools = Box::new(
            <<B as BitArray>::BitShape as ZipMap<u32, u32, bool>>::zip_map(
                &value_counters[0].1,
                &value_counters[1].1,
                |&a, &b| ((a + 1) as f32 / na) > ((b + 1) as f32 / nb),
            ),
        );
        //dbg!(&sign_bools);
        let values = Box::new(<B::BitShape as ZipMap<u32, u32, f32>>::zip_map(
            &value_counters[0].1,
            &value_counters[1].1,
            |&a, &b| {
                if ((a + 1) as f32 / na) > ((b + 1) as f32 / nb) {
                    (a + 1) as f32 / (a + b + 2) as f32
                } else {
                    let a = value_counters[0].0 - a as usize;
                    let b = value_counters[1].0 - b as usize;
                    (a + 1) as f32 / (a + b + 2) as f32
                }
            },
        ));

        //dbg!(&values);
        let edges = <Box<B::BitShape> as ZipMap<
            bool,
            <u32 as Element<B::BitShape>>::Array,
            <f32 as Element<B::BitShape>>::Array,
        >>::zip_map(&sign_bools, dist_matrix_counters, |outer_sign, row| {
            <B::BitShape as ZipMap<bool, u32, f32>>::zip_map(
                &sign_bools,
                row,
                |inner_sign, &count| {
                    if outer_sign ^ inner_sign {
                        count as f32 / n_examples as f32
                    } else {
                        (n_examples - count as usize) as f32 / n_examples as f32
                    }
                },
            )
        });
        let mut mask = <B::BitShape as Map<(), bool>>::map(&B::BitShape::default(), |_| true);
        // false: 18.855%
        // true:  49.566666%
        let mut cur_mse = std::f32::INFINITY;
        let mut is_optima = false;
        while !is_optima {
            let bit_flip_mses = B::BitShape::bit_flip_mses(&edges, &values, &mask);
            let (min_index, min_val) = <B::BitShape as Min>::min(&bit_flip_mses).unwrap();
            if min_val < cur_mse {
                <B::BitShape as FlipBool>::flip_bool(&mut mask, min_index);
                cur_mse = min_val;
            } else {
                is_optima = true;
            }
        }
        //dbg!(cur_mse);
        //dbg!(n_updates);
        let avg_input = {
            let threshold = ((value_counters[1].0 + value_counters[0].0) / 2) as u32;
            let avg_bools = Box::new(
                <<B as BitArray>::BitShape as ZipMap<u32, u32, bool>>::zip_map(
                    &value_counters[0].1,
                    &value_counters[1].1,
                    |&a, &b| (a + b) > threshold,
                ),
            );
            B::bitpack(&avg_bools)
        };
        let weights = <B::WordShape as ZipMap<
            B::WordType,
            B::WordType,
            (B::WordType, B::WordType),
        >>::zip_map(
            &B::bitpack(&sign_bools),
            &B::bitpack(&mask),
            |&sign_word, &mask_word| (sign_word, mask_word),
        );
        // the activation of the average input.
        let avg_act = weights.distance(&avg_input);
        (weights, avg_act)
    }
}

pub fn gen_partitions(depth: usize) -> Vec<HashSet<usize>> {
    assert_ne!(depth, 0);
    if depth == 1 {
        vec![HashSet::new()]
    } else {
        let a = gen_partitions(depth - 1);
        a.iter()
            .cloned()
            .chain(a.iter().cloned().map(|mut x| {
                x.insert(depth - 1);
                x
            }))
            .collect()
    }
}
