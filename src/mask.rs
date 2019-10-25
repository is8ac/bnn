use crate::bits::BitArray;
use crate::shape::{Element, Fold, Map, Shape, ZipMap};
use std::boxed::Box;
use std::ops::AddAssign;

// f64 values
pub trait Sum<T> {
    fn sum(&self) -> T;
}

impl Sum<f64> for f64 {
    fn sum(&self) -> f64 {
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
    f64: Element<Self>,
{
    fn min(values: &<f64 as Element<Self>>::Array) -> Option<(Self::Index, f64)>;
}

impl Min for () {
    fn min(&values: &f64) -> Option<((), f64)> {
        Some(((), values))
    }
}

impl<T: Min, const L: usize> Min for [T; L]
where
    f64: Element<T>,
    Self: Shape<Index = (usize, T::Index)>,
    T::Index: Copy,
{
    fn min(values: &[<f64 as Element<T>>::Array; L]) -> Option<((usize, T::Index), f64)> {
        let mut cur_min: Option<((usize, T::Index), f64)> = None;
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

fn bayes_magn<S: Shape + ZipMap<u32, u32, f64>>(
    counters: &[(usize, <u32 as Element<S>>::Array); 2],
) -> <f64 as Element<S>>::Array
where
    u32: Element<S>,
    f64: Element<S>,
{
    <S as ZipMap<u32, u32, f64>>::zip_map(&counters[0].1, &counters[1].1, |&a, &b| {
        let pab = (a + 1) as f64 / (a + b + 2) as f64;
        ((pab * 2f64) - 1f64).abs()
    })
}

pub trait Mse
where
    Self: Shape + Sized,
    bool: Element<Self>,
    f64: Element<Self>,
    <f64 as Element<Self>>::Array: Element<Self>,
{
    fn bit_flip_mses(
        edges: &Box<<<f64 as Element<Self>>::Array as Element<Self>>::Array>,
        local_avgs: &Box<<f64 as Element<Self>>::Array>,
        mask: &<bool as Element<Self>>::Array,
    ) -> <f64 as Element<Self>>::Array;
}

impl<
        S: Shape
            + Map<f64, f64>
            + ZipMap<f64, f64, f64>
            + ZipMap<bool, f64, f64>
            + ZipMap<f64, bool, f64>
            + Fold<<f64 as Element<S>>::Array, <f64 as Element<S>>::Array>
            + Fold<Box<<f64 as Element<S>>::Array>, <f64 as Element<S>>::Array>,
    > Mse for S
where
    bool: Element<S>,
    f64: Element<S>,
    <f64 as Element<S>>::Array: Element<S> + Sum<f64> + Default + std::fmt::Debug,
    Box<S>: Map<<f64 as Element<S>>::Array, <f64 as Element<S>>::Array>
        + ZipMap<<f64 as Element<S>>::Array, f64, <f64 as Element<S>>::Array>,
    Box<<<f64 as Element<S>>::Array as Element<S>>::Array>: std::fmt::Debug,
{
    fn bit_flip_mses(
        edges: &Box<<<f64 as Element<S>>::Array as Element<S>>::Array>,
        values: &Box<<f64 as Element<S>>::Array>,
        mask: &<bool as Element<S>>::Array,
    ) -> <f64 as Element<S>>::Array {
        let n: f64 = S::N as f64;
        let sum: f64 = values.sum();
        let avg = sum / n;
        let bit_flip_local_counts =
            <Box<S> as Map<<f64 as Element<S>>::Array, <f64 as Element<S>>::Array>>::map(
                &edges,
                |edge_set| {
                    let sum = masked_sum::<S>(edge_set, mask);
                    <S as ZipMap<bool, f64, f64>>::zip_map(
                        &mask,
                        &edge_set,
                        |&mask_bit, &edge| if mask_bit { sum - edge } else { sum + edge },
                    )
                },
            );
        //dbg!(&bit_flip_local_counts);
        let bit_flip_sums =
            <S as Fold<<f64 as Element<S>>::Array, <f64 as Element<S>>::Array>>::fold(
                &bit_flip_local_counts,
                <f64 as Element<S>>::Array::default(),
                |a, b| <S as ZipMap<f64, f64, f64>>::zip_map(&a, b, |x, y| x + y),
            );
        //dbg!(&bit_flip_local_counts);
        let bit_flip_scales = <S as Map<f64, f64>>::map(&bit_flip_sums, |sum| avg / (sum / n));
        //dbg!(&bit_flip_scales);
        let bit_flip_local_mses = <Box<S> as ZipMap<
            <f64 as Element<S>>::Array,
            f64,
            <f64 as Element<S>>::Array,
        >>::zip_map(
            &bit_flip_local_counts,
            values,
            |local_counts, local_avg| {
                <S as ZipMap<f64, f64, f64>>::zip_map(
                    local_counts,
                    &bit_flip_scales,
                    |count, scale| (local_avg - (count * scale)).powi(2),
                )
            },
        );
        //dbg!(&bit_flip_local_mses);
        <S as Fold<<f64 as Element<S>>::Array, <f64 as Element<S>>::Array>>::fold(
            &bit_flip_local_mses,
            <f64 as Element<S>>::Array::default(),
            |a, b| <S as ZipMap<f64, f64, f64>>::zip_map(&a, b, |x, y| x + y),
        )
    }
}

fn masked_sum<S: Shape + ZipMap<f64, bool, f64>>(
    edge_set: &<f64 as Element<S>>::Array,
    mask: &<bool as Element<S>>::Array,
) -> f64
where
    bool: Element<S>,
    f64: Element<S>,
    <f64 as Element<S>>::Array: Sum<f64>,
{
    <S as ZipMap<f64, bool, f64>>::zip_map(
        &edge_set,
        &mask,
        |&edge, &mask_bit| if mask_bit { edge } else { 0f64 },
    )
    .sum()
}

pub trait GenMask
where
    Self: BitArray,
    bool: Element<Self::BitShape>,
    u32: Element<Self::BitShape>,
    <u32 as Element<<Self as BitArray>::BitShape>>::Array:
        Element<<Self as BitArray>::BitShape>,
{
    fn gen_mask(
        dist_matrix_counters: &Box<
            <<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array,
        >,
        n: usize,
        value_counters: &Box<[(usize, <u32 as Element<Self::BitShape>>::Array); 2]>,
    ) -> Self;
}

impl<B: BitArray> GenMask for B
where
    B::BitShape: Mse
        + Min
        + FlipBool
        + Map<u32, f64>
        + ZipMap<u32, u32, f64>
        + Element<<B as BitArray>::BitShape>
        + Map<(), bool>
        + Default,
    Box<B::BitShape>:
        Map<<u32 as Element<B::BitShape>>::Array, <f64 as Element<B::BitShape>>::Array>,
    <B::BitShape as Element<<B as BitArray>::BitShape>>::Array: Shape,
    u32: Element<B::BitShape> + Element<<B::BitShape as Element<B::BitShape>>::Array>,
    f64: Element<B::BitShape> + Element<<B::BitShape as Element<B::BitShape>>::Array>,
    <u32 as Element<B::BitShape>>::Array: Element<<B as BitArray>::BitShape>,
    <f64 as Element<B::BitShape>>::Array:
        Element<<B as BitArray>::BitShape> + Sum<f64> + std::fmt::Debug,
    <<f64 as Element<B::BitShape>>::Array as Element<B::BitShape>>::Array:
        Default + std::fmt::Debug,
    bool: Element<B::BitShape>,
    (): Element<B::BitShape, Array = B::BitShape>,
    <B::BitShape as Shape>::Index: std::fmt::Debug,
{
    fn gen_mask(
        dist_matrix_counters: &Box<
            <<u32 as Element<B::BitShape>>::Array as Element<B::BitShape>>::Array,
        >,
        n_examples: usize,
        value_counters: &Box<[(usize, <u32 as Element<<B as BitArray>::BitShape>>::Array); 2]>,
    ) -> B {
        let values = Box::new(bayes_magn::<B::BitShape>(&value_counters));
        //dbg!(&values);
        let n_examples = n_examples as f64;
        let edges = <Box<B::BitShape> as Map<
            <u32 as Element<B::BitShape>>::Array,
            <f64 as Element<B::BitShape>>::Array,
        >>::map(dist_matrix_counters, |row| {
            <B::BitShape as Map<u32, f64>>::map(row, |&count| {
                ((count as f64 / n_examples) * 2f64 - 1f64).abs()
            })
        });
        let mut mask = <B::BitShape as Map<(), bool>>::map(&B::BitShape::default(), |_| true);
        // false: 30.5%
        // true:  33.6%
        let mut cur_mse = std::f64::INFINITY;
        let mut is_optima = false;
        let mut n_updates = 0;
        while !is_optima {
            let bit_flip_mses = B::BitShape::bit_flip_mses(&edges, &values, &mask);
            let (min_index, min_val) = <B::BitShape as Min>::min(&bit_flip_mses).unwrap();
            if min_val < cur_mse {
                <B::BitShape as FlipBool>::flip_bool(&mut mask, min_index);
                cur_mse = min_val;
                n_updates += 1;
            } else {
                is_optima = true;
            }
        }
        dbg!(cur_mse);
        dbg!(n_updates);
        B::bitpack(&mask)
    }
}
