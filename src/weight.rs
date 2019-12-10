use crate::bits::{BitArray, BitArrayOPs, BitWord, Distance};
use crate::count::ElementwiseAdd;
use crate::shape::{Element, Flatten, Fold, Map, MapMut, Shape, ZipMap, ZipMapMut};
use rayon::prelude::*;
use std::boxed::Box;
use std::collections::HashSet;
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

impl<I: BitArray + GenFilterSupervised, O: BitArray, const C: usize> GenWeights<I, O>
    for SupervisedWeightsGen<I, O, { C }>
where
    u32: Element<I::BitShape>,
    <u32 as Element<I::BitShape>>::Array: Element<I::BitShape>,
    (I::WordType, I::WordType): Element<I::WordShape>,
    (
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ): Element<O::BitShape>,
    (usize, <u32 as Element<I::BitShape>>::Array): Default + ElementwiseAdd,
    <(I::WordType, I::WordType) as Element<I::WordShape>>::Array: Send + Sync + Copy,
    O::BitShape: Flatten<(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    )>,
    (
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ): Element<O::BitShape>,
    Box<[(usize, <u32 as Element<I::BitShape>>::Array); 2]>: Send + Sync,
    <u32 as Element<I::BitShape>>::Array: Send + Sync,
    <<u32 as Element<I::BitShape>>::Array as Element<I::BitShape>>::Array: Send + Sync,
{
    type Accumulator = Box<(
        [(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C],
        <<u32 as Element<I::BitShape>>::Array as Element<I::BitShape>>::Array,
        usize,
    )>;
    fn gen_weights(
        acc: &Box<(
            [(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C],
            <<u32 as Element<I::BitShape>>::Array as Element<I::BitShape>>::Array,
            usize,
        )>,
    ) -> <(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ) as Element<O::BitShape>>::Array {
        let mut partitions = gen_partitions(C);
        partitions.sort_by_key(|x| x.len().min(C - x.len()));
        let partitions = &partitions[0..<O as BitArray>::BitShape::N];
        let weights: Vec<_> = partitions
            .par_iter()
            .map(|partition| {
                let mut split_counters =
                    Box::<[(usize, <u32 as Element<I::BitShape>>::Array); 2]>::default();
                for (class, class_counter) in acc.0.iter().enumerate() {
                    split_counters[partition.contains(&class) as usize]
                        .elementwise_add(class_counter);
                }
                <I as GenFilterSupervised>::gen_filter(&split_counters, &acc.1, acc.2)
            })
            .collect();
        <O as BitArray>::BitShape::from_vec(&weights)
    }
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

impl<B: BitArray + BitArrayOPs> GenFilterSupervised for B
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
    B::BitShape:
        ZipMapMut<bool, <u32 as Element<B::BitShape>>::Array, <f32 as Element<B::BitShape>>::Array>,
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
    fn gen_filter(
        value_counters: &[(usize, <u32 as Element<<B as BitArray>::BitShape>>::Array); 2],
        dist_matrix_counters: &<<u32 as Element<B::BitShape>>::Array as Element<B::BitShape>>::Array,
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
        let edges = {
            let mut target = Box::<
                <<f32 as Element<B::BitShape>>::Array as Element<B::BitShape>>::Array,
            >::default();
            <B::BitShape as ZipMapMut<
                bool,
                <u32 as Element<B::BitShape>>::Array,
                <f32 as Element<B::BitShape>>::Array,
            >>::zip_map_mut(
                &mut target,
                &sign_bools,
                dist_matrix_counters,
                |target, outer_sign, row| {
                    *target = <B::BitShape as ZipMap<bool, u32, f32>>::zip_map(
                        &sign_bools,
                        row,
                        |inner_sign, &count| {
                            if outer_sign ^ inner_sign {
                                count as f32 / n_examples as f32
                            } else {
                                (n_examples - count as usize) as f32 / n_examples as f32
                            }
                        },
                    );
                },
            );
            target
        };
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

/// `distances` is the square symmetrical matrix of distances between bits.
/// The value of distances[x][1][y][0] is the proportion of examples in which bit x is set and bit y is unset.
/// Given N input bits, and C clusters,
/// - the outer `Vec` must be of length N,
/// - the inner `Vec`s must be of length N,
///
/// `signs` is the allocation and sign of each of the input bits.
/// It must be of length N.
/// Each usize is of range 0-C.
fn cluster_mse(
    distances: &Vec<[Vec<[f32; 2]>; 2]>,
    assignments: &Vec<(usize, bool)>,
    n: usize,
    c: usize,
) -> f32 {
    assert_eq!(n, distances.len());
    assert_eq!(n, assignments.len());
    let clusters: Vec<Vec<(usize, bool)>> = assignments.iter().enumerate().fold(
        (0..c).map(|_| Vec::new()).collect(),
        |mut acc, (bit_index, (cluster_index, sign))| {
            acc[*cluster_index].push((bit_index, *sign));
            acc
        },
    );
    //dbg!(&clusters);
    let size_loss: f32 = clusters
        .iter()
        .map(|cluster| (cluster.len() as f32 / n as f32).powi(2))
        .sum();
    //dbg!(size_loss);
    let sum_cluster_loss: f32 = clusters
        .iter()
        .map(|cluster| {
            let cluster_loss: f32 = cluster
                .iter()
                .map(|&(x_index, x_sign)| {
                    cluster
                        .iter()
                        .map(|&(y_index, y_sign)| {
                            distances[x_index][x_sign as usize][y_index][y_sign as usize]
                        })
                        .sum::<f32>()
                    // cluster.len() as f32
                })
                .sum::<f32>()
                / cluster.len() as f32;
            cluster_loss.powi(2)
        })
        .sum();
    //dbg!(sum_cluster_loss);
    size_loss + sum_cluster_loss
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

impl<I: BitArray + BitWord + BitArrayOPs, O: BitArray + BitWord> UnsupervisedCluster<I, O>
    for <(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ) as Element<O::BitShape>>::Array
where
    O::BitShape: Flatten<(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    )>,
    I::BitShape: Map<u32, [f32; 2]>
        + Flatten<[f32; 2]>
        + Flatten<[<[f32; 2] as Element<I::BitShape>>::Array; 2]>
        + Flatten<bool>
        + Map<[(usize, <u32 as Element<I::BitShape>>::Array); 2], bool>,
    Box<I::BitShape>: Map<
        [(usize, <u32 as Element<I::BitShape>>::Array); 2],
        [<[f32; 2] as Element<I::BitShape>>::Array; 2],
    >,
    I::BitShape: MapMut<
        [(usize, <u32 as Element<I::BitShape>>::Array); 2],
        [<[f32; 2] as Element<I::BitShape>>::Array; 2],
    >,
    I::WordType: Copy,
    I::WordShape: ZipMap<I::WordType, I::WordType, (I::WordType, I::WordType)>,
    [f32; 2]: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    [(usize, <u32 as Element<I::BitShape>>::Array); 2]: Element<I::BitShape>,
    [<[f32; 2] as Element<I::BitShape>>::Array; 2]: Element<I::BitShape> + Copy,
    (I::WordType, I::WordType): Element<I::WordShape>,
    <(I::WordType, I::WordType) as Element<I::WordShape>>::Array: Default + Distance<Rhs = I>,
    (
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ): Element<O::BitShape> + Copy,
    <[f32; 2] as Element<I::BitShape>>::Array: Default,
    <[<[f32; 2] as Element<I::BitShape>>::Array; 2] as Element<I::BitShape>>::Array: Default,
    bool: Element<I::BitShape>,
    <bool as Element<<I as BitArray>::BitShape>>::Array: std::fmt::Debug,
{
    fn unsupervised_cluster(
        counts: &<[(usize, <u32 as Element<I::BitShape>>::Array); 2] as Element<I::BitShape>>::Array,
        n_examples: usize,
    ) -> Self {
        let distances = {
            let mut target = Box::<
                <[<[f32; 2] as Element<I::BitShape>>::Array; 2] as Element<I::BitShape>>::Array,
            >::default();
            <I::BitShape as MapMut<
                [(usize, <u32 as Element<I::BitShape>>::Array); 2],
                [<[f32; 2] as Element<I::BitShape>>::Array; 2],
            >>::map_mut(&mut target, &counts, |target, row| {
                *target = <[(); 2] as Map<
                    (usize, <u32 as Element<I::BitShape>>::Array),
                    <[f32; 2] as Element<I::BitShape>>::Array,
                >>::map(row, |(n, row)| {
                    <I::BitShape as Map<u32, [f32; 2]>>::map(row, |&c| {
                        [
                            (*n as u32 - c) as f32 / (n_examples) as f32,
                            (c) as f32 / (n_examples) as f32,
                        ]
                    })
                });
            });
            target
        };

        let vec_distances = {
            {
                let mut target: Vec<[<[f32; 2] as Element<I::BitShape>>::Array; 2]> = (0
                    ..I::BitShape::N)
                    .map(|_| <[<[f32; 2] as Element<I::BitShape>>::Array; 2]>::default())
                    .collect();
                <I::BitShape as Flatten<[<[f32; 2] as Element<I::BitShape>>::Array; 2]>>::to_vec(
                    &distances,
                    &mut target,
                );
                target
            }
            .iter()
            .map(|x| {
                let mut target: [Vec<[f32; 2]>; 2] = [
                    vec![[0f32; 2]; I::BitShape::N],
                    vec![[0f32; 2]; I::BitShape::N],
                ];
                for i in 0..2 {
                    <I::BitShape as Flatten<[f32; 2]>>::to_vec(&x[i], &mut target[i]);
                }
                target
            })
            .collect()
        };

        let mut assignments: Vec<(usize, bool)> = (0..I::BIT_LEN)
            .map(|i| (i % O::BIT_LEN, ((i / O::BIT_LEN) % 2) == 0))
            .collect();
        let mut cur_loss = cluster_mse(&vec_distances, &assignments, I::BIT_LEN, O::BIT_LEN);
        dbg!(cur_loss);
        for i in 0..3 {
            dbg!(i);
            for b in 0..I::BIT_LEN {
                for c in 0..O::BIT_LEN {
                    for &sign in &[true, false] {
                        let old_state = assignments[b];
                        assignments[b] = (c, sign);
                        let new_loss =
                            cluster_mse(&vec_distances, &assignments, I::BIT_LEN, O::BIT_LEN);
                        if new_loss < cur_loss {
                            cur_loss = new_loss;
                            //dbg!(cur_loss);
                        } else {
                            assignments[b] = old_state;
                        }
                    }
                }
            }
        }
        dbg!(cur_loss);
        let weights = {
            let avg_input = {
                let bools = <I::BitShape as Map<
                    [(usize, <u32 as Element<I::BitShape>>::Array); 2],
                    bool,
                >>::map(counts, |[(unset, _), (set, _)]| unset > set);
                I::bitpack(&bools)
            };

            let mut mask_target = vec![vec![false; I::BIT_LEN]; O::BIT_LEN];
            let mut signs_target = vec![vec![false; I::BIT_LEN]; O::BIT_LEN];
            for (i, &(c, sign)) in assignments.iter().enumerate() {
                mask_target[c][i] = true;
                signs_target[c][i] = sign;
            }
            let mask_bits_array: Vec<_> = signs_target
                .iter()
                .zip(mask_target.iter())
                .map(|(signs, mask)| {
                    let sign_bits = <I>::bitpack(&<I::BitShape>::from_vec(signs));
                    let mask_bits = <I>::bitpack(&<I::BitShape>::from_vec(mask));
                    let filter = <I::WordShape as ZipMap<
                        I::WordType,
                        I::WordType,
                        (I::WordType, I::WordType),
                    >>::zip_map(
                        &sign_bits,
                        &mask_bits,
                        |&sign_word, &mask_word| (sign_word, mask_word),
                    );
                    let dist: u32 = filter.distance(&avg_input);
                    (filter, dist)
                })
                .collect();
            <O::BitShape>::from_vec(&mask_bits_array)
        };
        weights
    }
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

impl<I: BitArray + GenFilterSupervised, const C: usize> GenClassify<I, [(); C]>
    for SimpleClassify<I, C>
where
    (I::WordType, I::WordType): Element<I::WordShape>,
    (
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ): Element<[(); C]> + Copy,
    u32: Element<I::BitShape>,
    <u32 as Element<I::BitShape>>::Array: Element<I::BitShape>,
    Box<[(usize, <u32 as Element<I::BitShape>>::Array); 2]>: Default,
    <(I::WordType, I::WordType) as Element<I::WordShape>>::Array: Send + Sync,
    (usize, <u32 as Element<I::BitShape>>::Array): ElementwiseAdd,
    [(); C]: Flatten<(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    )>,
    Box<(
        [(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C],
        <<u32 as Element<I::BitShape>>::Array as Element<I::BitShape>>::Array,
        usize,
    )>: Send + Sync,
    <(I::WordType, I::WordType) as Element<I::WordShape>>::Array: Copy,
{
    type Accumulator = Box<(
        [(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C],
        <<u32 as Element<I::BitShape>>::Array as Element<I::BitShape>>::Array,
        usize,
    )>;
    fn gen_classify(
        acc: &Self::Accumulator,
    ) -> <(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ) as Element<[(); C]>>::Array {
        let classes: Vec<usize> = (0..C).collect();
        let weights: Vec<_> = classes
            .par_iter()
            .map(|&class| {
                let mut split_counters =
                    Box::<[(usize, <u32 as Element<I::BitShape>>::Array); 2]>::default();
                for (part_class, class_counter) in acc.0.iter().enumerate() {
                    split_counters[(part_class == class) as usize].elementwise_add(class_counter);
                }
                <I as GenFilterSupervised>::gen_filter(&split_counters, &acc.1, acc.2)
            })
            .collect();
        let max_act: u32 = *weights.iter().map(|(_, t)| t).max().unwrap();
        let weights: Vec<_> = weights
            .iter()
            .map(|(weights, threshold)| (*weights, max_act - threshold))
            .collect();
        let weights = <[(); C] as Flatten<(
            <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
            u32,
        )>>::from_vec(&weights);
        weights
    }
}

pub struct UnsupervisedClusterWeightsGen<I, O, const C: usize> {
    input: PhantomData<I>,
    target: PhantomData<O>,
}

impl<I: BitArray, O: BitArray, const C: usize> GenWeights<I, O>
    for UnsupervisedClusterWeightsGen<I, O, C>
where
    (I::WordType, I::WordType): Element<I::WordShape>,
    (
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ): Element<O::BitShape>,
    u32: Element<I::BitShape>,
    [(usize, <u32 as Element<I::BitShape>>::Array); 2]: Element<I::BitShape>,
    <(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ) as Element<O::BitShape>>::Array: UnsupervisedCluster<I, O>,
{
    type Accumulator = Box<(
        <[(usize, <u32 as Element<I::BitShape>>::Array); 2] as Element<I::BitShape>>::Array,
        usize,
    )>;
    fn gen_weights(
        acc: &Self::Accumulator,
    ) -> <(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ) as Element<O::BitShape>>::Array {
        <<(
            <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
            u32,
        ) as Element<O::BitShape>>::Array as UnsupervisedCluster<I, O>>::unsupervised_cluster(
            &acc.0, acc.1,
        )
    }
}
