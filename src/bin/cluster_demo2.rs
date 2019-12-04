use bitnn::bits::{b4, b8, BitArray, BitArrayOPs, BitWord, Distance, IncrementCooccurrenceMatrix};
use bitnn::layer::CountBits;
use bitnn::shape::{Element, Flatten, Fold, IndexGet, Indexable, Map, Shape, ZipFold, ZipMap};
use bitnn::weight::FlipBool;

extern crate rand;
extern crate rand_hc;

use rand::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;

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
        counts: &Box<
            <[(usize, <u32 as Element<I::BitShape>>::Array); 2] as Element<I::BitShape>>::Array,
        >,
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
    <bool as bitnn::shape::Element<<I as bitnn::bits::BitArray>::BitShape>>::Array: std::fmt::Debug,
{
    fn unsupervised_cluster(
        counts: &Box<
            <[(usize, <u32 as Element<I::BitShape>>::Array); 2] as Element<I::BitShape>>::Array,
        >,
        n_examples: usize,
    ) -> Self {
        let distances = <Box<I::BitShape> as Map<
            [(usize, <u32 as Element<I::BitShape>>::Array); 2],
            [<[f32; 2] as Element<I::BitShape>>::Array; 2],
        >>::map(&counts, |row| {
            <[(); 2] as Map<
                (usize, <u32 as Element<I::BitShape>>::Array),
                <[f32; 2] as Element<I::BitShape>>::Array,
            >>::map(row, |(n, row)| {
                <I::BitShape as Map<u32, [f32; 2]>>::map(row, |&c| {
                    [
                        (*n as u32 - c) as f32 / (n_examples) as f32,
                        (c) as f32 / (n_examples) as f32,
                    ]
                })
            })
        });

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
        dbg!(&assignments);
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
                            dbg!(cur_loss);
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

type InputType = [b8; 2];
type TargetType = b4;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let examples: Vec<[b8; 2]> = vec![
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
    ];

    let mut counts = Box::<
        <[(
            usize,
            <u32 as Element<<InputType as BitArray>::BitShape>>::Array,
        ); 2] as Element<<InputType as BitArray>::BitShape>>::Array,
    >::default();
    for example in &examples {
        example.increment_cooccurrence_matrix(&mut *counts, example);
    }
    let weights = <[(
        <(
            <InputType as BitArray>::WordType,
            <InputType as BitArray>::WordType,
        ) as Element<<InputType as BitArray>::WordShape>>::Array,
        u32,
    ); TargetType::BIT_LEN] as UnsupervisedCluster<InputType, TargetType>>::unsupervised_cluster(
        &counts,
        examples.len(),
    );
    for cluster in &weights {
        println!("{:?}", cluster);
    }
    for example in &examples {
        let acts: Vec<_> = weights.iter().map(|x| x.0.distance(example)).collect();
        println!("{:?}", acts);
    }
}
