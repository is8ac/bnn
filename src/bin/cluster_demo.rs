use bitnn::bits::{b8, BitArrayOPs, IncrementCooccurrenceMatrix};
use bitnn::layer::CountBits;
use bitnn::shape::{Element, Fold, Indexable, Map, Shape, ZipFold, ZipMap};
use bitnn::weight::FlipBool;

extern crate rand;
extern crate rand_hc;

use rand::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;

trait ClusteringMse<T: Shape>
where
    Self: Shape + Sized,
    bool: Element<Self>,
    f32: Element<Self>,
    [f32; 2]: Element<Self>,
    <bool as Element<Self>>::Array: Element<T>,
    [<f32 as Element<Self>>::Array; 2]: Element<Self>,
    [<[f32; 2] as Element<Self>>::Array; 2]: Element<Self>,
{
    fn clustering_mse(
        distances: &Box<<[<[f32; 2] as Element<Self>>::Array; 2] as Element<Self>>::Array>,
        signs: &<<bool as Element<Self>>::Array as Element<T>>::Array,
    ) -> f32;
}

impl<
        S: Shape
            + Sized
            + ZipFold<u32, bool, bool>
            + ZipFold<f32, bool, f32>
            + ZipFold<f32, bool, [<[f32; 2] as Element<S>>::Array; 2]>
            + ZipFold<f32, bool, [f32; 2]>
            + ZipFold<f32, bool, [<f32 as Element<S>>::Array; 2]>,
        T: Shape + Fold<f32, <bool as Element<S>>::Array>,
    > ClusteringMse<T> for S
where
    bool: Element<Self>,
    f32: Element<Self>,
    <bool as Element<Self>>::Array: Element<T>,
    <f32 as Element<Self>>::Array: Element<S>,
    [f32; 2]: Element<Self>,
    [<f32 as Element<Self>>::Array; 2]: Element<S>,
    [<[f32; 2] as Element<Self>>::Array; 2]: Element<Self>,
{
    fn clustering_mse(
        distances: &Box<<[<[f32; 2] as Element<S>>::Array; 2] as Element<S>>::Array>,
        signs: &<<bool as Element<S>>::Array as Element<T>>::Array,
    ) -> f32 {
        <T as Fold<f32, <bool as Element<S>>::Array>>::fold(&signs, 0f32, |acc, feature_signs| {
            let intra_distance =
                <S as ZipFold<f32, bool, [<[f32; 2] as Element<S>>::Array; 2]>>::zip_fold(
                    feature_signs,
                    &distances,
                    0f32,
                    |acc, &source_sign, edges| {
                        acc + <S as ZipFold<f32, bool, [f32; 2]>>::zip_fold(
                            feature_signs,
                            &edges[source_sign as usize],
                            0f32,
                            |acc, &target_sign, &edge| edge[target_sign as usize],
                        )
                    },
                ) / InputShape::N as f32;
            let inter_closeness = <T as Fold<f32, <bool as Element<S>>::Array>>::fold(
                &signs,
                0f32,
                |acc, other_signs| {
                    acc + <S as ZipFold<u32, bool, bool>>::zip_fold(
                        feature_signs,
                        other_signs,
                        0u32,
                        |acc, a, b| acc + (a == b) as u32,
                    ) as f32
                        / InputShape::N as f32
                },
            ) / TargetShape::N as f32;
            //dbg!((intra_distance, inter_closeness));
            acc + inter_closeness.powi(2) + intra_distance.powi(2)
        }) / TargetShape::N as f32
    }
}

const N_CLASSES: usize = 6;
type InputShape = [[(); 8]; 1];
type TargetShape = [(); N_CLASSES];

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let examples: Vec<[b8; 1]> = vec![
        [b8(0b_1111_0000)],
        [b8(0b_1111_0000)],
        [b8(0b_1111_0000)],
        [b8(0b_1111_0000)],
        [b8(0b_1111_0000)],
        [b8(0b_1111_0000)],
        [b8(0b_1111_0000)],
        [b8(0b_1111_0000)],
        [b8(0b_1100_1100)],
        [b8(0b_1100_1100)],
        [b8(0b_1100_1100)],
        [b8(0b_1100_1100)],
        [b8(0b_1100_1100)],
        [b8(0b_1100_1100)],
        [b8(0b_1100_1100)],
        [b8(0b_1100_1100)],
        [b8(0b_1010_1010)],
        [b8(0b_1010_1010)],
        [b8(0b_1010_1010)],
        [b8(0b_1010_1010)],
        [b8(0b_1010_1010)],
        [b8(0b_1010_1010)],
        [b8(0b_1010_1010)],
        [b8(0b_1010_1010)],
        [b8(0b_1010_1010)],
    ];

    let mut counts = Box::<
        <[(usize, <u32 as Element<InputShape>>::Array); 2] as Element<InputShape>>::Array,
    >::default();
    for example in &examples {
        example.increment_cooccurrence_matrix(&mut *counts, example);
    }
    dbg!(&counts);
    let n_examples = examples.len();
    dbg!(n_examples);
    let distances = <Box<InputShape> as Map<
        [(usize, <u32 as Element<InputShape>>::Array); 2],
        [<[f32; 2] as Element<InputShape>>::Array; 2],
    >>::map(&counts, |row| {
        <[(); 2] as Map<
            (usize, <u32 as Element<InputShape>>::Array),
            <[f32; 2] as Element<InputShape>>::Array,
        >>::map(row, |(n, row)| {
            <InputShape as Map<u32, [f32; 2]>>::map(row, |&c| {
                [
                    (*n as u32 - c) as f32 / (n_examples) as f32,
                    (c) as f32 / (n_examples) as f32,
                ]
            })
        })
    });
    dbg!(&distances);
    let mut signs = <TargetShape as Map<(), <bool as Element<InputShape>>::Array>>::map(
        &TargetShape::default(),
        |_| <InputShape as Map<(), bool>>::map(&InputShape::default(), |_| true),
    );
    let mut indices = <InputShape as Element<TargetShape>>::Array::indices();
    indices.shuffle(&mut rng);
    let mut cur_loss =
        <InputShape as ClusteringMse<TargetShape>>::clustering_mse(&distances, &signs);
    dbg!(cur_loss);
    for i in 0..10 {
        dbg!(i);
        for index in &indices {
            <InputShape as Element<TargetShape>>::Array::flip_bool(&mut signs, *index);
            let new_loss = <InputShape as ClusteringMse<TargetShape>>::clustering_mse(&distances, &signs);
            if new_loss < cur_loss {
                cur_loss = new_loss;
                dbg!(new_loss);
                let bits = <[[b8; 1]; N_CLASSES]>::bitpack(&signs);
                for feature in &bits {
                    println!("{:?}", feature);
                }
            } else {
                <InputShape as Element<TargetShape>>::Array::flip_bool(&mut signs, *index);
            }
        }
    }
    let bits = <[[b8; 1]; N_CLASSES]>::bitpack(&signs);
    for feature in &bits {
        println!("{:?}", feature);
    }
    dbg!(cur_loss);
}
