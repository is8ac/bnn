#![feature(const_generics)]
extern crate bitnn;
extern crate num_cpus;
extern crate rayon;

use bitnn::bits::{b32, BitArray, IncrementFracCounters, IncrementHammingDistanceMatrix};
use bitnn::count::Counters;
use bitnn::datasets::mnist;
use bitnn::mask::{GenMask, Sum};
use bitnn::shape::{Element, ZipMap};
use rayon::prelude::*;
use std::boxed::Box;
use std::collections::HashSet;
use std::iter::Iterator;
use std::path::Path;

const N_EXAMPLES: usize = 60_00;
const N_CLASSES: usize = 10;
type InputType = [b32; 25];
type InputCountersShape = <InputType as BitArray>::BitShape;
type FracCounters = (usize, <u32 as Element<InputCountersShape>>::Array);
type ValueCountersType = [FracCounters; N_CLASSES];
type MatrixCountersType =
    <<u32 as Element<InputCountersShape>>::Array as Element<InputCountersShape>>::Array;

fn gen_partitions(depth: usize) -> Vec<HashSet<usize>> {
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

fn main() {
    println!(
        "MatrixCountersType: {}",
        std::any::type_name::<MatrixCountersType>()
    );
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(28))
        //.num_threads(2)
        .build_global()
        .unwrap();

    let images = mnist::load_images_bitpacked_u32(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"),
        N_EXAMPLES,
    );
    let classes = mnist::load_labels(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"),
        N_EXAMPLES,
    );
    let examples: Vec<(InputType, usize)> = images
        .iter()
        .cloned()
        .zip(classes.iter().map(|x| *x as usize))
        .collect();

    let (value_counters, matrix_counters): (Box<ValueCountersType>, Box<MatrixCountersType>) =
        examples
            .par_chunks(examples.len() / num_cpus::get_physical())
            .map(|chunk| {
                chunk.iter().fold(
                    (
                        Box::<ValueCountersType>::default(),
                        Box::<MatrixCountersType>::default(),
                    ),
                    |mut acc, (image, class)| {
                        image.increment_frac_counters(&mut acc.0[*class]);
                        image.increment_hamming_distance_matrix(&mut *acc.1, image);
                        acc
                    },
                )
            })
            .reduce(
                || {
                    dbg!();
                    (
                        Box::<ValueCountersType>::default(),
                        Box::<MatrixCountersType>::default(),
                    )
                },
                |mut a, b| {
                    (a.0).elementwise_add(&b.0);
                    (a.1).elementwise_add(&b.1);
                    a
                },
            );

    //let partitions = gen_partitions(10);
    let partitions: Vec<HashSet<usize>> = (0..N_CLASSES)
        .map(|c| {
            let mut set = HashSet::new();
            set.insert(c);
            set
        })
        .collect();
    //dbg!(&partitions);
    dbg!(partitions.len());
    let weights: Vec<_> = partitions
        .par_iter()
        .map(|partition| {
            let mut split_counters = Box::<[(usize, [[u32; 32]; 25]); 2]>::default();
            for (class, class_counter) in value_counters.iter().enumerate() {
                split_counters[partition.contains(&class) as usize].elementwise_add(class_counter);
            }
            let an = (split_counters[0].0 + 1) as f64;
            let bn = (split_counters[1].0 + 1) as f64;
            let sign_bools = <<InputType as BitArray>::BitShape as ZipMap<u32, u32, bool>>::zip_map(
                &split_counters[0].1,
                &split_counters[1].1,
                |&a, &b| {
                    //((a + 1) as f64 / (a + b + 2) as f64) > 0.5
                    ((a + 1) as f64 / an) > ((b + 1) as f64 / bn)
                },
            );
            let sign_bits = InputType::bitpack(&sign_bools);
            let mask_bits =
                <InputType as GenMask>::gen_mask(&matrix_counters, examples.len(), &split_counters);
            println!("{:?}", partition);
            mnist::display_mnist_b32(&sign_bits);
            mnist::display_mnist_b32(&mask_bits);
            <[(); 25] as ZipMap<b32, b32, (b32, b32)>>::zip_map(
                &sign_bits,
                &mask_bits,
                |&sign_word, &mask_word| (sign_word, mask_word),
            )
        })
        .collect();
    dbg!(weights.len());
    //dbg!(weights);
    let n_correct: u64 = examples
        .par_iter()
        .map(|(image, class)| {
            let activations: Vec<_> = weights
                .iter()
                .map(|class_weights| {
                    <[(); 25] as ZipMap<b32, (b32, b32), u32>>::zip_map(
                        &image,
                        &class_weights,
                        |&input_word, &(sign_word, mask_word)| {
                            ((input_word ^ sign_word) & mask_word).count_ones()
                        },
                    )
                    .sum()
                })
                .collect();
            dbg!(class);
            dbg!(&activations);
            let max_act_index = activations
                .iter()
                .enumerate()
                .max_by_key(|(_, x)| *x)
                .unwrap()
                .0;
            (max_act_index == *class) as u64
        })
        .sum();
    println!("acc: {}%", (n_correct as f64 / N_EXAMPLES as f64) * 100f64);
}
