#![feature(const_generics)]
extern crate bitnn;
extern crate num_cpus;
extern crate rayon;

use bitnn::bits::{b32, b8, BitArray, Distance};
use bitnn::count::{CountBits, Counters};
use bitnn::datasets::mnist;
use bitnn::image2d::PixelMap2D;
use bitnn::shape::{Element, ZipMap};
use bitnn::weight::{GenParamClasses, GenParamSet, Sum};
use rayon::prelude::*;
use std::boxed::Box;
use std::collections::HashSet;
use std::iter::Iterator;
use std::path::Path;

const N_EXAMPLES: usize = 60_000;
const N_CLASSES: usize = 10;
type InputWordShape = [(); 25];
//type InputWordShape = [[(); 28]; 28];
type InputWordType = b32;
//type InputWordType = b8;
type InputType = <InputWordType as Element<InputWordShape>>::Array;
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
    //let images = mnist::load_images_u8(
    //    Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"),
    //    N_EXAMPLES,
    //);
    let classes = mnist::load_labels(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"),
        N_EXAMPLES,
    );
    let examples: Vec<(InputType, usize)> = images
        .iter()
        //.map(|image|image.map_2d(|p|b8((!0) << (p / 8u8))))
        .cloned()
        .zip(classes.iter().map(|x| *x as usize))
        .collect();

    let (value_counters, matrix_counters): (Box<ValueCountersType>, Box<MatrixCountersType>) =
        InputType::count_bits(&examples);

    //let layer_weights = <InputType as GenParamSet<[b32; 1], [(); N_CLASSES]>>::gen_parm_set(
    //    examples.len(),
    //    &value_counters,
    //    &matrix_counters,
    //);
    //let foo: u8 = layer_weights;
    let weights = <InputType as GenParamClasses<[(); N_CLASSES]>>::gen_parm_classes(
        examples.len(),
        &value_counters,
        &matrix_counters,
    );

    let n_correct: u64 = examples
        .par_iter()
        .map(|(image, class)| {
            let activations: Vec<_> = weights
                .iter()
                .map(|(class_weights, threshold)| class_weights.distance(image) as i32 - *threshold as i32)
                .collect();
            //dbg!(class);
            //dbg!(&activations);
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
