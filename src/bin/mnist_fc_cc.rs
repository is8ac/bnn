extern crate bitnn;
extern crate num_cpus;
extern crate rayon;

use bitnn::bits::{b32, BitArray, BitMul, Classify, Distance};
use bitnn::count::CountBits;
use bitnn::datasets::mnist;
use bitnn::layer;
use bitnn::shape::Element;
use bitnn::weight::{GenParamClasses, GenParamSet};
use rayon::prelude::*;
use std::boxed::Box;
use std::iter::Iterator;
use std::path::Path;

const L1_SIZE: usize = 2;

const N_EXAMPLES: usize = 60_0;
const N_CLASSES: usize = 10;
type InputWordShape = [(); 25];
type InputWordType = b32;
type InputType = <InputWordType as Element<InputWordShape>>::Array;

fn main() {
    println!(
        "MatrixCountersType: {}",
        std::any::type_name::<u8>()
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

    let (layer_weights, examples) = <[[([(b32, b32); 25], u32); 32]; L1_SIZE] as layer::FC<
        [b32; 25],
        [b32; L1_SIZE],
        [(); 10],
    >>::apply("foobar", &examples);

    let (layer_weights, examples) = <[[([(b32, b32); L1_SIZE], u32); 32]; L1_SIZE] as layer::FC<
        [b32; L1_SIZE],
        [b32; L1_SIZE],
        [(); 10],
    >>::apply("foobar2", &examples);

    let (value_counters, matrix_counters, n_examples) = <[b32; L1_SIZE]>::count_bits(&examples);
    let weights = <[b32; L1_SIZE] as GenParamClasses<[(); N_CLASSES]>>::gen_parm_classes(
        n_examples,
        &value_counters,
        &matrix_counters,
    );
    let n_correct: u64 = examples
        .par_iter()
        .map(|(image, class)| (weights.max_class(image) == *class) as u64)
        .sum();
    println!("acc: {}%", (n_correct as f64 / N_EXAMPLES as f64) * 100f64);
}
