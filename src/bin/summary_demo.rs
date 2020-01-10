// if main overflows stack:
// ulimit -S -s 4194304

extern crate bitnn;
extern crate rayon;

use rayon::prelude::*;
use std::time::Instant;

use bitnn::bits::{b32, BitArray, BitMul, BitWord, Classify, IndexedFlipBit};
use bitnn::block::BlockCode;
use bitnn::count::CounterArray;
use bitnn::datasets::cifar;
use bitnn::image2d::StaticImage;
use bitnn::layer::{Apply, CountBits, Layer};
use bitnn::shape::Element;
use bitnn::unary::{Identity, Normalize, Unary};
use bitnn::weight::Objective;
use bitnn::weight::{DecendOneHiddenLayer, GenWeights};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::path::Path;

const N_EXAMPLES: usize = 50_000;
const N_CLASSES: usize = 10;
const K: usize = 27;
const TN: usize = 1;
type InputType = [[b32; 3]; 3];
type TargetType = [b32; TN];
type PreprocessorType = Unary<[[b32; 3]; 3]>;
//type PreprocessorType = Normalize<Unary<b32>>;

fn main() {
    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");

    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    //let examples = <()>::edges(&int_examples_32);

    let (l2_examples, layer, aux_weights) = <[[[[b32; 3]; 3]; 32]; 1] as Layer<
        StaticImage<[[[u8; 3]; 32]; 32]>,
        [[(); 3]; 3],
        [[b32; 3]; 3],
        Unary<[[b32; 3]; 3]>,
        DecendOneHiddenLayer<K, 3, 500, 9, { N_CLASSES }>,
        StaticImage<[[[b32; 1]; 32]; 32]>,
        [(); N_CLASSES],
    >>::gen(&int_examples_32);

    let acc_start = Instant::now();
    let n_correct: u64 = l2_examples
        .par_iter()
        .map(|(image, class)| {
            let max_class = <[TargetType; N_CLASSES] as Classify<
                StaticImage<[[TargetType; 32]; 32]>,
                (),
                [(); N_CLASSES],
            >>::max_class(&aux_weights, &image);
            (max_class == *class) as u64
        })
        .sum();
    dbg!(n_correct as f64 / int_examples_32.len() as f64);
    dbg!(acc_start.elapsed());
    //0.10128
    //50: 0.1522
    //500: 0.16752
    //1000: 0.15516
    //K24: 0.10548
    // 0.15516
    //TN2:0.1383
    //TN1:0.139

    // pow1: 0.12286
    // pow2: 0.0987
    // 100: 27000
    // 50: 0.1548 i: 103282
    // 0.1339
    // 0.1317
    // 1332
    //0.13826
    // s0: 0.13826
    // s1: 0.09848
    // s2: 0.1196
    // s3: 0.11136
}
