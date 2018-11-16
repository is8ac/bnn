extern crate bitnn;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::featuregen;
use bitnn::layers::{Layer2D, ObjectiveHeadFC, Patch, Pool2x2, WeightsMatrix, SimplifyBits};
use rayon::prelude::*;
use time::PreciseTime;

const TRAIN_SIZE: usize = 60_000;

fn main() {
    let images = mnist::load_images(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [[u128; 7]; 7])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image.simplify())).collect();

    let mut hidden_layer = <[([[u128; 7]; 7], [u32; 4]); 32]>::new_from_split(&examples);
    let l1_examples: Vec<(usize, _)> = examples.iter().map(|(class, input)| (*class, hidden_layer.vecmul(&input))).collect();
    let mut readout_layer = <[_; 10]>::new_from_split(&l1_examples);
    for _ in 0..5 {
        readout_layer.update(&l1_examples);
        println!("readout: {:?}", readout_layer.acc(&l1_examples));
    }
    let total_start = PreciseTime::now();
    for i in 0..20 {
        println!("starting iter: {:?}\n", i);
        hidden_layer.optimize(&mut readout_layer, &examples);
    }
    println!("total: {}", total_start.to(PreciseTime::now()));
}


// PT226 68%
// PT1145 78%
// PT304 76%
// 85%
// b32 u3: 70%
// b32 u4: 75%
// b32 u5: 72%
// b32 u6: 71%

// 60_000: b32 u4: 84.15%
// 60_000: b32 u4: 85.22% PT654
