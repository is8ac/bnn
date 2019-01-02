extern crate bitnn;
extern crate rand;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::featuregen;
use bitnn::layers::{Layer, NewFromRng, NewFromSplit, ObjectiveHead, Optimize};
use bitnn::{Patch, BitLen};
use rand::prelude::*;

use rayon::prelude::*;
use time::PreciseTime;

const TRAIN_SIZE: usize = 60_000;


const MOD: usize = 20;
const ITERS: usize = 3;
type HiddenLayer = u32;
const HIDDEN_SIZE: usize = HiddenLayer::BIT_LEN;

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

    //let mut rng = thread_rng();
    let model = Layer::<[u64; 13], [[u64; 13]; HIDDEN_SIZE], [i16; HIDDEN_SIZE], Layer<[i16; HIDDEN_SIZE], [i16; HIDDEN_SIZE], HiddenLayer, [HiddenLayer; 10]>>::new_from_split(&examples);

    let mut weights = model.data;
    let mut head = model.head;
    println!("{:?}", head.data);

    let start = PreciseTime::now();
    let mut acc = 0f64;
    for i in 0..ITERS {
        acc = weights.optimize(&mut head, &examples, MOD);
        println!("{} {:?}", i, acc);
    }
    println!("{:?}", head.data);
    println!("hidden: {}, mod: {}, iters {}, {}, acc: {}%", HIDDEN_SIZE, MOD, ITERS, start.to(PreciseTime::now()), acc * 100f64);
}

// u32 split:
// hidden: 32, mod: 7,  iters 4, PT243.810026379S, acc: 83.695%
// hidden: 32, mod: 14, iters 4, PT193.962831348S, acc: 84.016%
// hidden: 32, mod: 20, iters 3, PT142.640710833S, acc: 83.179%
// hidden: 32, mod: 40, iters 3, PT134.119394082S, acc: 82.331%
// hidden: 16, mod: 20, iters 3, PT61.776182200S, acc: 80.021%
// hidden: 16, mod: 10, iters 4, PT94.613469850S, acc: 80.185%
// hidden: 16, mod: 7,  iters 5, PT111.465620002S, acc: 74.621%
// hidden: 16, mod: 11, iters 4, PT88.429391688S, acc: 80.151%
// hidden: 16, mod: 7, iters 4, PT91.582400810S, acc: 74.414%
// hidden: 16, mod: 8, iters 4, PT95.706479591S, acc: 80.456%
// hidden: 16, mod: 5, iters 5, PT131.842399159S, acc: 79.75333333333333%
// hidden: 16, mod: 3, iters 6, PT195.520970420S, acc: 80.71333333333334%
// mut  biases: hidden: 32, mod: 20, iters 3, PT155.441772830S, acc: 83.17999999999999%
// stat biases: hidden: 32, mod: 20, iters 3, PT137.358122196S, acc: 83.24333333333334%
// stat /2 biases hidden: 32, mod: 20, iters 3, PT134.141783354S, acc: 84.47833333333334%

// 32 mod20: 84%
// rng 10 32 mod20: 83.113% PT393.209929585S
// split 10 32 mod7: 0.848% PT453.313864572S
