extern crate bitnn;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::featuregen;
use bitnn::layers::{Apply, Layer, NewFromSplit, ObjectiveHead, Optimize, SimplifyBits, VecApply};
use bitnn::Patch;
use rayon::prelude::*;
use time::PreciseTime;

const TRAIN_SIZE: usize = 60_00;

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

    let mut accs = vec![];

    let mut layer1 = <[([u64; 13], [u32; 4]); 32]>::new_from_split(&examples);
    let l1_examples: Vec<(usize, u128)> = layer1.vec_apply(&examples);
    //let mut l1_head = <[u128; 10]>::new_from_split(&l1_examples);
    let mut l1_head = Layer::<u128, u64, [(u128, u32); 64], [u64; 10]>::new_from_split(&l1_examples);
    let acc = l1_head.acc(&l1_examples);
    //accs.push(acc);
    //println!("acc: {}%", acc * 100f64);
    let start = PreciseTime::now();
    for i in 0..5 {
        layer1.optimize(&mut l1_head, &examples, 100);
        let l1_examples: Vec<(usize, u128)> = layer1.vec_apply(&examples);
        let acc = l1_head.acc(&l1_examples);
        accs.push((acc, start.to(PreciseTime::now())));
        println!("acc: {}% {:?}", acc * 100f64, start.to(PreciseTime::now()));
    }

    println!("{}", start.to(PreciseTime::now()));
    println!("{:?}", accs);
}
