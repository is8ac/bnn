extern crate bitnn;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::layers::{bitvecmul, pack_3x3, pixelmap, unary, Layer2d, Patch};
use bitnn::{featuregen, layers};
use rayon::prelude::*;
use time::PreciseTime;

fn split_by_label<T: Copy>(examples: &Vec<(usize, T)>, len: usize) -> Vec<Vec<T>> {
    let mut by_label: Vec<Vec<T>> = (0..len).map(|_| Vec::new()).collect();
    for (label, example) in examples {
        by_label[*label].push(*example);
    }
    let _: Vec<_> = by_label.iter_mut().map(|x| x.shrink_to_fit()).collect();
    by_label
}

const TRAIN_SIZE: usize = 60_000;

fn pack_u32_u128(input: [u32; 4]) -> u128 {
    let mut output = 0u128;
    for i in 0..4 {
        output = output | ((input[i] as u128) << (i * 32));
    }
    output
}

const L1_FEATURE_ITERS: usize = 4;
const l1_UNARY_SIZE: usize = 6;

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<_> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let mut shards = vec![split_by_label(&examples, 10)];
    let mut features_vec: Vec<([u64; 13], [u64; 13])> = vec![];
    let nil_mask = [!0u64; 13];
    for i in 0..L1_FEATURE_ITERS {
        for class in 0..10 {
            let (base_point, mask) = featuregen::gen_basepoint(&shards, 0.000000, class);
            //let mask = nil_mask;
            features_vec.push((base_point, mask));
            let split_threshold = featuregen::gen_threshold_masked(&images, &base_point, &mask);
            shards = featuregen::split_labels_set_by_distance(&shards, &base_point, &mask, split_threshold, 4);
            println!("{:064b} {:064b} {:?} \t {:?}", base_point[3], mask[3], shards.len(), class);
        }
    }
    let l1_thresholds: Vec<[u32; l1_UNARY_SIZE]> = features_vec.iter().map(|(base, mask)| featuregen::tm_6(&images, &base, &mask)).collect();
    let l2_examples: Vec<(usize, [u128; 2])> = examples
        .iter()
        .map(|(label, image)| {
            let distances = bitvecmul::vmbvm(&features_vec, &image);
            let mut bools = vec![false; 256];
            for c in 0..(L1_FEATURE_ITERS * 10) {
                for i in 0..l1_UNARY_SIZE {
                    bools[(c * l1_UNARY_SIZE) + i] = distances[c] > l1_thresholds[c][i];
                }
            }
            (*label, <[u128; 2]>::bitpack(&bools.as_slice()))
        }).collect();
    let mut shards = vec![split_by_label(&l2_examples, 10)];
    {
        let mut readout_features: Vec<([u128; 2], [u128; 2])> = vec![];
        for class in 0..10 {
            let (base_point, mask) = featuregen::gen_basepoint(&shards, 0.0000, class);
            //println!("{:0128b}", base_point);
            readout_features.push((base_point, mask));
        }
        let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
        let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
        let test_correct: u64 = test_labels
            .iter()
            .zip(test_images.iter())
            .map(|(&label, &image)| {
                let l2_state = {
                    let distances = bitvecmul::vmbvm(&features_vec, &image);
                    let mut bools = vec![false; 256];
                    for c in 0..(L1_FEATURE_ITERS * 10) {
                        for i in 0..l1_UNARY_SIZE {
                            bools[(c * l1_UNARY_SIZE) + i] = distances[c] > l1_thresholds[c][i];
                        }
                    }
                    <[u128; 2]>::bitpack(&bools.as_slice())
                };
                let max: usize = readout_features
                    .iter()
                    .map(|(class_feature, mask)| class_feature.masked_hamming_distance(&l2_state, &mask))
                    .enumerate()
                    .max_by_key(|(_, dist)| *dist)
                    .unwrap()
                    .0;
                (max == label) as u64
            }).sum();
        println!("{:?}%", test_correct as f64 / test_images.len() as f64 * 100.0);
    }
}
