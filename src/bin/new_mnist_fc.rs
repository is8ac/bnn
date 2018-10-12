extern crate bitnn;

use bitnn::datasets::mnist;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, Patch};

fn split_by_label<T: Copy>(examples: &Vec<(usize, T)>, len: usize) -> Vec<Vec<T>> {
    let mut by_label: Vec<Vec<T>> = (0..len).map(|_| Vec::new()).collect();
    for (label, example) in examples {
        by_label[*label].push(*example);
    }
    let _: Vec<_> = by_label.iter_mut().map(|x| x.shrink_to_fit()).collect();
    by_label
}

fn apply_unary<T: Patch, O: Patch>(input: &T, features_vec: &Vec<(T, T)>, thresholds: &Vec<Vec<u32>>) -> O {
    let distances = bitvecmul::vmbvm(&features_vec, &input);
    let mut bools = vec![false; O::bit_len()];
    for c in 0..distances.len() {
        for i in 0..thresholds[0].len() {
            bools[(c * thresholds[0].len()) + i] = distances[c] > thresholds[c][i];
        }
    }
    O::bitpack(&bools.as_slice())
}

const TRAIN_SIZE: usize = 60_000;

const L1_FEATURE_ITERS: usize = 4;
const L1_UNARY_SIZE: usize = 6;

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<_> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let mut shards = vec![split_by_label(&examples, 10)];
    let mut features_vec: Vec<([u64; 13], [u64; 13])> = vec![];
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
    let l1_thresholds: Vec<Vec<u32>> = features_vec
        .iter()
        .map(|(base, mask)| featuregen::vec_threshold(&images, &base, &mask, L1_UNARY_SIZE))
        .collect();
    let l2_examples: Vec<(usize, [u128; 2])> = examples
        .iter()
        .map(|(label, image)| {
            let distances = bitvecmul::vmbvm(&features_vec, &image);
            let mut bools = vec![false; 256];
            for c in 0..(L1_FEATURE_ITERS * 10) {
                for i in 0..L1_UNARY_SIZE {
                    bools[(c * L1_UNARY_SIZE) + i] = distances[c] > l1_thresholds[c][i];
                }
            }
            (*label, <[u128; 2]>::bitpack(&bools.as_slice()))
        }).collect();
    let l2_examples: Vec<(usize, [u128; 2])> = examples.iter().map(|(label, image)| (*label, apply_unary(image, &features_vec, &l1_thresholds))).collect();
    let shards = vec![split_by_label(&l2_examples, 10)];
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
            .map(|(&label, image)| {
                let l2_state = apply_unary(image, &features_vec, &l1_thresholds);
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
