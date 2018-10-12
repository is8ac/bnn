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

fn gen_readout_features<T: Patch + Sync + Clone + Default>(by_class: &Vec<Vec<T>>, n: usize) -> Vec<(T, T)> {
    (0..n).map(|class| {
        let grads = featuregen::grads_one_shard(by_class, class);
        let (base_point, mask) = featuregen::grads_to_bits(&grads, 0.0);
        (base_point, mask)
    }).collect()
}

fn eval_acc<T: Patch>(readout_features: &Vec<(T, T)>, test_examples: &Vec<(usize, T)>) -> f64 {
    let test_correct: u64 = test_examples
        .iter()
        .map(|(label, input)| {
            let max: usize = readout_features
                .iter()
                .map(|(class_feature, mask)| class_feature.masked_hamming_distance(&input, &mask))
                .enumerate()
                .max_by_key(|(_, dist)| *dist)
                .unwrap()
                .0;
            (max == *label) as u64
        }).sum();
    test_correct as f64 / test_examples.len() as f64
}

const TRAIN_SIZE: usize = 60_000;

const L2_SIZE: usize = 2;
const L1_FEATURE_ITERS: usize = 4;
const L1_UNARY_SIZE: usize = 6;

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

    let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
    let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
    let test_examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

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
    let l1_thresholds: Vec<Vec<u32>> = features_vec.iter().map(|(base, mask)| featuregen::vec_threshold(&images, &base, &mask, L1_UNARY_SIZE)).collect();
    let l2_examples: Vec<(usize, [u128; L2_SIZE])> = examples
        .iter()
        .map(|(label, image)| {
            let distances = bitvecmul::vmbvm(&features_vec, &image);
            let mut bools = vec![false; L2_SIZE * 128];
            for c in 0..(L1_FEATURE_ITERS * 10) {
                for i in 0..L1_UNARY_SIZE {
                    bools[(c * L1_UNARY_SIZE) + i] = distances[c] > l1_thresholds[c][i];
                }
            }
            (*label, <[u128; L2_SIZE]>::bitpack(&bools.as_slice()))
        }).collect();
    let l2_examples: Vec<(usize, [u128; L2_SIZE])> = examples.iter().map(|(label, image)| (*label, apply_unary(image, &features_vec, &l1_thresholds))).collect();
    let test_l2_examples: Vec<(usize, [u128; L2_SIZE])> = test_examples.iter().map(|(label, image)| (*label, apply_unary(image, &features_vec, &l1_thresholds))).collect();
    let readout_features = gen_readout_features(&split_by_label(&l2_examples, 10), 10);
    let acc = eval_acc(&readout_features, &test_l2_examples);
    println!("acc: {:?}%", acc * 100.0);
}
