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

//fn eval_acc<T>() -> f64{
//    let mut readout_features: Vec<([u64; 13], [u64; 13])> = vec![];
//    for class in 0..10 {
//        let (base_point, mask) = featuregen::gen_basepoint(&shards, 0.0, class);
//        readout_features.push((base_point, mask));
//    }
//    let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
//    let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
//    let test_correct: u64 = test_labels
//        .iter()
//        .zip(test_images.iter())
//        .map(|(&label, &image)| {
//            let max: usize = readout_features
//                .iter()
//                .map(|(class_feature, mask)| class_feature.masked_hamming_distance(&image, &mask))
//                .enumerate()
//                .max_by_key(|(_, dist)| *dist)
//                .unwrap()
//                .0;
//            (max == label) as u64
//        })
//        .sum();
//    test_correct as f64 / test_images.len()
//}

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<_> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let mut shards = vec![split_by_label(&examples, 10)];
    if false {
        let mut readout_features: Vec<([u64; 13], [u64; 13])> = vec![];
        for class in 0..10 {
            let (base_point, mask) = featuregen::gen_basepoint(&shards, 0.0, class);
            readout_features.push((base_point, mask));
        }
        let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
        let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
        let test_correct: u64 = test_labels
            .iter()
            .zip(test_images.iter())
            .map(|(&label, &image)| {
                let max: usize = readout_features
                    .iter()
                    .map(|(class_feature, mask)| class_feature.masked_hamming_distance(&image, &mask))
                    .enumerate()
                    .max_by_key(|(_, dist)| *dist)
                    .unwrap()
                    .0;
                (max == label) as u64
            })
            .sum();
        println!("l0 acc: {:?}%", test_correct as f64 / 10000.0 * 100.0);
    }
    let mut features_vec = vec![];
    let nil_mask = [!0u64; 13];
    for i in 0..5 {
        for class in 0..10 {
            let (base_point, mask) = featuregen::gen_basepoint(&shards, 0.0000001, class);
            //let mask = nil_mask;
            let threshold = featuregen::gen_threshold_masked(&images, &base_point, &mask);
            features_vec.push((base_point, mask, threshold));
            shards = featuregen::split_labels_set_by_distance(&shards, &base_point, &mask, threshold, 2);
            println!("{:064b} {:064b} {:} \t {:?} \t {:?}", base_point[3], mask[3], threshold, shards.len(), class);
        }
    }
    let mut features = [([0u64; 13], [0u64; 13], 0u32); 64];
    for (i, feature) in features_vec.iter().enumerate() {
        features[i] = *feature;
        //println!("{:064b} {:064b} {:?}", features[i].0[3], features[i].1[3], features[i].2);
    }
    let l2_examples: Vec<(usize, u64)> = examples.iter().map(|(label, image)| (*label, bitvecmul::mbvm_u64(&features, &image))).collect();
    let mut shards = vec![split_by_label(&l2_examples, 10)];

    {
        let mut readout_features: Vec<(u64, u64)> = vec![];
        for class in 0..10 {
            let (base_point, mask) = featuregen::gen_basepoint(&shards, 0.00004, class);
            println!("{:064b}", base_point);
            readout_features.push((base_point, mask));
        }
        let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
        let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
        let test_correct: u64 = test_labels
            .iter()
            .zip(test_images.iter())
            .map(|(&label, &image)| {
                let l2_state = bitvecmul::mbvm_u64(&features, &image);
                let max: usize = readout_features
                    .iter()
                    .map(|(class_feature, mask)| class_feature.masked_hamming_distance(&l2_state, &mask))
                    .enumerate()
                    .max_by_key(|(_, dist)| *dist)
                    .unwrap()
                    .0;
                (max == label) as u64
            })
            .sum();
        println!("{:?}%", test_correct as f64 / test_images.len() as f64 * 100.0);
    }
}
