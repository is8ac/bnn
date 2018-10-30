extern crate bitnn;
extern crate rayon;
extern crate time;
use rayon::prelude::*;

use bitnn::datasets::cifar;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, pack_3x3, pixelmap, unary, Layer2d, Patch};
use time::PreciseTime;

fn eval_acc<T: Patch>(readout_features: &Vec<(T, T)>, test_inputs: &Vec<Vec<T>>, biases: &Vec<i32>) -> f64 {
    let num_examples: usize = test_inputs.iter().map(|x| x.len()).sum();
    let test_correct: u64 = test_inputs
        .iter()
        .enumerate()
        .map(|x| {
            x.1.iter()
                .map(move |input| {
                    x.0 == readout_features
                        .iter()
                        .zip(biases.iter())
                        .map(|((base_point, mask), bias)| input.masked_hamming_distance(&base_point, &mask) as i32 - bias)
                        .enumerate()
                        .max_by_key(|(_, dist)| *dist)
                        .unwrap()
                        .0
                }).map(|x| x as u64)
        }).flatten()
        .sum();
    test_correct as f64 / num_examples as f64
}

fn bias<T: Patch>(readout_features: &Vec<(T, T)>, inputs: &Vec<Vec<T>>) -> Vec<i32> {
    let example_len: usize = inputs.iter().map(|x| x.len()).sum();
    let avg_activations: Vec<i32> = inputs
        .iter()
        .flatten()
        .fold(vec![0u32; 10], |acc, input| {
            readout_features
                .iter()
                .zip(acc.iter())
                .map(|((base_point, mask), class_sum)| class_sum + input.masked_hamming_distance(&base_point, &mask))
                .collect()
        }).iter()
        .map(|&x| (x as f64 / example_len as f64) as i32)
        .collect();

    avg_activations
}

fn apply_features_fc<T: Patch + Sync + Send, O: Patch + Sync + Send>(inputs: &Vec<Vec<T>>, features_vec: &Vec<(T, T)>, thresholds: &Vec<Vec<u32>>) -> Vec<Vec<O>> {
    inputs
        .par_iter()
        .map(|x| x.iter().map(|input| featuregen::apply_unary(input, &features_vec, thresholds)).collect())
        .collect()
}

fn nzip<X: Clone, Y: Clone>(a_vec: &Vec<Vec<X>>, b_vec: &Vec<Vec<Y>>) -> Vec<Vec<(X, Y)>> {
    a_vec.iter().zip(b_vec.iter()).map(|(a, b)| a.iter().cloned().zip(b.iter().cloned()).collect()).collect()
}

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let size: usize = 1000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images = Vec::with_capacity(size * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, size);
        images.append(&mut batch);
    }
    images
}

fn main() {
    let examples = load_data();
    let u14_packed_images: Vec<(usize, [[u16; 32]; 32])> = examples.par_iter().map(|(label, image)| (*label, pixelmap::pm_32(&image, &unary::rgb_to_u14))).collect();
    let l0_images = featuregen::split_by_label(&u14_packed_images, 10);
    let l0_patches: Vec<Vec<u128>> = l0_images
        .par_iter()
        .map(|imgs| imgs.iter().map(|image| image.to_3x3_patches()).flatten().map(|pixels| pack_3x3::p14(pixels)).collect())
        .collect();

    let l0_readout_features = featuregen::gen_readout_features(&l0_patches, 0.0);
    let biases = bias(&l0_readout_features, &l0_patches);
    let acc = eval_acc(&l0_readout_features, &l0_patches, &biases);
    println!("acc: {:?}%", acc * 100.0);

    let (l1_features_vec, l1_thresholds) = featuregen::gen_hidden_features(&l0_patches, 3, 4, 20);
    let l1_train_inputs = apply_features_fc::<_, u128>(&l0_patches, &l1_features_vec, &l1_thresholds);
    let l1_readout_features = featuregen::gen_readout_features(&l1_train_inputs, 0.0);
    let biases = bias(&l1_readout_features, &l1_train_inputs);
    let acc = eval_acc(&l1_readout_features, &l1_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);

    let l1_images: Vec<Vec<_>> = l0_images
        .par_iter()
        .map(|class| {
            class
                .iter()
                .map(|image| {
                    <[[u128; 32]; 32]>::from_pixels_1_padding(
                        &image
                            .to_3x3_patches()
                            .iter()
                            .map(|&pixels| featuregen::apply_unary(&pack_3x3::p14(pixels), &l1_features_vec, &l1_thresholds))
                            .collect(),
                    )
                }).collect()
        }).collect();
    let l1_pooled_images: Vec<Vec<[[_; 16]; 16]>> = l1_images.par_iter().map(|class| class.iter().map(|image| pixelmap::pool_or_32(&image)).collect()).collect();

    let l1_patches: Vec<Vec<_>> = l1_pooled_images
        .par_iter()
        .map(|imgs| imgs.iter().map(|image| image.to_3x3_patches()).flatten().collect())
        .collect();

    let l1_readout_features = featuregen::gen_readout_features(&l1_patches, 0.0);
    let biases = bias(&l1_readout_features, &l1_patches);
    let acc = eval_acc(&l1_readout_features, &l1_patches, &biases);
    println!("l1 acc: {:?}%", acc * 100.0);

    let (l2_features_vec, l2_thresholds) = featuregen::gen_hidden_features(&l1_patches, 4, 6, 7);
    let l2_train_inputs = apply_features_fc::<_, [u128; 2]>(&l1_patches, &l2_features_vec, &l2_thresholds);
    let l2_readout_features = featuregen::gen_readout_features(&l2_train_inputs, 0.0);
    let biases = bias(&l2_readout_features, &l2_train_inputs);
    let acc = eval_acc(&l2_readout_features, &l2_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);

    let l2_images: Vec<Vec<_>> = l1_pooled_images
        .par_iter()
        .map(|class| {
            class
                .iter()
                .map(|image| {
                    <[[[u128; 2]; 16]; 16]>::from_pixels_1_padding(
                        &image
                            .to_3x3_patches()
                            .iter()
                            .map(|&pixels| featuregen::apply_unary(&pixels, &l2_features_vec, &l2_thresholds))
                            .collect(),
                    )
                }).collect()
        }).collect();
    let l2_pooled_images: Vec<Vec<[[_; 8]; 8]>> = l2_images.par_iter().map(|class| class.iter().map(|image| pixelmap::pool_or_16(image)).collect()).collect();

    let l2_patches: Vec<Vec<_>> = l2_pooled_images
        .par_iter()
        .map(|imgs| imgs.iter().map(|image| image.to_3x3_patches()).flatten().collect())
        .collect();

    let l2_readout_features = featuregen::gen_readout_features(&l2_patches, 0.0);
    let biases = bias(&l2_readout_features, &l2_patches);
    let acc = eval_acc(&l2_readout_features, &l2_patches, &biases);
    println!("l2 acc: {:?}%", acc * 100.0);

    let (l3_features_vec, l3_thresholds) = featuregen::gen_hidden_features(&l2_patches, 5, 5, 3);
    let l3_train_inputs = apply_features_fc::<_, [u128; 3]>(&l2_patches, &l3_features_vec, &l3_thresholds);
    let l3_readout_features = featuregen::gen_readout_features(&l3_train_inputs, 0.0);
    let biases = bias(&l3_readout_features, &l3_train_inputs);
    let acc = eval_acc(&l3_readout_features, &l3_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);

    let l3_images: Vec<Vec<_>> = l2_pooled_images
        .par_iter()
        .map(|class| {
            class
                .iter()
                .map(|image| {
                    <[[[u128; 3]; 8]; 8]>::from_pixels_1_padding(
                        &image
                            .to_3x3_patches()
                            .iter()
                            .map(|&pixels| featuregen::apply_unary(&pixels, &l3_features_vec, &l3_thresholds))
                            .collect(),
                    )
                }).collect()
        }).collect();
    let l3_pooled_images: Vec<Vec<[[_; 4]; 4]>> = l3_images.par_iter().map(|class| class.iter().map(|image| pixelmap::pool_or_8(image)).collect()).collect();

    let l3_patches: Vec<Vec<_>> = l3_pooled_images
        .par_iter()
        .map(|imgs| imgs.iter().map(|image| image.to_3x3_patches()).flatten().collect())
        .collect();

    let l3_readout_features = featuregen::gen_readout_features(&l3_patches, 0.0);
    let biases = bias(&l3_readout_features, &l3_patches);
    let acc = eval_acc(&l3_readout_features, &l3_patches, &biases);
    println!("l3 acc: {:?}%", acc * 100.0);

    let (l4_features_vec, l4_thresholds) = featuregen::gen_hidden_features(&l3_patches, 5, 6, 2);
    let l4_train_inputs = apply_features_fc::<_, [u128; 3]>(&l3_patches, &l4_features_vec, &l4_thresholds);
    let l4_readout_features = featuregen::gen_readout_features(&l4_train_inputs, 0.0);
    let biases = bias(&l4_readout_features, &l4_train_inputs);
    let acc = eval_acc(&l4_readout_features, &l4_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);

    let test_set = cifar::load_images_10(&String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/test_batch.bin"), 10000);
    let u14_packed_test_images: Vec<(usize, [[u16; 32]; 32])> = test_set.par_iter().map(|(label, image)| (*label, pixelmap::pm_32(&image, &unary::rgb_to_u14))).collect();

    let start = PreciseTime::now();

    let l1_test_images: Vec<(usize, _)> = u14_packed_test_images
        .par_iter()
        .map(|(label, image)| {
            (
                *label,
                <[[u128; 32]; 32]>::from_pixels_1_padding(
                    &image
                        .to_3x3_patches()
                        .iter()
                        .map(|&pixels| featuregen::apply_unary(&pack_3x3::p14(pixels), &l1_features_vec, &l1_thresholds))
                        .collect(),
                ),
            )
        }).collect();
    let l1_pooled_images: Vec<(usize, [[_; 16]; 16])> = l1_test_images.par_iter().map(|(label, image)| (*label, pixelmap::pool_or_32(&image))).collect();
    let l2_test_images: Vec<(usize, _)> = l1_pooled_images
        .par_iter()
        .map(|(label, image)| {
            (
                *label,
                <[[[u128; 2]; 16]; 16]>::from_pixels_1_padding(
                    &image
                        .to_3x3_patches()
                        .iter()
                        .map(|&pixels| featuregen::apply_unary(&pixels, &l2_features_vec, &l2_thresholds))
                        .collect(),
                ),
            )
        }).collect();
    let l2_pooled_images: Vec<(usize, [[_; 8]; 8])> = l2_test_images.par_iter().map(|(label, image)| (*label, pixelmap::pool_or_16(&image))).collect();
    let l3_test_images: Vec<(usize, _)> = l2_pooled_images
        .par_iter()
        .map(|(label, image)| {
            (
                *label,
                <[[[u128; 3]; 8]; 8]>::from_pixels_1_padding(
                    &image
                        .to_3x3_patches()
                        .iter()
                        .map(|&pixels| featuregen::apply_unary(&pixels, &l3_features_vec, &l3_thresholds))
                        .collect(),
                ),
            )
        }).collect();
    let l3_pooled_images: Vec<(usize, [[_; 4]; 4])> = l3_test_images.par_iter().map(|(label, image)| (*label, pixelmap::pool_or_8(&image))).collect();
    //let l4_test_inputs = apply_features_fc::<_, [u128; 3]>(&l3_patches, &l4_features_vec, &l4_thresholds);
    //let l4_train_inputs =      l3_.map(|x| x.iter().map(|input| featuregen::apply_unary(input, &features_vec, thresholds)).collect())

    println!("inf time: {}", start.to(PreciseTime::now()));
}
// 29.28%
