#[macro_use]
extern crate bitnn;
extern crate rayon;
extern crate time;
use rayon::prelude::*;

use bitnn::datasets::cifar;
use std::collections::HashMap;
use time::PreciseTime;

#[allow(unused_macros)]
macro_rules! gen_fragment_features {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(examples_sets: &Vec<Vec<&[u64; $in_size]>>, mask_thresh: f64, level: usize) -> Vec<([u64; $in_size], [u64; $in_size])> {
            let examples_lens: Vec<usize> = examples_sets.par_iter().map(|x| x.len()).collect();
            let (largest_label, _): (usize, &usize) = examples_lens.iter().enumerate().max_by_key(|(_, &count)| count).unwrap();
            //println!("{:?} {:?}", level, largest_label);
            let mut label_bit_sums: Vec<(usize, Vec<u32>)> = examples_sets.par_iter().map(|patches| sum_bits!(patches, $in_size)).collect();
            let (target_len, target_bit_sums) = label_bit_sums.remove(largest_label);

            let (global_bit_len, global_bit_sums): (usize, Vec<u32>) = label_bit_sums.par_iter().cloned().reduce(
                || (0, vec![0u32; $in_size * 64]),
                |a, b| (a.0 + b.0, (a.1).iter().zip((b.1).iter()).map(|(a, b)| a + b).collect()),
            );

            let global_avg_bits: Vec<f64> = global_bit_sums.iter().map(|&sum| sum as f64 / global_bit_len as f64).collect();
            let target_avg_bits: Vec<f64> = target_bit_sums.iter().map(|&sum| sum as f64 / target_len as f64).collect();

            let mut weights = ([0u64; $in_size], [0u64; $in_size]);
            for i in 0..$in_size * 64 {
                let word = i / 64;
                let bit = i % 64;
                let grad = global_avg_bits[i] - target_avg_bits[i];
                let sign = grad > 0f64;
                weights.1[word] = weights.1[word] | ((sign as u64) << bit);
                let magn = grad.abs() > mask_thresh;
                weights.0[word] = weights.0[word] | ((magn as u64) << bit);
            }
            if level <= 0 {
                return vec![weights];
            }

            let mut activations: Vec<u32> = examples_sets
                .par_iter()
                .flatten()
                .map(|input| fc_infer!($in_size, input, weights.0, weights.1))
                .collect();
            activations.sort();
            let threshold = activations[activations.len() / 2];
            let over: Vec<Vec<&[u64; $in_size]>> = examples_sets
                .par_iter()
                .map(|example_set| {
                    example_set
                        .par_iter()
                        .filter(|input| fc_infer!($in_size, input, weights.0, weights.1) >= threshold)
                        .map(|&x| x)
                        .collect()
                }).collect();
            //print!("a: ");
            let mut over_elems = $name(&over, mask_thresh, level - 1);

            let under: Vec<Vec<&[u64; $in_size]>> = examples_sets
                .par_iter()
                .map(|example_set| {
                    example_set
                        .par_iter()
                        .filter(|input| fc_infer!($in_size, input, weights.0, weights.1) < threshold)
                        .map(|&x| x)
                        .collect()
                }).collect();

            //print!("b: ");
            let mut under_elems = $name(&under, mask_thresh, level - 1);

            over_elems.append(&mut under_elems);
            over_elems
        }
    };
}

macro_rules! readout_filters {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(patch_sets: &Vec<Vec<[u64; $in_size]>>, mask_thresh: f64) -> [([u64; $in_size], [u64; $in_size]); $n_labels] {
            let label_bit_sums: Vec<(usize, Vec<u32>)> = patch_sets.par_iter().map(|patches| sum_bits!(patches, $in_size)).collect();
            let (global_bit_len, global_bit_sums): (usize, Vec<u32>) = label_bit_sums.par_iter().cloned().reduce(
                || (0, vec![0u32; $in_size * 64]),
                |a, b| (a.0 + b.0, (a.1).iter().zip((b.1).iter()).map(|(a, b)| a + b).collect()),
            );

            let global_avg_bits: Vec<f64> = global_bit_sums.iter().map(|&sum| sum as f64 / global_bit_len as f64).collect();
            let label_avg_bits: Vec<Vec<f64>> = label_bit_sums
                .par_iter()
                .map(|(len, sums)| sums.iter().map(|&sum| sum as f64 / *len as f64).collect())
                .collect();

            let mut weights = [([0u64; $in_size], [0u64; $in_size]); $n_labels];
            for o in 0..$n_labels {
                for i in 0..$in_size * 64 {
                    let word = i / 64;
                    let bit = i % 64;
                    let grad = global_avg_bits[i] - label_avg_bits[o][i];
                    let sign = grad > 0f64;
                    weights[o].1[word] = weights[o].1[word] | ((sign as u64) << bit);
                    let magn = grad.abs() > mask_thresh;
                    weights[o].0[word] = weights[o].0[word] | ((magn as u64) << bit);
                }
            }
            weights
        }
    };
}

macro_rules! fc_infer {
    ($in_size:expr, $input:expr, $mask:expr, $signs:expr) => {{
        let mut sum = 0;
        for i in 0..$in_size {
            sum += (($signs[i] ^ $input[i]) & $mask[i]).count_ones();
        }
        sum
    }};
}

macro_rules! fc_is_correct {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(input: &[u64; $in_size], label: usize, weights: &[([u64; $in_size], [u64; $in_size]); $n_labels]) -> bool {
            let target_sum = fc_infer!($in_size, input, weights[label].0, weights[label].1);
            for o in 0..$n_labels {
                if o != label {
                    let label_sum = fc_infer!($in_size, input, weights[o].0, weights[o].1);
                    if label_sum >= target_sum {
                        return false;
                    }
                }
            }
            true
        }
    };
}

macro_rules! extract_3x3_patch_sets {
    ($name:ident, $image_size:expr, $in_chans:expr, $n_labels:expr) => {
        fn $name(images: &Vec<(u8, [[[u64; $in_chans]; $image_size]; $image_size])>) -> Vec<Vec<[u64; 9 * $in_chans]>> {
            fixed_dim_extract_3x3_patchs!(extract_patches, 32, 32, 1);
            let lables: Vec<usize> = (0..$n_labels).collect();
            let patch_sets: Vec<Vec<[u64; 9 * $in_chans]>> = lables
                .par_iter()
                .map(|&l| {
                    images
                        .par_iter()
                        .filter(|(label, _)| *label as usize == l)
                        .fold(
                            || Vec::new(),
                            |mut patches, (_, input)| {
                                extract_patches(&input, &mut patches);
                                patches
                            },
                        ).reduce(
                            || Vec::new(),
                            |mut a, mut b| {
                                a.append(&mut b);
                                a
                            },
                        )
                }).collect();
            patch_sets
        }
    };
}

#[macro_export]
macro_rules! fixed_dim_apply_3x3_patchs {
    ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
        fn $name(input: &[[[u64; $in_chans]; $y_size]; $x_size], weights: &Vec<([u64; 9 * $in_chans], [u64; 9], u32)>) -> [[[u64; $out_chans]; $y_size]; $x_size] {
            let mut output = [[[0u64; $out_chans]; $y_size]; $x_size];
            for x in 0..$x_size - 2 {
                for y in 0..$y_size - 2 {
                    let patch = {
                        let mut patch = [0u64; 3 * 3 * $in_chans];
                        for px in 0..3 {
                            let px_offset = px * 3 * $in_chans;
                            for py in 0..3 {
                                let py_offset = px_offset + (py * $in_chans);
                                for i in 0..$in_chans {
                                    patch[py_offset + i] = input[x + px][y + py][i];
                                }
                            }
                        }
                        patch
                    };
                    for (index, (signs, mask, threshold)) in weights.iter().enumerate() {
                        let bit = fc_infer!(9, patch, signs, mask) > *threshold;
                        output[x + 1][y + 1][index / 64] = output[x + 1][y + 1][index / 64] | ((bit as u64) << (index % 64));
                    }
                }
            }
            output
        }
    };
}

fixed_dim_extract_3x3_patchs!(extract_patches, 32, 32, 1);
readout_filters!(l1_readout, 9, 10);
fc_is_correct!(is_correct, 9, 10);
extract_3x3_patch_sets!(extract_patches_set, 32, 1, 10);
gen_fragment_features!(gen_features, 9, 10);
fixed_dim_apply_3x3_patchs!(conv_apply, 32, 32, 1, 1);

fn load_data() -> Vec<(u8, [[[u64; 1]; 32]; 32])> {
    let paths = vec![
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images = Vec::with_capacity(10000 * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_64chan_10(&path, 10000);
        images.append(&mut batch);
    }
    images
}

fn main() {
    let start = PreciseTime::now();
    let images = load_data();

    println!("load time: {}", start.to(PreciseTime::now()));

    let start = PreciseTime::now();
    let patch_sets = extract_patches_set(&images);
    println!("patch extraction: {}", start.to(PreciseTime::now()));
    {
        let readout_weights = l1_readout(&patch_sets, 0.054);
        for (signs, mask) in readout_weights.iter() {
            println!("filter", );
            for word in mask.iter() {
                println!("{:064b}", word);
            }
        }
        let data_path = String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/test_batch.bin");
        let images = cifar::load_images_64chan_10(&data_path, 10000);
        let patch_sets = extract_patches_set(&images);
        let sum_counts: Vec<(usize, u64)> = patch_sets
            .par_iter()
            .enumerate()
            .map(|(label, patch_set)| (patch_set.len(), patch_set.par_iter().map(|patch| is_correct(&patch, label, &readout_weights) as u64).sum()))
            .collect();
        let label_acc: Vec<f64> = sum_counts.iter().map(|(count, correct)| *correct as f64 / *count as f64).collect();
        println!("{:?}", label_acc);
        let (total_patches, total_correct): (usize, u64) = sum_counts.iter().fold((0usize, 0u64), |acc, x| (acc.0 + x.0, acc.1 + x.1));
        println!("total patches: {:?}", total_patches);

        let total_acc = total_correct as f64 / total_patches as f64;
        println!("acc {:?}%", total_acc * 100f64);
    }
    let patch_sets: Vec<Vec<&[u64; 9]>> = patch_sets.iter().map(|x| x.iter().collect()).collect();
    let start = PreciseTime::now();
    let features = gen_features(&patch_sets, 0f64, 6);
    println!("feature gen: {}", start.to(PreciseTime::now()));
    println!("{:?}", features.len());
    let start = PreciseTime::now();
    let mut thresholds: Vec<u32> = features
        .par_iter()
        .map(|(signs, mask)| {
            let mut activations: Vec<u32> = patch_sets.par_iter().flatten().map(|input| fc_infer!(9, input, signs, mask)).collect();
            activations.sort();
            activations[activations.len() / 2]
        }).collect();
    println!("thresholds: {}", start.to(PreciseTime::now()));
    println!("{:?}", thresholds);
    let features: Vec<([u64; 9], [u64; 9], u32)> = features.iter().zip(thresholds.iter()).map(|(&(sign, mask), &threshold)| (sign, mask, threshold)).collect();
    let start = PreciseTime::now();
    let images: Vec<(u8, [[[u64; 1]; 32]; 32])> = images.par_iter().map(|(label, input)| (*label, conv_apply(&input, &features))).collect();
    println!("update images: {}", start.to(PreciseTime::now()));
    //for x in 0..32 {
    //    for y in 0..32 {
    //        println!("{:064b}", images[0].1[x][y][0]);
    //    }
    //}

    let start = PreciseTime::now();
    let patch_sets = extract_patches_set(&images);
    println!("patch extraction: {}", start.to(PreciseTime::now()));
    for &thresh in [
        0.0f64, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.081, 0.082, 0.083, 0.084, 0.085, 0.086, 0.087, 0.088, 0.09, 0.091, 0.092, 0.093, 0.094, 0.095, 0.1,
        0.101, 0.012, 0.103, 0.104, 0.105, 0.106, 0.107, 0.108, 0.109, 0.11, 0.111, 0.112, 0.115, 0.12, 0.115, 0.12, 0.13, 0.14, 0.15, 0.2,
    ]
        .iter()
    {
        let readout_weights = l1_readout(&patch_sets, thresh);
        {
            let data_path = String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/test_batch.bin");
            let images = cifar::load_images_64chan_10(&data_path, 10000);
            let patch_sets = extract_patches_set(&images);
            let sum_counts: Vec<(usize, u64)> = patch_sets
                .par_iter()
                .enumerate()
                .map(|(label, patch_set)| (patch_set.len(), patch_set.par_iter().map(|patch| is_correct(&patch, label, &readout_weights) as u64).sum()))
                .collect();
            let label_acc: Vec<f64> = sum_counts.iter().map(|(count, correct)| *correct as f64 / *count as f64).collect();
            //println!("l1: {:?}", label_acc);
            let (total_patches, total_correct): (usize, u64) = sum_counts.iter().fold((0usize, 0u64), |acc, x| (acc.0 + x.0, acc.1 + x.1));
            //println!("l1 total patches: {:?}", total_patches);

            let total_acc = total_correct as f64 / total_patches as f64;
            println!("{:?} | {:?}%", thresh, total_acc * 100f64);
        }
    }
}
