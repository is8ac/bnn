#[macro_use]
extern crate bitnn;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use rayon::prelude::*;
use time::PreciseTime;

macro_rules! fc_infer {
    ($in_size:expr, $weights:expr, $input:expr) => {{
        let mut sum = 0;
        for i in 0..$in_size {
            sum += ($weights[i] ^ $input[i]).count_ones();
        }
        sum
    }};
}

macro_rules! fc_is_correct {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(input: &[u64; $in_size], label: usize, weights: [[u64; $in_size]; $n_labels]) -> bool {
            let target_sum = fc_infer!($in_size, weights[label], input);
            for o in 0..$n_labels {
                if o != label {
                    let sum = fc_infer!($in_size, weights[o], input);
                    if sum >= target_sum {
                        //println!("{:?} {:?}", target_sum, sum);
                        return false;
                    }
                }
            }
            true
        }
    };
}

macro_rules! fc_weights {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(examples: &Vec<&(usize, [u64; $in_size])>) -> [[u64; $in_size]; $n_labels] {
            let sums: Box<[(u32, [[u32; 64]; $in_size]); $n_labels]> = examples
                .par_iter()
                .fold(
                    || Box::new([(0u32, [[0u32; 64]; $in_size]); $n_labels]),
                    |mut counts, example| {
                        counts[example.0].0 += 1;
                        for i in 0..$in_size {
                            for b in 0..64 {
                                counts[example.0].1[i][b] += ((example.1[i] >> b) & 0b1u64) as u32;
                            }
                        }
                        counts
                    },
                )
                .reduce(
                    || Box::new([(0u32, [[0u32; 64]; $in_size]); $n_labels]),
                    |mut a, b| {
                        for l in 0..$n_labels {
                            a[l].0 += b[l].0;
                            for i in 0..$in_size {
                                for bit in 0..64 {
                                    a[l].1[i][bit] += b[l].1[i][bit];
                                }
                            }
                        }
                        a
                    },
                );
            let mut weights = [[0u64; $in_size]; $n_labels];
            for i in 0..$in_size {
                for b in 0..64 {
                    let total_avg_bits = {
                        let mut total_bits = 0;
                        let mut total_count = 0;
                        for l in 0..$n_labels {
                            total_bits += sums[l].1[i][b];
                            total_count += sums[l].0;
                        }
                        total_bits as f32 / total_count as f32
                    };
                    for l in 0..$n_labels {
                        let avg_bits = sums[l].1[i][b] as f32 / sums[l].0 as f32;
                        let grad = total_avg_bits - avg_bits;
                        let bit = grad > 0f32;
                        weights[l][i] = weights[l][i] | ((bit as u64) << b);
                    }
                }
            }
            weights
        }
    };
}

macro_rules! fc_boosted_features {
    ($name:ident, $in_size:expr, $n_labels:expr, $boosting_iters:expr) => {
        fn $name(examples: &Vec<(usize, [u64; $in_size])>) -> [[[u64; $in_size]; $n_labels]; $boosting_iters] {
            fc_is_correct!(correct, $in_size, $n_labels);
            fc_weights!(features, $in_size, $n_labels);

            let mut weights_set = [[[0u64; $in_size]; $n_labels]; $boosting_iters];
            let mut boosted_set = examples.iter().collect();
            weights_set[0] = features(&boosted_set);
            for boosting in 1..$boosting_iters {
                boosted_set = boosted_set
                    .iter()
                    .map(|&x| x)
                    .filter(|(label, image)| !correct(&image, *label, weights_set[boosting - 1]))
                    .collect();
                println!("boosted: {:?}", boosted_set.len());
                weights_set[boosting] = features(&boosted_set);
            }
            weights_set
        }
    };
}

macro_rules! reflow {
    ($name:ident, $in_size:expr, $n_labels:expr, $boostings:expr, $output_words:expr) => {
        fn $name(input: &[[[u64; $in_size]; $n_labels]; $boostings]) -> [[[u64; $in_size]; 64]; $output_words] {
            let mut output = [[[0u64; $in_size]; 64]; $output_words];
            for o in 0..$output_words {
                for b in 0..64 {
                    let index = (o * 64) + b;
                    output[o][b] = input[index / $n_labels][index % $n_labels];
                }
            }
            output
        }
    };
}

macro_rules! fused_xor_popcount_threshold_and_bitpack {
    ($name:ident, $in_size:expr, $out_size:expr) => {
        fn $name(input: &[u64; $in_size], weights: &[[[u64; $in_size]; 64]; $out_size], thresholds: &[[u32; 64]; $out_size]) -> [u64; $out_size] {
            let mut output = [0u64; $out_size];
            for o in 0..$out_size {
                for b in 0..64 {
                    let bit = fc_infer!($in_size, weights[o][b], input) > thresholds[o][b];
                    output[o] = output[o] | ((bit as u64) << b);
                }
            }
            output
        }
    };
}

macro_rules! activation_medians {
    ($name:ident, $in_size:expr, $out_size:expr) => {
        fn $name(inputs: &Vec<[u64; $in_size]>, weights: &[[[u64; $in_size]; 64]; $out_size]) -> [[u32; 64]; $out_size] {
            let mut output = [[0u32; 64]; $out_size];
            for o in 0..$out_size {
                for b in 0..64 {
                    let mut sums: Vec<u32> = inputs.par_iter().map(|input| fc_infer!($in_size, weights[o][b], input)).collect();
                    sums.sort();
                    output[o][b] = sums[inputs.len() / 2];
                }
            }
            output
        }
    };
}

const TRAIN_SIZE: usize = 60000;

const L1_B: usize = 14;
const L2_W: usize = 2;
const L2_B: usize = 20;

fc_is_correct!(l1_is_correct, 13, 10);
fc_boosted_features!(l1_boosted_features, 13, 10, L1_B);
reflow!(l1_reflow, 13, 10, L1_B, L2_W);
activation_medians!(l1_medians, 13, 2);
fused_xor_popcount_threshold_and_bitpack!(l1_fxoptbp, 13, L2_W);


fc_is_correct!(l2_is_correct, 2, 10);
fc_boosted_features!(l2_boosted_features, 2, 10, L2_B);
//reflow!(l2_reflow, 2, 10, L2_B, L2_W);
//fused_xor_popcount_and_bitpack!(l2_xopcbp, 13, L2_W);

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let l1_examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

    let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
    let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
    let test_examples: Vec<(usize, [u64; 13])> = test_labels
        .iter()
        .zip(test_images.iter())
        .map(|(&label, &image)| (label as usize, image))
        .collect();

    let weights = l1_boosted_features(&l1_examples);
    let l1_features = l1_reflow(&weights);
    let thresholds = l1_medians(&images, &l1_features);
    for i in 0..L1_B {
        let total_correct: u64 = test_examples
            .par_iter()
            .map(|(label, image)| l1_is_correct(&image, *label, weights[i]))
            .map(|x| x as u64)
            .sum();
        let avg_correct = total_correct as f32 / test_examples.len() as f32;
        println!("{:?} acc: {:?}%", i, avg_correct * 100f32);
    }
    let l2_images: Vec<[u64; 2]> = images.par_iter().map(|image| l1_fxoptbp(&image, &l1_features, &thresholds)).collect();
    let l2_examples: Vec<(usize, [u64; 2])> = labels
        .iter()
        .zip(l2_images.iter())
        .map(|(&label, &image)| (label as usize, image))
        .collect();

    let l2_test_images: Vec<[u64; 2]> = test_images.par_iter().map(|image| l1_fxoptbp(&image, &l1_features, &thresholds)).collect();
    let l2_test_examples: Vec<(usize, [u64; 2])> = test_labels
        .iter()
        .zip(l2_test_images.iter())
        .map(|(&label, &image)| (label as usize, image))
        .collect();

    println!("images: {:?}", &l2_images[0..20]);

    let weights = l2_boosted_features(&l2_examples);
    //let l2_features = l1_reflow(&weights);
    for i in 0..L2_B {
        let total_correct: u64 = l2_test_examples
            .par_iter()
            .map(|(label, image)| l2_is_correct(&image, *label, weights[i]))
            .map(|x| x as u64)
            .sum();
        let avg_correct = total_correct as f32 / test_examples.len() as f32;
        println!("{:?} acc: {:?}%", i, avg_correct * 100f32);
    }
}
