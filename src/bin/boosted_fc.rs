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
                    let label_sum = fc_infer!($in_size, weights[o], input);
                    //println!("{:?} {:?} {:?}", o, target_sum, label_sum);
                    //println!("{:?} {:?}", label, o);
                    if label_sum >= target_sum {
                        return false;
                    }
                }
            }
            //println!("falling", );
            true
        }
    };
}

macro_rules! fc_keep {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(input: &[u64; $in_size], label: usize, weights: [[u64; $in_size]; $n_labels]) -> bool {
            let target_sum = fc_infer!($in_size, weights[label], input);
            for o in 0..$n_labels {
                if o != label {
                    let label_sum = fc_infer!($in_size, weights[o], input);
                    if label_sum >= target_sum {
                        return true;
                    }
                }
            }
            false
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
                ).reduce(
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
                        //println!("grad: {:?}", grad);
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
            fc_keep!(keep, $in_size, $n_labels);
            fc_weights!(features, $in_size, $n_labels);

            let mut weights_set = [[[0u64; $in_size]; $n_labels]; $boosting_iters];
            let mut boosted_set = examples.iter().collect();
            weights_set[0] = features(&boosted_set);
            for boosting in 1..$boosting_iters {
                boosted_set = boosted_set
                    .iter()
                    .map(|&x| x)
                    .filter(|(label, image)| keep(&image, *label, weights_set[boosting - 1]))
                    .collect();
                //println!("boosted: {:?}", boosted_set.len());
                let freqs: [u32; 10] = boosted_set.iter().map(|(label, _)| label).fold([0u32; 10], |mut counts, &label| {
                    counts[label] += 1;
                    counts
                });
                println!("{:?} {:?}", boosted_set.len(), freqs);
                weights_set[boosting] = features(&boosted_set);
                println!("{:?}", weights_set[boosting]);
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

const L1_B: usize = 20;
const L2_W: usize = 3;
const L2_B: usize = 20;

fc_boosted_features!(l1_boosted_features, 13, 10, L1_B);
reflow!(l1_reflow, 13, 10, L1_B, L2_W);
activation_medians!(l1_medians, 13, L2_W);

fc_boosted_features!(l2_boosted_features, L2_W, 10, L2_B);
reflow!(l2_reflow, L2_W, 10, L2_B, L2_W);
activation_medians!(l2_medians, L2_W, L2_W);

macro_rules! update_examples {
    ($images:expr, $labels:expr, $features:expr, $thresholds:expr, $in_size:expr, $out_size:expr) => {{
        fused_xor_popcount_threshold_and_bitpack!(fxoptbp, $in_size, L2_W);

        let new_images: Vec<[u64; $out_size]> = $images.par_iter().map(|image| fxoptbp(&image, &$features, &$thresholds)).collect();
        let examples: Vec<(usize, [u64; $out_size])> = $labels.iter().zip(new_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
        (new_images, examples)
    }};
}

macro_rules! print_acc {
    ($test_examples:expr, $weights:expr, $boostings:expr, $in_size:expr, $n_labels:expr) => {{
        fc_is_correct!(is_correct, $in_size, $n_labels);
        for i in 0..$boostings {
            let total_correct: u64 = $test_examples
                .par_iter()
                .map(|(label, image)| is_correct(&image, *label, $weights[i]))
                .map(|x| x as u64)
                .sum();
            let avg_correct = total_correct as f32 / $test_examples.len() as f32;
            println!("{:?} acc: {:?}%", i, avg_correct * 100f32);
        }
    }};
}

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/fashion_mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

    let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/fashion_mnist/t10k-images-idx3-ubyte"), 10000);
    let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
    let test_examples: Vec<(usize, [u64; 13])> = test_labels.iter().zip(test_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

    let weights = l1_boosted_features(&examples);
    print_acc!(examples, weights, L1_B, 13, 10);
    let features = l1_reflow(&weights);
    let thresholds = l1_medians(&images, &features);
    let (images, examples) = update_examples!(images, labels, features, thresholds, 13, L2_W);
    let (test_images, test_examples) = update_examples!(test_images, test_labels, features, thresholds, 13, L2_W);

    let weights = l2_boosted_features(&examples);
    print_acc!(examples, weights, L2_B, L2_W, 10);
    let features = l2_reflow(&weights);
    let thresholds = l2_medians(&images, &features);
    let (images, examples) = update_examples!(images, labels, features, thresholds, L2_W, L2_W);
    let (test_images, test_examples) = update_examples!(test_images, test_labels, features, thresholds, L2_W, L2_W);

    println!("starting l3",);
    let weights = l2_boosted_features(&examples);
    print_acc!(examples, weights, L2_B, L2_W, 10);
    let features = l2_reflow(&weights);
    let thresholds = l2_medians(&images, &features);
    let (images, examples) = update_examples!(images, labels, features, thresholds, L2_W, L2_W);
    let (test_images, test_examples) = update_examples!(test_images, test_labels, features, thresholds, L2_W, L2_W);

    println!("starting l4",);
    let weights = l2_boosted_features(&examples);
    print_acc!(test_examples, weights, L2_B, L2_W, 10);
    let features = l2_reflow(&weights);
    let thresholds = l2_medians(&images, &features);
    let (images, examples) = update_examples!(images, labels, features, thresholds, L2_W, L2_W);
    //let (test_images, test_examples) = update_examples!(test_images, test_labels, features, thresholds, L2_W, L2_W);

    for (label, image) in examples.iter().take(30) {
        //println!("{:?} {:064b} {:064b} {:064b}", label, image[0], image[1], image[2]);
        println!("{:?} {:064b}", label, image[0]);
    }

    //println!("starting l5",);
    //let weights = l2_boosted_features(&examples);
    //print_acc!(examples, weights, L2_B, L2_W, 10);
    //let features = l2_reflow(&weights);
    //let thresholds = l2_medians(&images, &features);
    //let (images, examples) = update_examples!(images, labels, features, thresholds, L2_W, L2_W);
    //let (test_images, test_examples) = update_examples!(test_images, test_labels, features, thresholds, L2_W, L2_W);
}
