#[macro_use]
extern crate bitnn;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use rayon::prelude::*;
use time::PreciseTime;

macro_rules! bitpack_u64_2d {
    ($name:ident, $type:ty, $a_size:expr, $b_size:expr, $thresh:expr) => {
        fn $name(grads: &[[[$type; 64]; $b_size]; $a_size]) -> [[[u64; $c_size]; $b_size]; $a_size] {
            let mut params = [[[0u64; $c_size]; $b_size]; $a_size];
            for a in 0..$a_size {
                for b in 0..$b_size {
                    for i in 0..64 {
                        let bit = grads[a][b][i] > $thresh;
                        params[a][b] = params[a][b] | ((bit as u64) << i);
                    }
                }
            }
            params
        }
    };
}

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
                    if sum > target_sum {
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

const TRAIN_SIZE: usize = 60000;

fc_boosted_features!(boosted_grads, 13, 10, 20);

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let l1_weights = boosted_grads(&examples);
    for weights in &l1_weights {
        println!("{:064b}", weights[0][4]);
    }
}
