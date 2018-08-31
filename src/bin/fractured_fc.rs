extern crate bitnn;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use rayon::prelude::*;
use std::collections::HashMap;
//use time::PreciseTime;

#[allow(unused_macros)]
macro_rules! avg_bits {
    ($examples:expr, $in_size:expr) => {{
        let sums: Vec<u32> = $examples
            .par_iter()
            .fold(
                || vec![0u32; $in_size * 64],
                |mut counts, example| {
                    for i in 0..$in_size {
                        let offset = i * 64;
                        for b in 0..64 {
                            counts[offset + b] += ((example[i] >> b) & 0b1u64) as u32;
                        }
                    }
                    counts
                },
            ).reduce(
                || vec![0u32; $in_size * 64],
                |mut a, b| {
                    for i in 0..$in_size * 64 {
                        a[i] += b[i];
                    }
                    a
                },
            );
        let len = $examples.len() as f64;
        let mut avgs = [0f64; $in_size * 64];
        for i in 0..$in_size * 64 {
            avgs[i] = sums[i] as f64 / len;
        }
        avgs
    }};
}

#[allow(unused_macros)]
macro_rules! fc_split {
    ($name:ident, $in_size:expr) => {
        fn $name(a: &Vec<&[u64; $in_size]>, b: &Vec<&[u64; $in_size]>, mask_thresh: f64) -> ([u64; $in_size], [u64; $in_size]) {
            let a_avg = avg_bits!(a, $in_size);
            let b_avg = avg_bits!(b, $in_size);
            let mut weights = ([0u64; $in_size], [0u64; $in_size]);
            for i in 0..$in_size * 64 {
                let word = i / 64;
                let bit = i % 64;
                let grad = a_avg[i] - b_avg[i];
                let sign = grad > 0f64;
                weights.1[word] = weights.1[word] | ((sign as u64) << bit);
                let magn = grad.abs() > mask_thresh;
                weights.0[word] = weights.0[word] | ((magn as u64) << bit);
            }
            weights
        }
    };
}

#[allow(unused_macros)]
macro_rules! fc_infer {
    ($in_size:expr, $input:expr, $mask:expr, $signs:expr) => {{
        let mut sum = 0;
        for i in 0..$in_size {
            sum += (($signs[i] ^ $input[i]) & $mask[i]).count_ones();
        }
        sum
    }};
}

#[allow(unused_macros)]
macro_rules! fc_features {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(examples: &Vec<&(usize, [u64; $in_size])>, mask_thresh: f64, level: usize) -> Vec<([u64; $in_size], [u64; $in_size], u32)> {
            fc_split!(split, $in_size);
            let label_dist: [u32; 10] = examples
                .par_iter()
                .fold(
                    || [0u32; $n_labels],
                    |mut counts, (label, _)| {
                        counts[*label] += 1;
                        counts
                    },
                ).reduce(
                    || [0u32; $n_labels],
                    |mut a, b| {
                        for i in 0..$n_labels {
                            a[i] += b[i]
                        }
                        a
                    },
                );
            println!("{:?} {:?}", level, label_dist);
            let largest_label = label_dist.iter().enumerate().max_by_key(|(_, &count)| count).unwrap().0;
            let in_group: Vec<&[u64; $in_size]> = examples.par_iter().filter(|(label, _)| *label == largest_label).map(|(_, input)| input).collect();
            let out_group: Vec<&[u64; $in_size]> = examples.par_iter().filter(|(label, _)| *label != largest_label).map(|(_, input)| input).collect();
            let weights = split(&out_group, &in_group, mask_thresh);
            let mut activations: Vec<u32> = examples.par_iter().map(|&(_, input)| fc_infer!($in_size, input, weights.0, weights.1)).collect();
            activations.sort();
            let threshold = activations[examples.len() / 2];
            if level <= 0 {
                return vec![(weights.0, weights.1, threshold)];
            }
            println!("continuing {:?}", level);
            let over: Vec<&(usize, [u64; $in_size])> = examples
                .par_iter()
                .map(|&x| x)
                .filter(|&(_, input)| fc_infer!($in_size, input, weights.0, weights.1) >= threshold)
                .collect();
            let under: Vec<&(usize, [u64; $in_size])> = examples
                .par_iter()
                .map(|&x| x)
                .filter(|&(_, input)| fc_infer!($in_size, input, weights.0, weights.1) < threshold)
                .collect();
            let mut over_elems = $name(&over, mask_thresh, level - 1);
            let mut under_elems = $name(&under, mask_thresh, level - 1);
            over_elems.append(&mut under_elems);
            over_elems
        }
    };
}

//macro_rules! pack_filters {
//    ($name:ident, $in_size:expr, $out_size:expr) => {
//        fn $name(input: Vec<(u32, [u64; $in_size])>) -> ([[u32; 64]; $out_size], [[[u64; $in_size]; 64]; $out_size]) {
//            let mut packed_weights = [[[0u64; $in_size]; 64]; $out_size];
//            let mut thresholds = [[0u32; 64]; $out_size];
//            for o in 0..$out_size {
//                for b in 0..64 {
//                    thresholds[o][b] = input[(o * 64) + b].0;
//                    packed_weights[o][b] = input[(o * 64) + b].1;
//                }
//            }
//            (thresholds, packed_weights)
//        }
//    };
//}

#[allow(unused_macros)]
macro_rules! vector_fused_xor_popcount_threshold_and_bitpack {
    ($name:ident, $in_size:expr, $out_size:expr) => {
        fn $name(input: &[u64; $in_size], params: &Vec<([u64; $in_size], [u64; $in_size], u32)>) -> [u64; $out_size] {
            let mut output = [0u64; $out_size];
            for (i, (mask, signs, thresh)) in params.iter().enumerate() {
                let bit = fc_infer!($in_size, input, mask, signs) > *thresh;
                output[i / 64] = output[i / 64] | ((bit as u64) << (i % 64));
            }
            output
        }
    };
}

//macro_rules! fused_xor_popcount_threshold_and_bitpack {
//    ($name:ident, $in_size:expr, $out_size:expr) => {
//        fn $name(input: &[u64; $in_size], weights: &[[[u64; $in_size]; 64]; $out_size], thresholds: &[[u32; 64]; $out_size]) -> [u64; $out_size] {
//            let mut output = [0u64; $out_size];
//            for o in 0..$out_size {
//                for b in 0..64 {
//                    let bit = fc_infer!($in_size, weights[o][b], input) > thresholds[o][b];
//                    output[o] = output[o] | ((bit as u64) << b);
//                }
//            }
//            output
//        }
//    };
//}
#[allow(unused_macros)]
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

#[allow(unused_macros)]
macro_rules! readout_filters {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(examples: &Vec<&(usize, [u64; $in_size])>, mask_thresh: f64) -> [([u64; $in_size], [u64; $in_size]); $n_labels] {
            let mut weights = [([0u64; $in_size], [0u64; $in_size]); $n_labels];
            let nil_set: Vec<&[u64; $in_size]> = examples.par_iter().map(|(_, input)| input).collect();
            let nil_avg = avg_bits!(nil_set, $in_size);
            for o in 0..$n_labels {
                let label_set: Vec<&[u64; $in_size]> = examples.par_iter().filter(|(label, _)| *label == o).map(|(_, input)| input).collect();
                let label_avg = avg_bits!(label_set, $in_size);
                for i in 0..$in_size * 64 {
                    let word = i / 64;
                    let bit = i % 64;
                    let grad = nil_avg[i] - label_avg[i];
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

fn remove_index_from_2d_vec(matrix: &mut HashMap<usize, HashMap<usize, f64>>, index: usize) {
    matrix.remove(&index);
    for (_, map) in matrix {
        map.remove(&index);
    }
}

macro_rules! cov_matrix_bitpacked {
    ($name:ident, $in_size:expr) => {
        fn $name(inputs: &Vec<[u64; $in_size]>) -> HashMap<usize, HashMap<usize, f64>> {
            let len = inputs.len() as f64;
            let n_features = $in_size * 64;
            let mut matrix: HashMap<usize, HashMap<usize, f64>> = inputs
                .par_iter()
                .fold(
                    || vec![vec![0i32; n_features]; n_features],
                    |mut counts, input| {
                        // for each datum, calculate the features, storing each as an i8 of either -1 or +1 for latter convenience.
                        // fc_infer XORs the input with the provided filter and counts the bits set.
                        // We convert the bits into an i8 of either -1 or +1.
                        let features: Vec<i8> = (0..n_features)
                            .map(|index| (((input[index / 64] >> (index % 64)) & 0b1u64) as u8) as i8 * 2 - 1)
                            .collect();
                        // For each permutaion of features, multiply and increment the counts accordingly.
                        for a in 0..n_features {
                            for b in a..n_features {
                                counts[a][b] += (features[a] * features[b]) as i32;
                            }
                        }
                        counts
                    },
                ).reduce(
                    || vec![vec![0i32; n_features]; n_features],
                    |mut x, y| {
                        for a in 0..n_features {
                            for b in a..n_features {
                                x[a][b] += y[a][b];
                            }
                        }
                        x
                    },
                ).iter()
                .map(|x| x.iter().enumerate().map(|(index, &val)| (index, val.abs() as f64 / len)).collect())
                .enumerate()
                .collect();

            for a in 0..n_features {
                for b in a..n_features {
                    let value = matrix[&a][&b];
                    matrix.get_mut(&b).unwrap().insert(a, value);
                }
            }
            matrix
        }
    };
}

macro_rules! threshold_cov_matrix_filter {
    ($name:ident) => {
        fn $name(mut cov_matrix: HashMap<usize, HashMap<usize, f64>>, threshold: f64) -> Vec<usize> {
            let mut val: (usize, f64) = (0, 123f64);
            while val.1 > threshold {
                // we find the index of the largest average covariance.
                let n_features = cov_matrix.len() as f64;
                val = cov_matrix
                    .iter()
                    .map(|(&index, vals)| {
                        let sum: f64 = vals.iter().map(|(_, &val)| val).sum();
                        (index, sum / n_features)
                    }).max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                // and remove it from both dimentions of the matrix.
                if val.1 > threshold {
                    remove_index_from_2d_vec(&mut cov_matrix, val.0);
                }
            }
            cov_matrix.iter().map(|(&index, _)| index).collect()
        }
    };
}

#[allow(unused_macros)]
macro_rules! cov_mask {
    ($name:ident, $in_size:expr, $n_features:expr) => {
        fn $name(inputs: &Vec<[u64; $in_size]>, threshold: f64) -> [u64; $in_size] {
            cov_matrix_bitpacked!(cov_matrix, $in_size);
            threshold_cov_matrix_filter!(threshold_filter);
            let mut matrix = cov_matrix(&inputs);
            let indices = threshold_filter(matrix, threshold);
            println!("{:?} ", indices.len());
            let mut mask = [0u64; $in_size];
            for index in indices {
                mask[index / 64] = mask[index / 64] | (0b1u64 << (index % 64));
            }
            mask
        }
    };
}

// cov_filter creates a function to generate a subset of the features which have low covariance.
// inputs is the unlabeled raw data.
// parameters is the filters and corresponding thresholds generated by some dataset fracturing algorithm.
#[allow(unused_macros)]
macro_rules! cov_filter {
    ($name:ident, $in_size:expr, $n_features:expr) => {
        fn $name(inputs: &Vec<[u64; $in_size]>, parameters: &[(u32, [u64; $in_size]); $n_features], target_len: usize) -> Vec<(u32, [u64; $in_size])> {
            let len = inputs.len() as f64;
            // First we must create a nested HashMap of the covariances betwene the features.
            let mut matrix: HashMap<usize, HashMap<usize, f64>> = inputs
                .par_iter()
                .fold(
                    || Box::new([[0i32; $n_features]; $n_features]),
                    |mut counts, input| {
                        // for each datum, calculate the features, storing each as an i8 of either -1 or +1 for latter convenience.
                        let features: Vec<i8> = parameters
                            .iter()
                            .map(|(threshold, filter)| ((fc_infer!($in_size, filter, input) > *threshold) as u8) as i8 * 2 - 1)
                            .collect();
                        // for each permutaion of features, multiply and increment the counts accordingly.
                        // Note that, as multiplication is commutative, we are doing twice as much work as we need to do. But it's cheap.
                        for a in 0..$n_features {
                            for b in 0..$n_features {
                                counts[a][b] += (features[a] * features[b]) as i32;
                            }
                        }
                        counts
                    },
                ).reduce(
                    || Box::new([[0i32; $n_features]; $n_features]),
                    |mut x, y| {
                        for a in 0..$n_features {
                            for b in 0..$n_features {
                                x[a][b] += y[a][b];
                            }
                        }
                        x
                    },
                ).iter()
                .map(|x| x.iter().enumerate().map(|(index, &val)| (index, val.abs() as f64 / len)).collect())
                .enumerate()
                .collect();
            // Now, many times,
            for _i in 0..$n_features - target_len {
                // we find the index of the largest average avg covariance.
                let (largest_index, _val): (usize, f64) = matrix
                    .iter()
                    .map(|(&index, vals)| {
                        let sum: f64 = vals.iter().map(|(_, &val)| val).sum();
                        (index, sum / $n_features as f64)
                    }).max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                // and remove it from both dimentions of the matrix.
                remove_index_from_2d_vec(&mut matrix, largest_index);
            }
            matrix.iter().map(|(&index, _)| parameters[index]).collect()
        }
    };
}

#[allow(unused_macros)]
macro_rules! vec2array {
    ($vector:expr, $size:expr, $type:expr) => {{
        let mut array = [$type; $size];
        for i in 0..$size {
            array[i] = $vector[i];
        }
        array
    }};
}

#[allow(unused_macros)]
macro_rules! print_acc {
    ($prefix:expr, $in_size:expr, $n_labels:expr, $examples:expr, $test_examples:expr, $magn_threshold:expr) => {{
        readout_filters!(readouts, $in_size, $n_labels);
        fc_is_correct!(is_correct, $in_size, $n_labels);
        let readout_weights = readouts(&$examples, $magn_threshold);
        //for l in 0..10 {
        //    print_image(&readout_weights[l].0);
        //    print_image(&readout_weights[l].1);
        //}
        let total_correct: u64 = $test_examples
            .par_iter()
            .map(|(label, input)| is_correct(&input, *label, &readout_weights))
            .map(|x| x as u64)
            .sum();
        let avg_correct = total_correct as f32 / $test_examples.len() as f32;
        println!("{:}{:?}%", $prefix, avg_correct * 100f32);
    }};
}

const TRAIN_SIZE: usize = 60000;
const L1_POWER: usize = 6;
const L1_N_FILTERS: usize = 64;
const L1_SIZE: usize = 1;
//const L2_SIZE: usize = 2;
//const L2_N_FILTERS: usize = 128;

cov_mask!(l1_cov_mask, 13, 13 * 64);

fc_features!(l1_features, 13, 10);
////cov_filter!(l1_cov_filter, 13, L1_N_FILTERS);
vector_fused_xor_popcount_threshold_and_bitpack!(l1_fxoptbp, 13, L1_SIZE);

//fc_features!(l2_features, L1_SIZE, 10);
//cov_filter!(l2_cov_filter, L1_SIZE, L2_N_FILTERS);
//vector_fused_xor_popcount_threshold_and_bitpack!(l2_fxoptbp, L1_SIZE, L2_SIZE);

fn print_image(input: &[u64; 13]) {
    let mut bits = vec![];
    for word in input {
        let mut chars: Vec<char> = format!("{:064b}", word).chars().collect();
        chars.reverse();
        bits.append(&mut chars);
    }
    let rows = bits.chunks(28);
    for row in rows {
        let string: String = row.iter().collect();
        println!("{:}", string);
    }
}

macro_rules! apply_mask {
    ($in_size:expr, $mask:expr, $input:expr) => {{
        let mut output = [0u64; 13];
        for i in 0..$in_size {
            output[i] = $mask[i] & $input[i];
        }
        output
    }};
}

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let examples: Vec<&(usize, [u64; 13])> = examples.iter().collect();
    print_acc!("pre mask: ", 13, 10, examples, examples, 0f64);

    let mask = l1_cov_mask(&images, 0.14);

    let new_images: Vec<[u64; 13]> = images.par_iter().map(|image| apply_mask!(13, mask, image)).collect();
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(new_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let examples: Vec<&(usize, [u64; 13])> = examples.iter().collect();
    print_acc!("post mask: ", 13, 10, examples, examples, 0f64);

    let features_vec = l1_features(&examples, 0.0, L1_POWER);
    //let features_array = vec2array!(features_vec, L1_N_FILTERS, (0u32, [0u64; 13]));
    //let parameters = l1_cov_filter(&images, &features_array, 128);
    //println!("{:?}", parameters.len());

    let new_images: Vec<[u64; L1_SIZE]> = images.par_iter().map(|image| l1_fxoptbp(&image, &features_vec)).collect();
    let examples: Vec<(usize, [u64; L1_SIZE])> = labels.iter().zip(new_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let examples: Vec<&(usize, [u64; L1_SIZE])> = examples.iter().collect();
    for &thresh in [0.01f64, 0.02, 0.03, 0.1, 0.2].iter() {
        print!("{:?} | ", thresh);
        print_acc!("layer 1: ", L1_SIZE, 10, examples, examples, thresh);
    }

    //let features_vec = l2_features(&examples, 7);
    ////let features_array = vec2array!(features_vec, L2_N_FILTERS, (0u32, [0u64; L1_SIZE]));
    ////let parameters = l2_cov_filter(&new_images, &features_array, 128);
    //let new_images: Vec<[u64; L2_SIZE]> = new_images.par_iter().map(|image| l2_fxoptbp(&image, &features_vec)).collect();
    //let examples: Vec<(usize, [u64; L2_SIZE])> = labels.iter().zip(new_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    //let examples: Vec<&(usize, [u64; L2_SIZE])> = examples.iter().collect();
    //print_acc!(L2_SIZE, 10, examples, examples);
}
