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
        let sums: Box<[[u32; 64]; $in_size]> = $examples
            .par_iter()
            .fold(
                || Box::new([[0u32; 64]; $in_size]),
                |mut counts, example| {
                    for i in 0..$in_size {
                        for b in 0..64 {
                            counts[i][b] += ((example[i] >> b) & 0b1u64) as u32;
                        }
                    }
                    counts
                },
            )
            .reduce(
                || Box::new([[0u32; 64]; $in_size]),
                |mut a, b| {
                    for i in 0..$in_size {
                        for bit in 0..64 {
                            a[i][bit] += b[i][bit];
                        }
                    }
                    a
                },
            );
        let mut avgs = [[0f32; 64]; $in_size];
        let len = $examples.len() as f32;
        for i in 0..$in_size {
            for b in 0..64 {
                avgs[i][b] = sums[i][b] as f32 / len;
            }
        }
        avgs
    }};
}

#[allow(unused_macros)]
macro_rules! fc_split {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(a: &Vec<&[u64; $in_size]>, b: &Vec<&[u64; $in_size]>) -> [u64; $in_size] {
            let a_avg = avg_bits!(a, $in_size);
            let b_avg = avg_bits!(b, $in_size);
            let mut filter = [0u64; $in_size];
            for i in 0..$in_size {
                for b in 0..64 {
                    let grad = a_avg[i][b] - b_avg[i][b];
                    let bit = grad > 0f32;
                    filter[i] = filter[i] | ((bit as u64) << b);
                }
            }
            filter
        }
    };
}
#[allow(unused_macros)]
macro_rules! fc_infer {
    ($in_size:expr, $weights:expr, $input:expr) => {{
        let mut sum = 0;
        for i in 0..$in_size {
            sum += ($weights[i] ^ $input[i]).count_ones();
        }
        sum
    }};
}
#[allow(unused_macros)]
macro_rules! fc_features {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(examples: &Vec<&(usize, [u64; $in_size])>, level: usize) -> Vec<(u32, [u64; $in_size])> {
            fc_split!(split, $in_size, $n_labels);
            let label_dist: [u32; 10] = examples
                .par_iter()
                .fold(
                    || [0u32; $n_labels],
                    |mut counts, (label, _)| {
                        counts[*label] += 1;
                        counts
                    },
                )
                .reduce(
                    || [0u32; $n_labels],
                    |mut a, b| {
                        for i in 0..$n_labels {
                            a[i] += b[i]
                        }
                        a
                    },
                );
            //println!("{:?} {:?}", level, label_dist);
            let largest_label = label_dist.iter().enumerate().max_by_key(|(_, &count)| count).unwrap().0;
            let in_group: Vec<&[u64; $in_size]> = examples.par_iter().filter(|(label, _)| *label == largest_label).map(|(_, input)| input).collect();
            let out_group: Vec<&[u64; $in_size]> = examples.par_iter().filter(|(label, _)| *label != largest_label).map(|(_, input)| input).collect();
            let filter = split(&out_group, &in_group);
            let mut activations: Vec<u32> = examples.par_iter().map(|&(_, image)| fc_infer!($in_size, filter, image)).collect();
            activations.sort();
            let threshold = activations[examples.len() / 2];
            if level <= 0 {
                return vec![(threshold, filter)];
            }
            let over: Vec<&(usize, [u64; $in_size])> = examples
                .par_iter()
                .map(|&x| x)
                .filter(|&(_, image)| fc_infer!($in_size, filter, image) >= threshold)
                .collect();
            let under: Vec<&(usize, [u64; $in_size])> = examples
                .par_iter()
                .map(|&x| x)
                .filter(|&(_, image)| fc_infer!($in_size, filter, image) < threshold)
                .collect();
            let mut over_elems = $name(&over, level - 1);
            let mut under_elems = $name(&under, level - 1);
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
        fn $name(input: &[u64; $in_size], weights: &Vec<(u32, [u64; $in_size])>) -> [u64; $out_size] {
            let mut output = [0u64; $out_size];
            for (i, (thresh, filter)) in weights.iter().enumerate() {
                let bit = fc_infer!($in_size, filter, input) > *thresh;
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
        fn $name(input: &[u64; $in_size], label: usize, weights: &[[u64; $in_size]; $n_labels]) -> bool {
            let target_sum = fc_infer!($in_size, weights[label], input);
            for o in 0..$n_labels {
                if o != label {
                    let label_sum = fc_infer!($in_size, weights[o], input);
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
        fn $name(examples: &Vec<&(usize, [u64; $in_size])>) -> [[u64; $in_size]; $n_labels] {
            let mut filters = [[0u64; $in_size]; $n_labels];
            let nil_set: Vec<&[u64; $in_size]> = examples.par_iter().map(|(_, input)| input).collect();
            let nil_avg = avg_bits!(nil_set, $in_size);
            for o in 0..$n_labels {
                let label_set: Vec<&[u64; $in_size]> = examples.par_iter().filter(|(label, _)| *label == o).map(|(_, input)| input).collect();
                let label_avg = avg_bits!(label_set, $in_size);
                for i in 0..$in_size {
                    for b in 0..64 {
                        let grad = nil_avg[i][b] - label_avg[i][b];
                        let bit = grad > 0f32;
                        filters[o][i] = filters[o][i] | ((bit as u64) << b);
                    }
                }
            }
            filters
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
                )
                .reduce(
                    || vec![vec![0i32; n_features]; n_features],
                    |mut x, y| {
                        for a in 0..n_features {
                            for b in a..n_features {
                                x[a][b] += y[a][b];
                            }
                        }
                        x
                    },
                )
                .iter()
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

#[allow(unused_macros)]
macro_rules! cov_mask {
    ($name:ident, $in_size:expr, $n_features:expr) => {
        fn $name(inputs: &Vec<[u64; $in_size]>, target_len: usize) -> [bool; $n_features] {
            cov_matrix_bitpacked!(cov_matrix, $in_size);
            let mut matrix = cov_matrix(&inputs);
            // Now, many times,
            for _i in 0..$n_features - target_len {
                // we find the index of the largest average covariance.
                let (largest_index, _val): (usize, f64) = matrix
                    .iter()
                    .map(|(&index, vals)| {
                        let sum: f64 = vals.iter().map(|(_, &val)| val).sum();
                        (index, sum / $n_features as f64)
                    })
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                // and remove it from both dimentions of the matrix.
                remove_index_from_2d_vec(&mut matrix, largest_index);
            }
            let mut indices: Vec<usize> = matrix.iter().map(|(&index, _)| index).collect();
            indices.sort();
            println!("{:?}", indices);
            let mut filter = [false; $n_features];
            for i in indices {
                filter[i] = true;
            }
            filter
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
                )
                .reduce(
                    || Box::new([[0i32; $n_features]; $n_features]),
                    |mut x, y| {
                        for a in 0..$n_features {
                            for b in 0..$n_features {
                                x[a][b] += y[a][b];
                            }
                        }
                        x
                    },
                )
                .iter()
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
                    })
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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
    ($in_size:expr, $n_labels:expr, $examples:expr, $test_examples:expr) => {{
        readout_filters!(readouts, $in_size, $n_labels);
        fc_is_correct!(is_correct, $in_size, $n_labels);
        let readout_filters = readouts(&$examples);
        let total_correct: u64 = $test_examples
            .par_iter()
            .map(|(label, image)| is_correct(&image, *label, &readout_filters))
            .map(|x| x as u64)
            .sum();
        let avg_correct = total_correct as f32 / $test_examples.len() as f32;
        println!("acc: {:?}%", avg_correct * 100f32);
    }};
}

const TRAIN_SIZE: usize = 6000;
//const L1_POWER: usize = 7;
//const L1_N_FILTERS: usize = 128;
//const L1_SIZE: usize = L1_N_FILTERS / 64;
//const L2_SIZE: usize = 2;
//const L2_N_FILTERS: usize = 128;

cov_mask!(l1_cov_mask, 13, 13 * 64);

//fc_features!(l1_features, 13, 10);
////cov_filter!(l1_cov_filter, 13, L1_N_FILTERS);
//vector_fused_xor_popcount_threshold_and_bitpack!(l1_fxoptbp, 13, L1_SIZE);

//fc_features!(l2_features, L1_SIZE, 10);
//cov_filter!(l2_cov_filter, L1_SIZE, L2_N_FILTERS);
//vector_fused_xor_popcount_threshold_and_bitpack!(l2_fxoptbp, L1_SIZE, L2_SIZE);

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let examples: Vec<&(usize, [u64; 13])> = examples.iter().collect();
    //print_acc!(13, 10, examples, examples);
    println!("data loaded",);

    let mask = l1_cov_mask(&images, 400);
    for val in mask.iter() {
        print!("{:?}, ", val);
    }
    //let features_vec = l1_features(&examples, L1_POWER);
    ////let features_array = vec2array!(features_vec, L1_N_FILTERS, (0u32, [0u64; 13]));
    ////let parameters = l1_cov_filter(&images, &features_array, 128);
    ////println!("{:?}", parameters.len());

    //let new_images: Vec<[u64; L1_SIZE]> = images.par_iter().map(|image| l1_fxoptbp(&image, &features_vec)).collect();
    //let examples: Vec<(usize, [u64; L1_SIZE])> = labels.iter().zip(new_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    //let examples: Vec<&(usize, [u64; L1_SIZE])> = examples.iter().collect();
    //print_acc!(L1_SIZE, 10, examples, examples);

    //let features_vec = l2_features(&examples, 7);
    ////let features_array = vec2array!(features_vec, L2_N_FILTERS, (0u32, [0u64; L1_SIZE]));
    ////let parameters = l2_cov_filter(&new_images, &features_array, 128);
    //let new_images: Vec<[u64; L2_SIZE]> = new_images.par_iter().map(|image| l2_fxoptbp(&image, &features_vec)).collect();
    //let examples: Vec<(usize, [u64; L2_SIZE])> = labels.iter().zip(new_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    //let examples: Vec<&(usize, [u64; L2_SIZE])> = examples.iter().collect();
    //print_acc!(L2_SIZE, 10, examples, examples);
}
