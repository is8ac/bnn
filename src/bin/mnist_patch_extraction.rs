extern crate bitnn;
extern crate rayon;

use rayon::prelude::*;
use bitnn::datasets::mnist;
use std::collections::HashMap;

fn unary7bit(input: u8) -> u8 {
    !(255 << (input / 36))
}

fn patch_pack_7bit(pixels: [u8; 9]) -> u64 {
    let mut word = 0b0u64;
    for i in 0..9 {
        word = word | ((pixels[i] as u64) << (i * 7))
    }
    word
}

fn u64_sum_bits(examples: &Vec<u64>) -> [u32; 64] {
    examples
        .par_iter()
        .fold(
            || [0u32; 64],
            |mut counts, example| {
                for b in 0..64 {
                    counts[b] += ((example >> b) & 0b1u64) as u32;
                }
                counts
            },
        ).reduce(
            || [0u32; 64],
            |mut a, b| {
                for i in 0..64 {
                    a[i] += b[i];
                }
                a
            },
        )
}

macro_rules! map_2d {
    ($name:ident, $x_size:expr, $y_size:expr) => {
        fn $name<I: Copy, O: Copy + Default>(input: &[[I; $y_size]; $x_size], map_fn: &Fn(I) -> O) -> [[O; $y_size]; $x_size] {
            let mut output = [[O::default(); $y_size]; $x_size];
            for x in 0..$x_size {
                for y in 0..$y_size {
                    output[x][y] = map_fn(input[x][y]);
                }
            }
            output
        }
    };
}

macro_rules! fixed_dim_extract_3x3_patchs {
    ($name:ident, $x_size:expr, $y_size:expr) => {
        fn $name<P: Copy, O>(images: &Vec<[[P; $y_size]; $x_size]>, patch_fn: &Fn([P; 3 * 3]) -> O) -> Vec<O> {
            let mut patches = Vec::with_capacity(images.len() * ($x_size - 2) * ($y_size - 2));
            for image in images {
                for x in 0..$x_size - 2 {
                    for y in 0..$y_size - 2 {
                        let patch = patch_fn([
                            image[x + 0][y + 0],
                            image[x + 1][y + 0],
                            image[x + 2][y + 0],
                            image[x + 0][y + 1],
                            image[x + 1][y + 1],
                            image[x + 2][y + 1],
                            image[x + 0][y + 2],
                            image[x + 1][y + 2],
                            image[x + 2][y + 2],
                        ]);
                        patches.push(patch);
                    }
                }
            }
            patches
        }
    };
}

fn split_images_by_label(examples: &Vec<(usize, [[u8; 28]; 28])>) -> Vec<Vec<[[u8; 28]; 28]>> {
    let mut split_images: Vec<_> = (0..10).map(|_| Vec::new()).collect();
    for (label, image) in examples {
        split_images[*label].push(*image);
    }
    split_images
}

// split_labels_set_by_filter takes examples and a split func. It returns the
// examples is a vec of shards of labels of examples of patches.
// It returns the same data but with double the shards and with each Vec of patches (very approximately) half the length.
// split_fn is used to split each Vec of patches between two shards.
fn split_labels_set_by_filter<T: Copy + Send + Sync, D: Send + Sync>(examples: &Vec<Vec<Vec<T>>>, data: &D, split_fn: fn(&D, &T) -> bool) -> Vec<Vec<Vec<T>>> {
    examples
        .par_iter()
        .map(|by_label| {
            let pair: Vec<Vec<Vec<T>>> = vec![
                by_label
                    .par_iter()
                    .map(|label_examples| label_examples.iter().filter(|x| split_fn(data, x)).cloned().collect())
                    .collect(),
                by_label
                    .par_iter()
                    .map(|label_examples| label_examples.iter().filter(|x| !split_fn(data, x)).cloned().collect())
                    .collect(),
            ];
            pair
        }).flatten()
        .filter(|pair: &Vec<Vec<T>>| {
            let sum_len: usize = pair.iter().map(|x| x.len()).sum();
            sum_len > 0
        }).collect()
}

fn count_bits(examples: &Vec<Vec<Vec<u64>>>) -> Vec<Vec<(usize, [u32; 64])>> {
    examples
        .par_iter()
        .map(|shard| shard.par_iter().map(|patches| (patches.len(), u64_sum_bits(&patches))).collect())
        .collect()
}

fn avg_bits(bit_sums: &Vec<(usize, [u32; 64])>) -> Vec<[f64; 64]> {
    bit_sums
        .par_iter()
        .map(|(len, counts)| {
            let mut avg = [0f64; 64];
            let float_len = *len as f64;
            if float_len == 0f64 {
                avg
            } else {
                for i in 0..64 {
                    avg[i] = counts[i] as f64 / float_len;
                }
                avg
            }
        }).collect()
}

fn remove_index_from_2d_vec(matrix: &mut HashMap<usize, HashMap<usize, f64>>, index: usize) {
    matrix.remove(&index);
    for (_, map) in matrix {
        map.remove(&index);
    }
}

//for _i in 0..$n_features - target_len {
//    // we find the index of the largest average avg covariance.
//    let (largest_index, _val): (usize, f64) = matrix
//        .iter()
//        .map(|(&index, vals)| {
//            let sum: f64 = vals.iter().map(|(_, &val)| val).sum();
//            (index, sum / $n_features as f64)
//        }).max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//        .unwrap();
//    // and remove it from both dimentions of the matrix.
//    remove_index_from_2d_vec(&mut matrix, largest_index);
//}
//matrix.iter().map(|(&index, _)| parameters[index]).collect()

//fn ternary_similarly_matrix(weights: &Vec<(u64, u64)>, target_count: usize) -> Vec<usize> {
//    let mut matrix: HashMap<usize, HashMap<usize, f64>> =
//}

map_2d!(mnist_u8_map, 28, 28);
fixed_dim_extract_3x3_patchs!(mnist_extract_u64_packed_patches, 28, 28);

const TRAIN_SIZE: usize = 60000;

fn split_fn((weights, filter, threshold): &(u64, u64, u32), patch: &u64) -> bool {
    (patch ^ weights).count_ones() > *threshold
}

fn main() {
    let images = mnist::load_images(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let unary_images: Vec<_> = images.iter().map(|x| mnist_u8_map(x, &unary7bit)).collect();
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [[u8; 28]; 28])> = labels.iter().zip(unary_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let examples_sets = split_images_by_label(&examples);
    let mut patch_shards: Vec<Vec<Vec<u64>>> = vec![examples_sets.par_iter().map(|x| mnist_extract_u64_packed_patches(&x, &patch_pack_7bit)).collect()];
    let lens: Vec<Vec<usize>> = patch_shards.iter().map(|x| x.iter().map(|y| y.len()).collect()).collect();
    println!("shards: {:?}", lens);

    let mut params = vec![];

    let filter = !0b0u64;
    let weights = patch_pack_7bit([
        !0b0, 0b0, 0b0,
        !0b0, 0b0, 0b0,
        !0b0, 0b0, 0b0,
        ]);
    let mut activations: Vec<u32> = patch_shards.par_iter().flatten().flatten().map(|patch| (patch ^ weights).count_ones()).collect();
    activations.sort();
    let threshold = activations[activations.len() / 2];
    println!("{:064b} {:}", weights, threshold);

    params.push((weights, filter, threshold));
    patch_shards = split_labels_set_by_filter::<u64, _>(&patch_shards, &(weights, filter, threshold), split_fn);
    println!("n shards: {:?}", patch_shards.len());

    for i in 0..3 {
        for label in 0..10 {
            let mut shard_label_bit_sums = count_bits(&patch_shards);
            //let label = 0;
            let target_bit_sums: Vec<(usize, [u32; 64])> = shard_label_bit_sums.iter_mut().map(|x| x.remove(label)).collect();
            let global_bit_sums: Vec<(usize, [u32; 64])> = shard_label_bit_sums
                .iter()
                .map(|x| {
                    x.par_iter().cloned().reduce(
                        || (0, [0u32; 64]),
                        |mut a, b| {
                            a.0 += b.0;
                            for i in 0..64 {
                                a.1[i] += b.1[i];
                            }
                            a
                        },
                    )
                }).collect();

            let global_avg_bits: Vec<[f64; 64]> = avg_bits(&global_bit_sums);
            let target_avg_bits: Vec<[f64; 64]> = avg_bits(&target_bit_sums);

            let grads: Vec<f64> = global_avg_bits
                .iter()
                .zip(target_avg_bits.iter())
                .map(|(global, target)| global.iter().zip(target.iter()).map(|(g, t)| g - t).collect()) // for each bit, subtract the proportion in the target patches from the proportion in the global patches.
                .fold(vec![0f64; 64], |acc, grads: Vec<f64>| acc.iter().zip(grads.iter()).map(|(acc, grad)| acc + grad).collect()); // for all the shards, sum the grads of the corresponding bits.

            let mut weights = 0u64;
            let mut filter = 0u64;
            let mask_thresh = 0.000;
            // now we can pack the grads back into a u64.
            for (i, &grad) in grads.iter().enumerate() {
                let sign = grad > 0f64;
                weights = weights | ((sign as u64) << i);
                let magn = grad.abs() > mask_thresh;
                filter = filter | ((magn as u64) << i);
            }

            let mut activations: Vec<u32> = patch_shards.par_iter().flatten().flatten().map(|patch| (patch ^ weights).count_ones()).collect();
            activations.sort();
            let threshold = activations[activations.len() / 2];

            params.push((weights, filter, threshold));
            patch_shards = split_labels_set_by_filter::<u64, _>(&patch_shards, &(weights, filter, threshold), split_fn);
            //println!("n shards: {:?}", patch_shards.len());
            println!("{:064b} {:}\t{:}", weights, threshold, patch_shards.len());
            let lens: Vec<Vec<usize>> = patch_shards.iter().map(|x| x.iter().map(|y| y.len()).collect()).collect();
            //println!("shards: {:?}", lens);
        }
    }
    //for (weights, _, threshold) in &params {
    //    println!("{:064b} {:}", weights, threshold);
    //}
}
