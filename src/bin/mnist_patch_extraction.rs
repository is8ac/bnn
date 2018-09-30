#[macro_use]
extern crate bitnn;
extern crate rayon;

use bitnn::datasets::mnist;
use rayon::prelude::*;

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
            vec![
                by_label
                    .par_iter()
                    .map(|label_examples| label_examples.iter().filter(|x| split_fn(data, x)).cloned().collect())
                    .collect(),
                by_label
                    .par_iter()
                    .map(|label_examples| label_examples.iter().filter(|x| !split_fn(data, x)).cloned().collect())
                    .collect(),
            ]
        }).flatten()
        .collect()
}

fn count_bits(examples: &Vec<Vec<Vec<u64>>>) -> Vec<Vec<(usize, [u32; 64])>> {
    examples
        .par_iter()
        .map(|shard| shard.par_iter().map(|patches| (patches.len(), u64_sum_bits(&patches))).collect())
        .collect()
}

map_2d!(mnist_u8_map, 28, 28);
fixed_dim_extract_3x3_patchs!(mnist_extract_u64_packed_patches, 28, 28);

const TRAIN_SIZE: usize = 6000;

fn split_fn((weights, filter, threshold): &(u64, u64, u32), patch: &u64) -> bool {
    ((patch ^ weights) & filter).count_ones() > *threshold
}

fn main() {
    let images = mnist::load_images(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let unary_images: Vec<_> = images.iter().map(|x| mnist_u8_map(x, &unary7bit)).collect();
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [[u8; 28]; 28])> = labels.iter().zip(unary_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let examples_sets = split_images_by_label(&examples);
    let patches_set: Vec<Vec<Vec<u64>>> = vec![examples_sets.par_iter().map(|x| mnist_extract_u64_packed_patches(&x, &patch_pack_7bit)).collect()];

    let mut shard_label_bit_sums = count_bits(&patches_set);

    let (target_len, target_bit_sums) = shard_label_bit_sums[0].remove(0);
    let (global_bit_len, global_bit_sums): (usize, [u32; 64]) = shard_label_bit_sums[0].par_iter().cloned().reduce(
        || (0, [0u32; 64]),
        |mut a, b| {
            a.0 += b.0;
            for i in 0..64 {
                a.1[i] += b.1[i];
            }
            a
        },
    );
    let global_avg_bits: Vec<f64> = global_bit_sums.iter().map(|&sum| sum as f64 / global_bit_len as f64).collect();
    let target_avg_bits: Vec<f64> = target_bit_sums.iter().map(|&sum| sum as f64 / target_len as f64).collect();
    //println!("{:?}", global_avg_bits);
    //println!("{:?}", target_avg_bits);

    let mut weights = 0u64;
    let mut filter = 0u64;
    let mask_thresh = 0.05;
    for i in 0..64 {
        let grad = global_avg_bits[i] - target_avg_bits[i];
        let sign = grad > 0f64;
        weights = weights | ((sign as u64) << i);
        let magn = grad.abs() > mask_thresh;
        filter = filter | ((magn as u64) << i);
    }
    println!("{:064b}", weights);
    println!("{:064b}", filter);

    let mut activations: Vec<u32> = patches_set.par_iter().flatten().flatten().map(|patch| (patch ^ weights).count_ones()).collect();
    activations.sort();
    let threshold = activations[activations.len() / 2];
    println!("{:?}", threshold);

    let patch_shards = split_labels_set_by_filter(&patches_set, &(weights, filter, threshold), split_fn);
    println!("{:?}", patch_shards.len());
}
