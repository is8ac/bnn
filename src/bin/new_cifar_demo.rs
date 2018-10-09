extern crate bitnn;
extern crate rayon;
extern crate time;

use bitnn::datasets::cifar;
use rayon::prelude::*;
use time::PreciseTime;

use bitnn::layers;
use bitnn::layers::{pack_3x3, pixelmap, unary, Layer2d, Patch};

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images = Vec::with_capacity(10000 * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, 10000);
        images.append(&mut batch);
    }
    images
}

fn split_by_label<T: Copy>(examples: &Vec<(usize, T)>, len: usize) -> Vec<Vec<T>> {
    let mut by_label: Vec<Vec<T>> = (0..len).map(|_| Vec::new()).collect();
    for (label, example) in examples {
        by_label[*label].push(*example);
    }
    let _: Vec<_> = by_label.iter_mut().map(|x| x.shrink_to_fit()).collect();
    by_label
}

fn avg_bit_sums(len: usize, counts: &Vec<u32>) -> Vec<f64> {
    counts.iter().map(|&count| count as f64 / len as f64).collect()
}

// split_labels_set_by_filter takes examples and a split func. It returns the
// examples is a vec of shards of labels of examples of patches.
// It returns the same data but with double the shards and with each Vec of patches (very approximately) half the length.
// split_fn is used to split each Vec of patches between two shards.
fn split_labels_set_by_distance<T: Copy + Send + Sync + Patch>(
    examples: &Vec<Vec<Vec<T>>>,
    base_point: &T,
    threshold: u32,
) -> Vec<Vec<Vec<T>>> {
    examples
        .par_iter()
        .map(|by_label| {
            let pair: Vec<Vec<Vec<T>>> = vec![
                by_label
                    .par_iter()
                    .map(|label_examples| {
                        label_examples
                            .iter()
                            .filter(|x| base_point.hamming_distance(x) > threshold)
                            .cloned()
                            .collect()
                    })
                    .collect(),
                by_label
                    .par_iter()
                    .map(|label_examples| {
                        label_examples
                            .iter()
                            .filter(|x| base_point.hamming_distance(x) <= threshold)
                            .cloned()
                            .collect()
                    })
                    .collect(),
            ];
            pair
        })
        .flatten()
        .filter(|pair: &Vec<Vec<T>>| {
            let sum_len: usize = pair.iter().map(|x| x.len()).sum();
            sum_len > 100
        })
        .collect()
}

fn gen_basepoint<T: Patch + Sync + std::fmt::Binary>(shards: &Vec<Vec<Vec<T>>>, label: usize) -> (T, u32) {
    let mut sum_bits: Vec<Vec<(usize, Vec<u32>)>> = shards
        .iter()
        .map(|labels| {
            labels
                .iter()
                .map(|label_patches| (label_patches.len(), layers::count_bits(&label_patches)))
                .collect()
        })
        .collect();

    let grads: Vec<f64> = sum_bits
        .iter_mut()
        .map(|shard| {
            let target = shard.remove(label);
            let other = shard
                .iter()
                .fold((0usize, vec![0u32; T::bit_len()]), |(a_len, a_vals), (b_len, b_vals)| {
                    (a_len + b_len, a_vals.iter().zip(b_vals.iter()).map(|(x, y)| x + y).collect())
                });
            (target, other)
        })
        .filter(|((target_len, _), (other_len, _))| (*target_len > 0) & (*other_len > 0))
        .map(|((target_len, target_sums), (other_len, other_sums))| {
            let other_avg = avg_bit_sums(other_len, &other_sums);
            let target_avg = avg_bit_sums(target_len, &target_sums);
            other_avg
                .iter()
                .zip(target_avg.iter())
                .map(|(a, b)| (a - b) * (target_len.min(other_len) as f64))
                .collect()
        })
        .fold(vec![0f64; T::bit_len()], |acc, grads: Vec<f64>| {
            acc.iter().zip(grads.iter()).map(|(a, b)| a + b).collect()
        });
    let bits: Vec<bool> = grads.iter().map(|x| *x > 0f64).collect();

    let base_point = T::bitpack(&bits);
    let mut bit_distances: Vec<u32> = shards.iter().flatten().flatten().map(|y| y.hamming_distance(&base_point)).collect();
    bit_distances.par_sort();
    let threshold = bit_distances[bit_distances.len() / 2];
    (base_point, threshold)
}

fn main() {
    let start = PreciseTime::now();
    let images = load_data();
    let u14_packed_images: Vec<(usize, [[u16; 32]; 32])> = images
        .par_iter()
        .map(|(label, image)| (*label, pixelmap::pm_32(&image, &unary::rgb_to_u14)))
        .collect();
    let by_label = split_by_label(&u14_packed_images, 10);
    let mut shards: Vec<Vec<Vec<u128>>> = vec![
        by_label
            .par_iter()
            .map(|imgs| {
                imgs.iter()
                    .map(|image| image.to_3x3_patches())
                    .flatten()
                    .map(|pixels| pack_3x3::p14(pixels))
                    .collect()
            })
            .collect(),
    ];
    println!("load + data preparation time: {}", start.to(PreciseTime::now()));

    let mut features_vec = vec![];

    let seed = 0b11111111111111111111111111111111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000000000u128;
    let mut bit_distances: Vec<u32> = shards.iter().flatten().flatten().map(|y| y.hamming_distance(&seed)).collect();
    bit_distances.par_sort();
    let threshold = bit_distances[bit_distances.len() / 2];
    features_vec.push((seed, threshold));
    shards = split_labels_set_by_distance(&shards, &seed, threshold);
    println!("{:0128b} {:} \t {:?}", seed, threshold, shards.len());

    for i in 0..3 {
        for class in 0..10 {
            let (base_point, threshold) = gen_basepoint(&shards, class);
            features_vec.push((base_point, threshold));
            shards = split_labels_set_by_distance(&shards, &base_point, threshold);
            println!("{:0128b} {:} \t {:?} \t {:?}", base_point, threshold, shards.len(), class);
            let lens: Vec<Vec<usize>> = shards.iter().map(|x| x.iter().map(|y| y.len()).collect()).collect();
            //println!("{:?}", lens);
        }
    }
    let mut features = [(0u128, 0u32); 32];
    for (i, feature) in features_vec.iter().enumerate() {
        features[i] = *feature;
    }
    println!("{:?}", features);
}
