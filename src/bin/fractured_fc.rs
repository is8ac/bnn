#[macro_use]
extern crate bitnn;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use rayon::prelude::*;
use time::PreciseTime;

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

macro_rules! fc_infer {
    ($in_size:expr, $weights:expr, $input:expr) => {{
        let mut sum = 0;
        for i in 0..$in_size {
            sum += ($weights[i] ^ $input[i]).count_ones();
        }
        sum
    }};
}

macro_rules! fc_features {
    ($name:ident, $in_size:expr, $n_labels:expr) => {
        fn $name(examples: &Vec<&(usize, [u64; $in_size])>, level: usize) -> Vec<[u64; $in_size]> {
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
            println!("{:?} {:?}", level, label_dist);
            let largest_label = label_dist.iter().enumerate().max_by_key(|(_, &count)| count).unwrap().0;
            let in_group: Vec<&[u64; $in_size]> = examples.par_iter().filter(|(label, _)| *label == largest_label).map(|(_, input)| input).collect();
            let out_group: Vec<&[u64; $in_size]> = examples.par_iter().filter(|(label, _)| *label != largest_label).map(|(_, input)| input).collect();
            let filter = split(&out_group, &in_group);
            if level <= 0 {
                return vec![filter];
            }
            let mut activations: Vec<u32> = examples.par_iter().map(|&(_, image)| fc_infer!($in_size, filter, image)).collect();
            activations.sort();
            let threshold = activations[examples.len() / 2];
            let over: Vec<&(usize, [u64; $in_size])> = examples
                .par_iter()
                .map(|&x| x)
                .filter(|&(_, image)| fc_infer!($in_size, filter, image) > threshold)
                .collect();
            let under: Vec<&(usize, [u64; $in_size])> = examples
                .par_iter()
                .map(|&x| x)
                .filter(|&(_, image)| fc_infer!($in_size, filter, image) <= threshold)
                .collect();
            let mut over_elems = $name(&over, level - 1);
            let mut under_elems = $name(&under, level - 1);
            over_elems.append(&mut under_elems);
            over_elems
        }
    };
}

const TRAIN_SIZE: usize = 60000;

fc_features!(features, 13, 10);

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let examples: Vec<&(usize, [u64; 13])> = examples.iter().collect();
    let weights = features(&examples, 8);
    println!("{:?}", weights.len());
}
