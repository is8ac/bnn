#![feature(const_generics)]

use bincode::{deserialize_from, serialize_into};
use bitnn::bits::{BitLen, BitMul, FlipBit, GetBit, HammingDistance};
use bitnn::count::{Counters, IncrementCounters};
use bitnn::datasets::cifar;
use bitnn::image2d::{AvgPool, Concat2D, Conv2D, ExtractPixels, Normalize2D, PixelMap2D};
use bitnn::layer::{SupervisedLayer, UnsupervisedLayer};
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs::{create_dir_all, File};
use std::io::BufWriter;
use std::path::Path;
use time::PreciseTime;

trait Unary<Bits> {
    fn unary_encode(&self) -> Bits;
}

impl Unary<u32> for [u8; 3] {
    fn unary_encode(&self) -> u32 {
        let mut bits = 0b0u32;
        for c in 0..3 {
            bits |= (!0b0u32 << (self[c] / 23)) << (c * 10);
            bits &= !0b0u32 >> (32 - ((c + 1) * 10));
        }
        !bits
    }
}

trait NaiveBayes<const K: usize>
where
    Self: Sized,
{
    fn naive_bayes_classify(examples: &Vec<(Self, usize)>) -> [(Self, u32); K];
}

impl<T: HammingDistance + IncrementCounters + Sync + Send + Default + Copy, const K: usize>
    NaiveBayes<{ K }> for T
where
    [(usize, T::BitCounterType); K]: Default + Sync + Send,
    T::BitCounterType: Counters + Send + Sync + Default,
    [(T, u32); K]: Default,
{
    fn naive_bayes_classify(examples: &Vec<(T, usize)>) -> [(T, u32); K] {
        let counters: Vec<(usize, T::BitCounterType)> = examples
            .par_iter()
            .fold(
                || {
                    (0..K)
                        .map(|_| <(usize, T::BitCounterType)>::default())
                        .collect::<Vec<_>>()
                },
                |mut acc, (example, class)| {
                    acc[*class].0 += 1;
                    example.increment_counters(&mut acc[*class].1);
                    acc
                },
            )
            .reduce(
                || {
                    (0..K)
                        .map(|_| <(usize, T::BitCounterType)>::default())
                        .collect::<Vec<_>>()
                },
                |mut a, b| {
                    for c in 0..K {
                        a[c].0 += b[c].0;
                        (a[c].1).elementwise_add(&b[c].1);
                    }
                    a
                },
            );
        let (sum_n, sum_counters) =
            counters
                .iter()
                .fold((0usize, T::BitCounterType::default()), |mut a, b| {
                    a.0 += b.0;
                    (a.1).elementwise_add(&b.1);
                    a
                });
        let filters: Vec<T> = counters
            .par_iter()
            .map(|(n, bit_counter)| T::compare_and_bitpack(&sum_counters, sum_n, bit_counter, *n))
            .collect();

        let dist: Vec<u64> = examples
            .par_iter()
            .fold(
                || vec![0u64; 10],
                |mut acc, (image, _)| {
                    for c in 0..10 {
                        acc[c] += filters[c].hamming_distance(image) as u64;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0u64; 10],
                |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect(),
            );
        let max_avg_act: f64 = *dist.iter().max().unwrap() as f64 / examples.len() as f64;
        let biases: Vec<u32> = dist
            .iter()
            .map(|&x| (max_avg_act - (x as f64 / examples.len() as f64)) as u32)
            .collect();

        let mut target = <[(T, u32); K]>::default();
        for c in 0..K {
            target[c] = (filters[c], biases[c]);
        }
        target
    }
}

fn print_patch<const X: usize, const Y: usize>(patch: &[[u32; Y]; X]) {
    for x in 0..X {
        for y in 0..Y {
            print!("{:032b} | ", patch[x][y]);
        }
        print!("\n");
    }
    println!("-----------",);
}

const N_EXAMPLES: usize = 50_00;
const B0_CHANS: usize = 1;
const B1_CHANS: usize = 2;
const B2_CHANS: usize = 3;
const B3_CHANS: usize = 4;

fn main() {
    let start = PreciseTime::now();
    let mut rng = Hc128Rng::seed_from_u64(0);

    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(24))
        //.num_threads(4)
        .build_global()
        .unwrap();

    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    let examples: Vec<([[[u8; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);
    let labels: Vec<usize> = examples.par_iter().map(|(_, label)| *label).collect();
    let raw_images: Vec<[[[u8; 3]; 32]; 32]> =
        examples.par_iter().map(|(image, _)| *image).collect();

    let params_path = Path::new("params/cluster_split_1");
    create_dir_all(&params_path).unwrap();
    println!("init time: {}", start.to(PreciseTime::now()));

    let normalized_images: Vec<[[[[u32; 3]; 3]; 32]; 32]> = raw_images
        .par_iter()
        .map(|image| {
            let image: [[[[[u8; 3]; 3]; 3]; 32]; 32] = image.conv2d();
            image.map_2d(|patch| patch.normalize_2d().map_2d(|pixel| pixel.unary_encode()))
        })
        .collect();

    let (images, weights) = <[[([[u32; 3]; 3], u32); 32]; B0_CHANS]>::unsupervised_cluster(
        &mut rng,
        &normalized_images,
        &params_path.join("b0_l0.prms"),
    );

    //let conved_images: Vec<[[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32]> = images.par_iter().map(|image| image.conv2d()).collect();
    //let (images, weights) = <[[([[[u32; B0_CHANS]; 3]; 3], u32); 32]; B0_CHANS]>::unsupervised_cluster(
    //    &mut rng,
    //    &conved_images,
    //    &params_path.join("b0_l1.prms"),
    //);
    let conved_images: Vec<[[[[[u32; B0_CHANS]; 2]; 2]; 16]; 16]> = images
        .par_iter()
        .map(|image| {
            <[[[u32; B0_CHANS]; 32]; 32] as Conv2D<[[[[[u32; B0_CHANS]; 2]; 2]; 16]; 16]>>::conv2d(
                image,
            )
        })
        .collect();

    let raw_images: Vec<[[[u8; 3]; 16]; 16]> = raw_images
        .par_iter()
        .map(|image| image.avg_pool())
        .collect();
    let normalized_images: Vec<[[[[u32; 3]; 3]; 16]; 16]> = raw_images
        .par_iter()
        .map(|image| {
            let image: [[[[[u8; 3]; 3]; 3]; 16]; 16] = image.conv2d();
            image.map_2d(|patch| patch.normalize_2d().map_2d(|pixel| pixel.unary_encode()))
        })
        .collect();
    let zipped_images: Vec<_> = conved_images
        .par_iter()
        .zip(normalized_images.par_iter())
        .map(|(a, b)| <[[([[[u32; B0_CHANS]; 2]; 2], [[u32; 3]; 3]); 16]; 16]>::concat_2d(a, b))
        .collect();

    let (images, weights) = <[[(_, u32); 32]; B1_CHANS]>::unsupervised_cluster(
        &mut rng,
        &zipped_images,
        &params_path.join("b1_l0.prms"),
    );
    let conved_images: Vec<[[[[[u32; B1_CHANS]; 3]; 3]; 16]; 16]> =
        images.par_iter().map(|image| image.conv2d()).collect();

    let (images, weights) =
        <[[([[[u32; B1_CHANS]; 3]; 3], u32); 32]; B1_CHANS]>::unsupervised_cluster(
            &mut rng,
            &conved_images,
            &params_path.join("b1_l1.prms"),
        );

    let conved_images: Vec<[[[[[u32; B1_CHANS]; 2]; 2]; 8]; 8]> = images
        .par_iter()
        .map(|image| {
            <[[[u32; B1_CHANS]; 16]; 16] as Conv2D<[[[[[u32; B1_CHANS]; 2]; 2]; 8]; 8]>>::conv2d(
                image,
            )
        })
        .collect();

    let raw_images: Vec<[[[u8; 3]; 8]; 8]> = raw_images
        .par_iter()
        .map(|image| image.avg_pool())
        .collect();

    let normalized_images: Vec<[[[[u32; 3]; 3]; 8]; 8]> = raw_images
        .par_iter()
        .map(|image| {
            let image: [[[[[u8; 3]; 3]; 3]; 8]; 8] = image.conv2d();
            image.map_2d(|patch| patch.normalize_2d().map_2d(|pixel| pixel.unary_encode()))
        })
        .collect();

    let zipped_images: Vec<_> = conved_images
        .par_iter()
        .zip(normalized_images.par_iter())
        .map(|(a, b)| <[[([[[u32; B1_CHANS]; 2]; 2], [[u32; 3]; 3]); 8]; 8]>::concat_2d(a, b))
        .collect();

    let (images, weights) = <[[(_, u32); 32]; B2_CHANS]>::unsupervised_cluster(
        &mut rng,
        &zipped_images,
        &params_path.join("b2_l0.prms"),
    );

    let conved_images: Vec<[[[[[u32; B2_CHANS]; 3]; 3]; 8]; 8]> =
        images.par_iter().map(|image| image.conv2d()).collect();
    let (images, weights) =
        <[[([[[u32; B2_CHANS]; 3]; 3], u32); 32]; B2_CHANS]>::unsupervised_cluster(
            &mut rng,
            &conved_images,
            &params_path.join("b2_l1.prms"),
        );

    let conved_images: Vec<[[[[[u32; B2_CHANS]; 2]; 2]; 4]; 4]> = images
        .par_iter()
        .map(|image| {
            <[[[u32; B2_CHANS]; 8]; 8] as Conv2D<[[[[[u32; B2_CHANS]; 2]; 2]; 4]; 4]>>::conv2d(
                image,
            )
        })
        .collect();

    let raw_images: Vec<[[[u8; 3]; 4]; 4]> = raw_images
        .par_iter()
        .map(|image| image.avg_pool())
        .collect();

    let normalized_images: Vec<[[[[u32; 3]; 3]; 4]; 4]> = raw_images
        .par_iter()
        .map(|image| {
            let image: [[[[[u8; 3]; 3]; 3]; 4]; 4] = image.conv2d();
            image.map_2d(|patch| patch.normalize_2d().map_2d(|pixel| pixel.unary_encode()))
        })
        .collect();

    let zipped_images: Vec<_> = conved_images
        .par_iter()
        .zip(normalized_images.par_iter())
        .map(|(a, b)| <[[([[[u32; B2_CHANS]; 2]; 2], [[u32; 3]; 3]); 4]; 4]>::concat_2d(a, b))
        .collect();

    let (images, weights) = <[[(_, u32); 32]; B3_CHANS]>::unsupervised_cluster(
        &mut rng,
        &zipped_images,
        &params_path.join("b3_l0.prms"),
    );

    let examples: Vec<(_, usize)> = images
        .par_iter()
        .cloned()
        .zip(labels.par_iter().cloned())
        .collect();
    let mut fc_layer: [([[[u32; B3_CHANS]; 4]; 4], u32); 10] =
        <[[[u32; B3_CHANS]; 4]; 4]>::naive_bayes_classify(&examples);

    let n_correct: u64 = examples
        .par_iter()
        .map(|(image, class)| {
            let actual_class: usize = fc_layer
                .iter()
                .enumerate()
                .max_by_key(|(i, (filter, bias))| filter.hamming_distance(image) + bias)
                .unwrap()
                .0;
            (actual_class == *class) as u64
        })
        .sum();
    dbg!(n_correct as f64 / examples.len() as f64);
    //0.2984
    //0.29022
    //0.2692
    //0.2794
    //0.034
    //0.2396
    //0.2766
    //26636
}
