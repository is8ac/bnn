#![feature(const_generics)]

use bincode::{deserialize_from, serialize_into};
use bitnn::bits::{BitLen, BitMul, FlipBit, GetBit, HammingDistance};
use bitnn::count::{Counters, IncrementCounters};
use bitnn::datasets::cifar;
use bitnn::image2d::{AvgPool, Concat2D, Conv2D, ExtractPixels, Normalize2D, PixelMap2D};
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

trait Lloyds
where
    Self: IncrementCounters + Sized,
{
    fn update_centers(example: &Vec<Self>, centers: &Vec<Self>) -> Vec<Self>;
    fn lloyds<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Self>, k: usize) -> Vec<(Self, u32)>;
}

impl<T: IncrementCounters + Send + Sync + Copy + HammingDistance + BitLen + Eq> Lloyds for T
where
    <Self as IncrementCounters>::BitCounterType: Default + Send + Sync + Counters + std::fmt::Debug,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    fn update_centers(examples: &Vec<Self>, centers: &Vec<Self>) -> Vec<Self> {
        let counters: Vec<(usize, Self::BitCounterType)> = examples
            .par_iter()
            .fold(
                || {
                    centers
                        .iter()
                        .map(|_| {
                            (
                                0usize,
                                <Self as IncrementCounters>::BitCounterType::default(),
                            )
                        })
                        .collect()
                },
                |mut acc: Vec<(usize, Self::BitCounterType)>, example| {
                    let cell_index = centers
                        .iter()
                        .map(|center| center.hamming_distance(example))
                        .enumerate()
                        .min_by_key(|(_, hd)| *hd)
                        .unwrap()
                        .0;
                    acc[cell_index].0 += 1;
                    example.increment_counters(&mut acc[cell_index].1);
                    acc
                },
            )
            .reduce(
                || {
                    centers
                        .iter()
                        .map(|_| {
                            (
                                0usize,
                                <Self as IncrementCounters>::BitCounterType::default(),
                            )
                        })
                        .collect()
                },
                |mut a, b| {
                    a.iter_mut()
                        .zip(b.iter())
                        .map(|(a, b)| {
                            a.0 += b.0;
                            (a.1).elementwise_add(&b.1);
                        })
                        .count();
                    a
                },
            );
        counters
            .iter()
            .map(|(n_examples, counters)| {
                <Self>::threshold_and_bitpack(&counters, *n_examples as u32 / 2)
            })
            .collect()
    }
    fn lloyds<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Self>, k: usize) -> Vec<(Self, u32)> {
        let mut centroids: Vec<Self> = (0..k).map(|_| rng.gen()).collect();
        for i in 0..1000 {
            dbg!(i);
            let new_centroids = <Self>::update_centers(examples, &centroids);
            if new_centroids == centroids {
                break;
            }
            centroids = new_centroids;
        }
        centroids
            .iter()
            .map(|centroid| {
                let mut activations: Vec<u32> = examples
                    .par_iter()
                    .map(|example| example.hamming_distance(centroid))
                    .collect();
                activations.par_sort();
                (*centroid, activations[examples.len() / 2])
                //(*centroid, Self::BIT_LEN as u32 / 2)
            })
            .collect()
    }
}

trait Layer<InputImage, OutputImage> {
    fn cluster_features<RNG: rand::Rng>(
        rng: &mut RNG,
        examples: &Vec<InputImage>,
        path: &Path,
    ) -> (Vec<OutputImage>, Self);
}

impl<
        P: Send + Sync + HammingDistance + Lloyds + Copy,
        InputImage: Sync + PixelMap2D<P, [u32; C], OutputImage = OutputImage> + ExtractPixels<P>,
        OutputImage: Sync + Send,
        const C: usize,
    > Layer<InputImage, OutputImage> for [[(P, u32); 32]; C]
where
    for<'de> Self: serde::Deserialize<'de>,
    Self: BitMul<P, [u32; C]> + Default + serde::Serialize,
{
    fn cluster_features<RNG: rand::Rng>(
        rng: &mut RNG,
        images: &Vec<InputImage>,
        path: &Path,
    ) -> (Vec<OutputImage>, Self) {
        let weights: Self = File::open(&path)
            .map(|f| deserialize_from(f).unwrap())
            .ok()
            .unwrap_or_else(|| {
                println!("no params found, training.");
                let examples: Vec<P> = images
                    .par_iter()
                    .fold(
                        || vec![],
                        |mut pixels, image| {
                            image.extract_pixels(&mut pixels);
                            pixels
                        },
                    )
                    .reduce(
                        || vec![],
                        |mut a, mut b| {
                            a.append(&mut b);
                            a
                        },
                    );
                dbg!(examples.len());
                let clusters = <P>::lloyds(rng, &examples, C * 32);

                let mut weights = Self::default();
                for (i, filter) in clusters.iter().enumerate() {
                    weights[i / 32][i % 32] = *filter;
                }
                let mut f = BufWriter::new(File::create(path).unwrap());
                serialize_into(&mut f, &weights).unwrap();
                weights
            });
        println!("got params");
        let start = PreciseTime::now();
        let images: Vec<OutputImage> = images
            .par_iter()
            .map(|image| image.map_2d(|x| weights.bit_mul(x)))
            .collect();
        println!("time: {}", start.to(PreciseTime::now()));
        // PT2.04
        (images, weights)
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

fn is_correct<const C: usize>(input: &[u32; C], true_class: usize) -> bool {
    let max_act: usize = input
        .iter()
        .enumerate()
        .max_by_key(|(i, act)| *act)
        .unwrap()
        .0;
    max_act == true_class
}

trait Optimize<const C: usize>
where
    Self: Sized,
{
    fn optimize(weights: &mut [(Self, u32); C], examples: &Vec<(Self, usize)>);
}

impl<T: FlipBit + BitLen + Sized + Sync + HammingDistance + GetBit, const C: usize> Optimize<{ C }>
    for T
where
    [u32; C]: Default,
{
    fn optimize(weights: &mut [(Self, u32); C], examples: &Vec<(Self, usize)>) {
        let mut cache: Vec<[u32; C]> = examples
            .par_iter()
            .map(|example| {
                let mut activations = <[u32; C]>::default();
                for c in 0..C {
                    activations[c] = weights[c].0.hamming_distance(&example.0) + weights[c].1;
                }
                activations
            })
            .collect();
        let mut n_correct: u64 = cache
            .par_iter_mut()
            .zip(examples.par_iter())
            .map(|(activations, example)| is_correct(activations, example.1) as u64)
            .sum();
        dbg!(n_correct as f64 / examples.len() as f64);
        for c in 0..C {
            dbg!(c);
            for b in 0..T::BIT_LEN {
                //let bit = weights[c].0.bit(b);
                weights[c].0.flip_bit(b);
                let new_n_correct: u64 = cache
                    .par_iter_mut()
                    .zip(examples.par_iter())
                    .map(|(activations, example)| {
                        activations[c] = weights[c].0.hamming_distance(&example.0) + weights[c].1;
                        is_correct(activations, example.1) as u64
                    })
                    .sum();
                if new_n_correct > n_correct {
                    dbg!(n_correct as f64 / examples.len() as f64);
                    n_correct = new_n_correct;
                } else {
                    weights[c].0.flip_bit(b);
                }
            }
            cache
                .par_iter_mut()
                .zip(examples.par_iter())
                .map(|(activations, example)| {
                    activations[c] = weights[c].0.hamming_distance(&example.0) + weights[c].1;
                })
                .count();
        }
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

const N_EXAMPLES: usize = 50_000;
const B0_CHANS: usize = 2;
const B1_CHANS: usize = 4;
const B2_CHANS: usize = 8;
const B3_CHANS: usize = 16;

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

    let params_path = Path::new("params/lloyds_cluster_24");
    create_dir_all(&params_path).unwrap();
    println!("init time: {}", start.to(PreciseTime::now()));

    let normalized_images: Vec<[[[[u32; 3]; 3]; 32]; 32]> = raw_images
        .par_iter()
        .map(|image| {
            let image: [[[[[u8; 3]; 3]; 3]; 32]; 32] = image.conv2d();
            image.map_2d(|patch| patch.normalize_2d().map_2d(|pixel| pixel.unary_encode()))
        })
        .collect();

    let (images, weights) = <[[([[u32; 3]; 3], u32); 32]; B0_CHANS]>::cluster_features(
        &mut rng,
        &normalized_images,
        &params_path.join("b0_l0.prms"),
    );

    //let conved_images: Vec<[[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32]> = images.par_iter().map(|image| image.conv2d()).collect();
    //let (images, weights) = <[[([[[u32; B0_CHANS]; 3]; 3], u32); 32]; B0_CHANS]>::cluster_features(
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

    let (images, weights) = <[[(_, u32); 32]; B1_CHANS]>::cluster_features(
        &mut rng,
        &zipped_images,
        &params_path.join("b1_l0.prms"),
    );
    let conved_images: Vec<[[[[[u32; B1_CHANS]; 3]; 3]; 16]; 16]> =
        images.par_iter().map(|image| image.conv2d()).collect();

    let (images, weights) = <[[([[[u32; B1_CHANS]; 3]; 3], u32); 32]; B1_CHANS]>::cluster_features(
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

    let (images, weights) = <[[(_, u32); 32]; B2_CHANS]>::cluster_features(
        &mut rng,
        &zipped_images,
        &params_path.join("b2_l0.prms"),
    );

    let conved_images: Vec<[[[[[u32; B2_CHANS]; 3]; 3]; 8]; 8]> =
        images.par_iter().map(|image| image.conv2d()).collect();
    let (images, weights) = <[[([[[u32; B2_CHANS]; 3]; 3], u32); 32]; B2_CHANS]>::cluster_features(
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

    let (images, weights) = <[[(_, u32); 32]; B3_CHANS]>::cluster_features(
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

    Optimize::optimize(&mut fc_layer, &examples);
    Optimize::optimize(&mut fc_layer, &examples);
    Optimize::optimize(&mut fc_layer, &examples);
    Optimize::optimize(&mut fc_layer, &examples);
    Optimize::optimize(&mut fc_layer, &examples);

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
