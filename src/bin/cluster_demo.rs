#![feature(const_generics)]

use bitnn::datasets::cifar;
use bitnn::layers::SaveLoad;
//use bitnn::ExtractPatches;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;
use time::PreciseTime;

pub trait Counters {
    fn elementwise_add(&mut self, other: &Self);
}

impl Counters for u32 {
    fn elementwise_add(&mut self, other: &u32) {
        *self += other;
    }
}

impl<T: Counters, const L: usize> Counters for [T; L] {
    fn elementwise_add(&mut self, other: &[T; L]) {
        for i in 0..L {
            self[i].elementwise_add(&other[i]);
        }
    }
}

pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
}

impl<T: HammingDistance, const L: usize> HammingDistance for [T; L] {
    fn hamming_distance(&self, other: &[T; L]) -> u32 {
        let mut distance = 0u32;
        for i in 0..L {
            distance += self[i].hamming_distance(&other[i]);
        }
        distance
    }
}

trait IncrementCounters {
    type BitCounterType;
    fn increment_counters(&self, counters: &mut Self::BitCounterType);
    fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self;
    fn compare_and_bitpack(
        counters_0: &Self::BitCounterType,
        n_0: usize,
        counters_1: &Self::BitCounterType,
        n_1: usize,
    ) -> Self;
}

impl<T: IncrementCounters, const L: usize> IncrementCounters for [T; L]
where
    Self: Default,
{
    type BitCounterType = [T::BitCounterType; L];
    fn increment_counters(&self, counters: &mut [T::BitCounterType; L]) {
        for i in 0..L {
            self[i].increment_counters(&mut counters[i]);
        }
    }
    fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::threshold_and_bitpack(&counters[i], threshold);
        }
        target
    }
    fn compare_and_bitpack(
        counters_0: &Self::BitCounterType,
        n_0: usize,
        counters_1: &Self::BitCounterType,
        n_1: usize,
    ) -> Self {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::compare_and_bitpack(&counters_0[i], n_0, &counters_1[i], n_1);
        }
        target
    }
}

pub trait BitOr {
    fn bit_or(&self, other: &Self) -> Self;
}

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

macro_rules! array_bit_len {
    ($len:expr) => {
        impl<T: BitLen> BitLen for [T; $len] {
            const BIT_LEN: usize = $len * T::BIT_LEN;
        }
    };
}

array_bit_len!(1);
array_bit_len!(2);
array_bit_len!(3);
array_bit_len!(4);
array_bit_len!(5);
array_bit_len!(6);
array_bit_len!(7);
array_bit_len!(8);
array_bit_len!(13);
array_bit_len!(16);
array_bit_len!(25);
array_bit_len!(32);
pub trait GetBit {
    fn bit(&self, i: usize) -> bool;
}

macro_rules! impl_for_uint {
    ($type:ty, $len:expr) => {
        impl BitLen for $type {
            const BIT_LEN: usize = $len;
        }
        impl BitOr for $type {
            fn bit_or(&self, other: &Self) -> $type {
                self | other
            }
        }
        impl GetBit for $type {
            #[inline(always)]
            fn bit(&self, i: usize) -> bool {
                ((self >> i) & 1) == 1
            }
        }
        impl<I: HammingDistance + BitLen> BitMul<I, $type> for [(I, u32); $len] {
            fn bit_mul(&self, input: &I) -> $type {
                let mut target = <$type>::default();
                for i in 0..$len {
                    target |= ((self[i].0.hamming_distance(input) < self[i].1) as $type) << i;
                }
                target
            }
        }
        impl<I: HammingDistance + BitLen> BitMul<I, $type> for [I; $len] {
            fn bit_mul(&self, input: &I) -> $type {
                let mut target = <$type>::default();
                for i in 0..$len {
                    target |= ((self[i].hamming_distance(input) < (I::BIT_LEN as u32 / 2)) as $type) << i;
                }
                target
            }
        }
        impl IncrementCounters for $type {
            type BitCounterType = [u32; <$type>::BIT_LEN];
            fn increment_counters(&self, counters: &mut Self::BitCounterType) {
                for b in 0..<$type>::BIT_LEN {
                    counters[b] += ((self >> b) & 1) as u32
                }
            }
            fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
                let mut target = <$type>::default();
                for i in 0..$len {
                    target |= (counters[i] > threshold) as $type << i;
                }
                target
            }
            fn compare_and_bitpack(counters_0: &Self::BitCounterType, n_0: usize, counters_1: &Self::BitCounterType, n_1: usize) -> Self {
                let n0f = n_0 as f64;
                let n1f = n_1 as f64;
                let mut target = <$type>::default();
                for i in 0..$len {
                    target |= (counters_0[i] as f64 / n0f > counters_1[i] as f64 / n1f) as $type << i;
                }
                target
            }
        }
        impl HammingDistance for $type {
            #[inline(always)]
            fn hamming_distance(&self, other: &$type) -> u32 {
                (self ^ other).count_ones()
            }
        }
    };
}

impl_for_uint!(u32, 32);
impl_for_uint!(u16, 16);
impl_for_uint!(u8, 8);

pub trait BitMul<I, O> {
    fn bit_mul(&self, input: &I) -> O;
}

impl<I, O: Default + Copy, T: BitMul<I, O>, const L: usize> BitMul<I, [O; L]> for [T; L]
where
    [O; L]: Default,
{
    fn bit_mul(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_mul(input);
        }
        target
    }
}

pub trait OrPool<Output> {
    fn or_pool(&self) -> Output;
}
macro_rules! impl_orpool {
    ($x_size:expr, $y_size:expr) => {
        impl<Pixel: BitOr + Default + Copy> OrPool<[[Pixel; $y_size / 2]; $x_size / 2]>
            for [[Pixel; $y_size]; $x_size]
        {
            fn or_pool(&self) -> [[Pixel; $y_size / 2]; $x_size / 2] {
                let mut target = [[Pixel::default(); $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let x_index = x * 2;
                    for y in 0..$y_size / 2 {
                        let y_index = y * 2;
                        target[x][y] = self[x_index + 0][y_index + 0]
                            .bit_or(&self[x_index + 0][y_index + 1])
                            .bit_or(&self[x_index + 1][y_index + 0])
                            .bit_or(&self[x_index + 1][y_index + 1]);
                    }
                }
                target
            }
        }
    };
}

impl_orpool!(32, 32);
impl_orpool!(16, 16);
impl_orpool!(8, 8);

trait NormalizeAndEncode2D<const X: usize, const Y: usize> {
    fn normalize_and_unary_encode(&self) -> [[u32; Y]; X];
}

// slide the min to 0 but do not strech
impl<const X: usize, const Y: usize> NormalizeAndEncode2D<{ X }, { Y }> for [[[u8; 3]; Y]; X]
where
    [[u32; Y]; X]: Default,
{
    fn normalize_and_unary_encode(&self) -> [[u32; Y]; X] {
        let mut mins = [255u8; 3];
        for x in 0..X {
            for y in 0..Y {
                for c in 0..3 {
                    mins[c] = self[x][y][c].min(mins[c]);
                }
            }
        }
        let mut target = <[[u32; Y]; X]>::default();
        for x in 0..X {
            for y in 0..Y {
                let mut pixel = 0b0u32;
                for c in 0..3 {
                    pixel |= (!0b0u32 << ((self[x][y][c] - mins[c]) / 23)) << (c * 10);
                    pixel &= !0b0u32 >> (32 - ((c + 1) * 10));
                }
                // bit flip is just to make it more intuitive for humans.
                target[x][y] = !pixel;
            }
        }
        target
    }
}

trait ExtractPatches<Patch, const NORMALIZE: bool> {
    fn patches(&self, patches: &mut Vec<Patch>);
}

impl<P: Copy + Default, const X: usize, const Y: usize, const FX: usize, const FY: usize>
    ExtractPatches<[[P; FY]; FX], false> for [[P; Y]; X]
where
    [[P; FY]; FX]: Default,
{
    fn patches(&self, patches: &mut Vec<[[P; FY]; FX]>) {
        for x in 0..X - (FX - 1) {
            for y in 0..Y - (FY - 1) {
                let mut patch = <[[P; FY]; FX]>::default();
                for fx in 0..FX {
                    for fy in 0..FY {
                        patch[fx][fy] = self[x + fx][y + fy];
                    }
                }
                patches.push(patch);
            }
        }
    }
}

impl<const X: usize, const Y: usize, const FX: usize, const FY: usize>
    ExtractPatches<[[u32; FY]; FX], true> for [[[u8; 3]; Y]; X]
where
    [[[u8; 3]; FY]; FX]: Default + NormalizeAndEncode2D<{ FX }, { FY }>,
{
    fn patches(&self, patches: &mut Vec<[[u32; FY]; FX]>) {
        for x in 0..X - (FX - 1) {
            for y in 0..Y - (FY - 1) {
                let mut patch = <[[[u8; 3]; FY]; FX]>::default();
                for fx in 0..FX {
                    for fy in 0..FY {
                        patch[fx][fy] = self[x + fx][y + fy];
                    }
                }
                patches.push(patch.normalize_and_unary_encode());
            }
        }
    }
}

trait Conv2D<P, const NORMALIZE: bool, const X: usize, const Y: usize, const C: usize> {
    fn conv2d(&self, input: &[[P; Y]; X]) -> [[[u32; C]; Y]; X];
}

impl<
        P: HammingDistance + Copy,
        const X: usize,
        const Y: usize,
        const C: usize,
        const FX: usize,
        const FY: usize,
    > Conv2D<P, false, { X }, { Y }, { C }> for [[([[P; FY]; FX], u32); 32]; C]
where
    [[[u32; C]; Y]; X]: Default,
    [[P; FY]; FX]: Default,
    [[([[P; FY]; FX], u32); 32]; C]: BitMul<[[P; FY]; FX], [u32; C]>,
{
    fn conv2d(&self, input: &[[P; Y]; X]) -> [[[u32; C]; Y]; X] {
        let mut target = <[[[u32; C]; Y]; X]>::default();
        for x in 0..X - (FX - 1) {
            for y in 0..Y - (FY - 1) {
                let mut patch = <[[P; FY]; FX]>::default();
                for fx in 0..FX {
                    for fy in 0..FY {
                        patch[fx][fy] = input[x + fx][y + fy];
                    }
                }
                target[x + FX / 2][y + FY / 2] = self.bit_mul(&patch);
            }
        }
        target
    }
}

impl<const X: usize, const Y: usize, const C: usize, const FX: usize, const FY: usize>
    Conv2D<[u8; 3], true, { X }, { Y }, { C }> for [[([[u32; FY]; FX], u32); 32]; C]
where
    [[[u32; C]; Y]; X]: Default,
    [[[u8; 3]; FY]; FX]: Default + NormalizeAndEncode2D<{ FX }, { FY }>,
    [[([[u32; FY]; FX], u32); 32]; C]: BitMul<[[u32; FY]; FX], [u32; C]>,
{
    fn conv2d(&self, input: &[[[u8; 3]; Y]; X]) -> [[[u32; C]; Y]; X] {
        let mut target = <[[[u32; C]; Y]; X]>::default();
        for x in 0..X - (FX - 1) {
            for y in 0..Y - (FY - 1) {
                let mut patch = <[[[u8; 3]; FY]; FX]>::default();
                for fx in 0..FX {
                    for fy in 0..FY {
                        patch[fx][fy] = input[x + fx][y + fy];
                    }
                }
                target[x + FX / 2][y + FY / 2] = self.bit_mul(&patch.normalize_and_unary_encode());
            }
        }
        target
    }
}

trait AvgBits
where
    Self: IncrementCounters + Sized,
{
    fn count_bits(examples: &Vec<Self>) -> Self::BitCounterType;
    fn update_centers(example: &Vec<Self>, centers: &Vec<Self>) -> Vec<Self>;
    fn lloyds<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Self>, k: usize) -> Vec<(Self, u32)>;
}

impl<T: IncrementCounters + Send + Sync + Copy + HammingDistance + BitLen + Eq> AvgBits for T
where
    <Self as IncrementCounters>::BitCounterType:
        Default + Send + Sync + Counters + std::fmt::Debug,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    fn count_bits(examples: &Vec<Self>) -> Self::BitCounterType {
        examples
            .par_iter()
            .fold(
                || <Self as IncrementCounters>::BitCounterType::default(),
                |mut acc, example| {
                    example.increment_counters(&mut acc);
                    acc
                },
            )
            .reduce(
                || <Self as IncrementCounters>::BitCounterType::default(),
                |mut a, b| {
                    a.elementwise_add(&b);
                    a
                },
            )
    }
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
        for i in 0..100 {
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
            })
            .collect()
    }
}

trait Layer<InputImage, OutputImage> {
    fn layer<RNG: rand::Rng>(
        rng: &mut RNG,
        examples: &Vec<InputImage>,
        path: &Path,
    ) -> Vec<OutputImage>;
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
const FILTER_SIZE: usize = 3;
const N_CHANS: usize = 1;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    let images: Vec<([[[u8; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);

    let params_path = Path::new("params/lloyds_cluster/l0.prms");
    let weights =
        <[[([[u32; FILTER_SIZE]; FILTER_SIZE], u32); 32]; N_CHANS]>::new_from_fs(&params_path)
            .unwrap_or_else(|| {
                println!("no params found, training.");
                let examples: Vec<[[u32; FILTER_SIZE]; FILTER_SIZE]> = images
                    .par_iter()
                    .fold(
                        || vec![],
                        |mut acc, (image, _)| {
                            image.patches(&mut acc);
                            acc
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
                let clusters = <[[u32; FILTER_SIZE]; FILTER_SIZE]>::lloyds(&mut rng, &examples, N_CHANS * 32);
                for &filter in &clusters {
                    dbg!(filter.1);
                    print_patch(&filter.0);
                }

                let mut weights = [[([[0u32; FILTER_SIZE]; FILTER_SIZE], 0u32); 32]; N_CHANS];
                for (i, filter) in clusters.iter().enumerate() {
                    weights[i / 32][i % 32] = *filter;
                }
                weights.write_to_fs(&params_path);
                weights
            });

    println!("got params");
    let start = PreciseTime::now();
    let images: Vec<[[[u32; 1]; 32]; 32]> = images
        .par_iter()
        .map(|(image, _)| weights.conv2d(image))
        .collect();
    println!("time: {}", start.to(PreciseTime::now()));
    // PT2.04

    //for b in 0..32 {
    //    dbg!(b);
    //    for x in 0..32 {
    //        for y in 0..32 {
    //            print!("{}", if images[6][y][x][0].bit(b) { 1 } else { 0 });
    //        }
    //        print!("\n");
    //    }
    //    println!("-------------",);
    //}
    ////for row in &images[6] {
    ////    for pixel in row {
    ////        println!("{:032b}", pixel[0]);
    ////    }
    ////}
    //for (i, filter) in weights[0].iter().enumerate() {
    //    dbg!(i);
    //    dbg!(filter.1);
    //    print_patch(&filter.0);
    //}
}
