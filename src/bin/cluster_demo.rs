#![feature(const_generics)]

use bitnn::datasets::cifar;
//use bitnn::layers::SaveLoad;
//use bitnn::ExtractPatches;
use bincode::{deserialize_from, serialize_into};
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs::{create_dir_all, File};
use std::io::BufWriter;
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

impl<T: BitOr, const L: usize> BitOr for [T; L]
where
    [T; L]: Default,
{
    fn bit_or(&self, other: &Self) -> Self {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_or(&other[i]);
        }
        target
    }
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

trait Normalize2D<OutputPatch> {
    fn normalize_2d(&self) -> OutputPatch;
}

// slide the min to 0
impl<const X: usize, const Y: usize> Normalize2D<[[u32; Y]; X]> for [[[u8; 3]; Y]; X]
where
    [[u32; Y]; X]: Default,
{
    fn normalize_2d(&self) -> [[u32; Y]; X] {
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
                let mut bits = 0b0u32;
                for c in 0..3 {
                    bits |= (!0b0u32 << ((self[x][y][c] - mins[c]) / 23)) << (c * 10);
                    bits &= !0b0u32 >> (32 - ((c + 1) * 10));
                }
                target[x][y] = !bits;
            }
        }
        target
    }
}

// extracts patches and puts them in the pixels of the output image.
trait ExtractImagePatches<OutputImage> {
    fn extract_image_patches(&self) -> OutputImage;
}

impl<P: Copy, O, const X: usize, const Y: usize, const FX: usize, const FY: usize>
    ExtractImagePatches<[[[[O; FY]; FX]; Y]; X]> for [[P; Y]; X]
where
    [[P; FY]; FX]: Normalize2D<[[O; FY]; FX]>,
{
    fn extract_image_patches(&self) -> [[[[O; FY]; FX]; Y]; X] {}
}

trait ExtractPatches<Patch, const NORMALIZE: bool> {
    fn patches(&self, patches: &mut Vec<Patch>);
}

impl<
        I: Copy,
        O,
        const NORMALIZE: bool,
        const X: usize,
        const Y: usize,
        const FX: usize,
        const FY: usize,
    > ExtractPatches<[[O; FY]; FX], { NORMALIZE }> for [[I; Y]; X]
//where
//    [[I; FY]; FX]: Default + Encode2D<O, { NORMALIZE }, { FX }, { FY }>,
{
    fn patches(&self, patches: &mut Vec<[[O; FY]; FX]>) {
        for x in 0..X - (FX - 1) {
            for y in 0..Y - (FY - 1) {
                let mut patch = <[[I; FY]; FX]>::default();
                for fx in 0..FX {
                    for fy in 0..FY {
                        patch[fx][fy] = self[x + fx][y + fy];
                    }
                }
                patches.push(patch.encode_2d());
            }
        }
    }
}

trait Conv2D<InputImage, OutputImage, const NORMALIZE: bool> {
    fn conv2d(&self, input: &InputImage) -> OutputImage;
}

impl<
        I: Copy,
        O,
        const NORMALIZE: bool,
        const X: usize,
        const Y: usize,
        const C: usize,
        const FX: usize,
        const FY: usize,
    > Conv2D<[[I; Y]; X], [[[u32; C]; Y]; X], { NORMALIZE }> for [[([[O; FY]; FX], u32); 32]; C]
//where
//    [[[u32; C]; Y]; X]: Default,
//    [[([[O; FY]; FX], u32); 32]; C]: BitMul<[[O; FY]; FX], [u32; C]>,
//    [[I; FY]; FX]: Default + Encode2D<O, { NORMALIZE }, { FX }, { FY }>,
{
    fn conv2d(&self, input: &[[I; Y]; X]) -> [[[u32; C]; Y]; X] {
        let mut target = <[[[u32; C]; Y]; X]>::default();
        for x in 0..X - (FX - 1) {
            for y in 0..Y - (FY - 1) {
                let mut patch = <[[I; FY]; FX]>::default();
                for fx in 0..FX {
                    for fy in 0..FY {
                        patch[fx][fy] = input[x + fx][y + fy];
                    }
                }
                target[x + FX / 2][y + FY / 2] = self.bit_mul(&patch.encode_2d());
            }
        }
        target
    }
}

macro_rules! patch_2x2 {
    ($input:expr, $x:expr, $y:expr) => {
        [
            [$input[$x + 0][$y + 0], $input[$x + 0][$y + 1]],
            [$input[$x + 1][$y + 0], $input[$x + 1][$y + 1]],
        ]
    };
}

macro_rules! impl_strided_conv_2x2 {
    ($x:expr, $y:expr) => {
        impl<P: Copy, O, W: BitMul<[[P; 2]; 2], O>>
            Conv2D<[[P; $y]; $x], [[O; $y / 2]; $x / 2], false> for W
        where
            [[O; $y / 2]; $x / 2]: Default,
            [[P; 2]; 2]: Default + Encode2D<P, false, 2, 2>,
        {
            fn conv2d(&self, input: &[[P; $y]; $x]) -> [[O; $y / 2]; $x / 2] {
                let mut target = <[[O; $y / 2]; $x / 2]>::default();
                for x in 0..$x / 2 {
                    let x_index = x * 2;
                    for y in 0..$y / 2 {
                        let y_index = y * 2;
                        target[x][y] = self.bit_mul(&patch_2x2!(input, x_index, y_index));
                    }
                }
                target
            }
        }
    };
}

impl_strided_conv_2x2!(32, 32);
impl_strided_conv_2x2!(16, 16);
impl_strided_conv_2x2!(8, 8);

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

trait Layer<InputImage, OutputImage, const NORMALIZE: bool> {
    fn cluster_features<RNG: rand::Rng>(
        rng: &mut RNG,
        examples: &Vec<(InputImage, usize)>,
        path: &Path,
    ) -> (Vec<(OutputImage, usize)>, Self);
}

impl<
        //P: Sync + Send,
        W: Send + Sync,
        InputImage: Sync,
        OutputImage: Sync + Send,
        //const X: usize,
        //const Y: usize,
        const
        const C: usize,
        const FX: usize,
        const FY: usize,
        const NORMALIZE: bool,
    > Layer<InputImage, OutputImage, { NORMALIZE }> for [[([[W; FY]; FX], u32); 32]; C]
//where
//    for<'de> Self: serde::Deserialize<'de>,
//    Self: Conv2D<InputImage, OutputImage, { NORMALIZE }> + Default + serde::Serialize,
//    [[W; FY]; FX]: Lloyds + Copy,
//    InputImage: ExtractPatches<[[W; FY]; FX], { NORMALIZE }>,
{
    fn cluster_features<RNG: rand::Rng>(
        rng: &mut RNG,
        images: &Vec<(InputImage, usize)>,
        path: &Path,
    ) -> (Vec<(OutputImage, usize)>, Self) {
        let weights: Self = File::open(&path)
            .map(|f| deserialize_from(f).unwrap())
            .ok()
            .unwrap_or_else(|| {
                println!("no params found, training.");
                let examples: Vec<[[W; FY]; FX]> = images
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
                let clusters = <[[W; FY]; FX]>::lloyds(rng, &examples, C * 32);

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
        let images: Vec<(OutputImage, usize)> = images
            .par_iter()
            .map(|(image, class)| (weights.conv2d(image), *class))
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
    fn naive_bayes_classify(examples: &Vec<(Self, usize)>) -> [Self; K];
}

impl<T: HammingDistance + IncrementCounters + Sync + Send + Default + Copy, const K: usize>
    NaiveBayes<{ K }> for T
where
    [(usize, T::BitCounterType); K]: Default + Sync + Send,
    T::BitCounterType: Counters + Send + Sync + Default,
    [T; K]: Default,
{
    fn naive_bayes_classify(examples: &Vec<(T, usize)>) -> [T; K] {
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
            .map(|(n, bit_counter)| T::compare_and_bitpack(bit_counter, *n, &sum_counters, sum_n))
            .collect();
        let mut target = <[T; K]>::default();
        for c in 0..K {
            target[c] = filters[c];
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

const L1_CHANS: usize = 1;
const L2_CHANS: usize = 2;
const L3_CHANS: usize = 4;
const L4_CHANS: usize = 4;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    let images: Vec<([[[u8; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);

    let params_path = Path::new("params/lloyds_cluster_6");
    create_dir_all(&params_path).unwrap();

    let start = PreciseTime::now();
    let (images, weights): (
        Vec<([[[u32; L1_CHANS]; 32]; 32], usize)>,
        [[([[u32; 3]; 3], u32); 32]; L1_CHANS],
    ) = Layer::cluster_features(&mut rng, &images, &params_path.join("n0.prms"));
    println!("full time: {}", start.to(PreciseTime::now()));

    let (images, weights): (
        Vec<([[[u32; 2]; 16]; 16], usize)>,
        [[([[[u32; 1]; 2]; 2], u32); 32]; 2],
    ) = Layer::cluster_features(&mut rng, &images, &params_path.join("t1_c2x2.prms"));

    //let (images, weights): (
    //    Vec<([[[u32; L2_CHANS]; 16]; 16], usize)>,
    //    [[([[[u32; L1_CHANS]; 3]; 3], u32); 32]; L2_CHANS],
    //) = Layer::cluster_features(&mut rng, &images, &params_path.join("b2_l0.prms"));

    //let (images, weights): (
    //    Vec<([[[u32; 4]; 8]; 8], usize)>,
    //    [[([[[u32; 2]; 2]; 2], u32); 32]; 4],
    //) = Layer::cluster_features(&mut rng, &images, &params_path.join("t2_c2x2.prms"));

    //let (images, weights): (
    //    Vec<([[[u32; L3_CHANS]; 8]; 8], usize)>,
    //    [[([[[u32; L2_CHANS]; 3]; 3], u32); 32]; L3_CHANS],
    //) = Layer::cluster_features(&mut rng, &images, &params_path.join("l2.prms"));

    //let images: Vec<([[[u32; L3_CHANS]; 4]; 4], usize)> = images
    //    .par_iter()
    //    .map(|(image, class)| (image.or_pool(), *class))
    //    .collect();

    //let (images, weights): (
    //    Vec<([[[u32; L4_CHANS]; 4]; 4], usize)>,
    //    [[([[[u32; L3_CHANS]; 3]; 3], u32); 32]; L4_CHANS],
    //) = Layer::cluster_features(&mut rng, &images, &params_path.join("l3.prms"));

    //let fc_layer: [[[[u32; L4_CHANS]; 4]; 4]; 10] =
    //    <[[[u32; L4_CHANS]; 4]; 4]>::naive_bayes_classify(&images);

    //let dist: Vec<u64> = images
    //    .par_iter()
    //    .fold(
    //        || vec![0u64; 10],
    //        |mut acc, (image, _)| {
    //            for c in 0..10 {
    //                acc[c] += fc_layer[c].hamming_distance(image) as u64;
    //            }
    //            acc
    //        },
    //    )
    //    .reduce(
    //        || vec![0u64; 10],
    //        |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect(),
    //    );

    //let max_avg_act: f64 = *dist.iter().max().unwrap() as f64 / images.len() as f64;
    //dbg!(max_avg_act);
    //let biases: Vec<u32> = dist
    //    .iter()
    //    .map(|&x| (max_avg_act - (x as f64 / images.len() as f64)) as u32)
    //    .collect();
    //dbg!(&biases);

    //let n_correct: u64 = images
    //    .par_iter()
    //    .map(|(image, class)| {
    //        let actual_class: usize = fc_layer
    //            .iter()
    //            .enumerate()
    //            .min_by_key(|(i, filter)| filter.hamming_distance(image) + biases[*i])
    //            .unwrap()
    //            .0;
    //        (actual_class == *class) as u64
    //    })
    //    .sum();
    //dbg!(n_correct as f64 / images.len() as f64);
    // 0.235

    //for c in 0..2 {
    //    for b in 0..32 {
    //        dbg!((c, b));
    //        for x in 0..8 {
    //            for y in 0..8 {
    //                print!("{}", if images[6].0[y][x][c].bit(b) { 1 } else { 0 });
    //            }
    //            print!("\n");
    //        }
    //        println!("-------------",);
    //    }
    //}
    //for row in &images[6].0 {
    //    for pixel in row {
    //        for word in pixel {
    //            print!("{:032b}|", word);
    //        }
    //        print!("\n");
    //    }
    //}
    //for (i, (filter, bias)) in weights[0].iter().enumerate() {
    //    dbg!(i);
    //    dbg!(bias);
    //    for x in 0..3 {
    //        for y in 0..3 {
    //            print!("{:032b} | ", filter[x][y][0]);
    //        }
    //        print!("\n");
    //    }
    //    println!("-----------",);
    //}
}
