#![feature(const_generics)]

use bincode::{deserialize_from, serialize_into};
use bitnn::datasets::cifar;
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

impl<A: Counters, B: Counters> Counters for (A, B) {
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
    }
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

impl<A: HammingDistance, B: HammingDistance> HammingDistance for (A, B) {
    fn hamming_distance(&self, other: &(A, B)) -> u32 {
        self.0.hamming_distance(&other.0) + self.1.hamming_distance(&other.1)
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

impl<A: IncrementCounters, B: IncrementCounters> IncrementCounters for (A, B) {
    type BitCounterType = (A::BitCounterType, B::BitCounterType);
    fn increment_counters(&self, counters: &mut Self::BitCounterType) {
        self.0.increment_counters(&mut counters.0);
        self.1.increment_counters(&mut counters.1);
    }
    fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
        (
            A::threshold_and_bitpack(&counters.0, threshold),
            B::threshold_and_bitpack(&counters.1, threshold),
        )
    }
    fn compare_and_bitpack(
        counters_0: &Self::BitCounterType,
        n_0: usize,
        counters_1: &Self::BitCounterType,
        n_1: usize,
    ) -> Self {
        (
            A::compare_and_bitpack(&counters_0.0, n_0, &counters_1.0, n_1),
            B::compare_and_bitpack(&counters_0.1, n_0, &counters_1.1, n_1),
        )
    }
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

impl<A: BitLen, B: BitLen> BitLen for (A, B) {
    const BIT_LEN: usize = A::BIT_LEN + B::BIT_LEN;
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

pub trait AvgPool {
    type OutputImage;
    fn avg_pool(&self) -> Self::OutputImage;
}

macro_rules! impl_avgpool {
    ($x_size:expr, $y_size:expr) => {
        impl AvgPool for [[[u8; 3]; $y_size]; $x_size] {
            type OutputImage = [[[u8; 3]; $y_size / 2]; $x_size / 2];
            fn avg_pool(&self) -> Self::OutputImage {
                let mut target = [[[0u8; 3]; $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let x_index = x * 2;
                    for y in 0..$y_size / 2 {
                        let y_index = y * 2;
                        for c in 0..3 {
                            let sum = self[x_index + 0][y_index + 0][c] as u16
                                + self[x_index + 0][y_index + 1][c] as u16
                                + self[x_index + 1][y_index + 0][c] as u16
                                + self[x_index + 1][y_index + 1][c] as u16;
                            target[x][y][c] = (sum / 4) as u8;
                        }
                    }
                }
                target
            }
        }
    };
}

impl_avgpool!(32, 32);
impl_avgpool!(16, 16);
impl_avgpool!(8, 8);

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

trait Normalize2D<P> {
    type OutputImage;
    fn normalize_2d(&self) -> Self::OutputImage;
}

// slide the min to 0
impl<const X: usize, const Y: usize> Normalize2D<[u8; 3]> for [[[u8; 3]; Y]; X]
where
    [[[u8; 3]; Y]; X]: Default,
{
    type OutputImage = [[[u8; 3]; Y]; X];
    fn normalize_2d(&self) -> Self::OutputImage {
        let mut mins = [255u8; 3];
        for x in 0..X {
            for y in 0..Y {
                for c in 0..3 {
                    mins[c] = self[x][y][c].min(mins[c]);
                }
            }
        }
        let mut target = Self::OutputImage::default();
        for x in 0..X {
            for y in 0..Y {
                for c in 0..3 {
                    target[x][y][c] = self[x][y][c] - mins[c];
                }
            }
        }
        target
    }
}

trait ExtractPixels<P> {
    fn extract_pixels(&self, pixels: &mut Vec<P>);
}

impl<P: Copy, const X: usize, const Y: usize> ExtractPixels<P> for [[P; Y]; X] {
    fn extract_pixels(&self, pixels: &mut Vec<P>) {
        for x in 0..X {
            for y in 0..Y {
                pixels.push(self[x][y]);
            }
        }
    }
}

trait PixelMap2D<I, O> {
    type OutputImage;
    fn map_2d<F: Fn(&I) -> O>(&self, map_fn: F) -> Self::OutputImage;
}

impl<I, O, const X: usize, const Y: usize> PixelMap2D<I, O> for [[I; Y]; X]
where
    [[O; Y]; X]: Default,
{
    type OutputImage = [[O; Y]; X];
    fn map_2d<F: Fn(&I) -> O>(&self, map_fn: F) -> Self::OutputImage {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..X {
            for y in 0..Y {
                target[x][y] = map_fn(&self[x][y]);
            }
        }
        target
    }
}

trait Concat2D<A, B> {
    fn concat_2d(a: &A, b: &B) -> Self;
}

impl<A: Copy, B: Copy, const X: usize, const Y: usize> Concat2D<[[A; Y]; X], [[B; Y]; X]>
    for [[(A, B); Y]; X]
where
    Self: Default,
{
    fn concat_2d(a: &[[A; Y]; X], b: &[[B; Y]; X]) -> Self {
        let mut target = <Self>::default();
        for x in 0..X {
            for y in 0..Y {
                target[x][y] = (a[x][y], b[x][y]);
            }
        }
        target
    }
}

// extracts patches and puts them in the pixels of the output image.
trait Conv2D<OutputImage> {
    fn conv2d(&self) -> OutputImage;
}

macro_rules! impl_conv2d_2x2 {
    ($x:expr, $y:expr) => {
        impl<P: Copy + Default> Conv2D<[[[[P; 2]; 2]; $y / 2]; $x / 2]> for [[P; $y]; $x] {
            fn conv2d(&self) -> [[[[P; 2]; 2]; $y / 2]; $x / 2] {
                let mut target = <[[[[P; 2]; 2]; $y / 2]; $x / 2]>::default();
                for x in 0..$x / 2 {
                    let x_offset = x * 2;
                    for y in 0..$y / 2 {
                        let y_offset = y * 2;
                        for fx in 0..2 {
                            for fy in 0..2 {
                                target[x][y][fx][fy] = self[x_offset + fx][y_offset + fy];
                            }
                        }
                    }
                }
                target
            }
        }
    };
}

impl_conv2d_2x2!(32, 32);
impl_conv2d_2x2!(16, 16);
impl_conv2d_2x2!(8, 8);

impl<P: Copy, const X: usize, const Y: usize> Conv2D<[[[[P; 3]; 3]; Y]; X]> for [[P; Y]; X]
where
    [[[[P; 3]; 3]; Y]; X]: Default,
{
    fn conv2d(&self) -> [[[[P; 3]; 3]; Y]; X] {
        let mut target = <[[[[P; 3]; 3]; Y]; X]>::default();

        for fx in 1..3 {
            for fy in 1..3 {
                target[0][0][fx][fy] = self[0 + fx][0 + fy];
            }
        }
        for y in 0..Y - 2 {
            for fx in 1..3 {
                for fy in 0..3 {
                    target[0][y + 1][fx][fy] = self[0 + fx][y + fy];
                }
            }
        }
        for fx in 1..3 {
            for fy in 0..2 {
                target[0][Y - 1][fx][fy] = self[0 + fx][Y - 2 + fy];
            }
        }

        // begin center
        for x in 0..X - 2 {
            for fx in 0..3 {
                for fy in 1..3 {
                    target[x + 1][0][fx][fy] = self[x + fx][0 + fy];
                }
            }
            for y in 0..Y - 2 {
                for fx in 0..3 {
                    for fy in 0..3 {
                        target[x + 1][y + 1][fx][fy] = self[x + fx][y + fy];
                    }
                }
            }
            for fx in 0..3 {
                for fy in 0..2 {
                    target[x + 1][Y - 1][fx][fy] = self[x + fx][Y - 2 + fy];
                }
            }
        }
        // end center

        for fx in 0..2 {
            for fy in 1..3 {
                target[X - 1][0][fx][fy] = self[X - 2 + fx][0 + fy];
            }
        }
        for y in 0..Y - 2 {
            for fx in 0..2 {
                for fy in 0..3 {
                    target[X - 1][y + 1][fx][fy] = self[X - 2 + fx][y + fy];
                }
            }
        }
        for fx in 0..2 {
            for fy in 0..2 {
                target[X - 1][Y - 1][fx][fy] = self[X - 2 + fx][Y - 2 + fy];
            }
        }

        target
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

const N_EXAMPLES: usize = 50_0;
const B0_CHANS: usize = 1;
const B1_CHANS: usize = 2;
const B2_CHANS: usize = 4;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    let examples: Vec<([[[u8; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);

    let raw_images: Vec<[[[u8; 3]; 32]; 32]> =
        examples.par_iter().map(|(image, _)| *image).collect();

    let params_path = Path::new("params/lloyds_cluster_9");
    create_dir_all(&params_path).unwrap();

    let start = PreciseTime::now();
    println!("full time: {}", start.to(PreciseTime::now()));

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
    //let conved_images: Vec<[[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32]> =
    //    images.par_iter().map(|image| image.conv2d()).collect();

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

    //dbg!(&zipped_images[0..5]);

    let (images, weights) = <[[(_, u32); 32]; B2_CHANS]>::cluster_features(
        &mut rng,
        &zipped_images,
        &params_path.join("b2_l0.prms"),
    );
    dbg!(weights);
}
