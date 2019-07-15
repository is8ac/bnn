#![feature(const_generics)]

use bitnn::datasets::cifar;
use bitnn::ExtractPatches;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;

macro_rules! patch_3x3 {
    ($input:expr, $x:expr, $y:expr) => {
        [
            [
                $input[$x + 0][$y + 0],
                $input[$x + 0][$y + 1],
                $input[$x + 0][$y + 2],
            ],
            [
                $input[$x + 1][$y + 0],
                $input[$x + 1][$y + 1],
                $input[$x + 1][$y + 2],
            ],
            [
                $input[$x + 2][$y + 0],
                $input[$x + 2][$y + 1],
                $input[$x + 2][$y + 2],
            ],
        ]
    };
}

trait Median {
    fn median(&self) -> u32;
    fn range(&self) -> (u32, u32);
    fn count_sides(&self, pivot: u32) -> (usize, usize);
}

impl Median for u32 {
    fn range(&self) -> (u32, u32) {
        (*self, *self)
    }
    fn median(&self) -> u32 {
        *self
    }
    fn count_sides(&self, pivot: u32) -> (usize, usize) {
        ((*self < pivot) as usize, (*self >= pivot) as usize)
    }
}

impl<T: Median, const L: usize> Median for [T; L] {
    fn range(&self) -> (u32, u32) {
        let mut max = 0u32;
        let mut min = !0u32;
        for i in 0..L {
            let (sub_min, sub_max) = self[i].range();
            max = max.max(sub_max);
            min = min.min(sub_min);
        }
        (min, max)
    }
    fn median(&self) -> u32 {
        let (mut min, mut max) = self.range();
        loop {
            let pivot = min + (max - min) / 2;
            let (nlt, nget) = self.count_sides(pivot);
            if nlt > nget {
                max = pivot;
            } else {
                min = pivot;
            }
            if (max - min) <= 1 {
                if nlt > nget {
                    return max;
                } else {
                    return min;
                }
            }
        }
    }
    fn count_sides(&self, pivot: u32) -> (usize, usize) {
        let mut lt = 0usize;
        let mut gt = 0usize;
        for i in 0..L {
            let (l, g) = self[i].count_sides(pivot);
            lt += l;
            gt += g;
        }
        (lt, gt)
    }
}

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

trait Conv2D<I, O> {
    fn conv2d(&self, input: &I) -> O;
}

impl<I: Copy, O: Default + Copy, W: BitMul<[[I; 3]; 3], O>, const X: usize, const Y: usize>
    Conv2D<[[I; Y]; X], [[O; Y]; X]> for W
where
    [[O; Y]; X]: Default,
{
    fn conv2d(&self, input: &[[I; Y]; X]) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                target[x + 1][y + 1] = self.bit_mul(&patch_3x3!(input, x, y));
            }
        }
        target
    }
}

trait AvgBits
where
    Self: IncrementCounters + Sized,
{
    fn avg_bits(examples: &Vec<Self>) -> Self;
    fn count_bits(examples: &Vec<Self>) -> Self::BitCounterType;
    fn split(&self, examples: &Vec<Self>) -> (Vec<Self>, Vec<Self>);
    fn reextract(&self, examples: &Vec<Self>) -> Self;
    // input is examples, output is features.
    // It will generate up to `depth^2 + (depth-1)` features
    fn recursively_extract_features(examples: &Vec<Self>, depth: usize) -> Vec<Self>;
    fn extractive_feature_gen(examples: &Vec<Self>, n: usize) -> Vec<Self>;
    fn update_centers(example: &Vec<Self>, centers: &Vec<Self>) -> Vec<Self>;
    fn lloyds<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Self>, k: usize) -> Vec<Self>;
}

impl<T: IncrementCounters + Send + Sync + Copy + HammingDistance + BitLen + Eq> AvgBits for T
where
    <Self as IncrementCounters>::BitCounterType:
        Default + Send + Sync + Counters + Median + std::fmt::Debug,
    rand::distributions::Standard: rand::distributions::Distribution<T>,
{
    fn avg_bits(examples: &Vec<Self>) -> Self {
        let per_bit_counts = Self::count_bits(examples);
        //let threshold = examples.len() as u32 / 2;
        let threshold = per_bit_counts.median();
        <Self>::threshold_and_bitpack(&per_bit_counts, threshold)
    }
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
    fn split(&self, examples: &Vec<Self>) -> (Vec<Self>, Vec<Self>) {
        let mut activations: Vec<u32> = examples
            .par_iter()
            .map(|x| self.hamming_distance(x))
            .collect();
        activations.par_sort();
        let threshold = activations[activations.len() / 2];
        let side_0: Vec<_> = examples
            .par_iter()
            .filter(|x| self.hamming_distance(x) <= threshold)
            .cloned()
            .collect();
        let side_1: Vec<_> = examples
            .par_iter()
            .filter(|x| self.hamming_distance(x) > threshold)
            .cloned()
            .collect();
        (side_0, side_1)
    }
    fn reextract(&self, examples: &Vec<Self>) -> Self {
        let (side_0, side_1) = self.split(&examples);
        let bit_counts_0 = Self::count_bits(&side_0);
        let bit_counts_1 = Self::count_bits(&side_1);
        Self::compare_and_bitpack(&bit_counts_0, side_0.len(), &bit_counts_1, side_1.len())
    }
    fn recursively_extract_features(examples: &Vec<Self>, depth: usize) -> Vec<Self> {
        let mut filters = vec![];
        let avg_patch = Self::avg_bits(&examples);

        let split_filter = avg_patch.reextract(&examples);

        if depth == 0 {
            dbg!(examples.len());
            filters.push(split_filter);
        } else {
            let (side_0_examples, side_1_examples) = split_filter.split(&examples);
            filters.append(&mut Self::recursively_extract_features(
                &side_0_examples,
                depth - 1,
            ));
            filters.append(&mut Self::recursively_extract_features(
                &side_1_examples,
                depth - 1,
            ));
        }
        filters
    }
    fn extractive_feature_gen(examples: &Vec<Self>, n: usize) -> Vec<Self> {
        // first we need to get the average patch, that is, the patch where a bit is set iff >50% of the patches have that bit set.
        let per_bit_counts = Self::count_bits(examples);
        let avg_patch = <Self>::threshold_and_bitpack(&per_bit_counts, examples.len() as u32 / 2);

        // then we get the examples which have less then 1/3 of hte bits wrong.
        let near_corners: Vec<_> = examples
            .par_iter()
            .filter(|x| avg_patch.hamming_distance(x) < (Self::BIT_LEN as u32 / 3))
            .cloned()
            .collect();
        dbg!(near_corners.len() as f64 / examples.len() as f64);

        // and average them
        let per_bit_counts = Self::count_bits(&near_corners);
        let threshold = near_corners.len() as u32 / 2;
        let avg_patch = <Self>::threshold_and_bitpack(&per_bit_counts, threshold);
        let near_corners: Vec<_> = near_corners
            .par_iter()
            .filter(|x| avg_patch.hamming_distance(x) < 60)
            .cloned()
            .collect();
        dbg!(near_corners.len());
        dbg!(near_corners.len() as f64 / examples.len() as f64);

        vec![avg_patch]
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
            .map(|(i, x)| <Self>::threshold_and_bitpack(&x, *i as u32 / 2))
            .collect()
    }
    fn lloyds<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Self>, k: usize) -> Vec<Self> {
        let mut clusters: Vec<Self> = (0..32).map(|_| rng.gen()).collect();
        for i in 0..30 {
            dbg!(i);
            let new_clusters = <Self>::update_centers(examples, &clusters);
            if new_clusters == clusters {
                break;
            }
            clusters = new_clusters;
        }
        clusters
    }
}

fn print_patch(patch: &[[[u8; 3]; 3]; 3]) {
    for x in 0..3 {
        for y in 0..3 {
            for c in 0..3 {
                print!("{:08b}|", patch[x][y][c]);
            }
            print!(" ");
        }
        print!("\n");
    }
    println!("-----------",);
}

const N_EXAMPLES: usize = 50_00;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    let images: Vec<([[[u8; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);
    let examples: Vec<[[[u8; 3]; 3]; 3]> = images
        .par_iter()
        .map(|(image, _)| image.patches())
        .flatten()
        .collect();
    dbg!(examples.len());

    let clusters = <[[[u8; 3]; 3]; 3]>::lloyds(&mut rng, &examples, 32);
    for filter in &clusters {
        print_patch(&filter);
    }

    let mut weights = [([[[0u8; 3]; 3]; 3], 92u32); 32];
    //let mut weights = [[[[0u8; 3]; 3]; 3]; 32];
    for (i, filter) in clusters.iter().enumerate() {
        weights[i].0 = *filter;
    }

    let images: Vec<[[u32; 32]; 32]> = images
        .par_iter()
        .map(|(image, _)| weights.conv2d(image))
        .collect();
    let examples: Vec<[[u32; 3]; 3]> = images
        .par_iter()
        .map(|image| image.patches())
        .flatten()
        .collect();
    dbg!(examples.len());

    for b in 0..32 {
        for x in 0..32 {
            for y in 0..32 {
                print!("{}", if images[6][y][x].bit(b) { 1 } else { 0 });
            }
            print!("\n");
        }
        println!("-------------",);
    }
}
