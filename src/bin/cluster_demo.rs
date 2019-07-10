#![feature(const_generics)]

use bitnn::datasets::mnist;
use bitnn::ExtractPatches;
use rayon::prelude::*;
use std::path::Path;

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
    fn compare_and_bitpack(counters_0: &Self::BitCounterType, n_0: usize, counters_1: &Self::BitCounterType, n_1: usize) -> Self;
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
    fn compare_and_bitpack(counters_0: &Self::BitCounterType, n_0: usize, counters_1: &Self::BitCounterType, n_1: usize) -> Self {
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

macro_rules! impl_for_uint {
    ($type:ty, $len:expr) => {
        impl BitLen for $type {
            const BIT_LEN: usize = $len;
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
}

impl<T: IncrementCounters + Send + Sync + Copy + HammingDistance> AvgBits for T
where
    <Self as IncrementCounters>::BitCounterType: Default + Send + Sync + Counters + Median + std::fmt::Debug,
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
        let mut activations: Vec<u32> = examples.par_iter().map(|x| self.hamming_distance(x)).collect();
        activations.par_sort();
        let threshold = activations[activations.len() / 2];
        let side_0: Vec<_> = examples.par_iter().filter(|x| self.hamming_distance(x) <= threshold).cloned().collect();
        let side_1: Vec<_> = examples.par_iter().filter(|x| self.hamming_distance(x) > threshold).cloned().collect();
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
        if examples.len() > 0 {
            dbg!(examples.len());
            let avg_patch = Self::avg_bits(&examples);

            let split_filter = avg_patch.reextract(&examples);

            if depth > 0 {
                let (side_0_examples, side_1_examples) = split_filter.split(&examples);
                filters.append(&mut Self::recursively_extract_features(&side_0_examples, depth - 1));
                filters.append(&mut Self::recursively_extract_features(&side_1_examples, depth - 1));
            }
            filters.push(split_filter);
        }
        filters
    }
}

fn print_patch(patch: &[[u8; 3]; 3]) {
    for x in 0..3 {
        print!("{:08b}|", patch[x][0]);
        print!("{:08b}|", patch[x][1]);
        print!("{:08b}\n", patch[x][2]);
    }
    println!("-----------",);
}

const N_EXAMPLES: usize = 60_000;

fn main() {
    let images = mnist::load_images_u8_unary(Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), N_EXAMPLES);
    let examples: Vec<[[u8; 3]; 3]> = images.par_iter().map(|image| image.patches()).flatten().collect();
    dbg!(examples.len());
    let filters = <[[u8; 3]; 3]>::recursively_extract_features(&examples, 4);
    dbg!(filters.len());

    for filter in &filters {
        print_patch(&filter);
    }
}
