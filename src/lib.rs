extern crate rand;
extern crate rayon;
extern crate time;
use rand::{Rng, SeedableRng, StdRng};
#[macro_use]
pub mod datasets {
    pub mod cifar {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;

        pub fn encode_unary_rgb(bytes: [u8; 3]) -> u64 {
            let mut ones = 0u64;
            for color in 0..3 {
                for i in 0..bytes[color] / 12 {
                    ones = ones | 0b1u64 << (color * 21 + i as usize);
                }
            }
            ones
        }
        pub fn parse_rgb_u64(bits: u64) -> [u8; 3] {
            let mut bytes = [0u8; 3];
            for color in 0..3 {
                bytes[color] = (((bits >> (color * 21)) & 0b111111111111111111111u64).count_ones()
                    * 12) as u8; // 21 bit mask
            }
            bytes
        }
        pub fn load_images_10(path: &String, size: usize) -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open data");

            let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
            let mut label: [u8; 1] = [0; 1];
            let mut images: Vec<(usize, [[[u8; 3]; 32]; 32])> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut label).expect("can't read label");
                file.read_exact(&mut image_bytes)
                    .expect("can't read images");
                let mut image = [[[0u8; 3]; 32]; 32];
                for x in 0..32 {
                    for y in 0..32 {
                        image[x][y] = [
                            image_bytes[(0 * 1024) + (y * 32) + x],
                            image_bytes[(1 * 1024) + (y * 32) + x],
                            image_bytes[(2 * 1024) + (y * 32) + x],
                        ];
                    }
                }
                images.push((label[0] as usize, image));
            }
            return images;
        }
    }
    pub mod mnist {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;
        pub fn load_labels(path: &String, size: usize) -> Vec<usize> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 8] = [0; 8];
            file.read_exact(&mut header).expect("can't read header");

            let mut byte: [u8; 1] = [0; 1];
            let mut labels: Vec<usize> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut byte).expect("can't read label");
                labels.push(byte[0] as usize);
            }
            return labels;
        }
        pub fn load_images_u8_1chan(path: &String, size: usize) -> Vec<[[[u8; 1]; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[[[u8; 1]; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image = [[[0u8; 1]; 28]; 28];
                for p in 0..784 {
                    image[p / 28][p % 28][0] = images_bytes[p];
                }
                images.push(image);
            }
            return images;
        }
        pub fn load_images(path: &String, size: usize) -> Vec<[[u8; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[[u8; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image = [[0u8; 28]; 28];
                for p in 0..784 {
                    image[p / 28][p % 28] = images_bytes[p];
                }
                images.push(image);
            }
            return images;
        }
        pub fn load_images_bitpacked(path: &String, size: usize) -> Vec<[u64; 13]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in reverse order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[u64; 13]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image_words: [u64; 13] = [0; 13];
                for p in 0..784 {
                    let word_index = p / 64;
                    image_words[word_index] =
                        image_words[word_index] | (((images_bytes[p] > 128) as u64) << p % 64);
                }
                images.push(image_words);
            }
            return images;
        }
        pub fn load_images_64chan(path: &String, size: usize) -> Vec<[[[u64; 1]; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[[[u64; 1]; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image = [[[0u64; 1]; 28]; 28];
                for p in 0..784 {
                    let mut ones = 0u64;
                    for i in 0..images_bytes[p] / 4 {
                        ones = ones | 1 << i;
                    }
                    image[p / 28][p % 28][0] = ones;
                }
                images.push(image);
            }
            return images;
        }
    }
}

macro_rules! for_uints {
    ($tokens:tt) => {
        $tokens!(u8);
        $tokens!(u16);
        $tokens!(u32);
        $tokens!(u64);
        $tokens!(u128);
    };
}

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

macro_rules! primitive_bit_len {
    ($type:ty, $len:expr) => {
        impl BitLen for $type {
            const BIT_LEN: usize = $len;
        }
    };
}

primitive_bit_len!(u8, 8);
primitive_bit_len!(u16, 16);
primitive_bit_len!(u32, 32);
primitive_bit_len!(u64, 64);
primitive_bit_len!(u128, 128);

macro_rules! array_bit_len {
    ($len:expr) => {
        impl<T: BitLen> BitLen for [T; $len] {
            const BIT_LEN: usize = $len * T::BIT_LEN;
        }
    };
}

array_bit_len!(2);
array_bit_len!(3);
array_bit_len!(4);
array_bit_len!(5);
array_bit_len!(6);
array_bit_len!(7);
array_bit_len!(8);
array_bit_len!(9);
array_bit_len!(16);
array_bit_len!(32);
array_bit_len!(49);
array_bit_len!(28);
array_bit_len!(13);
array_bit_len!(64);
array_bit_len!(128);

pub trait Patch: Send + Sync + Sized + BitLen {
    fn hamming_distance(&self, &Self) -> u32;
    fn bit_increment(&self, &mut [u32]);
    fn bitpack(&[bool]) -> Self;
    fn bit_or(&self, &Self) -> Self;
    fn flip_bit(&mut self, usize);
    fn get_bit(&self, usize) -> bool;
}

macro_rules! primitive_patch {
    ($type:ty) => {
        impl Patch for $type {
            fn hamming_distance(&self, other: &$type) -> u32 {
                (self ^ other).count_ones()
            }
            fn bit_increment(&self, counters: &mut [u32]) {
                if counters.len() != <$type>::BIT_LEN {
                    panic!(
                        "primitive increment: counters is {:?}, should be {:?}",
                        counters.len(),
                        <$type>::BIT_LEN
                    );
                }
                for i in 0..<$type>::BIT_LEN {
                    counters[i] += ((self >> i) & 0b1 as $type) as u32;
                }
            }
            fn bitpack(bools: &[bool]) -> $type {
                if bools.len() != <$type>::BIT_LEN {
                    panic!(
                        "primitive bitpack: counters is {:?}, should be {:?}",
                        bools.len(),
                        <$type>::BIT_LEN
                    );
                }
                let mut val = 0 as $type;
                for i in 0..<$type>::BIT_LEN {
                    val = val | ((bools[i] as $type) << i);
                }
                val
            }
            fn bit_or(&self, other: &$type) -> $type {
                self | other
            }
            fn flip_bit(&mut self, index: usize) {
                *self ^= 1 << index
            }
            fn get_bit(&self, index: usize) -> bool {
                ((self >> index) & 0b1) == 1
            }
        }
    };
}

for_uints!(primitive_patch);

macro_rules! array_patch {
    ($len:expr) => {
        impl<T: Patch + Copy + Default> Patch for [T; $len] {
            fn hamming_distance(&self, other: &[T; $len]) -> u32 {
                let mut distance = 0;
                for i in 0..$len {
                    distance += self[i].hamming_distance(&other[i]);
                }
                distance
            }
            fn bit_increment(&self, counters: &mut [u32]) {
                if counters.len() != ($len * T::BIT_LEN) {
                    panic!(
                        "array increment: counters is {:?}, should be {:?}",
                        counters.len(),
                        $len * T::BIT_LEN
                    );
                }
                for i in 0..$len {
                    self[i].bit_increment(&mut counters[i * T::BIT_LEN..(i + 1) * T::BIT_LEN]);
                }
            }
            fn bitpack(bools: &[bool]) -> [T; $len] {
                if bools.len() != ($len * T::BIT_LEN) {
                    panic!(
                        "array bitpack: bools is {:?}, should be {:?}",
                        bools.len(),
                        $len * T::BIT_LEN
                    );
                }
                let mut val = [T::default(); $len];
                for i in 0..$len {
                    val[i] = T::bitpack(&bools[i * T::BIT_LEN..(i + 1) * T::BIT_LEN]);
                }
                val
            }
            fn bit_or(&self, other: &Self) -> Self {
                let mut output = [T::default(); $len];
                for i in 0..$len {
                    output[i] = self[i].bit_or(&other[i]);
                }
                output
            }
            fn flip_bit(&mut self, index: usize) {
                self[index / T::BIT_LEN].flip_bit(index % T::BIT_LEN);
            }
            fn get_bit(&self, index: usize) -> bool {
                self[index / T::BIT_LEN].get_bit(index % T::BIT_LEN)
            }
        }
    };
}

array_patch!(2);
array_patch!(3);
array_patch!(4);
array_patch!(5);
array_patch!(6);
array_patch!(7);
array_patch!(8);
array_patch!(9);
array_patch!(13);
array_patch!(16);
array_patch!(28);
array_patch!(32);
array_patch!(49);
array_patch!(64);

pub mod featuregen {
    use super::layers::bitvecmul;
    use super::Patch;
    use rayon::prelude::*;

    // splits labels examples by class.
    pub fn split_by_label<T: Copy>(examples: &Vec<(usize, T)>, len: usize) -> Vec<Vec<T>> {
        let mut by_label: Vec<Vec<T>> = (0..len).map(|_| Vec::new()).collect();
        for (label, example) in examples {
            by_label[*label].push(*example);
        }
        let _: Vec<_> = by_label.iter_mut().map(|x| x.shrink_to_fit()).collect();
        by_label
    }

    pub fn gen_readout_features<T: Patch + Sync + Clone>(by_class: &Vec<Vec<T>>) -> Vec<T> {
        let num_examples: usize = by_class.iter().map(|x| x.len()).sum();

        (0..by_class.len())
            .map(|class| {
                let grads = grads_one_shard(by_class, class);
                let scaled_grads: Vec<f64> =
                    grads.iter().map(|x| x / num_examples as f64).collect();
                //println!("{:?}", scaled_grads);
                grads_to_bits(&scaled_grads)
            })
            .collect()
    }

    // split_labels_set_by_filter takes examples and a split func. It returns the
    // examples is a vec of shards of labels of examples of patches.
    // It returns the same data but with double the shards and with each Vec of patches (very approximately) half the length.
    // split_fn is used to split each Vec of patches between two shards.
    pub fn split_labels_set_by_distance<T: Copy + Send + Sync + Patch>(
        examples: &Vec<Vec<Vec<T>>>,
        base_point: &T,
        threshold: u32,
        filter_thresh_len: usize,
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
                                .filter(|x| x.hamming_distance(base_point) > threshold)
                                .cloned()
                                .collect()
                        })
                        .collect(),
                    by_label
                        .par_iter()
                        .map(|label_examples| {
                            label_examples
                                .iter()
                                .filter(|x| x.hamming_distance(base_point) <= threshold)
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
                sum_len > filter_thresh_len
            })
            .collect()
    }
    fn avg_bit_sums(len: usize, counts: &Vec<u32>) -> Vec<f64> {
        counts
            .iter()
            .map(|&count| count as f64 / len as f64)
            .collect()
    }

    fn count_bits<T: Patch + Sync>(patches: &Vec<T>) -> Vec<u32> {
        patches
            .par_iter()
            .fold(
                || vec![0u32; T::BIT_LEN],
                |mut counts, example| {
                    example.bit_increment(&mut counts);
                    counts
                },
            )
            .reduce(
                || vec![0u32; T::BIT_LEN],
                |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect(),
            )
    }

    pub fn grads_one_shard<T: Patch + Sync>(by_class: &Vec<Vec<T>>, label: usize) -> Vec<f64> {
        let mut sum_bits: Vec<(usize, Vec<u32>)> = by_class
            .iter()
            .map(|label_patches| (label_patches.len(), count_bits(&label_patches)))
            .collect();

        let (target_len, target_sums) = sum_bits.remove(label);
        let (other_len, other_sums) = sum_bits.iter().fold(
            (0usize, vec![0u32; T::BIT_LEN]),
            |(a_len, a_vals), (b_len, b_vals)| {
                (
                    a_len + b_len,
                    a_vals
                        .iter()
                        .zip(b_vals.iter())
                        .map(|(x, y)| x + y)
                        .collect(),
                )
            },
        );
        let other_avg = avg_bit_sums(other_len, &other_sums);
        let target_avg = avg_bit_sums(target_len, &target_sums);

        let grads: Vec<f64> = other_avg
            .iter()
            .zip(target_avg.iter())
            .map(|(a, b)| (a - b) * (target_len.min(other_len) as f64))
            .collect();
        grads
    }

    pub fn grads_to_bits<T: Patch>(grads: &Vec<f64>) -> T {
        let sign_bits: Vec<bool> = grads.iter().map(|x| *x > 0f64).collect();
        T::bitpack(&sign_bits)
    }

    pub fn gen_basepoint<T: Patch + Sync>(shards: &Vec<Vec<Vec<T>>>, label: usize) -> T {
        let len: u64 = shards
            .par_iter()
            .map(|shard| {
                let sum: u64 = shard.iter().map(|class: &Vec<T>| class.len() as u64).sum();
                sum
            })
            .sum();

        let grads: Vec<f64> = shards
            .par_iter()
            .filter(|x| {
                let class_len = x[label].len();
                if class_len > 0 {
                    let total_len: usize = x.iter().map(|y| y.len()).sum();
                    return (total_len - class_len) > 0;
                }
                class_len > 0
            })
            .map(|shard| {
                grads_one_shard(&shard, label)
                    .iter()
                    .map(|&grad| grad * len as f64)
                    .collect()
            })
            .fold(
                || vec![0f64; T::BIT_LEN],
                |acc, grads: Vec<f64>| acc.iter().zip(grads.iter()).map(|(a, b)| a + b).collect(),
            )
            .reduce(
                || vec![0f64; T::BIT_LEN],
                |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect(),
            );

        let sign_bits: Vec<bool> = grads.iter().map(|x| *x > 0f64).collect();

        T::bitpack(&sign_bits)
    }
    pub fn gen_threshold<T: Patch + Sync>(patches: &Vec<T>, base_point: &T) -> u32 {
        let mut bit_distances: Vec<u32> = patches
            .par_iter()
            .map(|y| y.hamming_distance(&base_point))
            .collect();
        bit_distances.par_sort();
        bit_distances[bit_distances.len() / 2]
    }
}

#[macro_use]
pub mod layers {
    use super::featuregen;
    use super::{BitLen, Patch};
    use rayon::prelude::*;
    use std::marker::PhantomData;
    use std::mem::transmute;

    pub trait VecApply<I: Send + Sync, O: Send + Sync> {
        fn vec_apply(&self, &[(usize, I)]) -> Vec<(usize, O)>;
        fn vec_update(&self, inputs: &[(usize, I)], targets: &mut Vec<(usize, O)>, index: usize) {
            *targets = self.vec_apply(&inputs);
        }
    }

    macro_rules! vec_apply_trait {
        ($len:expr) => {
            impl<I: Send + Sync + Copy, O: Send + Sync> VecApply<I, O> for [I; $len]
            where
                Self: Apply<I, O>,
            {
                fn vec_apply(&self, inputs: &[(usize, I)]) -> Vec<(usize, O)> {
                    inputs
                        .par_iter()
                        .map(|(class, input)| (*class, self.apply(input)))
                        .collect()
                }
                fn vec_update(
                    &self,
                    inputs: &[(usize, I)],
                    targets: &mut Vec<(usize, O)>,
                    index: usize,
                ) {
                    let _: Vec<_> = targets
                        .par_iter_mut()
                        .zip(inputs.par_iter())
                        .map(|(x, input)| self.update(&input.1, &mut x.1, index))
                        .collect();
                }
            }
        };
    }

    vec_apply_trait!(8);
    vec_apply_trait!(16);
    vec_apply_trait!(32);
    vec_apply_trait!(64);
    vec_apply_trait!(128);

    pub trait Apply<I: Send + Sync, O: Send + Sync>
    where
        Self: Sync,
    {
        fn apply(&self, &I) -> O;
        fn update(&self, input: &I, target: &mut O, _: usize) {
            *target = self.apply(input);
        }
    }

    struct FusedLayer<A, T, B> {
        a: A,
        b: B,
        t: PhantomData<T>,
    }

    impl<I: Send + Sync, T: Send + Sync, O: Send + Sync, A: Apply<I, T>, B: Apply<T, O>>
        Apply<I, O> for FusedLayer<A, T, B>
    {
        fn apply(&self, input: &I) -> O {
            self.b.apply(&self.a.apply(input))
        }
        fn update(&self, input: &I, target: &mut O, i: usize) {
            *target = self.apply(input);
        }
    }

    macro_rules! primitive_dense_apply {
        ($type:ty) => {
            impl<I: Patch + Send + Sync + Default> Apply<I, $type> for [I; <$type>::BIT_LEN] {
                fn apply(&self, input: &I) -> $type {
                    let mut target = <$type>::default();
                    for i in 0..<$type>::BIT_LEN {
                        target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2))
                            as $type)
                            << i;
                    }
                    target
                }
                fn update(&self, input: &I, target: &mut $type, i: usize) {
                    *target &= !(1 << i); // unset the bit
                    *target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2))
                        as $type)
                        << i;
                }
            }
        };
    }

    primitive_dense_apply!(u8);
    primitive_dense_apply!(u16);
    primitive_dense_apply!(u32);
    primitive_dense_apply!(u64);
    primitive_dense_apply!(u128);

    macro_rules! array_dense_apply {
        ($len:expr) => {
            impl<I: Sync + Send, T: Default + Copy + Sync + Send + Patch, W: Apply<I, T>>
                Apply<I, [T; $len]> for [W; $len]
            {
                fn apply(&self, input: &I) -> [T; $len] {
                    let mut target = [T::default(); $len];
                    for i in 0..$len {
                        target[i] = self[i].apply(input);
                    }
                    target
                }
                fn update(&self, input: &I, target: &mut [T; $len], i: usize) {
                    self[i / T::BIT_LEN].update(
                        input,
                        &mut target[i / T::BIT_LEN],
                        i % T::BIT_LEN,
                    );
                }
            }
        };
    }

    array_dense_apply!(2);
    array_dense_apply!(3);
    array_dense_apply!(4);
    array_dense_apply!(5);

    pub struct PoolOrLayer;

    impl<I: Patch> NewFromSplit<I> for PoolOrLayer {
        fn new_from_split(_examples: &Vec<(usize, I)>) -> Self {
            PoolOrLayer
        }
    }

    impl<I: Sync + Send + Patch + Copy, O: Sync + Send> OptimizeLayer<I, O> for PoolOrLayer
    where
        Self: VecApply<I, O>,
    {
        fn optimize_layer<H: ObjectiveHead<O>>(
            &mut self,
            head: &mut H,
            examples: &[(usize, I)],
            update_freq: usize,
        ) -> f64 {
            let new_examples: Vec<(usize, O)> = (*self).vec_apply(examples);
            let acc = head.optimize(&new_examples, update_freq);
            acc
        }
    }

    use rand::{Rng, ThreadRng};
    pub trait NewFromRng<I> {
        fn new_from_rng(rng: &mut ThreadRng, input_bit_len: usize) -> Self;
    }

    macro_rules! primitive_activations_new_from_seed {
        ($len:expr) => {
            impl<I: Patch> NewFromRng<I> for [I; $len]
            where
                rand::distributions::Standard: rand::distributions::Distribution<I>,
            {
                fn new_from_rng(rng: &mut ThreadRng, _: usize) -> Self {
                    rng.gen::<[I; $len]>()
                }
            }
        };
    }
    primitive_activations_new_from_seed!(10);
    primitive_activations_new_from_seed!(8);
    primitive_activations_new_from_seed!(16);
    primitive_activations_new_from_seed!(32);

    pub trait NewFromSplit<I> {
        fn new_from_split(&Vec<(usize, I)>) -> Self;
    }

    macro_rules! primitive_activations_new_from_split {
        ($len:expr) => {
            impl<I: Patch + Copy + Default> NewFromSplit<I> for [I; $len] {
                fn new_from_split(examples: &Vec<(usize, I)>) -> Self {
                    let mut weights = [I::default(); $len];
                    let train_inputs = featuregen::split_by_label(&examples, 10);

                    let flat_inputs: Vec<I> = examples.iter().map(|(_, input)| *input).collect();
                    let mut shards: Vec<Vec<Vec<I>>> = vec![train_inputs.clone().to_owned()];
                    for i in 0..$len {
                        let class = i % 10;
                        let base_point = featuregen::gen_basepoint(&shards, class);
                        let mut bit_distances: Vec<u32> = flat_inputs
                            .par_iter()
                            .map(|y| y.hamming_distance(&base_point))
                            .collect();
                        bit_distances.par_sort();
                        let threshold = bit_distances[bit_distances.len() / 2];
                        weights[i] = base_point;

                        shards = featuregen::split_labels_set_by_distance(
                            &shards,
                            &base_point,
                            threshold,
                            2,
                        );
                    }
                    weights
                }
            }
        };
    }
    primitive_activations_new_from_split!(8);
    primitive_activations_new_from_split!(16);
    primitive_activations_new_from_split!(32);
    primitive_activations_new_from_split!(64);
    primitive_activations_new_from_split!(128);

    pub trait Mutate {
        fn mutate(&mut self, output_index: usize, input_index: usize);
        const OUTPUT_LEN: usize;
        const INPUT_LEN: usize;
    }

    macro_rules! primitive_mutate_matrix_trait {
        ($len:expr) => {
            impl<I: Patch> Mutate for [I; $len] {
                fn mutate(&mut self, output_index: usize, input_index: usize) {
                    self[output_index].flip_bit(input_index);
                }
                const OUTPUT_LEN: usize = $len;
                const INPUT_LEN: usize = I::BIT_LEN;
            }
        };
    }

    primitive_mutate_matrix_trait!(8);
    primitive_mutate_matrix_trait!(16);
    primitive_mutate_matrix_trait!(32);
    primitive_mutate_matrix_trait!(64);
    primitive_mutate_matrix_trait!(128);

    pub trait OptimizeLayer<I, O> {
        fn optimize_layer<H: ObjectiveHead<O>>(&mut self, &mut H, &[(usize, I)], usize) -> f64;
    }

    impl<I: Sync + Patch + Send + Copy, O: Sync + Send, W: Mutate + VecApply<I, O>>
        OptimizeLayer<I, O> for W
    {
        fn optimize_layer<H: ObjectiveHead<O>>(
            &mut self,
            head: &mut H,
            examples: &[(usize, I)],
            update_freq: usize,
        ) -> f64 {
            let new_examples: Vec<(usize, O)> = (*self).vec_apply(examples);
            let mut acc = head.acc(&new_examples);
            let mut iter = 0;
            for o in 0..W::OUTPUT_LEN {
                //println!("o: {:?}", o);
                let mut cache: Vec<(usize, O)> = (*self).vec_apply(&examples);
                acc = head.optimize(&cache, update_freq);
                //println!("{} output: {:?}%", o, acc * 100f64);
                for b in 0..W::INPUT_LEN {
                    if iter % update_freq == 0 {
                        (*self).vec_update(&examples, &mut cache, o);
                        acc = head.optimize(&cache, update_freq);
                        iter += 1;
                    }
                    self.mutate(o, b);
                    (*self).vec_update(&examples, &mut cache, o);
                    //println!("starting acc", );
                    let new_acc = head.acc(&cache);
                    //println!("end acc", );
                    if new_acc > acc {
                        acc = new_acc;
                        //println!("{} {} {:?}%", o, b, acc * 100f64);
                        iter += 1;
                    } else {
                        // revert
                        self.mutate(o, b);
                    }
                }
            }
            acc
        }
    }

    pub trait ObjectiveHead<I> {
        fn acc(&self, &Vec<(usize, I)>) -> f64;
        fn optimize(&mut self, &Vec<(usize, I)>, usize) -> f64;
    }

    impl<I: Patch + Send + Sync + Default + Copy> NewFromSplit<I> for [I; 10] {
        fn new_from_split(examples: &Vec<(usize, I)>) -> Self {
            let by_class = featuregen::split_by_label(&examples, 10);
            let mut readout = [I::default(); 10];
            for class in 0..10 {
                let grads = featuregen::grads_one_shard(&by_class, class);
                let sign_bits: Vec<bool> = grads.iter().map(|x| *x > 0f64).collect();
                readout[class] = I::bitpack(&sign_bits);
            }
            readout
        }
    }

    impl<I: Patch + Patch + Copy + Default> ObjectiveHead<I> for [I; 10] {
        fn acc(&self, examples: &Vec<(usize, I)>) -> f64 {
            let sum_correct: u64 = examples
                .par_iter()
                .map(|(class, input)| {
                    let target_act = self[*class].hamming_distance(input);
                    for i in 0..10 {
                        if i != *class {
                            let act = self[i].hamming_distance(input);
                            if act >= target_act {
                                return 0;
                            }
                        }
                    }
                    1
                })
                .sum();
            sum_correct as f64 / examples.len() as f64
        }
        fn optimize(&mut self, examples: &Vec<(usize, I)>, _update_freq: usize) -> f64 {
            let before_acc = self.acc(examples);
            for mut_class in 0..10 {
                let mut activation_diffs: Vec<(I, u32, bool)> = examples
                    .par_iter()
                    .map(|(targ_class, input)| {
                        let mut activations: Vec<u32> = self
                            .iter()
                            .map(|base_point| base_point.hamming_distance(&input))
                            .collect();

                        let targ_act = activations[*targ_class]; // the activation for target class of this example.
                        activations[*targ_class] = 0;
                        activations[mut_class] = 0;
                        let max_other_activations = activations.iter().max().unwrap(); // the max activation of all the classes not in the target class or mut class.
                        let max: u32 = {
                            if *targ_class == mut_class {
                                *max_other_activations
                            } else {
                                targ_act
                            }
                        };
                        (
                            input,
                            max,
                            *targ_class == mut_class,
                            (targ_act > *max_other_activations) | (*targ_class == mut_class),
                        ) // diff betwene the activation of the
                    })
                    .filter(|(_, _, _, keep)| *keep)
                    .map(|(input, diff, sign, _)| (*input, diff, sign))
                    .collect();

                // note that this sum correct is not the true acc, it is working on the subset that can be made correct or incorrect by this activation.
                let mut sum_correct: u64 = activation_diffs
                    .par_iter()
                    .map(|(input, max, sign)| {
                        let mut_act = self[mut_class].hamming_distance(input);
                        ((*sign ^ (mut_act < *max)) & (mut_act != *max)) as u64
                    })
                    .sum();
                let mut cur_acc = sum_correct as f64 / examples.len() as f64;
                for b in 0..I::BIT_LEN {
                    self[mut_class].flip_bit(b);
                    let new_sum_correct: u64 = activation_diffs
                        .par_iter()
                        .map(|(input, max, sign)| {
                            // new diff is the diff after flipping the weights bit
                            let mut_act = self[mut_class].hamming_distance(input);
                            // do we want mut_act to be smaller or larger?
                            // same as this statement:
                            //(if *sign { new_diff > 0 } else { new_diff < 0 }) as i64
                            ((*sign ^ (mut_act < *max)) & (mut_act != *max)) as u64
                        })
                        .sum();
                    let fast_new_acc = new_sum_correct as f64 / examples.len() as f64;
                    let real_new_acc = self.acc(&examples);
                    //println!("{:?} {:?}", fast_new_acc, real_new_acc);
                    assert_eq!(fast_new_acc, real_new_acc);
                    if new_sum_correct > sum_correct {
                        sum_correct = new_sum_correct;
                    } else {
                        // revert the bit
                        self[mut_class].flip_bit(b);
                    }
                }
            }
            let after_acc = self.acc(examples);
            if before_acc > after_acc {
                println!("reverting acc regression: {} > {}", before_acc, after_acc);
            }
            after_acc
        }
    }

    pub struct Layer<I: Sync + Send, L: VecApply<I, O>, O: Sync + Send, H: ObjectiveHead<O>> {
        input: PhantomData<I>,
        output: PhantomData<O>,
        pub data: L,
        pub head: H,
    }

    impl<
            I: Send + Sync + Copy + BitLen,
            O: Send + Sync,
            L: NewFromRng<I> + VecApply<I, O>,
            H: NewFromRng<O> + ObjectiveHead<O>,
        > NewFromRng<I> for Layer<I, L, O, H>
    {
        fn new_from_rng(rng: &mut ThreadRng, input_bit_len: usize) -> Self {
            let layer = L::new_from_rng(rng, input_bit_len);
            let head = H::new_from_rng(rng, I::BIT_LEN);
            Layer {
                input: PhantomData,
                output: PhantomData,
                data: layer,
                head: head,
            }
        }
    }

    impl<
            I: Sync + Send + Copy,
            O: Sync + Send,
            L: VecApply<I, O> + NewFromSplit<I>,
            H: NewFromSplit<O> + ObjectiveHead<O>,
        > NewFromSplit<I> for Layer<I, L, O, H>
    {
        fn new_from_split(examples: &Vec<(usize, I)>) -> Self {
            let layer = L::new_from_split(examples);
            let output_examples = layer.vec_apply(&examples);
            let head = H::new_from_split(&output_examples);
            Layer {
                input: PhantomData,
                output: PhantomData,
                data: layer,
                head: head,
            }
        }
    }

    impl<
            I: Sync + Send + Copy,
            O: Sync + Send,
            L: VecApply<I, O> + OptimizeLayer<I, O>,
            H: ObjectiveHead<O>,
        > ObjectiveHead<I> for Layer<I, L, O, H>
    {
        fn acc(&self, examples: &Vec<(usize, I)>) -> f64 {
            let output_examples = self.data.vec_apply(&examples);
            self.head.acc(&output_examples)
        }
        fn optimize(&mut self, examples: &Vec<(usize, I)>, update_freq: usize) -> f64 {
            self.data
                .optimize_layer(&mut self.head, examples, update_freq)
        }
    }

    pub struct OrPool2x2<P> {
        pixel_type: PhantomData<P>,
    }
    impl<P, I: Image2D<P>> NewFromSplit<I> for OrPool2x2<P> {
        fn new_from_split(_: &Vec<(usize, I)>) -> Self {
            OrPool2x2 {
                pixel_type: PhantomData,
            }
        }
    }

    impl<I: Send + Sync + Image2D<P>, O: Copy + Send + Sync + Image2D<P>, P> OptimizeLayer<I, O>
        for OrPool2x2<P>
    where
        Self: VecApply<I, O>,
    {
        fn optimize_layer<H: ObjectiveHead<O>>(
            &mut self,
            head: &mut H,
            examples: &[(usize, I)],
            update_freq: usize,
        ) -> f64 {
            let new_examples: Vec<(usize, O)> = (*self).vec_apply(examples);
            let acc = head.optimize(&new_examples, update_freq);
            acc
        }
    }

    macro_rules! or_pool2x2_trait {
        ($x_len:expr, $y_len:expr) => {
            impl<P: Patch + Copy + Default>
                Apply<[[P; $y_len]; $x_len], [[P; ($y_len / 2)]; ($x_len / 2)]> for OrPool2x2<P>
            {
                fn apply(
                    &self,
                    input: &[[P; $y_len]; $x_len],
                ) -> [[P; ($y_len / 2)]; ($x_len / 2)] {
                    let mut pooled = [[P::default(); $y_len / 2]; $x_len / 2];
                    for x in 0..($x_len / 2) {
                        let x_base = x * 2;
                        for y in 0..($y_len / 2) {
                            let y_base = y * 2;
                            pooled[x][y] = input[x_base + 0][y_base + 0]
                                .bit_or(&input[x_base + 0][y_base + 1])
                                .bit_or(&input[x_base + 1][y_base + 0])
                                .bit_or(&input[x_base + 1][y_base + 1]);
                        }
                    }
                    pooled
                }
            }
        };
    }

    or_pool2x2_trait!(32, 32);
    or_pool2x2_trait!(16, 16);
    or_pool2x2_trait!(8, 8);
    or_pool2x2_trait!(4, 4);

    impl<P, I: Send + Sync + Copy, O: Send + Sync> VecApply<I, O> for OrPool2x2<P>
    where
        Self: Apply<I, O>,
    {
        fn vec_apply(&self, inputs: &[(usize, I)]) -> Vec<(usize, O)> {
            inputs
                .par_iter()
                .map(|(class, input)| (*class, self.apply(input)))
                .collect()
        }
    }

    pub trait SimplifyBits<T> {
        fn simplify(&self) -> T;
    }

    macro_rules! simplify_bits_trait {
        ($in_type:ty, $out_type:ty) => {
            impl SimplifyBits<$out_type> for $in_type {
                fn simplify(&self) -> $out_type {
                    unsafe { transmute::<$in_type, $out_type>(*self) }
                }
            }
        };
    }

    simplify_bits_trait!([u8; 8], u64);
    simplify_bits_trait!([u16; 4], u64);
    simplify_bits_trait!([[u8; 28]; 28], [[u128; 7]; 7]);

    macro_rules! simplify_to_1u128_bits_trait {
        ($in_type:ty) => {
            simplify_bits_trait!($in_type, u128);
        };
    }

    simplify_to_1u128_bits_trait!([u8; 16]);
    simplify_to_1u128_bits_trait!([u16; 8]);
    simplify_to_1u128_bits_trait!([u32; 4]);
    simplify_to_1u128_bits_trait!([u64; 2]);

    macro_rules! simplify_bits_trait_array {
        ($in_type:ty, $len:expr) => {
            simplify_bits_trait!($in_type, [u128; $len]);
        };
    }

    simplify_bits_trait_array!([u8; 32], 2);
    simplify_bits_trait_array!([u8; 784], 49);
    simplify_bits_trait_array!([[u8; 28]; 28], 49);
    simplify_bits_trait_array!([u32; 8], 2);
    simplify_bits_trait_array!([u64; 8], 4);

    pub mod bitvecmul {
        macro_rules! primitive_bit_vecmul {
            ($name:ident, $type:ty, $len:expr) => {
                pub fn $name<T: super::Patch>(
                    weights: &[(T, u32); $len],
                    input: &T,
                ) -> [u32; $len] {
                    let mut output = [0u32; $len];
                    for b in 0..$len {
                        output[b] = (weights[b].0).hamming_distance(input);
                    }
                    output
                }
            };
        }

        primitive_bit_vecmul!(bvm_u3, u8, 3);
        primitive_bit_vecmul!(bvm_u7, u8, 7);
        primitive_bit_vecmul!(bvm_u8, u8, 8);
        primitive_bit_vecmul!(bvm_u14, u16, 14);
        primitive_bit_vecmul!(bvm_u16, u16, 16);
        primitive_bit_vecmul!(bvm_u32, u32, 32);
        primitive_bit_vecmul!(bvm_u64, u64, 64);
        primitive_bit_vecmul!(bvm_u128, u128, 128);

        // Vec Masked Bit Vector Multiply
        pub fn vbvm<T: super::Patch>(weights: &Vec<T>, input: &T) -> Vec<u32> {
            weights
                .iter()
                .map(|signs| input.hamming_distance(&signs))
                .collect()
        }

    }

    pub mod unary {
        macro_rules! to_unary {
            ($name:ident, $type:ty, $len:expr) => {
                pub fn $name(input: u8) -> $type {
                    !((!0) << (input / (255 / $len)))
                }
            };
        }

        to_unary!(to_3, u8, 3);
        to_unary!(to_4, u8, 4);
        to_unary!(to_5, u8, 5);
        to_unary!(to_6, u8, 6);
        to_unary!(to_7, u8, 7);
        to_unary!(to_8, u8, 8);
        to_unary!(to_14, u16, 14);

        pub fn rgb_to_u14(pixels: [u8; 3]) -> u16 {
            to_4(pixels[0]) as u16
                | ((to_5(pixels[1]) as u16) << 4)
                | ((to_5(pixels[2]) as u16) << 9)
        }
    }

    pub struct PixelMap<I: Send + Sync, FN: Apply<I, O>, O: Send + Sync> {
        input_type: PhantomData<I>,
        output_type: PhantomData<O>,
        map_fn: FN,
    }

    impl<I: Send + Sync, FN: Apply<I, O> + Mutate, O: Send + Sync> Mutate for PixelMap<I, FN, O> {
        fn mutate(&mut self, output_index: usize, input_index: usize) {
            self.map_fn.mutate(output_index, input_index);
        }
        const OUTPUT_LEN: usize = FN::OUTPUT_LEN;
        const INPUT_LEN: usize = FN::INPUT_LEN;
    }

    impl<
            II: Image2D<I> + Send + Sync,
            I: Send + Sync,
            O: Send + Sync,
            FN: Apply<I, O> + NewFromSplit<I>,
        > NewFromSplit<II> for PixelMap<I, FN, O>
    where
        ExtractPixels<I>: VecApply<II, I>,
    {
        fn new_from_split(examples: &Vec<(usize, II)>) -> Self {
            let pixels = ExtractPixels::new_from_split(&examples).vec_apply(&examples);
            PixelMap {
                input_type: PhantomData,
                output_type: PhantomData,
                map_fn: FN::new_from_split(&pixels),
            }
        }
    }

    macro_rules! pixel_map_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Send + Sync + Copy, FN: Apply<I, O>, O: Send + Sync + Default + Copy>
                Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]> for PixelMap<I, FN, O>
            {
                fn apply(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size]; $x_size] {
                    let mut target = [[O::default(); $y_size]; $x_size];
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            target[x][y] = self.map_fn.apply(&input[x][y]);
                        }
                    }
                    target
                }
            }
        };
    }

    pixel_map_apply_trait!(32, 32);
    pixel_map_apply_trait!(16, 16);

    impl<
            II: Image2D<IP> + Send + Sync,
            IP: Send + Sync,
            OI: Image2D<OP> + Send + Sync,
            OP: Send + Sync,
            FN: Apply<IP, OP>,
        > VecApply<II, OI> for PixelMap<IP, FN, OP>
    where
        Self: Apply<II, OI>,
    {
        fn vec_apply(&self, inputs: &[(usize, II)]) -> Vec<(usize, OI)> {
            inputs
                .par_iter()
                .map(|(class, input)| (*class, self.apply(input)))
                .collect()
        }
    }

    pub struct Patch3x3NotchedConv<
        I: Send + Sync,
        Patch: Send + Sync,
        FN: Apply<Patch, O>,
        O: Send + Sync,
    >
    where
        [I; 8]: SimplifyBits<Patch>,
    {
        input_pixel_type: PhantomData<I>,
        patch_type: PhantomData<Patch>,
        output_type: PhantomData<O>,
        map_fn: FN,
    }

    impl<I: Send + Sync, P: Send + Sync, FN: Apply<P, O> + Mutate, O: Send + Sync> Mutate
        for Patch3x3NotchedConv<I, P, FN, O>
    where
        [I; 8]: SimplifyBits<P>,
    {
        fn mutate(&mut self, output_index: usize, input_index: usize) {
            self.map_fn.mutate(output_index, input_index);
        }
        const OUTPUT_LEN: usize = FN::OUTPUT_LEN;
        const INPUT_LEN: usize = FN::INPUT_LEN;
    }

    impl<
            II: Image2D<I> + Send + Sync,
            I: Send + Sync,
            O: Send + Sync,
            P: Send + Sync,
            FN: Apply<P, O> + NewFromSplit<P>,
        > NewFromSplit<II> for Patch3x3NotchedConv<I, P, FN, O>
    where
        ExtractPatchesNotched3x3<P>: VecApply<II, P>,
        [I; 8]: SimplifyBits<P>,
    {
        fn new_from_split(examples: &Vec<(usize, II)>) -> Self {
            let patches = ExtractPatchesNotched3x3::new_from_split(&examples).vec_apply(&examples);
            Patch3x3NotchedConv {
                input_pixel_type: PhantomData,
                patch_type: PhantomData,
                output_type: PhantomData,
                map_fn: FN::new_from_split(&patches),
            }
        }
    }

    macro_rules! patch_conv_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<
                    I: Send + Sync + Copy,
                    P: Send + Sync,
                    FN: Apply<P, O>,
                    O: Send + Sync + Default + Copy,
                > Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]>
                for Patch3x3NotchedConv<I, P, FN, O>
            where
                [I; 8]: SimplifyBits<P>,
            {
                fn apply(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size]; $x_size] {
                    let mut target = [[O::default(); $y_size]; $x_size];
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            target[x][y] = self.map_fn.apply(
                                &([
                                    input[x + 1][y + 0],
                                    input[x + 2][y + 0],
                                    input[x + 0][y + 1],
                                    input[x + 1][y + 1],
                                    input[x + 2][y + 1],
                                    input[x + 0][y + 2],
                                    input[x + 1][y + 2],
                                    input[x + 2][y + 2],
                                ])
                                .simplify(),
                            );
                        }
                    }
                    target
                }
            }
        };
    }

    patch_conv_apply_trait!(32, 32);
    patch_conv_apply_trait!(16, 16);

    impl<
            II: Image2D<IP> + Send + Sync,
            IP: Send + Sync,
            OI: Image2D<OP> + Send + Sync,
            OP: Send + Sync,
            P: Send + Sync,
            FN: Apply<P, OP>,
        > VecApply<II, OI> for Patch3x3NotchedConv<IP, P, FN, OP>
    where
        Self: Apply<II, OI>,
        [IP; 8]: SimplifyBits<P>,
    {
        fn vec_apply(&self, inputs: &[(usize, II)]) -> Vec<(usize, OI)> {
            inputs
                .par_iter()
                .map(|(class, input)| (*class, self.apply(input)))
                .collect()
        }
    }

    pub struct ExtractPatchesNotched3x3<OP> {
        output_type: PhantomData<OP>,
    }

    impl<I, O> NewFromSplit<I> for ExtractPatchesNotched3x3<O> {
        fn new_from_split(_examples: &Vec<(usize, I)>) -> Self {
            ExtractPatchesNotched3x3 {
                output_type: PhantomData,
            }
        }
    }

    impl<I: Send + Sync, O: Copy + Send + Sync> OptimizeLayer<I, O> for ExtractPatchesNotched3x3<O>
    where
        Self: VecApply<I, O>,
    {
        fn optimize_layer<H: ObjectiveHead<O>>(
            &mut self,
            head: &mut H,
            examples: &[(usize, I)],
            update_freq: usize,
        ) -> f64 {
            let new_examples: Vec<(usize, O)> = (*self).vec_apply(examples);
            let acc = head.optimize(&new_examples, update_freq);
            acc
        }
    }

    // 3x3 with notch, flattened to [T; 8]
    macro_rules! extract_patch_8_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Send + Sync + Copy, O: Send + Sync> VecApply<[[I; $y_size]; $x_size], O>
                for ExtractPatchesNotched3x3<O>
            where
                [I; 8]: SimplifyBits<O>,
            {
                fn vec_apply(
                    &self,
                    examples: &[(usize, [[I; $y_size]; $x_size])],
                ) -> Vec<(usize, O)> {
                    let mut patches =
                        Vec::with_capacity(($y_size - 2) * ($x_size - 2) * examples.len());
                    for (class, image) in examples {
                        for x in 0..$x_size - 2 {
                            for y in 0..$y_size - 2 {
                                patches.push((
                                    *class,
                                    ([
                                        image[x + 1][y + 0],
                                        image[x + 2][y + 0],
                                        image[x + 0][y + 1],
                                        image[x + 1][y + 1],
                                        image[x + 2][y + 1],
                                        image[x + 0][y + 2],
                                        image[x + 1][y + 2],
                                        image[x + 2][y + 2],
                                    ])
                                    .simplify(),
                                ));
                            }
                        }
                    }
                    patches
                }
            }
        };
    }

    extract_patch_8_trait!(32, 32);
    extract_patch_8_trait!(16, 16);
    extract_patch_8_trait!(8, 8);

    pub struct ExtractPixels<P> {
        pixel_type: PhantomData<P>,
    }

    impl<I, O> NewFromSplit<I> for ExtractPixels<O> {
        fn new_from_split(_examples: &Vec<(usize, I)>) -> Self {
            ExtractPixels {
                pixel_type: PhantomData,
            }
        }
    }

    impl<I: Send + Sync, O: Copy + Send + Sync> OptimizeLayer<I, O> for ExtractPixels<O>
    where
        Self: VecApply<I, O>,
    {
        fn optimize_layer<H: ObjectiveHead<O>>(
            &mut self,
            head: &mut H,
            examples: &[(usize, I)],
            update_freq: usize,
        ) -> f64 {
            let new_examples: Vec<(usize, O)> = (*self).vec_apply(examples);
            let acc = head.optimize(&new_examples, update_freq);
            acc
        }
    }

    macro_rules! extract_pixel_vecapply {
        ($x_size:expr, $y_size:expr) => {
            impl<P: Copy + Send + Sync> VecApply<[[P; $y_size]; $x_size], P> for ExtractPixels<P> {
                fn vec_apply(
                    &self,
                    examples: &[(usize, [[P; $y_size]; $x_size])],
                ) -> Vec<(usize, P)> {
                    let mut patches = Vec::with_capacity($y_size * $x_size * examples.len());
                    for (class, image) in examples {
                        for x in 0..$x_size {
                            for y in 0..$y_size {
                                patches.push((*class, image[x][y]));
                            }
                        }
                    }
                    patches
                }
            }
        };
    }

    extract_pixel_vecapply!(32, 32);
    extract_pixel_vecapply!(16, 16);
    extract_pixel_vecapply!(8, 8);

    pub struct ExtractPixelsPadded<P> {
        pixel_type: PhantomData<P>,
    }

    impl<I, O> NewFromSplit<I> for ExtractPixelsPadded<O> {
        fn new_from_split(_examples: &Vec<(usize, I)>) -> Self {
            ExtractPixelsPadded {
                pixel_type: PhantomData,
            }
        }
    }

    impl<I: Send + Sync, O: Copy + Send + Sync> OptimizeLayer<I, O> for ExtractPixelsPadded<O>
    where
        Self: VecApply<I, O>,
    {
        fn optimize_layer<H: ObjectiveHead<O>>(
            &mut self,
            head: &mut H,
            examples: &[(usize, I)],
            update_freq: usize,
        ) -> f64 {
            let new_examples: Vec<(usize, O)> = (*self).vec_apply(examples);
            let acc = head.optimize(&new_examples, update_freq);
            acc
        }
    }

    macro_rules! extract_pixel_1x1_padded_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<P: Copy + Send + Sync> VecApply<[[P; $y_size]; $x_size], P>
                for ExtractPixelsPadded<P>
            {
                fn vec_apply(
                    &self,
                    examples: &[(usize, [[P; $y_size]; $x_size])],
                ) -> Vec<(usize, P)> {
                    let mut patches =
                        Vec::with_capacity(($y_size - 2) * ($x_size - 2) * examples.len());
                    for (class, image) in examples {
                        for x in 1..$x_size - 1 {
                            for y in 1..$y_size - 1 {
                                patches.push((*class, image[x][y]));
                            }
                        }
                    }
                    patches
                }
            }
        };
    }

    extract_pixel_1x1_padded_trait!(32, 32);
    extract_pixel_1x1_padded_trait!(16, 16);

    pub trait Image2D<P> {}

    macro_rules! image2d_trait {
        ($size:expr) => {
            impl<P> Image2D<P> for [[P; $size]; $size] {}
        };
    }
    image2d_trait!(32);
    image2d_trait!(16);
    image2d_trait!(8);

    pub trait PatchMap<IP, IA, O, OP> {
        fn patch_map(&self, &Fn(&IA, &mut OP)) -> O;
    }

    macro_rules! patch_map_trait_pixel {
        ($x_size:expr, $y_size:expr) => {
            impl<IP, OP: Copy + Default> PatchMap<IP, IP, [[OP; $y_size]; $x_size], OP>
                for [[IP; $y_size]; $x_size]
            {
                fn patch_map(&self, map_fn: &Fn(&IP, &mut OP)) -> [[OP; $y_size]; $x_size] {
                    let mut output = [[OP::default(); $y_size]; $x_size];
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            map_fn(&self[x][y], &mut output[x][y]);
                        }
                    }
                    output
                }
            }
        };
    }

    patch_map_trait_pixel!(32, 32);
    patch_map_trait_pixel!(28, 28);
    patch_map_trait_pixel!(16, 16);
}

#[cfg(test)]
mod tests {
    use super::layers::{unary, Apply};
    use super::Patch;
    #[test]
    fn patch_count() {
        let mut counters = vec![0u32; 32];
        [0b1011_1111u8, 0b0100_0000u8, 0b0100_0000u8, 0b1000_1010u8].bit_increment(&mut counters);
        let bools: Vec<_> = counters.iter().map(|&x| x != 0).collect();
        let avg = <[u16; 2]>::bitpack(&bools);

        let mut counters2 = vec![0u32; 32];
        avg.bit_increment(&mut counters2);
        println!("{:?}", counters);
        assert_eq!(counters, counters2)
    }
    #[test]
    fn patch_dist() {
        assert_eq!(123u8.hamming_distance(&123u8), 0);
        assert_eq!(0b1010_1000u8.hamming_distance(&0b1010_0111u8), 4);
        assert_eq!(
            [0b1111_0000u8, 0b1111_0000u8].hamming_distance(&[0b0000_1100u8, 0b1111_1111u8]),
            6 + 4
        );
    }
    #[test]
    fn unary() {
        assert_eq!(unary::to_3(128), 0b0000_0001u8);
        assert_eq!(unary::to_3(255), 0b0000_0111u8);
        assert_eq!(unary::to_7(255), 0b0111_1111u8);
        assert_eq!(unary::to_7(0), 0b0000_0000u8);
    }
    #[test]
    fn rgb_pack() {
        assert_eq!(unary::rgb_to_u14([255, 255, 255]), 0b0011_1111_1111_1111u16);
        assert_eq!(unary::rgb_to_u14([128, 128, 128]), 0b0000_0110_0011_0011u16);
        assert_eq!(unary::rgb_to_u14([0, 0, 0]), 0b0u16);
    }
}
