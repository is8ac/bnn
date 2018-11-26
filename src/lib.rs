extern crate rayon;

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
                bytes[color] = (((bits >> (color * 21)) & 0b111111111111111111111u64).count_ones() * 12) as u8; // 21 bit mask
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
                file.read_exact(&mut image_bytes).expect("can't read images");
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
        pub fn load_images_64chan_10(path: &String, size: usize) -> Vec<(u8, [[[u64; 1]; 32]; 32])> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open data");

            let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
            let mut label: [u8; 1] = [0; 1];
            let mut images: Vec<(u8, [[[u64; 1]; 32]; 32])> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut label).expect("can't read label");
                file.read_exact(&mut image_bytes).expect("can't read images");
                let mut image = [[[0u64; 1]; 32]; 32];
                for x in 0..32 {
                    for y in 0..32 {
                        image[x][y][0] = encode_unary_rgb([
                            image_bytes[(0 * 1024) + (y * 32) + x],
                            image_bytes[(1 * 1024) + (y * 32) + x],
                            image_bytes[(2 * 1024) + (y * 32) + x],
                        ]);
                    }
                }
                images.push((label[0], image));
            }
            return images;
        }
        pub fn load_images_64chan_100(path: &String, size: usize, fine: bool) -> Vec<(u8, [[[u64; 1]; 32]; 32])> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open data");

            let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
            let mut label: [u8; 2] = [0; 2];
            let mut images: Vec<(u8, [[[u64; 1]; 32]; 32])> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut label).expect("can't read label");
                file.read_exact(&mut image_bytes).expect("can't read images");
                let mut image = [[[0u64; 1]; 32]; 32];
                for x in 0..32 {
                    for y in 0..32 {
                        image[x][y][0] = encode_unary_rgb([
                            image_bytes[(0 * 1024) + (y * 32) + x],
                            image_bytes[(1 * 1024) + (y * 32) + x],
                            image_bytes[(2 * 1024) + (y * 32) + x],
                        ]);
                    }
                }
                images.push((label[fine as usize], image));
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
                file.read_exact(&mut images_bytes).expect("can't read images");
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
                file.read_exact(&mut images_bytes).expect("can't read images");
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
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image_words: [u64; 13] = [0; 13];
                for p in 0..784 {
                    let word_index = p / 64;
                    image_words[word_index] = image_words[word_index] | (((images_bytes[p] > 128) as u64) << p % 64);
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
                file.read_exact(&mut images_bytes).expect("can't read images");
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

pub trait Patch: Send + Sync + Sized {
    fn hamming_distance(&self, &Self) -> u32;
    fn bit_increment(&self, &mut [u32]);
    fn bit_len() -> usize;
    fn bitpack(&[bool]) -> Self;
    fn bit_or(&self, &Self) -> Self;
    fn flip_bit(&mut self, usize);
    fn get_bit(&self, usize) -> bool;
}

impl<A: Patch, B: Patch> Patch for (A, B) {
    fn hamming_distance(&self, other: &Self) -> u32 {
        self.0.hamming_distance(&other.0) + self.1.hamming_distance(&other.1)
    }
    fn bit_increment(&self, counters: &mut [u32]) {
        self.0.bit_increment(&mut counters[0..A::bit_len()]);
        self.1.bit_increment(&mut counters[A::bit_len()..]);
    }
    fn bit_len() -> usize {
        A::bit_len() + B::bit_len()
    }
    fn bitpack(bools: &[bool]) -> Self {
        if bools.len() != (A::bit_len() + B::bit_len()) {
            panic!("pair bitpack: counters is {:?}, should be {:?}", bools.len(), A::bit_len() + B::bit_len());
        }
        (A::bitpack(&bools[..A::bit_len()]), B::bitpack(&bools[A::bit_len()..]))
    }
    fn bit_or(&self, other: &Self) -> Self {
        (self.0.bit_or(&other.0), self.1.bit_or(&other.1))
    }
    fn flip_bit(&mut self, index: usize) {
        if index < A::bit_len() {
            self.0.flip_bit(index);
        } else {
            self.1.flip_bit(index - A::bit_len());
        }
    }
    fn get_bit(&self, index: usize) -> bool {
        if index < A::bit_len() {
            self.0.get_bit(index)
        } else {
            self.1.get_bit(index - A::bit_len())
        }
    }
}

macro_rules! primitive_patch {
    ($type:ty, $len:expr) => {
        impl Patch for $type {
            fn hamming_distance(&self, other: &$type) -> u32 {
                (self ^ other).count_ones()
            }
            fn bit_increment(&self, counters: &mut [u32]) {
                if counters.len() != $len {
                    panic!("primitive increment: counters is {:?}, should be {:?}", counters.len(), $len);
                }
                for i in 0..$len {
                    counters[i] += ((self >> i) & 0b1 as $type) as u32;
                }
            }
            fn bit_len() -> usize {
                $len
            }
            fn bitpack(bools: &[bool]) -> $type {
                if bools.len() != $len {
                    panic!("primitive bitpack: counters is {:?}, should be {:?}", bools.len(), $len);
                }
                let mut val = 0 as $type;
                for i in 0..$len {
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

primitive_patch!(u8, 8);
primitive_patch!(u16, 16);
primitive_patch!(u32, 32);
primitive_patch!(u64, 64);
primitive_patch!(u128, 128);

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
                if counters.len() != ($len * T::bit_len()) {
                    panic!("array increment: counters is {:?}, should be {:?}", counters.len(), $len * T::bit_len());
                }
                for i in 0..$len {
                    self[i].bit_increment(&mut counters[i * T::bit_len()..(i + 1) * T::bit_len()]);
                }
            }
            fn bit_len() -> usize {
                $len * T::bit_len()
            }
            fn bitpack(bools: &[bool]) -> [T; $len] {
                if bools.len() != ($len * T::bit_len()) {
                    panic!("array bitpack: bools is {:?}, should be {:?}", bools.len(), $len * T::bit_len());
                }
                let mut val = [T::default(); $len];
                for i in 0..$len {
                    val[i] = T::bitpack(&bools[i * T::bit_len()..(i + 1) * T::bit_len()]);
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
                self[index / T::bit_len()].flip_bit(index % T::bit_len());
            }
            fn get_bit(&self, index: usize) -> bool {
                self[index / T::bit_len()].get_bit(index % T::bit_len())
            }
        }
    };
}

array_patch!(1);
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

    // Takes patches split by class, and generates n_features_iters features for each.
    // Returns both a vec of features and a vec of unary activations.
    // Splits after each new feature.
    pub fn gen_hidden_features<T: Patch + Copy + Sync + Send>(
        train_inputs: &Vec<Vec<T>>,
        n_features_iters: usize,
        unary_size: usize,
        culling_threshold: usize,
    ) -> (Vec<T>, Vec<Vec<u32>>) {
        let flat_inputs: Vec<T> = train_inputs.iter().flatten().cloned().collect();
        let mut shards: Vec<Vec<Vec<T>>> = vec![train_inputs.clone().to_owned()];
        let mut features_vec: Vec<T> = vec![];
        let mut thresholds: Vec<Vec<u32>> = vec![];
        for i in 0..n_features_iters {
            for class in 0..train_inputs.len() {
                let base_point = gen_basepoint(&shards, class);
                features_vec.push(base_point);
                let (split_threshold, unary_thresholds) = vec_threshold(&flat_inputs, &base_point, unary_size);
                shards = split_labels_set_by_distance(&shards, &base_point, split_threshold, culling_threshold);
                thresholds.push(unary_thresholds);
                //println!("{:?} \t {:?}", shards.len(), class);
            }
        }
        (features_vec, thresholds)
    }
    // splits labels examples by class.
    pub fn split_by_label<T: Copy>(examples: &Vec<(usize, T)>, len: usize) -> Vec<Vec<T>> {
        let mut by_label: Vec<Vec<T>> = (0..len).map(|_| Vec::new()).collect();
        for (label, example) in examples {
            by_label[*label].push(*example);
        }
        let _: Vec<_> = by_label.iter_mut().map(|x| x.shrink_to_fit()).collect();
        by_label
    }

    pub fn apply_unary<T: Patch, O: Patch>(input: &T, features_vec: &Vec<T>, thresholds: &Vec<Vec<u32>>) -> O {
        let distances = bitvecmul::vbvm(&features_vec, &input);
        let mut bools = vec![false; O::bit_len()];
        for c in 0..distances.len() {
            for i in 0..thresholds[0].len() {
                bools[(c * thresholds[0].len()) + i] = distances[c] > thresholds[c][i];
            }
        }
        O::bitpack(&bools.as_slice())
    }

    pub fn gen_readout_features<T: Patch + Sync + Clone>(by_class: &Vec<Vec<T>>) -> Vec<T> {
        let num_examples: usize = by_class.iter().map(|x| x.len()).sum();

        (0..by_class.len())
            .map(|class| {
                let grads = grads_one_shard(by_class, class);
                let scaled_grads: Vec<f64> = grads.iter().map(|x| x / num_examples as f64).collect();
                //println!("{:?}", scaled_grads);
                grads_to_bits(&scaled_grads)
            }).collect()
    }

    // split_labels_set_by_filter takes examples and a split func. It returns the
    // examples is a vec of shards of labels of examples of patches.
    // It returns the same data but with double the shards and with each Vec of patches (very approximately) half the length.
    // split_fn is used to split each Vec of patches between two shards.
    pub fn split_labels_set_by_distance<T: Copy + Send + Sync + Patch>(examples: &Vec<Vec<Vec<T>>>, base_point: &T, threshold: u32, filter_thresh_len: usize) -> Vec<Vec<Vec<T>>> {
        examples
            .par_iter()
            .map(|by_label| {
                let pair: Vec<Vec<Vec<T>>> = vec![
                    by_label
                        .par_iter()
                        .map(|label_examples| label_examples.iter().filter(|x| x.hamming_distance(base_point) > threshold).cloned().collect())
                        .collect(),
                    by_label
                        .par_iter()
                        .map(|label_examples| label_examples.iter().filter(|x| x.hamming_distance(base_point) <= threshold).cloned().collect())
                        .collect(),
                ];
                pair
            }).flatten()
            .filter(|pair: &Vec<Vec<T>>| {
                let sum_len: usize = pair.iter().map(|x| x.len()).sum();
                sum_len > filter_thresh_len
            }).collect()
    }
    fn avg_bit_sums(len: usize, counts: &Vec<u32>) -> Vec<f64> {
        counts.iter().map(|&count| count as f64 / len as f64).collect()
    }

    fn count_bits<T: Patch + Sync>(patches: &Vec<T>) -> Vec<u32> {
        patches
            .par_iter()
            .fold(
                || vec![0u32; T::bit_len()],
                |mut counts, example| {
                    example.bit_increment(&mut counts);
                    counts
                },
            ).reduce(|| vec![0u32; T::bit_len()], |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect())
    }

    pub fn grads_one_shard<T: Patch + Sync>(by_class: &Vec<Vec<T>>, label: usize) -> Vec<f64> {
        let mut sum_bits: Vec<(usize, Vec<u32>)> = by_class.iter().map(|label_patches| (label_patches.len(), count_bits(&label_patches))).collect();

        let (target_len, target_sums) = sum_bits.remove(label);
        let (other_len, other_sums) = sum_bits.iter().fold((0usize, vec![0u32; T::bit_len()]), |(a_len, a_vals), (b_len, b_vals)| {
            (a_len + b_len, a_vals.iter().zip(b_vals.iter()).map(|(x, y)| x + y).collect())
        });
        let other_avg = avg_bit_sums(other_len, &other_sums);
        let target_avg = avg_bit_sums(target_len, &target_sums);

        let grads: Vec<f64> = other_avg.iter().zip(target_avg.iter()).map(|(a, b)| (a - b) * (target_len.min(other_len) as f64)).collect();
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
            }).sum();

        let grads: Vec<f64> = shards
            .par_iter()
            .filter(|x| {
                let class_len = x[label].len();
                if class_len > 0 {
                    let total_len: usize = x.iter().map(|y| y.len()).sum();
                    return (total_len - class_len) > 0;
                }
                class_len > 0
            }).map(|shard| grads_one_shard(&shard, label).iter().map(|&grad| grad * len as f64).collect())
            .fold(
                || vec![0f64; T::bit_len()],
                |acc, grads: Vec<f64>| acc.iter().zip(grads.iter()).map(|(a, b)| a + b).collect(),
            ).reduce(|| vec![0f64; T::bit_len()], |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect());

        let sign_bits: Vec<bool> = grads.iter().map(|x| *x > 0f64).collect();

        T::bitpack(&sign_bits)
    }
    pub fn gen_threshold<T: Patch + Sync>(patches: &Vec<T>, base_point: &T) -> u32 {
        let mut bit_distances: Vec<u32> = patches.par_iter().map(|y| y.hamming_distance(&base_point)).collect();
        bit_distances.par_sort();
        bit_distances[bit_distances.len() / 2]
    }

    pub fn vec_threshold<T: Patch + Sync>(patches: &Vec<T>, base_point: &T, n: usize) -> (u32, Vec<u32>) {
        let mut bit_distances: Vec<u32> = patches.par_iter().map(|y| y.hamming_distance(&base_point)).collect();
        bit_distances.par_sort();
        let mut split_points = vec![0u32; n];
        for i in 0..n {
            split_points[i] = bit_distances[(bit_distances.len() / (n + 1)) * (i + 1)];
        }
        (bit_distances[bit_distances.len() / 2], split_points)
    }

}

#[macro_use]
pub mod layers {
    use super::featuregen;
    use super::Patch;
    use rayon::prelude::*;
    use std::marker::PhantomData;
    use std::mem::transmute;

    pub trait VecApply<I: Send + Sync, O: Send + Sync> {
        fn vec_apply(&self, &Vec<(usize, I)>) -> Vec<(usize, O)>;
        fn vec_update(&self, &Vec<(usize, I)>, &mut Vec<(usize, O)>, usize);
    }

    impl<T: Apply<I, O>, I: Send + Sync + Patch + Copy, O: Send + Sync> VecApply<I, O> for T {
        fn vec_apply(&self, inputs: &Vec<(usize, I)>) -> Vec<(usize, O)> {
            inputs.par_iter().map(|(class, input)| (*class, self.apply(input))).collect()
        }
        fn vec_update(&self, inputs: &Vec<(usize, I)>, targets: &mut Vec<(usize, O)>, index: usize) {
            let _: Vec<_> = targets
                .par_iter_mut()
                .zip(inputs.par_iter())
                .map(|(x, input)| self.update(&input.1, &mut x.1, index))
                .collect();
        }
    }

    pub trait Apply<I: Send + Sync, O: Send + Sync>
    where
        Self: Sync,
    {
        fn apply(&self, &I) -> O;
        fn update(&self, &I, &mut O, usize);
    }

    macro_rules! primitive_apply {
        ($type:ty, $len:expr) => {
            impl<I: Patch + Default + Copy> Apply<I, $type> for [(I, u32); $len] {
                fn apply(&self, input: &I) -> $type {
                    let mut val = 0 as $type;
                    for i in 0..$len {
                        val = val | (((self[i].0.hamming_distance(&input) > self[i].1) as $type) << i);
                    }
                    val
                }
                fn update(&self, input: &I, target: &mut $type, index: usize) {
                    *target &= !(1 << index); // unset the bit
                    *target |= ((self[index].0.hamming_distance(&input) > self[index].1) as $type) << index; // set it to the updated value.
                }
            }
        };
    }

    primitive_apply!(u8, 8);
    primitive_apply!(u16, 16);
    primitive_apply!(u32, 32);
    primitive_apply!(u64, 64);
    primitive_apply!(u128, 128);

    macro_rules! primitive_apply_simplified_input {
        ($type:ty, $len:expr, $in_type:ty, $weights_type:ty) => {
            impl<W: Apply<$weights_type, $type>> Apply<$in_type, $type> for W
            where
                W: Apply<$weights_type, $type>,
            {
                fn apply(&self, input: &$in_type) -> $type {
                    let input = unsafe { transmute::<$in_type, $weights_type>(*input) };
                    self.apply(&input)
                }
                fn update(&self, input: &$in_type, target: &mut $type, index: usize) {
                    let input = unsafe { transmute::<$in_type, $weights_type>(*input) };
                    self.update(&input, target, index);
                }
            }
        };
    }

    macro_rules! primitive_apply_simplified_input_all {
        ($in_type:ty, $weights_type:ty) => {
            primitive_apply_simplified_input!(u8, 8, $in_type, $weights_type);
            primitive_apply_simplified_input!(u16, 16, $in_type, $weights_type);
            primitive_apply_simplified_input!(u32, 32, $in_type, $weights_type);
            primitive_apply_simplified_input!(u64, 64, $in_type, $weights_type);
            primitive_apply_simplified_input!(u128, 128, $in_type, $weights_type);
        };
    }
    primitive_apply_simplified_input_all!([u8; 8], u64);
    primitive_apply_simplified_input_all!([u16; 8], u128);
    primitive_apply_simplified_input_all!([u32; 8], [u128; 2]);
    primitive_apply_simplified_input_all!([u64; 8], [u128; 4]);

    macro_rules! primitive_apply_unary {
        ($type:ty, $len:expr, $unary_bits:expr) => {
            impl<I: Patch + Default + Copy> Apply<I, [$type; $unary_bits]> for [(I, [u32; $unary_bits]); $len] {
                fn apply(&self, input: &I) -> [$type; $unary_bits] {
                    let mut val = [0 as $type; $unary_bits];
                    for i in 0..$len {
                        let dist = self[i].0.hamming_distance(&input);
                        for b in 0..$unary_bits {
                            val[b] = val[b] | (((dist > self[i].1[b]) as $type) << i);
                        }
                    }
                    val
                }
                fn update(&self, input: &I, target: &mut [$type; $unary_bits], index: usize) {
                    let dist = self[index].0.hamming_distance(&input);
                    for b in 0..$unary_bits {
                        target[b] &= !(1 << index); // unset the bit
                        target[b] |= ((dist > self[index].1[b]) as $type) << index; // set it to the updated value.
                    }
                }
            }
        };
    }

    macro_rules! primitive_apply_n_bit_types {
        ($unary_bits:expr) => {
            primitive_apply_unary!(u8, 8, $unary_bits);
            primitive_apply_unary!(u16, 16, $unary_bits);
            primitive_apply_unary!(u32, 32, $unary_bits);
            primitive_apply_unary!(u64, 64, $unary_bits);
            primitive_apply_unary!(u128, 128, $unary_bits);
        };
    }

    primitive_apply_n_bit_types!(2);
    primitive_apply_n_bit_types!(3);
    primitive_apply_n_bit_types!(4);

    macro_rules! primitive_apply_unary_simplify {
        ($type:ty, $len:expr, $unary_bits:expr, $out_type:ty) => {
            impl<I: Patch + Default + Copy> Apply<I, $out_type> for [(I, [u32; $unary_bits]); $len] {
                fn apply(&self, input: &I) -> $out_type {
                    let mut val = [0 as $type; $unary_bits];
                    for i in 0..$len {
                        let dist = self[i].0.hamming_distance(&input);
                        for b in 0..$unary_bits {
                            val[b] = val[b] | (((dist > self[i].1[b]) as $type) << i);
                        }
                    }
                    unsafe { transmute(val) }
                }
                fn update(&self, input: &I, target: &mut $out_type, index: usize) {
                    let target = unsafe { transmute::<&mut $out_type, &mut [$type; $unary_bits]>(target) };
                    let dist = self[index].0.hamming_distance(&input);
                    for b in 0..$unary_bits {
                        target[b] &= !(1 << index); // unset the bit
                        target[b] |= ((dist > self[index].1[b]) as $type) << index; // set it to the updated value.
                    }
                }
            }
        };
    }

    primitive_apply_unary_simplify!(u8, 8, 2, u16);
    primitive_apply_unary_simplify!(u8, 8, 4, u32);
    primitive_apply_unary_simplify!(u16, 16, 2, u32);
    primitive_apply_unary_simplify!(u16, 16, 4, u64);
    primitive_apply_unary_simplify!(u32, 32, 2, u64);
    primitive_apply_unary_simplify!(u32, 32, 4, u128);
    primitive_apply_unary_simplify!(u64, 64, 2, u128);
    primitive_apply_unary_simplify!(u64, 64, 4, [u128; 2]);

    macro_rules! patch_apply_trait_8notched {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy + Send + Sync, OP: Copy + Default + Send + Sync, W: Apply<[IP; 8], OP>> Apply<[[IP; $y_size]; $x_size], [[OP; $y_size]; $x_size]> for W {
                fn apply(&self, input: &[[IP; $y_size]; $x_size]) -> [[OP; $y_size]; $x_size] {
                    let mut output = [[OP::default(); $y_size]; $x_size];
                    for x in 0..($x_size - 2) {
                        for y in 0..($y_size - 2) {
                            let patch = [
                                input[x + 1][y + 0],
                                input[x + 2][y + 0],
                                input[x + 0][y + 1],
                                input[x + 1][y + 1],
                                input[x + 2][y + 1],
                                input[x + 0][y + 2],
                                input[x + 1][y + 2],
                                input[x + 2][y + 2],
                            ];
                            output[x][y] = self.apply(&patch);
                        }
                    }
                    output
                }
                fn update(&self, input: &[[IP; $y_size]; $x_size], target: &mut [[OP; $y_size]; $x_size], index: usize) {
                    for x in 0..($x_size - 2) {
                        for y in 0..($y_size - 2) {
                            let patch = [
                                input[x + 1][y + 0],
                                input[x + 2][y + 0],
                                input[x + 0][y + 1],
                                input[x + 1][y + 1],
                                input[x + 2][y + 1],
                                input[x + 0][y + 2],
                                input[x + 1][y + 2],
                                input[x + 2][y + 2],
                            ];
                            self.update(&patch, &mut target[x][y], index);
                        }
                    }
                }
            }
        };
    }

    patch_apply_trait_8notched!(32, 32);
    patch_apply_trait_8notched!(28, 28);
    patch_apply_trait_8notched!(14, 14);
    patch_apply_trait_8notched!(16, 16);
    patch_apply_trait_8notched!(8, 8);

    macro_rules! pool_or_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<T: Patch + Default + Copy> Apply<[[T; $y_size]; $x_size], [[T; $y_size / 2]; $x_size / 2]> for () {
                fn apply(&self, input: &[[T; $y_size]; $x_size]) -> [[T; $y_size / 2]; $x_size / 2] {
                    let mut output = [[T::default(); $y_size / 2]; $x_size / 2];
                    for x in 0..$x_size / 2 {
                        let x_base = x * 2;
                        for y in 0..$y_size / 2 {
                            let y_base = y * 2;
                            output[x][y] = input[x_base + 0][y_base + 0]
                                .bit_or(&input[x_base + 0][y_base + 1])
                                .bit_or(&input[x_base + 1][y_base + 0])
                                .bit_or(&input[x_base + 1][y_base + 1]);
                        }
                    }
                    output
                }
                fn update(&self, input: &[[T; $y_size]; $x_size], output: &mut [[T; $y_size / 2]; $x_size / 2], _index: usize) {
                    *output = self.apply(input);
                }
            }
        };
    }

    pool_or_trait!(32, 32);
    pool_or_trait!(28, 28);
    pool_or_trait!(16, 16);

    pub trait NewFromSplit<I: Patch> {
        fn new_from_split(&Vec<(usize, I)>) -> Self;
    }

    macro_rules! primitive_new_from_split {
        ($type:ty, $len:expr) => {
            impl<I: Patch + Copy + Default> NewFromSplit<I> for [(I, u32); $len] {
                fn new_from_split(examples: &Vec<(usize, I)>) -> Self {
                    let mut weights = [(I::default(), 0u32); $len];
                    let train_inputs = featuregen::split_by_label(&examples, 10);

                    let flat_inputs: Vec<I> = examples.iter().map(|(_, input)| *input).collect();
                    let mut shards: Vec<Vec<Vec<I>>> = vec![train_inputs.clone().to_owned()];
                    for i in 0..$len {
                        let class = i % 10;
                        let base_point = featuregen::gen_basepoint(&shards, class);
                        let mut bit_distances: Vec<u32> = flat_inputs.par_iter().map(|y| y.hamming_distance(&base_point)).collect();
                        bit_distances.par_sort();
                        let threshold = bit_distances[bit_distances.len() / 2];
                        weights[i] = (base_point, threshold);

                        shards = featuregen::split_labels_set_by_distance(&shards, &base_point, threshold, 2);
                    }
                    weights
                }
            }
        };
    }
    primitive_new_from_split!(u8, 8);
    primitive_new_from_split!(u16, 16);
    primitive_new_from_split!(u32, 32);
    primitive_new_from_split!(u64, 64);
    primitive_new_from_split!(u128, 128);

    macro_rules! primitive_new_from_split_unary {
        ($type:ty, $len:expr, $unary_bits:expr) => {
            impl<I: Patch + Copy + Default> NewFromSplit<I> for [(I, [u32; $unary_bits]); $len] {
                fn new_from_split(examples: &Vec<(usize, I)>) -> Self {
                    let mut weights = [(I::default(), [0u32; $unary_bits]); $len];
                    let train_inputs = featuregen::split_by_label(&examples, 10);

                    let flat_inputs: Vec<I> = examples.iter().map(|(_, input)| *input).collect();
                    let mut shards: Vec<Vec<Vec<I>>> = vec![train_inputs.clone().to_owned()];
                    for i in 0..$len {
                        let class = i % 10;
                        let base_point = featuregen::gen_basepoint(&shards, class);
                        weights[i].0 = base_point;
                        let mut bit_distances: Vec<u32> = flat_inputs.par_iter().map(|y| y.hamming_distance(&base_point)).collect();
                        bit_distances.par_sort();
                        let threshold = bit_distances[bit_distances.len() / 2];
                        for b in 0..$unary_bits {
                            weights[i].1[b] = bit_distances[bit_distances.len() / ($unary_bits + 1)];
                        }
                        shards = featuregen::split_labels_set_by_distance(&shards, &base_point, threshold, 2);
                    }
                    weights
                }
            }
        };
    }

    primitive_new_from_split_unary!(u8, 8, 2);
    primitive_new_from_split_unary!(u16, 16, 2);
    primitive_new_from_split_unary!(u32, 32, 2);
    primitive_new_from_split_unary!(u64, 64, 2);
    primitive_new_from_split_unary!(u128, 128, 2);

    primitive_new_from_split_unary!(u8, 8, 4);
    primitive_new_from_split_unary!(u16, 16, 4);
    primitive_new_from_split_unary!(u32, 32, 4);
    primitive_new_from_split_unary!(u64, 64, 4);
    primitive_new_from_split_unary!(u128, 128, 4);

    pub trait Mutate {
        fn mutate(&mut self, output_index: usize, input_index: usize);
        fn output_len() -> usize;
        fn input_len() -> usize;
    }

    macro_rules! primitive_mutate_trait {
        ($len:expr) => {
            impl<I: Patch, T> Mutate for [(I, T); $len] {
                fn mutate(&mut self, output_index: usize, input_index: usize) {
                    self[output_index].0.flip_bit(input_index);
                }
                fn output_len() -> usize {
                    $len
                }
                fn input_len() -> usize {
                    I::bit_len()
                }
            }
        };
    }

    primitive_mutate_trait!(8);
    primitive_mutate_trait!(16);
    primitive_mutate_trait!(32);
    primitive_mutate_trait!(64);
    primitive_mutate_trait!(128);

    pub trait Optimize<I, O> {
        fn optimize<H: ObjectiveHead<O>>(&mut self, &mut H, &Vec<(usize, I)>, usize);
    }
    impl<I: Sync + Patch + Send + Copy, O: Sync + Send, W: Mutate + VecApply<I, O>> Optimize<I, O> for W
    where
        W: Apply<I, O>,
    {
        fn optimize<H: ObjectiveHead<O>>(&mut self, head: &mut H, examples: &Vec<(usize, I)>, update_freq: usize) {
            let mut iter = 0;
            for o in 0..W::output_len() {
                //println!("o: {:?}", o);
                let mut cache: Vec<(usize, O)> = (*self).vec_apply(examples);
                head.optimize(&cache, update_freq);
                head.optimize(&cache, update_freq);
                head.optimize(&cache, update_freq);
                let mut acc = head.acc(&cache);
                //println!("acc: {:?}", acc);
                for b in 0..W::input_len() {
                    if iter % update_freq == 0 {
                        head.optimize(&cache, update_freq);
                        acc = head.acc(&cache);
                    }
                    self.mutate(o, b);
                    (*self).vec_update(&examples, &mut cache, o);
                    let new_acc = head.acc(&cache);
                    if new_acc > acc {
                        acc = new_acc;
                        //println!("{:?}", acc);
                        iter += 1;
                    } else {
                        // revert
                        self.mutate(o, b);
                    }
                }
            }
        }
    }

    pub trait ObjectiveHead<I> {
        fn acc(&self, &Vec<(usize, I)>) -> f64;
        fn optimize(&mut self, &Vec<(usize, I)>, usize);
        fn new_from_split(&Vec<(usize, I)>) -> Self;
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
                }).sum();
            sum_correct as f64 / examples.len() as f64
        }
        fn optimize(&mut self, examples: &Vec<(usize, I)>, _update_freq: usize) {
            for mut_class in 0..10 {
                let mut activation_diffs: Vec<(I, i32, bool)> = examples
                    .par_iter()
                    .map(|(targ_class, input)| {
                        let mut activations: Vec<i32> = self.iter().map(|base_point| base_point.hamming_distance(&input) as i32).collect();

                        let targ_act = activations[*targ_class]; // the activation for target class of this example.
                        let mut_act = activations[mut_class]; // the activation which we are mutating.
                        activations[*targ_class] = -1;
                        activations[mut_class] = -1;
                        let max_other_activations = activations.iter().max().unwrap(); // the max activation of all the classes not in the target class or mut class.
                        let diff = {
                            if *targ_class == mut_class {
                                mut_act - max_other_activations
                            } else {
                                mut_act - targ_act
                            }
                        };
                        (input, diff, *targ_class == mut_class, (targ_act > *max_other_activations) | (*targ_class == mut_class)) // diff betwene the activation of the
                    }).filter(|(_, _, _, keep)| *keep)
                    .map(|(input, diff, sign, _)| (*input, diff, sign))
                    .collect();

                // note that this sum correct is not the true acc, it is working on the subset that can be made correct or incorrect by this activation.
                let mut sum_correct: i64 = activation_diffs
                    .par_iter()
                    .map(|(_, diff, sign)| {
                        if *sign {
                            // if we want the mut_act to be bigger,
                            *diff > 0 // count those which are bigger,
                        } else {
                            // otherwise,
                            *diff < 0 // count those which are smaller.
                        }
                    } as i64).sum();
                for b in 0..I::bit_len() {
                    // the new weights bit
                    let new_weights_bit = !self[mut_class].get_bit(b);
                    // if we were to flip the bit of the weights,
                    let new_sum_correct: i64 = activation_diffs
                        .par_iter()
                        .map(|(input, diff, sign)| {
                            // new diff is the diff after flipping the weights bit
                            let new_diff = {
                                if input.get_bit(b) ^ new_weights_bit {
                                    // flipping the bit would make mut_act larger
                                    diff + 2
                                } else {
                                    diff - 2
                                }
                            };
                            // do we want mut_act to be smaller or larger?
                            // same as this statement:
                            //(if *sign { new_diff > 0 } else { new_diff < 0 }) as i64
                            ((*sign ^ (new_diff < 0)) & (new_diff != 0)) as i64
                        }).sum();
                    if new_sum_correct > sum_correct {
                        sum_correct = new_sum_correct;
                        // actually flip the bit
                        self[mut_class].flip_bit(b);
                        // now update each
                        activation_diffs
                            .par_iter_mut()
                            .map(|i| {
                                if i.0.get_bit(b) ^ new_weights_bit {
                                    i.1 += 2;
                                } else {
                                    i.1 -= 2;
                                }
                            }).collect::<Vec<_>>();
                    }
                }
            }
        }
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

    struct Layer<I: Sync + Send, O: Sync + Send, L: VecApply<I, O>, H: ObjectiveHead<O>> {
        input: PhantomData<I>,
        output: PhantomData<O>,
        data: L,
        head: H,
    }

    impl<I: Sync + Send + Patch + Copy, O: Sync + Send, L: VecApply<I, O> + VecApply<I, O> + Optimize<I, O> + NewFromSplit<I>, H: ObjectiveHead<O>> ObjectiveHead<I>
        for Layer<I, O, L, H>
    {
        fn acc(&self, examples: &Vec<(usize, I)>) -> f64 {
            let output_examples = self.data.vec_apply(&examples);
            self.head.acc(&output_examples)
        }
        fn optimize(&mut self, examples: &Vec<(usize, I)>, update_freq: usize) {
            self.data.optimize(&mut self.head, examples, update_freq);
        }
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

    trait ArrayPack<I> {
        fn from_vec_padded(&Vec<I>) -> Self;
        fn from_vec_cropped(&Vec<I>) -> Self;
    }

    macro_rules! array_pack_trait {
        ($len:expr) => {
            impl<I: Default + Copy> ArrayPack<I> for [I; $len] {
                fn from_vec_padded(input: &Vec<I>) -> [I; $len] {
                    if input.len() > $len {
                        panic!(
                            "can't fit a vec of len {:} into an array of len {:}. Consider making the vec shorter or use a longer array.",
                            input.len(),
                            $len
                        );
                    }
                    let mut output = [I::default(); $len];
                    for (i, elem) in input.iter().enumerate() {
                        output[i] = *elem;
                    }
                    output
                }
                fn from_vec_cropped(input: &Vec<I>) -> [I; $len] {
                    if input.len() < $len {
                        panic!(
                            "can't get an array of len {:} from an array of len {:}. Consider making the vec longer or use a smaller array.",
                            $len,
                            input.len()
                        );
                    }
                    let mut output = [I::default(); $len];
                    for i in 0..$len {
                        output[i] = input[i];
                    }
                    output
                }
            }
        };
    }
    array_pack_trait!(8);
    array_pack_trait!(16);
    array_pack_trait!(32);
    array_pack_trait!(64);
    array_pack_trait!(128);

    pub trait Pool2x2<I, IP: Patch, O> {
        fn or_pool_2x2(&self) -> O;
    }

    macro_rules! or_pool2x2_trait {
        ($x_len:expr, $y_len:expr) => {
            impl<P: Patch + Copy + Default> Pool2x2<[[P; $y_len]; $x_len], P, [[P; ($y_len / 2)]; ($x_len / 2)]> for [[P; $y_len]; $x_len] {
                fn or_pool_2x2(&self) -> [[P; ($y_len / 2)]; ($x_len / 2)] {
                    let mut pooled = [[P::default(); $y_len / 2]; $x_len / 2];
                    for x in 0..($x_len / 2) {
                        let x_base = x * 2;
                        for y in 0..($y_len / 2) {
                            let y_base = y * 2;
                            pooled[x][y] = self[x_base + 0][y_base + 0]
                                .bit_or(&self[x_base + 0][y_base + 1])
                                .bit_or(&self[x_base + 1][y_base + 0])
                                .bit_or(&self[x_base + 1][y_base + 1]);
                        }
                    }
                    pooled
                }
            }
        };
    }

    or_pool2x2_trait!(4, 4);
    or_pool2x2_trait!(8, 8);
    or_pool2x2_trait!(16, 16);
    or_pool2x2_trait!(32, 32);

    pub trait SimplifyBits<T> {
        fn simplify(self) -> T;
    }

    macro_rules! simplify_bits_trait {
        ($in_type:ty, $out_type:ty) => {
            impl SimplifyBits<$out_type> for $in_type {
                fn simplify(self) -> $out_type {
                    unsafe { transmute::<$in_type, $out_type>(self) }
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
                pub fn $name<T: super::Patch>(weights: &[(T, u32); $len], input: &T) -> [u32; $len] {
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
            weights.iter().map(|signs| input.hamming_distance(&signs)).collect()
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
            to_4(pixels[0]) as u16 | ((to_5(pixels[1]) as u16) << 4) | ((to_5(pixels[2]) as u16) << 9)
        }
    }

    // From a 2D image, extract various different patches with stride of 1.
    pub trait ExtractPatches<O> {
        fn extract_patches(&self) -> Vec<O>;
    }

    macro_rules! extract_patch_8_simplify_trait {
        ($in_type:ty, $out_type:ty) => {
            impl<II: ExtractPatches<$in_type>> ExtractPatches<$out_type> for II {
                fn extract_patches(&self) -> Vec<$out_type> {
                    self.extract_patches()
                        .iter()
                        .map(|patch| unsafe { transmute::<$in_type, $out_type>(*patch) })
                        .collect()
                }
            }
        };
    }

    extract_patch_8_simplify_trait!([u8; 8], u64);
    extract_patch_8_simplify_trait!([u16; 8], u128);

    // 3x3 flattened to [T; 9] array.
    macro_rules! extract_patch_9_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy> ExtractPatches<[IP; 9]> for [[IP; $y_size]; $x_size] {
                fn extract_patches(&self) -> Vec<[IP; 9]> {
                    let mut patches = Vec::with_capacity(($x_size - 2) * ($y_size - 2));
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            patches.push([
                                self[x + 0][y + 0],
                                self[x + 1][y + 0],
                                self[x + 2][y + 0],
                                self[x + 0][y + 1],
                                self[x + 1][y + 1],
                                self[x + 2][y + 1],
                                self[x + 0][y + 2],
                                self[x + 1][y + 2],
                                self[x + 2][y + 2],
                            ]);
                        }
                    }
                    patches
                }
            }
        };
    }
    extract_patch_9_trait!(32, 32);
    extract_patch_9_trait!(28, 28);
    extract_patch_9_trait!(16, 16);

    // 3x3 in [[T; 3]; 3]
    macro_rules! extract_patch_3x3_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy> ExtractPatches<[[IP; 3]; 3]> for [[IP; $y_size]; $x_size] {
                fn extract_patches(&self) -> Vec<[[IP; 3]; 3]> {
                    let mut patches = Vec::with_capacity(($x_size - 2) * ($y_size - 2));
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            patches.push([
                                [self[x + 0][y + 0], self[x + 1][y + 0], self[x + 2][y + 0]],
                                [self[x + 0][y + 1], self[x + 1][y + 1], self[x + 2][y + 1]],
                                [self[x + 0][y + 2], self[x + 1][y + 2], self[x + 2][y + 2]],
                            ]);
                        }
                    }
                    patches
                }
            }
        };
    }

    extract_patch_3x3_trait!(32, 32);
    extract_patch_3x3_trait!(28, 28);
    extract_patch_3x3_trait!(16, 16);

    // 3x3 with notch, flattened to [T; 8]
    macro_rules! extract_patch_8_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy> ExtractPatches<[IP; 8]> for [[IP; $y_size]; $x_size] {
                fn extract_patches(&self) -> Vec<[IP; 8]> {
                    let mut patches = Vec::with_capacity(($y_size - 2) * ($x_size - 2));
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            patches.push([
                                self[x + 1][y + 0],
                                self[x + 2][y + 0],
                                self[x + 0][y + 1],
                                self[x + 1][y + 1],
                                self[x + 2][y + 1],
                                self[x + 0][y + 2],
                                self[x + 1][y + 2],
                                self[x + 2][y + 2],
                            ]);
                        }
                    }
                    patches
                }
            }
        };
    }

    extract_patch_8_trait!(32, 32);
    extract_patch_8_trait!(28, 28);
    extract_patch_8_trait!(16, 16);

    macro_rules! extract_patch_vec_apply_8_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Patch + Sync + Send + Copy> VecApply<[[IP; $y_size]; $x_size], [IP; 8]> for () {
                fn vec_apply(&self, examples: &Vec<(usize, [[IP; $y_size]; $x_size])>) -> Vec<(usize, [IP; 8])> {
                    examples
                        .iter()
                        .map(|(class, image)| image.extract_patches().iter().map(|patch| (*class, *patch)).collect::<Vec<(usize, [IP; 8])>>())
                        .flatten()
                        .collect()
                }
                fn vec_update(&self, inputs: &Vec<(usize, [[IP; $y_size]; $x_size])>, targets: &mut Vec<(usize, [IP; 8])>, _index: usize) {
                    *targets = self.vec_apply(inputs);
                }
            }
        };
    }

    extract_patch_vec_apply_8_trait!(32, 32);
    extract_patch_vec_apply_8_trait!(28, 28);
    extract_patch_vec_apply_8_trait!(16, 16);

    // One pixel, just T
    macro_rules! extract_patch_pixel_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy> ExtractPatches<IP> for [[IP; $y_size]; $x_size] {
                fn extract_patches(&self) -> Vec<IP> {
                    let mut patches = Vec::with_capacity($x_size * $y_size);
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            patches.push(self[x][y]);
                        }
                    }
                    patches
                }
            }
        };
    }

    extract_patch_pixel_trait!(32, 32);
    extract_patch_pixel_trait!(28, 28);
    extract_patch_pixel_trait!(16, 16);

    // 2x2 with stride of 2.
    impl<IP: Copy> ExtractPatches<[IP; 4]> for [[IP; 32]; 32] {
        fn extract_patches(&self) -> Vec<[IP; 4]> {
            let mut patches = Vec::with_capacity(32 * 32);
            for x in 0..32 / 2 {
                let x_base = x * 2;
                for y in 0..32 / 2 {
                    let y_base = y * 2;
                    patches.push([
                        self[x_base + 0][y_base + 0],
                        self[x_base + 0][y_base + 1],
                        self[x_base + 1][y_base + 0],
                        self[x_base + 1][y_base + 1],
                    ]);
                }
            }
            patches
        }
    }

    pub trait PatchMap<IP, IA, O, OP> {
        fn patch_map(&self, &Fn(&IA, &mut OP)) -> O;
    }

    macro_rules! patch_map_trait_pixel {
        ($x_size:expr, $y_size:expr) => {
            impl<IP, OP: Copy + Default> PatchMap<IP, IP, [[OP; $y_size]; $x_size], OP> for [[IP; $y_size]; $x_size] {
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

    macro_rules! patch_map_trait_notched {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy, OP: Copy + Default> PatchMap<IP, [IP; 8], [[OP; $y_size]; $x_size], OP> for [[IP; $y_size]; $x_size] {
                fn patch_map(&self, map_fn: &Fn(&[IP; 8], &mut OP)) -> [[OP; $y_size]; $x_size] {
                    let mut output = [[OP::default(); $y_size]; $x_size];
                    for x in 0..($x_size - 2) {
                        for y in 0..($y_size - 2) {
                            let patch = [
                                self[x + 1][y + 0],
                                self[x + 2][y + 0],
                                self[x + 0][y + 1],
                                self[x + 1][y + 1],
                                self[x + 2][y + 1],
                                self[x + 0][y + 2],
                                self[x + 1][y + 2],
                                self[x + 2][y + 2],
                            ];
                            map_fn(&patch, &mut output[x][y]);
                        }
                    }
                    output
                }
            }
        };
    }

    patch_map_trait_notched!(32, 32);
    patch_map_trait_notched!(28, 28);
    patch_map_trait_notched!(16, 16);

    macro_rules! patch_map_trait_3x3 {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy, OP: Copy + Default> PatchMap<IP, [IP; 9], [[OP; $y_size]; $x_size], OP> for [[IP; $y_size]; $x_size] {
                fn patch_map(&self, map_fn: &Fn(&[IP; 9], &mut OP)) -> [[OP; $y_size]; $x_size] {
                    let mut output = [[OP::default(); $y_size]; $x_size];
                    for x in 0..($x_size - 2) {
                        for y in 0..($y_size - 2) {
                            let patch = [
                                self[x + 0][y + 0],
                                self[x + 1][y + 0],
                                self[x + 2][y + 0],
                                self[x + 0][y + 1],
                                self[x + 1][y + 1],
                                self[x + 2][y + 1],
                                self[x + 0][y + 2],
                                self[x + 1][y + 2],
                                self[x + 2][y + 2],
                            ];
                            map_fn(&patch, &mut output[x][y]);
                        }
                    }
                    output
                }
            }
        };
    }

    patch_map_trait_3x3!(32, 32);
    patch_map_trait_3x3!(28, 28);
    patch_map_trait_3x3!(16, 16);

    macro_rules! patch_map_trait_2x2_pool {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy, OP: Copy + Default> PatchMap<IP, [IP; 4], [[OP; $y_size / 2]; $x_size / 2], OP> for [[IP; $y_size]; $x_size] {
                fn patch_map(&self, map_fn: &Fn(&[IP; 4], &mut OP)) -> [[OP; $y_size / 2]; $x_size / 2] {
                    let mut output = [[OP::default(); $y_size / 2]; $x_size / 2];
                    for x in 0..$x_size / 2 {
                        let x_base = x * 2;
                        for y in 0..$y_size / 2 {
                            let y_base = y * 2;
                            let patch = [
                                self[x_base + 0][y_base + 0],
                                self[x_base + 0][y_base + 1],
                                self[x_base + 1][y_base + 0],
                                self[x_base + 1][y_base + 1],
                            ];
                            map_fn(&patch, &mut output[x][y]);
                        }
                    }
                    output
                }
            }
        };
    }

    patch_map_trait_2x2_pool!(32, 32);
    patch_map_trait_2x2_pool!(28, 28);
    patch_map_trait_2x2_pool!(16, 16);

}

#[cfg(test)]
mod tests {
    use super::layers::{unary, Apply, ExtractPatches};
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
        assert_eq!([0b1111_0000u8, 0b1111_0000u8].hamming_distance(&[0b0000_1100u8, 0b1111_1111u8]), 6 + 4);
    }
    #[test]
    fn unary() {
        assert_eq!(unary::to_3(128), 0b0000_0001u8);
        assert_eq!(unary::to_3(255), 0b0000_0111u8);
        assert_eq!(unary::to_7(255), 0b0111_1111u8);
        assert_eq!(unary::to_7(0), 0b0000_0000u8);
    }
    #[test]
    fn extract_patches() {
        let value = 0b1011_0111u8;
        let layer = [[value; 28]; 28];
        let patches: Vec<[u8; 8]> = layer.extract_patches();
        assert_eq!(patches.len(), 676);
        assert_eq!(patches[0], [value; 8])
    }
    #[test]
    fn rgb_pack() {
        assert_eq!(unary::rgb_to_u14([255, 255, 255]), 0b0011_1111_1111_1111u16);
        assert_eq!(unary::rgb_to_u14([128, 128, 128]), 0b0000_0110_0011_0011u16);
        assert_eq!(unary::rgb_to_u14([0, 0, 0]), 0b0u16);
    }
    #[test]
    fn test_unary_apply() {
        let mut weights = [
            (0b0100110u16, [0u32, 3, 11, 15]),
            (!0b0u16, [0u32, 3, 11, 15]),
            (0b0u16, [0u32, 3, 11, 15]),
            (!0b0u16, [0u32, 3, 11, 15]),
            (!0b0u16, [0u32, 3, 11, 15]),
            (!0b0u16, [0u32, 3, 11, 15]),
            (0b0u16, [0u32, 3, 11, 15]),
            (!0b0u16, [0u32, 3, 11, 15]),
        ];
        let input = 0b0u16;
        let output: [u8; 4] = weights.apply(&input);
        println!("{:08b} {:08b} {:08b} {:08b}", output[0], output[1], output[2], output[3]);
        let mut output: u32 = weights.apply(&input);
        println!("{:032b}", output);

        weights[1] = (0u16, [0u32, 0, 11, 16]);
        weights.update(&input, &mut output, 1);
        println!("{:032b}", output);
        let orig_output: u32 = weights.apply(&input);
        assert_eq!(orig_output, output);
    }
    #[test]
    fn apply_2d() {
        let input = [[0u8; 28]; 28];
        let weights = [([0u8; 8], [0u32, 1234]); 16];
        let output: [[u32; 28]; 28] = weights.apply(&input);

        let mut weights = [(0u64, [0u32, 12345]); 16];
        let mut output: [[u32; 28]; 28] = weights.apply(&input);
        weights[2] = (!0u64, [0, 12345]);
        weights.update(&input, &mut output, 2);
        let orig_output: [[u32; 28]; 28] = weights.apply(&input);
        assert_eq!(orig_output, output);
    }
}
