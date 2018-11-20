extern crate image;
extern crate rand;
extern crate rayon;
extern crate time;

#[macro_use]
pub mod datasets {
    pub mod fake {
        extern crate image;
        use image::{ImageBuffer, Rgb};

        pub fn parse_rgb_u8(bits: u8) -> [u8; 3] {
            let mut bytes = [0u8; 3];
            for color in 0..3 {
                bytes[color] = ((bits >> color) & 0b1u8) * 255;
            }
            bytes
        }
        pub fn diag_rgb_u8_packed() -> [[u8; 32]; 32] {
            let mut image = [[0u8; 32]; 32];
            for x in 0..32 {
                for y in 0..32 {
                    image[x][y] = ((((((x + (y % 3) + 0) as u8) % 3) == 0) as u8) << 0)
                        | ((((((x + (y % 3) + 1) as u8) % 3) == 0) as u8) << 1)
                        | ((((((x + (y % 3) + 2) as u8) % 3) == 0) as u8) << 2);
                }
            }
            image
        }
        pub fn write_u8_packed_image(data: [[u8; 32]; 32], path: &String) {
            let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(32u32, 32u32);
            for x in 0..32 {
                for y in 0..32 {
                    let bits = data[x][y];
                    let pixel_bytes = parse_rgb_u8(bits);
                    image.get_pixel_mut(x as u32, y as u32).data = pixel_bytes;
                }
            }
            image.save(path).unwrap();
        }
    }
    pub mod cifar {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;
        extern crate image;
        use image::{ImageBuffer, Rgb};

        pub fn write_image(data: [[[u64; 1]; 32]; 32], path: &String) {
            let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(32u32, 32u32);
            for x in 0..32 {
                for y in 0..32 {
                    let bits = data[x][y][0];
                    let pixel_bytes = parse_rgb_u64(bits);
                    image.get_pixel_mut(x as u32, y as u32).data = pixel_bytes;
                }
            }
            image.save(path).unwrap();
        }
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

pub mod featuregen {
    use super::layers;
    use super::layers::{bitvecmul, Patch};
    use rayon::prelude::*;
    use time::PreciseTime;

    // Takes patches split by class, and generates n_features_iters features for each.
    // Returns both a vec of features and a vec of unary activations.
    // Splits after each new feature.
    pub fn gen_hidden_features<T: Patch + Copy + Sync + Send>(
        train_inputs: &Vec<Vec<T>>,
        n_features_iters: usize,
        unary_size: usize,
        culling_threshold: usize,
    ) -> (Vec<T>, Vec<Vec<u32>>) {
        let feature_gen_start = PreciseTime::now();
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

    pub fn grads_one_shard<T: Patch + Sync>(by_class: &Vec<Vec<T>>, label: usize) -> Vec<f64> {
        let mut sum_bits: Vec<(usize, Vec<u32>)> = by_class.iter().map(|label_patches| (label_patches.len(), layers::count_bits(&label_patches))).collect();

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
    use rayon::prelude::*;
    use std::marker::Sized;
    use std::mem::transmute;

    pub trait Patch: Send + Sync + Sized {
        fn hamming_distance(&self, &Self) -> u32;
        fn bit_increment(&self, &mut [u32]);
        fn bit_len() -> usize;
        fn bitpack(&[bool]) -> Self;
        fn bit_or(&self, &Self) -> Self;
        fn flip_bit(&mut self, usize);
        fn get_bit(&self, usize) -> bool;
        fn mul<O: Patch, W: WeightsMatrix<Self, O>>(&self, weights: &W) -> O {
            weights.vecmul(&self)
        }
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

    pub trait WeightsMatrix<I: Patch, O: Patch> {
        fn vecmul(&self, &I) -> O;
        fn update_vecmul(&self, &I, &mut O, usize);
        fn output_bit_len() -> usize;
        fn optimize<H: ObjectiveHeadFC<O>>(&mut self, &mut H, &Vec<(usize, I)>);
        fn new_from_split(&Vec<(usize, I)>) -> Self;
    }

    macro_rules! primitive_weights_matrix {
        ($type:ty, $len:expr) => {
            impl<I: Patch + Default + Copy> WeightsMatrix<I, $type> for [(I, u32); $len] {
                fn vecmul(&self, input: &I) -> $type {
                    let mut val = 0 as $type;
                    for i in 0..$len {
                        val = val | (((self[i].0.hamming_distance(&input) > self[i].1) as $type) << i);
                    }
                    val
                }
                fn output_bit_len() -> usize {
                    $len
                }
                fn update_vecmul(&self, input: &I, target: &mut $type, index: usize) {
                    *target &= !(1 << index); // unset the bit
                    *target |= (((self[index].0.hamming_distance(&input) > self[index].1) as $type) << index); // set it to the updated value.
                }
                fn optimize<H: ObjectiveHeadFC<$type>>(&mut self, head: &mut H, examples: &Vec<(usize, I)>) {
                    for o in 0..$len {
                        println!("o: {:?}", o);
                        let mut cache: Vec<(usize, $type)> = examples.par_iter().map(|(class, input)| (*class, self.vecmul(input))).collect();
                        head.update(&cache);
                        head.update(&cache);
                        let mut acc = head.acc(&cache);
                        println!("acc: {:?}", acc);
                        for b in 0..I::bit_len() {
                            self[o].flip_bit(b);
                            let _: Vec<_> = cache
                                .par_iter_mut()
                                .zip(examples.par_iter())
                                .map(|(x, input)| self.update_vecmul(&input.1, &mut x.1, o))
                                .collect();
                            let new_acc = head.acc(&cache);
                            //println!("{:?}", new_acc);
                            if new_acc > acc {
                                acc = new_acc;
                                println!("{:?}", acc);
                            } else {
                                self[o].flip_bit(b);
                            }
                        }
                    }
                }
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

    primitive_weights_matrix!(u8, 8);
    primitive_weights_matrix!(u16, 16);
    primitive_weights_matrix!(u32, 32);
    primitive_weights_matrix!(u64, 64);
    primitive_weights_matrix!(u128, 128);

    macro_rules! primitive_weights_matrix_unary {
        ($type:ty, $len:expr, $unary_bits:expr) => {
            impl<I: Patch + Default + Copy> WeightsMatrix<I, [$type; $unary_bits]> for [(I, [u32; $unary_bits]); $len] {
                fn vecmul(&self, input: &I) -> [$type; $unary_bits] {
                    let mut val = [0 as $type; $unary_bits];
                    for i in 0..$len {
                        let dist = self[i].0.hamming_distance(&input);
                        for b in 0..$unary_bits {
                            val[b] = val[b] | (((dist > self[i].1[b]) as $type) << i);
                        }
                    }
                    val
                }
                fn output_bit_len() -> usize {
                    $len * $unary_bits
                }
                fn update_vecmul(&self, input: &I, target: &mut [$type; $unary_bits], index: usize) {
                    let dist = self[index].0.hamming_distance(&input);
                    for b in 0..$unary_bits {
                        //val[b] = val[b] | (((dist > self[i].1[b]) as $type) << i);
                        target[b] &= !(1 << index); // unset the bit
                        target[b] |= ((dist > self[index].1[b]) as $type) << index; // set it to the updated value.
                    }
                }

                fn optimize<H: ObjectiveHeadFC<[$type; $unary_bits]>>(&mut self, head: &mut H, examples: &Vec<(usize, I)>) {
                    let mut n_updates = 0;
                    for o in 0..$len {
                        println!("o: {:?}", o);
                        let mut cache: Vec<(usize, [$type; $unary_bits])> = examples.par_iter().map(|(class, input)| (*class, self.vecmul(input))).collect();
                        head.update(&cache);
                        let mut acc = head.acc(&cache);
                        for b in 0..I::bit_len() {
                            if n_updates % 5 == 0 {
                                println!("updating",);
                                head.update(&cache);
                                acc = head.acc(&cache);
                                n_updates += 1;
                            }
                            self[o].flip_bit(b);
                            let _: Vec<_> = cache
                                .par_iter_mut()
                                .zip(examples.par_iter())
                                .map(|(x, input)| self.update_vecmul(&input.1, &mut x.1, o))
                                .collect();
                            let new_acc = head.acc(&cache);
                            //println!("{:?}", new_acc);
                            if new_acc > acc {
                                n_updates += 1;
                                acc = new_acc;
                                println!("{:?}", acc);
                            } else {
                                self[o].flip_bit(b);
                            }
                        }
                    }
                }
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

    macro_rules! primitive_weights_matrix_n_bit_types {
        ($unary_bits:expr) => {
            primitive_weights_matrix_unary!(u8, 8, $unary_bits);
            primitive_weights_matrix_unary!(u16, 16, $unary_bits);
            primitive_weights_matrix_unary!(u32, 32, $unary_bits);
            primitive_weights_matrix_unary!(u64, 64, $unary_bits);
            primitive_weights_matrix_unary!(u128, 128, $unary_bits);
        };
    }

    primitive_weights_matrix_n_bit_types!(2);
    primitive_weights_matrix_n_bit_types!(3);
    primitive_weights_matrix_n_bit_types!(4);
    primitive_weights_matrix_n_bit_types!(5);
    primitive_weights_matrix_n_bit_types!(6);
    primitive_weights_matrix_n_bit_types!(7);
    primitive_weights_matrix_n_bit_types!(8);

    pub trait ObjectiveHeadFC<I> {
        fn acc(&self, &Vec<(usize, I)>) -> f64;
        fn update(&mut self, &Vec<(usize, I)>);
        fn new_from_split(&Vec<(usize, I)>) -> Self;
    }

    impl<I: Patch + Patch + Copy + Default> ObjectiveHeadFC<I> for [I; 10] {
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
        fn update(&mut self, examples: &Vec<(usize, I)>) {
            for mut_class in 0..10 {
                let mut activation_diffs: Vec<(I, i32, bool)> = examples
                    .iter()
                    .map(|(targ_class, input)| {
                        let mut activations: Vec<i32> = self.iter().map(|base_point| base_point.hamming_distance(&input) as i32).collect();

                        let targ_act = activations[*targ_class]; // the activation for target class of this example.
                        let mut_act = activations[mut_class]; // the activation which we are mutating.
                        activations[*targ_class] = -10000;
                        activations[mut_class] = -10000;
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

    pub trait Layer2D<I, IP: Patch + Default + Copy, O, OP: Patch> {
        fn conv_8pixel<W: WeightsMatrix<[IP; 8], OP>>(&self, &W) -> O;
        fn conv_3x3<W: WeightsMatrix<[[IP; 3]; 3], OP>>(&self, &W) -> O;
        fn conv_1x1<W: WeightsMatrix<IP, OP>>(&self, &W) -> O;
    }

    macro_rules! layer2d_trait {
        ($x_len:expr, $y_len:expr) => {
            impl<IP: Patch + Copy + Default, OP: Patch + Default + Copy> Layer2D<[[IP; $y_len]; $x_len], IP, [[OP; $y_len]; $x_len], OP> for [[IP; $y_len]; $x_len] {
                fn conv_3x3<W: WeightsMatrix<[[IP; 3]; 3], OP>>(&self, weights: &W) -> [[OP; $y_len]; $x_len] {
                    let mut output = [[OP::default(); $y_len]; $x_len];
                    for x in 0..($x_len - 2) {
                        for y in 0..($y_len - 2) {
                            let patch = [
                                [self[x + 0][y + 0], self[x + 0][y + 1], self[x + 0][y + 2]],
                                [self[x + 1][y + 0], self[x + 1][y + 1], self[x + 1][y + 2]],
                                [self[x + 2][y + 0], self[x + 2][y + 1], self[x + 2][y + 2]],
                            ];
                            output[x + 1][y + 1] = weights.vecmul(&patch);
                        }
                    }
                    output
                }
                fn conv_8pixel<W: WeightsMatrix<[IP; 8], OP>>(&self, weights: &W) -> [[OP; $y_len]; $x_len] {
                    let mut output = [[OP::default(); $y_len]; $x_len];
                    for x in 0..($x_len - 2) {
                        for y in 0..($y_len - 2) {
                            let patch = [
                                self[x + 0][y + 0],
                                self[x + 0][y + 1],
                                self[x + 0][y + 2],
                                self[x + 1][y + 0],
                                self[x + 1][y + 1],
                                self[x + 1][y + 2],
                                self[x + 2][y + 0],
                                self[x + 2][y + 1],
                            ];
                            output[x + 1][y + 1] = weights.vecmul(&patch);
                        }
                    }
                    output
                }
                fn conv_1x1<W: WeightsMatrix<IP, OP>>(&self, weights: &W) -> [[OP; $y_len]; $x_len] {
                    let mut output = [[OP::default(); $y_len]; $x_len];
                    for x in 0..$x_len {
                        for y in 0..$y_len {
                            output[x][y] = weights.vecmul(&self[x][y]);
                        }
                    }
                    output
                }
            }
        };
    }

    layer2d_trait!(4, 4);
    layer2d_trait!(8, 8);
    layer2d_trait!(16, 16);
    layer2d_trait!(32, 32);

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

    pub mod pack_3x3 {
        macro_rules! pack_3x3 {
            ($name:ident, $in_type:ty, $out_type:ty, $out_len:expr) => {
                pub fn $name(pixels: [$in_type; 9]) -> $out_type {
                    let mut word = 0 as $out_type;
                    for i in 0..9 {
                        word = word | ((pixels[i] as $out_type) << (i * ($out_len / 9)))
                    }
                    word
                }
            };
        }

        pack_3x3!(p3, u8, u32, 32);
        pack_3x3!(p7, u8, u64, 64);
        pack_3x3!(p14, u16, u128, 128);

        macro_rules! array_pack_u32_u128 {
            ($name:ident, $size:expr) => {
                pub fn $name(input: &[u32; $size * 4]) -> [u128; $size] {
                    let mut output = [0u128; $size];
                    for i in 0..$size {
                        for shift in 0..4 {
                            output[i] = output[i] | ((input[i * 4 + shift] as u128) << (shift * 32));
                        }
                    }
                    output
                }
            };
        }
        array_pack_u32_u128!(array_pack_u32_u128_64, 64);

        macro_rules! array_pack_u8_u128 {
            ($name:ident, $size:expr) => {
                pub fn $name(input: &[u8; $size * 16]) -> [u128; $size] {
                    let mut output = [0u128; $size];
                    for i in 0..$size {
                        for shift in 0..16 {
                            output[i] = output[i] | ((input[i * 4 + shift] as u128) << (shift * 8));
                        }
                    }
                    output
                }
            };
        }
        array_pack_u8_u128!(array_pack_u8_u128_49, 49);

        macro_rules! flatten_2d {
            ($name:ident, $x_size:expr, $y_size:expr) => {
                pub fn $name<T: Default + Copy>(input: &[[T; $y_size]; $x_size]) -> [T; $x_size * $y_size] {
                    let mut output = [T::default(); $x_size * $y_size];
                    for x in 0..$x_size {
                        let offset = x * $y_size;
                        for y in 0..$y_size {
                            output[offset + y] = input[x][y];
                        }
                    }
                    output
                }
            };
        }
        flatten_2d!(flatten_4x4, 4, 4);
        flatten_2d!(flatten_8x8, 8, 8);
        flatten_2d!(flatten_16x16, 16, 16);
        flatten_2d!(flatten_28x28, 28, 28);
        flatten_2d!(flatten_32x32, 32, 32);
    }

    pub trait Layer2d<T> {
        fn to_pixels(&self) -> Vec<T>;
        fn to_3x3_patches(&self) -> Vec<[T; 9]>;
        fn from_pixels_1_padding(&Vec<T>) -> Self;
        fn from_pixels_0_padding(&Vec<T>) -> Self;
    }

    macro_rules! layer_2d {
        ($x_size:expr, $y_size:expr) => {
            impl<T: Copy + Default> Layer2d<T> for [[T; $y_size]; $x_size] {
                fn to_pixels(&self) -> Vec<T> {
                    let mut patches = Vec::with_capacity($x_size * $y_size);
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            patches.push(self[x][y]);
                        }
                    }
                    patches
                }
                fn to_3x3_patches(&self) -> Vec<[T; 9]> {
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
                fn from_pixels_0_padding(pixels: &Vec<T>) -> [[T; $y_size]; $x_size] {
                    if pixels.len() != ($x_size * $y_size) {
                        panic!("pixels is wrong len")
                    }
                    let mut output = [[T::default(); $y_size]; $x_size];
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            output[x][y] = pixels[x * $y_size + y];
                        }
                    }
                    output
                }
                fn from_pixels_1_padding(pixels: &Vec<T>) -> [[T; $y_size]; $x_size] {
                    if pixels.len() != (($x_size - 2) * ($y_size - 2)) {
                        panic!("pixels should be {:}, but is {:}", (($x_size - 2) * ($y_size - 2)), pixels.len());
                    }
                    let mut output = [[T::default(); $y_size]; $x_size];
                    for x in 0..($x_size - 2) {
                        for y in 0..($y_size - 2) {
                            output[x + 1][y + 1] = pixels[x * ($y_size - 2) + y];
                        }
                    }
                    output
                }
            }
        };
    }
    layer_2d!(4, 4);
    layer_2d!(5, 5);
    layer_2d!(8, 8);
    layer_2d!(16, 16);
    layer_2d!(28, 28);
    layer_2d!(32, 32);
    layer_2d!(64, 64);

    // From a 2D image, extract various different patches with stride of 1.
    pub trait ExtractPatches<I, IP, O> {
        fn extract_patches(&self) -> Vec<O>;
    }

    // 3x3 flattened to [T; 9] array.
    impl<IP: Copy> ExtractPatches<[[IP; 32]; 32], IP, [IP; 9]> for [[IP; 32]; 32] {
        fn extract_patches(&self) -> Vec<[IP; 9]> {
            let mut patches = Vec::with_capacity((32 - 2) * (32 - 2));
            for x in 0..32 - 2 {
                for y in 0..32 - 2 {
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

    // 3x3 in [[T; 3]; 3]
    impl<IP: Copy> ExtractPatches<[[IP; 32]; 32], IP, [[IP; 3]; 3]> for [[IP; 32]; 32] {
        fn extract_patches(&self) -> Vec<[[IP; 3]; 3]> {
            let mut patches = Vec::with_capacity((32 - 2) * (32 - 2));
            for x in 0..32 - 2 {
                for y in 0..32 - 2 {
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

    // 3x3 with notch, flattened to [T; 8]
    impl<IP: Copy> ExtractPatches<[[IP; 32]; 32], IP, [IP; 8]> for [[IP; 32]; 32] {
        fn extract_patches(&self) -> Vec<[IP; 8]> {
            let mut patches = Vec::with_capacity((32 - 2) * (32 - 2));
            for x in 0..32 - 2 {
                for y in 0..32 - 2 {
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

    // One pixel, just T
    impl<IP: Copy> ExtractPatches<[[IP; 32]; 32], IP, IP> for [[IP; 32]; 32] {
        fn extract_patches(&self) -> Vec<IP> {
            let mut patches = Vec::with_capacity(32 * 32);
            for x in 0..32 {
                for y in 0..32 {
                    patches.push(self[x][y]);
                }
            }
            patches
        }
    }

    // 2x2 with stride of 2.
    impl<IP: Copy> ExtractPatches<[[IP; 32]; 32], IP, [IP; 4]> for [[IP; 32]; 32] {
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

    pub trait PatchMap<I, IP, IA, O, OP> {
        fn patch_map(&self, &Fn(&IA, &mut OP)) -> O;
    }

    macro_rules! patch_map_trait_pixel {
        ($x_size:expr, $y_size:expr) => {
            impl<IP, OP: Copy + Default> PatchMap<[[IP; $y_size]; $x_size], IP, IP, [[OP; $y_size]; $x_size], OP> for [[IP; $y_size]; $x_size] {
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
    patch_map_trait_pixel!(16, 16);

    macro_rules! patch_map_trait_notched {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy, OP: Copy + Default> PatchMap<[[IP; $y_size]; $x_size], IP, [IP; 8], [[OP; $y_size]; $x_size], OP> for [[IP; $y_size]; $x_size] {
                fn patch_map(&self, map_fn: &Fn(&[IP; 8], &mut OP)) -> [[OP; $y_size]; $x_size] {
                    let mut output = [[OP::default(); $y_size]; $x_size];
                    for x in 0..$x_size {
                        for y in 0..$y_size {
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
    patch_map_trait_notched!(16, 16);

    macro_rules! patch_map_trait_3x3 {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy, OP: Copy + Default> PatchMap<[[IP; $y_size]; $x_size], IP, [IP; 9], [[OP; $y_size]; $x_size], OP> for [[IP; $y_size]; $x_size] {
                fn patch_map(&self, map_fn: &Fn(&[IP; 9], &mut OP)) -> [[OP; $y_size]; $x_size] {
                    let mut output = [[OP::default(); $y_size]; $x_size];
                    for x in 0..$x_size {
                        for y in 0..$y_size {
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
    patch_map_trait_3x3!(16, 16);

    macro_rules! patch_map_trait_2x2_pool {
        ($x_size:expr, $y_size:expr) => {
            impl<IP: Copy, OP: Copy + Default> PatchMap<[[IP; $y_size]; $x_size], IP, [IP; 4], [[OP; $y_size / 2]; $x_size / 2], OP> for [[IP; $y_size]; $x_size] {
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
    patch_map_trait_2x2_pool!(16, 16);

    pub mod pixelmap {
        use super::Patch;
        use featuregen;
        macro_rules! pixel_map_2d {
            ($name:ident, $x_size:expr, $y_size:expr) => {
                pub fn $name<I: Copy, O: Copy + Default>(input: &[[I; $y_size]; $x_size], map_fn: &Fn(I) -> O) -> [[O; $y_size]; $x_size] {
                    let mut output = [[O::default(); $y_size]; $x_size];
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            output[x][y] = map_fn(input[x][y]);
                        }
                    }
                    output
                }
            };
        }

        pixel_map_2d!(pm_28, 28, 28);
        pixel_map_2d!(pm_32, 32, 32);
        pixel_map_2d!(pm_64, 64, 64);

        macro_rules! or_pooling {
            ($name:ident, $x_size:expr, $y_size:expr) => {
                pub fn $name<T: Copy + Default + Patch>(image: &[[T; $y_size]; $x_size]) -> [[T; $y_size / 2]; $x_size / 2] {
                    let mut pooled = [[T::default(); $y_size / 2]; $x_size / 2];
                    for x in 0..($x_size / 2) {
                        let x_base = x * 2;
                        for y in 0..($y_size / 2) {
                            let y_base = y * 2;
                            pooled[x][y] = image[x_base + 0][y_base + 0]
                                .bit_or(&image[x_base + 0][y_base + 1])
                                .bit_or(&image[x_base + 1][y_base + 0])
                                .bit_or(&image[x_base + 1][y_base + 1]);
                        }
                    }
                    pooled
                }
            };
        }
        or_pooling!(pool_or_32, 32, 32);
        or_pooling!(pool_or_16, 16, 16);
        or_pooling!(pool_or_8, 8, 8);
    }

    pub fn count_bits<T: Patch + Sync>(patches: &Vec<T>) -> Vec<u32> {
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

    pub mod readout {
        use super::Patch;
        use featuregen;
        use rayon::prelude::*;

        pub struct ReadoutHead10<T: Patch> {
            weights: [T; 10],
            biases: [i32; 10],
        }

        impl<T: Patch + Default + Copy + Sync + Send> ReadoutHead10<T> {
            pub fn acc(&self, examples: &Vec<(usize, T)>) -> f64 {
                let test_correct: u64 = examples
                    .par_iter()
                    .map(|input| {
                        (input.0 == self
                            .weights
                            .iter()
                            .zip(self.biases.iter())
                            .map(|(base_point, bias)| base_point.hamming_distance(&input.1) as i32 - bias)
                            .enumerate()
                            .max_by_key(|(_, dist)| *dist)
                            .unwrap()
                            .0) as u64
                    }).sum();
                test_correct as f64 / examples.len() as f64
            }
            pub fn new_from_split(examples: &Vec<(usize, T)>) -> Self {
                let by_class = featuregen::split_by_label(&examples, 10);
                let mut readout = ReadoutHead10 {
                    weights: [T::default(); 10],
                    biases: [0i32; 10],
                };
                for class in 0..10 {
                    let grads = featuregen::grads_one_shard(&by_class, class);
                    let sign_bits: Vec<bool> = grads.iter().map(|x| *x > 0f64).collect();
                    readout.weights[class] = T::bitpack(&sign_bits);
                    let sum_activation: u64 = examples.iter().map(|(_, input)| readout.weights[class].hamming_distance(&input) as u64).sum();
                    readout.biases[class] = (sum_activation as f64 / examples.len() as f64) as i32;
                }
                readout
            }
            pub fn bitwise_ascend_acc(&mut self, examples: &Vec<(usize, T)>) {
                for mut_class in 0..10 {
                    let mut activation_diffs: Vec<(T, i32, bool)> = examples
                        .iter()
                        .map(|(targ_class, input)| {
                            let mut activations: Vec<i32> = self
                                .weights
                                .iter()
                                .zip(self.biases.iter())
                                .map(|(base_point, bias)| base_point.hamming_distance(&input) as i32 - bias)
                                .collect();

                            let targ_act = activations[*targ_class]; // the activation for target class of this example.
                            let mut_act = activations[mut_class]; // the activation which we are mutating.
                            activations[*targ_class] = -10000;
                            activations[mut_class] = -10000;
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
                    for b in 0..T::bit_len() {
                        // the new weights bit
                        let new_weights_bit = !self.weights[mut_class].get_bit(b);
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
                            self.weights[mut_class].flip_bit(b);
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::layers::{bitvecmul, pack_3x3, unary};
    use super::layers::{Layer2d, Patch};
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
    //#[test]
    //fn bit_vecmul() {
    //    let input = [0u128; 3];
    //    let weights = [([0u128; 3], 0u32); 7];
    //    let output = bitvecmul::bvm_u7(&weights, &input);
    //    assert_eq!(output, 0b0111_1111u8);
    //}
    #[test]
    fn unary() {
        assert_eq!(unary::to_3(128), 0b0000_0001u8);
        assert_eq!(unary::to_3(255), 0b0000_0111u8);
        assert_eq!(unary::to_7(255), 0b0111_1111u8);
        assert_eq!(unary::to_7(0), 0b0000_0000u8);
    }
    #[test]
    fn pack() {
        assert_eq!(
            pack_3x3::p3([0b11u8, 0b1u8, 0b1u8, 0b1u8, 0b1u8, 0b1u8, 0b1u8, 0b111u8, 0b1u8]),
            0b001111001_001001001_001001011u32
        );
    }
    #[test]
    fn extract_patches() {
        let value = 0b1011_0111u8;
        let layer = [[value; 28]; 28];
        let patches: Vec<_> = layer.to_3x3_patches().iter().map(|&x| x).collect();
        assert_eq!(patches.len(), 676);
        assert_eq!(patches[0], [value; 9])
    }
    #[test]
    fn rgb_pack() {
        assert_eq!(unary::rgb_to_u14([255, 255, 255]), 0b0011_1111_1111_1111u16);
        assert_eq!(unary::rgb_to_u14([128, 128, 128]), 0b0000_0110_0011_0011u16);
        assert_eq!(unary::rgb_to_u14([0, 0, 0]), 0b0u16);
    }
    #[test]
    fn layer_pixels() {
        let layer = [[0b1001100u8; 28]; 28];
        let patches = layer.to_pixels();
        let new_layer = <[[u8; 28]; 28]>::from_pixels_0_padding(&patches);
        assert_eq!(layer, new_layer);

        let layer = [[!0u8; 5], [0u8; 5], [!0u8; 5], [0u8; 5], [!0u8; 5]];
        let patches = layer.to_3x3_patches();
        let pixels: Vec<u8> = patches.iter().map(|x| x[4]).collect();
        assert_eq!(patches.len(), 9);
        let new_layer = <[[u8; 5]; 5]>::from_pixels_1_padding(&pixels);
        let target_new_layer = [[0u8; 5], [0u8; 5], [0u8, !0u8, !0u8, !0u8, 0u8], [0u8; 5], [0u8; 5]];
        assert_eq!(new_layer, target_new_layer);
    }
}
