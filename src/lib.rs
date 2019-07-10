extern crate bincode;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate serde_derive;
extern crate time;
use rayon::prelude::*;

pub mod datasets {
    pub mod mnist {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;
        pub fn load_labels(path: &Path, size: usize) -> Vec<usize> {
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
        pub fn load_images_bitpacked(path: &Path, size: usize) -> Vec<[u64; 13]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

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
        pub fn load_images_bitpacked_u32(path: &Path, size: usize) -> Vec<[u32; 25]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[u32; 25]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image_words: [u32; 25] = [0; 25];
                for p in 0..784 {
                    let word_index = p / 32;
                    image_words[word_index] = image_words[word_index] | (((images_bytes[p] > 128) as u32) << p % 32);
                }
                images.push(image_words);
            }
            return images;
        }

        pub fn load_images_u8_unary(path: &Path, size: usize) -> Vec<[[u8; 28]; 28]> {
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
    }

    pub mod cifar {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;

        macro_rules! to_unary {
            ($name:ident, $type:ty, $len:expr) => {
                fn $name(input: u8) -> $type {
                    !((!0) << (input / (256 / $len) as u8))
                }
            };
        }

        to_unary!(to_2, u8, 2);
        to_unary!(to_3, u8, 3);
        to_unary!(to_4, u8, 4);
        to_unary!(to_10, u32, 10);
        to_unary!(to_11, u32, 11);
        to_unary!(to_32, u32, 32);

        pub trait ConvertPixel {
            fn convert(pixel: [u8; 3]) -> Self;
        }
        impl ConvertPixel for [u8; 3] {
            fn convert(pixel: [u8; 3]) -> [u8; 3] {
                pixel
            }
        }
        impl ConvertPixel for [u32; 1] {
            fn convert(pixel: [u8; 3]) -> [u32; 1] {
                [to_11(pixel[0]) as u32 | ((to_11(pixel[1]) as u32) << 11) | ((to_10(pixel[2]) as u32) << 22)]
            }
        }
        impl ConvertPixel for u8 {
            fn convert(pixel: [u8; 3]) -> u8 {
                to_3(pixel[0]) | ((to_3(pixel[1])) << 3) | ((to_3(pixel[2])) << 6)
            }
        }

        impl ConvertPixel for [u32; 3] {
            fn convert(pixel: [u8; 3]) -> [u32; 3] {
                [to_32(pixel[0]), to_32(pixel[1]), to_32(pixel[2])]
            }
        }

        pub fn load_images_from_base<T: Default + Copy + ConvertPixel>(base_path: &Path, n: usize) -> Vec<([[T; 32]; 32], usize)> {
            if n > 50000 {
                panic!("n must be <= 50,000");
            }
            (1..6)
                .map(|i| {
                    let mut file = File::open(&base_path.join(format!("data_batch_{}.bin", i))).expect("can't open data");

                    let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
                    let mut label: [u8; 1] = [0; 1];
                    let mut images: Vec<([[T; 32]; 32], usize)> = Vec::new();
                    for _ in 0..10000 {
                        file.read_exact(&mut label).expect("can't read label");
                        file.read_exact(&mut image_bytes).expect("can't read images");
                        let mut image = [[T::default(); 32]; 32];
                        for x in 0..32 {
                            for y in 0..32 {
                                let pixel = [image_bytes[(0 * 1024) + (y * 32) + x], image_bytes[(1 * 1024) + (y * 32) + x], image_bytes[(2 * 1024) + (y * 32) + x]];
                                image[x][y] = T::convert(pixel);
                            }
                        }
                        images.push((image, label[0] as usize));
                    }
                    images
                })
                .flatten()
                .take(n)
                .collect()
        }
    }
}

pub trait BitOr {
    fn bit_or(&self, other: &Self) -> Self;
}

macro_rules! impl_bitor_for_uint {
    ($type:ty) => {
        impl BitOr for $type {
            fn bit_or(&self, other: &Self) -> $type {
                self | other
            }
        }
    };
}

impl_bitor_for_uint!(u8);
impl_bitor_for_uint!(u16);
impl_bitor_for_uint!(u32);
impl_bitor_for_uint!(u64);

pub trait GetBit {
    fn bit(&self, i: usize) -> bool;
}

impl GetBit for u8 {
    #[inline(always)]
    fn bit(&self, i: usize) -> bool {
        ((self >> i) & 1u8) == 1u8
    }
}

impl GetBit for u64 {
    #[inline(always)]
    fn bit(&self, i: usize) -> bool {
        ((self >> i) & 1u64) == 1u64
    }
}

impl GetBit for u32 {
    #[inline(always)]
    fn bit(&self, i: usize) -> bool {
        ((self >> i) & 1u32) == 1u32
    }
}

macro_rules! impl_getbit_for_array {
    ($len:expr) => {
        impl<T: GetBit + BitLen> GetBit for [T; $len] {
            #[inline(always)]
            fn bit(&self, i: usize) -> bool {
                self[i / T::BIT_LEN].bit(i % T::BIT_LEN)
            }
        }
    };
}

impl_getbit_for_array!(1);
impl_getbit_for_array!(2);
impl_getbit_for_array!(3);
impl_getbit_for_array!(4);
impl_getbit_for_array!(5);
impl_getbit_for_array!(6);
impl_getbit_for_array!(7);
impl_getbit_for_array!(8);
impl_getbit_for_array!(13);

impl<A: GetBit + BitLen, B: GetBit + BitLen> GetBit for (A, B) {
    fn bit(&self, i: usize) -> bool {
        if i < A::BIT_LEN {
            self.0.bit(i)
        } else {
            self.1.bit(i - A::BIT_LEN)
        }
    }
}

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

//impl BitLen for bool {
//    const BIT_LEN: usize = 1;
//}

impl BitLen for u8 {
    const BIT_LEN: usize = 8;
}

impl BitLen for u32 {
    const BIT_LEN: usize = 32;
}

impl BitLen for u64 {
    const BIT_LEN: usize = 64;
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
array_bit_len!(32);

impl<A: BitLen, B: BitLen> BitLen for (A, B) {
    const BIT_LEN: usize = A::BIT_LEN + B::BIT_LEN;
}

pub trait SetBit {
    fn set_bit(&mut self, index: usize, value: bool);
}

impl SetBit for u32 {
    fn set_bit(&mut self, index: usize, value: bool) {
        *self &= !(1 << index);
        *self |= (value as u32) << index;
    }
}
impl SetBit for u8 {
    fn set_bit(&mut self, index: usize, value: bool) {
        *self &= !(1 << index);
        *self |= (value as u8) << index;
    }
}

macro_rules! impl_setbit_for_array {
    ($len: expr) => {
        impl<T: SetBit + BitLen> SetBit for [T; $len] {
            fn set_bit(&mut self, index: usize, value: bool) {
                self[index / T::BIT_LEN].set_bit(index % T::BIT_LEN, value);
            }
        }
    };
}

impl_setbit_for_array!(1);
impl_setbit_for_array!(2);
impl_setbit_for_array!(3);
impl_setbit_for_array!(4);
impl_setbit_for_array!(5);
impl_setbit_for_array!(6);
impl_setbit_for_array!(7);
impl_setbit_for_array!(8);

impl<A: SetBit + BitLen, B: SetBit + BitLen> SetBit for (A, B) {
    fn set_bit(&mut self, i: usize, value: bool) {
        if i < A::BIT_LEN {
            self.0.set_bit(i, value);
        } else {
            self.1.set_bit(i - A::BIT_LEN, value);
        }
    }
}

pub trait FlipBit {
    fn flip_bit(&mut self, b: usize);
}
impl FlipBit for u8 {
    fn flip_bit(&mut self, index: usize) {
        *self ^= 1 << index
    }
}

impl FlipBit for u32 {
    fn flip_bit(&mut self, index: usize) {
        *self ^= 1 << index
    }
}
impl FlipBit for u64 {
    fn flip_bit(&mut self, index: usize) {
        *self ^= 1 << index
    }
}

macro_rules! array_flip_bit {
    ($len:expr) => {
        impl<T: BitLen + FlipBit> FlipBit for [T; $len] {
            fn flip_bit(&mut self, index: usize) {
                self[index / T::BIT_LEN].flip_bit(index % T::BIT_LEN);
            }
        }
    };
}

array_flip_bit!(1);
array_flip_bit!(2);
array_flip_bit!(3);
array_flip_bit!(4);
array_flip_bit!(5);
array_flip_bit!(6);
array_flip_bit!(7);
array_flip_bit!(8);
array_flip_bit!(13);

impl<A: FlipBit + BitLen, B: FlipBit + BitLen> FlipBit for (A, B) {
    fn flip_bit(&mut self, i: usize) {
        if i < A::BIT_LEN {
            self.0.flip_bit(i);
        } else {
            self.1.flip_bit(i - A::BIT_LEN);
        }
    }
}

pub trait FlipBitIndexed {
    fn flip_bit_indexed(&mut self, o: usize, b: usize);
    const INDEX_LEN: usize;
}

impl<T: FlipBit> FlipBitIndexed for [T; 32] {
    fn flip_bit_indexed(&mut self, o: usize, b: usize) {
        self[o].flip_bit(b);
    }
    const INDEX_LEN: usize = 32;
}

impl<T: FlipBit> FlipBitIndexed for [T; 16] {
    fn flip_bit_indexed(&mut self, o: usize, b: usize) {
        self[o].flip_bit(b);
    }
    const INDEX_LEN: usize = 16;
}

macro_rules! impl_flipbitindexed_for_array {
    ($len:expr) => {
        impl<T: FlipBit> FlipBitIndexed for [[T; 32]; $len] {
            fn flip_bit_indexed(&mut self, o: usize, b: usize) {
                self[o / 32][o % 32].flip_bit(b);
            }
            const INDEX_LEN: usize = 32 * $len;
        }
        impl<T: FlipBit> FlipBitIndexed for [[T; 16]; $len] {
            fn flip_bit_indexed(&mut self, o: usize, b: usize) {
                self[o / 16][o % 16].flip_bit(b);
            }
            const INDEX_LEN: usize = 16 * $len;
        }
    };
}

impl_flipbitindexed_for_array!(1);
impl_flipbitindexed_for_array!(2);
impl_flipbitindexed_for_array!(3);
impl_flipbitindexed_for_array!(4);
impl_flipbitindexed_for_array!(5);
impl_flipbitindexed_for_array!(6);
impl_flipbitindexed_for_array!(7);
impl_flipbitindexed_for_array!(8);

pub trait GetPatch<T> {
    fn get_patch(&self, index: usize) -> T;
}
macro_rules! impl_getpatch_for_weights {
    ($words:expr) => {
        impl<T: Copy> GetPatch<T> for [[T; 16]; $words] {
            fn get_patch(&self, index: usize) -> T {
                self[index / 16][index % 16]
            }
        }
        impl<T: Copy> GetPatch<T> for [[T; 32]; $words] {
            fn get_patch(&self, index: usize) -> T {
                self[index / 32][index % 32]
            }
        }
    };
}
impl_getpatch_for_weights!(1);
impl_getpatch_for_weights!(2);
impl_getpatch_for_weights!(3);
impl_getpatch_for_weights!(4);
impl_getpatch_for_weights!(5);
impl_getpatch_for_weights!(6);
impl_getpatch_for_weights!(7);
impl_getpatch_for_weights!(8);

pub trait WordLen {
    const WORD_LEN: usize;
}

impl WordLen for u32 {
    const WORD_LEN: usize = 1;
}

macro_rules! impl_wordlen_for_array {
    ($len:expr) => {
        impl<T: WordLen> WordLen for [T; $len] {
            const WORD_LEN: usize = T::WORD_LEN * $len;
        }
    };
}
impl_wordlen_for_array!(1);
impl_wordlen_for_array!(2);
impl_wordlen_for_array!(3);
impl_wordlen_for_array!(4);
impl_wordlen_for_array!(5);
impl_wordlen_for_array!(6);
impl_wordlen_for_array!(7);
impl_wordlen_for_array!(8);

pub trait GetWord {
    fn get_word(&self, i: usize) -> u32;
}

impl GetWord for u32 {
    #[inline(always)]
    fn get_word(&self, _i: usize) -> u32 {
        *self
    }
}

macro_rules! impl_getword_for_array {
    ($len:expr) => {
        impl<T: GetWord + WordLen> GetWord for [T; $len] {
            #[inline(always)]
            fn get_word(&self, i: usize) -> u32 {
                self[i / T::WORD_LEN].get_word(i % T::WORD_LEN)
            }
        }
    };
}
impl_getword_for_array!(1);
impl_getword_for_array!(2);
impl_getword_for_array!(3);
impl_getword_for_array!(4);
impl_getword_for_array!(5);
impl_getword_for_array!(6);
impl_getword_for_array!(7);
impl_getword_for_array!(8);

pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
}

//impl HammingDistance for bool {
//    fn hamming_distance(&self, other: &Self) -> u32 {
//        (self ^ other) as u32
//    }
//}

impl HammingDistance for u8 {
    #[inline(always)]
    fn hamming_distance(&self, other: &u8) -> u32 {
        (self ^ other).count_ones()
    }
}

impl HammingDistance for u32 {
    #[inline(always)]
    fn hamming_distance(&self, other: &u32) -> u32 {
        (self ^ other).count_ones()
    }
}

impl HammingDistance for u64 {
    #[inline(always)]
    fn hamming_distance(&self, other: &u64) -> u32 {
        (self ^ other).count_ones()
    }
}

macro_rules! array_hamming_distance {
    ($len:expr) => {
        impl<T: HammingDistance> HammingDistance for [T; $len] {
            fn hamming_distance(&self, other: &[T; $len]) -> u32 {
                let mut distance = 0u32;
                for i in 0..$len {
                    distance += self[i].hamming_distance(&other[i]);
                }
                distance
            }
        }
    };
}

array_hamming_distance!(1);
array_hamming_distance!(2);
array_hamming_distance!(3);
array_hamming_distance!(4);
array_hamming_distance!(5);
array_hamming_distance!(6);
array_hamming_distance!(7);
array_hamming_distance!(8);
array_hamming_distance!(10);
array_hamming_distance!(13);
array_hamming_distance!(32);

impl<A: HammingDistance, B: HammingDistance> HammingDistance for (A, B) {
    fn hamming_distance(&self, other: &(A, B)) -> u32 {
        self.0.hamming_distance(&other.0) + self.1.hamming_distance(&other.1)
    }
}

pub trait ElementwiseAdd {
    fn elementwise_add(&mut self, other: &Self);
}

impl ElementwiseAdd for u32 {
    fn elementwise_add(&mut self, other: &u32) {
        *self += other;
    }
}

impl<T: ElementwiseAdd> ElementwiseAdd for (T, T) {
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
    }
}

macro_rules! impl_elementwiseadd_for_array {
    ($len:expr) => {
        impl<T: ElementwiseAdd> ElementwiseAdd for [T; $len] {
            fn elementwise_add(&mut self, other: &[T; $len]) {
                for i in 0..$len {
                    self[i].elementwise_add(&other[i]);
                }
            }
        }
    };
}

impl_elementwiseadd_for_array!(1);
impl_elementwiseadd_for_array!(2);
impl_elementwiseadd_for_array!(3);
impl_elementwiseadd_for_array!(4);
impl_elementwiseadd_for_array!(8);
impl_elementwiseadd_for_array!(10);
impl_elementwiseadd_for_array!(32);

pub trait ArrayBitIncrement {
    type CountersType;
    type DiffType;
    fn increment_counters(&self, counters: &mut Self::CountersType);
    //fn compare(
    //    counters_0: &Self::CountersType,
    //    counters_1: &Self::CountersType,
    //    len: usize,
    //) -> Self::DiffType;
    fn compare_and_bitpack(&mut self, counters_0: &Self::CountersType, counters_1: &Self::CountersType, len: usize);
}

impl ArrayBitIncrement for u32 {
    type CountersType = [u32; 32];
    type DiffType = [f64; 32];
    fn increment_counters(&self, counters: &mut Self::CountersType) {
        for b in 0..32 {
            counters[b] += (self >> b) & 1
        }
    }
    //fn compare(
    //    counters_0: &Self::CountersType,
    //    counters_1: &Self::CountersType,
    //    len: usize,
    //) -> Self::DiffType {
    //    let mut diffs = Self::DiffType::default();
    //    for b in 0..32 {
    //        diffs[b] = (counters_0[b] as f64 - counters_1[b] as f64) / len as f64;
    //    }
    //    diffs
    //}
    fn compare_and_bitpack(&mut self, counters_0: &Self::CountersType, counters_1: &Self::CountersType, len: usize) {
        for b in 0..32 {
            let diff = (counters_0[b] as f64 - counters_1[b] as f64) / len as f64;
            if diff.abs() > 0.1 {
                self.set_bit(b, (counters_0[b] > counters_1[b]));
            }
        }
    }
}

impl ArrayBitIncrement for u8 {
    type CountersType = [u32; 8];
    type DiffType = [f64; 8];
    fn increment_counters(&self, counters: &mut Self::CountersType) {
        for b in 0..8 {
            counters[b] += ((self >> b) & 1) as u32
        }
    }
    //fn compare(
    //    counters_0: &Self::CountersType,
    //    counters_1: &Self::CountersType,
    //    len: usize,
    //) -> Self::DiffType {
    //    let mut diffs = Self::DiffType::default();
    //    for b in 0..8 {
    //        diffs[b] = (counters_0[b] as f64 - counters_1[b] as f64) / len as f64;
    //    }
    //    diffs
    //}
    fn compare_and_bitpack(&mut self, counters_0: &Self::CountersType, counters_1: &Self::CountersType, len: usize) {
        let mut target = 0u8;
        for b in 0..8 {
            let diff = (counters_0[b] as f64 - counters_1[b] as f64) / len as f64;
            if diff.abs() > 0.1 {
                target.set_bit(b, (counters_0[b] < counters_1[b]));
            }
        }
        *self = target;
    }
}

macro_rules! impl_bitincrement_for_array {
    ($len:expr) => {
        impl<T: ArrayBitIncrement + Default + Copy> ArrayBitIncrement for [T; $len] {
            type CountersType = [T::CountersType; $len];
            type DiffType = [T::DiffType; $len];
            fn increment_counters(&self, counters: &mut Self::CountersType) {
                for i in 0..$len {
                    self[i].increment_counters(&mut counters[i]);
                }
            }
            //fn compare(
            //    counters_0: &Self::CountersType,
            //    counters_1: &Self::CountersType,
            //    len: usize,
            //) -> Self::DiffType {
            //    let mut diffs = Self::DiffType::default();
            //    for i in 0..$len {
            //        diffs[i] = T::compare(&counters_0[i], &counters_1[i], len);
            //    }
            //    diffs
            //}
            fn compare_and_bitpack(&mut self, counters_0: &Self::CountersType, counters_1: &Self::CountersType, len: usize) {
                for i in 0..$len {
                    self[i].compare_and_bitpack(&counters_0[i], &counters_1[i], len);
                }
            }
        }
    };
}

impl_bitincrement_for_array!(1);
impl_bitincrement_for_array!(2);
impl_bitincrement_for_array!(3);
impl_bitincrement_for_array!(4);
impl_bitincrement_for_array!(8);
impl_bitincrement_for_array!(32);

pub trait MatrixBitIncrement<Input, Target> {
    type MatrixCountersType;
    fn increment_matrix_counters(&self, counters: &mut Self::MatrixCountersType, input: &Input, target: &Target, tanh_width: u32);
    fn bitpack(&mut self, counters: &Self::MatrixCountersType, len: usize);
}

// TODO pass current state.
impl<Input: ArrayBitIncrement + Copy + Default + HammingDistance + BitLen> MatrixBitIncrement<Input, u8> for [Input; 8] {
    type MatrixCountersType = [(Input::CountersType, Input::CountersType); 8];
    fn increment_matrix_counters(&self, counters: &mut Self::MatrixCountersType, input: &Input, target: &u8, tanh_width: u32) {
        for b in 0..8 {
            let activation = self[b].hamming_distance(&input);
            let threshold = Input::BIT_LEN as u32 / 2;
            // this patch only gets to vote if it is within tanh_width.
            let diff = activation.saturating_sub(threshold) | threshold.saturating_sub(activation);
            if diff < tanh_width {
                if target.bit(b) {
                    input.increment_counters(&mut counters[b].0);
                } else {
                    input.increment_counters(&mut counters[b].1);
                }
            }
        }
    }
    fn bitpack(&mut self, counters: &Self::MatrixCountersType, len: usize) {
        for b in 0..8 {
            self[b].compare_and_bitpack(&counters[b].0, &counters[b].1, len);
        }
    }
}

impl<Input: ArrayBitIncrement + Copy + Default + HammingDistance + BitLen> MatrixBitIncrement<Input, u32> for [Input; 32] {
    type MatrixCountersType = [(Input::CountersType, Input::CountersType); 32];
    fn increment_matrix_counters(&self, counters: &mut Self::MatrixCountersType, input: &Input, target: &u32, tanh_width: u32) {
        for b in 0..32 {
            // No grad is:
            //      _____
            //     |
            // ____|
            // current is:
            //       _____
            //      /
            // ____/
            // where the width is adjustable.
            let activation = self[b].hamming_distance(&input);
            let threshold = Input::BIT_LEN as u32 / 2;
            // this patch only gets to vote if it is within tanh_width.
            let diff = activation.saturating_sub(threshold) | threshold.saturating_sub(activation);
            if diff < tanh_width {
                if target.bit(b) {
                    input.increment_counters(&mut counters[b].0);
                } else {
                    input.increment_counters(&mut counters[b].1);
                }
            }
        }
    }
    fn bitpack(&mut self, counters: &Self::MatrixCountersType, len: usize) {
        for b in 0..32 {
            self[b].compare_and_bitpack(&counters[b].0, &counters[b].1, len);
        }
    }
}

macro_rules! impl_matrixbitincrement_for_array_of_matrixbitincrement {
    ($len:expr) => {
        impl<Input, Target, MatrixBits: MatrixBitIncrement<Input, Target> + Copy + Default> MatrixBitIncrement<Input, [Target; $len]> for [MatrixBits; $len] {
            type MatrixCountersType = [MatrixBits::MatrixCountersType; $len];
            fn increment_matrix_counters(&self, counters: &mut Self::MatrixCountersType, input: &Input, target: &[Target; $len], tanh_width: u32) {
                for i in 0..$len {
                    self[i].increment_matrix_counters(&mut counters[i], input, &target[i], tanh_width);
                }
            }
            fn bitpack(&mut self, counters: &Self::MatrixCountersType, len: usize) {
                for i in 0..$len {
                    self[i].bitpack(&counters[i], len);
                }
            }
        }
    };
}

impl_matrixbitincrement_for_array_of_matrixbitincrement!(1);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(2);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(3);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(4);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(8);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(10);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(32);

pub trait OptimizeInput<Weights, Target> {
    fn optimize(&mut self, weights: &Weights, target: &Target);
}

impl<Input: BitLen + FlipBit + GetBit, Weights: Apply<Input, Target>, Target: HammingDistance> OptimizeInput<Weights, Target> for Input {
    fn optimize(&mut self, weights: &Weights, target: &Target) {
        let mut cur_hd = weights.apply(self).hamming_distance(target);
        for b in 0..Input::BIT_LEN {
            self.flip_bit(b);
            let new_hd = weights.apply(self).hamming_distance(target);
            if new_hd < cur_hd {
                cur_hd = new_hd;
            } else {
                self.flip_bit(b);
            }
        }
    }
}

pub trait TrainAutoencoder<Input, Embedding, Decoder> {
    fn train_autoencoder<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Input>) -> Self;
}

impl<
        Input: Sync + Send + HammingDistance + BitLen,
        Encoder: MatrixBitIncrement<Input, Embedding> + Sync + Send + Apply<Input, Embedding> + HammingDistance + Copy,
        Embedding: Sync + Send + OptimizeInput<Decoder, Input>,
        Decoder: MatrixBitIncrement<Embedding, Input> + Sync + Send + Apply<Embedding, Input>,
    > TrainAutoencoder<Input, Embedding, Decoder> for Encoder
where
    Encoder::MatrixCountersType: Default + Sync + Send + ElementwiseAdd,
    Decoder::MatrixCountersType: Default + Sync + Send + ElementwiseAdd,
    rand::distributions::Standard: rand::distributions::Distribution<Encoder>,
    rand::distributions::Standard: rand::distributions::Distribution<Decoder>,
    <Decoder as MatrixBitIncrement<Embedding, Input>>::MatrixCountersType: std::marker::Send,
{
    fn train_autoencoder<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Input>) -> Self {
        let mut encoder: Encoder = rng.gen();
        let mut decoder: Decoder = rng.gen();

        for p in 0..7 {
            let decoder_counters = examples
                .par_iter()
                .fold(
                    || Decoder::MatrixCountersType::default(),
                    |mut counter, patch| {
                        let embedding = encoder.apply(&patch);
                        decoder.increment_matrix_counters(&mut counter, &embedding, patch, 3);
                        counter
                    },
                )
                .reduce(
                    || Decoder::MatrixCountersType::default(),
                    |mut a, b| {
                        a.elementwise_add(&b);
                        a
                    },
                );
            decoder.bitpack(&decoder_counters, examples.len());

            let encoder_counters = examples
                .par_iter()
                .fold(
                    || Encoder::MatrixCountersType::default(),
                    |mut counter, patch| {
                        let mut embedding = encoder.apply(&patch);
                        for i in 0..3 {
                            embedding.optimize(&decoder, patch);
                        }
                        encoder.increment_matrix_counters(&mut counter, patch, &embedding, 3);
                        counter
                    },
                )
                .reduce(
                    || Encoder::MatrixCountersType::default(),
                    |mut a, b| {
                        a.elementwise_add(&b);
                        a
                    },
                );

            let old_encoder: Encoder = encoder;
            encoder.bitpack(&encoder_counters, examples.len());
            dbg!(encoder.hamming_distance(&old_encoder));

            let sum_hd: u64 = examples
                .par_iter()
                .map(|patch| {
                    let embedding = encoder.apply(patch);
                    let output = decoder.apply(&embedding);
                    output.hamming_distance(patch) as u64
                })
                .sum();
            println!("avg hd: {} / {}", sum_hd as f64 / examples.len() as f64, Input::BIT_LEN);
        }
        encoder
    }
}

pub trait Apply<I, O> {
    fn apply(&self, input: &I) -> O;
}

impl<I: HammingDistance + BitLen> Apply<I, u8> for [I; 8] {
    fn apply(&self, input: &I) -> u8 {
        let mut target = 0u8;
        for i in 0..8 {
            target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2)) as u8) << i;
        }
        target
    }
}

impl<I: HammingDistance + BitLen> Apply<I, u32> for [I; 32] {
    fn apply(&self, input: &I) -> u32 {
        let mut target = 0u32;
        for i in 0..32 {
            target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2)) as u32) << i;
        }
        target
    }
}
impl<I: HammingDistance + BitLen> Apply<I, u32> for I {
    fn apply(&self, input: &I) -> u32 {
        self.hamming_distance(input)
    }
}

macro_rules! impl_apply_for_array_output {
    ($len:expr) => {
        impl<I, O: Default + Copy, T: Apply<I, O>> Apply<I, [O; $len]> for [T; $len] {
            fn apply(&self, input: &I) -> [O; $len] {
                let mut target = [O::default(); $len];
                for i in 0..$len {
                    target[i] = self[i].apply(input);
                }
                target
            }
        }
    };
}
impl_apply_for_array_output!(1);
impl_apply_for_array_output!(2);
impl_apply_for_array_output!(3);
impl_apply_for_array_output!(4);
impl_apply_for_array_output!(5);
impl_apply_for_array_output!(6);
impl_apply_for_array_output!(7);
impl_apply_for_array_output!(8);
impl_apply_for_array_output!(10);

#[macro_use]
pub mod layers {
    use super::{Apply, BitLen, BitOr, GetBit, HammingDistance, SetBit};
    use bincode::{deserialize_from, serialize_into};
    use std::fs::File;
    use std::io::BufWriter;
    use std::path::Path;

    macro_rules! patch_2x2 {
        ($input:expr, $x:expr, $y:expr) => {
            [[$input[$x + 0][$y + 0], $input[$x + 0][$y + 1]], [$input[$x + 1][$y + 0], $input[$x + 1][$y + 1]]]
        };
    }

    macro_rules! patch_3x3 {
        ($input:expr, $x:expr, $y:expr) => {
            [
                [$input[$x + 0][$y + 0], $input[$x + 0][$y + 1], $input[$x + 0][$y + 2]],
                [$input[$x + 1][$y + 0], $input[$x + 1][$y + 1], $input[$x + 1][$y + 2]],
                [$input[$x + 2][$y + 0], $input[$x + 2][$y + 1], $input[$x + 2][$y + 2]],
            ]
        };
    }

    pub trait Conv2D<I, O> {
        fn conv2d(&self, input: &I) -> O;
    }

    macro_rules! impl_conv2d_2x2 {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, O: Default + Copy, W: Apply<[[I; 2]; 2], O>> Conv2D<[[I; $y_size]; $x_size], [[O; $y_size / 2]; $x_size / 2]> for W {
                fn conv2d(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size / 2]; $x_size / 2] {
                    let mut target = [[O::default(); $y_size / 2]; $x_size / 2];
                    for x in 0..($x_size / 2) {
                        let x_base = x * 2;
                        for y in 0..($y_size / 2) {
                            let y_base = y * 2;
                            target[x][y] = self.apply(&patch_2x2!(input, x_base, y_base));
                        }
                    }
                    target
                }
            }
        };
    }

    impl_conv2d_2x2!(32, 32);
    impl_conv2d_2x2!(24, 24);
    impl_conv2d_2x2!(28, 28);
    impl_conv2d_2x2!(16, 16);
    impl_conv2d_2x2!(14, 14);
    impl_conv2d_2x2!(8, 8);
    impl_conv2d_2x2!(7, 7);

    macro_rules! conv2d_3x3_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, W: Apply<[[I; 3]; 3], O>, O: Default + Copy> Conv2D<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]> for W {
                fn conv2d(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size]; $x_size] {
                    let mut target = [[O::default(); $y_size]; $x_size];
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            target[x + 1][y + 1] = self.apply(&patch_3x3!(input, x, y));
                        }
                    }
                    target
                }
            }
        };
    }
    conv2d_3x3_apply_trait!(32, 32);
    conv2d_3x3_apply_trait!(16, 16);
    conv2d_3x3_apply_trait!(8, 8);

    conv2d_3x3_apply_trait!(28, 28);
    conv2d_3x3_apply_trait!(14, 14);

    macro_rules! conv2d_3x3_apply_trait_no_pad {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, W: Apply<[[I; 3]; 3], O>, O: Default + Copy> Conv2D<[[I; $y_size]; $x_size], [[O; $y_size - 2]; $x_size - 2]> for W {
                fn conv2d(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size - 2]; $x_size - 2] {
                    let mut target = [[O::default(); $y_size - 2]; $x_size - 2];
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            target[x][y] = self.apply(&patch_3x3!(input, x, y));
                        }
                    }
                    target
                }
            }
        };
    }

    conv2d_3x3_apply_trait_no_pad!(32, 32);
    conv2d_3x3_apply_trait_no_pad!(30, 30);
    conv2d_3x3_apply_trait_no_pad!(28, 28);
    conv2d_3x3_apply_trait_no_pad!(26, 26);
    conv2d_3x3_apply_trait_no_pad!(24, 24);
    conv2d_3x3_apply_trait_no_pad!(22, 22);
    //conv2d_3x3_apply_trait_no_pad!(20, 20);
    //conv2d_3x3_apply_trait_no_pad!(18, 18);
    //conv2d_3x3_apply_trait_no_pad!(16, 16);
    //conv2d_3x3_apply_trait_no_pad!(14, 14);
    //conv2d_3x3_apply_trait_no_pad!(12, 12);
    //conv2d_3x3_apply_trait_no_pad!(10, 10);
    //conv2d_3x3_apply_trait_no_pad!(8, 8);
    //conv2d_3x3_apply_trait_no_pad!(6, 5);
    //conv2d_3x3_apply_trait_no_pad!(4, 4);

    pub trait OrPool<Output> {
        fn or_pool(&self) -> Output;
    }
    macro_rules! impl_orpool {
        ($x_size:expr, $y_size:expr) => {
            impl<Pixel: BitOr + Default + Copy> OrPool<[[Pixel; $y_size / 2]; $x_size / 2]> for [[Pixel; $y_size]; $x_size] {
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

    pub trait SaveLoad
    where
        Self: Sized,
    {
        fn write_to_fs(&self, path: &Path);
        fn new_from_fs(path: &Path) -> Option<Self>;
    }

    impl<T: serde::Serialize> SaveLoad for T
    where
        for<'de> T: serde::Deserialize<'de>,
    {
        fn write_to_fs(&self, path: &Path) {
            //let vec_params: Vec<[[[[u32; $input_len]; 3]; 3]; 16]> = self.iter().cloned().collect();
            let mut f = BufWriter::new(File::create(path).unwrap());
            serialize_into(&mut f, self).unwrap();
        }
        // This will return:
        // - Some if the file exists and is good
        // - None of the file does not exist
        // and will panic if the file is exists but is bad.
        fn new_from_fs(path: &Path) -> Option<Self> {
            File::open(&path).map(|f| deserialize_from(f).unwrap()).ok()
        }
    }
}

pub trait Image2D {
    type Pixel;
}

macro_rules! impl_image2d {
    ($x_size:expr, $y_size:expr) => {
        impl<Pixel> Image2D for [[Pixel; $y_size]; $y_size] {
            type Pixel = Pixel;
        }
    };
}

impl_image2d!(32, 32);
impl_image2d!(16, 16);
impl_image2d!(8, 8);
//impl_image2d!(3, 3);
//impl_image2d!(2, 2);

pub trait ExtractPatches<Patch> {
    fn patches(&self) -> Vec<Patch>;
}

macro_rules! impl_extract_patch_trait {
    ($x_size:expr, $y_size:expr) => {
        impl<Pixel: Copy> ExtractPatches<[[Pixel; 3]; 3]> for [[Pixel; $y_size]; $x_size] {
            fn patches(&self) -> Vec<[[Pixel; 3]; 3]> {
                let mut patches = Vec::with_capacity(($y_size - 2) * ($x_size - 2));
                for x in 0..$x_size - 2 {
                    for y in 0..$y_size - 2 {
                        patches.push(patch_3x3!(self, x, y));
                    }
                }
                patches
            }
        }
        impl<Pixel: Copy> ExtractPatches<[[Pixel; 2]; 2]> for [[Pixel; $y_size]; $x_size] {
            fn patches(&self) -> Vec<[[Pixel; 2]; 2]> {
                let mut patches = Vec::with_capacity(($y_size - 1) * ($x_size - 1));
                for x in 0..($x_size - 1) {
                    for y in 0..($y_size - 1) {
                        patches.push(patch_2x2!(self, x, y));
                    }
                }
                patches
            }
        }
    };
}

impl_extract_patch_trait!(32, 32);
impl_extract_patch_trait!(30, 30);
impl_extract_patch_trait!(28, 28);
impl_extract_patch_trait!(26, 26);
impl_extract_patch_trait!(24, 24);
impl_extract_patch_trait!(22, 22);
impl_extract_patch_trait!(20, 20);
impl_extract_patch_trait!(18, 18);
impl_extract_patch_trait!(16, 16);
impl_extract_patch_trait!(14, 14);
impl_extract_patch_trait!(12, 12);
impl_extract_patch_trait!(10, 10);
impl_extract_patch_trait!(8, 8);
impl_extract_patch_trait!(7, 7);
impl_extract_patch_trait!(6, 6);
impl_extract_patch_trait!(4, 4);
