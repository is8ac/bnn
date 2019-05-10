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
        pub fn load_images_bitpacked_u32(path: &Path, size: usize) -> Vec<[u32; 25]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[u32; 25]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image_words: [u32; 25] = [0; 25];
                for p in 0..784 {
                    let word_index = p / 32;
                    image_words[word_index] =
                        image_words[word_index] | (((images_bytes[p] > 128) as u32) << p % 32);
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
                [to_11(pixel[0]) as u32
                    | ((to_11(pixel[1]) as u32) << 11)
                    | ((to_10(pixel[2]) as u32) << 22)]
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

        pub fn load_images_from_base<T: Default + Copy + ConvertPixel>(
            base_path: &Path,
            n: usize,
        ) -> Vec<([[T; 32]; 32], usize)> {
            if n > 50000 {
                panic!("n must be <= 50,000");
            }
            (1..6)
                .map(|i| {
                    let mut file = File::open(&base_path.join(format!("data_batch_{}.bin", i)))
                        .expect("can't open data");

                    let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
                    let mut label: [u8; 1] = [0; 1];
                    let mut images: Vec<([[T; 32]; 32], usize)> = Vec::new();
                    for _ in 0..10000 {
                        file.read_exact(&mut label).expect("can't read label");
                        file.read_exact(&mut image_bytes)
                            .expect("can't read images");
                        let mut image = [[T::default(); 32]; 32];
                        for x in 0..32 {
                            for y in 0..32 {
                                let pixel = [
                                    image_bytes[(0 * 1024) + (y * 32) + x],
                                    image_bytes[(1 * 1024) + (y * 32) + x],
                                    image_bytes[(2 * 1024) + (y * 32) + x],
                                ];
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

pub trait GetBit {
    fn bit(&self, i: usize) -> bool;
}

impl GetBit for bool {
    #[inline(always)]
    fn bit(&self, _i: usize) -> bool {
        *self
    }
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

impl BitLen for bool {
    const BIT_LEN: usize = 1;
}

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

pub trait GetMirroredWords {
    fn get_mirrored_words(&self, i: usize) -> [u32; 2];
}

impl<T: GetWord + WordLen> GetMirroredWords for [T; 3] {
    #[inline(always)]
    fn get_mirrored_words(&self, i: usize) -> [u32; 2] {
        let input_x = i / T::WORD_LEN;
        let strip_i = i % T::WORD_LEN;
        match input_x {
            0 => [self[0].get_word(strip_i), self[2].get_word(strip_i)],
            1 => [self[1].get_word(strip_i), self[1].get_word(strip_i)],
            2 => [self[2].get_word(strip_i), self[0].get_word(strip_i)],
            _ => panic!(input_x),
        }
    }
}

impl<T: GetWord + WordLen> GetMirroredWords for [T; 2] {
    #[inline(always)]
    fn get_mirrored_words(&self, i: usize) -> [u32; 2] {
        let input_x = i / T::WORD_LEN;
        let strip_i = i % T::WORD_LEN;
        match input_x {
            0 => [self[0].get_word(strip_i), self[1].get_word(strip_i)],
            1 => [self[1].get_word(strip_i), self[0].get_word(strip_i)],
            _ => panic!(input_x),
        }
    }
}

impl<T: GetWord + WordLen> GetMirroredWords for [T; 5] {
    #[inline(always)]
    fn get_mirrored_words(&self, i: usize) -> [u32; 2] {
        let input_x = i / T::WORD_LEN;
        let strip_i = i % T::WORD_LEN;
        match input_x {
            0 => [self[0].get_word(strip_i), self[4].get_word(strip_i)],
            1 => [self[1].get_word(strip_i), self[3].get_word(strip_i)],
            2 => [self[2].get_word(strip_i), self[2].get_word(strip_i)],
            3 => [self[3].get_word(strip_i), self[1].get_word(strip_i)],
            4 => [self[4].get_word(strip_i), self[0].get_word(strip_i)],
            _ => panic!(input_x),
        }
    }
}

pub trait SolidBits {
    fn solid_bits(sign: bool) -> Self;
}
impl SolidBits for u32 {
    fn solid_bits(sign: bool) -> u32 {
        if sign {
            !0
        } else {
            0
        }
    }
}

macro_rules! impl_solidbits_for_3x3 {
    ($len:expr) => {
        impl<T: Copy + SolidBits> SolidBits for [T; $len] {
            fn solid_bits(sign: bool) -> Self {
                [T::solid_bits(sign); $len]
            }
        }
    };
}

impl_solidbits_for_3x3!(1);
impl_solidbits_for_3x3!(2);
impl_solidbits_for_3x3!(3);
impl_solidbits_for_3x3!(4);

pub trait BinaryFilter {
    fn binary_filter() -> Self;
}

macro_rules! impl_binaryfilter_for_array {
    ($len:expr) => {
        impl<T: SolidBits + Default + Copy> BinaryFilter for [[[[T; 3]; 3]; 32]; $len] {
            fn binary_filter() -> Self {
                let mut weights = [[[[T::default(); 3]; 3]; 32]; $len];
                for w in 0..$len {
                    for b in 0..32 {
                        let c = w * 32 + b;
                        for x in 0..3 {
                            for y in 0..3 {
                                weights[w][b][x][y] = T::solid_bits((c as u32).bit(x * 3 + y));
                            }
                        }
                    }
                }
                weights
            }
        }
    };
}
impl_binaryfilter_for_array!(1);
impl_binaryfilter_for_array!(2);
impl_binaryfilter_for_array!(3);
impl_binaryfilter_for_array!(4);
impl_binaryfilter_for_array!(8);
impl_binaryfilter_for_array!(16);

pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
}

impl HammingDistance for bool {
    fn hamming_distance(&self, other: &Self) -> u32 {
        (self ^ other) as u32
    }
}

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

//pub trait MirrorHammingDistance {
//    fn normal_hamming_distance(&self, input: &Self) -> u32;
//    fn fliped_hamming_distance(&self, input: &Self) -> u32;
//}
//
//impl<T: HammingDistance> MirrorHammingDistance for [T; 3] {
//    fn normal_hamming_distance(&self, input: &[T; 3]) -> u32 {
//        self[0].hamming_distance(&input[0])
//            + self[1].hamming_distance(&input[1])
//            + self[2].hamming_distance(&input[2])
//    }
//    fn fliped_hamming_distance(&self, input: &[T; 3]) -> u32 {
//        self[0].hamming_distance(&input[2])
//            + self[1].hamming_distance(&input[1])
//            + self[2].hamming_distance(&input[0])
//    }
//}
//
//impl<T: HammingDistance> MirrorHammingDistance for [T; 2] {
//    fn normal_hamming_distance(&self, input: &[T; 2]) -> u32 {
//        self[0].hamming_distance(&input[0]) + self[1].hamming_distance(&input[1])
//    }
//    fn fliped_hamming_distance(&self, input: &[T; 2]) -> u32 {
//        self[0].hamming_distance(&input[1]) + self[1].hamming_distance(&input[0])
//    }
//}
//
//impl<T: HammingDistance> MirrorHammingDistance for [T; 5] {
//    fn normal_hamming_distance(&self, input: &[T; 5]) -> u32 {
//        self[0].hamming_distance(&input[0])
//            + self[1].hamming_distance(&input[1])
//            + self[2].hamming_distance(&input[2])
//            + self[3].hamming_distance(&input[3])
//            + self[4].hamming_distance(&input[4])
//    }
//    fn fliped_hamming_distance(&self, input: &[T; 5]) -> u32 {
//        self[0].hamming_distance(&input[4])
//            + self[1].hamming_distance(&input[3])
//            + self[2].hamming_distance(&input[2])
//            + self[3].hamming_distance(&input[1])
//            + self[4].hamming_distance(&input[0])
//    }
//}

pub trait ElementwiseAdd {
    fn elementwise_add(&mut self, other: &Self);
}

impl ElementwiseAdd for u32 {
    fn elementwise_add(&mut self, other: &u32) {
        *self += other;
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
    fn increment_counters(&self, counters: &mut Self::CountersType);
    fn compare_and_bitpack(
        &mut self,
        counters_0: &Self::CountersType,
        counters_1: &Self::CountersType,
        len: usize,
    );
}

impl ArrayBitIncrement for u32 {
    type CountersType = [u32; 32];
    fn increment_counters(&self, counters: &mut Self::CountersType) {
        for b in 0..32 {
            counters[b] += (self >> b) & 1
        }
    }
    fn compare_and_bitpack(
        &mut self,
        counters_0: &Self::CountersType,
        counters_1: &Self::CountersType,
        len: usize,
    ) {
        for b in 0..32 {
            let diff = (counters_0[b] as f64 - counters_1[b] as f64) / len as f64;
            if diff.abs() > 0.16 {
                self.set_bit(b, (counters_0[b] > counters_1[b]));
            }
        }
    }
}

//impl ArrayBitIncrement for u8 {
//    type CountersType = [u32; 8];
//    fn increment_counters(&self, counters: &mut Self::CountersType) {
//        for b in 0..8 {
//            counters[b] += ((self >> b) & 1) as u32
//        }
//    }
//    fn compare_and_bitpack(
//        &mut self,
//        counters_0: &Self::CountersType,
//        counters_1: &Self::CountersType,
//        len: usize,
//    ) {
//        let mut target = 0u8;
//        for b in 0..8 {
//            let diff = (counters_0[b] as f64 - counters_1[b] as f64) / len as f64;
//            if diff.abs() > 0.9 {
//                target |= ((counters_0[b] < counters_1[b]) as u8) << b;
//            }
//        }
//        *self = target;
//    }
//}

macro_rules! impl_bitincrement_for_array {
    ($len:expr) => {
        impl<T: ArrayBitIncrement + Default + Copy> ArrayBitIncrement for [T; $len] {
            type CountersType = [T::CountersType; $len];
            fn increment_counters(&self, counters: &mut Self::CountersType) {
                for i in 0..$len {
                    self[i].increment_counters(&mut counters[i]);
                }
            }
            fn compare_and_bitpack(
                &mut self,
                counters_0: &Self::CountersType,
                counters_1: &Self::CountersType,
                len: usize,
            ) {
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
    fn increment_matrix_counters(
        &self,
        counters: &mut Self::MatrixCountersType,
        input: &Input,
        target: &Target,
    );
    fn bitpack(&mut self, counters: &Self::MatrixCountersType, len: usize);
}

impl<Input: ArrayBitIncrement + BitLen + HammingDistance> MatrixBitIncrement<Input, bool>
    for Input
{
    type MatrixCountersType = [Input::CountersType; 2];
    fn increment_matrix_counters(
        &self,
        counters: &mut Self::MatrixCountersType,
        input: &Input,
        target: &bool,
    ) {
        let act = self.hamming_distance(&input) > (Input::BIT_LEN as u32 / 2);
        if act != *target {
            input.increment_counters(&mut counters[*target as usize]);
        }
    }
    fn bitpack(&mut self, counters: &Self::MatrixCountersType, len: usize) {
        self.compare_and_bitpack(&counters[0], &counters[1], len)
    }
}

// TODO pass current state.
impl<Input: ArrayBitIncrement + Copy + Default + HammingDistance + BitLen>
    MatrixBitIncrement<Input, u8> for [Input; 8]
{
    type MatrixCountersType = [[Input::CountersType; 2]; 8];
    fn increment_matrix_counters(
        &self,
        counters: &mut Self::MatrixCountersType,
        input: &Input,
        target: &u8,
    ) {
        for b in 0..8 {
            let act = self[b].hamming_distance(&input) > (Input::BIT_LEN as u32 / 2);
            if act != target.bit(b) {
                input.increment_counters(&mut counters[b][target.bit(b) as usize]);
            }
        }
    }
    fn bitpack(&mut self, counters: &Self::MatrixCountersType, len: usize) {
        for b in 0..8 {
            self[b].compare_and_bitpack(&counters[b][0], &counters[b][1], len);
        }
    }
}

impl<Input: ArrayBitIncrement + Copy + Default + HammingDistance + BitLen>
    MatrixBitIncrement<Input, u32> for [Input; 32]
{
    type MatrixCountersType = [[Input::CountersType; 2]; 32];
    fn increment_matrix_counters(
        &self,
        counters: &mut Self::MatrixCountersType,
        input: &Input,
        target: &u32,
    ) {
        for b in 0..32 {
            let act = self[b].hamming_distance(&input) > (Input::BIT_LEN as u32 / 2);
            if act != target.bit(b) {
                input.increment_counters(&mut counters[b][target.bit(b) as usize]);
            }
        }
    }
    fn bitpack(&mut self, counters: &Self::MatrixCountersType, len: usize) {
        for b in 0..32 {
            self[b].compare_and_bitpack(&counters[b][0], &counters[b][1], len);
        }
    }
}

macro_rules! impl_matrixbitincrement_for_array_of_matrixbitincrement {
    ($len:expr) => {
        impl<Input, Target, MatrixBits: MatrixBitIncrement<Input, Target> + Copy + Default>
            MatrixBitIncrement<Input, [Target; $len]> for [MatrixBits; $len]
        {
            type MatrixCountersType = [MatrixBits::MatrixCountersType; $len];
            fn increment_matrix_counters(
                &self,
                counters: &mut Self::MatrixCountersType,
                input: &Input,
                target: &[Target; $len],
            ) {
                for i in 0..$len {
                    self[i].increment_matrix_counters(&mut counters[i], input, &target[i]);
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

impl<Input: BitLen + FlipBit + GetBit, Weights: Apply<Input, Target>, Target: HammingDistance>
    OptimizeInput<Weights, Target> for Input
{
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
        Encoder: MatrixBitIncrement<Input, Embedding>
            + Sync
            + Send
            + Apply<Input, Embedding>
            + HammingDistance
            + Copy,
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
                        decoder.increment_matrix_counters(&mut counter, &embedding, patch);
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
                        encoder.increment_matrix_counters(&mut counter, patch, &embedding);
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
            println!(
                "avg hd: {} / {}",
                sum_hd as f64 / examples.len() as f64,
                Input::BIT_LEN
            );
        }
        encoder
    }
}

pub trait TrainEncoderSupervised<Input, Embedding, Decoder, Target> {
    fn train_encoder_supervised<RNG: rand::Rng>(
        rng: &mut RNG,
        examples: &Vec<(Input, Target)>,
    ) -> Self;
}

impl<
        Input: Sync + Send,
        Encoder: MatrixBitIncrement<Input, Embedding> + Sync + Send + Apply<Input, Embedding>,
        Embedding: Sync + Send + OptimizeInput<Decoder, Target>,
        Decoder: MatrixBitIncrement<Embedding, Target> + Sync + Send + Apply<Embedding, Target>,
        Target: Sync + Send + HammingDistance,
    > TrainEncoderSupervised<Input, Embedding, Decoder, Target> for Encoder
where
    Encoder::MatrixCountersType: Default + Sync + Send + ElementwiseAdd,
    Decoder::MatrixCountersType: Default + Sync + Send + ElementwiseAdd,
    rand::distributions::Standard: rand::distributions::Distribution<Encoder>,
    rand::distributions::Standard: rand::distributions::Distribution<Decoder>,
    <Decoder as MatrixBitIncrement<Embedding, Target>>::MatrixCountersType: Send,
{
    fn train_encoder_supervised<RNG: rand::Rng>(
        rng: &mut RNG,
        examples: &Vec<(Input, Target)>,
    ) -> Self {
        let mut encoder: Encoder = rng.gen();
        let mut decoder: Decoder = rng.gen();
        for p in 0..7 {
            let decoder_counters = examples
                .par_iter()
                .fold(
                    || Decoder::MatrixCountersType::default(),
                    |mut counter, (input, target)| {
                        let embedding = encoder.apply(&input);
                        decoder.increment_matrix_counters(&mut counter, &embedding, target);
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
                    |mut counter, (input, target)| {
                        let mut embedding = encoder.apply(&input);
                        for i in 0..3 {
                            embedding.optimize(&decoder, target);
                        }
                        encoder.increment_matrix_counters(&mut counter, input, &embedding);
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

            encoder.bitpack(&encoder_counters, examples.len());

            let sum_hd: u64 = examples
                .par_iter()
                .map(|(patch, target)| {
                    let embedding = encoder.apply(patch);
                    let output = decoder.apply(&embedding);
                    output.hamming_distance(target) as u64
                })
                .sum();

            println!("avg hd: {:}", sum_hd as f64 / examples.len() as f64);
        }
        encoder
    }
}

pub trait Apply<I, O> {
    fn apply(&self, input: &I) -> O;
}

impl<I: HammingDistance + BitLen> Apply<I, bool> for I {
    fn apply(&self, input: &I) -> bool {
        self.hamming_distance(input) > (I::BIT_LEN as u32 / 2)
    }
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
    use super::{Apply, BitLen, GetBit, HammingDistance, SetBit};
    use bincode::{deserialize_from, serialize_into};
    use std::fs::File;
    use std::io::BufWriter;
    use std::path::Path;

    //impl<I: HammingDistance + BitLen> Apply<[I; 3], u32> for [[I; 3]; 16]
    //where
    //    [I; 3]: MirrorHammingDistance,
    //{
    //    fn apply(&self, input: &[I; 3]) -> u32 {
    //        let threshold: u32 = (I::BIT_LEN * 3) as u32 / 2;
    //        let mut target = 0u32;
    //        for i in 0..16 {
    //            target |= ((self[i].normal_hamming_distance(input) > threshold) as u32) << i;
    //            target |= ((self[i].fliped_hamming_distance(input) > threshold) as u32) << (16 + i);
    //        }
    //        target
    //    }
    //}

    //impl<I: HammingDistance + BitLen> Apply<[I; 2], u32> for [[I; 2]; 16]
    //where
    //    [I; 2]: MirrorHammingDistance,
    //{
    //    fn apply(&self, input: &[I; 2]) -> u32 {
    //        let threshold: u32 = (I::BIT_LEN * 2) as u32 / 2;
    //        let mut target = 0u32;
    //        for i in 0..16 {
    //            target |= ((self[i].normal_hamming_distance(input) > threshold) as u32) << i;
    //            target |= ((self[i].fliped_hamming_distance(input) > threshold) as u32) << (16 + i);
    //        }
    //        target
    //    }
    //}

    //impl<I: HammingDistance + BitLen> Apply<[I; 5], u32> for [[I; 5]; 16]
    //where
    //    [I; 5]: MirrorHammingDistance,
    //{
    //    fn apply(&self, input: &[I; 5]) -> u32 {
    //        let threshold: u32 = (I::BIT_LEN * 5) as u32 / 2;
    //        let mut target = 0u32;
    //        for i in 0..16 {
    //            target |= ((self[i].normal_hamming_distance(input) > threshold) as u32) << i;
    //            target |= ((self[i].fliped_hamming_distance(input) > threshold) as u32) << (16 + i);
    //        }
    //        target
    //    }
    //}

    // The u8 is a base 2 integer. But the u32 is just 32 bits.
    //impl Apply<[[[u8; 3]; 3]; 3], u32> for [[[u32; 3]; 3]; 3] {
    //    fn apply(&self, input: &[[[u8; 3]; 3]; 3]) -> u32 {
    //        let mut target = 0u32;
    //        for b in 0..32 {
    //            let mut sum_act = 0i32;
    //            for x in 0..3 {
    //                for y in 0..3 {
    //                    for c in 0..3 {
    //                        if self[x][y][c].bit(b) {
    //                            sum_act += input[x][y][c] as i32;
    //                        } else {
    //                            sum_act -= input[x][y][c] as i32;
    //                        }
    //                    }
    //                }
    //            }
    //            target.set_bit(b, sum_act > 0);
    //        }
    //        target
    //    }
    //}

    macro_rules! patch_2x2 {
        ($input:expr, $x:expr, $y:expr) => {
            [
                [$input[$x + 0][$y + 0], $input[$x + 0][$y + 1]],
                [$input[$x + 1][$y + 0], $input[$x + 1][$y + 1]],
            ]
        };
    }

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
    macro_rules! patch_5x5 {
        ($input:expr, $x:expr, $y:expr) => {
            [
                [
                    $input[$x + 0][$y + 0],
                    $input[$x + 0][$y + 1],
                    $input[$x + 0][$y + 2],
                    $input[$x + 0][$y + 3],
                    $input[$x + 0][$y + 4],
                ],
                [
                    $input[$x + 1][$y + 0],
                    $input[$x + 1][$y + 1],
                    $input[$x + 1][$y + 2],
                    $input[$x + 1][$y + 3],
                    $input[$x + 1][$y + 4],
                ],
                [
                    $input[$x + 2][$y + 0],
                    $input[$x + 2][$y + 1],
                    $input[$x + 2][$y + 2],
                    $input[$x + 2][$y + 3],
                    $input[$x + 2][$y + 4],
                ],
                [
                    $input[$x + 3][$y + 0],
                    $input[$x + 3][$y + 1],
                    $input[$x + 3][$y + 2],
                    $input[$x + 3][$y + 3],
                    $input[$x + 3][$y + 4],
                ],
                [
                    $input[$x + 4][$y + 0],
                    $input[$x + 4][$y + 1],
                    $input[$x + 4][$y + 2],
                    $input[$x + 4][$y + 3],
                    $input[$x + 4][$y + 4],
                ],
            ]
        };
    }

    pub trait Conv2D<I, O> {
        fn conv2d(&self, input: &I) -> O;
    }

    macro_rules! patch_conv_2x2_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, O: Default + Copy, W: Apply<[[I; 2]; 2], O>>
                Apply<[[I; $y_size]; $x_size], [[O; $y_size / 2]; $x_size / 2]> for W
            {
                fn apply(
                    &self,
                    input: &[[I; $y_size]; $x_size],
                ) -> [[O; $y_size / 2]; $x_size / 2] {
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

    //patch_conv_2x2_apply_trait!(32, 32);
    //patch_conv_2x2_apply_trait!(28, 28);
    //patch_conv_2x2_apply_trait!(16, 16);
    //patch_conv_2x2_apply_trait!(14, 14);
    //patch_conv_2x2_apply_trait!(8, 8);
    //patch_conv_2x2_apply_trait!(7, 7);

    macro_rules! conv2d_3x3_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, W: Apply<[[I; 3]; 3], O>, O: Default + Copy>
                Conv2D<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]> for W
            {
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
            impl<I: Copy, W: Apply<[[I; 3]; 3], O>, O: Default + Copy>
                Conv2D<[[I; $y_size]; $x_size], [[O; $y_size - 2]; $x_size - 2]> for W
            {
                fn conv2d(
                    &self,
                    input: &[[I; $y_size]; $x_size],
                ) -> [[O; $y_size - 2]; $x_size - 2] {
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
    //conv2d_3x3_apply_trait_no_pad!(26, 26);
    //conv2d_3x3_apply_trait_no_pad!(24, 24);
    //conv2d_3x3_apply_trait_no_pad!(22, 22);
    //conv2d_3x3_apply_trait_no_pad!(20, 20);
    //conv2d_3x3_apply_trait_no_pad!(18, 18);
    //conv2d_3x3_apply_trait_no_pad!(16, 16);
    //conv2d_3x3_apply_trait_no_pad!(14, 14);
    //conv2d_3x3_apply_trait_no_pad!(12, 12);
    //conv2d_3x3_apply_trait_no_pad!(10, 10);
    //conv2d_3x3_apply_trait_no_pad!(8, 8);
    //conv2d_3x3_apply_trait_no_pad!(6, 5);
    //conv2d_3x3_apply_trait_no_pad!(4, 4);

    macro_rules! conv5x5_apply_trait_no_pad {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, W: Apply<[[I; 5]; 5], O>, O: Default + Copy>
                Apply<[[I; $y_size]; $x_size], [[O; $y_size - 4]; $x_size - 4]> for W
            {
                fn apply(
                    &self,
                    input: &[[I; $y_size]; $x_size],
                ) -> [[O; $y_size - 4]; $x_size - 4] {
                    let mut target = [[O::default(); $y_size - 4]; $x_size - 4];
                    for x in 0..$x_size - 4 {
                        for y in 0..$y_size - 4 {
                            target[x][y] = self.apply(&patch_5x5!(input, x, y));
                        }
                    }
                    target
                }
            }
        };
    }

    //conv5x5_apply_trait_no_pad!(32, 32);

    //macro_rules! conv5x5_apply_trait {
    //    ($x_size:expr, $y_size:expr, $output_len:expr) => {
    //        impl<I: Copy + BitLen + HammingDistance>
    //            Apply<[[I; $y_size]; $x_size], [[[u32; $output_len]; $y_size]; $x_size]>
    //            for [[[[I; 5]; 5]; 16]; $output_len]
    //        {
    //            fn apply(
    //                &self,
    //                input: &[[I; $y_size]; $x_size],
    //            ) -> [[[u32; $output_len]; $y_size]; $x_size] {
    //                let mut target = [[[0u32; $output_len]; $y_size]; $x_size];
    //                for x in 0..$x_size - 4 {
    //                    for y in 0..$y_size - 4 {
    //                        target[x + 2][y + 2] = self.apply(&patch_5x5!(input, x, y));
    //                    }
    //                }
    //                target
    //            }
    //        }
    //    };
    //}

    //conv5x5_apply_trait!(32, 32, 1);
    //conv5x5_apply_trait!(16, 16, 1);
    //conv5x5_apply_trait!(8, 8, 1);

    //conv5x5_apply_trait!(32, 32, 2);
    //conv5x5_apply_trait!(16, 16, 2);
    //conv5x5_apply_trait!(8, 8, 2);

    //conv5x5_apply_trait!(32, 32, 3);
    //conv5x5_apply_trait!(16, 16, 3);
    //conv5x5_apply_trait!(8, 8, 3);

    //conv5x5_apply_trait!(32, 32, 4);
    //conv5x5_apply_trait!(16, 16, 4);
    //conv5x5_apply_trait!(8, 8, 4);

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

    pub trait InvertibleDownsample<OutputImage> {
        fn i_rev_pool(&self) -> OutputImage;
    }

    macro_rules! impl_invertible_downsample_for_image {
        ($x_size:expr, $y_size:expr) => {
            impl<Pixel: Default + Copy>
                InvertibleDownsample<[[[[Pixel; 2]; 2]; $y_size / 2]; $x_size / 2]>
                for [[Pixel; $y_size]; $x_size]
            {
                fn i_rev_pool(&self) -> [[[[Pixel; 2]; 2]; $y_size / 2]; $x_size / 2] {
                    let mut target = [[[[Pixel::default(); 2]; 2]; $y_size / 2]; $x_size / 2];
                    for x in 0..($x_size / 2) {
                        let x_base = x * 2;
                        for y in 0..($y_size / 2) {
                            let y_base = y * 2;
                            target[x][y] = patch_2x2!(self, x_base, y_base);
                        }
                    }
                    target
                }
            }
        };
    }
    impl_invertible_downsample_for_image!(32, 32);
    impl_invertible_downsample_for_image!(16, 16);
    impl_invertible_downsample_for_image!(8, 8);
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
        impl<Pixel: Copy> ExtractPatches<[[Pixel; 5]; 5]> for [[Pixel; $y_size]; $x_size] {
            fn patches(&self) -> Vec<[[Pixel; 5]; 5]> {
                let mut patches = Vec::with_capacity(($y_size - 4) * ($x_size - 4));
                for x in 0..$x_size - 4 {
                    for y in 0..$y_size - 4 {
                        patches.push(patch_5x5!(self, x, y));
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

pub trait ConcatImages<A, B> {
    fn concat_images(input_a: &A, input_b: &B) -> Self;
}

macro_rules! impl_concat_image {
    ($x_size:expr, $y_size:expr) => {
        impl<A: Copy + Default, B: Copy + Default>
            ConcatImages<[[A; $y_size]; $x_size], [[B; $y_size]; $x_size]>
            for [[(A, B); $y_size]; $x_size]
        {
            fn concat_images(
                input_a: &[[A; $y_size]; $x_size],
                input_b: &[[B; $y_size]; $x_size],
            ) -> [[(A, B); $y_size]; $x_size] {
                let mut target = <[[(A, B); $y_size]; $x_size]>::default();
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        target[x][y] = (input_a[x][y], input_b[x][y]);
                    }
                }
                target
            }
        }
    };
}

impl_concat_image!(32, 32);
impl_concat_image!(16, 16);
impl_concat_image!(8, 8);

pub fn vec_concat_2_examples<A: Sync, B: Sync, T: ConcatImages<A, B> + Sync + Send>(
    a: &Vec<(A, usize)>,
    b: &Vec<(B, usize)>,
) -> Vec<(T, usize)> {
    assert_eq!(a.len(), b.len());
    a.par_iter()
        .zip(b.par_iter())
        .map(|((a_image, a_class), (b_image, b_class))| {
            assert_eq!(a_class, b_class);
            (T::concat_images(a_image, b_image), *a_class)
        })
        .collect()
}

pub mod train {
    use super::layers::{Conv2D, SaveLoad};
    use super::{
        Apply, BinaryFilter, BitLen, ExtractPatches, FlipBit, FlipBitIndexed, GetBit, GetPatch,
        HammingDistance, SetBit,
    };
    use rand::prelude::*;
    use rayon::prelude::*;
    use std::iter;
    use std::marker::PhantomData;
    use std::path::Path;
    use time::PreciseTime;

    struct CacheItem<Input, Weights, Embedding> {
        weights_type: PhantomData<Weights>,
        input: Input,
        class: usize,
        embedding: Embedding,
        output: [u32; 10],
        bit_losses: [f64; 2],
        cur_embedding_act: u32,
    }

    fn loss_from_scaled_output(input: &[f64; 10], true_class: usize) -> f64 {
        let mut exp = [0f64; 10];
        let mut sum_exp = 0f64;
        for c in 0..10 {
            exp[c] = (input[c]).exp();
            sum_exp += exp[c];
        }
        let sum_loss: f64 = exp
            .iter()
            .enumerate()
            .map(|(c, x)| (((x / sum_exp) - (c == true_class) as u8 as f64).powi(2)))
            .sum();
        //sum_loss / 10f64
        sum_loss
    }

    fn loss_from_embedding<Embedding: HammingDistance + BitLen>(
        embedding: &Embedding,
        head: &[Embedding; 10],
        true_class: usize,
    ) -> f64 {
        let mut scaled = [0f64; 10];
        for c in 0..10 {
            let sum = head[c].hamming_distance(&embedding);
            scaled[c] = sum as f64 / <Embedding>::BIT_LEN as f64;
        }

        loss_from_scaled_output(&scaled, true_class)
    }

    impl<
            Input: HammingDistance + BitLen + GetBit + Sync + Copy,
            Embedding: HammingDistance + BitLen + GetBit + SetBit + Default + Copy,
            Weights: Apply<Input, Embedding>,
        > CacheItem<Input, Weights, Embedding>
    {
        fn compute_embedding(&mut self, weights: &Weights) {
            self.embedding = weights.apply(&self.input);
        }
        fn update_embedding_bit(&mut self, weights_patch: &Input, embedding_bit_index: usize) {
            self.embedding.set_bit(
                embedding_bit_index,
                weights_patch.hamming_distance(&self.input) > (<Input>::BIT_LEN / 2) as u32,
            );
        }
        fn compute_embedding_act(&mut self, weights_patch: &Input) {
            self.cur_embedding_act = weights_patch.hamming_distance(&self.input);
        }
        fn compute_output(&mut self, head: &[Embedding; 10]) {
            for c in 0..10 {
                let sum = head[c].hamming_distance(&self.embedding);
                self.output[c] = sum;
            }
        }
        fn compute_bit_losses(&mut self, head: &[Embedding; 10], embedding_bit_index: usize) {
            let mut embedding = self.embedding;
            embedding.set_bit(embedding_bit_index, false);
            self.bit_losses[0] = loss_from_embedding(&embedding, head, self.class as usize);
            embedding.set_bit(embedding_bit_index, true);
            self.bit_losses[1] = loss_from_embedding(&embedding, head, self.class as usize);
        }
        fn update_head_output_from_bit(
            &mut self,
            class_index: usize,
            embedding_bit_index: usize,
            head_bit: bool,
        ) {
            let embedding_bit = self.embedding.bit(embedding_bit_index);
            if head_bit ^ embedding_bit {
                // if !=, changing will decrease activation
                self.output[class_index] -= 1;
            } else {
                self.output[class_index] += 1;
            };
        }
        fn update_cur_embedding_act_from_bit(&mut self, input_bit_index: usize, weights_bit: bool) {
            //let input_bit = self.input.bit(input_bit_index);
            if weights_bit ^ self.input.bit(input_bit_index) {
                // if !=, changing will decrease activation
                self.cur_embedding_act -= 1;
            } else {
                self.cur_embedding_act += 1;
            };
        }
        fn loss_from_head_bit(
            &self,
            embedding_bit_index: usize,
            class_index: usize,
            head_bit: bool,
        ) -> f64 {
            let input_bit = self.embedding.bit(embedding_bit_index);
            let mut output = self.output;
            if head_bit ^ input_bit {
                // if !=, changing will decrease activation
                output[class_index] = self.output[class_index] - 1;
            } else {
                output[class_index] = self.output[class_index] + 1;
            };
            let mut scaled = [0f64; 10];
            for c in 0..10 {
                scaled[c] = output[c] as f64 / <Embedding>::BIT_LEN as f64;
            }
            loss_from_scaled_output(&scaled, self.class as usize)
        }
        #[inline(always)]
        fn loss_from_bit(&self, input_bit_index: usize, weights_bit: bool) -> f64 {
            //let input_bit = self.input.bit(input_bit_index);
            let new_act = if weights_bit ^ self.input.bit(input_bit_index) {
                // if !=, changing will decrease activation
                self.cur_embedding_act - 1
            } else {
                self.cur_embedding_act + 1
            };
            self.bit_losses[(new_act > (<Input>::BIT_LEN / 2) as u32) as usize]
        }
        fn true_loss(&self, weights: &Weights, head: &[Embedding; 10]) -> f64 {
            let embedding = weights.apply(&self.input);
            loss_from_embedding(&embedding, head, self.class as usize)
        }
        fn is_correct(&self) -> bool {
            self.output
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| *v)
                .unwrap()
                .0
                == self.class as usize
        }
        fn new(input: &Input, class: usize) -> Self {
            CacheItem {
                weights_type: PhantomData,
                input: *input,
                embedding: Embedding::default(),
                output: [0u32; 10],
                bit_losses: [0f64; 2],
                class: class,
                cur_embedding_act: 0u32,
            }
        }
    }

    pub struct CacheBatch<Input, Weights, Embedding> {
        items: Vec<CacheItem<Input, Weights, Embedding>>,
        weights: Weights,
        head: [Embedding; 10],
        embedding_bit_index: usize,
        embedding_is_clean: bool,
        output_is_clean: bool,
    }

    impl<
            Input: HammingDistance + Sync + Send + BitLen + GetBit + Copy,
            Weights: GetPatch<Input> + Send + FlipBitIndexed + Sync + Copy + Apply<Input, Embedding>,
            Embedding: Send + Sync + Copy + FlipBit + HammingDistance + GetBit + BitLen + SetBit + Default,
        > CacheBatch<Input, Weights, Embedding>
    {
        fn new(weights: &Weights, head: &[Embedding; 10], examples: &[(Input, usize)]) -> Self {
            let weights_patch = weights.get_patch(0);
            CacheBatch {
                items: examples
                    .par_iter()
                    .map(|(input, class)| {
                        let mut cache = CacheItem::new(input, *class as usize);
                        cache.compute_embedding(weights);
                        cache.compute_output(head);
                        cache.compute_bit_losses(head, 0);
                        cache.compute_embedding_act(&weights_patch);
                        cache
                    })
                    .collect(),
                weights: *weights,
                head: *head,
                embedding_bit_index: 0,
                embedding_is_clean: true,
                output_is_clean: true,
            }
        }
        fn head_bit_loss(&mut self, embedding_bit_index: usize, class_index: usize) -> f64 {
            if !self.embedding_is_clean {
                let embedding_bit_index = self.embedding_bit_index;
                let weights_patch = self.weights.get_patch(self.embedding_bit_index);
                let _: Vec<_> = self
                    .items
                    .par_iter_mut()
                    .map(|cache| cache.update_embedding_bit(&weights_patch, embedding_bit_index))
                    .collect();
                self.embedding_is_clean = true;
            }
            let head = self.head;
            if !self.output_is_clean {
                let _: Vec<_> = self
                    .items
                    .par_iter_mut()
                    .map(|cache| cache.compute_output(&head))
                    .collect();
                self.output_is_clean = true;
            }
            let head_bit = self.head[class_index].bit(embedding_bit_index);
            let sum_loss: f64 = self
                .items
                .par_iter()
                .map(|cache| cache.loss_from_head_bit(embedding_bit_index, class_index, head_bit))
                .sum();
            sum_loss as f64 / self.items.len() as f64
        }
        fn bit_loss(&mut self, input_bit_index: usize) -> f64 {
            let cur_mut_bit_val = self
                .weights
                .get_patch(self.embedding_bit_index)
                .bit(input_bit_index);
            let sum_loss: f64 = self
                .items
                .par_iter()
                .map(|cache| cache.loss_from_bit(input_bit_index, cur_mut_bit_val))
                .sum();
            sum_loss / self.items.len() as f64
        }
        fn true_loss(&mut self) -> f64 {
            let sum_loss: f64 = self
                .items
                .par_iter()
                .map(|cache| cache.true_loss(&self.weights, &self.head))
                .sum();
            sum_loss / self.items.len() as f64
        }
        fn acc(&mut self) -> f64 {
            let head = self.head;
            let sum_iscorrect: u64 = self
                .items
                .par_iter_mut()
                .map(|cache| {
                    cache.compute_output(&head);
                    cache.is_correct() as u64
                })
                .sum();
            sum_iscorrect as f64 / self.items.len() as f64
        }
        fn flip_weights_bit(&mut self, input_bit_index: usize) {
            self.embedding_is_clean = false;
            self.output_is_clean = false;
            let cur_mut_bit_val = self
                .weights
                .get_patch(self.embedding_bit_index)
                .bit(input_bit_index);
            let _: Vec<_> = self
                .items
                .par_iter_mut()
                .map(|cache| {
                    cache.update_cur_embedding_act_from_bit(input_bit_index, cur_mut_bit_val)
                })
                .collect();
            self.weights
                .flip_bit_indexed(self.embedding_bit_index, input_bit_index);
        }
        fn flip_head_bit(&mut self, class: usize, embedding_bit_index: usize) {
            let head_bit = self.head[class].bit(embedding_bit_index);
            let _: Vec<_> = self
                .items
                .par_iter_mut()
                .map(|cache| {
                    cache.update_head_output_from_bit(class, embedding_bit_index, head_bit)
                })
                .collect();
            self.output_is_clean = true;
            self.head[class].flip_bit(embedding_bit_index);
        }
        fn transition_embedding_bit(&mut self, new_embedding_bit_index: usize) {
            let old_embedding_bit_index = self.embedding_bit_index;
            let old_weights_patch = self.weights.get_patch(old_embedding_bit_index);
            let new_weights_patch = self.weights.get_patch(new_embedding_bit_index);
            let head = self.head;
            let _: Vec<_> = self
                .items
                .par_iter_mut()
                .map(|cache| {
                    cache.update_embedding_bit(&old_weights_patch, old_embedding_bit_index);
                    cache.compute_embedding_act(&new_weights_patch);
                    cache.compute_bit_losses(&head, new_embedding_bit_index);
                })
                .collect();
            self.embedding_bit_index = new_embedding_bit_index;
        }
    }

    pub struct EmbeddingSplitCacheBatch<Input, Weights, Embedding> {
        pub cache_batch: CacheBatch<Input, Weights, Embedding>,
    }

    pub trait OptimizePass<Input, Weights, Head> {
        fn optimize(
            weights: &mut Weights,
            head: &mut Head,
            examples: &[(Input, usize)],
        ) -> (f64, u64);
    }

    impl<
            Input: BitLen + Send + Sync + GetBit + HammingDistance + Copy,
            Weights: Copy + Send + Sync + FlipBitIndexed + GetPatch<Input> + Apply<Input, Embedding>,
            Embedding: Copy + Send + Sync + GetBit + FlipBit + BitLen + Default + SetBit + HammingDistance,
        > OptimizePass<Input, Weights, [Embedding; 10]>
        for EmbeddingSplitCacheBatch<Input, Weights, Embedding>
    {
        fn optimize(
            weights: &mut Weights,
            head: &mut [Embedding; 10],
            examples: &[(Input, usize)],
        ) -> (f64, u64) {
            let mut updates = 0;
            let minibatch_size = examples.len() / Embedding::BIT_LEN;
            println!(
                "{} / {} = {}",
                examples.len(),
                Embedding::BIT_LEN,
                minibatch_size
            );
            for e in 0..<Embedding>::BIT_LEN {
                let mut cache_batch = CacheBatch::new(
                    weights,
                    head,
                    &examples[(e * minibatch_size)..((e + 1) * minibatch_size)],
                );
                // hack to get the current loss
                cache_batch.flip_weights_bit(0);
                let mut cur_loss = cache_batch.bit_loss(0);
                cache_batch.flip_weights_bit(0);
                //for c in 0..10 {
                //    let new_loss = cache_batch.head_bit_loss(e, c);
                //    if new_loss < cur_loss {
                //        cache_batch.flip_head_bit(c, e);
                //        head[c].flip_bit(e);
                //        cur_loss = new_loss;
                //    }
                //}
                cache_batch.transition_embedding_bit(e);
                for b in 0..<Input>::BIT_LEN {
                    let new_loss = cache_batch.bit_loss(b);
                    if new_loss < cur_loss {
                        cur_loss = new_loss;
                        cache_batch.flip_weights_bit(b);
                        weights.flip_bit_indexed(e, b);
                        updates += 1;
                    }
                }
            }
            let mut cache_batch = CacheBatch::new(weights, head, examples);
            (cache_batch.acc(), updates)
        }
    }

    impl<
            Input: BitLen + Send + Sync + GetBit + HammingDistance + Copy,
            Weights: Copy + Send + Sync + FlipBitIndexed + GetPatch<Input> + Apply<Input, Embedding>,
            Embedding: Copy + Send + Sync + GetBit + FlipBit + BitLen + Default + SetBit + HammingDistance,
        > OptimizePass<Input, Weights, [Embedding; 10]> for CacheBatch<Input, Weights, Embedding>
    {
        fn optimize(
            weights: &mut Weights,
            head: &mut [Embedding; 10],
            examples: &[(Input, usize)],
        ) -> (f64, u64) {
            let mut cache_batch = CacheBatch::new(weights, head, examples);
            let mut cur_loss = cache_batch.bit_loss(0);
            let mut updates = 0;
            for e in 0..<Embedding>::BIT_LEN {
                //let head_start = PreciseTime::now();
                //for c in 0..10 {
                //    let new_loss = cache_batch.head_bit_loss(e, c);
                //    if new_loss < cur_loss {
                //        cache_batch.flip_head_bit(c, e);
                //        head[c].flip_bit(e);
                //        cur_loss = new_loss;
                //        //println!("head {} {} {}", c, e, new_loss);
                //    }
                //}
                //println!("head time: {}", head_start.to(PreciseTime::now()));
                //let trans_start = PreciseTime::now();
                cache_batch.transition_embedding_bit(e);
                //println!("trans time: {}", trans_start.to(PreciseTime::now()));
                //let patch_start = PreciseTime::now();
                for b in 0..<Input>::BIT_LEN {
                    let new_loss = cache_batch.bit_loss(b);
                    if new_loss < cur_loss {
                        cur_loss = new_loss;
                        cache_batch.flip_weights_bit(b);
                        weights.flip_bit_indexed(e, b);
                        updates += 1;
                        //println!("{} {}: {:?}", e, b, new_loss);
                    }
                }
                //println!("patch time: {}", patch_start.to(PreciseTime::now()));
            }
            (cache_batch.acc(), updates)
        }
    }

    pub trait RecursiveTrain<Input, Embedding, Optimizer> {
        fn recurse_train(
            weights: &mut Self,
            head: &mut [Embedding; 10],
            examples: &[(Input, usize)],
            depth: usize,
        ) -> f64;
    }

    impl<
            Input: Sync + Send,
            Embedding: Sync + Send,
            Optimizer: OptimizePass<Input, Self, [Embedding; 10]>,
            Weights: SaveLoad + Apply<Input, Embedding> + Sync + Send,
        > RecursiveTrain<Input, Embedding, Optimizer> for Weights
    {
        fn recurse_train(
            weights: &mut Weights,
            head: &mut [Embedding; 10],
            examples: &[(Input, usize)],
            depth: usize,
        ) -> f64 {
            if depth == 0 {
                Optimizer::optimize(weights, head, &examples[0..examples.len() / 2]);
            } else {
                <Weights as RecursiveTrain<Input, Embedding, Optimizer>>::recurse_train(
                    weights,
                    head,
                    &examples[0..examples.len() / 2],
                    depth - 1,
                );
            }
            let (acc, updates) =
                Optimizer::optimize(weights, head, &examples[(examples.len() / 2)..]);
            println!("depth: {} {}", depth, acc * 100f64);
            acc
        }
    }

    pub trait TrainFC<Input, Weights, Embedding, Optimizer> {
        fn train<RNG: rand::Rng>(
            rng: &mut RNG,
            examples: &Vec<(Input, usize)>,
            weights_path: &Path,
            depth: usize,
            updates_thresh: u64,
        ) -> Vec<(Embedding, usize)>;
    }

    impl<
            Input: Sync + Send,
            Weights: SaveLoad
                + Apply<Input, Embedding>
                + Sync
                + Send
                + RecursiveTrain<Input, Embedding, Optimizer>,
            Embedding: Sync + Send,
            Optimizer: OptimizePass<Input, Weights, [Embedding; 10]>,
        > TrainFC<Input, Weights, Embedding, Optimizer> for Weights
    where
        rand::distributions::Standard: rand::distributions::Distribution<Weights>,
        rand::distributions::Standard: rand::distributions::Distribution<Embedding>,
    {
        fn train<RNG: rand::Rng>(
            rng: &mut RNG,
            examples: &Vec<(Input, usize)>,
            weights_path: &Path,
            depth: usize,
            updates_thresh: u64,
        ) -> Vec<(Embedding, usize)> {
            let weights = Self::new_from_fs(weights_path).unwrap_or_else(|| {
                println!("{} not found, training", &weights_path.to_str().unwrap());

                let mut weights: Weights = rng.gen();
                let mut head: [Embedding; 10] = rng.gen();
                let mut acc =
                    <Weights as RecursiveTrain<Input, Embedding, Optimizer>>::recurse_train(
                        &mut weights,
                        &mut head,
                        &examples,
                        depth,
                    );
                let mut updates = updates_thresh;
                while updates >= updates_thresh {
                    let result = Optimizer::optimize(&mut weights, &mut head, examples);
                    updates = result.1;
                    acc = result.0;
                    println!("updates: {} acc: {}%", updates, acc * 100f64);
                }
                println!("final acc: {}%", acc * 100f64);
                weights.write_to_fs(&weights_path);
                weights
            });

            examples
                .par_iter()
                .map(|(input, class)| (weights.apply(input), *class))
                .collect()
        }
    }

    pub trait TrainConv<InputImage, OutputImage, InputPatch, Weights, Embedding, Optimizer> {
        fn train<RNG: rand::Rng>(
            rng: &mut RNG,
            examples: &Vec<(InputImage, usize)>,
            weights_path: &Path,
            depth: usize,
            updates_thresh: u64,
        ) -> Vec<(OutputImage, usize)>;
    }
    impl<
            InputImage: ExtractPatches<InputPatch> + Sync + Send,
            OutputImage: Send + Sync,
            InputPatch: Sync + Send + Copy,
            Weights: SaveLoad
                + Apply<InputPatch, Embedding>
                + Conv2D<InputImage, OutputImage>
                + Sync
                + Send
                + RecursiveTrain<InputPatch, Embedding, Optimizer>,
            Embedding: Sync + Send,
            Optimizer: OptimizePass<InputPatch, Weights, [Embedding; 10]>,
        > TrainConv<InputImage, OutputImage, InputPatch, Weights, Embedding, Optimizer> for Weights
    where
        rand::distributions::Standard: rand::distributions::Distribution<Weights>,
        rand::distributions::Standard: rand::distributions::Distribution<Embedding>,
    {
        fn train<RNG: rand::Rng>(
            rng: &mut RNG,
            examples: &Vec<(InputImage, usize)>,
            weights_path: &Path,
            depth: usize,
            updates_thresh: u64,
        ) -> Vec<(OutputImage, usize)> {
            let weights = Self::new_from_fs(weights_path).unwrap_or_else(|| {
                println!("{} not found, training", &weights_path.to_str().unwrap());

                let mut weights: Weights = rng.gen();
                //let mut weights = Weights::binary_filter();
                let mut head: [Embedding; 10] = rng.gen();
                let mut patches = {
                    let mut patches: Vec<(InputPatch, usize)> = examples
                        .iter()
                        .map(|(image, class)| {
                            let patches: Vec<(InputPatch, usize)> = image
                                .patches()
                                .iter()
                                .cloned()
                                .zip(iter::repeat(*class))
                                .collect();
                            patches
                        })
                        .flatten()
                        .collect();
                    // and shuffle them
                    patches.shuffle(rng);
                    patches
                };

                let mut acc =
                    <Weights as RecursiveTrain<InputPatch, Embedding, Optimizer>>::recurse_train(
                        &mut weights,
                        &mut head,
                        &patches,
                        depth,
                    );

                let mut updates = updates_thresh;
                while updates >= updates_thresh {
                    let result = Optimizer::optimize(&mut weights, &mut head, &patches);
                    updates = result.1;
                    acc = result.0;
                    patches.shuffle(rng);
                    println!("updates: {} acc: {}%", updates, acc * 100f64);
                }
                println!("final acc: {}%", acc * 100f64);
                weights.write_to_fs(&weights_path);
                weights
            });

            examples
                .par_iter()
                .map(|(input, class)| (weights.conv2d(input), *class))
                .collect()
        }
    }
}
