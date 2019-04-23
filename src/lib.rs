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
    }

    pub mod cifar {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;

        macro_rules! to_unary {
            ($name:ident, $type:ty, $len:expr) => {
                fn $name(input: u8) -> $type {
                    !((!0) << (input / (255 / $len)))
                }
            };
        }

        to_unary!(to_10, u32, 10);
        to_unary!(to_11, u32, 11);
        to_unary!(to_32, u32, 32);

        pub trait ConvertPixel {
            fn convert(pixel: [u8; 3]) -> Self;
        }

        impl ConvertPixel for [u32; 1] {
            fn convert(pixel: [u8; 3]) -> [u32; 1] {
                [to_11(pixel[0]) as u32
                    | ((to_11(pixel[1]) as u32) << 11)
                    | ((to_10(pixel[2]) as u32) << 22)]
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
        ) -> Vec<(usize, [[T; 32]; 32])> {
            if n > 50000 {
                panic!("n must be <= 50,000");
            }
            (1..6)
                .map(|i| {
                    let mut file = File::open(&base_path.join(format!("data_batch_{}.bin", i)))
                        .expect("can't open data");

                    let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
                    let mut label: [u8; 1] = [0; 1];
                    let mut images: Vec<(usize, [[T; 32]; 32])> = Vec::new();
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
                        images.push((label[0] as usize, image));
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

pub trait BitLen: Sized {
    const BIT_LEN: usize;
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

pub trait FlipBit {
    fn flip_bit(&mut self, b: usize);
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

pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
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
array_hamming_distance!(13);

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

#[macro_use]
pub mod layers {
    use super::{BitLen, HammingDistance};
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

    impl<I: HammingDistance + BitLen> Apply<I, u32> for [I; 32] {
        fn apply(&self, input: &I) -> u32 {
            let mut target = 0u32;
            for i in 0..32 {
                target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2)) as u32) << i;
            }
            target
        }
    }

    macro_rules! impl_apply_for_array_output {
        ($len:expr) => {
            impl<I, T: Apply<I, u32>> Apply<I, [u32; $len]> for [T; $len] {
                fn apply(&self, input: &I) -> [u32; $len] {
                    let mut target = [0u32; $len];
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

    pub trait Apply<I, O> {
        fn apply(&self, input: &I) -> O;
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

    patch_conv_2x2_apply_trait!(32, 32);
    patch_conv_2x2_apply_trait!(16, 16);
    patch_conv_2x2_apply_trait!(8, 8);

    macro_rules! conv3x3_apply_trait {
        ($x_size:expr, $y_size:expr, $output_len:expr) => {
            impl<I: Copy + BitLen + HammingDistance>
                Apply<[[I; $y_size]; $x_size], [[[u32; $output_len]; $y_size]; $x_size]>
                for [[[[I; 3]; 3]; 16]; $output_len]
            {
                fn apply(
                    &self,
                    input: &[[I; $y_size]; $x_size],
                ) -> [[[u32; $output_len]; $y_size]; $x_size] {
                    let mut target = [[[0u32; $output_len]; $y_size]; $x_size];
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

    //conv3x3_apply_trait!(32, 32, 1);
    //conv3x3_apply_trait!(16, 16, 1);
    //conv3x3_apply_trait!(8, 8, 1);

    //conv3x3_apply_trait!(32, 32, 2);
    //conv3x3_apply_trait!(16, 16, 2);
    //conv3x3_apply_trait!(8, 8, 2);

    //conv3x3_apply_trait!(32, 32, 3);
    //conv3x3_apply_trait!(16, 16, 3);
    //conv3x3_apply_trait!(8, 8, 3);

    //conv3x3_apply_trait!(32, 32, 4);
    //conv3x3_apply_trait!(16, 16, 4);
    //conv3x3_apply_trait!(8, 8, 4);

    macro_rules! conv5x5_apply_trait {
        ($x_size:expr, $y_size:expr, $output_len:expr) => {
            impl<I: Copy + BitLen + HammingDistance>
                Apply<[[I; $y_size]; $x_size], [[[u32; $output_len]; $y_size]; $x_size]>
                for [[[[I; 5]; 5]; 16]; $output_len]
            {
                fn apply(
                    &self,
                    input: &[[I; $y_size]; $x_size],
                ) -> [[[u32; $output_len]; $y_size]; $x_size] {
                    let mut target = [[[0u32; $output_len]; $y_size]; $x_size];
                    for x in 0..$x_size - 4 {
                        for y in 0..$y_size - 4 {
                            target[x + 2][y + 2] = self.apply(&patch_5x5!(input, x, y));
                        }
                    }
                    target
                }
            }
        };
    }

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
}

pub trait Image2D<Pixel> {}

macro_rules! impl_image2d {
    ($x_size:expr, $y_size:expr) => {
        impl<Pixel> Image2D<Pixel> for [[Pixel; $y_size]; $y_size] {}
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
impl_extract_patch_trait!(16, 16);
impl_extract_patch_trait!(8, 8);
impl_extract_patch_trait!(4, 4);

pub trait ConcatImages<A, B> {
    fn concat_images(input_a: &A, input_b: &B) -> Self;
}

macro_rules! impl_concat_image {
    ($a_len:expr, $b_len:expr, $x_size:expr, $y_size:expr) => {
        impl ConcatImages<[[[u32; $a_len]; $y_size]; $x_size], [[[u32; $b_len]; $y_size]; $x_size]>
            for [[[u32; $a_len + $b_len]; $y_size]; $x_size]
        {
            fn concat_images(
                input_a: &[[[u32; $a_len]; $y_size]; $x_size],
                input_b: &[[[u32; $b_len]; $y_size]; $x_size],
            ) -> [[[u32; $a_len + $b_len]; $y_size]; $x_size] {
                let mut target = <[[[u32; $a_len + $b_len]; $y_size]; $x_size]>::default();
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        for i in 0..$a_len {
                            target[x][y][i] = input_a[x][y][i];
                        }
                        for i in 0..$b_len {
                            target[x][y][$a_len + i] = input_b[x][y][i];
                        }
                    }
                }
                target
            }
        }
    };
}

impl_concat_image!(1, 1, 32, 32);
impl_concat_image!(2, 1, 32, 32);
impl_concat_image!(3, 1, 32, 32);
impl_concat_image!(4, 1, 32, 32);
impl_concat_image!(5, 1, 32, 32);
impl_concat_image!(6, 1, 32, 32);

pub fn vec_concat_2_examples<A: Sync, B: Sync, T: ConcatImages<A, B> + Sync + Send>(
    a: &Vec<(usize, A)>,
    b: &Vec<(usize, B)>,
) -> Vec<(usize, T)> {
    assert_eq!(a.len(), b.len());
    a.par_iter()
        .zip(b.par_iter())
        .map(|((a_class, a_image), (b_class, b_image))| {
            assert_eq!(a_class, b_class);
            (*a_class, T::concat_images(a_image, b_image))
        })
        .collect()
}
