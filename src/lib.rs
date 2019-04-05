extern crate bincode;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate serde_derive;
extern crate time;
extern crate vulkano;
extern crate vulkano_shaders;
use rayon::prelude::*;

pub mod datasets {
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

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

impl BitLen for u32 {
    const BIT_LEN: usize = 32;
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

pub trait FlipBit {
    fn flip_bit(&mut self, b: usize);
}

impl FlipBit for u32 {
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

pub trait FlipBitIndexed {
    fn flip_bit_indexed(&mut self, o: usize, b: usize);
    const INDEX_LEN: usize;
}
impl<T: FlipBit> FlipBitIndexed for [T; 16] {
    fn flip_bit_indexed(&mut self, o: usize, b: usize) {
        self[o].flip_bit(b);
    }
    const INDEX_LEN: usize = 16;
}

impl<T: FlipBit> FlipBitIndexed for [T; 32] {
    fn flip_bit_indexed(&mut self, o: usize, b: usize) {
        self[o].flip_bit(b);
    }
    const INDEX_LEN: usize = 32;
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
    };
}
impl_getpatch_for_weights!(1);
impl_getpatch_for_weights!(2);
impl_getpatch_for_weights!(3);
impl_getpatch_for_weights!(4);

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

pub trait GetMirroredWords {
    fn get_mirrored_words(&self, i: usize) -> [u32; 2];
}

impl<T: GetWord + WordLen> GetMirroredWords for [T; 3] {
    #[inline(always)]
    fn get_mirrored_words(&self, i: usize) -> [u32; 2] {
        let input_x = i / T::WORD_LEN;
        let strip_i = i % T::WORD_LEN;
        let input_words = if input_x == 0 {
            [self[0].get_word(strip_i), self[2].get_word(strip_i)]
        } else if input_x == 2 {
            [self[2].get_word(strip_i), self[0].get_word(strip_i)]
        } else {
            [self[1].get_word(strip_i), self[1].get_word(strip_i)]
        };
        input_words
    }
}

impl<T: GetWord + WordLen> GetMirroredWords for [T; 2] {
    #[inline(always)]
    fn get_mirrored_words(&self, i: usize) -> [u32; 2] {
        let input_x = i / T::WORD_LEN;
        let strip_i = i % T::WORD_LEN;
        let input_words = if input_x == 0 {
            [self[0].get_word(strip_i), self[1].get_word(strip_i)]
        } else {
            [self[1].get_word(strip_i), self[0].get_word(strip_i)]
        };
        input_words
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

pub trait MirrorHammingDistance {
    fn normal_hamming_distance(&self, input: &Self) -> u32;
    fn fliped_hamming_distance(&self, input: &Self) -> u32;
}

impl<T: HammingDistance> MirrorHammingDistance for [T; 3] {
    fn normal_hamming_distance(&self, input: &[T; 3]) -> u32 {
        self[0].hamming_distance(&input[0])
            + self[1].hamming_distance(&input[1])
            + self[2].hamming_distance(&input[2])
    }
    fn fliped_hamming_distance(&self, input: &[T; 3]) -> u32 {
        self[0].hamming_distance(&input[2])
            + self[1].hamming_distance(&input[1])
            + self[2].hamming_distance(&input[0])
    }
}

impl<T: HammingDistance> MirrorHammingDistance for [T; 2] {
    fn normal_hamming_distance(&self, input: &[T; 2]) -> u32 {
        self[0].hamming_distance(&input[0]) + self[1].hamming_distance(&input[1])
    }
    fn fliped_hamming_distance(&self, input: &[T; 2]) -> u32 {
        self[0].hamming_distance(&input[1]) + self[1].hamming_distance(&input[0])
    }
}

#[macro_use]
pub mod layers {
    use super::{BitLen, HammingDistance, MirrorHammingDistance};
    use bincode::{deserialize_from, serialize_into};
    use std::fs::File;
    use std::io::BufWriter;
    use std::path::Path;

    impl<I: HammingDistance + BitLen> Apply<[I; 3], u32> for [[I; 3]; 16]
    where
        [I; 3]: MirrorHammingDistance,
    {
        fn apply(&self, input: &[I; 3]) -> u32 {
            let threshold: u32 = (I::BIT_LEN * 3) as u32 / 2;
            let mut target = 0u32;
            for i in 0..16 {
                target |= ((self[i].normal_hamming_distance(input) > threshold) as u32) << i;
                target |= ((self[i].fliped_hamming_distance(input) > threshold) as u32) << (16 + i);
            }
            target
        }
    }

    impl<I: HammingDistance + BitLen> Apply<[I; 2], u32> for [[I; 2]; 16]
    where
        [I; 2]: MirrorHammingDistance,
    {
        fn apply(&self, input: &[I; 2]) -> u32 {
            let threshold: u32 = (I::BIT_LEN * 2) as u32 / 2;
            let mut target = 0u32;
            for i in 0..16 {
                target |= ((self[i].normal_hamming_distance(input) > threshold) as u32) << i;
                target |= ((self[i].fliped_hamming_distance(input) > threshold) as u32) << (16 + i);
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
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, O: Copy + Default, W: Apply<[[I; 3]; 3], O>>
                Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]> for W
            {
                fn apply(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size]; $x_size] {
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

    conv3x3_apply_trait!(32, 32);
    conv3x3_apply_trait!(16, 16);
    conv3x3_apply_trait!(8, 8);

    pub trait SaveLoad
    where
        Self: Sized,
    {
        fn write_to_fs(&self, path: &Path);
        fn new_from_fs(path: &Path) -> Option<Self>;
    }

    macro_rules! impl_saveload_conv3x3_array {
        ($input_len:expr, $output_len:expr) => {
            impl SaveLoad for [[[[[u32; $input_len]; 3]; 3]; 16]; $output_len] {
                fn write_to_fs(&self, path: &Path) {
                    let vec_params: Vec<[[[[u32; $input_len]; 3]; 3]; 16]> =
                        self.iter().cloned().collect();
                    let mut f = BufWriter::new(File::create(path).unwrap());
                    serialize_into(&mut f, &vec_params).unwrap();
                }
                // This will return:
                // - Some if the file exists and is good
                // - None of the file does not exist
                // and will panic if the file is exists but is bad.
                fn new_from_fs(path: &Path) -> Option<Self> {
                    File::open(&path)
                        .map(|f| deserialize_from(f).unwrap())
                        .map(|vec_params: Vec<[[[[u32; $input_len]; 3]; 3]; 16]>| {
                            if vec_params.len() != $output_len {
                                panic!("input is of len {} not {}", vec_params.len(), $output_len);
                            }
                            let mut params =
                                [<[[[[u32; $input_len]; 3]; 3]; 16]>::default(); $output_len];
                            for i in 0..$output_len {
                                params[i] = vec_params[i];
                            }
                            params
                        })
                        .ok()
                }
            }
        };
    }
    impl_saveload_conv3x3_array!(1, 1);
    impl_saveload_conv3x3_array!(2, 1);
    impl_saveload_conv3x3_array!(2, 2);
    impl_saveload_conv3x3_array!(4, 2);
    impl_saveload_conv3x3_array!(4, 4);

    macro_rules! impl_saveload_conv2x2_array {
        ($input_len:expr, $output_len:expr) => {
            impl SaveLoad for [[[[[u32; $input_len]; 2]; 2]; 16]; $output_len] {
                fn write_to_fs(&self, path: &Path) {
                    let vec_params: Vec<[[[[u32; $input_len]; 2]; 2]; 16]> =
                        self.iter().cloned().collect();
                    let mut f = BufWriter::new(File::create(path).unwrap());
                    serialize_into(&mut f, &vec_params).unwrap();
                }
                // This will return:
                // - Some if the file exists and is good
                // - None of the file does not exist
                // and will panic if the file is exists but is bad.
                fn new_from_fs(path: &Path) -> Option<Self> {
                    File::open(&path)
                        .map(|f| deserialize_from(f).unwrap())
                        .map(|vec_params: Vec<[[[[u32; $input_len]; 2]; 2]; 16]>| {
                            if vec_params.len() != $output_len {
                                panic!("input is of len {} not {}", vec_params.len(), $output_len);
                            }
                            let mut params =
                                [<[[[[u32; $input_len]; 2]; 2]; 16]>::default(); $output_len];
                            for i in 0..$output_len {
                                params[i] = vec_params[i];
                            }
                            params
                        })
                        .ok()
                }
            }
        };
    }
    impl_saveload_conv2x2_array!(1, 1);
    impl_saveload_conv2x2_array!(1, 2);
    impl_saveload_conv2x2_array!(2, 1);
    impl_saveload_conv2x2_array!(2, 2);
    impl_saveload_conv2x2_array!(2, 4);
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
impl_image2d!(3, 3);
impl_image2d!(2, 2);

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
impl_extract_patch_trait!(16, 16);
impl_extract_patch_trait!(8, 8);

pub trait ConcatImages<I> {
    fn concat_images(inputs: I) -> Self;
}

macro_rules! impl_concat_image {
    ($len:expr, $depth:expr, $x_size:expr, $y_size:expr) => {
        impl<P: Default + Copy> ConcatImages<[&[[[P; $len]; $y_size]; $x_size]; $depth]>
            for [[[P; $depth * $len]; $y_size]; $x_size]
        {
            fn concat_images(
                input: [&[[[P; $len]; $y_size]; $x_size]; $depth],
            ) -> [[[P; $depth * $len]; $y_size]; $x_size] {
                let mut target = <[[[P; $depth * $len]; $y_size]; $x_size]>::default();
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        for i in 0..$depth {
                            for l in 0..$len {
                                target[x][y][(i * $len) + l] = input[i][x][y][l];
                            }
                        }
                    }
                }
                target
            }
        }
    };
}

impl_concat_image!(2, 2, 16, 16);
impl_concat_image!(1, 2, 32, 32);
impl_concat_image!(1, 2, 16, 16);

pub fn vec_concat_2_examples<'a, I: 'a + Sync, C: ConcatImages<[&'a I; 2]> + Sync + Send>(
    a: &'a Vec<(usize, I)>,
    b: &'a Vec<(usize, I)>,
) -> Vec<(usize, C)> {
    assert_eq!(a.len(), b.len());
    a.par_iter()
        .zip(b.par_iter())
        .map(|((a_class, a_image), (b_class, b_image))| {
            assert_eq!(a_class, b_class);
            (*a_class, C::concat_images([a_image, b_image]))
        })
        .collect()
}

pub mod optimize {
    use crate::layers::{Apply, SaveLoad};
    use crate::objective_eval::{ObjectiveEval, ObjectiveEvalCreator};
    use crate::{BitLen, ExtractPatches, FlipBit, FlipBitIndexed, Image2D};
    use rand::prelude::*;
    use rayon::prelude::*;
    use std::fs::OpenOptions;
    use std::io::prelude::*;
    use std::iter;
    use std::path::Path;
    use time::PreciseTime;

    pub trait Train<EvalCreator, InputPatch, Weights, Embedding> {
        fn train_pass(
            eval_creator: &EvalCreator,
            weights: &mut Weights,
            head: &mut [Embedding; 10],
            patches: &[(u8, InputPatch)],
            head_update_freq: usize,
        ) -> f64;
        fn recurs_train(
            eval_creator: &EvalCreator,
            weights: &mut Weights,
            head: &mut [Embedding; 10],
            patches: &[(u8, InputPatch)],
            depth: usize,
            head_update_freq: usize,
        ) -> f64;
    }

    impl<
            Patch: BitLen,
            Weights: FlipBitIndexed,
            Embedding: FlipBit,
            EvalCreator: ObjectiveEvalCreator<Patch, Weights, Embedding>,
        > Train<EvalCreator, Patch, Weights, Embedding> for Patch
    where
        EvalCreator::ObjectiveEvalType: ObjectiveEval<Patch, Weights, Embedding>,
    {
        fn train_pass(
            eval_creator: &EvalCreator,
            weights: &mut Weights,
            head: &mut [Embedding; 10],
            patches: &[(u8, Patch)],
            head_update_freq: usize,
        ) -> f64 {
            dbg!(patches.len());

            let mut gpu_obj_eval = eval_creator.new_obj_eval(&weights, &head, &patches);

            let start = PreciseTime::now();
            let mut cur_obj = gpu_obj_eval.obj();
            for e in 0..Weights::INDEX_LEN {
                let mut iter = 0;
                for i in 0..<Patch>::BIT_LEN {
                    if iter % head_update_freq == 0 {
                        for o in 0..10 {
                            gpu_obj_eval.flip_head_bit(o, e);
                            let new_obj = gpu_obj_eval.obj();
                            if new_obj >= cur_obj {
                                cur_obj = new_obj;
                                println!(
                                    "head: {} {}: {} {}",
                                    e,
                                    o,
                                    new_obj,
                                    new_obj as f64 / patches.len() as f64
                                );
                                head[o].flip_bit(e);
                            } else {
                                gpu_obj_eval.flip_head_bit(o, e);
                            }
                        }
                        iter += 1;
                    }
                    gpu_obj_eval.flip_weights_bit(e, i);
                    let new_obj = gpu_obj_eval.obj();
                    if new_obj >= cur_obj {
                        iter += 1;
                        cur_obj = new_obj;
                        println!(
                            "{} {}: {} {}",
                            e,
                            i,
                            new_obj,
                            new_obj as f64 / patches.len() as f64
                        );
                        weights.flip_bit_indexed(e, i);
                    } else {
                        gpu_obj_eval.flip_weights_bit(e, i);
                    }
                }
            }
            println!("{} {}", patches.len(), start.to(PreciseTime::now()));
            cur_obj as f64 / patches.len() as f64
        }
        // TODO: implement non 0.5 minibatch division.
        fn recurs_train(
            eval_creator: &EvalCreator,
            weights: &mut Weights,
            head: &mut [Embedding; 10],
            patches: &[(u8, Patch)],
            depth: usize,
            head_update_freq: usize,
        ) -> f64 {
            if depth == 0 {
                Self::train_pass(
                    eval_creator,
                    weights,
                    head,
                    &patches[0..patches.len() / 2],
                    head_update_freq,
                );
            } else {
                Self::recurs_train(
                    eval_creator,
                    weights,
                    head,
                    &patches[0..patches.len() / 2],
                    depth - 1,
                    head_update_freq,
                );
            }
            Self::train_pass(
                eval_creator,
                weights,
                head,
                &patches[patches.len() / 2..],
                head_update_freq,
            )
        }
    }
    pub trait TrainLayer<EvalCreator, Pixel, Patch, Embedding, InputImage, OutputImage> {
        fn train_from_images<RNG: rand::Rng>(
            rng: &mut RNG,
            eval_creator: &EvalCreator,
            images: &Vec<(usize, InputImage)>,
            fs_path: &Path,
            depth: usize,
            head_update_freq: usize,
            log_file_path: &Path,
            do_final_pass: bool,
        ) -> Vec<(usize, OutputImage)>;
    }

    impl<
            Pixel,
            Patch: Image2D<Pixel>,
            Weights: SaveLoad + Sync,
            Embedding,
            EvalCreator: ObjectiveEvalCreator<Patch, Weights, Embedding>,
            InputImage: Sync + ExtractPatches<Patch> + Image2D<Pixel>,
            OutputImage: Sync + Send,
        > TrainLayer<EvalCreator, Pixel, Patch, Embedding, InputImage, OutputImage> for Weights
    where
        Self: Apply<InputImage, OutputImage>,
        <EvalCreator as ObjectiveEvalCreator<Patch, Weights, Embedding>>::ObjectiveEvalType:
            ObjectiveEval<Patch, Weights, Embedding>,
        EvalCreator: ObjectiveEvalCreator<Patch, Weights, Embedding>,
        rand::distributions::Standard: rand::distributions::Distribution<Weights>,
        rand::distributions::Standard: rand::distributions::Distribution<Embedding>,
        Patch: Train<EvalCreator, Patch, Weights, Embedding>,
    {
        fn train_from_images<RNG: rand::Rng>(
            rng: &mut RNG,
            eval_creator: &EvalCreator,
            images: &Vec<(usize, InputImage)>,
            fs_path: &Path,
            depth: usize,
            head_update_freq: usize,
            log_file_path: &Path,
            do_final_pass: bool,
        ) -> Vec<(usize, OutputImage)> {
            let weights = Self::new_from_fs(fs_path).unwrap_or_else(|| {
                println!("{} not found, training", &fs_path.to_str().unwrap());
                let patches = {
                    let mut patches: Vec<(u8, Patch)> = images
                        .iter()
                        .map(|(class, image)| iter::repeat(*class as u8).zip(image.patches()))
                        .flatten()
                        .collect();
                    // and shuffle them
                    patches.shuffle(rng);
                    patches
                };

                let mut weights: Weights = rng.gen();
                let mut head: [Embedding; 10] = rng.gen();

                let start = PreciseTime::now();

                let mut avg_obj = <Patch>::recurs_train(
                    eval_creator,
                    &mut weights,
                    &mut head,
                    &patches,
                    depth,
                    head_update_freq,
                );
                if do_final_pass {
                    avg_obj =
                        <Patch>::train_pass(eval_creator, &mut weights, &mut head, &patches, 100);
                }
                println!("obj: {}, time: {}", avg_obj, start.to(PreciseTime::now()));
                write_to_log_event(
                    log_file_path,
                    fs_path,
                    start.to(PreciseTime::now()),
                    head_update_freq,
                    avg_obj,
                    depth,
                    images.len(),
                );
                weights.write_to_fs(&fs_path);
                weights
            });

            images
                .par_iter()
                .map(|(class, image)| (*class, weights.apply(image)))
                .collect()
        }
    }

    fn write_to_log_event(
        file_path: &Path,
        layer_name: &Path,
        duration: time::Duration,
        head_update_freq: usize,
        obj: f64,
        depth: usize,
        n: usize,
    ) {
        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(file_path)
            .unwrap();
        writeln!(
            file,
            "{} depth: {}, obj: {}, head_update_freq: {}, n: {}, {}",
            layer_name.to_str().unwrap(),
            depth,
            obj,
            head_update_freq,
            n,
            duration,
        )
        .unwrap();
    }
}

pub mod objective_eval {
    extern crate rand;
    extern crate time;

    use super::layers::Apply;
    use super::{
        BitLen, FlipBit, FlipBitIndexed, GetMirroredWords, GetPatch, GetWord, HammingDistance,
        MirrorHammingDistance, WordLen,
    };
    use rayon::prelude::*;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use std::thread;
    use std::thread::JoinHandle;
    use vulkano::buffer::BufferUsage;
    use vulkano::buffer::{CpuAccessibleBuffer, ImmutableBuffer};
    use vulkano::command_buffer::AutoCommandBufferBuilder;
    use vulkano::descriptor::descriptor_set::{
        PersistentDescriptorSet, PersistentDescriptorSetBuf,
    };
    use vulkano::descriptor::pipeline_layout::PipelineLayout;
    use vulkano::device::DeviceExtensions;
    use vulkano::device::{Device, Queue};
    use vulkano::instance::Instance;
    use vulkano::instance::InstanceExtensions;
    use vulkano::instance::PhysicalDevice;
    use vulkano::pipeline::ComputePipeline;
    use vulkano::sync;
    use vulkano::sync::GpuFuture;

    pub trait IsCorrect<I> {
        fn is_correct(&self, target: u8, input: I) -> bool;
    }

    impl<I: HammingDistance> IsCorrect<I> for [I; 10] {
        // the max activation is the target.
        #[inline(always)]
        fn is_correct(&self, target: u8, input: I) -> bool {
            let max = self[target as usize].hamming_distance(&input);
            for i in 0..10 {
                if i != target as usize {
                    if self[i].hamming_distance(&input) >= max {
                        return false;
                    }
                }
            }
            true
        }
    }

    pub trait ObjectiveEvalCreator<InputPatch, Weights, Embedding> {
        type ObjectiveEvalType;
        fn new_obj_eval(
            &self,
            weights: &Weights,
            head: &[Embedding; 10],
            examples: &[(u8, InputPatch)],
        ) -> Self::ObjectiveEvalType;
    }

    pub trait ObjectiveEval<InputPatch, Weights, Embedding> {
        fn flip_weights_bit(&mut self, o: usize, i: usize);
        fn flip_head_bit(&mut self, o: usize, i: usize);
        fn obj(&mut self) -> u64;
    }

    pub struct TestCPUObjectiveEvalCreator {}

    pub struct TestCPUObjectiveEval<InputPatch, Weights, Embedding> {
        weights: Weights,
        head: [Embedding; 10],
        examples: Vec<(u8, InputPatch)>,
    }
    impl TestCPUObjectiveEvalCreator {
        pub fn new() -> Self {
            TestCPUObjectiveEvalCreator {}
        }
    }

    impl<Patch: Copy, Weights: Apply<Patch, Embedding> + Copy, Embedding: Copy>
        ObjectiveEvalCreator<Patch, Weights, Embedding> for TestCPUObjectiveEvalCreator
    {
        type ObjectiveEvalType = TestCPUObjectiveEval<Patch, Weights, Embedding>;
        fn new_obj_eval(
            &self,
            weights: &Weights,
            head: &[Embedding; 10],
            examples: &[(u8, Patch)],
        ) -> TestCPUObjectiveEval<Patch, Weights, Embedding> {
            TestCPUObjectiveEval {
                weights: *weights,
                head: *head,
                examples: examples.iter().cloned().collect(),
            }
        }
    }

    // This is a slow implementation of obj() and should not be used if performance is desired.
    impl<
            Patch: Sync,
            Weights: Sync + Apply<Patch, Embedding> + FlipBitIndexed,
            Embedding: Sync + FlipBit,
        > ObjectiveEval<Patch, Weights, Embedding>
        for TestCPUObjectiveEval<Patch, Weights, Embedding>
    where
        [Embedding; 10]: IsCorrect<Embedding>,
    {
        fn flip_weights_bit(&mut self, o: usize, i: usize) {
            self.weights.flip_bit_indexed(o, i);
        }
        fn flip_head_bit(&mut self, o: usize, i: usize) {
            self.head[o].flip_bit(i);
        }
        fn obj(&mut self) -> u64 {
            self.examples
                .par_iter()
                .map(|(class, patch)| {
                    let embedding = self.weights.apply(patch);
                    self.head.is_correct(*class, embedding) as u64
                })
                .sum()
        }
    }

    mod fast_obj_update_e1 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_update_e1.glsl",
        }
    }
    mod fast_obj_update_e2 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_update_e2.glsl",
        }
    }
    mod fast_obj_update_e3 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_update_e3.glsl",
        }
    }


    mod transition_input_word_e1 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_input_word_trans_e1.glsl",
        }
    }
    mod transition_input_word_e2 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_input_word_trans_e2.glsl",
        }
    }
    mod transition_input_word_e3 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_input_word_trans_e3.glsl",
        }
    }


    mod clean_embedding_bit_e1 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_embedding_bit_clean_e1.glsl",
        }
    }
    mod clean_embedding_bit_e2 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_embedding_bit_clean_e2.glsl",
        }
    }
    mod clean_embedding_bit_e3 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_embedding_bit_clean_e3.glsl",
        }
    }


    mod head_obj_update_e1 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_head_update_e1.glsl",
        }
    }
    mod head_obj_update_e2 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_head_update_e2.glsl",
        }
    }
    mod head_obj_update_e3 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_head_update_e3.glsl",
        }
    }

    mod replace_cache_parts_e1 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_cache_replace_input_e1.glsl",
        }
    }
    mod replace_cache_parts_e2 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_cache_replace_input_e2.glsl",
        }
    }
    mod replace_cache_parts_e3 {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/fast_mirror_cache_replace_input_e3.glsl",
        }
    }

    // Note that when embedding word len get to be ~8, it may be worth switching to cached head partial sum so as to make head obj linear with embedding size.
    // However while it is small, the memory overhead of the 10 nonconstant partial sums is probably not worth the compute cost savings.
    // This is for 10 classes.
    // For 100 classes, the transition point will be far larger and it's probably not worth it.
    #[derive(Debug, Copy, Clone)]
    pub struct FastExampleMirroredCache<Embedding> {
        input_word: [u32; 2], // a pair of 32 bits. This is mirrored, so we need a normal and a flipped input word.
        input_partial_sums: [u32; 2], // a pair of integers, one for normal one for flipped.
        embedding: Embedding,
        true_class: u32,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct FastExampleMirroredParts {
        input_word: [u32; 2],
        input_partial_sums: [u32; 2],
    }

    pub trait NewMirrorFastCache<Weights, Embedding> {
        fn new_full(
            &self,
            weights: &Weights,
            class: usize,
            embedding_index: usize,
            patch_index: usize,
        ) -> FastExampleMirroredCache<Embedding>;
        fn new_parts(&self, weights_patch: &Self, patch_index: usize) -> FastExampleMirroredParts;
    }

    macro_rules! impl_NewMirrorFastCache_for_FastExampleMirrored {
        ($output_len:expr) => {
            impl<InputPatch: GetWord + MirrorHammingDistance + GetMirroredWords + Copy>
                NewMirrorFastCache<[[InputPatch; 16]; $output_len], [u32; $output_len]>
                for InputPatch
            where
                [InputPatch; 16]: Apply<InputPatch, u32>,
            {
                fn new_full(
                    &self,
                    weights: &[[InputPatch; 16]; $output_len],
                    class: usize,
                    embedding_index: usize,
                    patch_index: usize,
                ) -> FastExampleMirroredCache<[u32; $output_len]> {
                    let embedding = weights.apply(self);
                    let weights_patch = weights.get_patch(embedding_index);
                    let input_words = self.get_mirrored_words(patch_index);
                    FastExampleMirroredCache {
                        input_word: input_words,
                        input_partial_sums: [
                            weights_patch.normal_hamming_distance(self)
                                - weights_patch
                                    .get_word(patch_index)
                                    .hamming_distance(&input_words[0]),
                            weights_patch.fliped_hamming_distance(self)
                                - weights_patch
                                    .get_word(patch_index)
                                    .hamming_distance(&input_words[1]),
                        ],
                        embedding: embedding,
                        true_class: class as u32,
                    }
                }
                fn new_parts(
                    &self,
                    weights_patch: &InputPatch,
                    patch_index: usize,
                ) -> FastExampleMirroredParts {
                    let input_words = self.get_mirrored_words(patch_index);
                    FastExampleMirroredParts {
                        input_word: input_words,
                        input_partial_sums: [
                            weights_patch.normal_hamming_distance(&self)
                                - weights_patch
                                    .get_word(patch_index)
                                    .hamming_distance(&input_words[0]),
                            weights_patch.fliped_hamming_distance(&self)
                                - weights_patch
                                    .get_word(patch_index)
                                    .hamming_distance(&input_words[1]),
                        ],
                    }
                }
            }
        };
    }
    impl_NewMirrorFastCache_for_FastExampleMirrored!(1);
    impl_NewMirrorFastCache_for_FastExampleMirrored!(2);
    impl_NewMirrorFastCache_for_FastExampleMirrored!(3);

    pub struct FastCacheVKObjEval<ShaderLayout, InputPatch, Weights, Embedding> {
        n_examples: usize,
        examples: Arc<Vec<InputPatch>>,
        sum_batch_size: usize,
        weights: Weights,
        head: [Embedding; 10],
        input_patch_type: PhantomData<InputPatch>,
        embedding_type: PhantomData<Embedding>,
        weights_type: PhantomData<Weights>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        obj_sums_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
        cache_buffer: Arc<ImmutableBuffer<[FastExampleMirroredCache<Embedding>]>>,
        update_pipeline: Arc<ComputePipeline<PipelineLayout<ShaderLayout>>>,
        update_descriptor_set: Arc<
            PersistentDescriptorSet<
                Arc<ComputePipeline<PipelineLayout<ShaderLayout>>>,
                (
                    (
                        (),
                        PersistentDescriptorSetBuf<
                            Arc<ImmutableBuffer<[FastExampleMirroredCache<Embedding>]>>,
                        >,
                    ),
                    PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<[u32]>>>,
                ),
            >,
        >,
        embedding_index: usize,
        patch_index: usize,
        embedding_is_clean: bool,
        next_input_words_buffer_join_handle: Option<JoinHandle<Arc<ImmutableBuffer<[[u32; 2]]>>>>,
        next_input_words_buffer_patch_index: usize,
        next_input_words_buffer_embedding_index: usize,
    }

    pub struct VulkanFastCacheObjectiveEvalCreator {
        instance: Arc<Instance>,
        reduce_sum_batch_size: usize,
    }

    impl VulkanFastCacheObjectiveEvalCreator {
        pub fn new(reduce_sum_batch_size: usize) -> Self {
            let instance = Instance::new(None, &InstanceExtensions::none(), None)
                .expect("failed to create instance");

            Self {
                instance: instance,
                reduce_sum_batch_size: reduce_sum_batch_size,
            }
        }
    }

    macro_rules! impl_fastexamplemirrored_over_embedding {
        ($shader_mod_name:ident, $input_trans_shader_mod_name:ident, $clean_embedding_bit_mod_name:ident, $head_obj_update_mod_name:ident, $replace_cache_parts_shader_mod_name:ident, $embedding_len:expr) => {
            impl<InputPatch: Copy + Sync + Send + GetWord + GetMirroredWords + BitLen + Copy + NewMirrorFastCache<[[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>>
            ObjectiveEvalCreator<InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]> for VulkanFastCacheObjectiveEvalCreator
            {
                type ObjectiveEvalType = FastCacheVKObjEval<$shader_mod_name::Layout, InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>;
                fn new_obj_eval(
                    &self,
                    weights: &[[InputPatch; 16]; $embedding_len],
                    head: &[[u32; $embedding_len]; 10],
                    examples: &[(u8, InputPatch)],
                ) -> FastCacheVKObjEval<$shader_mod_name::Layout, InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]> {
                    let physical = PhysicalDevice::enumerate(&self.instance).next().unwrap();
                    let queue_family = physical.queue_families().find(|&q| q.supports_compute()).unwrap();

                    // Now initializing the device.
                    let (device, mut queues) = Device::new(
                        physical,
                        physical.supported_features(),
                        &DeviceExtensions::none(),
                        [(queue_family, 0.5)].iter().cloned(),
                    )
                    .unwrap();

                    let queue = queues.next().unwrap();

                    let pipeline = Arc::new({
                        let shader = $shader_mod_name::Shader::load(device.clone()).unwrap();
                        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
                    });

                    let sums_buffer = {
                        let data_iter = (0..(examples.len() as f64 / self.reduce_sum_batch_size as f64).ceil() as u32).map(|_| 0u32);
                        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data_iter).unwrap()
                    };

                    let cache_buffer = {
                        let cache_vec: Vec<_> = examples
                            .par_iter()
                            .map(|(class, patch)| patch.new_full(weights, *class as usize, 0, 0))
                            .collect();
                        let (cache_buffer, _) = ImmutableBuffer::from_iter(cache_vec.iter().cloned(), BufferUsage::all(), queue.clone()).unwrap();
                        cache_buffer
                    };

                    let set = Arc::new(
                        PersistentDescriptorSet::start(pipeline.clone(), 0)
                            .add_buffer(cache_buffer.clone())
                            .unwrap()
                            .add_buffer(sums_buffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );
                    FastCacheVKObjEval {
                        n_examples: examples.len(),
                        examples: Arc::new(examples.iter().map(|(_, patch)| patch).cloned().collect()),
                        sum_batch_size: self.reduce_sum_batch_size,
                        weights: *weights,
                        head: *head,
                        embedding_type: PhantomData,
                        input_patch_type: PhantomData,
                        weights_type: PhantomData,
                        device: device,
                        queue: queue,
                        obj_sums_buffer: sums_buffer,
                        cache_buffer: cache_buffer,
                        update_pipeline: pipeline,
                        update_descriptor_set: set,
                        embedding_index: 0,
                        patch_index: 0,
                        embedding_is_clean: true,
                        next_input_words_buffer_join_handle: None,
                        next_input_words_buffer_patch_index: 0,
                        next_input_words_buffer_embedding_index: 0,
                    }
                }
            }
            impl<InputPatch: 'static + Sync + Send + GetWord + GetMirroredWords + BitLen + Copy + NewMirrorFastCache<[[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>>
                FastCacheVKObjEval<$shader_mod_name::Layout, InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>
            {
                fn sum_obj(&self) -> u64 {
                    let pa_push_constants = $shader_mod_name::ty::PushConstantData {
                        head: self.head,
                        embedding_bit_index: self.embedding_index as u32 % 16,
                        embedding_word_index: self.embedding_index as u32 / 16,
                        threshold: (<InputPatch>::BIT_LEN / 2) as u32,
                        weights_word: self.weights.get_patch(self.embedding_index).get_word(self.patch_index),
                        batch_size: self.sum_batch_size as u32,
                    };
                    let n_workgroups = ((self.n_examples as f64 / 256f64) / self.sum_batch_size as f64).ceil() as u32;
                    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                        .unwrap()
                        .dispatch(
                            [n_workgroups, 1, 1],
                            self.update_pipeline.clone(),
                            self.update_descriptor_set.clone(),
                            pa_push_constants,
                        )
                        .unwrap()
                        .build()
                        .unwrap();

                    let future = sync::now(self.device.clone())
                        .then_execute(self.queue.clone(), command_buffer)
                        .unwrap()
                        .then_signal_fence_and_flush()
                        .unwrap();
                    future.wait(None).unwrap();

                    //dbg!(self.patch_index);
                    //let start = PreciseTime::now();
                    let data_buffer_content = self.obj_sums_buffer.read().unwrap();
                    let gpu_sum: u64 = data_buffer_content.par_iter().map(|&x| x as u64).sum();
                    //println!("sum ms: {}", start.to(PreciseTime::now()).num_milliseconds());
                    gpu_sum
                }
                fn transition_input_word(&mut self, target_weight_index: usize) {
                    //let start = PreciseTime::now();
                    if (self.next_input_words_buffer_patch_index != target_weight_index) | (self.next_input_words_buffer_embedding_index != self.embedding_index) {
                        println!("an input word buffer creator thread was started, but it was for the wrong input word", );
                        if let Some(handle) = self.next_input_words_buffer_join_handle.take() {
                            handle.join().expect("can't join thread");
                        }
                        println!("it is now dead", );
                        self.next_input_words_buffer_join_handle = None;
                    }
                    if self.next_input_words_buffer_join_handle.is_none() {
                        println!("no correct input words buffer had been created so creating now...", );
                        self.start_prepare_input_words_buffer(target_weight_index);
                    }
                    let new_words_buffer = self.next_input_words_buffer_join_handle.take().unwrap().join().expect("can't join thread");
                    //println!("get pre prepared input words ms: {}", start.to(PreciseTime::now()).num_milliseconds());

                    let pipeline = Arc::new({
                        let shader = $input_trans_shader_mod_name::Shader::load(self.device.clone()).unwrap();
                        ComputePipeline::new(self.device.clone(), &shader.main_entry_point(), &()).unwrap()
                    });
                    let set = Arc::new(
                        PersistentDescriptorSet::start(pipeline.clone(), 0)
                            .add_buffer(self.cache_buffer.clone())
                            .unwrap()
                            .add_buffer(new_words_buffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );
                    let push_constants = $input_trans_shader_mod_name::ty::PushConstantData {
                        new_weights_word: self.weights.get_patch(self.embedding_index).get_word(target_weight_index),
                        old_weights_word: self.weights.get_patch(self.embedding_index).get_word(self.patch_index),
                    };
                    let n_workgroups = (self.n_examples as f64 / 1024f64).ceil() as u32;
                    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                        .unwrap()
                        .dispatch([n_workgroups, 1, 1], pipeline.clone(), set.clone(), push_constants)
                        .unwrap()
                        .build()
                        .unwrap();

                    let future = sync::now(self.device.clone())
                        .then_execute(self.queue.clone(), command_buffer)
                        .unwrap()
                        .then_signal_fence_and_flush()
                        .unwrap();
                    future.wait(None).unwrap();
                    self.patch_index = target_weight_index;
                    //println!("full trans time: {}", start.to(PreciseTime::now()).num_milliseconds());
                }

                fn start_prepare_input_words_buffer(&mut self, target_weight_index: usize) {
                    if let Some(handle) = self.next_input_words_buffer_join_handle.take() {
                        println!("unneeded thread was created, joining before creating a new one", );
                        handle.join().expect("can't join thread");
                    }
                    let examples = self.examples.clone();
                    let queue = self.queue.clone();
                    let child = thread::spawn(move || {
                        //let start = PreciseTime::now();
                        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
                        let data_vec = pool.install(||{
                            let data_vec: Vec<[u32; 2]> = examples
                                .par_iter()
                                .map(|patch| patch.get_mirrored_words(target_weight_index))
                                .collect();
                            data_vec
                        });

                        let (new_words_buffer, _) = ImmutableBuffer::from_iter(data_vec.iter().cloned(), BufferUsage::all(), queue).unwrap();
                        //println!("new words prep time: {}", start.to(PreciseTime::now()).num_milliseconds());
                        new_words_buffer
                    });
                    self.next_input_words_buffer_join_handle = Some(child);
                    self.next_input_words_buffer_patch_index = target_weight_index;
                    self.next_input_words_buffer_embedding_index = self.embedding_index;
                }

                fn clean_embedding_bit(&mut self) {
                    //let start = PreciseTime::now();
                    let pipeline = Arc::new({
                        let shader = $clean_embedding_bit_mod_name::Shader::load(self.device.clone()).unwrap();
                        ComputePipeline::new(self.device.clone(), &shader.main_entry_point(), &()).unwrap()
                    });
                    let set = Arc::new(
                        PersistentDescriptorSet::start(pipeline.clone(), 0)
                            .add_buffer(self.cache_buffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );
                    let push_constants = $clean_embedding_bit_mod_name::ty::PushConstantData {
                        embedding_bit_index: self.embedding_index as u32 % 16,
                        embedding_word_index: self.embedding_index as u32 / 16,
                        threshold: (<InputPatch>::BIT_LEN / 2) as u32,
                        weights_word: self.weights.get_patch(self.embedding_index).get_word(self.patch_index),
                    };
                    let n_workgroups = (self.n_examples as f64 / 1024f64).ceil() as u32;
                    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                        .unwrap()
                        .dispatch([n_workgroups, 1, 1], pipeline.clone(), set.clone(), push_constants)
                        .unwrap()
                        .build()
                        .unwrap();

                    let future = sync::now(self.device.clone())
                        .then_execute(self.queue.clone(), command_buffer)
                        .unwrap()
                        .then_signal_fence_and_flush()
                        .unwrap();
                    future.wait(None).unwrap();
                    self.embedding_is_clean = true;
                    //println!("embedding bit clean ms: {}", start.to(PreciseTime::now()).num_milliseconds());
                }
                fn replace_cache_parts(&mut self, target_embedding_index: usize) {
                    if !self.embedding_is_clean {
                        self.clean_embedding_bit();
                    }
                    let new_weights_patch = self.weights.get_patch(target_embedding_index);
                    let new_cache_parts_buffer = {
                        let cache_vec: Vec<_> = self.examples
                            .par_iter()
                            .map(|patch| patch.new_parts(&new_weights_patch, 0))
                            .collect();
                        let (cache_buffer, _) = ImmutableBuffer::from_iter(cache_vec.iter().cloned(), BufferUsage::all(), self.queue.clone()).unwrap();
                        cache_buffer
                    };


                    //let start = PreciseTime::now();
                    let pipeline = Arc::new({
                        let shader = $replace_cache_parts_shader_mod_name::Shader::load(self.device.clone()).unwrap();
                        ComputePipeline::new(self.device.clone(), &shader.main_entry_point(), &()).unwrap()
                    });
                    let set = Arc::new(
                        PersistentDescriptorSet::start(pipeline.clone(), 0)
                            .add_buffer(self.cache_buffer.clone())
                            .unwrap()
                            .add_buffer(new_cache_parts_buffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );
                    let n_workgroups = (self.n_examples as f64 / 1024f64).ceil() as u32;
                    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                        .unwrap()
                        .dispatch([n_workgroups, 1, 1], pipeline.clone(), set.clone(), ())
                        .unwrap()
                        .build()
                        .unwrap();

                    let future = sync::now(self.device.clone())
                        .then_execute(self.queue.clone(), command_buffer)
                        .unwrap()
                        .then_signal_fence_and_flush()
                        .unwrap();
                    future.wait(None).unwrap();
                    self.embedding_index = target_embedding_index;
                    self.patch_index = 0;
                    dbg!(self.embedding_index);
                    dbg!(self.patch_index);
                }
                fn sum_head_obj(&mut self) -> u64 {
                    if !self.embedding_is_clean {
                        self.clean_embedding_bit();
                    }
                    let pipeline = Arc::new({
                        let shader = $head_obj_update_mod_name::Shader::load(self.device.clone()).unwrap();
                        ComputePipeline::new(self.device.clone(), &shader.main_entry_point(), &()).unwrap()
                    });
                    let set = Arc::new(
                        PersistentDescriptorSet::start(pipeline.clone(), 0)
                            .add_buffer(self.cache_buffer.clone())
                            .unwrap()
                            .add_buffer(self.obj_sums_buffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );
                    let push_constants = $head_obj_update_mod_name::ty::PushConstantData {
                        head: self.head,
                        batch_size: self.sum_batch_size as u32,
                    };
                    let n_workgroups = ((self.n_examples as f64 / 256f64) / self.sum_batch_size as f64).ceil() as u32;
                    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                        .unwrap()
                        .dispatch([n_workgroups, 1, 1], pipeline.clone(), set.clone(), push_constants)
                        .unwrap()
                        .build()
                        .unwrap();

                    let future = sync::now(self.device.clone())
                        .then_execute(self.queue.clone(), command_buffer)
                        .unwrap()
                        .then_signal_fence_and_flush()
                        .unwrap();
                    future.wait(None).unwrap();

                    let data_buffer_content = self.obj_sums_buffer.read().unwrap();
                    let gpu_sum: u64 = data_buffer_content.par_iter().map(|&x| x as u64).sum();
                    gpu_sum
                }
            }
            impl<InputPatch: 'static + BitLen + GetMirroredWords + GetWord + Copy + Sync + Send + NewMirrorFastCache<[[InputPatch; 16]; $embedding_len], [u32; $embedding_len]> + WordLen>
                ObjectiveEval<InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>
                for FastCacheVKObjEval<$shader_mod_name::Layout, InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>
            where
                [[InputPatch; 16]; $embedding_len]: FlipBitIndexed,
            {
                fn flip_weights_bit(&mut self, o: usize, i: usize) {
                    if o != self.embedding_index {
                        println!("transitioning to embedding bit {:?}", o);
                        self.replace_cache_parts(o);
                    }
                    let new_patch_index = i / 32;
                    if new_patch_index != self.patch_index {
                        self.transition_input_word(new_patch_index);
                        if (new_patch_index + 1) < InputPatch::WORD_LEN {
                            self.start_prepare_input_words_buffer(new_patch_index + 1);
                        }
                    }
                    self.weights.flip_bit_indexed(o, i);
                    self.embedding_is_clean = false;
                }
                fn flip_head_bit(&mut self, o: usize, i: usize) {
                    self.head[o].flip_bit(i);
                    if !self.embedding_is_clean {
                        self.clean_embedding_bit();
                    }
                }
                fn obj(&mut self) -> u64 {
                    if self.embedding_is_clean {
                        self.sum_head_obj()
                    } else {
                        self.sum_obj()
                    }
                }
            }
        };
    }

    impl_fastexamplemirrored_over_embedding!(
        fast_obj_update_e1,
        transition_input_word_e1,
        clean_embedding_bit_e1,
        head_obj_update_e1,
        replace_cache_parts_e1,
        1
    );
    impl_fastexamplemirrored_over_embedding!(
        fast_obj_update_e2,
        transition_input_word_e2,
        clean_embedding_bit_e2,
        head_obj_update_e2,
        replace_cache_parts_e2,
        2
    );
    impl_fastexamplemirrored_over_embedding!(
        fast_obj_update_e3,
        transition_input_word_e3,
        clean_embedding_bit_e3,
        head_obj_update_e3,
        replace_cache_parts_e3,
        3
    );

    #[cfg(test)]
    mod tests {
        use super::{
            ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEval, TestCPUObjectiveEvalCreator,
            VulkanFastCacheObjectiveEvalCreator,
        };
        use rand::prelude::*;
        use rand_hc::Hc128Rng;

        macro_rules! vk_test {
            ($weights_len:expr, $patch_size:expr, $input_len:expr, $output_len:expr) => {
                let mut rng = Hc128Rng::seed_from_u64(1);
                let weights: [[[[[u32; $input_len]; $patch_size]; $patch_size]; $weights_len];
                    $output_len] = rng.gen();
                let head: [[u32; $output_len]; 10] = rng.gen();
                let examples: Vec<(u8, [[[u32; $input_len]; $patch_size]; $patch_size])> = (0
                    ..104729)
                    .map(|_| (rng.gen_range(0, 10), rng.gen()))
                    .collect();

                let vk_eval_creator = VulkanFastCacheObjectiveEvalCreator::new(98);
                let mut vk_obj_eval = vk_eval_creator.new_obj_eval(&weights, &head, &examples);
                let test_eval_creator = TestCPUObjectiveEvalCreator::new();
                let mut test_obj_eval = test_eval_creator.new_obj_eval(&weights, &head, &examples);

                let vk_obj: u64 = vk_obj_eval.obj();
                let test_obj: u64 = test_obj_eval.obj();
                assert_eq!(vk_obj, test_obj);

                let mut rng = Hc128Rng::seed_from_u64(2);
                let weights: [[[[[u32; $input_len]; $patch_size]; $patch_size]; $weights_len];
                    $output_len] = rng.gen();
                let head: [[u32; $output_len]; 10] = rng.gen();
                let examples: Vec<(u8, [[[u32; $input_len]; $patch_size]; $patch_size])> = (0
                    ..100000)
                    .map(|_| (rng.gen_range(0, 10), rng.gen()))
                    .collect();
                let mut vk_obj_eval = vk_eval_creator.new_obj_eval(&weights, &head, &examples);
                let mut test_obj_eval = test_eval_creator.new_obj_eval(&weights, &head, &examples);

                for &(o, i) in &[
                    (5, 0),
                    (0, 0),
                    (0, 3),
                    (0, 0),
                    (3, 5),
                    (7, 3),
                    (1, 5),
                    (
                        ($weights_len * $output_len) - 1,
                        ($patch_size * $patch_size) * 32 * $input_len - 1,
                    ),
                ] {
                    vk_obj_eval.flip_weights_bit(o, i);
                    test_obj_eval.flip_weights_bit(o, i);

                    let vk_obj: u64 = vk_obj_eval.obj();
                    let test_obj: u64 = test_obj_eval.obj();
                    assert_eq!(vk_obj, test_obj);
                }

                for updates in &[
                    vec![(0, 0), (0, 0), (0, 0)],
                    vec![(0, 0), (2, 0), (2, 3)],
                    vec![(5, 0), (5, 1), (5, 2)],
                    vec![(3, 1), (3, 2), (4, 3)],
                    vec![(0, 0)],
                    vec![],
                ] {
                    for &(i, o) in updates {
                        vk_obj_eval.flip_weights_bit(o, i);
                        test_obj_eval.flip_weights_bit(o, i);
                    }

                    let vk_obj: u64 = vk_obj_eval.obj();
                    let test_obj: u64 = test_obj_eval.obj();
                    assert_eq!(vk_obj, test_obj);
                }
                for updates in &[
                    vec![(0, 0), (0, 0), (0, 0)],
                    vec![(0, 0), (2, 0), (2, 3)],
                    vec![(5, 0), (5, 1), (5, 2)],
                    vec![(3, 1), (3, 2), (4, 3)],
                    vec![(0, 0)],
                    vec![],
                ] {
                    for &(i, o) in updates {
                        vk_obj_eval.flip_head_bit(o, i);
                        test_obj_eval.flip_head_bit(o, i);
                    }

                    let vk_obj: u64 = vk_obj_eval.obj();
                    let test_obj: u64 = test_obj_eval.obj();
                    assert_eq!(vk_obj, test_obj);
                }
            };
        }

        #[test]
        fn test_cpu_obj_array() {
            // use a prime to make the shader code more likely to fail.
            const N_EXAMPLES: usize = 104729;
            let mut rng = Hc128Rng::seed_from_u64(42);
            let weights: [[[[[u32; 2]; 3]; 3]; 32]; 2] = rng.gen();
            let head: [[u32; 2]; 10] = rng.gen();
            let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..N_EXAMPLES)
                .map(|_| (rng.gen_range(0, 10), rng.gen()))
                .collect();

            let eval_creator: TestCPUObjectiveEvalCreator = TestCPUObjectiveEvalCreator::new();
            let mut obj_eval: TestCPUObjectiveEval<
                [[[u32; 2]; 3]; 3],
                [[[[[u32; 2]; 3]; 3]; 32]; 2],
                [u32; 2],
            > = eval_creator.new_obj_eval(&weights, &head, &examples);
            let obj1: u64 = obj_eval.obj();
            let avg_obj = obj1 as f64 / N_EXAMPLES as f64;
            assert!(avg_obj > 0.07);
            assert!(avg_obj < 0.09);

            obj_eval.flip_weights_bit(22, 5);
            let obj2: u64 = obj_eval.obj();
            let avg_obj = obj2 as f64 / N_EXAMPLES as f64;
            assert!(avg_obj > 0.07);
            assert!(avg_obj < 0.09);

            assert_ne!(obj1, obj2);

            obj_eval.flip_weights_bit(22, 5);
            let obj3: u64 = obj_eval.obj();
            assert_eq!(obj1, obj3);
        }
        #[test]
        fn test_cpu_obj_array_mirror() {
            // use a prime to make the shader code more likely to fail.
            const N_EXAMPLES: usize = 104729;
            let mut rng = Hc128Rng::seed_from_u64(42);
            let weights: [[[[[u32; 2]; 3]; 3]; 16]; 2] = rng.gen();
            let head: [[u32; 2]; 10] = rng.gen();
            let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..N_EXAMPLES)
                .map(|_| (rng.gen_range(0, 10), rng.gen()))
                .collect();

            let eval_creator: TestCPUObjectiveEvalCreator = TestCPUObjectiveEvalCreator::new();
            let mut obj_eval: TestCPUObjectiveEval<
                [[[u32; 2]; 3]; 3],
                [[[[[u32; 2]; 3]; 3]; 16]; 2],
                [u32; 2],
            > = eval_creator.new_obj_eval(&weights, &head, &examples);
            let obj1: u64 = obj_eval.obj();
            let avg_obj = obj1 as f64 / N_EXAMPLES as f64;
            assert!(avg_obj > 0.07);
            assert!(avg_obj < 0.09);

            obj_eval.flip_weights_bit(22, 5);
            let obj2: u64 = obj_eval.obj();
            let avg_obj = obj2 as f64 / N_EXAMPLES as f64;
            assert!(avg_obj > 0.07);
            assert!(avg_obj < 0.09);

            assert_ne!(obj1, obj2);

            obj_eval.flip_weights_bit(22, 5);
            let obj3: u64 = obj_eval.obj();
            assert_eq!(obj1, obj3);
        }

        #[test]
        fn vk_array_16_2x2_2_1() {
            vk_test!(16, 2, 2, 1);
        }
        #[test]
        fn vk_array_16_2x2_1_2() {
            vk_test!(16, 2, 1, 2);
        }
        #[test]
        fn vk_array_16_2x2_1_1() {
            vk_test!(16, 2, 1, 1);
        }
        #[test]
        fn vk_array_16_2x2_1_3() {
            vk_test!(16, 2, 1, 3);
        }
        #[test]
        fn vk_array_16_3x3_1_1() {
            vk_test!(16, 3, 1, 1);
        }
        #[test]
        fn vk_array_16_3x3_2_1() {
            vk_test!(16, 3, 2, 1);
        }
        #[test]
        fn vk_array_16_3x3_3_1() {
            vk_test!(16, 3, 3, 1);
        }
        #[test]
        fn vk_array_16_3x3_2_2() {
            vk_test!(16, 3, 2, 2);
        }
        #[test]
        fn vk_array_16_3x3_3_2() {
            vk_test!(16, 3, 3, 2);
        }
    }
}
