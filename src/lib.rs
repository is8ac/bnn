#[macro_use]
extern crate serde_derive;
extern crate bincode;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate time;
extern crate vulkano;
extern crate vulkano_shaders;
use rand::prelude::*;
use rayon::prelude::*;
use std::iter;

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

        fn rgb_to_u32(pixels: [u8; 3]) -> u32 {
            to_11(pixels[0]) as u32 | ((to_11(pixels[1]) as u32) << 11) | ((to_10(pixels[2]) as u32) << 22)
        }

        pub fn load_images_from_base(base_path: &Path, n: usize) -> Vec<(usize, [[[u32; 1]; 32]; 32])> {
            if n > 50000 {
                panic!("n must be <= 50,000");
            }
            (1..6)
                .map(|i| {
                    let mut file = File::open(&base_path.join(format!("data_batch_{}.bin", i))).expect("can't open data");

                    let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
                    let mut label: [u8; 1] = [0; 1];
                    let mut images: Vec<(usize, [[[u32; 1]; 32]; 32])> = Vec::new();
                    for _ in 0..10000 {
                        file.read_exact(&mut label).expect("can't read label");
                        file.read_exact(&mut image_bytes).expect("can't read images");
                        let mut image = [[[0u32]; 32]; 32];
                        for x in 0..32 {
                            for y in 0..32 {
                                let pixel = [
                                    image_bytes[(0 * 1024) + (y * 32) + x],
                                    image_bytes[(1 * 1024) + (y * 32) + x],
                                    image_bytes[(2 * 1024) + (y * 32) + x],
                                ];
                                image[x][y][0] = rgb_to_u32(pixel);
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

pub trait FlipBitIndexed {
    fn flip_bit_indexed(&mut self, o: usize, b: usize);
}

impl<T: FlipBit> FlipBitIndexed for [T; 32] {
    fn flip_bit_indexed(&mut self, o: usize, b: usize) {
        self[o].flip_bit(b);
    }
}

macro_rules! impl_flipbitindexed_for_array {
    ($len:expr) => {
        impl<T: FlipBit> FlipBitIndexed for [[T; 32]; $len] {
            fn flip_bit_indexed(&mut self, o: usize, b: usize) {
                self[o / 32][o % 32].flip_bit(b);
            }
        }
    };
}

impl_flipbitindexed_for_array!(1);
impl_flipbitindexed_for_array!(2);

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

#[macro_use]
pub mod layers {
    use super::{BitLen, HammingDistance};
    use bincode::{deserialize_from, serialize_into};
    use rayon::prelude::*;
    use std::fs::File;
    use std::io::BufWriter;
    use std::marker::PhantomData;
    use std::path::Path;
    use time::PreciseTime;

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
                [$input[$x + 0][$y + 0], $input[$x + 0][$y + 1], $input[$x + 0][$y + 2]],
                [$input[$x + 1][$y + 0], $input[$x + 1][$y + 1], $input[$x + 1][$y + 2]],
                [$input[$x + 2][$y + 0], $input[$x + 2][$y + 1], $input[$x + 2][$y + 2]],
            ]
        };
    }
    pub trait Apply<I, O> {
        fn apply(&self, input: &I) -> O;
    }

    pub trait SaveLoad
    where
        Self: Sized,
    {
        fn write_to_fs(&self, path: &Path);
        fn new_from_fs(path: &Path) -> Option<Self>;
    }

    macro_rules! conv3x3_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, O: Copy + Default, W: Apply<[[I; 3]; 3], O>> Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]> for W {
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


    pub trait NewFromRng {
        fn new_from_rng<RNG: rand::Rng>(rng: &mut RNG) -> Self;
    }

    macro_rules! primitive_activations_new_from_seed {
        ($len:expr) => {
            impl<I> NewFromRng for [I; $len]
            where
                rand::distributions::Standard: rand::distributions::Distribution<I>,
            {
                fn new_from_rng<RNG: rand::Rng>(rng: &mut RNG) -> Self {
                    rng.gen::<[I; $len]>()
                }
            }
        };
    }
    primitive_activations_new_from_seed!(8);
    primitive_activations_new_from_seed!(10);
    primitive_activations_new_from_seed!(16);
    primitive_activations_new_from_seed!(32);

    #[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
    pub struct Conv3x3<I, FN, O> {
        input_pixel_type: PhantomData<I>,
        pub map_fn: FN,
        output_type: PhantomData<O>,
    }

    macro_rules! conv3x3_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, FN: Apply<[[I; 3]; 3], O>, O: Default + Copy> Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]>
                for Conv3x3<I, FN, O>
            {
                fn apply(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size]; $x_size] {
                    let mut target = [[O::default(); $y_size]; $x_size];
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            target[x + 1][y + 1] = self.map_fn.apply(&patch_3x3!(input, x, y));
                        }
                    }
                    target
                }
            }
        };
    }

    //conv3x3_apply_trait!(32, 32);
    //conv3x3_apply_trait!(16, 16);
    //conv3x3_apply_trait!(8, 8);
    //conv3x3_apply_trait!(4, 4);

    macro_rules! impl_saveload_conv3x3_array {
        ($input_len:expr, $output_len:expr) => {
            impl SaveLoad for [[[[[u32; $input_len]; 3]; 3]; 32]; $output_len] {
                fn write_to_fs(&self, path: &Path) {
                    let vec_params: Vec<[[[[u32; $input_len]; 3]; 3]; 32]> = self.iter().cloned().collect();
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
                        .map(|vec_params: Vec<[[[[u32; $input_len]; 3]; 3]; 32]>| {
                            if vec_params.len() != $output_len {
                                panic!("input is of len {} not {}", vec_params.len(), $output_len);
                            }
                            let mut params = [<[[[[u32; $input_len]; 3]; 3]; 32]>::default(); $output_len];
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
    impl_saveload_conv3x3_array!(2, 1);
    impl_saveload_conv3x3_array!(1, 1);


    macro_rules! impl_saveload_conv3x3_32 {
        ($len:expr) => {
            impl SaveLoad for [[[[u32; $len]; 3]; 3]; 32] {
                fn write_to_fs(&self, path: &Path) {
                    let vec_params: Vec<[[[u32; $len]; 3]; 3]> = self.iter().cloned().collect();
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
                        .map(|vec_params: Vec<[[[u32; $len]; 3]; 3]>| {
                            if vec_params.len() != 32 {
                                panic!("input is of len {} not {}", vec_params.len(), 32);
                            }
                            let mut params = [<[[[u32; $len]; 3]; 3]>::default(); 32];
                            for i in 0..32 {
                                params[i] = vec_params[i];
                            }
                            params
                        })
                        .ok()
                }
            }
        };
    }
    impl_saveload_conv3x3_32!(1);
    impl_saveload_conv3x3_32!(2);
    impl_saveload_conv3x3_32!(3);

    macro_rules! impl_saveload_conv3x3 {
        ($len:expr) => {
            impl<I: Default + Copy, O> SaveLoad for Conv3x3<I, [[[I; 3]; 3]; $len], O>
            where
                I: serde::Serialize,
                for<'de> I: serde::Deserialize<'de>,
            {
                fn write_to_fs(&self, path: &Path) {
                    let vec_params: Vec<[[I; 3]; 3]> = self.map_fn.iter().map(|x| *x).collect();
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
                        .map(|vec_params: Vec<[[I; 3]; 3]>| {
                            if vec_params.len() != $len {
                                panic!("input is of len {} not {}", vec_params.len(), $len);
                            }
                            let mut params = [<[[I; 3]; 3]>::default(); $len];
                            for i in 0..$len {
                                params[i] = vec_params[i];
                            }
                            Conv3x3 {
                                input_pixel_type: PhantomData,
                                map_fn: params,
                                output_type: PhantomData,
                            }
                        })
                        .ok()
                }
            }
        };
    }
    impl_saveload_conv3x3!(8);
    impl_saveload_conv3x3!(16);
    impl_saveload_conv3x3!(32);
    impl_saveload_conv3x3!(64);
    impl_saveload_conv3x3!(128);

}

pub trait Extract3x3Patches<P> {
    fn patches(&self) -> Vec<[[P; 3]; 3]>;
}

macro_rules! extract_patch_3x3_trait {
    ($x_size:expr, $y_size:expr) => {
        impl<P: Copy> Extract3x3Patches<P> for [[P; $y_size]; $x_size] {
            fn patches(&self) -> Vec<[[P; 3]; 3]> {
                let mut patches = Vec::with_capacity(($y_size - 2) * ($x_size - 2));
                for x in 0..$x_size - 2 {
                    for y in 0..$y_size - 2 {
                        patches.push(patch_3x3!(self, x, y));
                    }
                }
                patches
            }
        }
    };
}

extract_patch_3x3_trait!(32, 32);
extract_patch_3x3_trait!(16, 16);
extract_patch_3x3_trait!(8, 8);
extract_patch_3x3_trait!(4, 4);

pub trait Extract2x2Patches<P> {
    fn patches(&self) -> Vec<[[P; 2]; 2]>;
}

macro_rules! extract_patch_2x2_trait {
    ($x_size:expr, $y_size:expr) => {
        impl<P: Copy> Extract2x2Patches<P> for [[P; $y_size]; $x_size] {
            fn patches(&self) -> Vec<[[P; 2]; 2]> {
                let mut patches = Vec::with_capacity(($y_size / 2) * ($x_size / 2));
                for x in 0..($x_size / 2) {
                    let x_base = x * 2;
                    for y in 0..($y_size / 2) {
                        let y_base = y * 2;
                        patches.push(patch_2x2!(self, x_base, y_base));
                    }
                }
                patches
            }
        }
    };
}

extract_patch_2x2_trait!(32, 32);
extract_patch_2x2_trait!(16, 16);
extract_patch_2x2_trait!(8, 8);

pub trait ConcatImages<I> {
    fn concat_images(inputs: I) -> Self;
}

macro_rules! impl_concat_image {
    ($len:expr, $depth:expr, $x_size:expr, $y_size:expr) => {
        impl<P: Default + Copy> ConcatImages<[&[[[P; $len]; $y_size]; $x_size]; $depth]> for [[[P; $depth]; $y_size]; $x_size] {
            fn concat_images(input: [&[[[P; $len]; $y_size]; $x_size]; $depth]) -> [[[P; $depth]; $y_size]; $x_size] {
                let mut target = <[[[P; $depth]; $y_size]; $x_size]>::default();
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

impl_concat_image!(1, 2, 32, 32);
impl_concat_image!(1, 3, 32, 32);
impl_concat_image!(1, 4, 32, 32);

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

pub fn vec_extract_patches<I: Extract3x3Patches<P>, P, RNG: rand::Rng>(rng: &mut RNG, images: &Vec<(usize, I)>) -> Vec<(u8, [[P; 3]; 3])> {
    // decompose the images into patches,
    let mut patches: Vec<(u8, [[P; 3]; 3])> = images
        .iter()
        .map(|(class, image)| iter::repeat(*class as u8).zip(image.patches()))
        .flatten()
        .collect();
    // and shuffle them
    patches.shuffle(rng);
    patches
}

pub mod objective_eval {
    extern crate rand;
    extern crate time;
    use super::layers::Apply;
    use super::{BitLen, FlipBit, FlipBitIndexed, HammingDistance};
    use rayon::prelude::*;
    use std::collections::HashSet;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use time::PreciseTime;
    use vulkano::buffer::BufferUsage;
    use vulkano::buffer::{CpuAccessibleBuffer, DeviceLocalBuffer, ImmutableBuffer};
    use vulkano::command_buffer::AutoCommandBufferBuilder;
    use vulkano::command_buffer::CommandBuffer;
    use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuf, StdDescriptorPoolAlloc};
    use vulkano::descriptor::pipeline_layout::PipelineLayout;
    use vulkano::device::DeviceExtensions;
    use vulkano::device::Features;
    use vulkano::device::{Device, Queue};
    use vulkano::instance::InstanceExtensions;
    use vulkano::instance::PhysicalDevice;
    use vulkano::instance::{Instance, QueueFamily};
    use vulkano::pipeline::ComputePipeline;
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
        fn new_obj_eval(&self, weights: &Weights, head: &[Embedding; 10], examples: &[(u8, InputPatch)]) -> Self::ObjectiveEvalType;
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

    impl<Patch: Copy, Weights: Apply<Patch, Embedding> + Copy, Embedding: Copy> ObjectiveEvalCreator<Patch, Weights, Embedding>
        for TestCPUObjectiveEvalCreator
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
    impl<Patch: Sync, Weights: Sync + Apply<Patch, Embedding> + FlipBitIndexed, Embedding: Sync + FlipBit> ObjectiveEval<Patch, Weights, Embedding>
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

    pub struct GlslLikeCPUObjectiveEvalCreator {}

    impl GlslLikeCPUObjectiveEvalCreator {
        pub fn new() -> Self {
            GlslLikeCPUObjectiveEvalCreator {}
        }
    }

    impl ObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> for GlslLikeCPUObjectiveEvalCreator {
        type ObjectiveEvalType = GlslLikeCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>;
        fn new_obj_eval(
            &self,
            weights: &[[[[u32; 2]; 3]; 3]; 32],
            head: &[u32; 10],
            examples: &[(u8, [[[u32; 2]; 3]; 3])],
        ) -> GlslLikeCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> {
            GlslLikeCPUObjectiveEval {
                weights: *weights,
                head: *head,
                examples: examples.iter().cloned().collect(),
            }
        }
    }

    pub struct GlslLikeCPUObjectiveEval<InputPatch, Weights, Embedding> {
        weights: Weights,
        head: [Embedding; 10],
        examples: Vec<(u8, InputPatch)>,
    }

    // This is an implementation of obj() that is almost identical to the glsl implementation.
    impl ObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
        for GlslLikeCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    {
        fn flip_weights_bit(&mut self, o: usize, i: usize) {
            self.weights[o].flip_bit(i);
        }
        fn flip_head_bit(&mut self, o: usize, i: usize) {
            self.head[o].flip_bit(i);
        }
        fn obj(&mut self) -> u64 {
            let threshold: u32 = ((32 * 2) * 9) / 2;
            self.examples
                .par_iter()
                .map(|(class, patch)| {
                    let mut target = 0b0u32;
                    for b in 0..32 {
                        let mut sum = 0;
                        for x in 0..3 {
                            for y in 0..3 {
                                for i in 0..2 {
                                    sum += (patch[x][y][i] ^ self.weights[b][x][y][i]).count_ones();
                                }
                            }
                        }
                        target |= ((sum > threshold) as u32) << b;
                    }
                    let mut is_good = true;
                    let max_obj = (self.head[*class as usize] ^ target).count_ones();
                    for o in 0..10 {
                        is_good = (((self.head[o] ^ target).count_ones() < max_obj) | (o == *class as usize)) & is_good;
                    }
                    is_good as u64
                })
                .sum()
        }
    }

    mod reduce_sum {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "shaders/reduce_sum.glsl",
        }
    }

    pub struct VulkanObjectiveEvalCreator {
        instance: Arc<Instance>,
        reduce_sum_batch_size: usize,
    }

    pub struct VulkanObjectiveEval<ApplyLayout, ReduceSumLayout, InputPatch, Weights, Embedding> {
        N_EXAMPLES: usize,
        input_patch_type: PhantomData<InputPatch>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        weights_buffer: Arc<CpuAccessibleBuffer<Weights>>,
        head_buffer: Arc<CpuAccessibleBuffer<[Embedding; 10]>>,
        obj_sums_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
        apply_compute_pipeline: Arc<ComputePipeline<PipelineLayout<ApplyLayout>>>,
        apply_descriptor_set: Arc<
            PersistentDescriptorSet<
                Arc<ComputePipeline<PipelineLayout<ApplyLayout>>>,
                (
                    (
                        (
                            (
                                (
                                    ((), PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<Weights>>>),
                                    PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<[Embedding; 10]>>>,
                                ),
                                PersistentDescriptorSetBuf<Arc<ImmutableBuffer<[InputPatch]>>>,
                            ),
                            PersistentDescriptorSetBuf<Arc<ImmutableBuffer<[u32]>>>,
                        ),
                        PersistentDescriptorSetBuf<Arc<DeviceLocalBuffer<[Embedding]>>>,
                    ),
                    PersistentDescriptorSetBuf<Arc<DeviceLocalBuffer<[u32]>>>,
                ),
                StdDescriptorPoolAlloc,
            >,
        >,
        reduce_sum_compute_pipeline: Arc<ComputePipeline<PipelineLayout<ReduceSumLayout>>>,
        reduce_sum_descriptor_set: Arc<
            PersistentDescriptorSet<
                Arc<ComputePipeline<PipelineLayout<reduce_sum::Layout>>>,
                (
                    ((), PersistentDescriptorSetBuf<Arc<DeviceLocalBuffer<[u32]>>>),
                    PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<[u32]>>>,
                ),
                StdDescriptorPoolAlloc,
            >,
        >,
        is_initialized: bool,
        embedding_is_clean: bool,
        obj_sum_is_clean: bool,
        unclean_output_bits: HashSet<usize>,
        reduce_sum_batch_size: usize,
    }

    impl VulkanObjectiveEvalCreator {
        pub fn new(reduce_sum_batch_size: usize) -> Self {
            let instance = Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

            Self {
                instance: instance,
                reduce_sum_batch_size: reduce_sum_batch_size,
            }
        }
    }

    macro_rules! impl_objectiveevalcreator_for_vulkanobjectiveevalcreator {
        ($input_len:expr, $output_len:expr, $shader_mod_name:ident, $shader_path:expr) => {
            mod $shader_mod_name {
                vulkano_shaders::shader! {
                    ty: "compute",
                    path: $shader_path,
                }
            }

            impl ObjectiveEvalCreator<[[[u32; $input_len]; 3]; 3], [[[[[u32; $input_len]; 3]; 3]; 32]; $output_len], [u32; $output_len]>
                for VulkanObjectiveEvalCreator
            {
                type ObjectiveEvalType = VulkanObjectiveEval<
                    $shader_mod_name::Layout,
                    reduce_sum::Layout,
                    [[[u32; $input_len]; 3]; 3],
                    [[[[[u32; $input_len]; 3]; 3]; 32]; $output_len],
                    [u32; $output_len],
                >;
                fn new_obj_eval(
                    &self,
                    weights: &[[[[[u32; $input_len]; 3]; 3]; 32]; $output_len],
                    head: &[[u32; $output_len]; 10],
                    examples: &[(u8, [[[u32; $input_len]; 3]; 3])],
                ) -> Self::ObjectiveEvalType {
                    let physical = PhysicalDevice::enumerate(&self.instance).next().expect("no device available");

                    let queue_family: QueueFamily = physical
                        .queue_families()
                        .find(|&q| q.supports_compute())
                        .expect("couldn't find a compute queue family");

                    let (device, mut queues) = Device::new(
                        physical,
                        &Features::none(),
                        &DeviceExtensions::none(),
                        [(queue_family, 0.5)].iter().cloned(),
                    )
                    .expect("failed to create device");

                    let queue = queues.next().unwrap();

                    let prm_buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), *weights).expect("failed to create buffer");

                    let head_buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), *head).expect("failed to create buffer");

                    let (input_buffer, _) =
                        ImmutableBuffer::from_iter(examples.iter().map(|(_, input)| input).cloned(), BufferUsage::all(), queue.clone())
                            .expect("failed to create buffer");

                    let (labels_buffer, _) =
                        ImmutableBuffer::from_iter(examples.iter().map(|(label, _)| *label as u32), BufferUsage::all(), queue.clone())
                            .expect("failed to create buffer");

                    let embeddings_buffer: Arc<DeviceLocalBuffer<[[u32; $output_len]]>> =
                        DeviceLocalBuffer::array(device.clone(), examples.len(), BufferUsage::all(), [queue_family].iter().cloned())
                            .expect("failed to create DeviceLocalBuffer");

                    let objs_buffer: Arc<DeviceLocalBuffer<[u32]>> =
                        DeviceLocalBuffer::array(device.clone(), examples.len(), BufferUsage::all(), [queue_family].iter().cloned())
                            .expect("failed to create DeviceLocalBuffer");

                    let obj_sums_iter = (0..(examples.len() as f64 / self.reduce_sum_batch_size as f64).ceil() as usize).map(|_| 0u32);
                    let obj_sums_buffer =
                        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), obj_sums_iter).expect("failed to create buffer");

                    let pa_shader = $shader_mod_name::Shader::load(device.clone()).expect("failed to create shader module");
                    let pa_compute_pipeline = Arc::new(
                        ComputePipeline::new(device.clone(), &pa_shader.main_entry_point(), &()).expect("failed to create compute pipeline"),
                    );

                    let pa_set = Arc::new(
                        PersistentDescriptorSet::start(pa_compute_pipeline.clone(), 0)
                            .add_buffer(prm_buffer.clone())
                            .unwrap()
                            .add_buffer(head_buffer.clone())
                            .unwrap()
                            .add_buffer(input_buffer.clone())
                            .unwrap()
                            .add_buffer(labels_buffer.clone())
                            .unwrap()
                            .add_buffer(embeddings_buffer.clone())
                            .unwrap()
                            .add_buffer(objs_buffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );

                    let rs_shader = reduce_sum::Shader::load(device.clone()).expect("failed to create shader module");
                    let rs_compute_pipeline = Arc::new(
                        ComputePipeline::new(device.clone(), &rs_shader.main_entry_point(), &()).expect("failed to create compute pipeline"),
                    );

                    let rs_set = Arc::new(
                        PersistentDescriptorSet::start(rs_compute_pipeline.clone(), 0)
                            .add_buffer(objs_buffer.clone())
                            .unwrap()
                            .add_buffer(obj_sums_buffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    );

                    Self::ObjectiveEvalType {
                        N_EXAMPLES: examples.len(),
                        input_patch_type: PhantomData,
                        device: device.clone(),
                        queue: queue,
                        weights_buffer: prm_buffer,
                        head_buffer: head_buffer,
                        obj_sums_buffer: obj_sums_buffer,
                        apply_compute_pipeline: pa_compute_pipeline,
                        apply_descriptor_set: pa_set,
                        reduce_sum_compute_pipeline: rs_compute_pipeline,
                        reduce_sum_descriptor_set: rs_set,
                        is_initialized: false,
                        embedding_is_clean: false,
                        obj_sum_is_clean: false,
                        unclean_output_bits: HashSet::new(),
                        reduce_sum_batch_size: self.reduce_sum_batch_size,
                    }
                }
            }

            impl
                VulkanObjectiveEval<
                    $shader_mod_name::Layout,
                    reduce_sum::Layout,
                    [[[u32; $input_len]; 3]; 3],
                    [[[[[u32; $input_len]; 3]; 3]; 32]; $output_len],
                    [u32; $output_len],
                >
            {
                fn patch_apply(&mut self, output_index: usize, apply_level: u32) {
                    let pa_push_constants = $shader_mod_name::ty::PushConstantData {
                        embedding_word_index: output_index as u32 / 32,
                        embedding_bit_index: output_index as u32 % 32,
                        full_apply: apply_level,
                    };
                    let pa_command_buffer = AutoCommandBufferBuilder::new(self.device.clone(), self.queue.family())
                        .unwrap()
                        .dispatch(
                            [(self.N_EXAMPLES as f64 / 64f64).ceil() as u32, 1, 1],
                            self.apply_compute_pipeline.clone(),
                            self.apply_descriptor_set.clone(),
                            pa_push_constants,
                        )
                        .unwrap()
                        .build()
                        .unwrap();

                    let finished = pa_command_buffer.execute(self.queue.clone()).unwrap();
                    finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();
                }
                fn reduce_sum_obj(&mut self) -> u64 {
                    let rs_push_constants = reduce_sum::ty::PushConstantData {
                        batch_size: self.reduce_sum_batch_size as u32,
                    };

                    let rs_command_buffer = AutoCommandBufferBuilder::new(self.device.clone(), self.queue.family())
                        .unwrap()
                        .dispatch(
                            [
                                ((self.N_EXAMPLES as f64 / 64f64) / self.reduce_sum_batch_size as f64).ceil() as u32,
                                1,
                                1,
                            ],
                            self.reduce_sum_compute_pipeline.clone(),
                            self.reduce_sum_descriptor_set.clone(),
                            rs_push_constants,
                        )
                        .unwrap()
                        .build()
                        .unwrap();

                    let finished = rs_command_buffer.execute(self.queue.clone()).unwrap();

                    finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();
                    self.obj_sum_is_clean = true;
                    let content = self.obj_sums_buffer.read().unwrap();
                    content.par_iter().map(|x| *x as u64).sum()
                }
            }

            impl ObjectiveEval<[[[u32; $input_len]; 3]; 3], [[[[[u32; $input_len]; 3]; 3]; 32]; $output_len], [u32; $output_len]>
                for VulkanObjectiveEval<
                    $shader_mod_name::Layout,
                    reduce_sum::Layout,
                    [[[u32; $input_len]; 3]; 3],
                    [[[[[u32; $input_len]; 3]; 3]; 32]; $output_len],
                    [u32; $output_len],
                >
            {
                fn flip_weights_bit(&mut self, o: usize, i: usize) {
                    let mut weights = self.weights_buffer.write().unwrap();
                    weights.flip_bit_indexed(o, i);
                    self.unclean_output_bits.insert(o);
                    self.embedding_is_clean = false;
                    self.obj_sum_is_clean = false;
                }
                fn flip_head_bit(&mut self, o: usize, i: usize) {
                    let mut head = self.head_buffer.write().unwrap();
                    head[o].flip_bit(i);
                    self.obj_sum_is_clean = false;
                }
                fn obj(&mut self) -> u64 {
                    if !self.is_initialized {
                        // if this the first time and the embedding is empty, do a full apply.
                        self.patch_apply(0, 1);
                    } else if !self.embedding_is_clean {
                        let indices: Vec<usize> = self.unclean_output_bits.iter().cloned().collect();
                        for o in &indices {
                            self.patch_apply(*o, 0);
                        }
                    } else if self.embedding_is_clean {
                        self.patch_apply(0, 2);
                    } else {

                    }
                    self.unclean_output_bits.clear();
                    self.is_initialized = true;
                    self.embedding_is_clean = true;
                    self.reduce_sum_obj()
                }
            }
        };
    }
    impl_objectiveevalcreator_for_vulkanobjectiveevalcreator!(1, 1, apply_shader_1_1, "shaders/patch_apply_1-1.glsl");
    impl_objectiveevalcreator_for_vulkanobjectiveevalcreator!(2, 1, apply_shader_2_1, "shaders/patch_apply_2-1.glsl");
    //impl_objectiveevalcreator_for_vulkanobjectiveevalcreator!(3,1, apply_shader_3_1, "shaders/patch_apply_3-1.glsl");

    #[cfg(test)]
    mod tests {
        use super::{FlipBit, FlipBitIndexed};
        use super::{
            ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEval, TestCPUObjectiveEvalCreator, VulkanObjectiveEval, VulkanObjectiveEvalCreator,
        };
        use crate::layers::Apply;
        use rand::prelude::*;
        use rand_hc::Hc128Rng;

        #[test]
        fn test_cpu_obj_array() {
            // use a prime to make the shader code more likely to fail.
            const N_EXAMPLES: usize = 104729;
            let mut rng = Hc128Rng::seed_from_u64(42);
            let mut weights: [[[[[u32; 2]; 3]; 3]; 32]; 2] = rng.gen();
            let mut head: [[u32; 2]; 10] = rng.gen();
            let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..N_EXAMPLES).map(|_| (rng.gen_range(0, 10), rng.gen())).collect();

            let eval_creator: TestCPUObjectiveEvalCreator = TestCPUObjectiveEvalCreator::new();
            let mut obj_eval: TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[[u32; 2]; 3]; 3]; 32]; 2], [u32; 2]> =
                eval_creator.new_obj_eval(&weights, &head, &examples);
            let obj1: u64 = obj_eval.obj();
            dbg!(obj1);
            let avg_obj = obj1 as f64 / N_EXAMPLES as f64;
            dbg!(avg_obj);
            assert!(avg_obj > 0.07);
            assert!(avg_obj < 0.09);

            obj_eval.flip_weights_bit(22, 5);
            let obj2: u64 = obj_eval.obj();
            dbg!(obj2);
            let avg_obj = obj2 as f64 / N_EXAMPLES as f64;
            dbg!(avg_obj);
            assert!(avg_obj > 0.07);
            assert!(avg_obj < 0.09);

            assert_ne!(obj1, obj2);

            obj_eval.flip_weights_bit(22, 5);
            let obj3: u64 = obj_eval.obj();
            assert_eq!(obj1, obj3);
        }
        #[test]
        fn vulkan_obj() {
            let mut rng = Hc128Rng::seed_from_u64(1);
            let weights: [[[[[u32; 2]; 3]; 3]; 32]; 1] = rng.gen();
            let head: [[u32; 1]; 10] = rng.gen();
            let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..104729).map(|_| (rng.gen_range(0, 10), rng.gen())).collect();

            let vk_eval_creator = VulkanObjectiveEvalCreator::new(98);
            let mut vk_obj_eval = vk_eval_creator.new_obj_eval(&weights, &head, &examples);
            let test_eval_creator = TestCPUObjectiveEvalCreator::new();
            let mut test_obj_eval = test_eval_creator.new_obj_eval(&weights, &head, &examples);

            let vk_obj: u64 = vk_obj_eval.obj();
            let test_obj: u64 = test_obj_eval.obj();
            assert_eq!(vk_obj, test_obj);

            let mut rng = Hc128Rng::seed_from_u64(2);
            let weights: [[[[[u32; 2]; 3]; 3]; 32]; 1] = rng.gen();
            let head: [[u32; 1]; 10] = rng.gen();
            let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..100000).map(|_| (rng.gen_range(0, 10), rng.gen())).collect();
            let mut vk_obj_eval = vk_eval_creator.new_obj_eval(&weights, &head, &examples);
            let mut test_obj_eval = test_eval_creator.new_obj_eval(&weights, &head, &examples);

            for &(o, i) in &[(0, 0), (0, 5), (1, 5), (0, 0), (31, 9 * 32 * 2 - 1)] {
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
        }
    }
}
