extern crate bincode;
extern crate bitnn;
extern crate image;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;
use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::{
    Apply, IsCorrect, Mutate, NewFromRng, Objective, OptimizeHead, OptimizeLayer, PixelMap,
    SaveLoad, VecApply,
};
use bitnn::{BitLen, HalfLen, HammingDist, Patch};
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::convert::{From, Into};
use std::marker::PhantomData;
use std::ops::{Add, BitAndAssign, BitOrAssign, BitXor, Not, Shl};
use std::ops::{Index, IndexMut};
use std::path::Path;
use time::PreciseTime;

pub fn u32_to_rbg(bits: u32) -> [u8; 3] {
    [
        ((bits & ((!0b0u32) << 11)).count_ones() * 11) as u8,
        (((bits << 11) & ((!0b0u32) << 11)).count_ones() * 11) as u8,
        (((bits << 22) & ((!0b0u32) << 10)).count_ones() * 10) as u8,
    ]
}

fn save_image(input: &[[u32; 32]; 32]) {
    let imgx = 32;
    let imgy = 32;

    let mut imgbuf = image::ImageBuffer::new(imgx, imgy);

    for x in 0..imgx {
        for y in 0..imgy {
            let pixel = imgbuf.get_pixel_mut(x, y);
            *pixel = image::Rgb(u32_to_rbg(input[x as usize][y as usize]));
        }
    }

    imgbuf.save("cifar_image.png").unwrap();
}

// A TrainCache stores intermediate stages of objective computation
// so as to speed up computation when only one bit of the params has been changed.
pub trait TrainCache<I, L: Apply<I, O> + Mutate, O, H: Objective<O>>: Sized {
    fn new(layer: &L, class: usize, input: &I) -> Self;
    fn output(&mut self, &L) -> (usize, O);
    // old is the part of layer that is dirty from time.
    // new is the part of layer that was mutated this time.
    fn obj(&mut self, layer: &L, head: &H, i: usize, o: usize) -> f64;
    // self_test is used for testing that the optimized implementation is correct.
    fn self_test(layer: &mut L, class: usize, head: &H, input: &I) {
        let mut cache = Self::new(layer, class, input);
        let real_obj = head.objective(&(class, layer.apply(input)));
        let cache_obj = cache.obj(layer, head, 0, 0);
        assert_eq!(cache_obj, real_obj);

        layer.mutate(0, 1);
        let cache_obj = cache.obj(layer, head, 1, 0);
        let real_obj = head.objective(&(class, layer.apply(input)));
        assert_eq!(cache_obj, real_obj);

        layer.mutate(0, L::INPUT_LEN - 1);
        let cache_obj = cache.obj(layer, head, L::INPUT_LEN - 1, 0);
        let real_obj = head.objective(&(class, layer.apply(input)));
        assert_eq!(cache_obj, real_obj);

        layer.mutate(1, 0);
        let cache_obj = cache.obj(layer, head, 0, 1);
        let real_obj = head.objective(&(class, layer.apply(input)));
        assert_eq!(cache_obj, real_obj);
    }
}

pub struct Conv3x3Mirror<I, O: HalfLen<[[I; 3]; 3]>> {
    input_pixel_type: PhantomData<I>,
    kernel: O::HalfArray,
    output_type: PhantomData<O>,
}

trait Cacheable {
    type CacheStruct;
}

//impl<I, O, > Cacheable for Conv3x3Mirror<I,>

impl<I: BitLen, O: BitLen + HalfLen<[[I; 3]; 3]>> Mutate for Conv3x3Mirror<I, O>
where
    O::HalfArray: Mutate,
{
    fn mutate(&mut self, b: usize, i: usize) {
        self.kernel.mutate(b, i);
    }
    const OUTPUT_LEN: usize = O::BIT_LEN / 2;
    const INPUT_LEN: usize = I::BIT_LEN * 3 * 3;
}

//fn simple_apply<I: Patch + Copy + Default>(
//    weights: &Conv3x3Mirror<I, [[[I; 3]; 3]; 64], u128>,
//    input: &[[I; 32]; 32],
//) -> [[u128; 32]; 32] {
//    let mut flip_kern = [[[I::default(); 3]; 3]; 64];
//    for b in 0..64 {
//        flip_kern[b] = [
//            weights.kernel[b][2],
//            weights.kernel[b][1],
//            weights.kernel[b][0],
//        ];
//    }
//
//    let mut target = [[0u128; 32]; 32];
//    for x in 0..32 - 2 {
//        for y in 0..32 - 2 {
//            let patch = [
//                [
//                    input[x + 0][y + 0],
//                    input[x + 0][y + 1],
//                    input[x + 0][y + 2],
//                ],
//                [
//                    input[x + 1][y + 0],
//                    input[x + 1][y + 1],
//                    input[x + 1][y + 2],
//                ],
//                [
//                    input[x + 2][y + 0],
//                    input[x + 2][y + 1],
//                    input[x + 2][y + 2],
//                ],
//            ];
//            for i in 0..64 {
//                target[x + 1][y + 1] |=
//                    (weights.kernel[i].distance_and_threshold(&patch) as u128) << i;
//                target[x + 1][y + 1] |=
//                    (flip_kern[i].distance_and_threshold(&patch) as u128) << (i + 64);
//            }
//        }
//    }
//    target
//}

macro_rules! impl_conv3x3mirror_apply {
    ($x_size:expr, $y_size:expr, $output_type:ty) => {
        impl<
                I: Copy + Patch + BitLen + Default,
                O: HalfLen<[[I; 3]; 3]>
                    + BitLen
                    + Default
                    + BitOrAssign
                    + Copy
                    + From<bool>
                    + Shl<usize, Output = O>,
            > Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]> for Conv3x3Mirror<I, O>
        where
            [I; 3]: Patch,
            O::HalfArray: Index<usize, Output = [[I; 3]; 3]>,
        {
            fn apply(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size]; $x_size] {
                let threshold: u16 = From::from(9 * I::BIT_LEN as u16 / 2);
                let mut target = [[O::default(); $y_size]; $x_size];
                for y in 0..$y_size - 2 {
                    for b in 0..O::BIT_LEN / 2 {
                        let mut buf = [[0u16; 3]; $x_size];

                        let strip = [input[0][y + 0], input[0][y + 1], input[0][y + 2]];
                        buf[0][0] = self.kernel[b][0].hamming_distance(&strip) as u16;
                        buf[0][2] = self.kernel[b][2].hamming_distance(&strip) as u16;

                        for x in 1..$x_size - 1 {
                            let strip = [input[x][y + 0], input[x][y + 1], input[x][y + 2]];
                            buf[x] = [
                                self.kernel[b][0].hamming_distance(&strip) as u16,
                                self.kernel[b][1].hamming_distance(&strip) as u16,
                                self.kernel[b][2].hamming_distance(&strip) as u16,
                            ];
                        }
                        let strip = [
                            input[$x_size - 1][y + 0],
                            input[$x_size - 1][y + 1],
                            input[$x_size - 1][y + 2],
                        ];
                        buf[$x_size - 1][0] = self.kernel[b][0].hamming_distance(&strip) as u16;
                        buf[$x_size - 1][2] = self.kernel[b][2].hamming_distance(&strip) as u16;

                        for x in 0..$x_size - 2 {
                            let bit: O = O::from(
                                (buf[x + 0][0] + buf[x + 1][1] + buf[x + 2][2]) > threshold,
                            );
                            target[x + 1][y + 1] |= bit << From::from(b);

                            target[x + 1][y + 1] |= O::from(
                                (buf[x + 0][2] + buf[x + 1][1] + buf[x + 2][0]) > threshold,
                            ) << From::from(b + (O::BIT_LEN / 2));
                        }
                    }
                }
                target
            }
        }
    };
}
//impl_conv3x3mirror_apply!(32, 32, u128);
//impl_conv3x3mirror_apply!(32, 32, u64);
impl_conv3x3mirror_apply!(32, 32, u32);

struct Mirror3x3Cache<L, I, S, O> {
    unclean_i_section: usize,
    unclean_b: usize,
    layer: PhantomData<L>,
    class: usize,
    input: I,
    sums: S,
    output: O,
}

macro_rules! impl_mirror3x3cache {
    ($x_size:expr, $y_size:expr, $output_type:ty) => {
        impl<
                I: BitLen + Copy + BitXor + Patch + HammingDist + Default,
                O: HalfLen<[[I; 3]; 3]>
                    + BitOrAssign
                    + BitAndAssign
                    + BitLen
                    + Not<Output = O>
                    + Shl<usize, Output = O>
                    + From<u8>
                    + From<bool>,
            >
            Mirror3x3Cache<
                Conv3x3Mirror<I, O>,
                [[I; $y_size]; $x_size],
                [[[u16; 3]; $x_size]; $y_size - 2],
                [[O; $y_size]; $x_size],
            >
        where
            Conv3x3Mirror<I, O>: Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]>,
            [I; 3]: Patch,
            O::HalfArray: Index<usize, Output = [[I; 3]; 3]>,
        {
            fn clean_all_sums_all_rows(&mut self, layer: &Conv3x3Mirror<I, O>, b: usize) {
                for y in 0..$y_size - 2 {
                    self.clean_all_sums_row(layer, y, b);
                }
            }
            // clean one side of the input.
            fn clean_sums_row(
                &mut self,
                layer: &Conv3x3Mirror<I, O>,
                y: usize,
                b: usize,
                i_seg: usize,
            ) {
                for x in 0..32 {
                    self.sums[y][x][i_seg] = ((layer.kernel[b][i_seg][0].hd(self.input[x][y + 0]))
                        + (layer.kernel[b][i_seg][1].hd(self.input[x][y + 1]))
                        + (layer.kernel[b][i_seg][2].hd(self.input[x][y + 2])))
                        as u16;
                    //self.sums[y][x][i_seg] = layer.kernel[b][i_seg].hamming_distance(&[
                    //    self.input[x][y + 0],
                    //    self.input[x][y + 1],
                    //    self.input[x][y + 2],
                    //]) as u16;
                }
            }
            // clean all of the input. Same as calling 0, 2 and center. but fused and faster.
            fn clean_all_sums_row(&mut self, layer: &Conv3x3Mirror<I, O>, y: usize, b: usize) {
                let strip = [
                    self.input[0][y + 0],
                    self.input[0][y + 1],
                    self.input[0][y + 2],
                ];
                self.sums[y][0][0] = layer.kernel[b][0].hamming_distance(&strip) as u16;
                self.sums[y][0][2] = layer.kernel[b][2].hamming_distance(&strip) as u16;

                for x in 1..32 - 1 {
                    let strip = [
                        self.input[x][y + 0],
                        self.input[x][y + 1],
                        self.input[x][y + 2],
                    ];
                    self.sums[y][x] = [
                        layer.kernel[b][0].hamming_distance(&strip) as u16,
                        layer.kernel[b][1].hamming_distance(&strip) as u16,
                        layer.kernel[b][2].hamming_distance(&strip) as u16,
                    ];
                }
                let strip = [
                    self.input[31][y + 0],
                    self.input[31][y + 1],
                    self.input[31][y + 2],
                ];
                self.sums[y][31][0] = layer.kernel[b][0].hamming_distance(&strip) as u16;
                self.sums[y][31][2] = layer.kernel[b][2].hamming_distance(&strip) as u16;
            }
            fn clean_b_row(&mut self, layer: &Conv3x3Mirror<I, O>, y: usize, b: usize) {
                self.clean_all_sums_row(layer, y, b);
                for x in 0..$x_size - 2 {
                    self.output[x + 1][y + 1] &= O::from(1u8) << From::from(b);
                    self.output[x + 1][y + 1] |= O::from(
                        self.sums[y][x + 0][0] + self.sums[y][x + 1][1] + self.sums[y][x + 2][2]
                            > From::from(9 * I::BIT_LEN as u16 / 2),
                    ) << b;
                    let mask: O = O::from(1u8) << (b + O::BIT_LEN / 2);
                    self.output[x + 1][y + 1] &= !mask;
                    self.output[x + 1][y + 1] |= O::from(
                        self.sums[y][x + 0][2] + self.sums[y][x + 1][1] + self.sums[y][x + 2][0]
                            > From::from(9 * I::BIT_LEN as u16 / 2),
                    ) << (b + O::BIT_LEN / 2);
                }
            }
        }
    };
}

//impl_mirror3x3cache!(32, 32, u128);
//impl_mirror3x3cache!(32, 32, u64);
impl_mirror3x3cache!(32, 32, u32);

macro_rules! impl_traincache_for_mirror3x3cache {
    ($x_size:expr, $y_size:expr, $output_type:ty) => {
        impl<
                I: BitLen + Copy + Default + BitXor + Patch + HammingDist,
                O: HalfLen<[[I; 3]; 3]>
                    + BitLen
                    + Sync
                    + Copy
                    + HammingDist
                    + BitAndAssign
                    + BitOrAssign
                    + Not<Output = O>
                    + Shl<usize, Output = O>
                    + Default
                    + Add
                    + From<u8>
                    + From<bool>,
            >
            TrainCache<
                [[I; $y_size]; $x_size],
                Conv3x3Mirror<I, O>,
                [[O; $y_size]; $x_size],
                [O; 10],
            >
            for Mirror3x3Cache<
                Conv3x3Mirror<I, O>,
                [[I; $y_size]; $x_size],
                [[[u16; 3]; $x_size]; $y_size - 2],
                [[O; $y_size]; $x_size],
            >
        where
            Conv3x3Mirror<I, O>: Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]>,
            [I; 3]: Patch,
            O::HalfArray: Mutate + Index<usize, Output = [[I; 3]; 3]>,
        {
            fn new(
                layer: &Conv3x3Mirror<I, O>,
                class: usize,
                input: &[[I; $y_size]; $x_size],
            ) -> Self {
                let mut cache = Mirror3x3Cache {
                    unclean_i_section: 0,
                    unclean_b: 0,
                    layer: PhantomData,
                    class: class,
                    input: *input,
                    sums: [[[0u16; 3]; $x_size]; $y_size - 2],
                    output: layer.apply(input),
                };
                cache.clean_all_sums_all_rows(layer, 0);
                cache
            }
            fn output(&mut self, layer: &Conv3x3Mirror<I, O>) -> (usize, [[O; $y_size]; $x_size]) {
                for y in 0..$y_size - 2 {
                    let unclean_b = self.unclean_b;
                    let unclean_i_section = self.unclean_i_section;
                    self.clean_sums_row(layer, y, unclean_b, unclean_i_section);
                }
                (self.class, self.output)
            }
            fn obj(
                &mut self,
                layer: &Conv3x3Mirror<I, O>,
                head: &[O; 10],
                i: usize,
                b: usize,
            ) -> f64 {
                let mut sum_obj = 0;
                let unclean_b = self.unclean_b;
                let unclean_i_seg = self.unclean_i_section;
                let i_seg = i / (I::BIT_LEN * 3);
                for y in 0..$y_size - 2 {
                    // if the unclean output is not the output we are going to recompute,
                    // we must clean it first.
                    if unclean_b != b {
                        self.clean_b_row(layer, y, unclean_b);
                        // also clean all the sums in the row for the new output which we are to now compute.
                        self.clean_all_sums_row(layer, y, b);
                    // but if the output is not changed, we need only clean some of the cached sums.
                    } else {
                        // if the input is in a new segment, we must clean the old segment.
                        if unclean_i_seg != i_seg {
                            self.clean_sums_row(layer, y, b, unclean_i_seg);
                        }
                        // but we need to clean the current segment of the sums anyway.
                        self.clean_sums_row(layer, y, b, i_seg);
                    }
                    for x in 0..$x_size - 2 {
                        self.output[x + 1][y + 1] &= !(O::from(1u8) << b);
                        self.output[x + 1][y + 1] |= O::from(
                            self.sums[y][x + 0][0]
                                + self.sums[y][x + 1][1]
                                + self.sums[y][x + 2][2]
                                > (9 * I::BIT_LEN as u16 / 2),
                        ) << b;
                        self.output[x + 1][y + 1] &= !(O::from(1u8) << (b + O::BIT_LEN / 2));
                        self.output[x + 1][y + 1] |= O::from(
                            self.sums[y][x + 0][2]
                                + self.sums[y][x + 1][1]
                                + self.sums[y][x + 2][0]
                                > (9 * I::BIT_LEN as u16 / 2),
                        ) << (b + O::BIT_LEN / 2);

                        sum_obj += head.is_correct(self.class, self.output[x + 1][y + 1]) as u64;
                    }
                }
                self.unclean_b = b;
                self.unclean_i_section = i / (I::BIT_LEN * 3);
                sum_obj as f64 / (($x_size - 2) * ($y_size - 2)) as f64
            }
        }
    };
}

//impl_traincache_for_mirror3x3cache!(32, 32, u128);
//impl_traincache_for_mirror3x3cache!(32, 32, u64);
impl_traincache_for_mirror3x3cache!(32, 32, u32);

//fn update(
//    weights: &Conv3x3Mirror<u32, [[[u32; 3]; 3]; 64], u128>,
//    input: &[[u32; 32]; 32],
//    target: &mut [[u128; 32]; 32],
//    b: usize,
//) {
//    for y in 0..32 - 2 {
//        let mut buf = [[0u16; 3]; 32];
//        let strip = [input[0][y + 0], input[0][y + 1], input[0][y + 2]];
//        buf[0][0] = weights.kernel[b][0].hamming_distance(&strip) as u16;
//        buf[0][2] = weights.kernel[b][2].hamming_distance(&strip) as u16;
//
//        for x in 1..32 - 1 {
//            let strip = [input[x][y + 0], input[x][y + 1], input[x][y + 2]];
//            buf[x] = [
//                weights.kernel[b][0].hamming_distance(&strip) as u16,
//                weights.kernel[b][1].hamming_distance(&strip) as u16,
//                weights.kernel[b][2].hamming_distance(&strip) as u16,
//            ];
//        }
//        let strip = [input[31][y + 0], input[31][y + 1], input[31][y + 2]];
//        buf[31][0] = weights.kernel[b][0].hamming_distance(&strip) as u16;
//        buf[31][2] = weights.kernel[b][2].hamming_distance(&strip) as u16;
//
//        for x in 0..32 - 2 {
//            target[x + 1][y + 1] |=
//                ((buf[x + 0][0] + buf[x + 1][1] + buf[x + 2][2] > (9 * 16)) as u128) << b;
//
//            target[x + 1][y + 1] |=
//                ((buf[x + 0][2] + buf[x + 1][1] + buf[x + 2][0] > (9 * 16)) as u128) << (b + 64);
//        }
//    }
//}

fn load_data() -> Vec<(usize, [[u64; 32]; 32])> {
    let size: usize = 10_000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images: Vec<(usize, [[[u8; 3]; 32]; 32])> = Vec::with_capacity(size * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, size);
        images.append(&mut batch);
    }
    images
        .iter()
        .map(|(class, image)| (*class, image.pixel_map(&|input| unary::rgb_to_u64(*input))))
        .collect()
}

fn flip_image<P: Copy + Default>(input: &[[P; 32]; 32]) -> [[P; 32]; 32] {
    let mut target = [<[P; 32]>::default(); 32];
    for i in 0..32 {
        target[((i as i64 * -1) + 31) as usize] = input[i];
    }
    target
}

//impl<I: Default + Copy, O: BitLen + HalfLen<[[I; 3]; 3]>> NewFromRng for Conv3x3Mirror<I, O>
//where
//    rand::distributions::Standard: rand::distributions::Distribution<[[I; 3]; 3]>,
//    O::HalfArray: Default + IndexMut<usize, Output = [[I; 3]; 3]>,
//{
//    fn new_from_rng<RNG: rand::Rng>(rng: &mut RNG) -> Self {
//        let mut kernel = O::HalfArray::default();
//        for i in 0..O::BIT_LEN / 2 {
//            kernel[i] = rng.gen();
//        }
//        Conv3x3Mirror {
//            input_pixel_type: PhantomData,
//            kernel: kernel,
//            output_type: PhantomData,
//        }
//    }
//}

impl<I: Default + Copy> NewFromRng for Conv3x3Mirror<I, u32>
where
    rand::distributions::Standard: rand::distributions::Distribution<[[I; 3]; 3]>,
    //<u32 as HalfLen<[[I; 3]; 3]>>::HalfArray: Default + IndexMut<usize, Output = [[I; 3]; 3]>,
{
    fn new_from_rng<RNG: rand::Rng>(rng: &mut RNG) -> Self {
        let mut kernel = <u32 as HalfLen<[[I; 3]; 3]>>::HalfArray::default();
        for i in 0..u32::BIT_LEN / 2 {
            kernel[i] = rng.gen();
        }
        Conv3x3Mirror {
            input_pixel_type: PhantomData,
            kernel: kernel,
            output_type: PhantomData,
        }
    }
}

trait Train<I, O, H: Objective<O>>: Mutate + Apply<I, O> + Sized {
    fn train_new_layer<RNG: rand::Rng, C: TrainCache<I, Self, O, H> + Objective<O> + Sync + Send>(
        examples: &Vec<(usize, I)>,
        fs_path: &Path,
        rng: &mut RNG,
        head_update_freq: usize,
        depth: usize,
    ) -> Vec<(usize, O)>;
    fn train(
        layer: &mut Self,
        head: &mut H,
        examples_batch: &[(usize, I)],
        head_update_freq: usize,
        depth: usize,
    );
}

impl<
        I: Sync + Copy + Patch,
        O: Sync + Send,
        P: Copy + Default + Sync,
        L: SaveLoad + NewFromRng + Mutate + Apply<I, O> + Sync + VecApply<I, O>,
    > Train<I, O, [P; 10]> for L
where
    [P; 10]: OptimizeHead<O> + IsCorrect<P> + Objective<O> + Sync + NewFromRng,
{
    fn train_new_layer<
        RNG: rand::Rng,
        C: TrainCache<I, L, O, [P; 10]> + Objective<O> + Sync + Send,
    >(
        examples: &Vec<(usize, I)>,
        fs_path: &Path,
        rng: &mut RNG,
        head_update_freq: usize,
        depth: usize,
    ) -> Vec<(usize, O)> {
        let layer = Self::new_from_fs(fs_path).unwrap_or_else(|| {
            println!("{} not found, training", &fs_path.to_str().unwrap());
            let mut layer: Self = Self::new_from_rng(rng);
            let mut head = <[P; 10]>::new_from_rng(rng);

            let start = PreciseTime::now();
            Self::train(&mut layer, &mut head, examples, head_update_freq, depth);
            let obj = train_layer_with_cache::<I, L, O, [P; 10], C>(
                &examples,
                &mut layer,
                &mut head,
                UPDATE_FREQ,
            );

            let (obj, updates) =
                layer.optimize_layer::<[P; 10]>(&mut head, &examples, head_update_freq);
            println!(
                "obj: {}, depth: {}, {}",
                obj,
                depth,
                start.to(PreciseTime::now())
            );
            //write_to_log_event(start.to(PreciseTime::now()), obj, depth, &fs_path);
            layer.write_to_fs(&fs_path);
            layer
        });
        layer.vec_apply(&examples)
    }
    // run time is ~constant with depth.
    fn train(
        layer: &mut Self,
        head: &mut [P; 10],
        examples: &[(usize, I)],
        head_update_freq: usize,
        depth: usize,
    ) {
        dbg!(depth);
        if depth > 0 {
            Self::train(
                layer,
                head,
                &examples[0..examples.len() / 2],
                head_update_freq,
                depth - 1,
            );
        } else {
            // train half
        }
        dbg!(examples.len());
        // train other half
    }
}

fn train_layer_with_cache<
    I: Sync,
    L: Sync + Mutate + Apply<I, O>,
    O: Sync + Send,
    H: Sync + Objective<O> + OptimizeHead<O>,
    C: Sync + Send + TrainCache<I, L, O, H>,
>(
    examples: &[(usize, I)],
    layer: &mut L,
    head: &mut H,
    head_update_freq: usize,
) -> f64 {
    let mut cache: Vec<C> = examples
        .par_iter()
        .map(|(class, input_image)| C::new(&layer, *class, input_image))
        .collect();

    let obj_sum: f64 = cache
        .par_iter_mut()
        .map(|cache| cache.obj(&layer, &head, 0, 0))
        .sum();
    let mut cur_obj = obj_sum / cache.len() as f64;
    let mut iter = 0;
    for o in 0..L::OUTPUT_LEN {
        for i in 0..L::INPUT_LEN {
            if iter % head_update_freq == 0 {
                let outputs: Vec<(usize, O)> = cache
                    .par_iter_mut()
                    .map(|cache| cache.output(layer))
                    .collect();
                cur_obj = head.optimize_head(&outputs, 0);
                println!("head update: {} {} {}", o, i, cur_obj);
                iter += 1;
            }
            layer.mutate(o, i);
            let obj_sum: f64 = cache
                .par_iter_mut()
                .map(|cache| cache.obj(&layer, &head, i, o))
                .sum();
            let new_obj = obj_sum / cache.len() as f64;
            if new_obj > cur_obj {
                cur_obj = new_obj;
                iter += 1;
                println!("{} {} {}", o, i, cur_obj);
            } else {
                layer.mutate(o, i);
                //dbg!(cur_obj);
            }
        }
    }
    cur_obj
}

type IP = u64;
type OP = u32;

const UPDATE_FREQ: usize = 20;

// head update: 63 569 0.18607844444444444

fn main() {
    // In rayon 1.0.3 we need to tell rayon to give the workers a larger stack.
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(22))
        .build_global()
        .unwrap();

    let mut rng = Hc128Rng::seed_from_u64(42);
    let examples = load_data();

    let mut layer = Conv3x3Mirror::<IP, OP>::new_from_rng(&mut rng);
    let mut head = <[OP; 10]>::new_from_rng(&mut rng);

    //let examples2 = Conv3x3Mirror::<IP, [[[IP; 3]; 3]; OP::BIT_LEN / 2], OP>::train_new_layer::<
    //    Hc128Rng,
    //    Mirror3x3Cache<
    //        Conv3x3Mirror<IP, [[[IP; 3]; 3]; OP::BIT_LEN / 2], OP>,
    //        [[IP; 32]; 32],
    //        [[[u16; 3]; 32]; 30],
    //        [[OP; 32]; 32],
    //    >,
    //>(
    //    &examples,
    //    &Path::new("params/mirror_test/l1"),
    //    &mut rng,
    //    UPDATE_FREQ,
    //    5,
    //);

    let prms = [[[0u64; 3]; 3]; 16];
    let foo = prms[5usize];

    let start = PreciseTime::now();
    for &(start, end) in [
        (0usize, 1000usize),
        (1000, 2000),
        (2000, 4000),
        (4000, 8000),
        (8000, 16000),
        (16000, 25000),
        (25000, 50000),
        (0, 50000),
    ]
    .iter()
    {
        println!("examples[{}..{}]", start, end);
        let obj = train_layer_with_cache::<
            [[IP; 32]; 32],
            Conv3x3Mirror<IP, OP>,
            [[OP; 32]; 32],
            [OP; 10],
            Mirror3x3Cache<
                Conv3x3Mirror<IP, OP>,
                [[IP; 32]; 32],
                [[[u16; 3]; 32]; 30],
                [[OP; 32]; 32],
            >,
        >(&examples[start..end], &mut layer, &mut head, UPDATE_FREQ);
    }
    println!("cached train duration: {:?}", start.to(PreciseTime::now()));

    //let obj = layer.optimize_layer(&mut head, &examples, UPDATE_FREQ);

    //let l1 = Conv3x3Mirror::<IP, [[[IP; 3]; 3]; 64], u128>::train_new_layer(&mut rng);

    dbg!("creating cache");
    //let mut cache_vec: Vec<_> = examples
    //    .par_iter()
    //    .map(|(class, input_image)| {
    //        Mirror3x3Cache::<
    //            Conv3x3Mirror<IP, OP>,
    //            [[IP; 32]; 32],
    //            [[[u16; 3]; 32]; 30],
    //            [[OP; 32]; 32],
    //        >::new(&layer, *class, input_image)
    //    })
    //    .collect();

    //let input_index: usize = 1;
    //let output_index: usize = 0;
    //dbg!("starting cached update");
    //let start = PreciseTime::now();
    //let cache_objs: Vec<f64> = cache_vec
    //    .par_iter_mut()
    //    .map(|cache| cache.obj(&layer, &head, input_index, output_index))
    //    .collect();
    //println!(
    //    "cached obj duration: {:?}",
    //    start.to(PreciseTime::now()).num_milliseconds()
    //);

    //let start = PreciseTime::now();
    //let apply_objs: Vec<f64> = examples
    //    .par_iter()
    //    .map(|(class, input_image)| head.objective(&(*class, simple_apply(&layer, input_image))))
    //    //.map(|(class, input_image)| head.objective(&(*class, layer.apply(input_image))))Conv3x3Mirror<IP, [[[IP; 3]; 3]; 64], u128>
    //    .collect();
    //println!(
    //    "apply duration: {:?}",
    //    start.to(PreciseTime::now()).num_milliseconds()
    //);
    //let cache_sum: f64 = cache_objs.iter().sum();
    //let apply_sum: f64 = apply_objs.iter().sum();
    //dbg!(cache_sum / cache_objs.len() as f64);
    //dbg!(apply_sum / apply_objs.len() as f64);
    //assert_eq!(cache_objs, apply_objs);

    //for example in examples.iter().take(100) {
    //    Mirror3x3Cache::<
    //        Conv3x3Mirror<IP, [[[IP; 3]; 3]; OP::BIT_LEN / 2], OP>,
    //        [[IP; 32]; 32],
    //        [[[u16; 3]; 32]; 30],
    //        [[OP; 32]; 32],
    //    >::self_test(&mut layer, example.0, &head, &example.1);
    //}
    ////apply duration: 11747
    ////update duration: 193

    //let start = PreciseTime::now();
    //let _: Vec<_> = opt_outputs
    //    .iter_mut()
    //    .zip(examples.iter())
    //    .map(|(target, input)| update(&weights, &input.1, target, 17))
    //    .collect();
    //println!(
    //    "update duration: {:?}",
    //    start.to(PreciseTime::now()).num_milliseconds()
    //);

    //let start = PreciseTime::now();
    //let simple_outputs: Vec<_> = examples
    //    .iter()
    //    .map(|(_, input_image)| simple_apply(&weights, input_image))
    //    .collect();
    //println!("simple apply duration: {:?}", start.to(PreciseTime::now()));
    //let _: Vec<_> = simple_outputs
    //    .iter()
    //    .zip(opt_outputs.iter())
    //    .map(|(simple, opt)| assert_eq!(simple, opt))
    //    .collect();
}
