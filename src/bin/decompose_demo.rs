extern crate bincode;
extern crate bitnn;
extern crate image;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate serde_derive;
extern crate time;
use bincode::{deserialize_from, serialize_into};
use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::vec_concat_2_examples;
use bitnn::layers::{
    Apply, Extract3x3Patches, IsCorrect, Mutate, NewFromRng, OptimizeHead, PixelMap, SaveLoad,
};
use bitnn::{BitLen, HammingDist, Patch};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::convert::From;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufWriter;
use std::iter;
use std::marker::PhantomData;
use std::path::Path;
use time::PreciseTime;

pub struct Conv3x3Mirror<I, K, O> {
    input_pixel_type: PhantomData<I>,
    kernel: K,
    output_type: PhantomData<O>,
}

impl<I: BitLen, K: Mutate, O: BitLen> Mutate for Conv3x3Mirror<I, K, O> {
    fn mutate(&mut self, b: usize, i: usize) {
        self.kernel.mutate(b, i);
    }
    const OUTPUT_LEN: usize = O::BIT_LEN / 2;
    const INPUT_LEN: usize = I::BIT_LEN * 3 * 3;
}

macro_rules! impl_conv3x3mirror_patch_apply {
    ($output_type:ty) => {
        impl<I: Copy + Patch + BitLen + Default>
            Apply<[[I; 3]; 3], $output_type>
            for Conv3x3Mirror<I, [[[I; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>
        where
            [[I; 3]; 3]: Patch,
        {
            fn apply(&self, input: &[[I; 3]; 3]) -> $output_type {
                let threshold = 9 * I::BIT_LEN as u32 / 2;
                let mut target = <$output_type>::default();
                for b in 0..<$output_type>::BIT_LEN / 2 {
                    target |=
                        (self.kernel[b].hamming_distance(input) > threshold) as $output_type << b;

                    let mirror_filter = [self.kernel[b][2], self.kernel[b][1], self.kernel[b][0]];
                    target |=
                    (mirror_filter.hamming_distance(input) > threshold) as $output_type
                        << (b + (<$output_type>::BIT_LEN / 2));
                    }
                    target
            }
        }
    };
}

impl_conv3x3mirror_patch_apply!(u16);
impl_conv3x3mirror_patch_apply!(u32);
impl_conv3x3mirror_patch_apply!(u64);

macro_rules! impl_conv3x3mirror_apply {
    ($x_size:expr, $y_size:expr, $output_type:ty) => {
        impl<I: Copy + Patch + BitLen + Default>
            Apply<[[I; $y_size]; $x_size], [[$output_type; $y_size]; $x_size]>
            for Conv3x3Mirror<I, [[[I; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>
        where
            [I; 3]: Patch,
        {
            fn apply(
                &self,
                input: &[[I; $y_size]; $x_size],
            ) -> [[$output_type; $y_size]; $x_size] {
                let threshold: u16 = From::from(9 * I::BIT_LEN as u16 / 2);
                let mut target = [[<$output_type>::default(); $y_size]; $x_size];
                for y in 0..$y_size - 2 {
                    for b in 0..<$output_type>::BIT_LEN / 2 {
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
                            target[x + 1][y + 1] |=
                                ((buf[x + 0][0] + buf[x + 1][1] + buf[x + 2][2]) > threshold) as $output_type << b;

                            target[x + 1][y + 1] |=
                                ((buf[x + 0][2] + buf[x + 1][1] + buf[x + 2][0]) > threshold) as $output_type
                                    << (b + (<$output_type>::BIT_LEN / 2));
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

struct Mirror3x3PatchCache<P, L, O> {
    layer: PhantomData<L>,
    class: u8,
    patch: [[P; 3]; 3],
    // flipped center is left empty
    pixel_sums: [[[u16; 3]; 3]; 2],
    strip_sums: [[u16; 3]; 2],
    output: O,
}

macro_rules! impl_mirror3x3patchcache_clean {
    ($output_type:ty) => {
        impl<P: HammingDist + BitLen + Copy> Mirror3x3PatchCache<P, Conv3x3Mirror<P, [[[P; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>, $output_type>
        where
            Conv3x3Mirror<P, [[[P; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>: Apply<[[P; 3]; 3], $output_type>,
        {
            fn new(layer: &Conv3x3Mirror<P, [[[P; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>, class: u8, patch: &[[P; 3]; 3]) -> Self {
                let pixel_sums = [[[0u16; 3]; 3]; 2];
                let strip_sums = [[0u16; 3]; 2];
                let mut cache = Mirror3x3PatchCache{
                    layer: PhantomData,
                    class: class,
                    patch: *patch,
                    pixel_sums: pixel_sums,
                    strip_sums: strip_sums,
                    output: layer.apply(patch),
                };
                cache.switch_output_bit(layer, 0);
                cache
            }
            #[inline(always)]
            fn clean_output(&mut self, b: usize) {
                let threshold: u16 = 9 * P::BIT_LEN as u16 / 2;
                self.output &= !(1 << b);
                self.output |= (self.strip_sums[0][0] + self.strip_sums[0][1] + self.strip_sums[0][2]
                    > threshold) as $output_type
                    << b;
                self.output &= !(1 << (b + <$output_type>::BIT_LEN / 2));
                self.output |=
                    (self.strip_sums[1][0] + self.strip_sums[0][1] + self.strip_sums[1][2]
                        > threshold) as $output_type
                        << (b + <$output_type>::BIT_LEN / 2);
            }
            #[inline(always)]
            fn clean_one_pixel_sum(
                &mut self,
                layer: &Conv3x3Mirror<P, [[[P; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>,
                x: usize,
                y: usize,
                b: usize,
            ) {
                if x ==0 {
                    self.pixel_sums[0][0][y] = self.patch[0][y].hd(layer.kernel[b][0][y]) as u16;
                    self.strip_sums[0][0] = self.pixel_sums[0][0][0] + self.pixel_sums[0][0][1] + self.pixel_sums[0][0][2];

                    self.pixel_sums[1][2][y] = self.patch[2][y].hd(layer.kernel[b][0][y]) as u16;
                    self.strip_sums[1][2] = self.pixel_sums[1][2][0] + self.pixel_sums[1][2][1] + self.pixel_sums[1][2][2];
                } else if x ==2 {
                    self.pixel_sums[0][2][y] = self.patch[2][y].hd(layer.kernel[b][2][y]) as u16;
                    self.strip_sums[0][2] = self.pixel_sums[0][2][0] + self.pixel_sums[0][2][1] + self.pixel_sums[0][2][2];

                    self.pixel_sums[1][0][y] = self.patch[0][y].hd(layer.kernel[b][2][y]) as u16;
                    self.strip_sums[1][0] = self.pixel_sums[1][0][0] + self.pixel_sums[1][0][1] + self.pixel_sums[1][0][2];
                } else {
                    self.pixel_sums[0][1][y] = self.patch[1][y].hd(layer.kernel[b][1][y]) as u16;
                    self.strip_sums[0][1] = self.pixel_sums[0][1][0] + self.pixel_sums[0][1][1] + self.pixel_sums[0][1][2];
                }
                self.clean_output(b);
            }
            #[inline(always)]
            fn switch_output_bit(&mut self, layer: &Conv3x3Mirror<P, [[[P; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>, b: usize) {
                for y in 0..3 {
                    // normal way
                    self.pixel_sums[0][0][y] = self.patch[0][y].hd(layer.kernel[b][0][y]) as u16;
                    self.pixel_sums[0][2][y] = self.patch[2][y].hd(layer.kernel[b][2][y]) as u16;

                    // flipped
                    self.pixel_sums[1][0][y] = self.patch[0][y].hd(layer.kernel[b][2][y]) as u16;
                    self.pixel_sums[1][2][y] = self.patch[2][y].hd(layer.kernel[b][0][y]) as u16;

                    // center
                    self.pixel_sums[0][1][y] = self.patch[1][y].hd(layer.kernel[b][1][y]) as u16;
                }
                // normal way
                self.strip_sums[0][0] = self.pixel_sums[0][0][0] + self.pixel_sums[0][0][1] + self.pixel_sums[0][0][2];
                self.strip_sums[0][2] = self.pixel_sums[0][2][0] + self.pixel_sums[0][2][1] + self.pixel_sums[0][2][2];

                // flipped
                self.strip_sums[1][0] = self.pixel_sums[1][0][2] + self.pixel_sums[1][0][1] + self.pixel_sums[1][0][0];
                self.strip_sums[1][2] = self.pixel_sums[1][2][2] + self.pixel_sums[1][2][1] + self.pixel_sums[1][2][0];

                // center
                self.strip_sums[0][1] = self.pixel_sums[0][1][0] + self.pixel_sums[0][1][1] + self.pixel_sums[0][1][2];

                self.clean_output(b);
            }
        }
    };
}

//impl_mirror3x3patchcache_clean!(u16);
impl_mirror3x3patchcache_clean!(u32);
//impl_mirror3x3patchcache_clean!(u64);

trait TrainLayerWithCache<I: Sync, O> {
    fn train_layer_with_cache(
        &mut self,
        patches: &[(u8, [[I; 3]; 3])],
        head: &mut [O; 10],
        head_update_freq: usize,
    ) -> f64;
}

macro_rules! impl_trainlayerwithcache_for_conv3x3mirror {
    ($output_type:ty) => {
        impl<I: Patch + Sync + Send + Copy + HammingDist + BitLen + Default>
            TrainLayerWithCache<I, $output_type>
            for Conv3x3Mirror<I, [[[I; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>
        where
            Self: Apply<[[I; 3]; 3], $output_type>,
        {
            fn train_layer_with_cache(
                &mut self,
                patches: &[(u8, [[I; 3]; 3])],
                head: &mut [$output_type; 10],
                head_update_freq: usize,
            ) -> f64 {
                println!("constructing cache with {} patches", patches.len());
                let mut caches: Vec<_> = patches
                    .par_iter()
                    .map(|(class, patch)| {
                        Mirror3x3PatchCache::<I, Self, $output_type>::new(self, *class, patch)
                    })
                    .collect();
                let mut cur_n_isgood: u64 = caches
                    .par_iter()
                    .map(|cache| head.is_correct(cache.class, cache.output) as u64)
                    .sum();

                let mut iters = 0;
                let mut cur_dirty_x = 0;
                let mut cur_dirty_y = 0;
                for b in 0..<$output_type>::BIT_LEN / 2 {
                    dbg!(b);
                    let new_cur_n_isgood: u64 = caches
                        .par_iter_mut()
                        .map(|cache| {
                            cache.switch_output_bit(self, b);
                            head.is_correct(cache.class, cache.output) as u64
                        })
                        .sum();
                    assert_eq!(new_cur_n_isgood, cur_n_isgood);

                    for x in 0..3 {
                        for y in 0..3 {
                            for i in 0..I::BIT_LEN {
                                if iters % head_update_freq == 0 {
                                    let outputs: Vec<(u8, $output_type)> = caches
                                        .par_iter_mut()
                                        .map(|cache| {
                                            cache.clean_one_pixel_sum(
                                                self,
                                                cur_dirty_x,
                                                cur_dirty_y,
                                                b,
                                            );
                                            (cache.class, cache.output)
                                        })
                                        .collect();
                                    cur_n_isgood = head.optimize_head(&outputs);
                                    iters += 1;
                                }
                                //dbg!("mutating");
                                self.kernel[b][x][y].flip_bit(i);
                                let new_n_isgood: u64 = caches
                                    .par_iter_mut()
                                    .map(|cache| {
                                        cache.clean_one_pixel_sum(self, x, y, b);
                                        head.is_correct(cache.class, cache.output) as u64
                                    })
                                    .sum();
                                if new_n_isgood > cur_n_isgood {
                                    cur_n_isgood = new_n_isgood;
                                    iters += 1;
                                    println!("{} {} {} {}: {}", b, x, y, i, cur_n_isgood);
                                } else {
                                    //dbg!(new_n_isgood);
                                    cur_dirty_x = x;
                                    cur_dirty_y = y;
                                    // revert
                                    self.kernel[b][x][y].flip_bit(i);
                                }
                            }
                            let cleanup_n_isgood: u64 = caches
                                .par_iter_mut()
                                .map(|cache| {
                                    cache.clean_one_pixel_sum(self, x, y, b);
                                    head.is_correct(cache.class, cache.output) as u64
                                })
                                .sum();
                            assert_eq!(cleanup_n_isgood, cur_n_isgood);
                        }
                    }

                    println!("{} {}%", b, cur_n_isgood as f64 / caches.len() as f64);
                }

                cur_n_isgood as f64 / caches.len() as f64
            }
        }
    };
}

impl_trainlayerwithcache_for_conv3x3mirror!(u32);

fn load_data() -> Vec<(usize, [[u32; 32]; 32])> {
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
        .map(|(class, image)| (*class, image.pixel_map(&|input| unary::rgb_to_u32(*input))))
        .collect()
}

macro_rules! impl_newfromrand_for_conv3x3mirror {
    ($output_type:ty) => {
        impl<I: Default + Copy> NewFromRng
            for Conv3x3Mirror<I, [[[I; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>
        where
            rand::distributions::Standard: rand::distributions::Distribution<[[I; 3]; 3]>,
        {
            fn new_from_rng<RNG: rand::Rng>(rng: &mut RNG) -> Self {
                let mut kernel = [[[I::default(); 3]; 3]; <$output_type>::BIT_LEN / 2];
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
    };
}

impl_newfromrand_for_conv3x3mirror!(u32);
impl_newfromrand_for_conv3x3mirror!(u64);

macro_rules! impl_saveload_conv3x3mirror {
    ($output_type:ty) => {
        impl<I: Default + Copy, O> SaveLoad
            for Conv3x3Mirror<I, [[[I; 3]; 3]; <$output_type>::BIT_LEN / 2], O>
        where
            I: serde::Serialize,
            for<'de> I: serde::Deserialize<'de>,
        {
            fn write_to_fs(&self, path: &Path) {
                let vec_params: Vec<[[I; 3]; 3]> = self.kernel.iter().map(|x| *x).collect();
                let mut f = BufWriter::new(File::create(path).unwrap());
                serialize_into(&mut f, &vec_params).unwrap();
            }
            // This will return:
            // - Some if the file exists and is good
            // - None of the file does not exist
            // and will panic if the file is exists but is bad.
            fn new_from_fs(path: &Path) -> Option<Self> {
                let len = <$output_type>::BIT_LEN / 2;
                File::open(&path)
                    .map(|f| deserialize_from(f).unwrap())
                    .map(|vec_params: Vec<[[I; 3]; 3]>| {
                        if vec_params.len() != len {
                            panic!("input is of len {} not {}", vec_params.len(), len);
                        }
                        let mut kernel = [<[[I; 3]; 3]>::default(); <$output_type>::BIT_LEN / 2];
                        for i in 0..len {
                            kernel[i] = vec_params[i];
                        }
                        Conv3x3Mirror {
                            input_pixel_type: PhantomData,
                            kernel: kernel,
                            output_type: PhantomData,
                        }
                    })
                    .ok()
            }
        }
    };
}
impl_saveload_conv3x3mirror!(u8);
impl_saveload_conv3x3mirror!(u16);
impl_saveload_conv3x3mirror!(u32);

trait Train<II, I, OI, O>: Sized {
    fn train_new_layer<RNG: rand::Rng>(
        examples: &Vec<(usize, II)>,
        fs_path: &Path,
        rng: &mut RNG,
        head_update_freq: usize,
        depth: usize,
    ) -> Vec<(usize, OI)>;
    fn train(
        layer: &mut Self,
        head: &mut [O; 10],
        examples_batch: &[(u8, [[I; 3]; 3])],
        head_update_freq: usize,
        depth: usize,
    ) -> f64;
}

impl<
        II: Extract3x3Patches<I> + Sync,
        I: Sync + Copy + Patch,
        OI: Sync + Send,
        O: Sync + Send,
        L: SaveLoad
            + NewFromRng
            + Apply<[[I; 3]; 3], O>
            + Sync
            + TrainLayerWithCache<I, O>
            + Apply<II, OI>,
    > Train<II, I, OI, O> for L
where
    [O; 10]: OptimizeHead<O> + IsCorrect<O> + Sync + NewFromRng,
{
    fn train_new_layer<RNG: rand::Rng>(
        examples: &Vec<(usize, II)>,
        fs_path: &Path,
        rng: &mut RNG,
        head_update_freq: usize,
        depth: usize,
    ) -> Vec<(usize, OI)> {
        // if we can't load the layer from disk, train it.
        let layer = Self::new_from_fs(fs_path).unwrap_or_else(|| {
            println!("{} not found, training", &fs_path.to_str().unwrap());
            let patches = {
                // decompose the images into patches,
                let mut patches: Vec<(u8, [[I; 3]; 3])> = examples
                    .iter()
                    .map(|(class, image)| iter::repeat(*class as u8).zip(image.patches()))
                    .flatten()
                    .collect();
                // and shuffle them
                patches.shuffle(rng);
                //rng.shuffle(&mut patches);
                patches
            };
            // construct the layer and the head.
            let mut layer: Self = Self::new_from_rng(rng);
            let mut head = <[O; 10]>::new_from_rng(rng);

            let start = PreciseTime::now();
            // now train. This will recursively train with halves.
            let obj = <Self as Train<II, I, OI, O>>::train(
                &mut layer,
                &mut head,
                &patches,
                head_update_freq,
                depth,
            );
            // uncomment to take twice as long and be produce only a slightly better results.
            //println!("begining final full pass with {} examples", patches.len());
            //let obj = layer.train_layer_with_cache(examples, &mut head, head_update_freq);

            println!(
                "obj: {}, depth: {}, {}",
                obj,
                depth,
                start.to(PreciseTime::now())
            );
            // now log and write the layer to disk.
            write_to_log_event(start.to(PreciseTime::now()), obj, depth, &fs_path);
            layer.write_to_fs(&fs_path);
            layer
        });
        // now that we have the parameters, we can use it to generate the next state.
        examples
            .par_iter()
            .map(|(class, input_image)| (*class, layer.apply(input_image)))
            .collect()
    }
    // run time is ~constant with depth.
    fn train(
        layer: &mut Self,
        head: &mut [O; 10],
        examples: &[(u8, [[I; 3]; 3])],
        head_update_freq: usize,
        depth: usize,
    ) -> f64 {
        dbg!(depth);
        // train with the first half.
        if depth > 0 {
            <Self as Train<II, I, OI, O>>::train(
                layer,
                head,
                &examples[0..examples.len() / 2],
                head_update_freq,
                depth - 1,
            );
        } else {
            layer.train_layer_with_cache(&examples[0..examples.len() / 2], head, head_update_freq);
        }
        dbg!(examples.len());
        // train other half
        let obj =
            layer.train_layer_with_cache(&examples[examples.len() / 2..], head, head_update_freq);
        println!("depth: {}", depth);
        obj
    }
}

fn write_to_log_event(duration: time::Duration, obj: f64, depth: usize, layer_name: &Path) {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("train_log_mirror.txt")
        .unwrap();
    writeln!(
        file,
        "{}, {}, obj: {}, {}",
        layer_name.to_str().unwrap(),
        depth,
        obj,
        duration
    )
    .unwrap();
}

type MC3_32_32 = Conv3x3Mirror<u32, [[[u32; 3]; 3]; 16], u32>;
//type MC3_64_32 = Conv3x3Mirror<u64, [[[u64; 3]; 3]; 16], u32>;

const MOD: usize = 20;
const DEPTH: usize = 12;

fn main() {
    //let base_path = Path::new("params/mirror_conv_decompose_1skip");
    let base_path = Path::new("params/mirror_conv_decompose_0skip");
    fs::create_dir_all(base_path).unwrap();
    // We need to tell rayon to give the workers a larger stack.
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(22))
        .build_global()
        .unwrap();

    let mut rng = Hc128Rng::seed_from_u64(42);
    let mut s0: Vec<(usize, [[u32; 32]; 32])> = load_data();
    //let mut s1 =
    //    MC3_32_32::train_new_layer(&s0, &base_path.join("l0_c3m_32-32"), &mut rng, MOD, DEPTH);

    let start = PreciseTime::now();
    for i in 0..100 {
        //let cc: Vec<(usize, [[u64; 32]; 32])> = vec_concat_2_examples(&s1, &s0);
        //s0 = s1;
        s0 = MC3_32_32::train_new_layer(
            &s0,
            &base_path.join(format!("l{}_c3m_32-32", i)),
            &mut rng,
            MOD,
            DEPTH,
        );
    }
    println!("cached train duration: {:?}", start.to(PreciseTime::now()));
}
