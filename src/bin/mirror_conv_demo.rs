extern crate bincode;
extern crate bitnn;
extern crate image;
extern crate rand;
extern crate rayon;
extern crate time;
use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::{Apply, Image2D, IsCorrect, PixelMap};
use bitnn::{BitLen, Patch};
use rand::prng::Hc128Rng;
use rand::SeedableRng;
use std::marker::PhantomData;
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

pub struct Conv3x3Mirror<I, K, O> {
    input_pixel_type: PhantomData<I>,
    kernel: K,
    output_type: PhantomData<O>,
}

fn simple_apply(
    weights: &Conv3x3Mirror<u32, [[[u32; 3]; 3]; 64], u128>,
    input: &[[u32; 32]; 32],
) -> [[u128; 32]; 32] {
    let mut flip_kern = [[[0u32; 3]; 3]; 64];
    for b in 0..64 {
        flip_kern[b] = [
            weights.kernel[b][2],
            weights.kernel[b][1],
            weights.kernel[b][0],
        ];
    }

    let mut target = [[0u128; 32]; 32];
    for x in 0..32 - 2 {
        for y in 0..32 - 2 {
            let patch = [
                [
                    input[x + 0][y + 0],
                    input[x + 0][y + 1],
                    input[x + 0][y + 2],
                ],
                [
                    input[x + 1][y + 0],
                    input[x + 1][y + 1],
                    input[x + 1][y + 2],
                ],
                [
                    input[x + 2][y + 0],
                    input[x + 2][y + 1],
                    input[x + 2][y + 2],
                ],
            ];
            for i in 0..64 {
                target[x + 1][y + 1] |=
                    (weights.kernel[i].distance_and_threshold(&patch) as u128) << i;
                target[x + 1][y + 1] |=
                    (flip_kern[i].distance_and_threshold(&patch) as u128) << (i + 64);
            }
        }
    }
    target
}

macro_rules! impl_conv3x3mirror_apply {
    ($x_size:expr, $y_size:expr, $output_type:ty) => {
        impl<I: Copy + Patch + BitLen>
            Apply<[[I; $y_size]; $x_size], [[$output_type; $y_size]; $x_size]>
            for Conv3x3Mirror<I, [[[I; 3]; 3]; (<$output_type>::BIT_LEN / 2)], $output_type>
        where
            [I; 3]: Patch,
        {
            fn apply(&self, input: &[[I; $y_size]; $x_size]) -> [[u128; $y_size]; $x_size] {
                let mut target = [[0u128; $y_size]; $x_size];
                for y in 0..$y_size - 2 {
                    for b in 0..64 {
                        let mut buf = [[0u32; 3]; $x_size];

                        let strip = [input[0][y + 0], input[0][y + 1], input[0][y + 2]];
                        buf[0][0] = self.kernel[b][0].hamming_distance(&strip);
                        buf[0][2] = self.kernel[b][2].hamming_distance(&strip);

                        for x in 1..$x_size - 1 {
                            let strip = [input[x][y + 0], input[x][y + 1], input[x][y + 2]];
                            buf[x] = [
                                self.kernel[b][0].hamming_distance(&strip),
                                self.kernel[b][1].hamming_distance(&strip),
                                self.kernel[b][2].hamming_distance(&strip),
                            ];
                        }
                        let strip = [
                            input[$x_size - 1][y + 0],
                            input[$x_size - 1][y + 1],
                            input[$x_size - 1][y + 2],
                        ];
                        buf[$x_size - 1][0] = self.kernel[b][0].hamming_distance(&strip);
                        buf[$x_size - 1][2] = self.kernel[b][2].hamming_distance(&strip);

                        for x in 0..$x_size - 2 {
                            target[x + 1][y + 1] |=
                                ((buf[x + 0][0] + buf[x + 1][1] + buf[x + 2][2]
                                    > (9 * I::BIT_LEN as u32 / 2))
                                    as u128)
                                    << b;

                            target[x + 1][y + 1] |=
                                ((buf[x + 0][2] + buf[x + 1][1] + buf[x + 2][0]
                                    > (9 * I::BIT_LEN as u32 / 2))
                                    as u128)
                                    << (b + 64);
                        }
                    }
                }
                target
            }
        }
    };
}
impl_conv3x3mirror_apply!(32, 32, u128);

struct ExampleCache<L, I, S, O> {
    layer: PhantomData<L>,
    class: usize,
    input: I,
    sums: S,
    output: O,
}

macro_rules! impl_examplecache {
    ($x_size:expr, $y_size:expr, $output_type:ty) => {
        impl<I: BitLen>
            ExampleCache<
                Conv3x3Mirror<I, [[[I; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>,
                [[I; $y_size]; $x_size],
                [[[u32; 3]; $x_size]; $y_size - 2],
                [[$output_type; $y_size]; $x_size],
            >
        where
            Conv3x3Mirror<I, [[[I; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>:
                Apply<[[I; $y_size]; $x_size], [[$output_type; $y_size]; $x_size]>,
            [I; 3]: Patch,
        {
            fn new(
                layer: &Conv3x3Mirror<I, [[[I; 3]; 3]; <$output_type>::BIT_LEN / 2], $output_type>,
                class: usize,
                input: &[[I; $y_size]; $x_size],
            ) -> Self {
                ExampleCache {
                    layer: PhantomData,
                    class: class,
                    input: *input,
                    sums: [[[0u32; 3]; $x_size]; $y_size - 2],
                    output: layer.apply(input),
                }
            }
            fn update_all_sums_all_rows(
                &self,
                weights: &[[[I; 3]; 3]; <$output_type>::BIT_LEN / 2],
                b: usize,
            ) {
                for y in 0..$y_size {
                    self.update_all_sums_row(weights, y, b);
                }
            }
            fn update_all_sums_row(
                &self,
                weights: &[[[I; 3]; 3]; <$output_type>::BIT_LEN / 2],
                y: usize,
                b: usize,
            ) {
                let strip = [
                    self.input[0][y + 0],
                    self.input[0][y + 1],
                    self.input[0][y + 2],
                ];
                self.sums[y][0][0] = weights[b][0].hamming_distance(&strip);
                self.sums[y][0][2] = weights[b][2].hamming_distance(&strip);

                for x in 1..32 - 1 {
                    let strip = [
                        self.input[x][y + 0],
                        self.input[x][y + 1],
                        self.input[x][y + 2],
                    ];
                    self.sums[y][x] = [
                        weights[b][0].hamming_distance(&strip),
                        weights[b][1].hamming_distance(&strip),
                        weights[b][2].hamming_distance(&strip),
                    ];
                }
                let strip = [
                    self.input[31][y + 0],
                    self.input[31][y + 1],
                    self.input[31][y + 2],
                ];
                self.sums[y][31][0] = weights[b][0].hamming_distance(&strip);
                self.sums[y][31][2] = weights[b][2].hamming_distance(&strip);
            }
            fn avg_obj<H: IsCorrect<$output_type>>(
                &self,
                weights: &[[[I; 3]; 3]; <$output_type>::BIT_LEN / 2],
                b: usize,
                head: &H,
            ) -> f64 {
                let mut sum_obj = 0u32;
                for y in 0..$y_size - 2 {
                    self.update_all_sums_row(weights, y, b);
                    for x in 0..$x_size - 2 {
                        let output: $output_type = (self.output[x + 1][y + 1]
                            & (!(1 << b) | !(1 << b + <$output_type>::BIT_LEN / 2)))
                            | (((self.sums[y][x + 0][0]
                                + self.sums[y][x + 1][1]
                                + self.sums[y][x + 2][2]
                                > (9 * I::BIT_LEN as u32 / 2)) as u128)
                                << b)
                            | (((self.sums[y][x + 0][2]
                                + self.sums[y][x + 1][1]
                                + self.sums[y][x + 2][0]
                                > (9 * I::BIT_LEN as u32 / 2)) as u128)
                                << (b + <$output_type>::BIT_LEN / 2));
                    }
                }
                sum_obj as f64 / (($x_size - 2) * ($y_size - 2)) as f64
            }
        }
    };
}

impl_examplecache!(32, 32, u128);

struct ObjCacheConv3x3Mirror<II: Image2D<I>, OI: Image2D<O>, S, I, K, O>
where
    Conv3x3Mirror<I, K, O>: Apply<II, OI>,
{
    params: Conv3x3Mirror<I, K, O>,
    examples: Vec<ExampleCache<Conv3x3Mirror<I, K, O>, II, S, OI>>,
    output_index: usize,
    input_index: usize,
}

impl<II: Copy + Image2D<I>, OI: Image2D<O>, S, I, K, O> ObjCacheConv3x3Mirror<II, OI, S, I, K, O>
where
    Conv3x3Mirror<I, K, O>: Apply<II, OI>,
    [I; 3]: Patch,
{
    fn init(params: Conv3x3Mirror<I, K, O>, examples: &Vec<(usize, II)>) -> Self {
        let caches: Vec<ExampleCache<Conv3x3Mirror<I, K, O>, II, S, OI>> = examples
            .iter()
            .map(|(class, input)| {
                ExampleCache::<Conv3x3Mirror<I, K, O>, II, S, OI>::new(*class, input)
            })
            .collect();

        ObjCacheConv3x3Mirror {
            params: params,
            examples: caches,
            output_index: 0,
            input_index: 0,
        }
    }
    fn avg_obj(input_index: usize, output_index: usize) -> f64 {
        0f64
    }
    fn mutate(input_index: usize, output_index: usize) {}
}

fn update(
    weights: &Conv3x3Mirror<u32, [[[u32; 3]; 3]; 64], u128>,
    input: &[[u32; 32]; 32],
    target: &mut [[u128; 32]; 32],
    b: usize,
) {
    for y in 0..32 - 2 {
        let mut buf = [[0u32; 3]; 32];
        let strip = [input[0][y + 0], input[0][y + 1], input[0][y + 2]];
        buf[0][0] = weights.kernel[b][0].hamming_distance(&strip);
        buf[0][2] = weights.kernel[b][2].hamming_distance(&strip);

        for x in 1..32 - 1 {
            let strip = [input[x][y + 0], input[x][y + 1], input[x][y + 2]];
            buf[x] = [
                weights.kernel[b][0].hamming_distance(&strip),
                weights.kernel[b][1].hamming_distance(&strip),
                weights.kernel[b][2].hamming_distance(&strip),
            ];
        }
        let strip = [input[31][y + 0], input[31][y + 1], input[31][y + 2]];
        buf[31][0] = weights.kernel[b][0].hamming_distance(&strip);
        buf[31][2] = weights.kernel[b][2].hamming_distance(&strip);

        for x in 0..32 - 2 {
            target[x + 1][y + 1] |=
                ((buf[x + 0][0] + buf[x + 1][1] + buf[x + 2][2] > (9 * 16)) as u128) << b;

            target[x + 1][y + 1] |=
                ((buf[x + 0][2] + buf[x + 1][1] + buf[x + 2][0] > (9 * 16)) as u128) << (b + 64);
        }
    }
}

fn load_data() -> Vec<(usize, [[u32; 32]; 32])> {
    let size: usize = 10_000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
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

fn flip_image<P: Copy + Default>(input: &[[P; 32]; 32]) -> [[P; 32]; 32] {
    let mut target = [<[P; 32]>::default(); 32];
    for i in 0..32 {
        target[((i as i64 * -1) + 31) as usize] = input[i];
    }
    target
}

fn new_rand_weights<RNG: rand::Rng>(rng: &mut RNG) -> [[[u32; 3]; 3]; 64] {
    let mut output = [[[0u32; 3]; 3]; 64];
    for i in 0..64 {
        output[i] = rng.gen();
    }
    output
}

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(42);

    let weights: Conv3x3Mirror<u32, [[[u32; 3]; 3]; 64], u128> = Conv3x3Mirror {
        input_pixel_type: PhantomData,
        kernel: new_rand_weights(&mut rng),
        output_type: PhantomData,
    };

    //let weights: Conv3x3Mirror<u32, [[u32; 3]; 64], u128> = Conv3x3Mirror {
    //    input_pixel_type: PhantomData,
    //    center: [[!0u32, !0u32, 0u32]; 64],
    //    a_side: [[!0u32, 0u32, 0u32]; 64],
    //    b_side: [[!0u32, 0u32, 0u32]; 64],
    //    output_type: PhantomData,
    //};
    let examples = load_data();

    let start = PreciseTime::now();
    let mut opt_outputs: Vec<_> = examples
        .iter()
        .map(|(_, input_image)| weights.apply(input_image))
        .collect();
    println!(
        "apply duration: {:?}",
        start.to(PreciseTime::now()).num_milliseconds()
    );

    //apply duration: 11747
    //update duration: 193

    let start = PreciseTime::now();
    let _: Vec<_> = opt_outputs
        .iter_mut()
        .zip(examples.iter())
        .map(|(target, input)| update(&weights, &input.1, target, 17))
        .collect();
    println!(
        "update duration: {:?}",
        start.to(PreciseTime::now()).num_milliseconds()
    );

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
