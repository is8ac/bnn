extern crate bitnn;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::layers::{Apply, SaveLoad};
use bitnn::{
    BitLen, ExtractPatches, FlipBit, GetBit, HammingDistance,
};
use rand::prelude::*;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs;
use std::iter;
use std::path::Path;
use time::PreciseTime;

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
impl_elementwiseadd_for_array!(32);

pub trait ArrayBitIncrement {
    type CountersType;
    fn increment_counters(&self, counters: &mut Self::CountersType);
    fn compare_and_bitpack(
        counters_0: &Self::CountersType,
        counters_1: &Self::CountersType,
    ) -> Self;
}

impl ArrayBitIncrement for u32 {
    type CountersType = [u32; 32];
    fn increment_counters(&self, counters: &mut Self::CountersType) {
        for b in 0..32 {
            counters[b] += (self >> b) & 1
        }
    }
    fn compare_and_bitpack(
        counters_0: &Self::CountersType,
        counters_1: &Self::CountersType,
    ) -> Self {
        let mut target = 0u32;
        for b in 0..32 {
            target |= ((counters_0[b] > counters_1[b]) as u32) << b;
        }
        target
    }
}

impl ArrayBitIncrement for u8 {
    type CountersType = [u32; 8];
    fn increment_counters(&self, counters: &mut Self::CountersType) {
        for b in 0..8 {
            counters[b] += ((self >> b) & 1) as u32
        }
    }
    fn compare_and_bitpack(
        counters_0: &Self::CountersType,
        counters_1: &Self::CountersType,
    ) -> Self {
        let mut target = 0u8;
        for b in 0..8 {
            target |= ((counters_0[b] > counters_1[b]) as u8) << b;
        }
        target
    }
}

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
                counters_0: &Self::CountersType,
                counters_1: &Self::CountersType,
            ) -> Self {
                let mut target = [T::default(); $len];
                for i in 0..$len {
                    target[i] = T::compare_and_bitpack(&counters_0[i], &counters_1[i]);
                }
                target
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
        counters: &mut Self::MatrixCountersType,
        input: &Input,
        target: &Target,
    );
    fn bitpack(counters: &Self::MatrixCountersType) -> Self;
}

impl<Input: ArrayBitIncrement> MatrixBitIncrement<Input, bool> for Input {
    type MatrixCountersType = [Input::CountersType; 2];
    fn increment_matrix_counters(
        counters: &mut Self::MatrixCountersType,
        input: &Input,
        target: &bool,
    ) {
        input.increment_counters(&mut counters[*target as usize]);
    }
    fn bitpack(counters: &Self::MatrixCountersType) -> Self {
        Self::compare_and_bitpack(&counters[0], &counters[1])
    }
}

impl<Input: ArrayBitIncrement + Copy + Default> MatrixBitIncrement<Input, u8> for [Input; 8] {
    type MatrixCountersType = [[Input::CountersType; 2]; 8];
    fn increment_matrix_counters(
        counters: &mut Self::MatrixCountersType,
        input: &Input,
        target: &u8,
    ) {
        for b in 0..8 {
            input.increment_counters(&mut counters[b][target.bit(b) as usize]);
        }
    }
    fn bitpack(counters: &Self::MatrixCountersType) -> Self {
        let mut output = [Input::default(); 8];
        for b in 0..8 {
            output[b] = Input::compare_and_bitpack(&counters[b][0], &counters[b][1]);
        }
        output
    }
}

impl<Input: ArrayBitIncrement + Copy + Default> MatrixBitIncrement<Input, u32> for [Input; 32] {
    type MatrixCountersType = [[Input::CountersType; 2]; 32];
    fn increment_matrix_counters(
        counters: &mut Self::MatrixCountersType,
        input: &Input,
        target: &u32,
    ) {
        for b in 0..32 {
            input.increment_counters(&mut counters[b][target.bit(b) as usize]);
        }
    }
    fn bitpack(counters: &Self::MatrixCountersType) -> Self {
        let mut output = [Input::default(); 32];
        for b in 0..32 {
            output[b] = Input::compare_and_bitpack(&counters[b][0], &counters[b][1]);
        }
        output
    }
}

macro_rules! impl_matrixbitincrement_for_array_of_matrixbitincrement {
    ($len:expr) => {
        impl<Input, Target, MatrixBits: MatrixBitIncrement<Input, Target> + Copy + Default>
            MatrixBitIncrement<Input, [Target; $len]> for [MatrixBits; $len]
        {
            type MatrixCountersType = [MatrixBits::MatrixCountersType; $len];
            fn increment_matrix_counters(
                counters: &mut Self::MatrixCountersType,
                input: &Input,
                target: &[Target; $len],
            ) {
                for i in 0..$len {
                    MatrixBits::increment_matrix_counters(&mut counters[i], input, &target[i]);
                }
            }
            fn bitpack(counters: &Self::MatrixCountersType) -> Self {
                let mut output = [MatrixBits::default(); $len];
                for i in 0..$len {
                    output[i] = MatrixBits::bitpack(&counters[i]);
                }
                output
            }
        }
    };
}

impl_matrixbitincrement_for_array_of_matrixbitincrement!(1);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(2);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(3);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(4);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(8);
impl_matrixbitincrement_for_array_of_matrixbitincrement!(32);

pub trait OptimizeInput<Weights, Target> {
    fn optimize(&mut self, weights: &Weights, target: &Target);
}

impl<Input: BitLen + FlipBit, Weights: Apply<Input, Target>, Target: HammingDistance>
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

pub trait TrainEncoder<Input, Embedding, Decoder> {
    fn train_encoder<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Input>) -> Self;
}

impl<
        Input: Sync + Send + HammingDistance,
        Encoder: MatrixBitIncrement<Input, Embedding> + Sync + Send + Apply<Input, Embedding>,
        Embedding: Sync + Send + OptimizeInput<Decoder, Input>,
        Decoder: MatrixBitIncrement<Embedding, Input> + Sync + Send + Apply<Embedding, Input>,
    > TrainEncoder<Input, Embedding, Decoder> for Encoder
where
    Encoder::MatrixCountersType: Default + Sync + Send + ElementwiseAdd,
    Decoder::MatrixCountersType: Default + Sync + Send + ElementwiseAdd,
    rand::distributions::Standard: rand::distributions::Distribution<Encoder>,
    rand::distributions::Standard: rand::distributions::Distribution<Decoder>,
    <Decoder as MatrixBitIncrement<Embedding, Input>>::MatrixCountersType: std::marker::Send,
{
    fn train_encoder<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Input>) -> Self {
        let mut encoder: Encoder = rng.gen();
        //let mut decoder: Decoder = rng.gen();

        let start = PreciseTime::now();
        let decoder_counters = examples
            .par_iter()
            .fold(
                || Decoder::MatrixCountersType::default(),
                |mut counter, patch| {
                    let embedding = encoder.apply(&patch);
                    <Decoder>::increment_matrix_counters(&mut counter, &embedding, patch);
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
        println!("decoder time: {}", start.to(PreciseTime::now()));
        let decoder = <Decoder as MatrixBitIncrement<Embedding, Input>>::bitpack(&decoder_counters);

        let start = PreciseTime::now();
        let encoder_counters = examples
            .par_iter()
            .fold(
                || Encoder::MatrixCountersType::default(),
                |mut counter, patch| {
                    let mut embedding = encoder.apply(&patch);
                    embedding.optimize(&decoder, patch);
                    Encoder::increment_matrix_counters(&mut counter, patch, &embedding);
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

        println!("encoder time: {}", start.to(PreciseTime::now()));
        let encoder2 =
            <Encoder as MatrixBitIncrement<Input, Embedding>>::bitpack(&encoder_counters);
        encoder = encoder2;

        let sum_hd: u64 = examples
            .par_iter()
            .map(|patch| {
                let embedding = encoder.apply(patch);
                let output = decoder.apply(&embedding);
                output.hamming_distance(patch) as u64
            })
            .sum();
        println!("avg hd: {:}", sum_hd as f64 / examples.len() as f64);
        encoder
    }
}

const N_EXAMPLES: usize = 60_00;
const EMBEDDING_LEN: usize = 1;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(26))
        .build_global()
        .unwrap();

    let base_path = Path::new("params/fc_test_1");
    fs::create_dir_all(base_path).unwrap();

    let mut rng = Hc128Rng::seed_from_u64(8);
    let images = mnist::load_images_u8_unary(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"),
        N_EXAMPLES,
    );
    let classes = mnist::load_labels(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"),
        N_EXAMPLES,
    );
    let examples: Vec<([[u8; 28]; 28], usize)> = images
        .iter()
        .cloned()
        .zip(classes.iter().map(|x| *x as usize))
        .collect();

    let patches = {
        let mut patches: Vec<([[u8; 3]; 3], usize)> = examples
            .iter()
            .map(|(image, class)| {
                let patches: Vec<([[u8; 3]; 3], usize)> = image
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
        patches.shuffle(&mut rng);
        patches
    };
    let unlabeled_patches: Vec<_> = patches.par_iter().map(|(patch, _)| *patch).collect();
    dbg!(patches.len());

    let encoder = <[[[[u8; 3]; 3]; 32]; EMBEDDING_LEN] as TrainEncoder<
        _,
        _,
        [[[[u32; EMBEDDING_LEN]; 8]; 3]; 3],
    >>::train_encoder(&mut rng, &unlabeled_patches);
}
