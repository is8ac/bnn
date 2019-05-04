extern crate bitnn;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::layers::{Apply, SaveLoad};
use bitnn::train::OptimizePass;
use bitnn::{
    BitLen, ExtractPatches, FlipBit, FlipBitIndexed, GetBit, GetPatch, HammingDistance, SetBit,
};
use rand::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use rayon::prelude::*;
use std::fs;
use std::iter;
use std::marker::PhantomData;
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
    type BitsType;
    fn increment_counters(&mut self, bits: &Self::BitsType);
    fn compare_and_bitpack(&self, other: &Self) -> Self::BitsType;
}

impl ArrayBitIncrement for [u32; 32] {
    type BitsType = u32;
    fn increment_counters(&mut self, bits: &Self::BitsType) {
        for b in 0..32 {
            self[b] += (bits >> b) & 1
        }
    }
    fn compare_and_bitpack(&self, other: &Self) -> Self::BitsType {
        let mut target = 0u32;
        for b in 0..32 {
            target |= ((self[b] > other[b]) as u32) << b;
        }
        target
    }
}

impl ArrayBitIncrement for [u32; 8] {
    type BitsType = u8;
    fn increment_counters(&mut self, bits: &Self::BitsType) {
        for b in 0..8 {
            self[b] += (bits >> b) as u32 & 1u32
        }
    }
    fn compare_and_bitpack(&self, other: &Self) -> Self::BitsType {
        let mut target = 0u8;
        for b in 0..8 {
            target |= ((self[b] > other[b]) as u8) << b;
        }
        target
    }
}

macro_rules! impl_bitincrement_for_array {
    ($len:expr) => {
        impl<T: ArrayBitIncrement> ArrayBitIncrement for [T; $len]
        where
            T::BitsType: Default + Copy,
        {
            type BitsType = [T::BitsType; $len];
            fn increment_counters(&mut self, bits: &Self::BitsType) {
                for i in 0..$len {
                    self[i].increment_counters(&bits[i]);
                }
            }
            fn compare_and_bitpack(&self, other: &Self) -> Self::BitsType {
                let mut target = [T::BitsType::default(); $len];
                for i in 0..$len {
                    target[i] = self[i].compare_and_bitpack(&other[i]);
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

pub trait MatrixBitIncrement<InputCounters, Target>
where
    InputCounters: ArrayBitIncrement,
{
    type BitsType;
    fn increment_matrix_counters(&mut self, input: &InputCounters::BitsType, target: &Target);
    fn bitpack(&self) -> Self::BitsType;
}

impl<InputCounters: ArrayBitIncrement> MatrixBitIncrement<InputCounters, bool>
    for [InputCounters; 2]
{
    type BitsType = InputCounters::BitsType;
    fn increment_matrix_counters(&mut self, input: &InputCounters::BitsType, target: &bool) {
        self[*target as usize].increment_counters(input);
    }
    fn bitpack(&self) -> InputCounters::BitsType {
        self[0].compare_and_bitpack(&self[1])
    }
}

impl<InputCounters: ArrayBitIncrement> MatrixBitIncrement<InputCounters, u32>
    for [[InputCounters; 2]; 32]
where
    InputCounters::BitsType: Default + Copy,
{
    type BitsType = [InputCounters::BitsType; 32];
    fn increment_matrix_counters(&mut self, input: &InputCounters::BitsType, target: &u32) {
        for b in 0..32 {
            self[b][target.bit(b) as usize].increment_counters(input);
        }
    }
    fn bitpack(&self) -> [InputCounters::BitsType; 32] {
        let mut output = [InputCounters::BitsType::default(); 32];
        for b in 0..32 {
            output[b] = self[b][0].compare_and_bitpack(&self[b][1]);
        }
        output
    }
}
impl<InputCounters: ArrayBitIncrement> MatrixBitIncrement<InputCounters, u8>
    for [[InputCounters; 2]; 8]
where
    InputCounters::BitsType: Default + Copy,
{
    type BitsType = [InputCounters::BitsType; 8];
    fn increment_matrix_counters(&mut self, input: &InputCounters::BitsType, target: &u8) {
        for b in 0..8 {
            self[b][target.bit(b) as usize].increment_counters(input);
        }
    }
    fn bitpack(&self) -> [InputCounters::BitsType; 8] {
        let mut output = [InputCounters::BitsType::default(); 8];
        for b in 0..8 {
            output[b] = self[b][0].compare_and_bitpack(&self[b][1]);
        }
        output
    }
}

macro_rules! impl_matrixbitincrement_for_array_of_matrixbitincrement {
    ($len:expr) => {
        impl<
                InputCounters: ArrayBitIncrement,
                Target,
                MatrixCounter: MatrixBitIncrement<InputCounters, Target>,
            > MatrixBitIncrement<InputCounters, [Target; $len]> for [MatrixCounter; $len]
        where
            MatrixCounter::BitsType: Default + Copy,
        {
            type BitsType = [MatrixCounter::BitsType; $len];
            fn increment_matrix_counters(
                &mut self,
                input: &InputCounters::BitsType,
                target: &[Target; $len],
            ) {
                for i in 0..$len {
                    self[i].increment_matrix_counters(input, &target[i]);
                }
            }
            fn bitpack(&self) -> [MatrixCounter::BitsType; $len] {
                let mut output = [MatrixCounter::BitsType::default(); $len];
                for i in 0..$len {
                    output[i] = self[i].bitpack();
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
//impl_matrixbitincrement_for_array_of_matrixbitincrement!(8);
//impl_matrixbitincrement_for_array_of_matrixbitincrement!(32);

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

pub trait TrainEncoder<Input, Embedded, Decoder> {
    fn train_encoder<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Input>) -> Self;
}

impl<
        Input: Sync,
        InputCounters,
        Embedding,
        EmbeddingCounters,
        Decoder: Apply<Embedding, Input> + MatrixBitIncrement<EmbeddingCounters, Input>,
    > TrainEncoder<Input, Embedding, Decoder> for [[Input; 32]; EMBEDDING_LEN]
where
    Self: Apply<Input, Embedding> + MatrixBitIncrement<InputCounters, Embedding>,
    rand::distributions::Standard: rand::distributions::Distribution<Self>,
    rand::distributions::Standard: rand::distributions::Distribution<Decoder>,
{
    fn train_encoder<RNG: rand::Rng>(rng: &mut RNG, examples: &Vec<Input>) -> Self {
        let mut encoder: Self = rng.gen();
        let mut decoder: Decoder = rng.gen();

        let start = PreciseTime::now();
        let decoder_counters = examples
            .par_iter()
            .fold(
                || [[[[[[0u32; 32]; EMBEDDING_LEN]; 2]; 8]; 3]; 3],
                |mut counter, patch| {
                    let embedding = encoder.apply(patch);
                    counter.increment_matrix_counters(&embedding, patch);
                    counter
                },
            )
            .reduce(
                || [[[[[[0u32; 32]; EMBEDDING_LEN]; 2]; 8]; 3]; 3],
                |mut a, b| {
                    a.elementwise_add(&b);
                    a
                },
            );
        println!("decoder time: {}", start.to(PreciseTime::now()));
        decoder = decoder_counters.bitpack();

        let start = PreciseTime::now();
        let encoder_counters = examples
            .par_iter()
            .fold(
                || [[[[[[0u32; 8]; 3]; 3]; 2]; 32]; EMBEDDING_LEN],
                |mut counter, patch| {
                    let mut embedding = encoder.apply(patch);
                    embedding.optimize(&decoder, patch);
                    counter.increment_matrix_counters(patch, &embedding);
                    counter
                },
            )
            .reduce(
                || [[[[[[0u32; 8]; 3]; 3]; 2]; 32]; EMBEDDING_LEN],
                |mut a, b| {
                    a.elementwise_add(&b);
                    a
                },
            );

        println!("encoder time: {}", start.to(PreciseTime::now()));
        encoder = encoder_counters.bitpack();
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
const EMBEDDING_LEN: usize = 2;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(25))
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
    dbg!(patches.len());
}
