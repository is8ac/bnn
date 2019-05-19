extern crate bitnn;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::layers::{Conv2D, SaveLoad};
use bitnn::{Apply, HammingDistance};
use bitnn::{BitLen, ElementwiseAdd, ExtractPatches, FlipBit, GetBit};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs;
use std::iter::repeat;
use std::path::Path;
use time::PreciseTime;

trait Wrap<T> {
    type Wrapped;
}

macro_rules! impl_wrap_for_uint {
    ($type:ty) => {
        impl<I> Wrap<$type> for I {
            type Wrapped = [I; <$type>::BIT_LEN];
        }
    };
}
impl_wrap_for_uint!(u8);
impl_wrap_for_uint!(u32);

macro_rules! impl_wrap_for_array {
    ($len:expr) => {
        impl<T, I: Wrap<T>> Wrap<[T; $len]> for I {
            type Wrapped = [<I as Wrap<T>>::Wrapped; $len];
        }
    };
}

impl_wrap_for_array!(1);
impl_wrap_for_array!(2);
impl_wrap_for_array!(3);
impl_wrap_for_array!(4);
impl_wrap_for_array!(8);
impl_wrap_for_array!(32);

trait InputBits
where
    Self: Sized,
    u32: Wrap<Self>,
    f64: Wrap<Self>,
{
    fn compare(
        counters_0: &<u32 as Wrap<Self>>::Wrapped,
        counters_1: &<u32 as Wrap<Self>>::Wrapped,
        len: usize,
    ) -> <f64 as Wrap<Self>>::Wrapped;
    fn increment_counters(&self, counters: &mut <u32 as Wrap<Self>>::Wrapped);
    fn backprop<Target: TargetBits<Self> + BitLen>(
        &mut self,
        target: &Target,
        weights: &<Self as Wrap<Target>>::Wrapped,
        tanh_width: u32,
        n_updates: usize,
    ) where
        <u32 as Wrap<Self>>::Wrapped: Default,
        <f64 as Wrap<Self>>::Wrapped: Wrap<Target>,
        Self: Wrap<Target>,
        (<u32 as Wrap<Self>>::Wrapped, <u32 as Wrap<Self>>::Wrapped): Wrap<Target>,
        (<f64 as Wrap<Self>>::Wrapped, <f64 as Wrap<Self>>::Wrapped): Wrap<Target>,
        f64: Wrap<<Self as Wrap<Target>>::Wrapped>,
        Self: BitMatrix,
        Self::IndexType: Sized + Copy,
    {
        let mut counters_0 = <u32 as Wrap<Self>>::Wrapped::default();
        let mut counters_1 = <u32 as Wrap<Self>>::Wrapped::default();
        target.increment_input_counters(
            &mut counters_0,
            &mut counters_1,
            self,
            weights,
            tanh_width,
        );
        let diffs = Self::compare(&counters_0, &counters_1, Target::BIT_LEN);
        let mut grads = self.indexed_grads(&diffs);
        grads.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
        for (_, index) in grads.iter().filter(|(x, _)| *x > 0f64).take(n_updates) {
            self.flip_indexed_bit(*index);
        }
    }
}

macro_rules! impl_inputbits_for_uint {
    ($type:ty) => {
        impl InputBits for $type {
            fn compare(
                counters_0: &<u32 as Wrap<Self>>::Wrapped,
                counters_1: &<u32 as Wrap<Self>>::Wrapped,
                len: usize,
            ) -> <f64 as Wrap<Self>>::Wrapped {
                let mut diffs = <f64 as Wrap<Self>>::Wrapped::default();
                for b in 0..<$type>::BIT_LEN {
                    diffs[b] = (counters_1[b] as f64 - counters_0[b] as f64) / len as f64;
                }
                diffs
            }
            fn increment_counters(&self, counters: &mut <u32 as Wrap<Self>>::Wrapped) {
                for b in 0..<$type>::BIT_LEN {
                    counters[b] += ((self >> b) & 1) as u32
                }
            }
        }
    };
}

impl_inputbits_for_uint!(u8);
impl_inputbits_for_uint!(u32);

macro_rules! impl_inputbits_for_len {
    ($len:expr) => {
        impl<T: InputBits> InputBits for [T; $len]
        where
            Self: Sized,
            u32: Wrap<T>,
            f64: Wrap<T>,
            <f64 as Wrap<Self>>::Wrapped: Default,
        {
            fn compare(
                counters_0: &<u32 as Wrap<Self>>::Wrapped,
                counters_1: &<u32 as Wrap<Self>>::Wrapped,
                len: usize,
            ) -> <f64 as Wrap<Self>>::Wrapped {
                let mut diffs = <f64 as Wrap<Self>>::Wrapped::default();
                for i in 0..$len {
                    diffs[i] = T::compare(&counters_0[i], &counters_1[i], len);
                }
                diffs
            }
            fn increment_counters(&self, counters: &mut <u32 as Wrap<Self>>::Wrapped) {
                for i in 0..$len {
                    self[i].increment_counters(&mut counters[i]);
                }
            }
        }
    };
}

impl_inputbits_for_len!(1);
impl_inputbits_for_len!(2);
impl_inputbits_for_len!(3);
impl_inputbits_for_len!(4);
impl_inputbits_for_len!(8);
impl_inputbits_for_len!(32);

trait TargetBits<Input>
where
    f64: Wrap<Input>,
    //(<f64 as Wrap<Input>>::Wrapped, <f64 as Wrap<Input>>::Wrapped): Wrap<Self>,
    u32: Wrap<Input>,
    (<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped): Wrap<Self>,
    Input: Wrap<Self>,
    Self: Sized,
    <f64 as Wrap<Input>>::Wrapped: Wrap<Self>,
{
    fn matrix_compare(
        counters: &<(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<
            Self,
        >>::Wrapped,
        len: usize,
    ) -> <<f64 as Wrap<Input>>::Wrapped as Wrap<Self>>::Wrapped;
    fn increment_matrix_counters(
        &self,
        counters: &mut <(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<
            Self,
        >>::Wrapped,
        weights: &<Input as Wrap<Self>>::Wrapped,
        input: &Input,
        tanh_width: u32,
    );
    fn increment_input_counters(
        &self,
        counters_0: &mut <u32 as Wrap<Input>>::Wrapped,
        counters_1: &mut <u32 as Wrap<Input>>::Wrapped,
        input: &Input,
        weights: &<Input as Wrap<Self>>::Wrapped,
        tanh_width: u32,
    );
}

macro_rules! impl_targetbits_for_uint {
    ($type:ty) => {
        impl<Input: InputBits + Wrap<Self, Wrapped=[Input; <$type>::BIT_LEN]> + BitLen + HammingDistance> TargetBits<Input> for $type
        where
            u32: Wrap<Input>,
            f64: Wrap<Input>,
            <<f64 as Wrap<Input>>::Wrapped as Wrap<Self>>::Wrapped: Default,
        {
            fn matrix_compare(
                counters: &<(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<
                    Self,
                >>::Wrapped,
                len: usize,
            ) -> <<f64 as Wrap<Input>>::Wrapped as Wrap<Self>>::Wrapped {
                let mut diffs = <<f64 as Wrap<Input>>::Wrapped as Wrap<Self>>::Wrapped::default();
                for b in 0..<$type>::BIT_LEN {
                    diffs[b] = Input::compare(&counters[b].0, &counters[b].1, len);
                }
                diffs
            }
            fn increment_matrix_counters(
                &self,
                counters: &mut <(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<Self>>::Wrapped,
                weights: &<Input as Wrap<$type>>::Wrapped,
                input: &Input,
                tanh_width: u32,
            ) {
                for b in 0..<$type>::BIT_LEN {
                    let activation = weights[b].hamming_distance(&input);
                    let threshold = Input::BIT_LEN as u32 / 2;
                    // this patch only gets to vote if it is within tanh_width.
                    let diff = activation.saturating_sub(threshold) | threshold.saturating_sub(activation);
                    if diff < tanh_width {
                        if self.bit(b) {
                            input.increment_counters(&mut counters[b].0);
                        } else {
                            input.increment_counters(&mut counters[b].1);
                        }
                    }
                }
            }
            fn increment_input_counters(
                &self,
                counters_0: &mut <u32 as Wrap<Input>>::Wrapped,
                counters_1: &mut <u32 as Wrap<Input>>::Wrapped,
                input: &Input,
                weights: &<Input as Wrap<$type>>::Wrapped,
                tanh_width: u32
            ) {
                for b in 0..<$type>::BIT_LEN {
                    let activation = weights[b].hamming_distance(&input);
                    let threshold = Input::BIT_LEN as u32 / 2;
                    // this patch only gets to vote if it is within tanh_width.
                    let diff = activation.saturating_sub(threshold) | threshold.saturating_sub(activation);
                    if diff < tanh_width {
                        if self.bit(b) {
                            weights[b].increment_counters(counters_0);
                        } else {
                            weights[b].increment_counters(counters_1);
                        }
                    }
                }
            }
        }
    };
}

impl_targetbits_for_uint!(u8);
impl_targetbits_for_uint!(u32);

macro_rules! impl_targetbits_for_array {
    ($len:expr) => {
        impl<Input: InputBits + Wrap<T>, T: TargetBits<Input>> TargetBits<Input> for [T; $len]
        where
            u32: Wrap<Input>,
            f64: Wrap<Input>,
            (<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped): Wrap<T>,
            (<f64 as Wrap<Input>>::Wrapped, <f64 as Wrap<Input>>::Wrapped): Wrap<T>,
            <f64 as Wrap<Input>>::Wrapped: Wrap<T>,
            <<f64 as Wrap<Input>>::Wrapped as Wrap<Self>>::Wrapped: Default,
        {
            fn matrix_compare(
                counters: &<(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<
                    Self,
                >>::Wrapped,
                len: usize,
            ) -> <<f64 as Wrap<Input>>::Wrapped as Wrap<Self>>::Wrapped {
                let mut diffs = <<f64 as Wrap<Input>>::Wrapped as Wrap<Self>>::Wrapped::default();
                for i in 0..$len {
                    diffs[i] = T::matrix_compare(&counters[i], len);
                }
                diffs
            }
            fn increment_matrix_counters(
                &self,
                counters: &mut <(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<Self>>::Wrapped,
                weights: &<Input as Wrap<Self>>::Wrapped,
                input: &Input,
                tanh_width: u32,
            ) {
                for i in 0..$len {
                    self[i].increment_matrix_counters(&mut counters[i], &weights[i], input, tanh_width);
                }
            }
            fn increment_input_counters(
                &self,
                counters_0: &mut <u32 as Wrap<Input>>::Wrapped,
                counters_1: &mut <u32 as Wrap<Input>>::Wrapped,
                input: &Input,
                weights: &<Input as Wrap<Self>>::Wrapped,
                tanh_width: u32
            ) {
                for b in 0..$len {
                    self[b].increment_input_counters(counters_0, counters_1, input, &weights[b], tanh_width);
                }
            }
        }
    }
}

impl_targetbits_for_array!(1);
impl_targetbits_for_array!(2);
impl_targetbits_for_array!(3);
impl_targetbits_for_array!(4);
impl_targetbits_for_array!(8);
impl_targetbits_for_array!(32);

trait UpdateMatrix<Input, Output> {
    fn update(&mut self, examples: &Vec<(Input, Output)>, n_updates: usize, tanh_width: u32);
}

impl<Input: Wrap<Output> + Sync, Output: Sync + TargetBits<Input>> UpdateMatrix<Input, Output>
    for <Input as Wrap<Output>>::Wrapped
where
    u32: Wrap<Input>,
    f64: Wrap<Input> + Wrap<<Input as Wrap<Output>>::Wrapped>,
    <f64 as Wrap<Input>>::Wrapped:
        Wrap<Output, Wrapped = <f64 as Wrap<<Input as Wrap<Output>>::Wrapped>>::Wrapped>,
    (<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped): Wrap<Output>,
    <(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<Output>>::Wrapped:
        Default + Sync + Send + ElementwiseAdd,
    Self: BitMatrix,
    <Input as Wrap<Output>>::Wrapped: Sync,
    <<Input as Wrap<Output>>::Wrapped as BitMatrix>::IndexType: Copy,
{
    fn update(&mut self, examples: &Vec<(Input, Output)>, n_updates: usize, tanh_width: u32) {
        let counters = examples
            .par_iter()
            .fold(
                || {
                    <(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<
                        Output,
                    >>::Wrapped::default()
                },
                |mut counters, (input, target)| {
                    target.increment_matrix_counters(&mut counters, self, &input, tanh_width);
                    counters
                },
            )
            .reduce(
                || {
                    <(<u32 as Wrap<Input>>::Wrapped, <u32 as Wrap<Input>>::Wrapped) as Wrap<
                        Output,
                    >>::Wrapped::default()
                },
                |mut a, b| {
                    a.elementwise_add(&b);
                    a
                },
            );
        let diffs = <Output as TargetBits<Input>>::matrix_compare(&counters, examples.len());
        let mut grads = self.indexed_grads(&diffs);
        dbg!(grads.len());
        grads.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
        for (_, index) in grads.iter().filter(|(x, _)| *x > 0f64).take(n_updates) {
            self.flip_indexed_bit(*index);
        }
    }
}

// Things we can do to a (N-dim) bit matrix.
trait BitMatrix
where
    Self: Sized,
    f64: Wrap<Self>,
{
    type IndexType;
    fn flip_indexed_bit(&mut self, index: Self::IndexType);
    fn indexed_grads(&self, grads: &<f64 as Wrap<Self>>::Wrapped) -> Vec<(f64, Self::IndexType)>;
}

macro_rules! impl_bitmatrix_for_uint {
    ($type:ty) => {
        impl BitMatrix for $type
        where
            Self: Sized,
            f64: Wrap<Self>,
        {
            type IndexType = usize;
            fn flip_indexed_bit(&mut self, index: usize) {
                self.flip_bit(index);
            }
            fn indexed_grads(&self, grads: &<f64 as Wrap<Self>>::Wrapped) -> Vec<(f64, usize)> {
                grads
                    .iter()
                    .enumerate()
                    .map(|(i, g)| (*g * if self.bit(i) { -1f64 } else { 1f64 }, i))
                    .collect()
            }
        }
    };
}

impl_bitmatrix_for_uint!(u8);
impl_bitmatrix_for_uint!(u32);

macro_rules! impl_bitmatrix_for_array {
    ($len:expr) => {
        impl<T: BitMatrix> BitMatrix for [T; $len]
        where
            Self: Sized,
            f64: Wrap<T>,
            T::IndexType: Copy + Sized,
        {
            type IndexType = (usize, T::IndexType);
            fn flip_indexed_bit(&mut self, index: Self::IndexType) {
                self[index.0].flip_indexed_bit(index.1);
            }
            fn indexed_grads(
                &self,
                grads: &<f64 as Wrap<Self>>::Wrapped,
            ) -> Vec<(f64, Self::IndexType)> {
                grads
                    .iter()
                    .zip(self.iter())
                    .enumerate()
                    .map(|(i, (grads, bits))| {
                        let indices: Vec<(f64, (usize, T::IndexType))> = bits
                            .indexed_grads(grads)
                            .iter()
                            .map(|(v, indexes)| (*v, (i, *indexes)))
                            .collect();
                        indices
                    })
                    .flatten()
                    .collect()
            }
        }
    };
}

impl_bitmatrix_for_array!(1);
impl_bitmatrix_for_array!(2);
impl_bitmatrix_for_array!(3);
impl_bitmatrix_for_array!(4);
impl_bitmatrix_for_array!(8);
impl_bitmatrix_for_array!(32);

fn log_hd<
    Encoder: Apply<Patch, Embedding> + Sync,
    Patch: HammingDistance + Sync,
    Decoder: Apply<Embedding, Patch> + Sync,
>(
    encoder: &Encoder,
    decoder: &Decoder,
    patches: &Vec<Patch>,
) {
    let sum_hd: u64 = patches
        .par_iter()
        .map(|patch| {
            let embedding = encoder.apply(patch);
            let output = decoder.apply(&embedding);
            output.hamming_distance(patch) as u64
        })
        .sum();
    println!("avg hd: {}", sum_hd as f64 / patches.len() as f64,);
}

type Embedding = [u32; 2];
type Patch = [[u8; 3]; 3];
type Encoder = <Patch as Wrap<Embedding>>::Wrapped;
type Decoder = <Embedding as Wrap<Patch>>::Wrapped;

const N_EXAMPLES: usize = 100;

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

    let patches: Vec<Patch> = images
        .iter()
        .map(|image| {
            let patches: Vec<Patch> = image.patches().iter().cloned().collect();
            patches
        })
        .flatten()
        .collect();
    dbg!(patches.len());

    let mut encoder: Encoder = rng.gen();
    let mut decoder: Decoder = rng.gen();
    let sum_hd: u64 = patches
        .par_iter()
        .map(|patch| {
            let embedding = encoder.apply(patch);
            let output = decoder.apply(&embedding);
            output.hamming_distance(patch) as u64
        })
        .sum();
    println!("avg hd: {}", sum_hd as f64 / patches.len() as f64,);

    let decoder_counters = patches
        .par_iter()
        .fold(
            || {
                <(
                    <u32 as Wrap<Embedding>>::Wrapped,
                    <u32 as Wrap<Embedding>>::Wrapped,
                ) as Wrap<Patch>>::Wrapped::default()
            },
            |mut counter, patch| {
                let embedding = encoder.apply(patch);
                patch.increment_matrix_counters(&mut counter, &decoder, &embedding, 3);
                counter
            },
        )
        .reduce(
            || {
                <(
                    <u32 as Wrap<Embedding>>::Wrapped,
                    <u32 as Wrap<Embedding>>::Wrapped,
                ) as Wrap<Patch>>::Wrapped::default()
            },
            |mut a, b| {
                a.elementwise_add(&b);
                a
            },
        );
    let diffs = <Patch as TargetBits<Embedding>>::matrix_compare(&decoder_counters, patches.len());
    let mut grads = decoder.indexed_grads(&diffs);
    dbg!(grads.len());
    grads.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
    //dbg!(&grads[..30]);
    let mut updates = grads.iter().filter(|(x, _)| *x > 0f64);
    for _ in 0..1100 {
        let (_, index) = updates.next().unwrap();
        decoder.flip_indexed_bit(*index);
    }
    dbg!(Decoder::BIT_LEN);
    log_hd(&encoder, &decoder, &patches);
    for _ in 0..10 {
        let (_, index) = updates.next().unwrap();
        decoder.flip_indexed_bit(*index);
    }
    log_hd(&encoder, &decoder, &patches);

    let start = PreciseTime::now();
    let encoder_counters = patches
        .par_iter()
        .fold(
            || {
                <(<u32 as Wrap<Patch>>::Wrapped, <u32 as Wrap<Patch>>::Wrapped) as Wrap<
                    Embedding,
                >>::Wrapped::default()
            },
            |mut counter, patch| {
                let mut embedding = encoder.apply(patch);
                embedding.backprop(patch, &decoder, 2, 9);
                embedding.increment_matrix_counters(&mut counter, &encoder, patch, 3);
                counter
            },
        )
        .reduce(
            || {
                <(<u32 as Wrap<Patch>>::Wrapped, <u32 as Wrap<Patch>>::Wrapped) as Wrap<
                    Embedding,
                >>::Wrapped::default()
            },
            |mut a, b| {
                a.elementwise_add(&b);
                a
            },
        );
    println!("time: {}", start.to(PreciseTime::now()));
    let diffs = <Embedding as TargetBits<Patch>>::matrix_compare(&encoder_counters, patches.len());
    let mut grads = encoder.indexed_grads(&diffs);
    dbg!(grads.len());
    grads.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());
    //dbg!(&grads[..30]);
    let mut updates = grads.iter().filter(|(x, _)| *x > 0f64);
    for _ in 0..900 {
        let (_, index) = updates.next().unwrap();
        encoder.flip_indexed_bit(*index);
    }
    dbg!(Encoder::BIT_LEN);
    log_hd(&encoder, &decoder, &patches);
    for _ in 0..10 {
        let (_, index) = updates.next().unwrap();
        encoder.flip_indexed_bit(*index);
    }
    log_hd(&encoder, &decoder, &patches);
}
