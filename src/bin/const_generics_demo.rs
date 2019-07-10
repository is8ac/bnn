#![feature(const_generics)]
use bitnn::datasets::mnist;
use bitnn::ExtractPatches;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;

macro_rules! patch_3x3 {
    ($input:expr, $x:expr, $y:expr) => {
        [
            [$input[$x + 0][$y + 0], $input[$x + 0][$y + 1], $input[$x + 0][$y + 2]],
            [$input[$x + 1][$y + 0], $input[$x + 1][$y + 1], $input[$x + 1][$y + 2]],
            [$input[$x + 2][$y + 0], $input[$x + 2][$y + 1], $input[$x + 2][$y + 2]],
        ]
    };
}

macro_rules! patch_5x6 {
    ($input:expr, $x:expr, $y:expr) => {
        [
            [$input[$x + 0][$y + 0], $input[$x + 0][$y + 1], $input[$x + 0][$y + 2], $input[$x + 0][$y + 3], $input[$x + 0][$y + 4]],
            [$input[$x + 1][$y + 0], $input[$x + 1][$y + 1], $input[$x + 1][$y + 2], $input[$x + 1][$y + 3], $input[$x + 1][$y + 4]],
            [$input[$x + 2][$y + 0], $input[$x + 2][$y + 1], $input[$x + 2][$y + 2], $input[$x + 2][$y + 3], $input[$x + 2][$y + 4]],
            [$input[$x + 3][$y + 0], $input[$x + 3][$y + 1], $input[$x + 3][$y + 2], $input[$x + 3][$y + 3], $input[$x + 3][$y + 4]],
            [$input[$x + 4][$y + 0], $input[$x + 4][$y + 1], $input[$x + 4][$y + 2], $input[$x + 4][$y + 3], $input[$x + 4][$y + 4]],
        ]
    };
}

pub trait ElementwiseAdd {
    fn elementwise_add(&mut self, other: &Self);
}
impl ElementwiseAdd for u16 {
    fn elementwise_add(&mut self, other: &u16) {
        *self += other;
    }
}

impl ElementwiseAdd for u32 {
    fn elementwise_add(&mut self, other: &u32) {
        *self += other;
    }
}

impl<A: ElementwiseAdd, B: ElementwiseAdd> ElementwiseAdd for (A, B) {
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
    }
}

impl<T: ElementwiseAdd, const L: usize> ElementwiseAdd for [T; L] {
    fn elementwise_add(&mut self, other: &[T; L]) {
        for i in 0..L {
            self[i].elementwise_add(&other[i]);
        }
    }
}

trait WeightsMatrix<Input> {
    type Weights;
}

impl<Input, T: WeightsMatrix<Input>, const L: usize> WeightsMatrix<Input> for [T; L] {
    type Weights = [T::Weights; L];
}

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

pub trait BitMul<I, O> {
    fn bit_mul(&self, input: &I) -> O;
}

pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
}

trait NestedIndex {
    type IndexType;
    fn flip_bit_index(&mut self, index: &Self::IndexType);
}

//impl<T: NestedIndex> NestedIndex for (T, u32) {
//    type IndexType = T::IndexType;
//    fn flip_bit_index(&mut self, index: &Self::IndexType) {
//        self.0.flip_bit_index(index);
//    }
//}

trait IncrementCounters
where
    Self: NestedIndex,
{
    type BitCounterType;
    fn increment_counters(&self, counters: &mut Self::BitCounterType);
    fn top1_index(&self, a: &Self::BitCounterType, b: &Self::BitCounterType) -> (Option<Self::IndexType>, u16);
}

trait NewWeights {
    fn rand_weights<RNG: rand::Rng>(rng: &mut RNG) -> Self;
}

impl<T: BitLen + NewWeights> NewWeights for (T, u32) {
    fn rand_weights<RNG: rand::Rng>(rng: &mut RNG) -> Self {
        (T::rand_weights(rng), T::BIT_LEN as u32 / 2)
    }
}

impl<T: NewWeights, const L: usize> NewWeights for [T; L]
where
    Self: Default,
{
    fn rand_weights<RNG: rand::Rng>(rng: &mut RNG) -> Self {
        let mut target = Self::default();
        for i in 0..L {
            target[i] = T::rand_weights(rng);
        }
        target
    }
}

pub trait GetBit {
    fn bit(&self, i: usize) -> bool;
}

macro_rules! impl_bitlen_for_uint {
    ($type:ty, $len:expr) => {
        impl GetBit for $type {
            #[inline(always)]
            fn bit(&self, i: usize) -> bool {
                ((self >> i) & 1) == 1
            }
        }
        impl NestedIndex for $type {
            type IndexType = usize;
            fn flip_bit_index(&mut self, &index: &Self::IndexType) {
                *self ^= 1 << index;
            }
        }
        impl<Input> WeightsMatrix<Input> for $type {
            type Weights = [(Input, u32); $len];
        }

        impl NewWeights for $type {
            fn rand_weights<RNG: rand::Rng>(rng: &mut RNG) -> Self {
                rng.gen()
            }
        }
        impl IncrementCounters for $type {
            type BitCounterType = [u16; <$type>::BIT_LEN];
            fn increment_counters(&self, counters: &mut Self::BitCounterType) {
                for b in 0..<$type>::BIT_LEN {
                    counters[b] += ((self >> b) & 1) as u16
                }
            }
            fn top1_index(&self, a: &Self::BitCounterType, b: &Self::BitCounterType) -> (Option<Self::IndexType>, u16) {
                let mut max_diff = 0u16;
                let mut index = None;
                for i in 0..$len {
                    let grad_sign = a[i] <= b[i];
                    let bit_sign = self.bit(i);
                    // if the current sign is not the same as the gradient, then we can update.
                    if bit_sign ^ grad_sign {
                        let diff = a[i].saturating_sub(b[i]) | b[i].saturating_sub(a[i]);
                        if diff > max_diff {
                            max_diff = diff;
                            index = Some(i);
                        }
                    }
                }
                (index, max_diff)
            }
        }
        impl BitLen for $type {
            const BIT_LEN: usize = $len;
        }
        impl<I: HammingDistance + BitLen> BitMul<I, $type> for [(I, u32); $len] {
            fn bit_mul(&self, input: &I) -> $type {
                let mut target = <$type>::default();
                for i in 0..$len {
                    target |= ((self[i].0.hamming_distance(input) > self[i].1) as $type) << i;
                }
                target
            }
        }

        impl<I: HammingDistance + BitLen> BitMul<I, $type> for [I; $len] {
            fn bit_mul(&self, input: &I) -> $type {
                let mut target = <$type>::default();
                for i in 0..$len {
                    target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2)) as $type) << i;
                }
                target
            }
        }
        impl HammingDistance for $type {
            #[inline(always)]
            fn hamming_distance(&self, other: &$type) -> u32 {
                (self ^ other).count_ones()
            }
        }
        impl<Input: IncrementCounters + BitLen + HammingDistance> IncrementMatrixCounters<Input, $type> for [(Input, u32); $len] {
            type MatrixBitCounterType = [((Input::BitCounterType, Input::BitCounterType), (u16, u16)); $len];
            // No grad is:
            //      _____
            //     |
            // ____|
            // current is:
            //       _____
            //      /
            // ____/
            // where the width is adjustable.
            fn backprop(
                &self,
                counters: &mut Self::MatrixBitCounterType,
                input: &Input,
                input_counters_0: &mut Input::BitCounterType,
                input_counters_1: &mut Input::BitCounterType,
                target: &$type,
                tanh_width: u32,
            ) {
                for b in 0..<$type>::BIT_LEN {
                    let activation = self[b].0.hamming_distance(&input);
                    let threshold = self[b].1;
                    // this patch only gets to vote if it is within tanh_width.
                    let diff = activation.saturating_sub(threshold) | threshold.saturating_sub(activation);
                    if diff < tanh_width {
                        if target.bit(b) {
                            (counters[b].1).0 += 1;
                            self[b].0.increment_counters(input_counters_0);
                            input.increment_counters(&mut (counters[b].0).0);
                        } else {
                            (counters[b].1).1 += 1;
                            self[b].0.increment_counters(input_counters_1);
                            input.increment_counters(&mut (counters[b].0).1);
                        }
                    }
                }
            }
            fn lesser_backprop(&self, matrix_counters: &mut Self::MatrixBitCounterType, input: &Input, &target_index: &usize, target: &$type, tanh_width: u32) {
                let activation = self[target_index].0.hamming_distance(&input);
                let threshold = self[target_index].1;
                // this patch only gets to vote if it is within tanh_width.
                let diff = activation.saturating_sub(threshold) | threshold.saturating_sub(activation);
                if diff < tanh_width {
                    if target.bit(target_index) {
                        (matrix_counters[target_index].1).0 += 1;
                        input.increment_counters(&mut (matrix_counters[target_index].0).0);
                    } else {
                        (matrix_counters[target_index].1).1 += 1;
                        input.increment_counters(&mut (matrix_counters[target_index].0).1);
                    }
                }
            }
            type WeightsMutationType = (Input::IndexType, usize);
            fn weights_top1_index(&self, counters: &Self::MatrixBitCounterType) -> (Option<Self::WeightsMutationType>, u16) {
                let mut max_diff = 0u16;
                let mut index = None;
                for i in 0..$len {
                    let (sub_index, diff) = self[i].0.top1_index(&(counters[i].0).0, &(counters[i].0).1);
                    if diff > max_diff {
                        max_diff = diff;
                        index = sub_index.map(|x| (x, i));
                    }
                }
                (index, max_diff)
            }
            fn apply_wights_update(&mut self, (input_index, target_index): &Self::WeightsMutationType) {
                self[*target_index].0.flip_bit_index(input_index);
            }

            type BiasMutationType = (usize, bool);
            fn bias_top1_index(counters: &Self::MatrixBitCounterType) -> (Option<Self::BiasMutationType>, u16) {
                let mut max_diff = 0u16;
                let mut mutation = None;
                for i in 0..$len {
                    let diff = (counters[i].1).0.saturating_sub((counters[i].1).1) | (counters[i].1).1.saturating_sub((counters[i].1).0);
                    if diff > max_diff {
                        max_diff = diff;
                        let sub_sign = (counters[i].1).0 < (counters[i].1).1;
                        mutation = Some((i, sub_sign));
                    }
                }
                (mutation, max_diff)
            }
            fn apply_bias_update(&mut self, (index, sign): &Self::BiasMutationType) {
                if *sign {
                    self[*index].1 += 1;
                } else {
                    self[*index].1 -= 1;
                }
            }
        }
    };
}

impl_bitlen_for_uint!(u8, 8);
impl_bitlen_for_uint!(u16, 16);
impl_bitlen_for_uint!(u32, 32);
impl_bitlen_for_uint!(u64, 64);
impl_bitlen_for_uint!(u128, 128);

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
array_bit_len!(25);
array_bit_len!(32);

impl<T: NestedIndex, const L: usize> NestedIndex for [T; L] {
    type IndexType = (T::IndexType, usize);
    fn flip_bit_index(&mut self, (sub_index, i): &Self::IndexType) {
        self[*i].flip_bit_index(sub_index);
    }
}

impl<T: IncrementCounters, const L: usize> IncrementCounters for [T; L]
where
    Self: Default,
{
    type BitCounterType = [T::BitCounterType; L];
    fn increment_counters(&self, counters: &mut [T::BitCounterType; L]) {
        for i in 0..L {
            self[i].increment_counters(&mut counters[i]);
        }
    }
    fn top1_index(&self, a: &Self::BitCounterType, b: &Self::BitCounterType) -> (Option<Self::IndexType>, u16) {
        let mut max_diff = 0u16;
        let mut index = None;
        for i in 0..L {
            let (sub_index, diff) = self[i].top1_index(&a[i], &b[i]);
            if diff > max_diff {
                max_diff = diff;
                index = sub_index.map(|x| (x, i));
            }
        }
        (index, max_diff)
    }
}

impl<T: HammingDistance, const L: usize> HammingDistance for [T; L] {
    fn hamming_distance(&self, other: &[T; L]) -> u32 {
        let mut distance = 0u32;
        for i in 0..L {
            distance += self[i].hamming_distance(&other[i]);
        }
        distance
    }
}

trait IncrementMatrixCounters<Input: IncrementCounters, Target: NestedIndex> {
    type MatrixBitCounterType;
    fn backprop(
        &self,
        matrix_counters: &mut Self::MatrixBitCounterType,
        input: &Input,
        input_counters_0: &mut Input::BitCounterType,
        input_counters_1: &mut Input::BitCounterType,
        target: &Target,
        tanh_width: u32,
    );
    fn lesser_backprop(&self, matrix_counters: &mut Self::MatrixBitCounterType, input: &Input, target_index: &Target::IndexType, target: &Target, tanh_width: u32);

    type WeightsMutationType;
    fn weights_top1_index(&self, counters: &Self::MatrixBitCounterType) -> (Option<Self::WeightsMutationType>, u16);
    fn apply_wights_update(&mut self, index: &Self::WeightsMutationType);

    type BiasMutationType;
    fn bias_top1_index(counters: &Self::MatrixBitCounterType) -> (Option<Self::BiasMutationType>, u16);
    fn apply_bias_update(&mut self, index: &Self::BiasMutationType);
}

impl<Input: IncrementCounters, Target: NestedIndex, MatrixBits: IncrementMatrixCounters<Input, Target> + Copy + Default, const L: usize> IncrementMatrixCounters<Input, [Target; L]>
    for [MatrixBits; L]
{
    type MatrixBitCounterType = [MatrixBits::MatrixBitCounterType; L];
    fn backprop(
        &self,
        counters: &mut Self::MatrixBitCounterType,
        input: &Input,
        input_counters_0: &mut Input::BitCounterType,
        input_counters_1: &mut Input::BitCounterType,
        target: &[Target; L],
        tanh_width: u32,
    ) {
        for i in 0..L {
            self[i].backprop(&mut counters[i], input, input_counters_0, input_counters_1, &target[i], tanh_width);
        }
    }
    fn lesser_backprop(&self, matrix_counters: &mut Self::MatrixBitCounterType, input: &Input, (sub_index, target_index): &(Target::IndexType, usize), target: &[Target; L], tanh_width: u32) {
        self[*target_index].lesser_backprop(&mut matrix_counters[*target_index], input, &sub_index, &target[*target_index], tanh_width);
    }
    type WeightsMutationType = (MatrixBits::WeightsMutationType, usize);
    fn weights_top1_index(&self, counters: &Self::MatrixBitCounterType) -> (Option<Self::WeightsMutationType>, u16) {
        let mut max_diff = 0u16;
        let mut index = None;
        for i in 0..L {
            let (sub_index, diff) = self[i].weights_top1_index(&counters[i]);
            if diff > max_diff {
                max_diff = diff;
                index = sub_index.map(|x| (x, i));
            }
        }
        (index, max_diff)
    }
    fn apply_wights_update(&mut self, (sub_index, index): &Self::WeightsMutationType) {
        self[*index].apply_wights_update(sub_index);
    }
    type BiasMutationType = (MatrixBits::BiasMutationType, usize);
    fn bias_top1_index(counters: &Self::MatrixBitCounterType) -> (Option<Self::BiasMutationType>, u16) {
        let mut max_diff = 0u16;
        let mut index = None;
        for i in 0..L {
            let (sub_index, diff) = MatrixBits::bias_top1_index(&counters[i]);
            if diff > max_diff {
                max_diff = diff;
                index = sub_index.map(|x| (x, i));
            }
        }
        (index, max_diff)
    }
    fn apply_bias_update(&mut self, (sub_index, index): &Self::BiasMutationType) {
        self[*index].apply_bias_update(sub_index);
    }
}

impl<I, O: Default + Copy, T: BitMul<I, O>, const L: usize> BitMul<I, [O; L]> for [T; L]
where
    [O; L]: Default,
{
    fn bit_mul(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_mul(input);
        }
        target
    }
}

trait Conv2D<I, O> {
    fn conv2d(&self, input: &I) -> O;
}

impl<I: Copy, O: Default + Copy, W: BitMul<[[I; 3]; 3], O>, const X: usize, const Y: usize> Conv2D<[[I; Y]; X], [[O; Y]; X]> for W
where
    [[O; Y]; X]: Default,
{
    fn conv2d(&self, input: &[[I; Y]; X]) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                target[x + 1][y + 1] = self.bit_mul(&patch_3x3!(input, x, y));
            }
        }
        target
    }
}

trait AutoencoderSample<Embedding>
where
    Embedding: WeightsMatrix<Self> + Sized + IncrementCounters,
    Self: WeightsMatrix<Embedding> + Sized + IncrementCounters,
    <Embedding as WeightsMatrix<Self>>::Weights: IncrementMatrixCounters<Self, Embedding> + BitMul<Self, Embedding>,
    <Self as WeightsMatrix<Embedding>>::Weights: IncrementMatrixCounters<Embedding, Self> + BitMul<Embedding, Self>,
{
    fn update_counters(
        &self,
        encoder: &<Embedding as WeightsMatrix<Self>>::Weights,
        encoder_counter: &mut <<Embedding as WeightsMatrix<Self>>::Weights as IncrementMatrixCounters<Self, Embedding>>::MatrixBitCounterType,
        decoder: &<Self as WeightsMatrix<Embedding>>::Weights,
        decoder_counter: &mut <<Self as WeightsMatrix<Embedding>>::Weights as IncrementMatrixCounters<Embedding, Self>>::MatrixBitCounterType,
        tanh_width: u32,
    ) -> bool;
}

impl<Input: IncrementCounters + HammingDistance, Embedding: IncrementCounters> AutoencoderSample<Embedding> for Input
where
    Self: WeightsMatrix<Embedding> + Sized,
    Embedding: WeightsMatrix<Self> + Sized,
    <Self as WeightsMatrix<Embedding>>::Weights: IncrementMatrixCounters<Embedding, Self> + BitMul<Embedding, Self>,
    <Embedding as WeightsMatrix<Self>>::Weights: IncrementMatrixCounters<Self, Embedding> + BitMul<Input, Embedding>,
    Embedding::BitCounterType: Default,
{
    fn update_counters(
        &self,
        encoder: &<Embedding as WeightsMatrix<Self>>::Weights,
        encoder_counter: &mut <<Embedding as WeightsMatrix<Self>>::Weights as IncrementMatrixCounters<Self, Embedding>>::MatrixBitCounterType,
        decoder: &<Self as WeightsMatrix<Embedding>>::Weights,
        decoder_counter: &mut <<Self as WeightsMatrix<Embedding>>::Weights as IncrementMatrixCounters<Embedding, Self>>::MatrixBitCounterType,
        tanh_width: u32,
    ) -> bool {
        let embedding = encoder.bit_mul(self);
        let mut embedding_counters_0 = Embedding::BitCounterType::default();
        let mut embedding_counters_1 = Embedding::BitCounterType::default();
        decoder.backprop(decoder_counter, &embedding, &mut embedding_counters_0, &mut embedding_counters_1, self, tanh_width);
        let (embedding_index, _) = embedding.top1_index(&embedding_counters_0, &embedding_counters_1);
        if let Some(index) = embedding_index {
            encoder.lesser_backprop(encoder_counter, self, &index, &embedding, tanh_width);
            true
        } else {
            false
        }
    }
}

trait AutoencoderMinibatch<Example>
where
    Example: WeightsMatrix<Self> + Sized,
    Self: WeightsMatrix<Example> + Sized,
{
    fn minibatch(encoder: &mut <Self as WeightsMatrix<Example>>::Weights, decoder: &mut <Example as WeightsMatrix<Self>>::Weights, examples: &[Example], tanh_width: u32) -> u64;
    fn epoch(encoder: &mut <Self as WeightsMatrix<Example>>::Weights, decoder: &mut <Example as WeightsMatrix<Self>>::Weights, examples: &[Example], minibatch_size: usize, tanh_width: u32) {
        let mut chunks = examples.chunks_exact(minibatch_size);
        let mut update_frac = 1f64;
        let mut n_minibatches = 0;
        for chunk in chunks {
            n_minibatches += 1;
            let n_updates = <Self as AutoencoderMinibatch<Example>>::minibatch(encoder, decoder, chunk, tanh_width);
            update_frac = n_updates as f64 / minibatch_size as f64;
        }
        dbg!(n_minibatches);
    }
}

impl<Example: Send + Sync + WeightsMatrix<Embedding> + IncrementCounters + HammingDistance, Embedding: WeightsMatrix<Example> + Sized + IncrementCounters> AutoencoderMinibatch<Example> for Embedding
where
    <Embedding as WeightsMatrix<Example>>::Weights: IncrementMatrixCounters<Example, Embedding> + BitMul<Example, Embedding> + Sync,
    <Example as WeightsMatrix<Embedding>>::Weights: IncrementMatrixCounters<Embedding, Example> + BitMul<Embedding, Example> + Sync,
    <<Embedding as WeightsMatrix<Example>>::Weights as IncrementMatrixCounters<Example, Embedding>>::MatrixBitCounterType: Default + Send + ElementwiseAdd,
    <<Example as WeightsMatrix<Embedding>>::Weights as IncrementMatrixCounters<Embedding, Example>>::MatrixBitCounterType: Default + Send + ElementwiseAdd,
    Embedding::BitCounterType: Default,
{
    fn minibatch(encoder: &mut <Embedding as WeightsMatrix<Example>>::Weights, decoder: &mut <Example as WeightsMatrix<Embedding>>::Weights, examples: &[Example], tanh_width: u32) -> u64 {
        let (updates, (encoder_counters, decoder_counters)) = examples
            .par_iter()
            .fold(
                || {
                    (
                        0u64,
                        (
                            <<Embedding as WeightsMatrix<Example>>::Weights as IncrementMatrixCounters<Example, Embedding>>::MatrixBitCounterType::default(),
                            <<Example as WeightsMatrix<Embedding>>::Weights as IncrementMatrixCounters<Embedding, Example>>::MatrixBitCounterType::default(),
                        ),
                    )
                },
                |mut counters, example| {
                    let is_update = <Example as AutoencoderSample<Embedding>>::update_counters(&example, &encoder, &mut (counters.1).0, &decoder, &mut (counters.1).1, tanh_width);
                    counters.0 += is_update as u64;
                    counters
                },
            )
            .reduce(
                || {
                    (
                        0u64,
                        (
                            <<Embedding as WeightsMatrix<Example>>::Weights as IncrementMatrixCounters<Example, Embedding>>::MatrixBitCounterType::default(),
                            <<Example as WeightsMatrix<Embedding>>::Weights as IncrementMatrixCounters<Embedding, Example>>::MatrixBitCounterType::default(),
                        ),
                    )
                },
                |mut a, b| {
                    a.0 += b.0;
                    (a.1).0.elementwise_add(&(b.1).0);
                    (a.1).1.elementwise_add(&(b.1).1);
                    a
                },
            );
        if let (Some(index), _) = encoder.weights_top1_index(&encoder_counters) {
            encoder.apply_wights_update(&index);
        }
        //if let (Some(index), val) = <Embedding as WeightsMatrix<Example>>::Weights::bias_top1_index(&encoder_counters) {
        //    encoder.apply_bias_update(&index);
        //}
        if let (Some(index), _) = decoder.weights_top1_index(&decoder_counters) {
            decoder.apply_wights_update(&index);
        }
        //if let (Some(index), val) = <Example as WeightsMatrix<Embedding>>::Weights::bias_top1_index(&decoder_counters) {
        //    decoder.apply_bias_update(&index);
        //}
        updates
    }
}

macro_rules! avg_hd {
    ($examples:expr, $encoder:expr, $decoder:expr) => {{
        let sum_hd: u32 = $examples
            .par_iter()
            .map(|example| {
                let embedding = $encoder.bit_mul(&example);
                let output = $decoder.bit_mul(&embedding);
                example.hamming_distance(&output)
            })
            .sum();
        sum_hd as f64 / $examples.len() as f64
    }};
}

const EMBEDDING_LEN: usize = 4;
//type InputPatchType = [[u8; 3]; 3];
type InputPatchType = [u32; 25];
type EmbeddingType = [u32; EMBEDDING_LEN];
type EncoderType = <EmbeddingType as WeightsMatrix<InputPatchType>>::Weights;
type DecoderType = <InputPatchType as WeightsMatrix<EmbeddingType>>::Weights;

const N_EXAMPLES: usize = 60_000;
const N_STAGES: usize = 6;
//const MINIBATCH_SIZE: usize = 512;
// 6.64 2m8
// 6.09 3m34
// 5.36 6m34
// 4.9
// 7.09 0m30
// 6.9 0m38
// 5.97 0.25

// full: 5.915 1m10

// fc: 105.7

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    rayon::ThreadPoolBuilder::new().stack_size(2usize.pow(30)).num_threads(4).build_global().unwrap();

    //let images = mnist::load_images_u8_unary(Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), N_EXAMPLES);

    //let mut examples: Vec<InputPatchType> = images.par_iter().map(|image| image.patches()).flatten().collect();
    let mut examples = mnist::load_images_bitpacked_u32(Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), N_EXAMPLES);
    examples.shuffle(&mut rng);
    //let mut encoder = <[[[[u8; 3]; 3]; 32]; EMBEDDING_LEN]>::rand_weights(&mut rng);
    //let mut decoder = <[[[[u32; EMBEDDING_LEN]; 8]; 3]; 3]>::rand_weights(&mut rng);

    let mut encoder = EncoderType::rand_weights(&mut rng);
    let mut decoder = DecoderType::rand_weights(&mut rng);

    dbg!(avg_hd!(examples, encoder, decoder));
    let stage_size = examples.len() / N_STAGES;
    for (i, t) in (0..N_STAGES).rev().enumerate() {
        let minibatch_size = 2usize.pow(i as u32 + 7);
        dbg!(minibatch_size);
        let tanh_width = t as u32 + 2;
        dbg!(tanh_width);
        for _ in 0..12 {
            <EmbeddingType as AutoencoderMinibatch<InputPatchType>>::epoch(&mut encoder, &mut decoder, &examples, minibatch_size, tanh_width);
            //<EmbeddingType as AutoencoderMinibatch<InputPatchType>>::epoch(&mut encoder, &mut decoder, &examples[(i * stage_size)..((i + 1) * stage_size)], minibatch_size, tanh_width);
            dbg!(avg_hd!(examples, encoder, decoder));
        }
    }
    //let classes = mnist::load_labels(
    //    Path::new("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"),
    //    N_EXAMPLES,
    //);
    //let image_index = classes
    //    .iter()
    //    .enumerate()
    //    .find(|(_, c)| **c == 0)
    //    .unwrap()
    //    .0;
    //let image2: [[[u32; EMBEDDING_LEN]; 28]; 28] = encoder.conv2d(&images[image_index]);

    //for w in 0..EMBEDDING_LEN {
    //    for b in 0..32 {
    //        for x in 0..3 {
    //            for y in 0..3 {
    //                print!("{:08b} ", encoder[w][b].0[x][y]);
    //            }
    //            print!("\n");
    //        }
    //        dbg!(b);
    //        for x in 0..3 {
    //            for y in 0..3 {
    //                for wb in 0..8 {
    //                    print!("{}", if decoder[x][y][wb].0[w].bit(b) { 1 } else { 0 });
    //                }
    //                print!(" ");
    //            }
    //            print!("\n");
    //        }

    //        for row in &image2 {
    //            for pixel in row {
    //                print!("{}", if pixel[w].bit(b) { 1 } else { 0 });
    //            }
    //            print!("\n");
    //        }
    //    }
    //}
}

// only one bit of embedding has gradient
// only incrament one output of the encoder matrix counters.
// Backprop on decoder return a bit and an index.
// Then encoder matrix update is only for one output

// for each example in minibatch
// value -> encoder_matrix -> example_embedding
// example_embedding -> decoder_matrix -> value_activations
// filter value activations to close to center
// outputs that are close to center incrament minibatch decoder matrix counters _and_ example_embedding counters
// Take top mag example embedding index and use it to incrament encoder matrix counters.
// This looses information, but allows us to move slowly, allowing small minibatches.
