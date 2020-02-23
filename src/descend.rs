use crate::bits::{BitArray, BitMul, BitWord, IncrementFracCounters, IndexedFlipBit};
use crate::count::ElementwiseAdd;
use crate::float::{BFFVMMtanh, FFFVMMtanh, Mutate, SoftMaxLoss, FFFVMM};
use crate::image2d::{Conv2D, Image2D, PixelFold, StaticImage};
use crate::shape::{Element, Map, Shape};
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::f64;
use std::marker::PhantomData;

pub trait DescendMod2<I: Element<O::BitShape>, O: BitArray, C: Shape>
where
    f32: Element<O::BitShape>,
    <f32 as Element<O::BitShape>>::Array: Element<C>,
{
    fn descend<R: Rng>(
        &mut self,
        rng: &mut R,
        aux_weights: &<<f32 as Element<O::BitShape>>::Array as Element<C>>::Array,
        centroids: &Vec<I>,
        examples: &[(Vec<(u32, u32)>, usize)],
        window_size: usize,
        window_thresh: usize,
        rate: f64,
    );
    fn avg_loss(
        &self,
        aux_weights: &<<f32 as Element<O::BitShape>>::Array as Element<C>>::Array,
        centroids: &Vec<I>,
        examples: &[(Vec<(u32, u32)>, usize)],
    ) -> f64;
}

impl<
        I: Element<O::BitShape> + BitWord + Sync,
        O: BitArray + BitWord + IncrementFracCounters + Sync + Send,
        const C: usize,
    > DescendMod2<I, O, [(); C]> for <I as Element<O::BitShape>>::Array
where
    [f32; C]: SoftMaxLoss,
    <I as Element<O::BitShape>>::Array: BitMul<I, O> + Sync + IndexedFlipBit<I, O>,
    f32: Element<O::BitShape>,
    [<f32 as Element<O::BitShape>>::Array; C]:
        FFFVMM<[f32; C], InputType = <f32 as Element<O::BitShape>>::Array> + Sync,
    u32: Element<O::BitShape>,
    (usize, <u32 as Element<<O as BitArray>::BitShape>>::Array): Default,
    O::BitShape: Map<u32, f32>,
{
    fn descend<RNG: Rng>(
        &mut self,
        rng: &mut RNG,
        aux_weights: &[<f32 as Element<O::BitShape>>::Array; C],
        centroids: &Vec<I>,
        examples: &[(Vec<(u32, u32)>, usize)],
        window_size: usize,
        window_thresh: usize,
        rate: f64,
    ) {
        assert!(window_size <= examples.len());
        assert!(rate < 1.0);
        assert!(rate > 0.0);
        if window_size > window_thresh {
            self.descend(
                rng,
                aux_weights,
                centroids,
                examples,
                (window_size as f64 * rate) as usize,
                window_thresh,
                rate,
            );
        }
        let indices = {
            let mut indices: Vec<usize> = (0..I::BIT_LEN)
                .map(|i| (i * (examples.len() - window_size)) / I::BIT_LEN)
                .collect();
            indices.shuffle(rng);
            indices
        };
        for (ib, &index) in indices.iter().enumerate() {
            let minibatch = &examples[index..index + window_size];
            let mut cur_loss = (*self).avg_loss(aux_weights, centroids, minibatch);
            //dbg!(cur_sum_loss);
            for ob in 0..O::BIT_LEN {
                self.indexed_flip_bit(ob, ib);
                let new_loss = (*self).avg_loss(aux_weights, centroids, minibatch);
                if new_loss < cur_loss {
                    cur_loss = new_loss;
                } else {
                    self.indexed_flip_bit(ob, ib);
                }
            }
        }
        let full_loss = (*self).avg_loss(aux_weights, centroids, examples);
        println!("w:{}: l:{}", window_size, full_loss);
    }
    fn avg_loss(
        &self,
        aux_weights: &[<f32 as Element<O::BitShape>>::Array; C],
        centroids: &Vec<I>,
        examples: &[(Vec<(u32, u32)>, usize)],
    ) -> f64 {
        let centroid_acts: Vec<O> = centroids
            .par_iter()
            .map(|input| self.bit_mul(input))
            .collect();

        let sum_image_loss: f64 = examples
            .par_iter()
            .map(|(patch_bag, class)| {
                let act_counts = patch_bag.iter().fold(
                    <(usize, <u32 as Element<<O as BitArray>::BitShape>>::Array)>::default(),
                    |mut bit_count, (patch_index, patch_count)| {
                        centroid_acts[*patch_index as usize]
                            .weighted_increment_frac_counters(*patch_count, &mut bit_count);
                        bit_count
                    },
                );
                let n = act_counts.0 as f32;
                let float_hidden =
                    <<O as BitArray>::BitShape as Map<u32, f32>>::map(&act_counts.1, |&count| {
                        count as f32 / n
                    });
                let class_acts: [f32; C] = aux_weights.fffvmm(&float_hidden);
                class_acts.softmax_loss(*class) as f64
            })
            .sum();
        sum_image_loss / examples.len() as f64
    }
}

pub trait DescendFloat<I, C: Shape> {
    fn float_descend(
        &mut self,
        examples: &Vec<(I, usize)>,
        noise: &[f32],
        cur_loss: &mut f64,
        n_workers: usize,
    ) -> usize;
    fn train<R: Rng>(
        rng: &mut R,
        examples: &Vec<(I, usize)>,
        n_workers: usize,
        n_iters: usize,
        noise_sdev: f32,
        sdev_decay_rate: f32,
    ) -> Self;
}

impl<T: Model<I, C> + Mutate + Default + Send + Sync, I: Sync, const C: usize>
    DescendFloat<I, [(); C]> for T
where
    [f32; C]: SoftMaxLoss,
{
    fn float_descend(
        &mut self,
        examples: &Vec<(I, usize)>,
        noise: &[f32],
        cur_loss: &mut f64,
        n_workers: usize,
    ) -> usize {
        assert_eq!(noise.len(), Self::NOISE_LEN + n_workers);
        let worker_ids: Vec<_> = (0..n_workers).collect();
        let worker_losses: Vec<(usize, f64)> = worker_ids
            .par_iter()
            .map(|&worker_id| {
                let perturbed_model = self.mutate(&noise[worker_id..]);
                let new_loss: f64 = examples
                    .par_iter()
                    .map(|(image, class)| perturbed_model.loss(image, *class) as f64)
                    .sum();
                (worker_id, new_loss)
            })
            .filter(|&(_, l)| l < *cur_loss)
            .collect();

        if let Some(&(best_seed, new_loss)) = worker_losses
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            *cur_loss = new_loss;
            *self = self.mutate(&noise[best_seed..best_seed + Self::NOISE_LEN]);
        } else {
            //println!("miss: {} seeds", worker_losses.len());
        }
        worker_losses.len()
    }
    fn train<R: Rng>(
        rng: &mut R,
        examples: &Vec<(I, usize)>,
        n_workers: usize,
        n_iters: usize,
        noise_sdev: f32,
        sdev_decay_rate: f32,
    ) -> T {
        let mut model = T::default();

        let mut cur_loss = f64::MAX;
        let mut cur_sdev = noise_sdev;
        for i in 0..n_iters {
            cur_sdev *= sdev_decay_rate;
            let normal = Normal::new(0f32, cur_sdev).unwrap();
            let noise: Vec<f32> = (0..Self::NOISE_LEN + n_workers)
                .map(|_| normal.sample(rng))
                .collect();
            let n_good_seeds = model.float_descend(&examples, &noise, &mut cur_loss, n_workers);
            println!(
                "{:3} {:3} sdev:{:.4} loss: {:.5}",
                i,
                n_good_seeds,
                cur_sdev,
                cur_loss / examples.len() as f64
            );
        }
        model
    }
}

pub trait Model<I, const C: usize>
where
    [f32; C]: SoftMaxLoss,
{
    fn apply(&self, input: &I) -> [f32; C];
    fn is_correct(&self, input: &I, class: usize) -> bool {
        let acts = self.apply(input);
        let max_act = acts
            .iter()
            .enumerate()
            .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        max_act == class
    }
    fn loss(&self, input: &I, class: usize) -> f32 {
        let acts = self.apply(input);
        acts.softmax_loss(class)
    }
}

pub struct OneHiddenLayerFCmodel<I, H: Shape, O: Shape>
where
    f32: Element<H>,
    (I, f32): Element<H>,
    (<f32 as Element<H>>::Array, f32): Element<O>,
{
    pub l1: <(I, f32) as Element<H>>::Array,
    pub l2: <(<f32 as Element<H>>::Array, f32) as Element<O>>::Array,
}

impl<I: Element<H>, H: Shape, const C: usize> Default for OneHiddenLayerFCmodel<I, H, [(); C]>
where
    f32: Element<H>,
    (I, f32): Element<H>,
    <(I, f32) as Element<H>>::Array: Default,
    [(<f32 as Element<H>>::Array, f32); C]: Default,
{
    fn default() -> Self {
        Self {
            l1: <(I, f32) as Element<H>>::Array::default(),
            l2: <[(<f32 as Element<H>>::Array, f32); C]>::default(),
        }
    }
}

impl<I, H: Shape, const C: usize> Mutate for OneHiddenLayerFCmodel<I, H, [(); C]>
where
    f32: Element<H>,
    (I, f32): Element<H>,
    <(I, f32) as Element<H>>::Array: Mutate,
    [(<f32 as Element<H>>::Array, f32); C]: Mutate,
{
    const NOISE_LEN: usize = <(I, f32) as Element<H>>::Array::NOISE_LEN
        + <[(<f32 as Element<H>>::Array, f32); C]>::NOISE_LEN;
    fn mutate(&self, noise: &[f32]) -> Self {
        Self {
            l1: self
                .l1
                .mutate(&noise[0..<(I, f32) as Element<H>>::Array::NOISE_LEN]),
            l2: self
                .l2
                .mutate(&noise[<(I, f32) as Element<H>>::Array::NOISE_LEN..Self::NOISE_LEN]),
        }
    }
}

impl<I, H: Shape, const C: usize> Model<I, C> for OneHiddenLayerFCmodel<I, H, [(); C]>
where
    f32: Element<H>,
    (I, f32): Element<H>,
    <(I, f32) as Element<H>>::Array: FFFVMMtanh<<f32 as Element<H>>::Array, InputType = I>,
    [(<f32 as Element<H>>::Array, f32); C]:
        FFFVMM<[f32; C], InputType = <f32 as Element<H>>::Array>,
    [f32; C]: SoftMaxLoss,
{
    fn apply(&self, input: &I) -> [f32; C] {
        let h1 = self.l1.fffvmm_tanh(input);
        self.l2.fffvmm(&h1)
    }
}

pub struct OneHiddenLayerConvPooledModel<I, H: Shape, O: Shape>
where
    f32: Element<H>,
    ([[I; 3]; 3], f32): Element<H>,
    (<f32 as Element<H>>::Array, f32): Element<O>,
{
    pub l1: <([[I; 3]; 3], f32) as Element<H>>::Array,
    pub l2: <(<f32 as Element<H>>::Array, f32) as Element<O>>::Array,
}

impl<I, H: Shape + Map<f32, f32>, const C: usize> Model<StaticImage<I, 32, 32>, C>
    for OneHiddenLayerConvPooledModel<I, H, [(); C]>
where
    f32: Element<H>,
    ([[I; 3]; 3], f32): Element<H>,
    <([[I; 3]; 3], f32) as Element<H>>::Array:
        FFFVMMtanh<<f32 as Element<H>>::Array, InputType = [[I; 3]; 3]>,
    [(<f32 as Element<H>>::Array, f32); C]:
        FFFVMM<[f32; C], InputType = <f32 as Element<H>>::Array>,
    [f32; C]: SoftMaxLoss,
    <f32 as Element<H>>::Array: Default + ElementwiseAdd + std::fmt::Debug,
    StaticImage<<f32 as Element<H>>::Array, 32, 32>: Image2D<PixelType = <f32 as Element<H>>::Array>
        + PixelFold<(usize, <f32 as Element<H>>::Array), [[(); 3]; 3]>,
    StaticImage<I, 32, 32>: Conv2D<
            [[(); 3]; 3],
            <f32 as Element<H>>::Array,
            OutputType = StaticImage<<f32 as Element<H>>::Array, 32, 32>,
        > + Image2D<PixelType = I>,
{
    fn apply(&self, input: &StaticImage<I, 32, 32>) -> [f32; C] {
        let h1: StaticImage<<f32 as Element<H>>::Array, 32, 32> =
            input.conv2d(|patch| self.l1.fffvmm_tanh(patch));
        let pooled = h1.pixel_fold(
            (0usize, <f32 as Element<H>>::Array::default()),
            |mut acc, pixel| {
                acc.0 += 1;
                acc.1.elementwise_add(pixel);
                acc
            },
        );
        let n = pooled.0 as f32;
        let avgs = <H as Map<f32, f32>>::map(&pooled.1, |x| x / n);
        self.l2.fffvmm(&avgs)
    }
}

impl<I, H: Shape, const C: usize> Default for OneHiddenLayerConvPooledModel<I, H, [(); C]>
where
    f32: Element<H>,
    ([[I; 3]; 3], f32): Element<H>,
    <([[I; 3]; 3], f32) as Element<H>>::Array: Default,
    [(<f32 as Element<H>>::Array, f32); C]: Default,
{
    fn default() -> Self {
        Self {
            l1: <([[I; 3]; 3], f32) as Element<H>>::Array::default(),
            l2: <[(<f32 as Element<H>>::Array, f32); C]>::default(),
        }
    }
}

impl<I, H: Shape, const C: usize> Mutate for OneHiddenLayerConvPooledModel<I, H, [(); C]>
where
    f32: Element<H>,
    ([[I; 3]; 3], f32): Element<H>,
    <([[I; 3]; 3], f32) as Element<H>>::Array: Mutate,
    [(<f32 as Element<H>>::Array, f32); C]: Mutate,
{
    const NOISE_LEN: usize = <([[I; 3]; 3], f32) as Element<H>>::Array::NOISE_LEN
        + <[(<f32 as Element<H>>::Array, f32); C]>::NOISE_LEN;
    fn mutate(&self, noise: &[f32]) -> Self {
        Self {
            l1: self
                .l1
                .mutate(&noise[0..<([[I; 3]; 3], f32) as Element<H>>::Array::NOISE_LEN]),
            l2: self.l2.mutate(
                &noise[<([[I; 3]; 3], f32) as Element<H>>::Array::NOISE_LEN..Self::NOISE_LEN],
            ),
        }
    }
}

impl<I, H: Shape + Map<f32, f32>, const C: usize> Model<StaticImage<I, 32, 32>, C>
    for (
        <([[I; 3]; 3], f32) as Element<H>>::Array,
        [(<f32 as Element<H>>::Array, f32); C],
        PhantomData<H>,
    )
where
    f32: Element<H>,
    ([[I; 3]; 3], f32): Element<H>,
    <([[I; 3]; 3], f32) as Element<H>>::Array:
        FFFVMMtanh<<f32 as Element<H>>::Array, InputType = [[I; 3]; 3]>,
    [(<f32 as Element<H>>::Array, f32); C]:
        FFFVMM<[f32; C], InputType = <f32 as Element<H>>::Array>,
    [f32; C]: SoftMaxLoss,
    <f32 as Element<H>>::Array: Default + ElementwiseAdd + std::fmt::Debug,
    StaticImage<<f32 as Element<H>>::Array, 32, 32>: Image2D<PixelType = <f32 as Element<H>>::Array>
        + PixelFold<(usize, <f32 as Element<H>>::Array), [[(); 3]; 3]>,
    StaticImage<I, 32, 32>: Conv2D<
            [[(); 3]; 3],
            <f32 as Element<H>>::Array,
            OutputType = StaticImage<<f32 as Element<H>>::Array, 32, 32>,
        > + Image2D<PixelType = I>,
{
    fn apply(&self, input: &StaticImage<I, 32, 32>) -> [f32; C] {
        let h1: StaticImage<<f32 as Element<H>>::Array, 32, 32> =
            input.conv2d(|patch| self.0.fffvmm_tanh(patch));
        let pooled = h1.pixel_fold(
            (0usize, <f32 as Element<H>>::Array::default()),
            |mut acc, pixel| {
                acc.0 += 1;
                acc.1.elementwise_add(pixel);
                acc
            },
        );
        let n = pooled.0 as f32;
        let avgs = <H as Map<f32, f32>>::map(&pooled.1, |x| x / n);
        self.1.fffvmm(&avgs)
    }
}

pub trait DescendFloatCentroidsPatchBag<I, C: Shape> {
    type LayerType;
    type AuxType;
    fn float_descend(
        model: &mut (Self::LayerType, Self::AuxType),
        centroids: &Vec<I>,
        examples: &[(Vec<(u32, u32)>, usize)],
        noise: &[f32],
        cur_loss: &mut f64,
        n_workers: usize,
    ) -> usize;
    fn train<R: Rng>(
        rng: &mut R,
        centroids: &Vec<I>,
        examples: &[(Vec<(u32, u32)>, usize)],
        n_workers: usize,
        n_iters: usize,
        noise_sdev: f32,
        sdev_decay_rate: f32,
    ) -> (Self::LayerType, Self::AuxType);
}

impl<H: Shape + Map<f32, f32>, I: BitArray + BFFVMMtanh<H> + Sync, const C: usize>
    DescendFloatCentroidsPatchBag<I, [(); C]> for H
where
    [f32; C]: SoftMaxLoss,
    f32: Element<I::BitShape> + Element<H>,
    (<f32 as Element<I::BitShape>>::Array, f32): Element<H>,
    (
        <(<f32 as Element<I::BitShape>>::Array, f32) as Element<H>>::Array,
        [(<f32 as Element<H>>::Array, f32); C],
    ): Mutate + Sync + Default,
    <f32 as Element<H>>::Array: Send + Default + ElementwiseAdd + Sync,
    [(<f32 as Element<H>>::Array, f32); C]:
        FFFVMM<[f32; C], InputType = <f32 as Element<H>>::Array>,
{
    type LayerType = <(<f32 as Element<I::BitShape>>::Array, f32) as Element<H>>::Array;
    type AuxType = [(<f32 as Element<H>>::Array, f32); C];
    fn float_descend(
        model: &mut (Self::LayerType, Self::AuxType),
        centroids: &Vec<I>,
        examples: &[(Vec<(u32, u32)>, usize)],
        noise: &[f32],
        cur_loss: &mut f64,
        n_workers: usize,
    ) -> usize {
        assert_eq!(
            noise.len(),
            <(Self::LayerType, Self::AuxType)>::NOISE_LEN + n_workers
        );
        let worker_ids: Vec<_> = (0..n_workers).collect();
        let worker_losses: Vec<(usize, f64)> = worker_ids
            .par_iter()
            .map(|&worker_id| {
                let perturbed_model = model.mutate(&noise[worker_id..]);
                let acts: Vec<<f32 as Element<H>>::Array> = centroids
                    .par_iter()
                    .map(|patch| patch.bffvmm_tanh(&model.0))
                    .collect();
                let new_loss: f64 = examples
                    .par_iter()
                    .map(|(bag, class)| {
                        let pooled = bag.iter().fold(
                            (0usize, <f32 as Element<H>>::Array::default()),
                            |mut acc, (patch_index, patch_weight)| {
                                acc.0 += 1;
                                acc.1.elementwise_add(&acts[*patch_index as usize]);
                                acc
                            },
                        );
                        let n = pooled.0 as f32;
                        let avgs = <H as Map<f32, f32>>::map(&pooled.1, |x| x / n);
                        let acts = model.1.fffvmm(&avgs);
                        acts.softmax_loss(*class) as f64
                    })
                    .sum();
                (worker_id, new_loss)
            })
            .filter(|&(_, l)| l < *cur_loss)
            .collect();

        if let Some(&(best_seed, new_loss)) = worker_losses
            .iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        {
            *cur_loss = new_loss;
            *model = model.mutate(
                &noise[best_seed..best_seed + <(Self::LayerType, Self::AuxType)>::NOISE_LEN],
            );
        } else {
            //println!("miss: {} seeds", worker_losses.len());
        }
        worker_losses.len()
    }
    fn train<R: Rng>(
        rng: &mut R,
        centroids: &Vec<I>,
        examples: &[(Vec<(u32, u32)>, usize)],
        n_workers: usize,
        n_iters: usize,
        noise_sdev: f32,
        sdev_decay_rate: f32,
    ) -> (Self::LayerType, Self::AuxType) {
        let mut model = <(Self::LayerType, Self::AuxType)>::default();

        let mut cur_loss = f64::MAX;
        let mut cur_sdev = noise_sdev;
        for i in 0..n_iters {
            cur_sdev *= sdev_decay_rate;
            let normal = Normal::new(0f32, cur_sdev).unwrap();
            let noise: Vec<f32> = (0..<(Self::LayerType, Self::AuxType)>::NOISE_LEN + n_workers)
                .map(|_| normal.sample(rng))
                .collect();
            let n_good_seeds = H::float_descend(
                &mut model,
                &centroids,
                &examples,
                &noise,
                &mut cur_loss,
                n_workers,
            );
            println!(
                "{:3} {:3} sdev:{:.4} loss: {:.5}",
                i,
                n_good_seeds,
                cur_sdev,
                cur_loss / examples.len() as f64
            );
        }
        model
    }
}
