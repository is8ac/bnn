use crate::bits::{BitArray, BitMul, Classify};
use crate::count::{CountBits, CountBitsConv};
use crate::image2d::{Conv2D, Image2D, NCorrectConv};
use crate::shape::Element;
use crate::weight::{GenParamClasses, GenParamSet, InputBits};
use rayon::prelude::*;
use std::boxed::Box;
use std::time::{Duration, Instant};

pub trait FC<I, O, C>
where
    Self: Sized,
{
    fn apply(name: &str, examples: &Vec<(I, usize)>) -> (Self, Vec<(O, usize)>);
}

impl<
        T: BitMul<O, Input = I> + Sized + Send + Sync,
        I: BitArray
            + GenParamClasses<[(); C]>
            + GenParamSet<O, [(); C]>
            + Send
            + Sync
            + CountBits<{ C }>
            + InputBits<O, TrinaryWeights = T>,
        O: BitArray + Sync + Send,
        const C: usize,
    > FC<I, O, [(); C]> for T
where
    <(I::WordType, I::WordType) as Element<I::WordShape>>::Array: Send + Sync,
    u32: Element<I::BitShape> + Element<O::BitShape>,
    (I::WordType, I::WordType): Element<I::WordShape>,
    [(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ); C]: Classify<Input = I, ClassesShape = [(); C]>,
    bool: Element<I::BitShape> + Element<O::BitShape>,
    <u32 as Element<I::BitShape>>::Array: Element<I::BitShape>,
{
    fn apply(name: &str, examples: &Vec<(I, usize)>) -> (Self, Vec<(O, usize)>) {
        let (value_counters, matrix_counters, n_examples): (
            Box<[(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C]>,
            Box<<<u32 as Element<I::BitShape>>::Array as Element<I::BitShape>>::Array>,
            usize,
        ) = I::count_bits(&examples);
        let layer_weights = <I as GenParamSet<O, [(); C]>>::gen_parm_set(
            n_examples,
            &value_counters,
            &matrix_counters,
        );
        let class_weights = <I as GenParamClasses<[(); C]>>::gen_parm_classes(
            n_examples,
            &value_counters,
            &matrix_counters,
        );
        let n_correct: u64 = examples
            .par_iter()
            .map(|(image, class)| (class_weights.max_class(image) == *class) as u64)
            .sum();
        println!("acc: {}%", (n_correct as f64 / n_examples as f64) * 100f64);
        let new_examples: Vec<_> = examples
            .par_iter()
            .map(|(image, class)| (layer_weights.bit_mul(image), *class))
            .collect();
        (layer_weights, new_examples)
    }
}

pub trait Conv2D3x3<Image: Image2D, Patch, O: Element<Image::ImageShape>, C>
where
    Self: Sized,
{
    fn apply(
        name: &str,
        examples: &Vec<(Image, usize)>,
    ) -> (Self, Vec<(<O as Element<Image::ImageShape>>::Array, usize)>);
}

impl<
        Image: Image2D + Send + Sync,
        Patch: BitArray
            + InputBits<OP>
            + CountBitsConv<Image, { C }>
            + GenParamClasses<[(); C]>
            + GenParamSet<OP, [(); C]>,
        OP: Element<Image::ImageShape> + BitArray,
        const C: usize,
    > Conv2D3x3<Image, Patch, OP, [(); C]> for <Patch as InputBits<OP>>::TrinaryWeights
where
    Self: Conv2D<Image, OP>,
    <OP as Element<Image::ImageShape>>::Array: Send + Sync,
    bool: Element<OP::BitShape> + Element<Patch::BitShape>,
    u32: Element<OP::BitShape> + Element<Patch::BitShape>,
    <u32 as Element<Patch::BitShape>>::Array: Element<Patch::BitShape>,
    (Patch::WordType, Patch::WordType): Element<Patch::WordShape>,
    [(
        <(Patch::WordType, Patch::WordType) as Element<Patch::WordShape>>::Array,
        u32,
    ); C]: NCorrectConv<Image>,
    <Patch as InputBits<OP>>::TrinaryWeights: Send + Sync,
    <(Patch::WordType, Patch::WordType) as Element<Patch::WordShape>>::Array: Send + Sync,
{
    fn apply(
        name: &str,
        examples: &Vec<(Image, usize)>,
    ) -> (
        Self,
        Vec<(<OP as Element<Image::ImageShape>>::Array, usize)>,
    ) {
        let total_start = Instant::now();
        println!(
            "Training {} on {} images",
            std::any::type_name::<Self>(),
            examples.len(),
        );
        let start = Instant::now();
        let (value_counters, matrix_counters, n_examples) = Patch::count_bits_conv(&examples);
        let count_time = start.elapsed();
        let start = Instant::now();
        let layer_weights = <Patch as GenParamSet<OP, [(); C]>>::gen_parm_set(
            n_examples,
            &value_counters,
            &matrix_counters,
        );
        let weights_time = start.elapsed();
        let class_weights = <Patch as GenParamClasses<[(); C]>>::gen_parm_classes(
            n_examples,
            &value_counters,
            &matrix_counters,
        );
        let (n_examples, n_correct): (usize, u64) = examples
            .par_iter()
            .fold(
                || (0usize, 0u64),
                |acc, (image, class)| {
                    let (n_examples, n_correct) = class_weights.n_correct(image, *class);
                    (acc.0 + n_examples, acc.1 + n_correct)
                },
            )
            .reduce(|| (0usize, 0u64), |a, b| (a.0 + b.0, a.1 + b.1));
        println!("acc: {}%", (n_correct as f64 / n_examples as f64) * 100f64);

        let start = Instant::now();
        let new_examples: Vec<_> = examples
            .par_iter()
            .map(|(image, class)| {
                (
                    <Self as Conv2D<Image, OP>>::conv2d(&layer_weights, image),
                    *class,
                )
            })
            .collect();
        let apply_time = start.elapsed();
        let total_time = total_start.elapsed();
        println!(
            "count: {:?}, weights: {:?}, apply: {:?}, total: {:?}",
            count_time, weights_time, apply_time, total_time
        );
        (layer_weights, new_examples)
    }
}
