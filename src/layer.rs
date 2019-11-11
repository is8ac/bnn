use crate::bits::{BitArray, BitMul, Classify};
use crate::count::{CountBits, CountBitsConv};
use crate::image2d::{AvgPool, BitPool, Conv2D, Image2D, NCorrectConv, Poolable};
use crate::shape::{Element, Merge, Shape, ZipMap};
use crate::weight::{GenParamClasses, GenParamSet, InputBits};
use rayon::prelude::*;
use std::boxed::Box;
use std::time::Instant;

use bincode::{deserialize_from, serialize_into};
use std::collections::hash_map::DefaultHasher;
use std::fs::create_dir_all;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::io::Read;
use std::path::Path;

pub trait FC<C>
where
    Self: Sized + BitMul,
{
    fn apply(examples: &Vec<(Self::Input, usize)>) -> (Self, Vec<(Self::Target, usize)>);
}

impl<T: BitMul + Sized + Send + Sync, const C: usize> FC<[(); C]> for T
where
    T::Input: BitArray
        + GenParamClasses<[(); C]>
        + GenParamSet<T::Target, [(); C]>
        + Send
        + Sync
        + CountBits<{ C }>
        + InputBits<T::Target, TrinaryWeights = T>,
    T::Target: BitArray + Sync + Send,
    <(
        <T::Input as BitArray>::WordType,
        <T::Input as BitArray>::WordType,
    ) as Element<<T::Input as BitArray>::WordShape>>::Array: Send + Sync,
    u32: Element<<T::Input as BitArray>::BitShape>
        + Element<<T::Input as BitArray>::BitShape>
        + Element<<T::Target as BitArray>::BitShape>,
    bool: Element<<T::Input as BitArray>::BitShape>
        + Element<<T::Input as BitArray>::BitShape>
        + Element<<<T as BitMul>::Target as BitArray>::BitShape>,
    (
        <T::Input as BitArray>::WordType,
        <T::Input as BitArray>::WordType,
    ): Element<<T::Input as BitArray>::WordShape>,
    [(
        <(
            <T::Input as BitArray>::WordType,
            <T::Input as BitArray>::WordType,
        ) as Element<<T::Input as BitArray>::WordShape>>::Array,
        u32,
    ); C]: Classify<Input = T::Input, ClassesShape = [(); C]>,
    <u32 as Element<<T::Input as BitArray>::BitShape>>::Array:
        Element<<T::Input as BitArray>::BitShape>,
{
    fn apply(examples: &Vec<(T::Input, usize)>) -> (Self, Vec<(T::Target, usize)>) {
        let (value_counters, matrix_counters, n_examples) = T::Input::count_bits(&examples);
        let layer_weights = <T::Input as GenParamSet<T::Target, [(); C]>>::gen_parm_set(
            n_examples,
            &value_counters,
            &matrix_counters,
        );
        let class_weights = <T::Input as GenParamClasses<[(); C]>>::gen_parm_classes(
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

pub trait Conv2D3x3<Image: Image2D, C>
where
    Self: Sized + BitMul,
    <Self as BitMul>::Target: Element<<Image as Image2D>::ImageShape>,
{
    fn apply(
        examples: &Vec<(Image, usize)>,
    ) -> (
        Self,
        Vec<(<Self::Target as Element<Image::ImageShape>>::Array, usize)>,
    );
}

impl<
        T: BitMul + Conv2D<Image> + Send + Sync + serde::Serialize,
        Image: Image2D + Send + Sync + Hash,
        const C: usize,
    > Conv2D3x3<Image, [(); C]> for T
where
    <T as BitMul>::Input: BitArray
        + CountBitsConv<Image, { C }>
        + GenParamClasses<[(); C]>
        + GenParamSet<T::Target, [(); C]>
        + InputBits<T::Target, TrinaryWeights = T>,
    <T::Target as Element<Image::ImageShape>>::Array: Send + Sync,
    bool: Element<<T::Target as BitArray>::BitShape> + Element<<T::Input as BitArray>::BitShape>,
    u32: Element<<T::Target as BitArray>::BitShape> + Element<<T::Input as BitArray>::BitShape>,
    <u32 as Element<<T::Input as BitArray>::BitShape>>::Array:
        Element<<T::Input as BitArray>::BitShape>,
    (
        <T::Input as BitArray>::WordType,
        <T::Input as BitArray>::WordType,
    ): Element<<T::Input as BitArray>::WordShape>,
    [(
        <(
            <T::Input as BitArray>::WordType,
            <T::Input as BitArray>::WordType,
        ) as Element<<T::Input as BitArray>::WordShape>>::Array,
        u32,
    ); C]: NCorrectConv<Image>,
    <(
        <T::Input as BitArray>::WordType,
        <T::Input as BitArray>::WordType,
    ) as Element<<T::Input as BitArray>::WordShape>>::Array: Send + Sync,
    for<'de> T: serde::Deserialize<'de>,
    <T as BitMul>::Target: Element<<Image as Image2D>::ImageShape>,
    <<T as BitMul>::Target as Element<<Image as Image2D>::ImageShape>>::Array: Send,
    <T as BitMul>::Target: BitArray + Element<Image::ImageShape>,
{
    fn apply(
        examples: &Vec<(Image, usize)>,
    ) -> (
        Self,
        Vec<(<T::Target as Element<Image::ImageShape>>::Array, usize)>,
    ) {
        let total_start = Instant::now();

        let input_hash = {
            let mut s = DefaultHasher::new();
            examples.hash(&mut s);
            s.finish()
        };
        let dataset_path = format!("params/{}", input_hash);
        let dataset_path = &Path::new(&dataset_path);
        create_dir_all(&dataset_path).unwrap();

        let weights_path = dataset_path.join(std::any::type_name::<Self>());
        let layer_weights = if let Some(weights_file) = File::open(&weights_path).ok() {
            //println!("reading {} from disk", std::any::type_name::<Self>());
            let layer_weights =
                deserialize_from(weights_file).expect("can't deserialize from file");
            let acc: f64 = {
                let mut acc_file = File::open(dataset_path.join("acc")).unwrap();
                let mut acc_string = String::new();
                acc_file.read_to_string(&mut acc_string).unwrap();
                acc_string.parse().unwrap()
            };
            println!(
                "[cache] acc: {:.8}% {}  {}",
                acc * 100f64,
                std::any::type_name::<Image::ImageShape>(),
                std::any::type_name::<Self>()
            );
            layer_weights
        } else {
            let start = Instant::now();
            let (value_counters, matrix_counters, n_examples) =
                T::Input::count_bits_conv(&examples);
            let count_time = start.elapsed();
            {
                let class_weights = <T::Input as GenParamClasses<[(); C]>>::gen_parm_classes(
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
                let mut acc_file = File::create(dataset_path.join("acc")).unwrap();
                write!(&mut acc_file, "{}", n_correct as f64 / n_examples as f64).unwrap();
            }
            let start = Instant::now();
            let layer_weights =
                <<T as BitMul>::Input as GenParamSet<T::Target, [(); C]>>::gen_parm_set(
                    n_examples,
                    &value_counters,
                    &matrix_counters,
                );
            let weights_time = start.elapsed();
            let total_time = total_start.elapsed();
            println!(
                "count: {:?}, weights: {:?}, total: {:?}",
                count_time, weights_time, total_time
            );
            serialize_into(File::create(&weights_path).unwrap(), &layer_weights).unwrap();
            layer_weights
        };
        let new_examples: Vec<_> = examples
            .par_iter()
            .map(|(image, class)| {
                (
                    <Self as Conv2D<Image>>::conv2d(&layer_weights, image),
                    *class,
                )
            })
            .collect();
        (layer_weights, new_examples)
    }
}

pub trait BitPoolLayer<P: Element<Self>, O: Merge<P, P>>
where
    Self: Shape + Sized + Poolable,
    Self::Pooled: Shape,
    O: Element<Self::Pooled>,
{
    fn bit_pool(
        examples: &Vec<(<P as Element<Self>>::Array, usize)>,
    ) -> Vec<(<O as Element<Self::Pooled>>::Array, usize)>;
}

impl<S: Shape + Poolable, P: Element<S> + Element<S::Pooled> + Copy, O: Merge<P, P>>
    BitPoolLayer<P, O> for S
where
    <P as Element<S>>::Array: Image2D<ImageShape = S, PixelType = P>,
    S::Pooled: Shape + ZipMap<P, P, O>,
    O: Element<S::Pooled>,
    <P as Element<S>>::Array: BitPool + Send + Sync,
    <O as Element<S::Pooled>>::Array: Send + Sync,
{
    fn bit_pool(
        examples: &Vec<(<P as Element<S>>::Array, usize)>,
    ) -> Vec<(<O as Element<<S as Poolable>::Pooled>>::Array, usize)> {
        examples
            .par_iter()
            .map(|(image, class)| {
                (
                    <Self::Pooled as ZipMap<P, P, O>>::zip_map(
                        &image.or_pool(),
                        &image.and_pool(),
                        |a, b| O::merge(a, b),
                    ),
                    *class,
                )
            })
            .collect()
    }
}

pub trait AvgPoolLayer<P: Element<Self> + Element<Self::Pooled>>
where
    Self: Poolable + Shape + Sized,
    Self::Pooled: Shape,
{
    fn avg_pool(
        examples: &Vec<(<P as Element<Self>>::Array, usize)>,
    ) -> Vec<(<P as Element<Self::Pooled>>::Array, usize)>;
}

impl<P: Element<Self> + Element<Self::Pooled>, S: Poolable + Shape + Sized> AvgPoolLayer<P> for S
where
    Self: Poolable + Shape + Sized,
    Self::Pooled: Shape,
    <P as Element<Self>>::Array: AvgPool + Image2D<ImageShape = Self, PixelType = P> + Send + Sync,
    <P as Element<Self::Pooled>>::Array: Send + Sync,
{
    fn avg_pool(
        examples: &Vec<(<P as Element<Self>>::Array, usize)>,
    ) -> Vec<(<P as Element<Self::Pooled>>::Array, usize)> {
        examples
            .par_iter()
            .map(|(image, class)| (image.avg_pool(), *class))
            .collect()
    }
}

pub trait ConcatLayer<A: Element<Self>, B: Element<Self>, O: Element<Self>>
where
    Self: Shape + Sized,
{
    fn concat(
        examples_a: &Vec<(<A as Element<Self>>::Array, usize)>,
        examples_b: &Vec<(<B as Element<Self>>::Array, usize)>,
    ) -> Vec<(<O as Element<Self>>::Array, usize)>;
}

impl<
        A: Element<Self>,
        B: Element<Self>,
        O: Element<Self> + Merge<A, B>,
        S: Shape + Sized + ZipMap<A, B, O>,
    > ConcatLayer<A, B, O> for S
where
    Self: Shape + Sized,
    <A as Element<S>>::Array: Sync + Send,
    <B as Element<S>>::Array: Sync + Send,
    <O as Element<S>>::Array: Sync + Send,
{
    fn concat(
        examples_a: &Vec<(<A as Element<Self>>::Array, usize)>,
        examples_b: &Vec<(<B as Element<Self>>::Array, usize)>,
    ) -> Vec<(<O as Element<Self>>::Array, usize)> {
        examples_a
            .par_iter()
            .zip(examples_b.par_iter())
            .map(|(a, b)| {
                assert_eq!(a.1, b.1);
                (
                    <S as ZipMap<A, B, O>>::zip_map(&a.0, &b.0, |a, b| O::merge(a, b)),
                    a.1,
                )
            })
            .collect()
    }
}
