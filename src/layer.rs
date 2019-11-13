use crate::bits::{BitArray, BitMul, Classify};
use crate::count::{Counters, IncrementCounters};
use crate::image2d::{AvgPool, BitPool, Conv2D, Image2D, NCorrectConv, Poolable};
use crate::shape::{Element, Flatten, Merge, Shape, ZipMap};
use crate::weight::{gen_partitions, GenWeights, InputBits};
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

pub trait Apply<Example, Patch, I> {
    type Output;
    fn apply(&self, input: &Example) -> Self::Output;
}

/// CountBits turns a bit Vec of `Example`s into a fixed size counters.
/// For each `Example`, we extract 'Patch's from it, normalise to 'Self' and use them to increment counters.
pub trait CountBits<Example: IncrementCounters<Patch, Self, { C }>, Patch, const C: usize>
where
    Self: BitArray,
    u32: Element<Self::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<Self::BitShape>,
{
    fn count_bits(
        examples: &Vec<(Example, usize)>,
    ) -> (
        Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>,
        Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>,
        usize,
    );
}

impl<
        Example: Hash + IncrementCounters<Patch, T, { C }> + Send + Sync,
        Patch,
        T,
        const C: usize,
    > CountBits<Example, Patch, { C }> for T
where
    T: BitArray,
    u32: Element<T::BitShape>,
    bool: Element<T::BitShape>,
    <u32 as Element<T::BitShape>>::Array: Element<T::BitShape>,
    Box<[(usize, <u32 as Element<T::BitShape>>::Array); C]>: Default + Sync + Send + Counters,
    Box<<<u32 as Element<T::BitShape>>::Array as Element<T::BitShape>>::Array>:
        Default + Send + Sync + Counters,
    for<'de> (
        Box<[(usize, <u32 as Element<T::BitShape>>::Array); C]>,
        Box<<<u32 as Element<T::BitShape>>::Array as Element<T::BitShape>>::Array>,
        usize,
    ): serde::Deserialize<'de>,
    (
        Box<[(usize, <u32 as Element<T::BitShape>>::Array); C]>,
        Box<<<u32 as Element<T::BitShape>>::Array as Element<T::BitShape>>::Array>,
        usize,
    ): serde::Serialize,
{
    fn count_bits(
        examples: &Vec<(Example, usize)>,
    ) -> (
        Box<[(usize, <u32 as Element<T::BitShape>>::Array); C]>,
        Box<<<u32 as Element<T::BitShape>>::Array as Element<T::BitShape>>::Array>,
        usize,
    ) {
        let input_hash = {
            let mut s = DefaultHasher::new();
            examples.hash(&mut s);
            s.finish()
        };
        let count_path = format!("params/{}.count", input_hash);
        let count_path = &Path::new(&count_path);
        create_dir_all("params").unwrap();

        if let Some(counts_file) = File::open(&count_path).ok() {
            println!("reading counts from disk");
            deserialize_from(counts_file).expect("can't deserialize from file")
        } else {
            let counts = examples
                .par_chunks(examples.len() / num_cpus::get_physical())
                .map(|chunk| {
                    chunk.iter().fold(
                        (
                            Box::<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>::default(),
                            Box::<
                                <<u32 as Element<Self::BitShape>>::Array as Element<
                                    Self::BitShape,
                                >>::Array,
                            >::default(),
                            0usize,
                        ),
                        |mut acc, (image, class)| {
                            image.increment_counters(*class, &mut acc.0, &mut acc.1, &mut acc.2);
                            acc
                        },
                    )
                })
                .reduce(
                    || {
                        (
                            Box::<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>::default(),
                            Box::<
                                <<u32 as Element<Self::BitShape>>::Array as Element<
                                    Self::BitShape,
                                >>::Array,
                            >::default(),
                            0usize,
                        )
                    },
                    |mut a, b| {
                        (a.0).elementwise_add(&b.0);
                        (a.1).elementwise_add(&b.1);
                        (a.2).elementwise_add(&b.2);
                        a
                    },
                );
            serialize_into(File::create(&count_path).unwrap(), &counts).unwrap();
            counts
        }
    }
}

pub trait Layer<Example, Patch, I, C>
where
    Self: Sized + Apply<Example, Patch, I>,
{
    fn gen(examples: &Vec<(Example, usize)>) -> (Self, Vec<(Self::Output, usize)>);
}

impl<
        Example: Send + Sync + Hash + IncrementCounters<Patch, I, { C }>,
        Patch,
        I: BitArray + CountBits<Example, Patch, { C }> + GenWeights + Copy,
        T: BitMul<Input = I> + Send + Sync + serde::Serialize + Apply<Example, Patch, I>,
        const C: usize,
    > Layer<Example, Patch, I, [(); C]> for T
where
    for<'de> T: serde::Deserialize<'de>,
    T::Target: BitArray,
    T::Output: Send + Sync,
    (I::WordType, I::WordType): Element<I::WordShape>,
    bool: Element<I::BitShape> + Element<I::BitShape>,
    u32: Element<I::BitShape> + Element<I::BitShape>,
    <u32 as Element<I::BitShape>>::Array: Element<I::BitShape> + Send + Sync,
    Box<[(usize, <u32 as Element<I::BitShape>>::Array); 2]>: Default,
    Box<<<u32 as Element<I::BitShape>>::Array as Element<I::BitShape>>::Array>: Send + Sync,
    (usize, <u32 as Element<I::BitShape>>::Array): Counters,
    <T::Target as BitArray>::BitShape: Flatten<(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    )>,
    (
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ): Element<<T::Target as BitArray>::BitShape, Array = T> + Send + Sync + Copy,
{
    fn gen(examples: &Vec<(Example, usize)>) -> (Self, Vec<(Self::Output, usize)>) {
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
                "[cache] acc: {:.8}% {}",
                acc * 100f64,
                std::any::type_name::<Self>()
            );
            layer_weights
        } else {
            let start = Instant::now();
            let (value_counters, matrix_counters, n_examples) = I::count_bits(&examples);
            let count_time = start.elapsed();
            let start = Instant::now();
            let layer_weights = {
                let mut partitions = gen_partitions(C);
                partitions.sort_by_key(|x| x.len().min(C - x.len()));
                let partitions = &partitions[0..<T::Target as BitArray>::BitShape::N];
                let weights: Vec<_> = partitions
                    .par_iter()
                    .map(|partition| {
                        let mut split_counters =
                            Box::<[(usize, <u32 as Element<I::BitShape>>::Array); 2]>::default();
                        for (class, class_counter) in value_counters.iter().enumerate() {
                            split_counters[partition.contains(&class) as usize]
                                .elementwise_add(class_counter);
                        }
                        <I as GenWeights>::gen_weights(
                            &split_counters,
                            &matrix_counters,
                            n_examples,
                        )
                    })
                    .collect();
                <T::Target as BitArray>::BitShape::from_vec(&weights)
            };
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
                    <Self as Apply<Example, Patch, I>>::apply(&layer_weights, image),
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

pub trait AvgPoolLayer<Image: AvgPool>
where
    Image::Pooled: Send + Sync,
{
    fn avg_pool(examples: &Vec<(Image, usize)>) -> Vec<(Image::Pooled, usize)>;
}

impl<Image: AvgPool + Send + Sync> AvgPoolLayer<Image> for ()
where
    Image::Pooled: Send + Sync,
{
    fn avg_pool(examples: &Vec<(Image, usize)>) -> Vec<(Image::Pooled, usize)> {
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
