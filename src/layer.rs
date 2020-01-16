use crate::bits::BitArray;
use crate::count::{ElementwiseAdd, IncrementCounters};
use crate::image2d::{AvgPool, BitPool, Concat, Image2D};
use crate::shape::{Element, Shape};
use crate::weight::GenWeights;
use rayon::prelude::*;
use std::time::Instant;

use bincode::{deserialize_from, serialize_into};
use std::collections::hash_map::DefaultHasher;
use std::fs::create_dir_all;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub trait Apply<Example, Patch, Preprocessor, Output> {
    fn apply(&self, input: &Example) -> Output;
}

/// CountBits turns a bit Vec of `Example`s into a fixed size counters.
/// For each `Example`, we extract 'Patch's from it, normalise to 'Self' and use them to increment counters.
pub trait CountBits<Example, Patch, Preprocessor, Accumulator> {
    fn count_bits(examples: &Vec<(Example, usize)>) -> Accumulator;
}

impl<
        Example: Hash + Send + Sync + IncrementCounters<Patch, Preprocessor, Accumulator>,
        Preprocessor,
        Accumulator: Send + Sync + Default + ElementwiseAdd,
        Patch,
        T,
    > CountBits<Example, Patch, Preprocessor, Accumulator> for T
where
    T: BitArray,
    u32: Element<T::BitShape>,
    bool: Element<T::BitShape>,
    <u32 as Element<T::BitShape>>::Array: Element<T::BitShape>,
    //for<'de> Accumulator: serde::Deserialize<'de>,
    //Accumulator: serde::Serialize,
{
    fn count_bits(examples: &Vec<(Example, usize)>) -> Accumulator {
        let input_hash = {
            let mut s = DefaultHasher::new();
            examples.hash(&mut s);
            s.finish()
        };
        let dataset_path = format!("params/{}", input_hash);
        let dataset_path = &Path::new(&dataset_path);
        create_dir_all(&dataset_path).unwrap();

        //if let Some(counts_file) = File::open(&count_path).ok() {
        //let count_path = dataset_path.join(std::any::type_name::<Accumulator>());
        //    println!("reading counts from file: {:?}", counts_file);
        //    deserialize_from(counts_file).expect("can't deserialize from file")
        //} else {
        println!("counting {} examples...", examples.len());
        let start = Instant::now();
        let parallelize = false;
        let counts = if parallelize {
            let sub_accs: Vec<Accumulator> = examples
                .par_chunks(examples.len() / num_cpus::get_physical())
                .map(|chunk| {
                    let foo = chunk.iter().fold(
                        Accumulator::default(),
                        |mut accumulator, (image, class)| {
                            image.increment_counters(*class, &mut accumulator);
                            accumulator
                        },
                    );
                    foo
                })
                .collect();
            dbg!("done counting");
            sub_accs.iter().fold(Accumulator::default(), |mut a, b| {
                a.elementwise_add(&b);
                a
            })
        } else {
            examples
                .iter()
                .fold(Accumulator::default(), |mut accumulator, (image, class)| {
                    image.increment_counters(*class, &mut accumulator);
                    accumulator
                })
        };
        let count_time = start.elapsed();
        dbg!(count_time);
        //serialize_into(File::create(&count_path).unwrap(), &counts).unwrap();
        counts
        //}
    }
}

pub trait Layer<Example, Patch, I, Preprocessor, WeightsAlgorithm, Output: Image2D, C: Shape>
where
    Self: Sized,
    Output::PixelType: Element<C>,
{
    fn gen(
        examples: &Vec<(Example, usize)>,
    ) -> (
        Vec<(Output, usize)>,
        Self,
        <Output::PixelType as Element<C>>::Array,
    );
}

impl<
        T: Copy + Send + Sync + Apply<Example, Patch, Preprocessor, Output> + Copy,
        Preprocessor,
        Example: Send + Sync + Hash,
        Patch,
        WeightsAlgorithm: GenWeights<I, Output::PixelType, [(); C]>,
        I: CountBits<Example, Patch, Preprocessor, WeightsAlgorithm::Accumulator>
            + Copy
            + BitArray
            + Sync
            + Element<<<Output as Image2D>::PixelType as BitArray>::BitShape, Array = T>,
        Output: Send + Sync + Image2D,
        const C: usize,
    > Layer<Example, Patch, I, Preprocessor, WeightsAlgorithm, Output, [(); C]> for T
where
    for<'de> (Self, [Output::PixelType; C]): serde::Deserialize<'de>,
    (Self, [Output::PixelType; C]): serde::Serialize,
    bool: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    <Output as Image2D>::PixelType: BitArray,
{
    fn gen(
        examples: &Vec<(Example, usize)>,
    ) -> (Vec<(Output, usize)>, Self, [Output::PixelType; C]) {
        let total_start = Instant::now();
        let input_hash = {
            let mut s = DefaultHasher::new();
            examples.hash(&mut s);
            s.finish()
        };
        let dataset_path = format!("params/{}", input_hash);
        let dataset_path = &Path::new(&dataset_path);
        create_dir_all(&dataset_path).unwrap();

        let weights_path = dataset_path.join(WeightsAlgorithm::string_name());
        let (layer_weights, aux_weights) =
            if let Some(weights_file) = File::open(&weights_path).ok() {
                println!("loading {:?} from disk", &weights_path);
                deserialize_from(weights_file).expect("can't deserialize from file")
            } else {
                println!("training {}", WeightsAlgorithm::string_name());
                let start = Instant::now();
                let accumulator: WeightsAlgorithm::Accumulator = I::count_bits(&examples);
                let count_time = start.elapsed();
                let start = Instant::now();
                let weights = WeightsAlgorithm::gen_weights(&accumulator);
                let weights_time = start.elapsed();
                let total_time = total_start.elapsed();
                println!(
                    "count: {:?}, weights: {:?}, total: {:?}",
                    count_time, weights_time, total_time
                );
                serialize_into(File::create(&weights_path).unwrap(), &weights).unwrap();
                weights
            };
        let new_examples: Vec<_> = examples
            .par_iter()
            .map(|(image, class)| {
                (
                    <Self as Apply<Example, Patch, Preprocessor, Output>>::apply(
                        &layer_weights,
                        image,
                    ),
                    *class,
                )
            })
            .collect();
        (new_examples, layer_weights, aux_weights)
    }
}

pub trait BitPoolLayer<Image: BitPool> {
    fn bit_pool(examples: &Vec<(Image::Input, usize)>) -> Vec<(Image, usize)>;
}

impl<Image: BitPool + Sync + Send> BitPoolLayer<Image> for ()
where
    Image::Input: Sync,
{
    fn bit_pool(examples: &Vec<(Image::Input, usize)>) -> Vec<(Image, usize)> {
        examples
            .par_iter()
            .map(|(image, class)| (Image::andor_pool(image), *class))
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

pub trait ConcatImages<A, B, O> {
    fn concat(examples_a: &Vec<(A, usize)>, examples_b: &Vec<(B, usize)>) -> Vec<(O, usize)>;
}

impl<A: Sync, B: Sync, O: Sync + Send + Concat<A, B>> ConcatImages<A, B, O> for () {
    fn concat(examples_a: &Vec<(A, usize)>, examples_b: &Vec<(B, usize)>) -> Vec<(O, usize)> {
        examples_a
            .par_iter()
            .zip(examples_b.par_iter())
            .map(|(a, b)| {
                assert_eq!(a.1, b.1);
                (O::concat(&a.0, &b.0), a.1)
            })
            .collect()
    }
}

pub trait ClassifyLayer<Example, I, WeightsAlgorithm, C>
where
    Self: Sized,
{
    fn gen_classify(examples: &Vec<(Example, usize)>) -> f64;
}
