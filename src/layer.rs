use crate::bits::{BitArray, BitArrayOPs, BitMul, BitWord, Distance, IndexedFlipBit};
use crate::cluster::{ImageCountByCentroids, ImagePatchLloyds};
use crate::image2d::{AvgPool, BitPool, Concat, Image2D};
use crate::shape::{Element, Shape};
use crate::weight::{decend, FloatObj, Noise};
use bincode::{deserialize_from, serialize_into};
use rand::distributions;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::fs::create_dir_all;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::Instant;

pub trait Apply<Example, Patch, Preprocessor, Output> {
    fn apply(&self, input: &Example) -> Output;
}

#[derive(Copy, Clone, Debug)]
pub struct TrainParams {
    pub lloyds_seed: u64,
    pub k: usize,
    pub lloyds_iters: usize,
    pub weights_seed: u64,
    pub decend_window_thresh: usize,
    pub noise_sdev: f32,
}

pub trait Layer<InputImage, PatchShape, Preprocessor, O: BitArray, OutputImage, C: Shape>
where
    Self: Element<O::BitShape>,
    O: Element<C>,
    f32: Element<O::BitShape>,
    <f32 as Element<O::BitShape>>::Array: Element<C>,
{
    fn gen(
        examples: &Vec<(InputImage, usize)>,
        hyper_params: TrainParams,
    ) -> (
        Vec<(OutputImage, usize)>,
        <Self as Element<O::BitShape>>::Array,
        <<f32 as Element<O::BitShape>>::Array as Element<C>>::Array,
    );
}

impl<
        InputImage: Send + Sync + Hash + Image2D,
        PatchShape: Shape,
        Preprocessor,
        I: Copy
            + Sync
            + Send
            + BitWord
            + Distance
            + BitArray
            + BitArrayOPs
            + serde::Serialize
            + Element<O::BitShape>
            + ImagePatchLloyds<InputImage, PatchShape, Preprocessor>
            + ImageCountByCentroids<InputImage, PatchShape, Preprocessor, [(); C]>,
        O: BitArray + Sync + Send + BitWord,
        OutputImage: Send + Sync + Image2D,
        const C: usize,
    > Layer<InputImage, PatchShape, Preprocessor, O, OutputImage, [(); C]> for I
where
    distributions::Standard: distributions::Distribution<[I; C]>
        + distributions::Distribution<I>
        + distributions::Distribution<[<f32 as Element<O::BitShape>>::Array; C]>
        + distributions::Distribution<<I as Element<O::BitShape>>::Array>,
    <I as Element<O::BitShape>>::Array: Apply<InputImage, PatchShape, Preprocessor, OutputImage>
        + Sync
        + Apply<InputImage, PatchShape, Preprocessor, OutputImage>
        + BitMul<I, O>
        + IndexedFlipBit<I, O>,
    for<'de> I: serde::Deserialize<'de>,
    for<'de> [u32; C]: serde::Deserialize<'de>,
    for<'de> (
        <I as Element<O::BitShape>>::Array,
        [<f32 as Element<O::BitShape>>::Array; C],
    ): serde::Deserialize<'de>,
    (
        <I as Element<O::BitShape>>::Array,
        [<f32 as Element<O::BitShape>>::Array; C],
    ): serde::Serialize,
    bool: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    <u32 as Element<I::BitShape>>::Array: Sync + Default,
    <I as Element<O::BitShape>>::Array: Copy + Sync,
    [<f32 as Element<O::BitShape>>::Array; C]:
        FloatObj<O, [(); C]> + Copy + Noise + std::fmt::Debug,
    [u32; C]: Default + serde::Serialize,
    InputImage::PixelType: Element<PatchShape>,
    f32: Element<O::BitShape>,
    <f32 as Element<O::BitShape>>::Array: Sync,
{
    fn gen(
        examples: &Vec<(InputImage, usize)>,
        params: TrainParams,
    ) -> (
        Vec<(OutputImage, usize)>,
        <I as Element<O::BitShape>>::Array,
        [<f32 as Element<O::BitShape>>::Array; C],
    ) {
        let total_start = Instant::now();
        let input_hash = {
            let mut s = DefaultHasher::new();
            examples.hash(&mut s);
            s.finish()
        };
        let dataset_path = format!("params/{}", input_hash);
        let dataset_path = &Path::new(&dataset_path);

        let counts_dir = dataset_path.join(format!(
            "counts:{}, n:{}, seed:{}, k{}, li{}",
            std::any::type_name::<I>(),
            examples.len(),
            params.lloyds_seed,
            params.k,
            params.lloyds_iters,
        ));

        create_dir_all(&counts_dir).unwrap();
        let weights_path = counts_dir.join(format!(
            "{}, seed:{}, sdev:{}, wt:{}.weights",
            std::any::type_name::<O>(),
            params.weights_seed,
            params.decend_window_thresh,
            params.noise_sdev,
        ));
        let (layer_weights, aux_weights) =
            if let Some(weights_file) = File::open(&weights_path).ok() {
                println!("loading {:?} from disk", &weights_path);
                deserialize_from(weights_file).expect("can't deserialize from file")
            } else {
                let counts_path = counts_dir.join("counts");
                let counts = if let Some(counts_file) = File::open(&counts_path).ok() {
                    println!("loading {:?} from disk", &counts_path);
                    deserialize_from(counts_file).expect("can't deserialize counts from file")
                } else {
                    let mut rng = Hc128Rng::seed_from_u64(params.lloyds_seed);
                    let centroids =
                        <I as ImagePatchLloyds<InputImage, PatchShape, Preprocessor>>::lloyds(
                            &mut rng,
                            &examples,
                            params.k,
                            params.lloyds_iters,
                        );

                    let counts = <I as ImageCountByCentroids<
                        InputImage,
                        PatchShape,
                        Preprocessor,
                        [(); C],
                    >>::count_by_centroids(&examples, &centroids);
                    serialize_into(File::create(&counts_path).unwrap(), &counts).unwrap();
                    counts
                };
                let mut rng = Hc128Rng::seed_from_u64(params.weights_seed);
                let mut layer_weights: <I as Element<<O as BitArray>::BitShape>>::Array = rng.gen();
                //let mut aux_weights: [<f32 as Element<O::BitShape>>::Array; C] = rng.gen();
                let aux_weights =
                    <[<f32 as Element<O::BitShape>>::Array; C]>::noise(&mut rng, params.noise_sdev);
                //dbg!(&aux_weights);

                decend(
                    &mut rng,
                    &mut layer_weights,
                    &aux_weights,
                    &counts,
                    params.k,
                    params.decend_window_thresh,
                );
                serialize_into(
                    File::create(&weights_path).unwrap(),
                    &(layer_weights, aux_weights),
                )
                .unwrap();
                (layer_weights, aux_weights)
            };
        let new_examples: Vec<_> = examples
            .par_iter()
            .map(|(image, class)| {
                (
                    <<I as Element<O::BitShape>>::Array as Apply<
                        InputImage,
                        PatchShape,
                        Preprocessor,
                        OutputImage,
                    >>::apply(&layer_weights, image),
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
