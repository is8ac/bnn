use crate::bits::{BitArray, BitArrayOPs, BitWord, Distance};
use crate::cluster::{CentroidCountPerImage, ImagePatchLloyds};
use crate::descend::DescendMod2;
use crate::float::Noise;
use crate::image2d::{AvgPool, BitPool, Concat, Image2D};
use crate::shape::{Element, Shape};
use bincode::{deserialize_from, serialize_into};
use rand::distributions;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::f64;
use std::fs::create_dir_all;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub trait Apply<Input, Patch, Preprocessor, Output> {
    fn apply(&self, input: &Input) -> Output;
}

#[derive(Copy, Clone, Debug)]
pub struct TrainParams {
    pub lloyds_seed: u64,
    pub k: usize,
    pub lloyds_iters: usize,

    pub weights_init_seed: u64,

    pub minibatch_shuff_seed: u64,
    pub descend_minibatch_max: usize,
    pub descend_minibatch_threshold: usize,

    pub descend_rate: f64,

    pub aux_seed: u64,
    pub aux_sdev: f32,
}

pub trait Layer<InputImage, PatchShape, Preprocessor, O: BitArray, OutputImage, C: Shape> {
    type WeightsType;
    fn gen(
        examples: &Vec<(InputImage, usize)>,
        hyper_params: TrainParams,
    ) -> (Vec<(OutputImage, usize)>, Self::WeightsType);
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
            + serde::Serialize
            + BitArray
            + BitArrayOPs
            + Element<O::BitShape>
            + ImagePatchLloyds<InputImage, PatchShape, Preprocessor>
            + CentroidCountPerImage<InputImage, PatchShape, Preprocessor, [(); C]>,
        O: BitArray + Sync + Send + BitWord,
        OutputImage: Send + Sync + Image2D,
        const C: usize,
    > Layer<InputImage, PatchShape, Preprocessor, O, OutputImage, [(); C]> for I
where
    bool: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    f32: Element<O::BitShape>,
    <I as Element<O::BitShape>>::Array: Copy + Sync,
    <u32 as Element<I::BitShape>>::Array: Sync + Default,
    <I as Element<O::BitShape>>::Array: Apply<InputImage, PatchShape, Preprocessor, OutputImage>
        + Sync
        + DescendMod2<I, O, [(); C]>,
    [<f32 as Element<O::BitShape>>::Array; C]: Sync + Noise,
    distributions::Standard: distributions::Distribution<I>
        + distributions::Distribution<<I as Element<O::BitShape>>::Array>,
    for<'de> I: serde::Deserialize<'de>,
    for<'de> <I as Element<O::BitShape>>::Array: serde::Deserialize<'de>,
    <I as Element<O::BitShape>>::Array: serde::Serialize,
    InputImage::PixelType: Element<PatchShape>,
{
    type WeightsType = (
        <I as Element<O::BitShape>>::Array,
        [<f32 as Element<O::BitShape>>::Array; C],
    );
    fn gen(
        examples: &Vec<(InputImage, usize)>,
        params: TrainParams,
    ) -> (Vec<(OutputImage, usize)>, Self::WeightsType) {
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
            "{}, wseed:{}, aux_seed:{}, aux_sdev:{}, shuffseed:{}, minibatch({}-{}):{}.mutations",
            std::any::type_name::<O>(),
            params.weights_init_seed,
            params.aux_seed,
            params.aux_sdev,
            params.minibatch_shuff_seed,
            params.descend_minibatch_max,
            params.descend_minibatch_threshold,
            params.descend_rate,
        ));
        let (layer_weights, aux_weights): Self::WeightsType = if let Some(weights_file) =
            File::open(&weights_path).ok()
        {
            println!("loading {:?} from disk", &weights_path);
            let weights =
                deserialize_from(weights_file).expect("can't deserialize weights from file");

            let aux_weights = <[<f32 as Element<<O as BitArray>::BitShape>>::Array; C]>::noise(
                &mut Hc128Rng::seed_from_u64(params.aux_seed),
                params.aux_sdev,
            );
            (weights, aux_weights)
        } else {
            let counts_path = counts_dir.join("counts");
            let (centroids, image_patch_bags): (Vec<I>, Vec<(Vec<(u16, u32)>, usize)>) =
                if let Some(counts_file) = File::open(&counts_path).ok() {
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

                    let patch_bags =
                        <I as CentroidCountPerImage<
                            InputImage,
                            PatchShape,
                            Preprocessor,
                            [(); C],
                        >>::centroid_count_per_image(&examples, &centroids);
                    serialize_into(
                        File::create(&counts_path).unwrap(),
                        &(centroids.clone(), patch_bags.clone()),
                    )
                    .unwrap();
                    (centroids, patch_bags)
                };

            let mut weights: <I as Element<<O as BitArray>::BitShape>>::Array =
                Hc128Rng::seed_from_u64(params.weights_init_seed).gen();

            let aux_weights = <[<f32 as Element<<O as BitArray>::BitShape>>::Array; C]>::noise(
                &mut Hc128Rng::seed_from_u64(params.aux_seed),
                params.aux_sdev,
            );

            <<I as Element<<O as BitArray>::BitShape>>::Array as DescendMod2<
                I,
                O,
                [(); C],
            >>::descend(
                &mut weights,
                &mut Hc128Rng::seed_from_u64(params.minibatch_shuff_seed),
                &aux_weights,
                &centroids,
                &image_patch_bags,
                params.descend_minibatch_max,
                params.descend_minibatch_threshold,
                params.descend_rate,
            );
            dbg!("done training");

            serialize_into(File::create(&weights_path).unwrap(), &weights).unwrap();
            (weights, aux_weights)
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
        (new_examples, (layer_weights, aux_weights))
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
