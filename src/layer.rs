use crate::bits::{BitArray, BitArrayOPs, BitMap, BitMul, BitWord, Distance, BFBVMM};
use crate::cluster::{CentroidCountPerImage, ImagePatchLloyds, NullCluster};
use crate::descend::{DescendFloat, DescendMod2};
use crate::float::{FFFVMMtanh, Noise};
use crate::image2d::{AvgPool, BitPool, Concat, Conv2D, Image2D, PixelMap, StaticImage};
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
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

pub trait Apply<I, O> {
    fn apply(&self, input: &I) -> O;
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
pub trait ConvLayer<InputImage, PatchShape, O: BitArray, OutputImage, C: Shape> {
    type WeightsType;
    fn gen(
        examples: &Vec<(InputImage, usize)>,
        hyper_params: TrainParams,
    ) -> (Vec<(OutputImage, usize)>, Self::WeightsType);
}

impl<
        InputImage: Send + Sync + Hash + Image2D + Conv2D<PatchShape, O, OutputType = OutputImage>,
        PatchShape: Shape,
        I: Copy
            + Sync
            + Send
            + BitWord
            + Distance
            + serde::Serialize
            + BitArray
            + BitArrayOPs
            + Element<O::BitShape>
            + ImagePatchLloyds<InputImage, PatchShape>
            + CentroidCountPerImage<InputImage, PatchShape, [(); C]>,
        O: BitArray + Sync + Send + BitWord,
        OutputImage: Send + Sync + Image2D,
        const C: usize,
    > ConvLayer<InputImage, PatchShape, O, OutputImage, [(); C]> for I
where
    bool: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    f32: Element<O::BitShape>,
    <I as Element<O::BitShape>>::Array: Copy + Sync,
    <u32 as Element<I::BitShape>>::Array: Sync + Default,
    <I as Element<O::BitShape>>::Array: Sync + DescendMod2<I, O, [(); C]>,
    [<f32 as Element<O::BitShape>>::Array; C]: Sync + Noise,
    distributions::Standard: distributions::Distribution<I>
        + distributions::Distribution<<I as Element<O::BitShape>>::Array>,
    for<'de> I: serde::Deserialize<'de>,
    for<'de> <I as Element<O::BitShape>>::Array: serde::Deserialize<'de>,
    <I as Element<O::BitShape>>::Array: serde::Serialize + BitMul<I, O>,
    InputImage::PixelType: Element<PatchShape>,
    <InputImage as Image2D>::PixelType: Element<PatchShape, Array = I>,
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
            let (centroids, image_patch_bags): (Vec<I>, Vec<(Vec<(u32, u32)>, usize)>) =
                if let Some(counts_file) = File::open(&counts_path).ok() {
                    println!("loading {:?} from disk", &counts_path);
                    deserialize_from(counts_file).expect("can't deserialize counts from file")
                } else {
                    let mut rng = Hc128Rng::seed_from_u64(params.lloyds_seed);
                    let centroids = <I as ImagePatchLloyds<InputImage, PatchShape>>::lloyds(
                        &mut rng,
                        &examples,
                        params.k,
                        params.lloyds_iters,
                    );

                    let patch_bags = <I as CentroidCountPerImage<
                        InputImage,
                        PatchShape,
                        [(); C],
                    >>::centroid_count_per_image(
                        &examples, &centroids
                    );
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

            <<I as Element<<O as BitArray>::BitShape>>::Array as DescendMod2<I, O, [(); C]>>::descend(
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
            .map(|(image, class)| (image.conv2d(|patch| layer_weights.bit_mul(patch)), *class))
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

#[derive(Debug)]
pub struct FullFloatConvLayerParams {
    pub seed: u64,
    pub n_workers: usize,
    pub n_iters: usize,
    pub noise_sdev: f32,
    pub sdev_decay_rate: f32,
}

pub trait FullFloatConvLayer<InputImage: Image2D, PatchShape, H: Shape, OutputImage, C: Shape>
where
    f32: Element<H>,
    (<f32 as Element<H>>::Array, f32): Element<C>,
{
    type WeightsType;
    fn gen(
        examples: &Vec<(InputImage, usize)>,
        dataset_name: &Path,
        hyper_params: &FullFloatConvLayerParams,
    ) -> ((Vec<(OutputImage, usize)>, PathBuf), Self::WeightsType);
}

impl<I, H: Shape, const C: usize>
    FullFloatConvLayer<
        StaticImage<I, 32, 32>,
        [[(); 3]; 3],
        H,
        StaticImage<<f32 as Element<H>>::Array, 32, 32>,
        [(); C],
    > for [[I; 3]; 3]
where
    (
        <([[I; 3]; 3], f32) as Element<H>>::Array,
        [(<f32 as Element<H>>::Array, f32); C],
        PhantomData<H>,
    ): Default + Sync + DescendFloat<StaticImage<I, 32, 32>, [(); C]> + serde::Serialize,
    for<'de> (
        <([[I; 3]; 3], f32) as Element<H>>::Array,
        [(<f32 as Element<H>>::Array, f32); C],
        PhantomData<H>,
    ): serde::Deserialize<'de>,

    f32: Element<H>,
    ([[I; 3]; 3], f32): Element<H>,
    <([[I; 3]; 3], f32) as Element<H>>::Array:
        FFFVMMtanh<<f32 as Element<H>>::Array, InputType = Self>,
    StaticImage<I, 32, 32>: Image2D<PixelType = I>
        + Sync
        + Conv2D<
            [[(); 3]; 3],
            <f32 as Element<H>>::Array,
            OutputType = StaticImage<<f32 as Element<H>>::Array, 32, 32>,
        >,
    <f32 as Element<H>>::Array: Sync,
    StaticImage<<f32 as Element<H>>::Array, 32, 32>:
        Image2D<PixelType = <f32 as Element<H>>::Array> + Sync + Send,
{
    type WeightsType = (
        <([[I; 3]; 3], f32) as Element<H>>::Array,
        [(<f32 as Element<H>>::Array, f32); C],
        PhantomData<H>,
    );
    fn gen(
        examples: &Vec<(StaticImage<I, 32, 32>, usize)>,
        dataset_name: &Path,
        params: &FullFloatConvLayerParams,
    ) -> (
        (
            Vec<(StaticImage<<f32 as Element<H>>::Array, 32, 32>, usize)>,
            PathBuf,
        ),
        Self::WeightsType,
    ) {
        let dataset_name = dataset_name.join(format!(
            "{}, n{}, seed:{}, n_workers:{}, n_iters:{}, noise_sdev:{}, noise_decay_rate:{}.params",
            std::any::type_name::<H>(),
            examples.len(),
            params.seed,
            params.n_workers,
            params.n_iters,
            params.noise_sdev,
            params.sdev_decay_rate,
        ));
        create_dir_all(&dataset_name).unwrap();
        let weights_path = &dataset_name.join("float_weights.prms");
        let weights: Self::WeightsType = if let Some(weights_file) = File::open(&weights_path).ok()
        {
            println!("loading {:?} from disk", &weights_path);
            deserialize_from(weights_file).expect("can't deserialize weights from file")
        } else {
            let mut rng = Hc128Rng::seed_from_u64(0);
            dbg!();
            let weights =
                <Self::WeightsType as DescendFloat<StaticImage<I, 32, 32>, [(); C]>>::train(
                    &mut rng,
                    &examples,
                    params.n_workers,
                    params.n_iters,
                    params.noise_sdev,
                    params.sdev_decay_rate,
                );

            dbg!("done training");

            serialize_into(File::create(&weights_path).unwrap(), &weights).unwrap();
            weights
        };
        let new_examples: Vec<(StaticImage<<f32 as Element<H>>::Array, 32, 32>, usize)> = examples
            .par_iter()
            .map(|(image, class)| (image.conv2d(|patch| weights.0.fffvmm_tanh(patch)), *class))
            .collect();
        ((new_examples, dataset_name), weights)
    }
}

pub trait BFBConvLayer<ImageShape, I, PatchShape, C: Shape> {
    type InputImage;
    type OutputImage;
    type FloatWeightsType;
    type ObjType;
    fn gen(
        examples: &Vec<(Self::InputImage, usize)>,
        hyper_params: &FullFloatConvLayerParams,
    ) -> (
        Vec<(Self::OutputImage, usize)>,
        (Self::FloatWeightsType, Self::ObjType),
    );
}

impl<I: BitArray + BitMap<f32>, O: BitArray, const C: usize>
    BFBConvLayer<StaticImage<(), 32, 32>, I, [[(); 3]; 3], [(); C]> for O
where
    f32: Element<O::BitShape> + Element<I::BitShape>,
    ([[<f32 as Element<I::BitShape>>::Array; 3]; 3], f32): Element<O::BitShape>,
    <([[<f32 as Element<I::BitShape>>::Array; 3]; 3], f32) as Element<O::BitShape>>::Array:
        BFBVMM<[[I; 3]; 3], O> + Sync + Copy,
    [(<f32 as Element<O::BitShape>>::Array, f32); C]: Sync + Copy,
    (
        <([[<f32 as Element<I::BitShape>>::Array; 3]; 3], f32) as Element<O::BitShape>>::Array,
        [(<f32 as Element<O::BitShape>>::Array, f32); C],
        PhantomData<O::BitShape>,
    ): DescendFloat<StaticImage<<f32 as Element<I::BitShape>>::Array, 32, 32>, [(); C]>,

    for<'de> (
        <([[<f32 as Element<I::BitShape>>::Array; 3]; 3], f32) as Element<O::BitShape>>::Array,
        [(<f32 as Element<O::BitShape>>::Array, f32); C],
    ): serde::Deserialize<'de>,
    (
        <([[<f32 as Element<I::BitShape>>::Array; 3]; 3], f32) as Element<O::BitShape>>::Array,
        [(<f32 as Element<O::BitShape>>::Array, f32); C],
    ): serde::Serialize,

    ([[I; 3]; 3], f32): Element<O::BitShape>,
    StaticImage<I, 32, 32>: Sync
        + Conv2D<[[(); 3]; 3], O, OutputType = StaticImage<O, 32, 32>>
        + Image2D<PixelType = I>
        + PixelMap<<f32 as Element<I::BitShape>>::Array>
        + Image2D<ImageShape = StaticImage<(), 32, 32>>
        + Hash,
    StaticImage<O, 32, 32>: Sync + Send,
    <f32 as Element<I::BitShape>>::Array: Element<
        StaticImage<(), 32, 32>,
        Array = StaticImage<<f32 as Element<I::BitShape>>::Array, 32, 32>,
    >,
    StaticImage<<f32 as Element<I::BitShape>>::Array, 32, 32>: Sync + Send,
    <<f32 as Element<I::BitShape>>::Array as Element<StaticImage<(), 32usize, 32usize>>>::Array:
        Send,
    [[I; 3]; 3]: ImagePatchLloyds<StaticImage<I, 32, 32>, [[(); 3]; 3]>
        + CentroidCountPerImage<StaticImage<I, 32, 32>, [[(); 3]; 3], [(); C]>
        + NullCluster<StaticImage<I, 32, 32>, [[(); 3]; 3]>,
    rand::distributions::Standard: rand::distributions::Distribution<[[I; 3]; 3]>,
{
    type InputImage = StaticImage<I, 32, 32>;
    type OutputImage = StaticImage<O, 32, 32>;
    type FloatWeightsType =
        <([[<f32 as Element<I::BitShape>>::Array; 3]; 3], f32) as Element<O::BitShape>>::Array;
    type ObjType = [(<f32 as Element<O::BitShape>>::Array, f32); C];
    fn gen(
        examples: &Vec<(Self::InputImage, usize)>,
        params: &FullFloatConvLayerParams,
    ) -> (
        Vec<(Self::OutputImage, usize)>,
        (Self::FloatWeightsType, Self::ObjType),
    ) {
        let input_hash = {
            let mut s = DefaultHasher::new();
            examples.hash(&mut s);
            s.finish()
        };
        let dataset_path = format!("params/{}", input_hash);
        let dataset_path = &Path::new(&dataset_path);

        let dataset_name = dataset_path.join(format!(
            "{}, n{}, seed:{}, n_workers:{}, n_iters:{}, noise_sdev:{}, noise_decay_rate:{}.params",
            std::any::type_name::<O>(),
            examples.len(),
            params.seed,
            params.n_workers,
            params.n_iters,
            params.noise_sdev,
            params.sdev_decay_rate,
        ));
        create_dir_all(&dataset_name).unwrap();
        let weights_path = &dataset_name.join("float_weights.prms");
        let weights: (Self::FloatWeightsType, Self::ObjType) =
            if let Some(weights_file) = File::open(&weights_path).ok() {
                println!("loading {:?} from disk", &weights_path);
                deserialize_from(weights_file).expect("can't deserialize weights from file")
            } else {
                let mut rng = Hc128Rng::seed_from_u64(0);
                //let centroids =
                //    <[[I; 3]; 3] as ImagePatchLloyds<Self::InputImage, [[(); 3]; 3]>>::lloyds(
                //        &mut rng, &examples, 100, 10,
                //    );
                let centroids =
                    <[[I; 3]; 3] as NullCluster<Self::InputImage, [[(); 3]; 3]>>::null_cluster(
                        &examples,
                    );

                let patch_bags = <[[I; 3]; 3] as CentroidCountPerImage<
                    Self::InputImage,
                    [[(); 3]; 3],
                    [(); C],
                >>::centroid_count_per_image(&examples, &centroids);

                let float_examples: Vec<(
                    StaticImage<<f32 as Element<I::BitShape>>::Array, 32, 32>,
                    usize,
                )> = examples
                    .par_iter()
                    .map(|(image, class)| {
                        (
                            image.pixel_map(|p| p.bit_map(|sign| if sign { -1f32 } else { 1f32 })),
                            *class,
                        )
                    })
                    .collect();
                let mut rng = Hc128Rng::seed_from_u64(0);
                let (weights, obj, _) = <(
                    Self::FloatWeightsType,
                    Self::ObjType,
                    PhantomData<O::BitShape>,
                ) as DescendFloat<
                    StaticImage<<f32 as Element<I::BitShape>>::Array, 32, 32>,
                    [(); C],
                >>::train(
                    &mut rng,
                    &float_examples,
                    params.n_workers,
                    params.n_iters,
                    params.noise_sdev,
                    params.sdev_decay_rate,
                );

                dbg!("done training");

                serialize_into(File::create(&weights_path).unwrap(), &(weights, obj)).unwrap();
                (weights, obj)
            };
        let new_examples: Vec<(StaticImage<O, 32, 32>, usize)> = examples
            .par_iter()
            .map(|(image, class)| (image.conv2d(|patch| weights.0.bfbvmm(patch)), *class))
            .collect();
        (new_examples, weights)
    }
}
