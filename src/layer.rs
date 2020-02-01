use crate::bits::{BitArray, BitArrayOPs, BitWord, Distance, IndexedFlipBit, BFBVM};
use crate::cluster::{ImageCountByCentroids, ImagePatchLloyds};
use crate::float::{FloatLoss, Mutate, Noise};
use crate::image2d::{AvgPool, BitPool, Concat, Image2D};
use crate::shape::{Element, Shape};
use bincode::{deserialize_from, serialize_into};
use rand::distributions;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::f64;
use std::fs::create_dir_all;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::path::Path;

pub trait Apply<Example, Patch, Preprocessor, Output> {
    fn apply(&self, input: &Example) -> Output;
}

#[derive(Copy, Clone, Debug)]
pub struct TrainParams {
    pub lloyds_seed: u64,
    pub k: usize,
    pub lloyds_iters: usize,
    pub weights_seed: u64,
    pub noise_sdev: f32,
    pub decend_iters: usize,
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
            + BitArray
            + BitArrayOPs
            + Element<O::BitShape>
            + ImagePatchLloyds<InputImage, PatchShape, Preprocessor>
            + ImageCountByCentroids<InputImage, PatchShape, Preprocessor, [(); C]>,
        O: BitArray + Sync + Send + BitWord,
        OutputImage: Send + Sync + Image2D,
        const C: usize,
    > Layer<InputImage, PatchShape, Preprocessor, O, OutputImage, [(); C]> for I
where
    bool: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    f32: Element<I::BitShape> + Element<O::BitShape>,
    <I as Element<O::BitShape>>::Array: Copy + Sync,
    <u32 as Element<I::BitShape>>::Array: Sync + Default,
    (<f32 as Element<I::BitShape>>::Array, f32): Element<O::BitShape>,
    <(<f32 as Element<I::BitShape>>::Array, f32) as Element<O::BitShape>>::Array:
        Apply<InputImage, PatchShape, Preprocessor, OutputImage> + Sync + BFBVM<I, O>,
    (I, [u32; C]): serde::Serialize + std::fmt::Debug,
    [(<f32 as Element<O::BitShape>>::Array, f32); C]: FloatLoss<O, C> + Sync,
    distributions::Standard: distributions::Distribution<I>,
    for<'de> (I, [u32; C]): serde::Deserialize<'de>,
    (
        <(<f32 as Element<I::BitShape>>::Array, f32) as Element<O::BitShape>>::Array,
        [(<f32 as Element<O::BitShape>>::Array, f32); C],
    ): Default + Mutate,
    InputImage::PixelType: Element<PatchShape>,
{
    type WeightsType = (
        <(<f32 as Element<I::BitShape>>::Array, f32) as Element<O::BitShape>>::Array,
        [(<f32 as Element<O::BitShape>>::Array, f32); C],
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
            "{}, seed:{}, sdev:{}, ds:{}.mutations",
            std::any::type_name::<O>(),
            params.weights_seed,
            params.decend_iters,
            params.noise_sdev,
        ));
        let (layer_weights, aux_weights): Self::WeightsType = if let Some(mutation_log_file) =
            File::open(&weights_path).ok()
        {
            println!("loading {:?} from disk", &weights_path);
            let mutation_log: Vec<usize> =
                deserialize_from(mutation_log_file).expect("can't deserialize from file");
            let normal = Normal::new(0f32, 0.03).unwrap();
            let mut rng = Hc128Rng::seed_from_u64(params.weights_seed);
            let mut noise: Vec<f32> = (0..<Self::WeightsType>::NOISE_LEN + params.decend_iters)
                .map(|_| normal.sample(&mut rng))
                .collect();

            mutation_log
                .iter()
                .fold(<Self::WeightsType>::default(), |weights, mutation| {
                    weights.mutate(&noise[*mutation..])
                })
        } else {
            let counts_path = counts_dir.join("counts");
            let counts: Vec<(I, [u32; C])> =
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

                    let counts = <I as ImageCountByCentroids<
                        InputImage,
                        PatchShape,
                        Preprocessor,
                        [(); C],
                    >>::count_by_centroids(&examples, &centroids);
                    serialize_into(File::create(&counts_path).unwrap(), &counts).unwrap();
                    counts
                };
            dbg!(counts.len());
            let n_examples: u64 = counts
                .iter()
                .map(|(_, c)| c.iter().sum::<u32>() as u64)
                .sum();
            dbg!(n_examples);
            //dbg!(&counts);
            let mut weights = <Self::WeightsType>::default();
            let mut cur_sum_loss = f64::MAX;
            let mut rng = Hc128Rng::seed_from_u64(params.weights_seed);
            let normal = Normal::new(0f32, params.noise_sdev).unwrap();
            let mut noise: Vec<f32> = (0..<Self::WeightsType>::NOISE_LEN + params.decend_iters)
                .map(|_| normal.sample(&mut rng))
                .collect();
            let mut mutations_log = Vec::<usize>::new();
            dbg!("start training");
            for i in 0..params.decend_iters {
                let perturbed_weights = weights.mutate(&noise[i..]);
                let new_sum_loss: f64 = counts
                    .par_iter()
                    .map(|(input, class_counts)| {
                        perturbed_weights
                            .1
                            .counts_loss(&perturbed_weights.0.bfbvm(input), class_counts)
                            as f64
                    })
                    .sum();
                //dbg!(new_sum_loss);
                if new_sum_loss < cur_sum_loss {
                    println!("{} {}", i, (new_sum_loss / n_examples as f64));
                    cur_sum_loss = new_sum_loss;
                    weights = perturbed_weights;
                    mutations_log.push(i);
                }
            }
            dbg!("done training");

            serialize_into(File::create(&weights_path).unwrap(), &mutations_log).unwrap();
            weights
        };
        let new_examples: Vec<_> = examples
            .par_iter()
            .map(|(image, class)| {
                (
                    <<(<f32 as Element<I::BitShape>>::Array, f32) as Element<O::BitShape>>::Array as Apply<
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
