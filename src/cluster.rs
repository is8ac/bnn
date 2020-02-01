use crate::bits::{BitArray, BitArrayOPs, Distance, IncrementFracCounters};
use crate::count::ElementwiseAdd;
use crate::image2d::{Image2D, PatchFold};
use crate::shape::{Element, Map, Shape};
use crate::unary::Preprocess;
use rand::distributions;
use rand::Rng;
use rayon::prelude::*;

pub trait ImagePatchLloyds<Image: Image2D, PatchShape, Preprocessor>
where
    Self: BitArray,
    PatchShape: Shape,
    u32: Element<Self::BitShape>,
    Image::PixelType: Element<PatchShape>,
{
    fn avgs(examples: &Vec<(Image, usize)>, centroids: &Vec<Self>) -> Vec<Self>;
    fn lloyds<RNG: Rng>(
        rng: &mut RNG,
        examples: &Vec<(Image, usize)>,
        k: usize,
        i: usize,
    ) -> Vec<Self>
    where
        distributions::Standard: distributions::Distribution<Self>,
    {
        let mut centroids: Vec<Self> = (0..k).map(|_| rng.gen()).collect();
        for e in 0..i {
            dbg!(e);
            centroids = Self::avgs(examples, &centroids);
        }
        centroids
    }
}

impl<
        T: Distance + BitArray + IncrementFracCounters + BitArrayOPs + Sync,
        Image: PatchFold<Vec<(usize, <u32 as Element<T::BitShape>>::Array)>, PatchShape> + Image2D + Sync,
        PatchShape: Shape,
        Preprocessor: Preprocess<<Image::PixelType as Element<PatchShape>>::Array, Output = T>,
    > ImagePatchLloyds<Image, PatchShape, Preprocessor> for T
where
    u32: Element<T::BitShape>,
    Image::PixelType: Element<PatchShape>,
    <u32 as Element<<T as BitArray>::BitShape>>::Array: Default + Sync + Send + ElementwiseAdd,
    T::BitShape: Map<u32, bool>,
    bool: Element<T::BitShape>,
{
    fn avgs(examples: &Vec<(Image, usize)>, centroids: &Vec<T>) -> Vec<T> {
        examples
            .par_iter()
            .fold(
                || {
                    (0..centroids.len())
                        .map(|_| {
                            <(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array)>::default()
                        })
                        .collect()
                },
                |acc: Vec<(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array)>,
                 (image, _)| {
                    <Image as PatchFold<_, PatchShape>>::patch_fold(
                        image,
                        acc,
                        |mut sub_acc, patch| {
                            let preprocesed = Preprocessor::preprocess(patch);
                            let closest_centroid = centroids
                                .iter()
                                .map(|centroid| preprocesed.distance(centroid))
                                .enumerate()
                                .min_by_key(|(_, count)| *count)
                                .unwrap()
                                .0;
                            preprocesed.increment_frac_counters(&mut sub_acc[closest_centroid]);
                            sub_acc
                        },
                    )
                },
            )
            .reduce(
                || {
                    (0..centroids.len())
                        .map(|_| {
                            <(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array)>::default()
                        })
                        .collect()
                },
                |mut a, b| {
                    a.elementwise_add(&b);
                    a
                },
            )
            .iter()
            .map(|(n, counts)| {
                let threshold = *n as u32 / 2;
                let bools =
                    <<T as BitArray>::BitShape as Map<u32, bool>>::map(&counts, |&x| x > threshold);
                T::bitpack(&bools)
            })
            .collect()
    }
}

pub trait ImageCountByCentroids<Image, PatchShape, Preprocessor, C: Shape>
where
    Self: Sized,
    u32: Element<C>,
{
    fn count_by_centroids(
        examples: &Vec<(Image, usize)>,
        centroids: &Vec<Self>,
    ) -> Vec<(Self, <u32 as Element<C>>::Array)>;
}

impl<
        T: Sized + Distance + Copy + Send + Sync,
        Image: PatchFold<Vec<<u32 as Element<[(); C]>>::Array>, PatchShape> + Image2D + Sync,
        PatchShape: Shape,
        Preprocessor: Preprocess<<Image::PixelType as Element<PatchShape>>::Array, Output = T>,
        const C: usize,
    > ImageCountByCentroids<Image, PatchShape, Preprocessor, [(); C]> for T
where
    u32: Element<[(); C], Array = [u32; C]>,
    Image::PixelType: Element<PatchShape>,
    [u32; C]: Default,
{
    fn count_by_centroids(
        examples: &Vec<(Image, usize)>,
        centroids: &Vec<Self>,
    ) -> Vec<(Self, <u32 as Element<[(); C]>>::Array)> {
        examples
            .par_iter()
            .fold(
                || {
                    (0..centroids.len())
                        .map(|_| <[u32; C]>::default())
                        .collect()
                },
                |acc, (image, class)| {
                    <Image as PatchFold<_, PatchShape>>::patch_fold(
                        image,
                        acc,
                        |mut sub_acc, patch| {
                            let preprocesed = Preprocessor::preprocess(patch);
                            let closest_centroid = centroids
                                .iter()
                                .map(|centroid| preprocesed.distance(centroid))
                                .enumerate()
                                .min_by_key(|(_, count)| *count)
                                .unwrap()
                                .0;
                            sub_acc[closest_centroid][*class] += 1;
                            sub_acc
                        },
                    )
                },
            )
            .reduce(
                || {
                    (0..centroids.len())
                        .map(|_| <[u32; C]>::default())
                        .collect()
                },
                |mut a, b| {
                    a.elementwise_add(&b);
                    a
                },
            )
            .iter()
            .zip(centroids.iter())
            .map(|(counts, centroid)| (*centroid, *counts))
            .collect()
    }
}
