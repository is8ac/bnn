use crate::bits::{BitArray, BitArrayOPs, Distance, IncrementFracCounters};
use crate::count::ElementwiseAdd;
use crate::image2d::{Image2D, PatchFold};
use crate::shape::{Element, Map, Shape};
use rand::distributions;
use rand::Rng;
use rayon::prelude::*;

pub trait ImagePatchLloyds<Image: Image2D, PatchShape>
where
    Self: BitArray,
    PatchShape: Shape,
    //u32: Element<Self::BitShape>,
    Image::PixelType: Element<PatchShape>,
{
    fn avgs(
        examples: &Vec<(Image, usize)>,
        centroids: &Vec<Self>,
        prune_threshold: usize,
    ) -> Vec<Self>;
    fn lloyds<RNG: Rng>(
        rng: &mut RNG,
        examples: &Vec<(Image, usize)>,
        k: usize,
        i: usize,
        prune_threshold: usize,
    ) -> Vec<Self>
    where
        distributions::Standard: distributions::Distribution<Self>,
    {
        (0..i).fold((0..k).map(|_| rng.gen()).collect(), |centroids, e| {
            let centroids = Self::avgs(examples, &centroids, prune_threshold);
            println!(
                "{}: n:{} {}",
                e,
                centroids.len(),
                centroids.len() as f64 / k as f64
            );
            centroids
        })
    }
}

impl<
        T: Distance + BitArray + IncrementFracCounters + BitArrayOPs + Sync + Send,
        Image: PatchFold<Vec<(usize, <u32 as Element<T::BitShape>>::Array)>, PatchShape> + Image2D + Sync,
        PatchShape: Shape,
    > ImagePatchLloyds<Image, PatchShape> for T
where
    u32: Element<T::BitShape>,
    Image::PixelType: Element<PatchShape, Array = T>,
    <u32 as Element<<T as BitArray>::BitShape>>::Array: Default + Sync + Send + ElementwiseAdd,
    T::BitShape: Map<u32, bool>,
    bool: Element<T::BitShape>,
{
    fn avgs(examples: &Vec<(Image, usize)>, centroids: &Vec<T>, prune_threshold: usize) -> Vec<T> {
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
                            let closest_centroid = centroids
                                .iter()
                                .map(|centroid| patch.distance(centroid))
                                .enumerate()
                                .min_by_key(|(_, count)| *count)
                                .unwrap()
                                .0;
                            patch.increment_frac_counters(&mut sub_acc[closest_centroid]);
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
            .par_iter()
            .filter(|(n, _)| *n > prune_threshold)
            .map(|(n, counts)| {
                let threshold = *n as u32 / 2;
                let bools =
                    <<T as BitArray>::BitShape as Map<u32, bool>>::map(&counts, |&x| x > threshold);
                T::bitpack(&bools)
            })
            .collect()
    }
}

/// examples is of length `n`.
/// centroids is of length `k`.
/// The return Vec is of length `n`.
/// Each element is a pair of (bag, class).
/// The Vec<(u32, u32)> bag is of length `k`.
/// The u32 is the index into the centroids. The u32 is the number of patches in that cell.
/// The bag will be filtered of empty cells.
pub trait CentroidCountPerImage<Image, PatchShape, C: Shape>
where
    Self: Sized,
    u32: Element<C>,
{
    fn centroid_count_per_image(
        examples: &Vec<(Image, usize)>,
        centroids: &Vec<Self>,
    ) -> Vec<(Vec<(u32, u32)>, usize)>;
}

impl<
        T: Sized + Distance + Copy + Send + Sync,
        Image: PatchFold<Vec<u32>, PatchShape> + Image2D + Sync,
        PatchShape: Shape,
        const C: usize,
    > CentroidCountPerImage<Image, PatchShape, [(); C]> for T
where
    Image::PixelType: Element<PatchShape, Array = T>,
    [u32; C]: Default,
{
    fn centroid_count_per_image(
        examples: &Vec<(Image, usize)>,
        centroids: &Vec<Self>,
    ) -> Vec<(Vec<(u32, u32)>, usize)> {
        examples
            .par_iter()
            .map(|(image, class)| {
                let counts = <Image as PatchFold<Vec<u32>, PatchShape>>::patch_fold(
                    image,
                    vec![0u32; centroids.len()],
                    |mut counts, patch| {
                        let closest_centroid = centroids
                            .iter()
                            .map(|centroid| patch.distance(centroid))
                            .enumerate()
                            .min_by_key(|(_, count)| *count)
                            .unwrap()
                            .0;
                        counts[closest_centroid] += 1;
                        counts
                    },
                );
                (
                    counts
                        .iter()
                        .enumerate()
                        .filter(|&(_, c)| *c > 0)
                        .map(|(i, c)| (i as u32, *c))
                        .collect(),
                    *class,
                )
            })
            .collect()
    }
}

pub trait NullCluster<Image, PatchShape>
where
    Self: Sized,
{
    fn null_cluster(examples: &Vec<(Image, usize)>) -> Vec<Self>;
}

impl<
        T: Sized + Copy + Send + Sync,
        Image: PatchFold<Vec<T>, PatchShape> + Image2D + Sync,
        PatchShape: Shape,
    > NullCluster<Image, PatchShape> for T
where
    Image::PixelType: Element<PatchShape, Array = T>,
{
    fn null_cluster(examples: &Vec<(Image, usize)>) -> Vec<Self> {
        examples
            .par_iter()
            .fold(
                || Vec::new(),
                |acc, (image, _)| {
                    image.patch_fold(acc, |mut a, patch| {
                        a.push(*patch);
                        a
                    })
                },
            )
            .reduce(
                || Vec::new(),
                |mut a, mut b| {
                    a.append(&mut b);
                    a
                },
            )
    }
}
