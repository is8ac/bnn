use crate::bits::{BitArray, BitArrayOPs, Distance, IncrementFracCounters};
use crate::count::ElementwiseAdd;
use crate::image2d::{Image2D, PatchFold};
use crate::shape::{Element, Map, Shape};
use crate::unary::Preprocess;
use rand::distributions;
use rand::Rng;
use rayon::prelude::*;

pub trait ImagePatchLloyds<Image: Image2D, PatchShape>
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
        (0..i).fold((0..k).map(|_| rng.gen()).collect(), |centroids, e| {
            dbg!(e);
            Self::avgs(examples, &centroids)
        })
    }
}

impl<
        T: Distance + BitArray + IncrementFracCounters + BitArrayOPs + Sync,
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

/// examples is of length `n`.
/// centroids is of length `k`.
/// The return Vec is of length `n`.
/// Each element is a pair of (bag, class).
/// The Vec<(u16, u32)> bag is of length `k`.
/// The u16 is the index into the centroids. The u32 is the number of patches in that cell.
/// The bag will be filtered of empty cells.
/// k must never be < 2^16 !!!
pub trait CentroidCountPerImage<Image, PatchShape, C: Shape>
where
    Self: Sized,
    u32: Element<C>,
{
    fn centroid_count_per_image(
        examples: &Vec<(Image, usize)>,
        centroids: &Vec<Self>,
    ) -> Vec<(Vec<(u16, u32)>, usize)>;
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
    ) -> Vec<(Vec<(u16, u32)>, usize)> {
        assert!(centroids.len() < 2usize.pow(16));
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
                        .map(|(i, c)| (i as u16, *c))
                        .collect(),
                    *class,
                )
            })
            .collect()
    }
}
