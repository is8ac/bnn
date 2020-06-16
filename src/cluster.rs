use crate::bits::{BitArray, BitArrayOPs, Distance, IncrementFracCounters};
use crate::count::ElementwiseAdd;
use crate::image2d::{Image2D, PatchFold};
use crate::shape::{Element, Map, Shape};
use rand::distributions;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;

pub fn class_dist<const C: usize>(examples: &Vec<(Vec<u32>, usize)>, centroids: &Vec<Vec<u32>>) -> Vec<[u32; C]>
where
    [u32; C]: Default + Copy,
{
    examples
        .par_iter()
        .fold(
            || centroids.iter().map(|_| <[u32; C]>::default()).collect(),
            |mut acc: Vec<[u32; C]>, (example, class)| {
                let closest_centroid: usize = centroids
                    .iter()
                    .map(|centroid| vec_distance(centroid, example))
                    .enumerate()
                    .min_by_key(|(_, d)| *d)
                    .unwrap()
                    .0;
                acc[closest_centroid][*class] += 1;
                acc
            },
        )
        .reduce_with(|a, b| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| {
                    let mut acc = *x;
                    acc.iter_mut().zip(y.iter()).for_each(|(a, b)| *a += b);
                    acc
                })
                .collect()
        })
        .unwrap()
}

fn cluster_avgs(examples: &Vec<(Vec<u32>, usize)>, centroids: &Vec<Vec<u32>>, n: usize, prune_threshold: usize) -> Vec<Vec<u32>> {
    examples
        .par_iter()
        .fold(
            || centroids.iter().map(|_| (0usize, vec![0u32; n])).collect(),
            |mut acc: Vec<(usize, Vec<u32>)>, (example, _)| {
                let closest_centroid: usize = centroids
                    .iter()
                    .map(|centroid| vec_distance(centroid, example))
                    .enumerate()
                    .min_by_key(|(_, d)| *d)
                    .unwrap()
                    .0;
                acc[closest_centroid].0 += 1;
                acc[closest_centroid].1.iter_mut().zip(example.iter()).for_each(|(count, d)| *count += *d);
                acc
            },
        )
        .reduce_with(
            //|| centroids.iter().map(|_| (0usize, vec![0u32; n])).collect(),
            |a, b| {
                a.iter()
                    .zip(b.iter())
                    .map(|((xc, x), (yc, y))| (xc + yc, x.iter().zip(y.iter()).map(|(a, b)| a + b).collect()))
                    .collect()
            },
        )
        .unwrap()
        .par_iter()
        .filter(|(n, _)| *n > prune_threshold)
        .map(|(n_examples, sums)| sums.iter().map(|x| x / *n_examples as u32).collect())
        .collect()
}

pub fn vec_distance(a: &Vec<u32>, b: &Vec<u32>) -> u32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&c, &e)| c.saturating_sub(e) | e.saturating_sub(c)).sum()
}

// the class is not used.
pub fn patch_count_lloyds(
    examples: &Vec<(Vec<u32>, usize)>,
    seed: u64,
    i: usize,
    n_patches: usize,
    n_centroids: usize,
    prune_threshold: usize,
) -> Vec<Vec<u32>> {
    (0..i).fold(
        examples
            .choose_multiple(&mut Hc128Rng::seed_from_u64(seed), n_centroids)
            .map(|(x, _)| x.clone())
            .collect(),
        |centroids, i| {
            println!("{}: {}", i, centroids.len());
            cluster_avgs(examples, &centroids, n_patches, prune_threshold)
        },
    )
}

pub trait ImagePatchLloyds<Image: Image2D, PatchShape>
where
    Self: BitArray,
    PatchShape: Shape,
    //u32: Element<Self::BitShape>,
    Image::PixelType: Element<PatchShape>,
{
    fn avgs(examples: &Vec<(Image, usize)>, centroids: &Vec<Self>, prune_threshold: usize) -> Vec<Self>;
    fn lloyds<RNG: Rng>(rng: &mut RNG, examples: &Vec<(Image, usize)>, k: usize, i: usize, prune_threshold: usize) -> Vec<Self>
    where
        distributions::Standard: distributions::Distribution<Self>,
    {
        (0..i).fold((0..k).map(|_| rng.gen()).collect(), |centroids, e| {
            let centroids = Self::avgs(examples, &centroids, prune_threshold);
            println!("{}: n:{} {}", e, centroids.len(), centroids.len() as f64 / k as f64);
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
                        .map(|_| <(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array)>::default())
                        .collect()
                },
                |acc: Vec<(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array)>, (image, _)| {
                    <Image as PatchFold<_, PatchShape>>::patch_fold(image, acc, |mut sub_acc, patch| {
                        let closest_centroid = centroids
                            .iter()
                            .map(|centroid| patch.distance(centroid))
                            .enumerate()
                            .min_by_key(|(_, count)| *count)
                            .unwrap()
                            .0;
                        patch.increment_frac_counters(&mut sub_acc[closest_centroid]);
                        sub_acc
                    })
                },
            )
            .reduce(
                || {
                    (0..centroids.len())
                        .map(|_| <(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array)>::default())
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
                let bools = <<T as BitArray>::BitShape as Map<u32, bool>>::map(&counts, |&x| x > threshold);
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
pub trait CentroidCount<Image, PatchShape, C: Shape>
where
    Self: Sized,
    u32: Element<C>,
{
    fn centroid_count(image: &Image, centroids: &Vec<Self>) -> Vec<u32>;
}

impl<T: Sized + Distance + Copy + Send + Sync, Image: PatchFold<Vec<u32>, PatchShape> + Image2D + Sync, PatchShape: Shape, const C: usize>
    CentroidCount<Image, PatchShape, [(); C]> for T
where
    Image::PixelType: Element<PatchShape, Array = T>,
    [u32; C]: Default,
{
    fn centroid_count(image: &Image, centroids: &Vec<Self>) -> Vec<u32> {
        <Image as PatchFold<Vec<u32>, PatchShape>>::patch_fold(image, centroids.iter().map(|_| 0u32).collect(), |mut counts, patch| {
            let closest_centroid = centroids
                .iter()
                .map(|centroid| patch.distance(centroid))
                .enumerate()
                .min_by_key(|(_, count)| *count)
                .unwrap()
                .0;
            counts[closest_centroid] += 1;
            counts
        })
    }
}

pub fn sparsify_centroid_count(counts: &Vec<u32>, filter_threshold: u32) -> Vec<(usize, u32)> {
    counts
        .iter()
        .enumerate()
        .filter(|&(_, c)| *c > filter_threshold)
        .map(|(i, c)| (i, *c))
        .collect()
}

pub trait NullCluster<Image, PatchShape>
where
    Self: Sized,
{
    fn null_cluster(examples: &Vec<(Image, usize)>) -> Vec<Self>;
}

impl<T: Sized + Copy + Send + Sync, Image: PatchFold<Vec<T>, PatchShape> + Image2D + Sync, PatchShape: Shape> NullCluster<Image, PatchShape> for T
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
