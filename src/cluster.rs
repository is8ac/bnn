use crate::bits::{BitArray, BitArrayOPs, Distance, IncrementFracCounters};
use crate::count::ElementwiseAdd;
use crate::image2d::{Image2D, PatchFold, RandomPatch};
use crate::shape::{Element, Map, Shape};
use rand::distributions;
use rand::Rng;
use rayon::prelude::*;

pub trait ImagePatchLloyds<ImageShape: Shape, PatchShape: Shape, Pixel>
where
    Pixel: Element<PatchShape> + Element<ImageShape>,
{
    fn avgs(
        examples: &[(<Pixel as Element<ImageShape>>::Array, usize)],
        centroids: &[<Pixel as Element<PatchShape>>::Array],
    ) -> Vec<<Pixel as Element<PatchShape>>::Array>;
    fn lloyds<RNG: Rng>(
        rng: &mut RNG,
        examples: &[(<Pixel as Element<ImageShape>>::Array, usize)],
        k: usize,
        i: usize,
    ) -> Vec<<Pixel as Element<PatchShape>>::Array>
    where
        distributions::Standard: distributions::Distribution<<Pixel as Element<PatchShape>>::Array>;
}

impl<ImageShape: Shape, PatchShape: Shape, Pixel> ImagePatchLloyds<ImageShape, PatchShape, Pixel>
    for ()
where
    bool: Element<<<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape>,
    u32: Element<<<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape>,
    <u32 as Element<<<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape>>::Array:
        Default + Sync + Send + ElementwiseAdd,
    Pixel: Element<ImageShape> + Element<PatchShape>,
    <Pixel as Element<PatchShape>>::Array:
        BitArray + Send + Distance + IncrementFracCounters + BitArrayOPs + Sync + Clone + Copy,
    <<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape: Map<u32, bool>,
    <Pixel as Element<ImageShape>>::Array:
        Sync
            + Image2D<PixelType = Pixel>
            + RandomPatch<PatchShape>
            + PatchFold<Vec<<Pixel as Element<PatchShape>>::Array>, PatchShape>
            + PatchFold<
                Vec<(
                    usize,
                    <u32 as Element<
                        <<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape,
                    >>::Array,
                )>,
                PatchShape,
            >,
{
    fn avgs(
        examples: &[(<Pixel as Element<ImageShape>>::Array, usize)],
        centroids: &[<Pixel as Element<PatchShape>>::Array],
    ) -> Vec<<Pixel as Element<PatchShape>>::Array> {
        examples
            .par_iter()
            .fold(
                || {
                    (0..centroids.len())
                        .map(|_| {
                            <(
                                usize,
                                <u32 as Element<
                                    <<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape,
                                >>::Array,
                            )>::default()
                        })
                        .collect()
                },
                |acc: Vec<(
                    usize,
                    <u32 as Element<
                        <<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape,
                    >>::Array,
                )>,
                 (image, _)| {
                    <<Pixel as Element<ImageShape>>::Array as PatchFold<
                        Vec<(
                            usize,
                            <u32 as Element<
                                <<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape,
                            >>::Array,
                        )>,
                        PatchShape,
                    >>::patch_fold(image, acc, |mut sub_acc, patch| {
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
            .reduce_with(|mut a, b| {
                a.elementwise_add(&b);
                a
            })
            .unwrap()
            .par_iter()
            .map(|(n, counts)| {
                let threshold = *n as u32 / 2;
                let bools =
                    <<<Pixel as Element<PatchShape>>::Array as BitArray>::BitShape as Map<
                        u32,
                        bool,
                    >>::map(&counts, |&x| x > threshold);
                <<Pixel as Element<PatchShape>>::Array>::bitpack(&bools)
            })
            .collect()
    }
    fn lloyds<RNG: Rng>(
        rng: &mut RNG,
        examples: &[(<Pixel as Element<ImageShape>>::Array, usize)],
        k: usize,
        i: usize,
    ) -> Vec<<Pixel as Element<PatchShape>>::Array>
    where
        distributions::Standard: distributions::Distribution<<Pixel as Element<PatchShape>>::Array>,
    {
        (0..i).fold(
            examples
                .iter()
                .take(k)
                .map(|(image, _)| image.random_patch(rng))
                .collect(),
            |centroids, _| {
                <() as ImagePatchLloyds<ImageShape, PatchShape, Pixel>>::avgs(examples, &centroids)
            },
        )
    }
}
