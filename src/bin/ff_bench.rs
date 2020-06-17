#![feature(const_generics)]
use bitnn::bits::{
    b32, BitArray, BitArrayOPs, BitMapPack, BitWord, IncrementFracCounters, MaskedDistance,
    TritArray,
};
use bitnn::cluster::{
    self, patch_count_lloyds, sparsify_centroid_count, CentroidCount, ImagePatchLloyds,
};
use bitnn::datasets::cifar;
//use bitnn::descend::Descend;
use bitnn::image2d::{Image2D, PatchFold, PixelMap, StaticImage};
use bitnn::shape::{Element, IndexGet, Map, MapMut, Shape, ZipFold, ZipMap};
use bitnn::unary::to_10;
use rand::distributions;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;
use std::time::{Duration, Instant};

trait IIIVMM<I: BitArray, C: Shape>
where
    i32: Element<C> + Element<I::BitShape>,
    (usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])): Element<C>,
    [u32; 2]: Element<I::BitShape>,
    u32: Element<C>,
    i8: Element<C>,
{
    fn iiivmm(&self, input: &<i32 as Element<I::BitShape>>::Array) -> <i32 as Element<C>>::Array;
    fn update_iiivmm_acts(
        weights: &<i8 as Element<C>>::Array,
        input: i32,
        acts: &<i32 as Element<C>>::Array,
    ) -> <i32 as Element<C>>::Array;
    fn subtract_input_from_iiivmm_acts(
        weights: &<i8 as Element<C>>::Array,
        input: i32,
        acts: &<i32 as Element<C>>::Array,
    ) -> <i32 as Element<C>>::Array;
    fn increment_grads(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        class_dist: &<u32 as Element<C>>::Array,
        grads: &mut <(usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])) as Element<
            C,
        >>::Array,
        input_threshold: i32,
    ) -> u64;
    fn update(
        &self,
        grads: &<(usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])) as Element<C>>::Array,
        update_threshold: u32,
    ) -> Self;
    fn descend(
        &self,
        patch_acts: &Vec<I>,
        sparse_patch_count_class_dists: &Vec<(Vec<(usize, u32)>, i32, <u32 as Element<C>>::Array)>,
        i: usize,
        it: i32,
        ut: u32,
    ) -> Self;
}

impl<I, const C: usize> IIIVMM<I, [(); C]> for [(<i8 as Element<I::BitShape>>::Array, i8); C]
where
    I: BitArray + IncrementFracCounters,
    I::BitShape: MapMut<i32, [u32; 2]> + ZipFold<i32, i32, i8> + ZipMap<i8, [u32; 2], i8>,
    [i32; C]: Default,
    [(); C]: ZipMap<
        (<i8 as Element<I::BitShape>>::Array, i8),
        (usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])),
        (<i8 as Element<I::BitShape>>::Array, i8),
    >,
    i8: Element<I::BitShape>,
    i32: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    bool: Element<I::BitShape>,
    [u32; 2]: Element<I::BitShape>,
    [(
        usize,
        (
            <[u32; 2] as Element<<I as BitArray>::BitShape>>::Array,
            [u32; 2],
        ),
    ); C]: Default,
    [(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8); C]: Default,
    <I as BitArray>::BitShape: Map<u32, i32>,
    <i8 as Element<I::BitShape>>::Array: IndexGet<<I::BitShape as Shape>::Index, Element = i8>,
    <i32 as Element<I::BitShape>>::Array: IndexGet<<I::BitShape as Shape>::Index, Element = i32>,
    <u32 as Element<<I as BitArray>::BitShape>>::Array: Default,
{
    fn iiivmm(&self, input: &<i32 as Element<I::BitShape>>::Array) -> [i32; C] {
        <[(); C] as Map<(<i8 as Element<I::BitShape>>::Array, i8), i32>>::map(
            self,
            |(weights, bias)| {
                <I::BitShape as ZipFold<i32, i32, i8>>::zip_fold(
                    &input,
                    weights,
                    0,
                    |sum, i, &w| sum + i * w as i32,
                ) + *bias as i32
            },
        )
    }
    fn update_iiivmm_acts(weights: &[i8; C], input: i32, partial_acts: &[i32; C]) -> [i32; C] {
        <[(); C] as ZipMap<i32, i8, i32>>::zip_map(partial_acts, weights, |partial_act, &weight| {
            partial_act + weight as i32 * input
        })
    }
    fn subtract_input_from_iiivmm_acts(
        weights: &[i8; C],
        input: i32,
        partial_acts: &[i32; C],
    ) -> [i32; C] {
        <[(); C] as ZipMap<i32, i8, i32>>::zip_map(partial_acts, weights, |partial_act, &weight| {
            partial_act - weight as i32 * input
        })
    }
    fn increment_grads(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        class_dist: &[u32; C],
        grads: &mut [(usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])); C],
        input_threshold: i32,
    ) -> u64 {
        let class_acts =
            <[(<i8 as Element<I::BitShape>>::Array, i8); C] as IIIVMM<I, [(); C]>>::iiivmm(
                self, input,
            );
        class_dist
            .iter()
            .enumerate()
            .map(|(class, &count)| {
                let (max_index, &max_val) = class_acts
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != class)
                    .max_by_key(|(_, v)| *v)
                    .unwrap();
                if class_acts[class] <= max_val {
                    for &(i, sign) in &[(class, false), (max_index, true)] {
                        (grads[i].1).1[sign as usize] += count;
                        grads[i].0 += count as usize;
                        <I::BitShape as MapMut<i32, [u32; 2]>>::map_mut(
                            &mut (grads[i].1).0,
                            &input,
                            |grad, &input| {
                                grad[((input.is_positive()) ^ sign) as usize] +=
                                    (input.abs() > input_threshold) as u32 * count;
                            },
                        );
                    }
                    0
                } else {
                    count as u64
                }
            })
            .sum()
    }
    fn update(
        &self,
        grads: &[(usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])); C],
        update_threshold: u32,
    ) -> Self {
        <[(); C] as ZipMap<
            (<i8 as Element<I::BitShape>>::Array, i8),
            (usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])),
            (<i8 as Element<I::BitShape>>::Array, i8),
        >>::zip_map(self, grads, |(w, b), (c, (wg, bg))| {
            (
                <I::BitShape as ZipMap<i8, [u32; 2], i8>>::zip_map(w, wg, |&w, &g| {
                    w.saturating_add(grad_counts_to_update(g, update_threshold))
                }),
                b.saturating_add(grad_counts_to_update(*bg, update_threshold)),
            )
        })
    }
    fn descend(
        &self,
        patch_acts: &Vec<I>,
        sparse_patch_count_class_dists: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
        n_iters: usize,
        it: i32,
        ut: u32,
    ) -> Self {
        let total_n: u32 = sparse_patch_count_class_dists
            .iter()
            .map(|(_, _, c)| c.iter())
            .flatten()
            .sum();
        (0..n_iters).fold(
            <[(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8); C]>::default(),
            |aux_weights, i| {
                let (n_correct, aux_grads) = sparse_patch_count_class_dists.iter().fold(
                    <(
                        u64,
                        [(
                            usize,
                            (
                                <[u32; 2] as Element<<I as BitArray>::BitShape>>::Array,
                                [u32; 2],
                            ),
                        ); C],
                    )>::default(),
                    |mut grads, (patch_bag, _, class)| {
                        let (hidden_act_n, hidden_act_counts) = patch_bag.iter().fold(
                            <(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array)>::default(
                            ),
                            |mut acc, &(index, count)| {
                                patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                                acc
                            },
                        );
                        let n = hidden_act_n as i32 / 2;
                        let hidden_acts = <<I as BitArray>::BitShape as Map<u32, i32>>::map(
                            &hidden_act_counts,
                            |count| *count as i32 - n,
                        );
                        //dbg!(&hidden_acts);
                        let is_correct = <[(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8);
                            C] as IIIVMM<I, [(); C]>>::increment_grads(
                            &aux_weights,
                            &hidden_acts,
                            class,
                            &mut grads.1,
                            it,
                        );
                        //dbg!(loss);
                        grads.0 += is_correct as u64;
                        grads
                    },
                );
                println!("{}: {}", i, n_correct as f64 / total_n as f64);
                <[(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<
                    I,
                    [(); C],
                >>::update(&aux_weights, &aux_grads, ut)
            },
        )
    }
}

fn grad_counts_to_update(g: [u32; 2], threshold: u32) -> i8 {
    SIGNS[((g[0].saturating_sub(g[1]) | g[1].saturating_sub(g[0])) > threshold) as usize]
        [(g[0] > g[1]) as usize]
}

const SIGNS: [[i8; 2]; 2] = [[0, 0], [1, -1]];

// Bit Trit Bit Vector Matrix Multipply
pub trait BTBVMM<I: BitArray, O: BitArray> {
    fn btbvmm(&self, input: &I) -> O;
    fn btbvmm_one_bit(weights: &I::TritArrayType, threshold: u32, input: &I) -> bool;
}

impl<I, O> BTBVMM<I, O> for <(I::TritArrayType, u32) as Element<O::BitShape>>::Array
where
    bool: Element<I::BitShape>,
    I: BitArray,
    O: BitArray + BitMapPack<(I::TritArrayType, u32)>,
    I::TritArrayType: MaskedDistance + TritArray<BitArrayType = I>,
    (I::TritArrayType, u32): Element<O::BitShape>,
    (I, u32): Element<O::BitShape>,
    <(I::TritArrayType, u32) as Element<O::BitShape>>::Array:
        IndexGet<<O::BitShape as Shape>::Index, Element = (I::TritArrayType, u32)>,
{
    fn btbvmm(&self, input: &I) -> O {
        <O as BitMapPack<(I::TritArrayType, u32)>>::bit_map_pack(self, |(weights, threshold)| {
            weights.masked_distance(input) > *threshold
        })
    }
    fn btbvmm_one_bit(weights: &I::TritArrayType, threshold: u32, input: &I) -> bool {
        weights.masked_distance(input) > threshold
    }
}

fn unary(input: [u8; 3]) -> b32 {
    to_10(input[0]) | (to_10(input[1]) << 10) | (to_10(input[2]) << 20)
}

fn act_loss<const C: usize>(acts: &[i32; C], class: usize) -> u64 {
    let (max_index, max_act) = acts
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != class)
        .max_by_key(|(_, v)| *v)
        .unwrap();
    (max_act - acts[class]).abs() as u64
}

// This need only called once per layer.
// return in (patch_centroids, image_patch_bags)
// where image_patch_bags is a Vec<(sparse_patch_bag, center, class_dist)>
fn cluster_patches_and_images<
    P: CentroidCount<ImageType, [[(); 3]; 3], [(); C]>
        + BitArray
        + Sync
        + Send
        + ImagePatchLloyds<ImageType, [[(); 3]; 3]>,
    ImageType: Image2D + Sync + Send,
    const C: usize,
>(
    examples: &Vec<(ImageType, usize)>,
) -> (Vec<P>, Vec<(Vec<(usize, u32)>, i32, [u32; C])>)
where
    distributions::Standard: distributions::Distribution<P>,
    [u32; C]: Default + Copy + Sync + Send,
{
    let mut rng = Hc128Rng::seed_from_u64(0);
    let patch_centroids =
        <P as ImagePatchLloyds<_, [[(); 3]; 3]>>::lloyds(&mut rng, &examples, 1000, 3, 1);
    let patch_dists: Vec<(Vec<u32>, usize)> = examples
        .par_iter()
        .map(|(image, class)| {
            (
                <P as CentroidCount<ImageType, [[(); 3]; 3], [(); C]>>::centroid_count(
                    image,
                    &patch_centroids,
                ),
                *class,
            )
        })
        .collect();
    let patch_dist_centroids: Vec<Vec<u32>> =
        patch_count_lloyds(&patch_dists, 0, 3, patch_centroids.len(), 1000, 1);
    let patch_bag_cluster_class_dists: Vec<[u32; C]> =
        cluster::class_dist::<C>(&patch_dists, &patch_dist_centroids);
    //{
    //    // sanity checks
    //    assert_eq!(patch_bag_cluster_class_dists.len(), patch_dist_centroids.len());
    //    let sum: u32 = patch_bag_cluster_class_dists.iter().flatten().sum();
    //    assert_eq!(sum as usize, N_EXAMPLES);
    //}

    // the i32 is 1/2 the sum of the counts
    let sparse_patch_bags: Vec<(Vec<(usize, u32)>, i32, [u32; C])> = patch_dist_centroids
        .par_iter()
        .zip(patch_bag_cluster_class_dists.par_iter())
        .map(|(patch_counts, class_dist)| {
            let bag = sparsify_centroid_count(patch_counts, 0);
            let n: u32 = bag.iter().map(|(_, c)| *c).sum();
            (bag, n as i32 / 2, *class_dist)
        })
        .collect();
    (patch_centroids, sparse_patch_bags)
}

// with no hidden bit.
fn init_class_act_cache<H: BitArray + IncrementFracCounters + Sync, const C: usize>(
    patch_acts: &Vec<H>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
    aux_weights: &[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C],
) -> Vec<[i32; C]>
where
    i8: Element<[(); C]> + Element<H::BitShape>,
    u32: Element<H::BitShape>,
    i32: Element<[(); C], Array = [i32; C]> + Element<H::BitShape>,
    H::BitShape: Map<u32, i32>,
    [u32; 2]: Element<H::BitShape>,
    [i32; C]: Sync + Send,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, [(); C]>,
    <i8 as Element<<H as BitArray>::BitShape>>::Array: Sync,
    <i32 as bitnn::shape::Element<[(); C]>>::Array: Sync + Send,
    <u32 as Element<<H as BitArray>::BitShape>>::Array: Default,
{
    patch_bags
        .par_iter()
        .map(|(patch_bag, n, _)| {
            let (hidden_act_n, hidden_act_counts) = patch_bag
                .iter()
                .fold(<(usize, <u32 as Element<<H as BitArray>::BitShape>>::Array)>::default(), |mut acc, &(index, count)| {
                    patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                    acc
                });
            let hidden_acts = <<H as BitArray>::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| *count as i32 - n);
            <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<H, [(); C]>>::iiivmm(&aux_weights, &hidden_acts)
        })
        .collect()
}

fn prepare_class_act_cache<P: BitArray + Sync, H: BitArray, const C: usize>(
    patch_centroids: &Vec<P>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
    class_act_cache: &Vec<[i32; C]>,
    patch_weights_one_chan: &P::TritArrayType,
    patch_chan_threshold: u32,
    aux_weights_one_chan: &[i8; C],
) -> Vec<[i32; C]>
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    [u32; 2]: Element<H::BitShape>,
    (<P as BitArray>::TritArrayType, u32): Element<H::BitShape>,
    <(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array:
        BTBVMM<P, H>,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, [(); C]>,
    <P as BitArray>::TritArrayType: Sync,
{
    let patch_act_hidden_bits: Vec<bool> = patch_centroids
        .par_iter()
        .map(|patch| {
            <<(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array as BTBVMM<P, H>>::btbvmm_one_bit(
                patch_weights_one_chan,
                patch_chan_threshold,
                patch,
            )
        })
        .collect();

    patch_bags
        .par_iter()
        .zip(class_act_cache.par_iter())
        .map(|((patch_bag, n, _), partial_acts)| {
            let updated_act_count: u32 = patch_bag.iter().map(|&(index, count)| patch_act_hidden_bits[index] as u32 * count).sum();
            let updated_act = updated_act_count as i32 - n;
            <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<H, [(); C]>>::subtract_input_from_iiivmm_acts(&aux_weights_one_chan, updated_act, &partial_acts)
        })
        .collect()
}

fn sum_loss_one_chan<P: BitArray + Sync, H: BitArray, const C: usize>(
    patch_centroids: &Vec<P>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
    class_act_cache: &Vec<[i32; C]>,
    patch_weights_one_chan: &P::TritArrayType,
    patch_chan_threshold: u32,
    aux_weights_one_chan: &[i8; C],
) -> u64
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    [u32; 2]: Element<H::BitShape>,
    (<P as BitArray>::TritArrayType, u32): Element<H::BitShape>,
    <(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array:
        BTBVMM<P, H>,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, [(); C]>,
    <P as BitArray>::TritArrayType: Sync,
{
    let patch_act_hidden_bits: Vec<bool> = patch_centroids
        .par_iter()
        .map(|patch| {
            <<(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array as BTBVMM<P, H>>::btbvmm_one_bit(
                patch_weights_one_chan,
                patch_chan_threshold,
                patch,
            )
        })
        .collect();

    patch_bags
        .par_iter()
        .zip(class_act_cache.par_iter())
        .map(|((patch_bag, n, class_counts), partial_acts)| {
            let updated_act_count: u32 = patch_bag
                .iter()
                .map(|&(index, count)| patch_act_hidden_bits[index] as u32 * count)
                .sum();
            let updated_act = updated_act_count as i32 - n;
            let class_acts: [i32; C] =
                <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<
                    H,
                    [(); C],
                >>::update_iiivmm_acts(
                    &aux_weights_one_chan, updated_act, &partial_acts
                );

            class_counts
                .iter()
                .enumerate()
                .map(|(class, &count)| act_loss(&class_acts, class) * count as u64)
                .sum::<u64>()
        })
        .sum()
}

type PatchType = [[b32; 3]; 3];
type HiddenType = [b32; 2];

const N_EXAMPLES: usize = 5_000;

fn main() {
    let start = Instant::now();
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(22))
        //.num_threads(16)
        .build_global()
        .unwrap();

    let mut rng = Hc128Rng::seed_from_u64(0);

    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let examples: Vec<_> = int_examples_32
        .par_iter()
        .map(|(image, class)| (image.pixel_map(|&p| unary(p)), *class))
        .collect();

    let cluster_start = Instant::now();
    let (patch_centroids, patch_bags) =
        cluster_patches_and_images::<PatchType, StaticImage<b32, 32usize, 32usize>, 10>(&examples);
    println!("cluster time: {:?}", cluster_start.elapsed());

    let mut patch_weights = {
        let trits: <<PatchType as BitArray>::TritArrayType as Element<
            <HiddenType as BitArray>::BitShape,
        >>::Array = rng.gen();

        <<HiddenType as BitArray>::BitShape as Map<
            <PatchType as BitArray>::TritArrayType,
            (<PatchType as BitArray>::TritArrayType, u32),
        >>::map(&trits, |&weights| {
            let mut target = <(<PatchType as BitArray>::TritArrayType, u32)>::default();
            target.0 = weights;
            target.1 = (<<PatchType as BitArray>::BitShape as Shape>::N as u32
                - target.0.mask_zeros())
                / 2;
            target
        })
    };
    let aux_weights = <[(
        <i8 as Element<<HiddenType as BitArray>::BitShape>>::Array,
        i8,
    ); 10]>::default();

    let patch_start = Instant::now();
    let patch_acts: Vec<HiddenType> = patch_centroids
        .iter()
        .map(|patch| patch_weights.btbvmm(patch))
        .collect();
    println!("patch act time: {:?}", patch_start.elapsed());
    //dbg!(patch_acts);

    let aux_start = Instant::now();
    let aux_weights = aux_weights.descend(&patch_acts, &patch_bags, 20, 3, 50);
    //dbg!(&aux_weights);
    println!("aux time: {:?}", aux_start.elapsed());

    // for all hidden chans
    let class_act_cache: Vec<[i32; 10]> =
        init_class_act_cache::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);

    let bit_index = (1, (7, ()));
    let aux_weights_one_chan: [i8; 10] = <[(); 10] as Map<
        (
            <i8 as Element<<HiddenType as BitArray>::BitShape>>::Array,
            i8,
        ),
        i8,
    >>::map(&aux_weights, |(class_row, _)| {
        *class_row.index_get(&bit_index)
    });
    let (patch_weights_one_chan, patch_chan_threshold) = patch_weights.index_get(&bit_index);

    // for one hidden chan
    let class_act_cache_one_chan = prepare_class_act_cache::<PatchType, HiddenType, 10>(
        &patch_centroids,
        &patch_bags,
        &class_act_cache,
        &patch_weights_one_chan,
        *patch_chan_threshold,
        &aux_weights_one_chan,
    );

    let n_iters = 1000;
    let sum_loss_start = Instant::now();
    for i in 0..n_iters {
        let sum_loss = sum_loss_one_chan::<PatchType, HiddenType, 10>(
            &patch_centroids,
            &patch_bags,
            &class_act_cache_one_chan,
            &patch_weights_one_chan,
            *patch_chan_threshold,
            &aux_weights_one_chan,
        );
    }
    println!("sum loss time: {:?}", sum_loss_start.elapsed() / n_iters);
}
