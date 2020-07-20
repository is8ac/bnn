#![feature(move_ref_pattern)]
#![feature(const_generics)]
use bitnn::bits::{
    b32, BitArray, BitArrayOPs, BitMap, BitMapPack, IncrementFracCounters, MaskedDistance, SetTrit,
    TritArray,
};
use bitnn::cluster::{
    self, patch_count_lloyds, sparsify_centroid_count, CentroidCount, ImagePatchLloyds,
};
use bitnn::datasets::cifar;
use bitnn::image2d::{Image2D, PixelMap, StaticImage};
use bitnn::shape::{Element, IndexGet, Map, MapMut, Shape, ZipFold, ZipMap};
use bitnn::unary::to_10;
use rand::distributions;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

trait IIIVMM<H: BitArray, const C: usize>
where
    [u32; 2]: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    Self: Sized,
{
    fn iiivmm(&self, input: &<i32 as Element<H::BitShape>>::Array) -> [i32; C];
    fn update_iiivmm_acts(weights: &[i8; C], input: i32, acts: &[i32; C]) -> [i32; C];
    fn egd_descend(
        self,
        cur_sum_loss: u64,
        class_counts: &Vec<[u32; C]>,
        aux_inputs: &Vec<<i32 as Element<<H as BitArray>::BitShape>>::Array>,
        class_acts: Vec<[i32; C]>,
        weight_delta: i8,
        slow_and_panicky: bool,
    ) -> (Self, Vec<[i32; C]>, u64);
    fn optimize(
        //weights: [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C],
        self,
        class_act_cache: Vec<[i32; C]>,
        class_counts: &Vec<[u32; C]>,
        aux_inputs: &Vec<<i32 as Element<H::BitShape>>::Array>,
        sum_loss: u64,
        i: usize,
    ) -> (Self, Vec<[i32; C]>, u64)
    where
        [u32; 2]: Element<H::BitShape>,
        i32: Element<H::BitShape>,
        i8: Element<H::BitShape>,
    {
        (0..i).fold(
            (self, class_act_cache, sum_loss),
            |(weights, class_act_cache, sum_loss), _| {
                let (weights, class_act_cache, sum_loss) = weights.egd_descend(
                    sum_loss,
                    &class_counts,
                    &aux_inputs,
                    class_act_cache,
                    1,
                    false,
                );
                weights.egd_descend(
                    sum_loss,
                    &class_counts,
                    &aux_inputs,
                    class_act_cache,
                    -1,
                    false,
                )
            },
        )
    }
}

impl<I, const C: usize> IIIVMM<I, C> for [(<i8 as Element<I::BitShape>>::Array, i8); C]
where
    Self: Sync + Sized,
    <i8 as Element<I::BitShape>>::Array: Sync,
    I: BitArray + IncrementFracCounters + Sync,
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
    <i32 as Element<I::BitShape>>::Array:
        IndexGet<<I::BitShape as Shape>::Index, Element = i32> + Sync + Send,
    <u32 as Element<<I as BitArray>::BitShape>>::Array: Default,
    <<I as BitArray>::BitShape as Shape>::Index: Sync,
    <<I as BitArray>::BitShape as Shape>::IndexIter:
        Iterator<Item = <<I as BitArray>::BitShape as Shape>::Index>,
    [i32; C]: std::fmt::Debug + Eq,
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
    fn egd_descend(
        self,
        cur_sum_loss: u64,
        class_counts: &Vec<[u32; C]>,
        aux_inputs: &Vec<<i32 as Element<<I as BitArray>::BitShape>>::Array>,
        class_acts: Vec<[i32; C]>,
        weight_delta: i8,
        slow_and_panicky: bool,
    ) -> (Self, Vec<[i32; C]>, u64) {
        (0..C).fold(
            (self, class_acts, cur_sum_loss),
            |(mut aux_weights, class_acts_cache, cur_sum_loss), class_index| {
                let acc = {
                    let new_class_acts: Vec<[i32; C]> = class_acts_cache
                        .par_iter()
                        .cloned()
                        .map(|mut new_class_acts| {
                            new_class_acts[class_index] += weight_delta as i32;
                            new_class_acts
                        })
                        .collect();

                    let new_sum_loss: u64 = new_class_acts
                        .par_iter()
                        .zip(class_counts.par_iter())
                        .map(|(new_class_acts, class_counts)| {
                            class_counts
                                .iter()
                                .enumerate()
                                .map(|(class, &count)| {
                                    act_loss(&new_class_acts, class) * count as u64
                                })
                                .sum::<u64>()
                        })
                        .sum();
                    //if slow_and_panicky {
                    //    let backup = aux_weights[class_index].1;
                    //    aux_weights[class_index].1 = aux_weights[class_index].1.saturating_add(weight_delta);
                    //    let (true_sum_loss, _, _) = sum_loss_correct::<I, C>(patch_acts, &patch_bags, &aux_weights);
                    //    aux_weights[class_index].1 = backup;
                    //    assert_eq!(new_sum_loss, true_sum_loss);
                    //}

                    if new_sum_loss < cur_sum_loss {
                        //dbg!(new_sum_loss);
                        aux_weights[class_index].1 =
                            aux_weights[class_index].1.saturating_add(weight_delta);
                        (aux_weights, new_class_acts, new_sum_loss)
                    } else {
                        (aux_weights, class_acts_cache, cur_sum_loss)
                    }
                };
                <I as BitArray>::BitShape::indices().fold(
                    acc,
                    |(mut aux_weights, class_acts_cache, cur_sum_loss), chan_index| {
                        let new_sum_loss = aux_sum_loss_one_class::<I, C>(
                            &aux_inputs,
                            &class_counts,
                            &class_acts_cache,
                            &chan_index,
                            class_index,
                            weight_delta as i32,
                        );
                        //if slow_and_panicky {
                        //    let backup = *aux_weights[class_index].0.index_get(&chan_index);
                        //    *aux_weights[class_index].0.index_get_mut(&chan_index) = aux_weights[class_index].0.index_get(&chan_index).saturating_add(weight_delta);
                        //    let (true_sum_loss, _, _) = sum_loss_correct::<I, C>(patch_acts, &patch_bags, &aux_weights);
                        //    *aux_weights[class_index].0.index_get_mut(&chan_index) = backup;
                        //    assert_eq!(new_sum_loss, true_sum_loss);
                        //}
                        if new_sum_loss < cur_sum_loss {
                            //dbg!(new_sum_loss);
                            *aux_weights[class_index].0.index_get_mut(&chan_index) = aux_weights
                                [class_index]
                                .0
                                .index_get(&chan_index)
                                .saturating_add(weight_delta);
                            let class_acts_cache: Vec<[i32; C]> = aux_inputs
                                .par_iter()
                                .zip(class_acts_cache.par_iter().cloned())
                                .map(|(input, mut new_acts)| {
                                    new_acts[class_index] +=
                                        input.index_get(&chan_index) * weight_delta as i32;
                                    new_acts
                                })
                                .collect();
                            (aux_weights, class_acts_cache, new_sum_loss)
                        } else {
                            (aux_weights, class_acts_cache, cur_sum_loss)
                        }
                    },
                )
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

trait DescendPatchWeights<P: BitArray, H: BitArray, const C: usize>
where
    i8: Element<H::BitShape>,
    Self: Sized,
    i32: Element<H::BitShape>,
{
    fn descend(
        self,
        patch_centroids: &Vec<P>,
        patch_bags: &Vec<(Vec<(usize, u32)>, i32)>,
        class_counts: &Vec<[u32; C]>,
        aux_weights: &[(<i8 as Element<H::BitShape>>::Array, i8); C],
        class_act_cache: Vec<[i32; C]>,
        aux_inputs: Vec<<i32 as Element<<H as BitArray>::BitShape>>::Array>,
        sum_loss: u64,
        delta_sign: bool,
    ) -> (
        Self,
        Vec<<i32 as Element<<H as BitArray>::BitShape>>::Array>,
        Vec<[i32; C]>,
        u64,
        (usize, usize),
    );
    fn descend_patch_chan(
        chan_weights: (P::TritArrayType, u32),
        aux_weights_one_chan: &[i8; C],
        delta_sign: bool,
        patch_centroids: &Vec<P>,
        patch_bags: &Vec<(Vec<(usize, u32)>, i32)>,
        class_counts: &Vec<[u32; C]>,
        class_act_cache_one_chan: &Vec<[i32; C]>,
        sum_loss: u64,
    ) -> ((P::TritArrayType, u32), u64, (usize, usize));
}

impl<P, H, const C: usize> DescendPatchWeights<P, H, { C }>
    for <(P::TritArrayType, u32) as Element<H::BitShape>>::Array
where
    P: BitArray + Sync,
    H: Sync + BitArray + BitArrayOPs + IncrementFracCounters,
    i8: Element<H::BitShape>,
    (<P as BitArray>::TritArrayType, u32): Element<<H as BitArray>::BitShape>,
    Self: BTBVMM<P, H>,
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: IIIVMM<H, C> + Sync,
    i32: Element<H::BitShape>,
    [u32; 2]: Element<H::BitShape>,
    u32: Element<H::BitShape>,
    bool: Element<H::BitShape>,
    <i8 as Element<H::BitShape>>::Array: Sync,
    <u32 as Element<H::BitShape>>::Array: Default,
    H::BitShape: Map<u32, i32>,
    <<H as BitArray>::BitShape as Shape>::IndexIter:
        Iterator<Item = <<H as BitArray>::BitShape as Shape>::Index>,
    [i32; C]: Default,
    [i8; C]: Default,
    <i8 as Element<H::BitShape>>::Array: IndexGet<<H::BitShape as Shape>::Index, Element = i8>,
    <(P::TritArrayType, u32) as Element<H::BitShape>>::Array:
        IndexGet<<H::BitShape as Shape>::Index, Element = (P::TritArrayType, u32)>,
    P::TritArrayType: Sync + SetTrit + Copy,
    <<P as BitArray>::BitShape as Shape>::IndexIter: Iterator<Item = <P::BitShape as Shape>::Index>,
    <i32 as Element<<H as BitArray>::BitShape>>::Array: Send,
    <i32 as Element<H::BitShape>>::Array: IndexGet<<H::BitShape as Shape>::Index, Element = i32>,
    <H::BitShape as Shape>::Index: Sync,
    <i32 as Element<<H as BitArray>::BitShape>>::Array: std::fmt::Debug + Eq,
    P::TritArrayType: Eq + std::fmt::Debug,
{
    fn descend(
        self,
        patch_centroids: &Vec<P>,
        patch_bags: &Vec<(Vec<(usize, u32)>, i32)>,
        class_counts: &Vec<[u32; C]>,
        aux_weights: &[(<i8 as Element<H::BitShape>>::Array, i8); C],
        class_act_cache: Vec<[i32; C]>,
        mut aux_inputs: Vec<<i32 as Element<<H as BitArray>::BitShape>>::Array>,
        sum_loss: u64,
        delta_sign: bool,
    ) -> (
        Self,
        Vec<<i32 as Element<<H as BitArray>::BitShape>>::Array>,
        Vec<[i32; C]>,
        u64,
        (usize, usize),
    ) {
        <H as BitArray>::BitShape::indices().fold(
            (self, aux_inputs, class_act_cache, sum_loss, (0, 0)),
            |(mut weights, mut aux_inputs, class_act_cache, cur_sum_loss, (n_iters, n_updates)),
             chan_index: <H::BitShape as Shape>::Index| {
                let aux_weights_one_chan: [i8; C] = <[(); C] as Map<
                    (<i8 as Element<<H as BitArray>::BitShape>>::Array, i8),
                    i8,
                >>::map(
                    &aux_weights,
                    |(class_row, _)| *class_row.index_get(&chan_index),
                );
                let chan_weights = *weights.index_get(&chan_index);
                let class_act_cache_one_chan = prepare_chan_class_act_cache::<P, H, C>(
                    &patch_centroids,
                    &patch_bags,
                    &class_act_cache,
                    &chan_weights,
                    &aux_weights_one_chan,
                    -1,
                );

                let (chan_weights, new_sum_loss, (chan_iters, chan_updates)) =
                    Self::descend_patch_chan(
                        chan_weights.clone(),
                        &aux_weights_one_chan,
                        delta_sign,
                        patch_centroids,
                        patch_bags,
                        class_counts,
                        &class_act_cache_one_chan,
                        cur_sum_loss,
                    );

                *weights.index_get_mut(&chan_index) = chan_weights;
                // the bit of the patch layer outputs which we have mutated.
                let patch_act_hidden_bits: Vec<bool> = patch_centroids
                    .par_iter()
                    .map(|patch| {
                        <<(<P as BitArray>::TritArrayType, u32) as Element<
                            <H as BitArray>::BitShape,
                        >>::Array as BTBVMM<P, H>>::btbvmm_one_bit(
                            &chan_weights.0,
                            chan_weights.1,
                            patch,
                        )
                    })
                    .collect();

                // we now mutate aux_inputs, rewriting the `chan_index`th element.
                patch_bags
                    .par_iter()
                    .zip(aux_inputs.par_iter_mut())
                    .for_each(|((patch_bag, n), aux_input)| {
                        *aux_input.index_get_mut(&chan_index) = patch_bag
                            .iter()
                            .map(|&(index, count)| patch_act_hidden_bits[index] as u32 * count)
                            .sum::<u32>()
                            as i32
                            - n;
                    });

                let clean_class_act_cache = prepare_chan_class_act_cache::<P, H, C>(
                    &patch_centroids,
                    &patch_bags,
                    &class_act_cache_one_chan,
                    &chan_weights,
                    &aux_weights_one_chan,
                    1,
                );
                (
                    weights,
                    aux_inputs,
                    clean_class_act_cache,
                    new_sum_loss,
                    (n_iters + chan_iters, n_updates + chan_updates),
                )
            },
        )
    }
    fn descend_patch_chan(
        chan_weights: (P::TritArrayType, u32),
        aux_weights_one_chan: &[i8; C],
        delta_sign: bool,
        patch_centroids: &Vec<P>,
        patch_bags: &Vec<(Vec<(usize, u32)>, i32)>,
        class_counts: &Vec<[u32; C]>,
        class_act_cache_one_chan: &Vec<[i32; C]>,
        sum_loss: u64,
    ) -> ((P::TritArrayType, u32), u64, (usize, usize)) {
        let (sum_loss, chan_weights) = {
            let new_chan_threshold = if delta_sign {
                chan_weights.1 + 1
            } else {
                chan_weights.1.saturating_sub(1)
            };
            let new_chan_weights = (chan_weights.0, new_chan_threshold);
            let threshold_new_sum_loss = sum_loss_one_chan::<P, H, C>(
                &patch_centroids,
                &patch_bags,
                &class_counts,
                &class_act_cache_one_chan,
                &new_chan_weights,
                &aux_weights_one_chan,
            );
            if threshold_new_sum_loss < sum_loss {
                (threshold_new_sum_loss, new_chan_weights)
            } else {
                (sum_loss, chan_weights)
            }
        };
        <P as BitArray>::BitShape::indices().fold(
            (chan_weights, sum_loss, (0, 0)),
            |((chan_weights, chan_threshold), cur_sum_loss, (chan_iters, chan_updates)), index| {
                let new_trit = if let Some(_) = chan_weights.get_trit(&index) {
                    None
                } else {
                    Some(delta_sign)
                };
                let perturbed_chan_weights =
                    (chan_weights.set_trit(new_trit, &index), chan_threshold);
                let new_sum_loss = sum_loss_one_chan::<P, H, C>(
                    &patch_centroids,
                    &patch_bags,
                    &class_counts,
                    &class_act_cache_one_chan,
                    &perturbed_chan_weights,
                    &aux_weights_one_chan,
                );
                if new_sum_loss < cur_sum_loss {
                    (
                        perturbed_chan_weights,
                        new_sum_loss,
                        (chan_iters + 1, chan_updates + 1),
                    )
                } else {
                    (
                        (chan_weights, chan_threshold),
                        cur_sum_loss,
                        (chan_iters + 1, chan_updates),
                    )
                }
            },
        )
    }
}

fn unary(input: [u8; 3]) -> b32 {
    to_10(input[0]) | (to_10(input[1]) << 10) | (to_10(input[2]) << 20)
}

fn act_loss<const C: usize>(acts: &[i32; C], class: usize) -> u64 {
    let max_act = acts
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != class)
        .max_by_key(|(_, v)| *v)
        .unwrap()
        .1;
    (max_act - acts[class]).max(0) as u64
}

fn is_correct<const C: usize>(acts: &[i32; C], class: usize) -> bool {
    let (_, max_act) = acts
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != class)
        .max_by_key(|(_, v)| *v)
        .unwrap();
    acts[class] >= *max_act
}

struct ClusterParams {
    /// seed for patch level clustering. Example: 0
    patch_lloyds_seed: u64,
    /// initial number of patch centroids. Actual number after pruning will generally be smaller. Increase for accurecy, decrese for performance. Example: 500
    patch_lloyds_k: usize,
    /// number of clustering iterations. Increase for accuracy, decrease for performance. Example: 3
    patch_lloyds_i: usize,
    /// Patch level clustering prune threshold. Increase for performance, decrese for accurecy. Set to 0 to effectively disable pruning. Example: 1
    patch_lloyds_prune_threshold: usize,
    /// image / patch bag level clustering seed. Example: 0
    image_lloyds_seed: u64,
    /// initial number of image centroids. Actual number after pruning will generally be smaller. Increase for accurecy, decrese for performance. Example: 500
    image_lloyds_k: usize,
    /// Number of image cluster iters. Example: 3
    image_lloyds_i: usize,
    /// image cluster prune threshold. Increase for performance, decrese for accuracy. Set to 0 to effectively disable. Example: 1
    image_lloyds_prune_threshold: usize,
    /// Prune patches within patch bags. Increase for performance, decrease for accuracy. Example: 1
    sparsify_centroid_count_filter_threshold: u32,
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
    params: &ClusterParams,
) -> (Vec<P>, (Vec<(Vec<(usize, u32)>, i32)>, Vec<[u32; C]>))
where
    distributions::Standard: distributions::Distribution<P>,
    [u32; C]: Default + Copy + Sync + Send,
{
    let mut rng = Hc128Rng::seed_from_u64(params.patch_lloyds_seed);
    let patch_centroids = <P as ImagePatchLloyds<_, [[(); 3]; 3]>>::lloyds(
        &mut rng,
        &examples,
        params.patch_lloyds_k,
        params.patch_lloyds_i,
        params.patch_lloyds_prune_threshold,
    );
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
    dbg!();
    let patch_dist_centroids: Vec<Vec<u32>> = patch_count_lloyds(
        &patch_dists,
        params.image_lloyds_seed,
        params.image_lloyds_i,
        patch_centroids.len(),
        params.image_lloyds_k,
        params.image_lloyds_prune_threshold,
    );
    dbg!();
    let patch_bag_cluster_class_dists: Vec<[u32; C]> =
        cluster::class_dist::<C>(&patch_dists, &patch_dist_centroids);
    //{
    //    // sanity checks
    //    assert_eq!(patch_bag_cluster_class_dists.len(), patch_dist_centroids.len());
    //    let sum: u32 = patch_bag_cluster_class_dists.iter().flatten().sum();
    //    assert_eq!(sum as usize, N_EXAMPLES);
    //}

    // the i32 is 1/2 the sum of the counts
    let sparse_patch_bags: Vec<(Vec<(usize, u32)>, i32)> = patch_dist_centroids
        .par_iter()
        .map(|patch_counts| {
            let bag = sparsify_centroid_count(
                patch_counts,
                params.sparsify_centroid_count_filter_threshold,
            );
            let n: u32 = bag.iter().map(|(_, c)| *c).sum();
            (bag, n as i32 / 2)
        })
        .collect();
    (
        patch_centroids,
        (sparse_patch_bags, patch_bag_cluster_class_dists),
    )
}

// with no hidden bit.
fn init_class_act_cache<H: BitArray + IncrementFracCounters + Sync, const C: usize>(
    patch_acts: &Vec<H>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32)>,
    aux_weights: &[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C],
) -> Vec<[i32; C]>
where
    i8: Element<[(); C]> + Element<H::BitShape>,
    u32: Element<H::BitShape>,
    i32: Element<[(); C], Array = [i32; C]> + Element<H::BitShape>,
    H::BitShape: Map<u32, i32>,
    [u32; 2]: Element<H::BitShape>,
    [i32; C]: Sync + Send,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, C>,
    <i8 as Element<<H as BitArray>::BitShape>>::Array: Sync,
    <i32 as bitnn::shape::Element<[(); C]>>::Array: Sync + Send,
    <u32 as Element<<H as BitArray>::BitShape>>::Array: Default,
{
    patch_bags
        .par_iter()
        .map(|(patch_bag, n)| {
            let (_, hidden_act_counts) = patch_bag.iter().fold(
                <(usize, <u32 as Element<<H as BitArray>::BitShape>>::Array)>::default(),
                |mut acc, &(index, count)| {
                    patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                    acc
                },
            );
            let hidden_acts =
                <<H as BitArray>::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| {
                    *count as i32 - n
                });
            <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<H, C>>::iiivmm(
                &aux_weights,
                &hidden_acts,
            )
        })
        .collect()
}

fn prepare_chan_class_act_cache<P: BitArray + Sync, H: BitArray, const C: usize>(
    patch_centroids: &Vec<P>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32)>,
    class_act_cache: &Vec<[i32; C]>,
    (chan_patch_weights, chan_threshold): &(P::TritArrayType, u32),
    aux_weights_one_chan: &[i8; C],
    sign: i32,
) -> Vec<[i32; C]>
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    [u32; 2]: Element<H::BitShape>,
    (<P as BitArray>::TritArrayType, u32): Element<H::BitShape>,
    <(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array:
        BTBVMM<P, H>,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, C>,
    <P as BitArray>::TritArrayType: Sync,
{
    let patch_act_hidden_bits: Vec<bool> = patch_centroids
        .par_iter()
        .map(|patch| <<(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array as BTBVMM<P, H>>::btbvmm_one_bit(chan_patch_weights, *chan_threshold, patch))
        .collect();

    patch_bags
        .par_iter()
        .zip(class_act_cache.par_iter())
        .map(|((patch_bag, n), partial_acts)| {
            let updated_act_count: u32 = patch_bag.iter().map(|&(index, count)| patch_act_hidden_bits[index] as u32 * count).sum();
            let updated_act = updated_act_count as i32 - n;
            <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<H, C>>::update_iiivmm_acts(&aux_weights_one_chan, updated_act * sign, &partial_acts)
        })
        .collect()
}

fn sum_loss_one_chan<P: BitArray + Sync, H: BitArray, const C: usize>(
    patch_centroids: &Vec<P>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32)>,
    class_counts: &Vec<[u32; C]>,
    class_act_cache: &Vec<[i32; C]>,
    chan_patch_weights: &(P::TritArrayType, u32),
    aux_weights_one_chan: &[i8; C],
) -> u64
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    [u32; 2]: Element<H::BitShape>,
    (<P as BitArray>::TritArrayType, u32): Element<H::BitShape>,
    <(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array:
        BTBVMM<P, H>,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, C>,
    <P as BitArray>::TritArrayType: Sync,
{
    // the bit of the patch layer outputs which we have mutated.
    let patch_act_hidden_bits: Vec<bool> = patch_centroids
        .par_iter()
        .map(|patch| <<(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array as BTBVMM<P, H>>::btbvmm_one_bit(&chan_patch_weights.0, chan_patch_weights.1, patch))
        .collect();

    // patch bags are (sparce_patch_counts, sparce_patch_counts_threshold, class_dist)
    patch_bags
        .par_iter()
        .zip(class_counts.par_iter())
        .zip(class_act_cache.par_iter()) // zip with the class_act_cache which have that hidden bit subtracted.
        .map(|(((patch_bag, n), class_counts), partial_acts)| {
            let updated_act_count: u32 = patch_bag.iter().map(|&(index, count)| patch_act_hidden_bits[index] as u32 * count).sum();
            let updated_act = updated_act_count as i32 - n;
            let class_acts: [i32; C] = <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<H, C>>::update_iiivmm_acts(&aux_weights_one_chan, updated_act, &partial_acts);

            class_counts.iter().enumerate().map(|(class, &count)| act_loss(&class_acts, class) * count as u64).sum::<u64>()
        })
        .sum()
}

fn aux_sum_loss_one_class<H: BitArray, const C: usize>(
    inputs: &Vec<<i32 as Element<H::BitShape>>::Array>,
    class_dists: &Vec<[u32; C]>,
    class_acts: &Vec<[i32; C]>,
    chan_index: &<H::BitShape as Shape>::Index,
    class_index: usize,
    weight_delta: i32,
) -> u64
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    [u32; 2]: Element<H::BitShape>,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, C>,
    (<i32 as Element<H::BitShape>>::Array, [u32; C]): Sync,
    <i32 as Element<H::BitShape>>::Array: IndexGet<<H::BitShape as Shape>::Index, Element = i32>,
    [i32; C]: Sync,
    <H::BitShape as Shape>::Index: Sync,
    <i32 as Element<H::BitShape>>::Array: Sync,
{
    inputs
        .par_iter()
        .zip(class_dists.par_iter())
        .zip(class_acts.par_iter())
        .map(|((input, class_counts), acts_cache)| {
            let new_acts = {
                let mut new_acts = *acts_cache;
                new_acts[class_index] =
                    acts_cache[class_index] + input.index_get(chan_index) * weight_delta;
                new_acts
            };

            class_counts
                .iter()
                .enumerate()
                .map(|(class, &count)| act_loss(&new_acts, class) * count as u64)
                .sum::<u64>()
        })
        .sum()
}

fn sum_loss_correct<H: BitArray, const C: usize>(
    patch_acts: &Vec<H>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32)>,
    class_counts: &Vec<[u32; C]>,
    aux_weights: &[(<i8 as Element<H::BitShape>>::Array, i8); C],
) -> (u64, usize, usize)
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    u32: Element<H::BitShape>,
    H: IncrementFracCounters + Sync,
    [u32; 2]: Element<H::BitShape>,
    <i8 as bitnn::shape::Element<<H as bitnn::bits::BitArray>::BitShape>>::Array: Sync,
    <u32 as Element<H::BitShape>>::Array: Default,
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: IIIVMM<H, C>,
    H::BitShape: Map<u32, i32>,
{
    patch_bags
        .par_iter()
        .zip(class_counts.par_iter())
        .map(|((patch_bag, n), class_counts)| {
            let (_, hidden_act_counts) = patch_bag
                .iter()
                .fold(<(usize, <u32 as Element<<H as BitArray>::BitShape>>::Array)>::default(), |mut acc, &(index, count)| {
                    patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                    acc
                });
            let hidden_acts = <<H as BitArray>::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| *count as i32 - n);
            let class_acts = <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<H, C>>::iiivmm(&aux_weights, &hidden_acts);
            let loss = class_counts.iter().enumerate().map(|(class, &count)| act_loss(&class_acts, class) * count as u64).sum::<u64>();
            let n_correct = class_counts
                .iter()
                .enumerate()
                .map(|(class, &count)| is_correct(&class_acts, class) as usize * count as usize)
                .sum::<usize>();
            let n: u32 = class_counts.iter().sum();
            (loss, n_correct, n as usize)
        })
        .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
}

type PatchType = [[b32; 3]; 3];
type HiddenType = [b32; 4];

const N_EXAMPLES: usize = 1_000;

fn main() {
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

    let cluster_params = ClusterParams {
        patch_lloyds_seed: 0,
        patch_lloyds_k: 500,
        patch_lloyds_i: 3,
        patch_lloyds_prune_threshold: 1,
        image_lloyds_seed: 0,
        image_lloyds_i: 3,
        image_lloyds_k: 500,
        image_lloyds_prune_threshold: 1,
        sparsify_centroid_count_filter_threshold: 1,
    };
    dbg!();
    let cluster_start = Instant::now();
    let (patch_centroids, (patch_bags, class_counts)): (
        Vec<PatchType>,
        (Vec<(Vec<(usize, u32)>, i32)>, Vec<[u32; 10]>),
    ) = cluster_patches_and_images::<PatchType, StaticImage<b32, 32usize, 32usize>, 10>(
        &examples,
        &cluster_params,
    );
    dbg!(patch_centroids.len());
    dbg!(patch_bags.len());
    println!("cluster time: {:?}", cluster_start.elapsed());

    let patch_weights = {
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
    //let mut aux_weights = <[(<i8 as Element<<HiddenType as BitArray>::BitShape>>::Array, i8); 10]>::default();
    let aux_weights = <[(); 10] as Map<
        HiddenType,
        (
            <i8 as Element<<HiddenType as BitArray>::BitShape>>::Array,
            i8,
        ),
    >>::map(&rng.gen(), |signs| {
        (signs.bit_map(|sign| SIGNS[1][sign as usize] * 20), 0)
    });

    let aux_start = Instant::now();
    let patch_acts: Vec<HiddenType> = patch_centroids
        .iter()
        .map(|patch| patch_weights.btbvmm(patch))
        .collect();
    let (sum_loss, _, _) =
        sum_loss_correct::<HiddenType, 10>(&patch_acts, &patch_bags, &class_counts, &aux_weights);
    let class_act_cache: Vec<[i32; 10]> =
        init_class_act_cache::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);

    let aux_inputs: Vec<<i32 as Element<<HiddenType as BitArray>::BitShape>>::Array> = patch_bags
        .par_iter()
        .map(|(patch_bag, n)| {
            let (_, hidden_act_counts) = patch_bag.iter().fold(
                <(
                    usize,
                    <u32 as Element<<HiddenType as BitArray>::BitShape>>::Array,
                )>::default(),
                |mut acc, &(index, count)| {
                    patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                    acc
                },
            );
            <<HiddenType as BitArray>::BitShape as Map<u32, i32>>::map(
                &hidden_act_counts,
                |count| *count as i32 - n,
            )
        })
        .collect();

    let (aux_weights, class_act_cache, sum_loss) = <[(
        <i8 as Element<<HiddenType as BitArray>::BitShape>>::Array,
        i8,
    ); 10] as IIIVMM<HiddenType, 10>>::optimize(
        aux_weights,
        class_act_cache,
        &class_counts,
        &aux_inputs,
        sum_loss,
        13,
    );

    let (sum_loss, n_correct, n) =
        sum_loss_correct::<HiddenType, 10>(&patch_acts, &patch_bags, &class_counts, &aux_weights);
    dbg!(n_correct as f64 / n as f64);
    println!("aux time: {:?}", aux_start.elapsed());

    let patch_start = Instant::now();
    let (patch_weights, aux_inputs, class_act_cache, sum_loss, (n_weights, n_updates)) =
        <<(<PatchType as BitArray>::TritArrayType, u32) as Element<
            <HiddenType as BitArray>::BitShape,
        >>::Array as DescendPatchWeights<PatchType, HiddenType, 10>>::descend(
            patch_weights,
            &patch_centroids,
            &patch_bags,
            &class_counts,
            &aux_weights,
            class_act_cache,
            aux_inputs,
            sum_loss,
            true,
        );
    dbg!(patch_start.elapsed() / n_weights as u32);

    dbg!(n_weights);
    dbg!(n_updates);

    let patch_acts: Vec<HiddenType> = patch_centroids
        .iter()
        .map(|patch| patch_weights.btbvmm(patch))
        .collect();
    {
        let (true_sum_loss, n_correct, n) = sum_loss_correct::<HiddenType, 10>(
            &patch_acts,
            &patch_bags,
            &class_counts,
            &aux_weights,
        );
        assert_eq!(sum_loss, true_sum_loss);
        dbg!(n_correct as f64 / n as f64);
    }

    let (aux_weights, class_act_cache, sum_loss) = <[(
        <i8 as Element<<HiddenType as BitArray>::BitShape>>::Array,
        i8,
    ); 10] as IIIVMM<HiddenType, 10>>::optimize(
        aux_weights,
        class_act_cache,
        &class_counts,
        &aux_inputs,
        sum_loss,
        13,
    );

    let patch_start = Instant::now();
    let (patch_weights, aux_inputs, class_act_cache, sum_loss, (n_weights, n_updates)) =
        <<(<PatchType as BitArray>::TritArrayType, u32) as Element<
            <HiddenType as BitArray>::BitShape,
        >>::Array as DescendPatchWeights<PatchType, HiddenType, 10>>::descend(
            patch_weights,
            &patch_centroids,
            &patch_bags,
            &class_counts,
            &aux_weights,
            class_act_cache,
            aux_inputs,
            sum_loss,
            true,
        );
    dbg!(patch_start.elapsed() / n_weights as u32);
    dbg!(n_weights);
    dbg!(n_updates);

    let patch_acts: Vec<HiddenType> = patch_centroids
        .iter()
        .map(|patch| patch_weights.btbvmm(patch))
        .collect();

    let (aux_weights, class_act_cache, sum_loss) = <[(
        <i8 as Element<<HiddenType as BitArray>::BitShape>>::Array,
        i8,
    ); 10] as IIIVMM<HiddenType, 10>>::optimize(
        aux_weights,
        class_act_cache,
        &class_counts,
        &aux_inputs,
        sum_loss,
        13,
    );

    {
        let (true_sum_loss, n_correct, n) = sum_loss_correct::<HiddenType, 10>(
            &patch_acts,
            &patch_bags,
            &class_counts,
            &aux_weights,
        );
        assert_eq!(sum_loss, true_sum_loss);
        dbg!(n_correct as f64 / n as f64);
    }
}
