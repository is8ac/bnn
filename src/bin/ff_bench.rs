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
    fn egd_descend(
        self,
        patch_bags: &Vec<(Vec<(usize, u32)>, i32, <u32 as Element<C>>::Array)>,
        patch_acts: &Vec<I>,
        weight_delta: i8,
        slow_and_panicky: bool,
    ) -> Self;
}

impl<I, const C: usize> IIIVMM<I, [(); C]> for [(<i8 as Element<I::BitShape>>::Array, i8); C]
where
    Self: Sync,
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
        >>::zip_map(self, grads, |(w, b), (_, (wg, bg))| {
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
    fn egd_descend(
        self,
        patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
        patch_acts: &Vec<I>,
        weight_delta: i8,
        slow_and_panicky: bool,
    ) -> Self {
        let aux_inputs: Vec<(<i32 as Element<<I as BitArray>::BitShape>>::Array, [u32; C])> =
            patch_bags
                .par_iter()
                .map(|(patch_bag, n, class_counts)| {
                    let (_, hidden_act_counts) = patch_bag.iter().fold(
                        <(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array)>::default(),
                        |mut acc, &(index, count)| {
                            patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                            acc
                        },
                    );
                    let hidden_acts = <<I as BitArray>::BitShape as Map<u32, i32>>::map(
                        &hidden_act_counts,
                        |count| *count as i32 - n,
                    );
                    (hidden_acts, *class_counts)
                })
                .collect();

        let (cur_sum_loss, n_correct, n) = sum_loss_correct::<I, C>(&patch_acts, patch_bags, &self);
        dbg!(n_correct as f64 / n as f64);

        let class_acts: Vec<[i32; C]> = aux_inputs
            .par_iter()
            .map(|(input, _)| {
                <[(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<
                    I,
                    [(); C],
                >>::iiivmm(&self, &input)
            })
            .collect();

        (0..C)
            .fold(
                (self, class_acts, cur_sum_loss),
                |(mut aux_weights, class_acts_cache, cur_sum_loss), class_index| {
                    let acc = {
                        let new_class_acts: Vec<[i32; C]> = class_acts_cache
                            .par_iter()
                            .cloned()
                            .map(|(mut new_class_acts)| {
                                new_class_acts[class_index] += weight_delta as i32;
                                new_class_acts
                            })
                            .collect();

                        let new_sum_loss: u64 = new_class_acts
                            .par_iter()
                            .zip(patch_bags.par_iter())
                            .map(|(new_class_acts, (_, _, class_counts))| {
                                class_counts
                                    .iter()
                                    .enumerate()
                                    .map(|(class, &count)| {
                                        act_loss(&new_class_acts, class) * count as u64
                                    })
                                    .sum::<u64>()
                            })
                            .sum();
                        if slow_and_panicky {
                            let backup = aux_weights[class_index].1;
                            aux_weights[class_index].1 =
                                aux_weights[class_index].1.saturating_add(weight_delta);
                            let (true_sum_loss, _, _) =
                                sum_loss_correct::<I, C>(patch_acts, &patch_bags, &aux_weights);
                            aux_weights[class_index].1 = backup;
                            assert_eq!(new_sum_loss, true_sum_loss);
                        }

                        if new_sum_loss < cur_sum_loss {
                            //if false {
                            dbg!(new_sum_loss);
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
                                &class_acts_cache,
                                &chan_index,
                                class_index,
                                weight_delta as i32,
                            );
                            if slow_and_panicky {
                                let backup = *aux_weights[class_index].0.index_get(&chan_index);
                                *aux_weights[class_index].0.index_get_mut(&chan_index) =
                                    aux_weights[class_index]
                                        .0
                                        .index_get(&chan_index)
                                        .saturating_add(weight_delta);
                                let (true_sum_loss, n_correct, n) =
                                    sum_loss_correct::<I, C>(patch_acts, &patch_bags, &aux_weights);
                                *aux_weights[class_index].0.index_get_mut(&chan_index) = backup;
                                assert_eq!(new_sum_loss, true_sum_loss);
                            }
                            if new_sum_loss < cur_sum_loss {
                                //dbg!(new_sum_loss);
                                *aux_weights[class_index].0.index_get_mut(&chan_index) =
                                    aux_weights[class_index]
                                        .0
                                        .index_get(&chan_index)
                                        .saturating_add(weight_delta);
                                let class_acts_cache: Vec<[i32; C]> = aux_inputs
                                    .par_iter()
                                    .zip(class_acts_cache.par_iter().cloned())
                                    .map(|((input, _), mut new_acts)| {
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
            .0
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

trait DescendPatchWeights<P, H: BitArray, const C: usize>
where
    i8: Element<H::BitShape>,
{
    fn descend(
        self,
        patch_centroids: &Vec<P>,
        patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
        aux_weights: &[(<i8 as Element<H::BitShape>>::Array, i8); C],
        class_act_cache: &Vec<[i32; C]>,
        sum_loss: u64,
        delta_sign: bool,
    ) -> Self;
}

impl<P, H, const C: usize> DescendPatchWeights<P, H, { C }>
    for <(P::TritArrayType, u32) as Element<H::BitShape>>::Array
where
    P: BitArray + Sync,
    H: Sync + BitArray + BitArrayOPs + IncrementFracCounters,
    i8: Element<H::BitShape>,
    (<P as BitArray>::TritArrayType, u32): Element<<H as BitArray>::BitShape>,
    Self: BTBVMM<P, H>,
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: IIIVMM<H, [(); C]> + Sync,
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
{
    fn descend(
        self,
        patch_centroids: &Vec<P>,
        patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
        aux_weights: &[(<i8 as Element<H::BitShape>>::Array, i8); C],
        class_act_cache: &Vec<[i32; C]>,
        sum_loss: u64,
        delta_sign: bool,
    ) -> Self {
        let patch_acts: Vec<H> = patch_centroids
            .iter()
            .map(|patch| self.btbvmm(patch))
            .collect();
        let class_act_cache: Vec<[i32; C]> =
            init_class_act_cache::<H, C>(&patch_acts, &patch_bags, &aux_weights);
        let (n_bits, patch_weights, _class_act_cache, _new_sum_loss) =
            <H as BitArray>::BitShape::indices().fold(
                (0, self, class_act_cache, sum_loss),
                |(n_bits, mut weights, class_act_cache, cur_sum_loss),
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

                    let (cur_sum_loss, chan_weights) = {
                        let new_chan_threshold = if delta_sign {
                            chan_weights.1 + 1
                        } else {
                            chan_weights.1.saturating_sub(1)
                        };
                        let new_chan_weights = (chan_weights.0, new_chan_threshold);
                        let threshold_new_sum_loss = sum_loss_one_chan::<P, H, C>(
                            &patch_centroids,
                            &patch_bags,
                            &class_act_cache_one_chan,
                            &new_chan_weights,
                            &aux_weights_one_chan,
                        );
                        if threshold_new_sum_loss < cur_sum_loss {
                            dbg!(threshold_new_sum_loss);
                            (threshold_new_sum_loss, new_chan_weights)
                        } else {
                            (cur_sum_loss, chan_weights)
                        }
                    };
                    //dbg!();
                    let (chan_bits, chan_weights, new_sum_loss) =
                        <P as BitArray>::BitShape::indices().fold(
                            (0, chan_weights, cur_sum_loss),
                            |(n_bits, (chan_weights, chan_threshold), cur_sum_loss), index| {
                                //let new_trit = chan_weights.get_trit(&index);
                                //let new_trit = None;
                                let new_trit = if let Some(_) = chan_weights.get_trit(&index) {
                                    None
                                } else {
                                    Some(delta_sign)
                                };
                                //dbg!(new_trit);
                                let perturbed_chan_weights =
                                    (chan_weights.set_trit(new_trit, &index), chan_threshold);

                                let new_sum_loss = sum_loss_one_chan::<P, H, C>(
                                    &patch_centroids,
                                    &patch_bags,
                                    &class_act_cache_one_chan,
                                    &perturbed_chan_weights,
                                    &aux_weights_one_chan,
                                );
                                //dbg!();

                                //println!("{} {}", cur_sum_loss, new_sum_loss);
                                if new_sum_loss < cur_sum_loss {
                                    dbg!(new_sum_loss);
                                    (n_bits + 1, perturbed_chan_weights, new_sum_loss)
                                } else {
                                    (n_bits + 1, (chan_weights, chan_threshold), cur_sum_loss)
                                }
                            },
                        );

                    *weights.index_get_mut(&chan_index) = chan_weights;

                    if true {
                        let patch_acts: Vec<H> = patch_centroids
                            .iter()
                            .map(|patch| weights.btbvmm(patch))
                            .collect();
                        let (true_sum_loss, n_correct, n) =
                            sum_loss_correct::<H, C>(&patch_acts, &patch_bags, &aux_weights);
                        assert_eq!(true_sum_loss, new_sum_loss);
                    }

                    let clean_class_act_cache = prepare_chan_class_act_cache::<P, H, C>(
                        &patch_centroids,
                        &patch_bags,
                        &class_act_cache_one_chan,
                        &chan_weights,
                        &aux_weights_one_chan,
                        1,
                    );
                    (
                        n_bits + chan_bits,
                        weights,
                        clean_class_act_cache,
                        new_sum_loss,
                    )
                },
            );
        patch_weights
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
) -> (Vec<P>, Vec<(Vec<(usize, u32)>, i32, [u32; C])>)
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
    let sparse_patch_bags: Vec<(Vec<(usize, u32)>, i32, [u32; C])> = patch_dist_centroids
        .par_iter()
        .zip(patch_bag_cluster_class_dists.par_iter())
        .map(|(patch_counts, class_dist)| {
            let bag = sparsify_centroid_count(
                patch_counts,
                params.sparsify_centroid_count_filter_threshold,
            );
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
            let (_, hidden_act_counts) = patch_bag
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

fn prepare_chan_class_act_cache<P: BitArray + Sync, H: BitArray, const C: usize>(
    patch_centroids: &Vec<P>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
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
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, [(); C]>,
    <P as BitArray>::TritArrayType: Sync,
{
    let patch_act_hidden_bits: Vec<bool> = patch_centroids
        .par_iter()
        .map(|patch| {
            <<(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array as BTBVMM<P, H>>::btbvmm_one_bit(chan_patch_weights, *chan_threshold, patch)
        })
        .collect();

    patch_bags
        .par_iter()
        .zip(class_act_cache.par_iter())
        .map(|((patch_bag, n, _), partial_acts)| {
            let updated_act_count: u32 = patch_bag.iter().map(|&(index, count)| patch_act_hidden_bits[index] as u32 * count).sum();
            let updated_act = updated_act_count as i32 - n;
            <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<H, [(); C]>>::update_iiivmm_acts(&aux_weights_one_chan, updated_act * sign, &partial_acts)
        })
        .collect()
}

fn sum_loss_one_chan<P: BitArray + Sync, H: BitArray, const C: usize>(
    patch_centroids: &Vec<P>,
    patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
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
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, [(); C]>,
    <P as BitArray>::TritArrayType: Sync,
{
    // the bit of the patch layer outputs which we have mutated.
    let patch_act_hidden_bits: Vec<bool> = patch_centroids
        .par_iter()
        .map(|patch| {
            <<(<P as BitArray>::TritArrayType, u32) as Element<<H as BitArray>::BitShape>>::Array as BTBVMM<P, H>>::btbvmm_one_bit(
                &chan_patch_weights.0,
                chan_patch_weights.1,
                patch,
            )
        })
        .collect();

    // patch bags are (sparce_patch_counts, sparce_patch_counts_threshold, class_dist)
    patch_bags
        .par_iter()
        .zip(class_act_cache.par_iter()) // zip with the class_act_cache which have that hidden bit subtracted.
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

fn aux_sum_loss_one_class<H: BitArray, const C: usize>(
    inputs: &Vec<(<i32 as Element<H::BitShape>>::Array, [u32; C])>,
    class_acts: &Vec<[i32; C]>,
    chan_index: &<H::BitShape as Shape>::Index,
    class_index: usize,
    weight_delta: i32,
) -> u64
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    [u32; 2]: Element<H::BitShape>,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, [(); C]>,
    (<i32 as Element<H::BitShape>>::Array, [u32; C]): Sync,
    <i32 as Element<H::BitShape>>::Array: IndexGet<<H::BitShape as Shape>::Index, Element = i32>,
    [i32; C]: Sync,
    <H::BitShape as Shape>::Index: Sync,
{
    inputs
        .par_iter()
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
    patch_bags: &Vec<(Vec<(usize, u32)>, i32, [u32; C])>,
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
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: IIIVMM<H, [(); C]>,
    H::BitShape: Map<u32, i32>,
{
    patch_bags
        .par_iter()
        .map(|(patch_bag, n, class_counts)| {
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
            let class_acts =
                <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<
                    H,
                    [(); C],
                >>::iiivmm(&aux_weights, &hidden_acts);
            let loss = class_counts
                .iter()
                .enumerate()
                .map(|(class, &count)| act_loss(&class_acts, class) * count as u64)
                .sum::<u64>();
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
    let (patch_centroids, patch_bags): (Vec<PatchType>, Vec<(Vec<(usize, u32)>, i32, [u32; 10])>) =
        cluster_patches_and_images::<PatchType, StaticImage<b32, 32usize, 32usize>, 10>(
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

    let patch_start = Instant::now();
    let patch_acts: Vec<HiddenType> = patch_centroids
        .iter()
        .map(|patch| patch_weights.btbvmm(patch))
        .collect();
    println!("patch act time: {:?}", patch_start.elapsed());
    //dbg!(patch_acts);

    let aux_start = Instant::now();
    //let aux_weights = aux_weights.descend(&patch_acts, &patch_bags, 3, 5, 5);
    //dbg!(&aux_weights);

    let class_act_cache: Vec<[i32; 10]> =
        init_class_act_cache::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);

    let aux_weights = (0..15).fold(aux_weights, |weights, _| {
        weights
            .egd_descend(&patch_bags, &patch_acts, 1, false)
            .egd_descend(&patch_bags, &patch_acts, -1, false)
    });

    let (sum_loss, n_correct, n) =
        sum_loss_correct::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);
    dbg!(n_correct as f64 / n as f64);
    println!("aux time: {:?}", aux_start.elapsed());

    // patch
    /*
    let bit_index = (0, (5, ()));
    let class_act_cache: Vec<[i32; 10]> = init_class_act_cache::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);
    dbg!(&class_act_cache[0..3]);
    // for one hidden chan
    let new_sum_loss = sum_loss_one_chan::<PatchType, HiddenType, 10>(
        &patch_centroids,
        &patch_bags,
        &class_act_cache_one_chan,
        chan_weights,
        *patch_chan_threshold,
        &aux_weights_one_chan,
    );

    let patch_acts: Vec<HiddenType> = patch_centroids.iter().map(|patch| patch_weights.btbvmm(patch)).collect();
    let (true_sum_loss, _, _) = sum_loss_correct::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);
    assert_eq!(true_sum_loss, new_sum_loss);
    */

    let patch_start = Instant::now();
    let class_act_cache: Vec<[i32; 10]> =
        init_class_act_cache::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);
    let patch_delta_sign = false;
    let (n_bits, patch_weights, _class_act_cache, _new_sum_loss) = <HiddenType as BitArray>::BitShape::indices().fold(
        (0, patch_weights, class_act_cache, sum_loss),
        |(n_bits, mut weights, class_act_cache, cur_sum_loss), chan_index| {
            let aux_weights_one_chan: [i8; 10] =
                <[(); 10] as Map<(<i8 as Element<<HiddenType as BitArray>::BitShape>>::Array, i8), i8>>::map(&aux_weights, |(class_row, _)| *class_row.index_get(&chan_index));
            let chan_weights = *weights.index_get(&chan_index);

            let class_act_cache_one_chan =
                prepare_chan_class_act_cache::<PatchType, HiddenType, 10>(&patch_centroids, &patch_bags, &class_act_cache, &chan_weights, &aux_weights_one_chan, -1);

            let (cur_sum_loss, chan_weights) = {
                let new_chan_threshold = if patch_delta_sign { chan_weights.1 + 1 } else { chan_weights.1 - 1 };
                let new_chan_weights = (chan_weights.0, new_chan_threshold);
                let threshold_new_sum_loss =
                    sum_loss_one_chan::<PatchType, HiddenType, 10>(&patch_centroids, &patch_bags, &class_act_cache_one_chan, &new_chan_weights, &aux_weights_one_chan);
                if threshold_new_sum_loss < cur_sum_loss {
                    dbg!(threshold_new_sum_loss);
                    (threshold_new_sum_loss, new_chan_weights)
                } else {
                    (cur_sum_loss, chan_weights)
                }
            };
            //dbg!();
            let (chan_bits, chan_weights, new_sum_loss) =
                <PatchType as BitArray>::BitShape::indices().fold((0, chan_weights, cur_sum_loss), |(n_bits, (chan_weights, chan_threshold), cur_sum_loss), index| {
                    //let new_trit = chan_weights.get_trit(&index);
                    //let new_trit = None;
                    let new_trit = if let Some(_) = chan_weights.get_trit(&index) { None } else { Some(patch_delta_sign) };
                    //dbg!(new_trit);
                    let perturbed_chan_weights = chan_weights.set_trit(new_trit, &index);

                    let new_sum_loss = sum_loss_one_chan::<PatchType, HiddenType, 10>(
                        &patch_centroids,
                        &patch_bags,
                        &class_act_cache_one_chan,
                        &(perturbed_chan_weights, chan_threshold),
                        &aux_weights_one_chan,
                    );
                    //dbg!();

                    //println!("{} {}", cur_sum_loss, new_sum_loss);
                    if new_sum_loss < cur_sum_loss {
                        dbg!(new_sum_loss);
                        (n_bits + 1, (perturbed_chan_weights, chan_threshold), new_sum_loss)
                    } else {
                        (n_bits + 1, (chan_weights, chan_threshold), cur_sum_loss)
                    }
                });

            *weights.index_get_mut(&chan_index) = chan_weights;

            if true {
                let patch_acts: Vec<HiddenType> = patch_centroids.iter().map(|patch| weights.btbvmm(patch)).collect();
                let (true_sum_loss, n_correct, n) = sum_loss_correct::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);
                assert_eq!(true_sum_loss, new_sum_loss);
            }

            let clean_class_act_cache =
                prepare_chan_class_act_cache::<PatchType, HiddenType, 10>(&patch_centroids, &patch_bags, &class_act_cache_one_chan, &chan_weights, &aux_weights_one_chan, 1);
            (n_bits + chan_bits, weights, clean_class_act_cache, new_sum_loss)
        },
    );
    dbg!(patch_start.elapsed());
    dbg!(n_bits);

    let patch_acts: Vec<HiddenType> = patch_centroids
        .iter()
        .map(|patch| patch_weights.btbvmm(patch))
        .collect();
    let (_, n_correct, n) =
        sum_loss_correct::<HiddenType, 10>(&patch_acts, &patch_bags, &aux_weights);
    dbg!(n_correct as f64 / n as f64);
}
