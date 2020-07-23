use crate::bits::{
    BitArray, BitArrayOPs, BitMap, BitMapPack, IncrementFracCounters, MaskedDistance, SetTrit,
    TritArray,
};
use crate::shape::{Element, IndexGet, Map, Shape, ZipFold, ZipMap};
use rand::distributions;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::time::Instant;

pub trait IIIVMM<H: BitArray, const C: usize>
where
    i32: Element<H::BitShape>,
    Self: Sized,
{
    fn iiivmm(&self, input: &<i32 as Element<H::BitShape>>::Array) -> [i32; C];
    fn update_iiivmm_acts(weights: &[i8; C], input: i32, acts: &[i32; C]) -> [i32; C];
    fn egd_descend(
        self,
        cur_sum_loss: u64,
        class_counts: &[[u32; C]],
        aux_inputs: &[<i32 as Element<<H as BitArray>::BitShape>>::Array],
        class_acts: Vec<[i32; C]>,
        weight_delta: i8,
    ) -> (Self, Vec<[i32; C]>, u64);
    fn optimize(
        //weights: [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C],
        self,
        class_act_cache: Vec<[i32; C]>,
        class_counts: &[[u32; C]],
        aux_inputs: &[<i32 as Element<H::BitShape>>::Array],
        sum_loss: u64,
        i: usize,
    ) -> (Self, Vec<[i32; C]>, u64) {
        (0..i).fold(
            (self, class_act_cache, sum_loss),
            |(weights, class_act_cache, sum_loss), _| {
                let (weights, class_act_cache, sum_loss) =
                    weights.egd_descend(sum_loss, &class_counts, &aux_inputs, class_act_cache, 1);
                weights.egd_descend(sum_loss, &class_counts, &aux_inputs, class_act_cache, -1)
            },
        )
    }
}

impl<I, const C: usize> IIIVMM<I, C> for [(<i8 as Element<I::BitShape>>::Array, i8); C]
where
    Self: Sync + Sized,
    <i8 as Element<I::BitShape>>::Array: Sync,
    I: BitArray + IncrementFracCounters + Sync,
    I::BitShape: ZipFold<i32, i32, i8>,
    [i32; C]: Default,
    i8: Element<I::BitShape>,
    i32: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    bool: Element<I::BitShape>,
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
        class_counts: &[[u32; C]],
        aux_inputs: &[<i32 as Element<<I as BitArray>::BitShape>>::Array],
        class_acts: Vec<[i32; C]>,
        weight_delta: i8,
    ) -> (Self, Vec<[i32; C]>, u64) {
        (0..C).fold(
            (self, class_acts, cur_sum_loss),
            |(mut aux_weights, class_acts_cache, cur_sum_loss), class_index| {
                let acc = {
                    if let Some(sum) = aux_weights[class_index].1.checked_add(weight_delta) {
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

                        if new_sum_loss < cur_sum_loss {
                            aux_weights[class_index].1 = sum;
                            (aux_weights, new_class_acts, new_sum_loss)
                        } else {
                            (aux_weights, class_acts_cache, cur_sum_loss)
                        }
                    } else {
                        (aux_weights, class_acts_cache, cur_sum_loss)
                    }
                };
                <I as BitArray>::BitShape::indices().fold(
                    acc,
                    |(mut aux_weights, class_acts_cache, cur_sum_loss), chan_index| {
                        // if overflow i8, skip.
                        if let Some(sum) = aux_weights[class_index]
                            .0
                            .index_get(&chan_index)
                            .checked_add(weight_delta)
                        {
                            let new_sum_loss = aux_sum_loss_one_class::<I, C>(
                                aux_inputs,
                                class_counts,
                                &class_acts_cache,
                                &chan_index,
                                class_index,
                                weight_delta as i32,
                            );
                            if new_sum_loss < cur_sum_loss {
                                //dbg!(sum);
                                *aux_weights[class_index].0.index_get_mut(&chan_index) = sum;
                                let new_class_acts_cache: Vec<[i32; C]> = aux_inputs
                                    .par_iter()
                                    .zip(class_acts_cache.par_iter().cloned())
                                    .map(|(input, mut new_acts)| {
                                        new_acts[class_index] +=
                                            input.index_get(&chan_index) * weight_delta as i32;
                                        new_acts
                                    })
                                    .collect();
                                (aux_weights, new_class_acts_cache, new_sum_loss)
                            } else {
                                (aux_weights, class_acts_cache, cur_sum_loss)
                            }
                        } else {
                            (aux_weights, class_acts_cache, cur_sum_loss)
                        }
                    },
                )
            },
        )
    }
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

pub trait DescendPatchWeights<P: BitArray, H: BitArray, const C: usize>
where
    i8: Element<H::BitShape>,
    Self: Sized,
    i32: Element<H::BitShape>,
{
    fn descend(
        self,
        patch_centroids: &[P],
        patch_bags: &[(Vec<(usize, u32)>, i32)],
        class_counts: &[[u32; C]],
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
        patch_centroids: &[P],
        patch_bags: &[(Vec<(usize, u32)>, i32)],
        class_counts: &[[u32; C]],
        class_act_cache_one_chan: &[[i32; C]],
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
        patch_centroids: &[P],
        patch_bags: &[(Vec<(usize, u32)>, i32)],
        class_counts: &[[u32; C]],
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
        patch_centroids: &[P],
        patch_bags: &[(Vec<(usize, u32)>, i32)],
        class_counts: &[[u32; C]],
        class_act_cache_one_chan: &[[i32; C]],
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

// with no hidden bit.
fn init_class_act_cache<H: BitArray + IncrementFracCounters + Sync, const C: usize>(
    patch_acts: &[H],
    patch_bags: &[(Vec<(usize, u32)>, i32)],
    aux_weights: &[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C],
) -> Vec<[i32; C]>
where
    i8: Element<[(); C]> + Element<H::BitShape>,
    u32: Element<H::BitShape>,
    i32: Element<[(); C], Array = [i32; C]> + Element<H::BitShape>,
    H::BitShape: Map<u32, i32>,
    [i32; C]: Sync + Send,
    [(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]: IIIVMM<H, C>,
    <i8 as Element<<H as BitArray>::BitShape>>::Array: Sync,
    <i32 as Element<[(); C]>>::Array: Sync + Send,
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
    patch_centroids: &[P],
    patch_bags: &[(Vec<(usize, u32)>, i32)],
    class_act_cache: &[[i32; C]],
    (chan_patch_weights, chan_threshold): &(P::TritArrayType, u32),
    aux_weights_one_chan: &[i8; C],
    sign: i32,
) -> Vec<[i32; C]>
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
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
    patch_centroids: &[P],
    patch_bags: &[(Vec<(usize, u32)>, i32)],
    class_counts: &[[u32; C]],
    class_act_cache: &[[i32; C]],
    chan_patch_weights: &(P::TritArrayType, u32),
    aux_weights_one_chan: &[i8; C],
) -> u64
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
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
    inputs: &[<i32 as Element<H::BitShape>>::Array],
    class_dists: &[[u32; C]],
    class_acts: &[[i32; C]],
    chan_index: &<H::BitShape as Shape>::Index,
    class_index: usize,
    weight_delta: i32,
) -> u64
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
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

pub fn sum_loss_correct<H: BitArray, const C: usize>(
    patch_acts: &[H],
    patch_bags: &[(Vec<(usize, u32)>, i32)],
    class_counts: &[[u32; C]],
    aux_weights: &[(<i8 as Element<H::BitShape>>::Array, i8); C],
) -> (u64, usize, usize)
where
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    u32: Element<H::BitShape>,
    H: IncrementFracCounters + Sync,
    <i8 as Element<<H as BitArray>::BitShape>>::Array: Sync,
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

pub trait TrainWeights<P, H, const C: usize>
where
    P: BitArray,
    H: BitArray,
    (P::TritArrayType, u32): Element<H::BitShape>,
    i8: Element<H::BitShape>,
{
    fn train_weights(
        patch_centroids: &[P],
        patch_bags: &[(Vec<(usize, u32)>, i32)],
        class_counts: &[[u32; C]],
        seed: u64,
        aux_iters: usize,
        iters: usize,
    ) -> (
        <(P::TritArrayType, u32) as Element<H::BitShape>>::Array,
        [(<i8 as Element<H::BitShape>>::Array, i8); C],
        u64,
    );
}

impl<P, H, const C: usize> TrainWeights<P, H, C> for ()
where
    i8: Element<H::BitShape>,
    u32: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    <i8 as Element<H::BitShape>>::Array: Sync,
    <P as BitArray>::TritArrayType: Element<H::BitShape>,
    <u32 as Element<H::BitShape>>::Array: Default,
    <i32 as Element<H::BitShape>>::Array: Default + Send,
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: IIIVMM<H, C> + Default,
    (P::TritArrayType, u32): Element<H::BitShape>,
    P: BitArray,
    H: BitArray + Sync + BitMap<i8> + IncrementFracCounters,
    H::BitShape: Map<u32, i32>
        + Map<P::TritArrayType, (P::TritArrayType, u32)>
        + Map<(P::TritArrayType, u32), u32>,
    P::TritArrayType: Default + Copy,
    distributions::Standard: distributions::Distribution<[H; C]>,
    distributions::Standard:
        distributions::Distribution<<P::TritArrayType as Element<H::BitShape>>::Array>,
    <(P::TritArrayType, u32) as Element<H::BitShape>>::Array:
        BTBVMM<P, H> + DescendPatchWeights<P, H, C> + std::fmt::Debug,
    <u32 as Element<H::BitShape>>::Array: std::fmt::Debug,
{
    fn train_weights(
        patch_centroids: &[P],
        patch_bags: &[(Vec<(usize, u32)>, i32)],
        class_counts: &[[u32; C]],
        seed: u64,
        aux_iters: usize,
        iters: usize,
    ) -> (
        <(P::TritArrayType, u32) as Element<H::BitShape>>::Array,
        [(<i8 as Element<H::BitShape>>::Array, i8); C],
        u64,
    ) where {
        let mut rng = Hc128Rng::seed_from_u64(seed);

        // now we init the weights
        let patch_weights = {
            let trits: <<P as BitArray>::TritArrayType as Element<<H as BitArray>::BitShape>>::Array = rng.gen();

            <<H as BitArray>::BitShape as Map<
                <P as BitArray>::TritArrayType,
                (<P as BitArray>::TritArrayType, u32),
            >>::map(&trits, |&weights| {
                let mut target = <(<P as BitArray>::TritArrayType, u32)>::default();
                target.0 = weights;
                target.1 =
                    (<<P as BitArray>::BitShape as Shape>::N as u32 - target.0.mask_zeros()) / 2;
                target
            })
        };
        let aux_weights =
            <[(); C] as Map<H, (<i8 as Element<<H as BitArray>::BitShape>>::Array, i8)>>::map(
                &rng.gen(),
                |signs| (signs.bit_map(|sign| SIGNS[1][sign as usize] * 20), 0),
            );

        // init the caches
        let patch_acts: Vec<H> = patch_centroids
            .iter()
            .map(|patch| patch_weights.btbvmm(patch))
            .collect();
        let (sum_loss, _, _) =
            sum_loss_correct::<H, C>(&patch_acts, &patch_bags, &class_counts, &aux_weights);
        let class_act_cache: Vec<[i32; C]> =
            init_class_act_cache::<H, C>(&patch_acts, &patch_bags, &aux_weights);

        let aux_inputs: Vec<<i32 as Element<<H as BitArray>::BitShape>>::Array> = patch_bags
            .par_iter()
            .map(|(patch_bag, n)| {
                let (_, hidden_act_counts) = patch_bag.iter().fold(
                    <(usize, <u32 as Element<<H as BitArray>::BitShape>>::Array)>::default(),
                    |mut acc, &(index, count)| {
                        patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                        acc
                    },
                );
                <<H as BitArray>::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| {
                    *count as i32 - n
                })
            })
            .collect();

        // now train for real
        let (patch_weights, _, aux_weights, _, sum_loss) =
            (0..iters).fold(
                (
                    patch_weights,
                    aux_inputs,
                    aux_weights,
                    class_act_cache,
                    sum_loss,
                ),
                |(patch_weights, aux_inputs, aux_weights, class_act_cache, sum_loss), _| {
                    let (aux_weights, class_act_cache, sum_loss) =
                        <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<
                            H,
                            C,
                        >>::optimize(
                            aux_weights,
                            class_act_cache,
                            &class_counts,
                            &aux_inputs,
                            sum_loss,
                            aux_iters,
                        );

                    let patch_start = Instant::now();
                    let (
                        patch_weights,
                        aux_inputs,
                        class_act_cache,
                        sum_loss,
                        (n_weights, n_updates),
                    ) = <<(<P as BitArray>::TritArrayType, u32) as Element<
                        <H as BitArray>::BitShape,
                    >>::Array as DescendPatchWeights<P, H, C>>::descend(
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

                    {
                        let patch_acts: Vec<H> = patch_centroids
                            .iter()
                            .map(|patch| patch_weights.btbvmm(patch))
                            .collect();
                        let (true_sum_loss, n_correct, n) = sum_loss_correct::<H, C>(
                            &patch_acts,
                            &patch_bags,
                            &class_counts,
                            &aux_weights,
                        );
                        dbg!(n_correct as f64 / n as f64);
                        assert_eq!(sum_loss, true_sum_loss);
                    }

                    dbg!(patch_start.elapsed() / n_weights as u32);
                    dbg!(n_updates);
                    let (aux_weights, class_act_cache, sum_loss) =
                        <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<
                            H,
                            C,
                        >>::optimize(
                            aux_weights,
                            class_act_cache,
                            &class_counts,
                            &aux_inputs,
                            sum_loss,
                            aux_iters,
                        );

                    //let thresholds = <H::BitShape as Map<(P::TritArrayType, u32), u32>>::map(&patch_weights, |(_, t)| *t);
                    //dbg!(thresholds);
                    //aux_weights.iter().for_each(|(_, t)| {
                    //    dbg!(t);
                    //});

                    {
                        let patch_acts: Vec<H> = patch_centroids
                            .iter()
                            .map(|patch| patch_weights.btbvmm(patch))
                            .collect();
                        let (true_sum_loss, n_correct, n) = sum_loss_correct::<H, C>(
                            &patch_acts,
                            &patch_bags,
                            &class_counts,
                            &aux_weights,
                        );
                        dbg!(n_correct as f64 / n as f64);
                        assert_eq!(sum_loss, true_sum_loss);
                    }

                    let (patch_weights, aux_inputs, class_act_cache, sum_loss, (_, n_updates)) =
                        <<(<P as BitArray>::TritArrayType, u32) as Element<
                            <H as BitArray>::BitShape,
                        >>::Array as DescendPatchWeights<P, H, C>>::descend(
                            patch_weights,
                            &patch_centroids,
                            &patch_bags,
                            &class_counts,
                            &aux_weights,
                            class_act_cache,
                            aux_inputs,
                            sum_loss,
                            false,
                        );
                    dbg!(n_updates);
                    {
                        let patch_acts: Vec<H> = patch_centroids
                            .iter()
                            .map(|patch| patch_weights.btbvmm(patch))
                            .collect();
                        let (true_sum_loss, n_correct, n) = sum_loss_correct::<H, C>(
                            &patch_acts,
                            &patch_bags,
                            &class_counts,
                            &aux_weights,
                        );
                        dbg!(n_correct as f64 / n as f64);
                        assert_eq!(sum_loss, true_sum_loss);
                    }

                    (
                        patch_weights,
                        aux_inputs,
                        aux_weights,
                        class_act_cache,
                        sum_loss,
                    )
                },
            );

        (patch_weights, aux_weights, sum_loss)
    }
}
