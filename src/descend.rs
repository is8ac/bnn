use crate::bits::{
    BitArray, BitArrayOPs, BitMap, BitMapPack, IncrementFracCounters, MaskedDistance, TritArray,
    TritPack,
};
use crate::count::ElementwiseAdd;
use crate::shape::{Element, Flatten, Map, MapMut, Shape, ZipFold, ZipMap};
use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::num::Wrapping;
use std::ops::Add;
use std::thread::sleep;
use std::time::{Duration, Instant};

const SIGNS: [i8; 2] = [1, -1];

trait IIIVMM<I: BitArray, C: Shape>
where
    i32: Element<C> + Element<I::BitShape>,
    (usize, (<u32 as Element<I::BitShape>>::Array, u32)): Element<C>,
    u32: Element<I::BitShape>,
{
    fn iiivmm(&self, input: &<i32 as Element<I::BitShape>>::Array) -> <i32 as Element<C>>::Array;
    fn grads(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        grads: &mut <(usize, (<u32 as Element<I::BitShape>>::Array, u32)) as Element<C>>::Array,
        up_index: usize,
        down_index: usize,
    ) -> I;
}

impl<I: BitArray + BitArrayOPs, const C: usize> IIIVMM<I, [(); C]>
    for [(<i8 as Element<I::BitShape>>::Array, i8); C]
where
    I::BitShape: MapMut<i32, u32> + ZipMap<i8, i8, bool> + ZipFold<i32, i32, i8>,
    [i32; C]: Default,
    i8: Element<I::BitShape>,
    i32: Element<I::BitShape>,
    bool: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    <bool as Element<I::BitShape>>::Array: std::fmt::Debug,
{
    fn iiivmm(&self, input: &<i32 as Element<I::BitShape>>::Array) -> [i32; C] {
        let mut target = <[i32; C]>::default();
        for c in 0..C {
            target[c] = <I::BitShape as ZipFold<i32, i32, i8>>::zip_fold(
                &input,
                &self[c].0,
                0,
                |sum, i, &w| sum + i * w as i32,
            ) + self[c].1 as i32;
        }
        target
    }
    fn grads(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        grads: &mut [(usize, (<u32 as Element<I::BitShape>>::Array, u32)); C],
        up_index: usize,
        down_index: usize,
    ) -> I {
        for &(i, sign) in &[(up_index, false), (down_index, true)] {
            grads[i].0 += 1;
            (grads[i].1).1 += sign as u32;
            <I::BitShape as MapMut<i32, u32>>::map_mut(
                &mut (grads[i].1).0,
                &input,
                |grad, &input| {
                    *grad += ((input < 0) ^ sign) as u32;
                },
            );
        }
        let grads = <I::BitShape as ZipMap<i8, i8, bool>>::zip_map(
            &self[up_index].0,
            &self[down_index].0,
            |&up, &down| down < up,
        );
        //dbg!(&grads);
        I::bitpack(&grads)
    }
}

trait BTBVMM<I: BitArray, O: BitArray>
where
    u32: Element<I::BitShape>,
    (<u32 as Element<I::BitShape>>::Array, u32): Element<O::BitShape>,
    //(I, bool): Element<O::BitShape>,
    (I::TritArrayType, Option<bool>): Element<O::BitShape>,
{
    fn btbvmm(&self, input: &I) -> O;
    fn grads(
        input: &I,
        output_grads: &O,
        counts: &mut (
            usize,
            <(<u32 as Element<I::BitShape>>::Array, u32) as Element<O::BitShape>>::Array,
        ),
    );
}

impl<I, O: BitArray> BTBVMM<I, O> for <(I::TritArrayType, u32, u32) as Element<O::BitShape>>::Array
where
    u32: Element<I::BitShape> + Element<<I::WordType as BitArray>::BitShape>,
    bool: Element<I::BitShape>,
    I: BitArray + BitArrayOPs,
    O: BitArray
        + BitMapPack<(I::TritArrayType, u32, u32)>
        + BitMap<(<u32 as Element<I::BitShape>>::Array, u32)>,
    O::BitShape: ZipMap<
        (I::TritArrayType, u32, u32),
        (I::TritArrayType, Option<bool>),
        (I::TritArrayType, u32, u32),
    >,
    I::WordType: BitArray,
    I::WordShape: ZipMap<
        <I::WordType as BitArray>::TritArrayType,
        <I::WordType as BitArray>::TritArrayType,
        <I::WordType as BitArray>::TritArrayType,
    >,
    I::TritArrayType: MaskedDistance + TritArray<BitArrayType = I>,
    (I::TritArrayType, u32, u32): Element<O::BitShape>,
    (I, bool): Element<O::BitShape>,
    (I, u32): Element<O::BitShape>,
    (<u32 as Element<I::BitShape>>::Array, u32): Element<O::BitShape>,
    <I::WordType as BitArray>::TritArrayType: Element<I::WordShape, Array = I::TritArrayType>
        + Copy
        + Add<Output = <I::WordType as BitArray>::TritArrayType>,
    (I::TritArrayType, Option<bool>): Element<O::BitShape>,
{
    fn btbvmm(&self, input: &I) -> O {
        <O as BitMapPack<(I::TritArrayType, u32, u32)>>::bit_map_pack(
            self,
            |(weights, threshold, _)| weights.masked_distance(input) > *threshold,
        )
    }
    fn grads(
        input: &I,
        output_grads: &O,
        counts: &mut (
            usize,
            <(<u32 as Element<I::BitShape>>::Array, u32) as Element<O::BitShape>>::Array,
        ),
    ) {
        counts.0 += 1;
        output_grads.bit_map_mut(&mut counts.1, |mut counts, sign| {
            counts.1 += sign as u32;
            input.flipped_increment_counters(sign, &mut counts.0);
        });
    }
}

fn threshold_to_option_bool(count: u32, up: u32, low: u32) -> Option<bool> {
    if count > up {
        Some(true)
    } else if count < low {
        Some(false)
    } else {
        None
    }
}

pub trait Descend<P: BitArray, H: BitArray, C: Shape>
where
    i8: Element<H::BitShape>,
    (P::TritArrayType, u32, u32): Element<H::BitShape>,
    (<i8 as Element<H::BitShape>>::Array, i8): Element<C>,
{
    fn descend(
        patches: &Vec<P>,
        examples: &Vec<(Vec<(usize, u32)>, usize)>,
        bias_diff_threshold: u32,
        patch_learning_update_n: usize,
        n_iters: usize,
        minibatch_take: usize,
        seed: u64,
    ) -> (
        <(P::TritArrayType, u32, u32) as Element<H::BitShape>>::Array,
        <(<i8 as Element<H::BitShape>>::Array, i8) as Element<C>>::Array,
    );
}

impl<P, H, const C: usize> Descend<P, H, [(); C]> for ()
where
    P: BitArray + Sync + Send,
    H: BitArray
        + IncrementFracCounters
        + BitMapPack<u32>
        + BitMap<(<u32 as Element<P::BitShape>>::Array, u32)>
        + Sync
        + Send,
    (): Element<H::BitShape>,
    i8: Element<H::BitShape>,
    u32: Element<P::BitShape>
        + Element<H::BitShape>
        + Element<<P::BitShape as Element<H::BitShape>>::Array>
        + Element<
            <P::BitShape as Element<H::BitShape>>::Array,
            Array = <<u32 as Element<P::BitShape>>::Array as Element<H::BitShape>>::Array,
        >,
    i32: Element<H::BitShape>,
    Option<bool>: Element<P::BitShape>
        + Element<H::BitShape>
        + Element<
            <P::BitShape as Element<H::BitShape>>::Array,
            Array = <<Option<bool> as Element<P::BitShape>>::Array as Element<H::BitShape>>::Array,
        >,
    H::BitShape: Map<u32, i32>
        + Map<u32, Option<bool>>
        + Map<P::TritArrayType, (P::TritArrayType, u32, u32)>
        + Map<(P::TritArrayType, u32, u32), <Option<bool> as Element<P::BitShape>>::Array>
        + Map<(<u32 as Element<P::BitShape>>::Array, u32), (P::TritArrayType, Option<bool>)>
        + Map<(<u32 as Element<P::BitShape>>::Array, u32), <u32 as Element<P::BitShape>>::Array>
        + ZipMap<i8, u32, i8>,
    P::TritArrayType: Element<H::BitShape> + Copy + TritPack + TritArray<TritShape = P::BitShape>,
    P::BitShape: Element<H::BitShape> + Map<u32, u32> + Map<u32, Option<bool>>,
    [(); C]: ZipMap<
        (<i8 as Element<H::BitShape>>::Array, H),
        (usize, (<u32 as Element<H::BitShape>>::Array, u32)),
        (<i8 as Element<H::BitShape>>::Array, H),
    >,
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: Default + IIIVMM<H, [(); C]>,
    [(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C]: Default,
    (P, u32, u32): Element<H::BitShape>,
    (usize, <u32 as Element<H::BitShape>>::Array): Copy,
    (P::TritArrayType, u32, u32): Default + Element<H::BitShape>,
    (P::TritArrayType, Option<bool>): Element<H::BitShape>,
    <() as Element<H::BitShape>>::Array: Default,
    (<u32 as Element<P::BitShape>>::Array, u32): Element<H::BitShape>,
    (usize, <u32 as Element<H::BitShape>>::Array): Default,
    <(<u32 as Element<P::BitShape>>::Array, u32) as Element<H::BitShape>>::Array: Default,
    <(P::TritArrayType, u32, u32) as Element<H::BitShape>>::Array: BTBVMM<P, H>,
    <(P::TritArrayType, Option<bool>) as Element<H::BitShape>>::Array: std::fmt::Debug,
    <u32 as Element<P::BitShape>>::Array: Element<H::BitShape>,
    <Option<bool> as Element<P::BitShape>>::Array: Element<H::BitShape>,
    <P::BitShape as Element<H::BitShape>>::Array: Flatten<Option<bool>> + Flatten<u32>,
    <P::TritArrayType as Element<H::BitShape>>::Array: TritPack,
    Option<bool>:
        Element<<<P::TritArrayType as Element<H::BitShape>>::Array as TritArray>::TritShape>,
    rand::distributions::Standard:
        rand::distributions::Distribution<<P::TritArrayType as Element<H::BitShape>>::Array>,
    Option<bool>: Element<
        <<P::TritArrayType as Element<H::BitShape>>::Array as TritArray>::TritShape,
        Array = <Option<bool> as Element<<P::BitShape as Element<H::BitShape>>::Array>>::Array,
    >,
    H::BitShape: ZipMap<u32, P::TritArrayType, (P::TritArrayType, u32, u32)>
        + ZipMap<(P::TritArrayType, u32, u32), (<u32 as Element<P::BitShape>>::Array, u32), u32>,
    <u32 as Element<H::BitShape>>::Array: std::fmt::Debug,
    <i32 as Element<H::BitShape>>::Array: std::fmt::Debug,
    [i32; C]: std::fmt::Debug,
    H: std::fmt::Debug,
    <i32 as Element<H::BitShape>>::Array: Sync + Send,
    <u32 as Element<H::BitShape>>::Array: Sync + Send,
    [(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C]: Sync + Send,
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: Sync,
    (
        usize,
        Vec<(usize, <u32 as Element<H::BitShape>>::Array)>,
        [(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C],
    ): ElementwiseAdd,
{
    fn descend(
        patches: &Vec<P>,
        examples: &Vec<(Vec<(usize, u32)>, usize)>,
        bias_diff_threshold: u32,
        patch_learning_update_n: usize,
        n_iters: usize,
        minibatch_take: usize,
        seed: u64,
    ) -> (
        <(P::TritArrayType, u32, u32) as Element<H::BitShape>>::Array,
        [(<i8 as Element<H::BitShape>>::Array, i8); C],
    ) {
        let mut rng = Hc128Rng::seed_from_u64(seed);
        let trits: <P::TritArrayType as Element<H::BitShape>>::Array = rng.gen();
        let init_patch_weights = <H::BitShape as Map<
            P::TritArrayType,
            (P::TritArrayType, u32, u32),
        >>::map(&trits, |&weights| {
            let mut target = <(P::TritArrayType, u32, u32)>::default();
            target.0 = weights;
            target.2 = P::BitShape::N as u32;
            target.1 = (target.2 - target.0.mask_zeros()) / 2;
            target
        });
        let init_aux_weights =
            <[(<i8 as Element<<H as BitArray>::BitShape>>::Array, i8); C]>::default();

        (0..n_iters).fold(
            (init_patch_weights, init_aux_weights),
            |(patch_weights, aux_weights), i| {
                let now = Instant::now();
                // binary patch layer outputs
                let patch_acts: Vec<H> = patches
                    .iter()
                    .map(|patch| patch_weights.btbvmm(patch))
                    .collect();
                println!("patches: {:?}", now.elapsed());

                let now = Instant::now();
                // for each example,
                // 1thread: 215ms
                // auto par: 170ms
                let (n_examples, patch_act_counts, aux_weight_counts): (
                    usize,
                    Vec<(usize, <u32 as Element<H::BitShape>>::Array)>,
                    [(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C],
                ) = examples
                    .par_iter()
                    // check if it is wrong.
                    .filter_map(|(patch_counts, class)| {
                        //.filter_map(|(patch_counts, class)| {
                        // elementwise sum the patch acts.
                        let (hidden_act_n, hidden_act_counts) = patch_counts.iter().fold(
                            <(usize, <u32 as Element<H::BitShape>>::Array)>::default(),
                            |mut acc, &(index, count)| {
                                patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                                acc
                            },
                        );
                        let n = hidden_act_n as i32 / 2;
                        let hidden_acts =
                            <H::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| {
                                *count as i32 - n
                            });
                        //dbg!(&hidden_acts);
                        let acts = <[(<i8 as Element<H::BitShape>>::Array, i8); C] as IIIVMM<
                            H,
                            [(); C],
                        >>::iiivmm(&aux_weights, &hidden_acts);
                        let (max_index, &max_val) = acts
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| i != class)
                            .max_by_key(|(_, v)| *v)
                            .unwrap();
                        if acts[*class] > max_val {
                            None
                        } else {
                            Some((patch_counts, hidden_acts, *class, max_index))
                        }
                    })
                    //.take(minibatch_take)
                    .fold(
                        || {
                            (
                            0usize,
                            vec![
                                <(usize, <u32 as Element<H::BitShape>>::Array)>::default();
                                patches.len()
                            ],
                            <[(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C]>::default(),
                        )
                        },
                        |(mut n, mut patch_act_count, mut aux_weight_grads): (
                            usize,
                            Vec<(usize, <u32 as Element<H::BitShape>>::Array)>,
                            [(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C],
                        ),
                         (patch_counts, hidden_input, up_index, down_index): (
                            &Vec<(usize, u32)>,
                            <i32 as Element<H::BitShape>>::Array,
                            usize,
                            usize,
                        )| {
                            n += 1;
                            let input_grads =
                                <[(<i8 as Element<H::BitShape>>::Array, i8); C] as IIIVMM<
                                    H,
                                    [(); C],
                                >>::grads(
                                    &aux_weights,
                                    &hidden_input,
                                    &mut aux_weight_grads,
                                    up_index,
                                    down_index,
                                );
                            //dbg!(&input_grads);

                            patch_counts.iter().for_each(|&(index, count)| {
                                input_grads.weighted_increment_frac_counters(
                                    count,
                                    &mut patch_act_count[index],
                                );
                            });
                            //dbg!(&patch_act_count);
                            (n, patch_act_count, aux_weight_grads)
                        },
                    )
                    .reduce(
                        || {
                            (
                        0usize,
                        vec![
                            <(usize, <u32 as Element<H::BitShape>>::Array)>::default();
                            patches.len()
                        ],
                        <[(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C]>::default(),
                    )
                        },
                        |mut a, b| {
                            a.elementwise_add(&b);
                            a
                        },
                    );
                println!("fold: {:?}", now.elapsed());
                dbg!(n_examples);

                let new_aux_weights = <[(); C] as ZipMap<
                    (<i8 as Element<H::BitShape>>::Array, i8),
                    (usize, (<u32 as Element<H::BitShape>>::Array, u32)),
                    (<i8 as Element<H::BitShape>>::Array, i8),
                >>::zip_map(
                    &aux_weights,
                    &aux_weight_counts,
                    |(w, b), (n, (wg, bg))| {
                        let n = *n as u32 / 2;
                        (
                            <H::BitShape as ZipMap<i8, u32, i8>>::zip_map(&w, &wg, |w, &g| {
                                w.saturating_add(SIGNS[(g > n) as usize])
                            }),
                            b.saturating_add(SIGNS[(*bg > n) as usize]),
                        )
                    },
                );
                //dbg!(&patch_act_counts);
                let patch_weight_counts = patch_act_counts.iter().zip(patches.iter()).fold(
                    <(
                        usize,
                        <(<u32 as Element<P::BitShape>>::Array, u32) as Element<
                            H::BitShape,
                        >>::Array,
                    )>::default(),
                    |mut acc, ((n, counts), patch)| {
                        let threshold = *n as u32 / 2;
                        //println!("t: {} {:?}", n, counts);
                        let grad = <H>::bit_map_pack(counts, |&c: &u32| c > threshold);
                        <<(P::TritArrayType, u32, u32) as Element<H::BitShape>>::Array as BTBVMM<
                            P,
                            H,
                        >>::grads(patch, &grad, &mut acc);
                        acc
                    },
                );
                //println!("act: {:?}", now.elapsed());

                let new_patch_trits = {
                    let patch_weights_counts_vec = {
                        let patch_weights_counts = <H::BitShape as Map<
                            (<u32 as Element<P::BitShape>>::Array, u32),
                            <u32 as Element<P::BitShape>>::Array,
                        >>::map(
                            &patch_weight_counts.1,
                            |(trits, _)| <P::BitShape as Map<u32, u32>>::map(trits, |&t| t),
                        );
                        let mut patch_weights_counts_vec: Vec<u32> =
                            vec![0; <P::BitShape as Element<H::BitShape>>::Array::N];

                        <<P::BitShape as Element<H::BitShape>>::Array>::to_vec(
                            &patch_weights_counts,
                            &mut patch_weights_counts_vec,
                        );
                        patch_weights_counts_vec
                    };

                    let ob_patch_weights_vec = {
                        let ob_patch_weights = <H::BitShape as Map<
                            (P::TritArrayType, u32, u32),
                            <Option<bool> as Element<P::BitShape>>::Array,
                        >>::map(
                            &patch_weights,
                            |(trits, _, _)| trits.trit_expand(),
                        );
                        let mut ob_patch_weights_vec: Vec<Option<bool>> =
                            vec![None; <P::BitShape as Element<H::BitShape>>::Array::N];
                        <<P::BitShape as Element<H::BitShape>>::Array>::to_vec(
                            &ob_patch_weights,
                            &mut ob_patch_weights_vec,
                        );
                        ob_patch_weights_vec
                    };
                    let threshold = patch_weight_counts.0 as u32 / 2;
                    // (magn, index, old_trit, update_sign)
                    let mut weight_grads_vec: Vec<(u32, usize, Option<bool>, bool)> =
                        ob_patch_weights_vec
                            .iter()
                            .zip(patch_weights_counts_vec.iter())
                            .enumerate()
                            .map(|(i, (&trit, &count))| {
                                let pos = trit.unwrap_or(false);
                                let neg = !trit.unwrap_or(true);

                                let pos_magn = count.saturating_sub(threshold)
                                    & (Wrapping(0u32) - Wrapping(!pos as u32)).0;
                                let neg_magn = threshold.saturating_sub(count)
                                    & (Wrapping(0u32) - Wrapping(!neg as u32)).0;

                                let magn = pos_magn | neg_magn;
                                //println!("{} {}", threshold, count);
                                let sign = count > threshold;
                                (magn, i, trit, sign)
                            })
                            .collect();
                    weight_grads_vec.sort();
                    weight_grads_vec.reverse();
                    //dbg!(&weight_grads_vec[0..50]);

                    let mut new_weights_vec: Vec<(usize, Option<bool>)> = weight_grads_vec
                        .iter()
                        .enumerate()
                        .map(|(i, &(magn, w_index, trit, update_sign))| {
                            (
                                w_index,
                                if (i < patch_learning_update_n) & (magn > 0) {
                                    if let None = trit {
                                        Some(update_sign)
                                    } else {
                                        None
                                    }
                                } else {
                                    trit
                                },
                            )
                        })
                        .collect();
                    new_weights_vec.sort();
                    //dbg!(&new_weights_vec[0..50]);

                    let weight_vec: Vec<Option<bool>> =
                        new_weights_vec.iter().map(|(_, t)| *t).collect();

                    let new_ob_weights =
                        <<P::BitShape as Element<H::BitShape>>::Array>::from_vec(&weight_vec);
                    <<P::TritArrayType as Element<H::BitShape>>::Array>::trit_pack(&new_ob_weights)
                };
                let threshold = patch_weight_counts.0 as u32 / 2;

                let patch_weight_offsets = <H::BitShape as ZipMap<
                    (P::TritArrayType, u32, u32),
                    (<u32 as Element<P::BitShape>>::Array, u32),
                    u32,
                >>::zip_map(
                    &patch_weights,
                    &patch_weight_counts.1,
                    |(_, _, offset), (_, bias_count)| {
                        //println!("{} {}", threshold, bias_count);
                        if *bias_count > (threshold + bias_diff_threshold) {
                            offset.saturating_sub(1)
                        } else if (*bias_count + bias_diff_threshold) < threshold {
                            offset.saturating_add(1)
                        } else {
                            *offset
                        }
                        //*offset
                    },
                );

                let new_patch_weights = <H::BitShape as ZipMap<
                    u32,
                    P::TritArrayType,
                    (P::TritArrayType, u32, u32),
                >>::zip_map(
                    &patch_weight_offsets,
                    &new_patch_trits,
                    |&offset, &trits| {
                        let n_zeros = trits.mask_zeros();
                        (trits, (offset.max(n_zeros) - n_zeros) / 2, offset)
                    },
                );
                //println!("iter: {:?}", now.elapsed());
                if i < 10 {
                    (patch_weights, new_aux_weights)
                } else {
                    (new_patch_weights, aux_weights)
                }
            },
        )
    }
}
