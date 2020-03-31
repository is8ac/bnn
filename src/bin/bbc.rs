#![feature(const_generics)]

use bitnn::bits::{
    b16, b8, t16, t8, BitArray, BitArrayOPs, BitMap, BitMapPack, BitZipMap, Distance,
    IncrementFracCounters, MaskedDistance, TritArray, TritPack,
};
use bitnn::shape::{Element, Map, MapMut, Shape, ZipFold, ZipMap};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_hc::Hc128Rng;
use std::ops::Add;

//trait BBBVMM<I, O>
//where
//    Self: BitArray,
//    u32: Element<Self::BitShape>,
//{
//    fn bbbvmm(&self, input: &I) -> O;
//    fn weight_grads(
//        &self,
//        input: &I,
//        target_signs: &O,
//        target_mask: &O,
//        counters: &mut (usize, <u32 as Element<Self::BitShape>>::Array),
//    );
//}

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
            |&up, &down| down > up,
        );
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
    fn update_weights(
        &self,
        grads: &<(I::TritArrayType, Option<bool>) as Element<O::BitShape>>::Array,
    ) -> Self;
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
    fn update_weights(
        &self,
        grads: &<(I::TritArrayType, Option<bool>) as Element<O::BitShape>>::Array,
    ) -> Self {
        <O::BitShape as ZipMap<
            (I::TritArrayType, u32, u32),
            (I::TritArrayType, Option<bool>),
            (I::TritArrayType, u32, u32),
        >>::zip_map(self, &grads, |weights, (weight_grads, bias_grad)| {
            let offset = match bias_grad {
                Some(true) => weights.2 + 1,
                None => weights.2,
                Some(false) => weights.2 - 1,
            };
            let n_zeros = weights.0.mask_zeros();
            (
                I::WordShape::zip_map(&weights.0, weight_grads, |&weight, &bias| weight + bias),
                (offset.max(n_zeros) - weights.0.mask_zeros()) / 2,
                offset,
            )
        })
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
        examples: &Vec<(Vec<usize>, usize)>,
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
    P: BitArray,
    H: BitArray
        + IncrementFracCounters
        + BitMapPack<u32>
        + BitMap<(<u32 as Element<P::BitShape>>::Array, u32)>,
    (): Element<H::BitShape>,
    i8: Element<H::BitShape>,
    u32: Element<P::BitShape> + Element<H::BitShape>,
    i32: Element<H::BitShape>,
    Option<bool>: Element<P::BitShape> + Element<H::BitShape>,
    H::BitShape: Map<P::TritArrayType, (P::TritArrayType, u32, u32)>
        + Map<u32, i32>
        + ZipMap<i8, u32, i8>
        + Map<u32, Option<bool>>
        + Map<(<u32 as Element<P::BitShape>>::Array, u32), (P::TritArrayType, Option<bool>)>,
    P::TritArrayType:
        TritArray + Element<H::BitShape> + Copy + TritPack + TritArray<TritShape = P::BitShape>,
    P::BitShape: Map<u32, Option<bool>>,
    [(); C]: ZipMap<
        (<i8 as Element<H::BitShape>>::Array, H),
        (usize, (<u32 as Element<H::BitShape>>::Array, u32)),
        (<i8 as Element<H::BitShape>>::Array, H),
    >,
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: Default + IIIVMM<H, [(); C]>,
    [(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C]: Default,
    (P, u32, u32): Element<H::BitShape>,
    (usize, <u32 as Element<H::BitShape>>::Array): Copy,
    (P::TritArrayType, u32, u32): Element<H::BitShape> + Default,
    (P::TritArrayType, Option<bool>): Element<H::BitShape>,
    <() as Element<H::BitShape>>::Array: Default,
    (<u32 as Element<P::BitShape>>::Array, u32): Element<H::BitShape>,
    (usize, <u32 as Element<H::BitShape>>::Array): Default,
    <(<u32 as Element<P::BitShape>>::Array, u32) as Element<H::BitShape>>::Array: Default,
    <(P::TritArrayType, u32, u32) as Element<H::BitShape>>::Array: BTBVMM<P, H>,
    rand::distributions::Standard:
        rand::distributions::Distribution<<P::TritArrayType as Element<H::BitShape>>::Array>,
{
    fn descend(
        patches: &Vec<P>,
        examples: &Vec<(Vec<usize>, usize)>,
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
                dbg!(i);
                let patch_acts: Vec<H> = patches
                    .iter()
                    .map(|patch| patch_weights.btbvmm(patch))
                    .collect();

                let (n_examples, patch_act_counts, aux_weight_counts): (
                    usize,
                    Vec<(usize, <u32 as Element<H::BitShape>>::Array)>,
                    [(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C],
                ) = examples
                    .iter()
                    .filter_map(|(patch_counts, class)| {
                        let (hidden_act_n, hidden_act_counts) =
                            patch_counts.iter().zip(patch_acts.iter()).fold(
                                <(usize, <u32 as Element<H::BitShape>>::Array)>::default(),
                                |mut acc, (count, act)| {
                                    act.weighted_increment_frac_counters(*count as u32, &mut acc);
                                    acc
                                },
                            );
                        let n = hidden_act_n as i32 / 2;
                        let hidden_acts =
                            <H::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| {
                                *count as i32 - n
                            });
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
                    .take(minibatch_take)
                    .fold(
                        (
                            0usize,
                            vec![
                                <(usize, <u32 as Element<H::BitShape>>::Array)>::default();
                                patches.len()
                            ],
                            <[(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C]>::default(),
                        ),
                        |(mut n, mut patch_act_count, mut aux_weight_grads): (
                            usize,
                            Vec<(usize, <u32 as Element<H::BitShape>>::Array)>,
                            [(usize, (<u32 as Element<H::BitShape>>::Array, u32)); C],
                        ),
                         (patch_counts, hidden_input, up_index, down_index): (
                            &Vec<usize>,
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

                            patch_act_count
                                .iter_mut()
                                .zip(patch_counts.iter())
                                .for_each(|(act_counters, count)| {
                                    input_grads.weighted_increment_frac_counters(
                                        *count as u32,
                                        act_counters,
                                    );
                                });
                            //dbg!(&patch_act_count);
                            (n, patch_act_count, aux_weight_grads)
                        },
                    );
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
                let patch_weight_counts = patch_act_counts.iter().zip(patches.iter()).fold(
                    <(
                        usize,
                        <(<u32 as Element<<P as BitArray>::BitShape>>::Array, u32) as Element<
                            H::BitShape,
                        >>::Array,
                    )>::default(),
                    |mut acc, ((n, counts), patch)| {
                        let threshold = *n as u32 / 2;
                        let grad = <H>::bit_map_pack(counts, |&c: &u32| c < threshold);
                        <<(P::TritArrayType, u32, u32) as Element<H::BitShape>>::Array as BTBVMM<
                            P,
                            H,
                        >>::grads(patch, &grad, &mut acc);
                        acc
                    },
                );
                let n = patch_weight_counts.0 as u32 / 2;
                let patch_weigth_grads = <H::BitShape as Map<
                    (<u32 as Element<P::BitShape>>::Array, u32),
                    (P::TritArrayType, Option<bool>),
                >>::map(
                    &patch_weight_counts.1,
                    |(weights_counts, bias_count)| {
                        let weights_grads =
                            <<P as BitArray>::BitShape as Map<u32, Option<bool>>>::map(
                                weights_counts,
                                |&count| {
                                    //dbg!(n);
                                    threshold_to_option_bool(count, n + THRESHOLD, n - THRESHOLD)
                                },
                            );
                        (
                            <P::TritArrayType>::trit_pack(&weights_grads),
                            threshold_to_option_bool(*bias_count, n + THRESHOLD, n - THRESHOLD),
                        )
                    },
                );

                let new_patch_weights =
                    <<(P::TritArrayType, u32, u32) as Element<H::BitShape>>::Array as BTBVMM<
                        P,
                        H,
                    >>::update_weights(&patch_weights, &patch_weigth_grads);

                (new_patch_weights, new_aux_weights)
            },
        )
    }
}

const SIGNS: [i8; 2] = [1, -1];
const THRESHOLD: u32 = 1;
const N_PATCHES: usize = 4;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(4);
    let patches = vec![
        [
            b16(0b_1101_1100_1100_1110u16),
            b16(0b_0111_1110_1101_0010u16),
        ],
        [
            b16(0b_1001_0010_0011_0001u16),
            b16(0b_1100_0011_0101_0001u16),
        ],
        [
            b16(0b_0010_0011_0011_1001u16),
            b16(0b_0000_0001_0011_1110u16),
        ],
        [
            b16(0b_1111_1100_1010_1101u16),
            b16(0b_0111_1001_0011_0100u16),
        ],
    ];
    let examples: Vec<(Vec<usize>, usize)> = vec![
        (vec![5, 0, 1, 0], 0),
        (vec![0, 5, 1, 0], 1),
        (vec![0, 1, 4, 1], 2),
        (vec![5, 0, 0, 1], 0),
        (vec![0, 2, 1, 3], 1),
        (vec![1, 1, 4, 0], 2),
        (vec![4, 1, 1, 0], 0),
        (vec![0, 4, 1, 1], 1),
        (vec![1, 0, 4, 1], 2),
        (vec![4, 0, 1, 0], 0),
        (vec![1, 4, 1, 0], 1),
        (vec![0, 1, 4, 1], 2),
        (vec![4, 0, 1, 1], 0),
        (vec![0, 6, 0, 0], 1),
        (vec![1, 1, 4, 0], 2),
        (vec![4, 1, 1, 0], 0),
        (vec![0, 4, 1, 1], 1),
        (vec![1, 0, 4, 1], 2),
    ];
    let weights = <() as Descend<[b16; 2], b8, [(); 3]>>::descend(&patches, &examples, 30, 100, 0);
}
