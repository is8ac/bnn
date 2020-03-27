#![feature(const_generics)]

use bitnn::bits::{b16, b8, t16, t8, BitArray, BitArrayOPs, BitMap, BitMapPack, BitZipMap, Distance, MaskedDistance, TritArray, TritPack};
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

impl<I: BitArray + BitArrayOPs, const C: usize> IIIVMM<I, [(); C]> for [(<i8 as Element<I::BitShape>>::Array, i8); C]
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
            target[c] = <I::BitShape as ZipFold<i32, i32, i8>>::zip_fold(&input, &self[c].0, 0, |sum, i, &w| sum + i * w as i32) + self[c].1 as i32;
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
            <I::BitShape as MapMut<i32, u32>>::map_mut(&mut (grads[i].1).0, &input, |grad, &input| {
                *grad += ((input < 0) ^ sign) as u32;
            });
        }
        let grads = <I::BitShape as ZipMap<i8, i8, bool>>::zip_map(&self[up_index].0, &self[down_index].0, |&up, &down| down > up);
        I::bitpack(&grads)
    }
}

trait BTBVMM<I: BitArray, O: BitArray>
where
    u32: Element<I::BitShape>,
    (<u32 as Element<I::BitShape>>::Array, u32): Element<O::BitShape>,
    (I, bool): Element<O::BitShape>,
    (I::TritArrayType, Option<bool>): Element<O::BitShape>,
{
    fn btbvmm(&self, input: &I) -> O;
    fn grads(input: &I, output_grads: &O, counts: &mut (usize, <(<u32 as Element<I::BitShape>>::Array, u32) as Element<O::BitShape>>::Array));
    fn update_weights(&self, grads: &<(I::TritArrayType, Option<bool>) as Element<O::BitShape>>::Array) -> Self;
}

impl<I, O: BitArray> BTBVMM<I, O> for <(I::TritArrayType, u32, u32) as Element<O::BitShape>>::Array
where
    u32: Element<I::BitShape> + Element<<I::WordType as BitArray>::BitShape>,
    bool: Element<I::BitShape>,
    I: BitArray + BitArrayOPs,
    O: BitArray + BitMapPack<(I::TritArrayType, u32, u32)> + BitMap<(<u32 as Element<I::BitShape>>::Array, u32)>,
    O::BitShape: ZipMap<(I::TritArrayType, u32, u32), (I::TritArrayType, Option<bool>), (I::TritArrayType, u32, u32)>,
    I::WordType: BitArray,
    I::WordShape: ZipMap<<I::WordType as BitArray>::TritArrayType, <I::WordType as BitArray>::TritArrayType, <I::WordType as BitArray>::TritArrayType>,
    I::TritArrayType: MaskedDistance + TritArray<BitArrayType = I>,
    (I::TritArrayType, u32, u32): Element<O::BitShape>,
    (I, bool): Element<O::BitShape>,
    (I, u32): Element<O::BitShape>,
    (<u32 as Element<I::BitShape>>::Array, u32): Element<O::BitShape>,
    <I::WordType as BitArray>::TritArrayType: Element<I::WordShape, Array = I::TritArrayType> + Copy + Add<Output = <I::WordType as BitArray>::TritArrayType>,
    (I::TritArrayType, Option<bool>): Element<O::BitShape>,
{
    fn btbvmm(&self, input: &I) -> O {
        <O as BitMapPack<(I::TritArrayType, u32, u32)>>::bit_map_pack(self, |(weights, threshold, _)| weights.masked_distance(input) > *threshold)
    }
    fn grads(input: &I, output_grads: &O, counts: &mut (usize, <(<u32 as Element<I::BitShape>>::Array, u32) as Element<O::BitShape>>::Array)) {
        counts.0 += 1;
        output_grads.bit_map_mut(&mut counts.1, |mut counts, sign| {
            counts.1 += sign as u32;
            input.flipped_increment_counters(sign, &mut counts.0);
        });
    }
    fn update_weights(&self, grads: &<(I::TritArrayType, Option<bool>) as Element<O::BitShape>>::Array) -> Self {
        <O::BitShape as ZipMap<(I::TritArrayType, u32, u32), (I::TritArrayType, Option<bool>), (I::TritArrayType, u32, u32)>>::zip_map(
            self,
            &grads,
            |weights, (weight_grads, bias_grad)| {
                let offset = match bias_grad {
                    Some(true) => weights.2 - 1,
                    None => weights.2,
                    Some(false) => weights.2 + 1,
                };
                (
                    I::WordShape::zip_map(&weights.0, weight_grads, |&weight, &bias| weight + bias),
                    (offset - weights.0.mask_zeros()) / 2,
                    offset,
                )
            },
        )
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

const SIGNS: [i8; 2] = [1, -1];
const THRESHOLD: u32 = 0;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(4);
    let patches = vec![
        [b16(0b_1101_1100_1100_1110u16), b16(0b_0111_1110_1101_0010u16)],
        [b16(0b_1001_0010_0011_0001u16), b16(0b_1100_0011_0101_0001u16)],
        [b16(0b_0010_0011_0011_1001u16), b16(0b_0000_0001_0011_1110u16)],
    ];
    //let patch_weights: [([t16; 2], u32, u32); 8] = rng.gen();
    let patch_weights = {
        let mut target = <[([t16; 2], u32, u32); 8]>::default();
        for i in 0..8 {
            target[i].0 = rng.gen();
            target[i].2 = <[t16; 2]>::N as u32;
            target[i].1 = (target[i].2 - target[i].0.mask_zeros()) / 2;
        }
        target
    };
    dbg!(&patch_weights);

    let hidden_grads = vec![
        (0, &b8(0b_0110_1100_u8)),
        (1, &b8(0b_1001_0011_u8)),
        (2, &b8(0b_0011_1001_u8)),
        (0, &b8(0b_1110_1100_u8)),
        (1, &b8(0b_1011_0010_u8)),
        (2, &b8(0b_1101_0000_u8)),
        (0, &b8(0b_1110_1101_u8)),
        (1, &b8(0b_1101_0001_u8)),
        (2, &b8(0b_0001_1101_u8)),
        (0, &b8(0b_1111_0100_u8)),
        (1, &b8(0b_1010_1010_u8)),
        (2, &b8(0b_1011_0100_u8)),
    ];

    let new_patch_weights = (0..5).fold(patch_weights, |weights, i| {
        dbg!(i);
        dbg!(weights[4]);
        let mut grad_counts = <(usize, [(<u32 as Element<<[b16; 2] as BitArray>::BitShape>>::Array, u32); 8])>::default();
        for (patch_index, hidden_grad) in &hidden_grads {
            <[([t16; 2], u32, u32); 8] as BTBVMM<[b16; 2], b8>>::grads(&patches[*patch_index], hidden_grad, &mut grad_counts);
        }

        //dbg!(&grad_counts);
        let n = grad_counts.0 as u32 / 2;
        let patch_weigth_grads = <[(); 8] as Map<(<u32 as Element<<[b16; 2] as BitArray>::BitShape>>::Array, u32), ([t16; 2], Option<bool>)>>::map(
            &grad_counts.1,
            |(weights_counts, bias_count)| {
                let weights_grads = <<[b16; 2] as BitArray>::BitShape as Map<u32, Option<bool>>>::map(weights_counts, |&count| {
                    threshold_to_option_bool(count, n + THRESHOLD, n - THRESHOLD)
                });
                (
                    <[t16; 2]>::trit_pack(&weights_grads),
                    threshold_to_option_bool(*bias_count, n + THRESHOLD, n - THRESHOLD),
                )
            },
        );

        <[([t16; 2], u32, u32); 8] as BTBVMM<[b16; 2], b8>>::update_weights(&patch_weights, &patch_weigth_grads)
    });
    dbg!(&new_patch_weights);
    //for patch in &patches {
    //    let output: b8 = weights.btbvmm(patch);
    //    dbg!(output);
    //}

    let examples = vec![
        ([-1i32, 0, 1, 2, -3, 3, 2, -3], 0usize),
        ([-1, 2, 0, 2, -3, 3, 2, -3], 0),
        ([-2, 1, 1, 1, -1, 3, 2, -3], 0),
        ([-1, 0, 1, 2, -3, 3, 2, -3], 0),
        ([1, -2, 0, 2, -3, 3, 2, -3], 0),
        ([-2, 1, -5, 1, -1, 3, 2, -3], 0),
        ([-1, 0, 1, -2, 3, 3, -2, 1], 1),
        ([1, -1, -2, -2, -2, -2, 2, -3], 1),
        ([-1, 1, 1, 3, 2, 3, -1, 2], 1),
        ([-1, 0, 1, 2, 3, 3, -2, 1], 1),
        ([1, -1, -2, -2, -2, -2, 2, -3], 1),
        ([-1, 1, 1, 3, -2, 3, -1, 2], 1),
        ([1, 0, -1, 2, -3, -3, -1, -3], 2),
        ([2, 3, -1, 2, -3, 3, -9, -2], 2),
        ([1, 0, 0, -2, 3, -3, -2, 3], 2),
        ([1, 1, -1, 2, -3, 3, 1, 3], 2),
        ([2, 3, -1, 2, -3, 3, -9, -2], 2),
        ([0, 0, 1, 2, 3, 3, -2, 1], 2),
    ];
    let weights = [
        ([-1i8, -1, 1, 1, -1, 1, -1, 1], 0i8),
        ([-1, 1, -1, 2, -1, 1, -1, -1], 0),
        ([1, 1, 1, -1, -1, -1, 1, -1], 0),
    ];

    let new_weights = (0..3).fold(weights, |weights, i| {
        dbg!(i);
        let weight_grads = examples
            .iter()
            .filter_map(|(input, class)| {
                let acts = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&weights, &input);
                let (max_index, &max_val) = acts.iter().enumerate().filter(|(i, _)| i != class).max_by_key(|(_, v)| *v).unwrap();
                if acts[*class] > max_val {
                    None
                } else {
                    Some((input, *class, max_index))
                }
            })
            .take(50)
            .fold(
                [(0usize, ([0u32; 8], 0u32)); 3],
                |mut grads, (input, up_index, down_index): (&[i32; 8], usize, usize)| {
                    <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::grads(&weights, input, &mut grads, up_index, down_index);
                    grads
                },
            );

        //dbg!(weight_grads);
        let new_weights =
            <[(); 3] as ZipMap<([i8; 8], i8), (usize, ([u32; 8], u32)), ([i8; 8], i8)>>::zip_map(&weights, &weight_grads, |(w, b), (n, (wg, bg))| {
                let n = *n as u32 / 2;
                (
                    <[(); 8] as ZipMap<i8, u32, i8>>::zip_map(w, wg, |w, &g| w.saturating_add(SIGNS[(g > n) as usize])),
                    b.saturating_add(SIGNS[(*bg > n) as usize]),
                )
            });
        let n_correct = examples
            .iter()
            .filter(|(input, class)| {
                let acts = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&weights, &input);
                let (_, &max_val) = acts.iter().enumerate().filter(|(i, _)| i != class).max_by_key(|(_, v)| *v).unwrap();
                acts[*class] > max_val
            })
            .count();
        dbg!(n_correct);
        new_weights
    });
    dbg!(examples.len());
    //dbg!(new_weights);

    //let acts = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&weights, &input);
    //dbg!(&acts);
    //let mut weight_grads = [(0usize, ([0u32; 8], 0u32)); 3];
    //let grads = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::grads(&weights, &input, &mut weight_grads, 2, 0);
    //let new_input = grads.bit_zip_map(&input, |sign, i: i32| i.saturating_add(SIGNS[sign as usize] as i32));
    //let acts = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&weights, &new_input);
    //dbg!(&acts);
    //dbg!(&new_weights);
    //let acts = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&new_weights, &input);
    //dbg!(&acts);

    //let acts = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&new_weights, &new_input);
    //dbg!(&acts);
    let signs = b8(0b_1010_1010);
    let mut trits = t8(0b_1100_1100, 0b_1111_0000);
    trits.flip(&b8(0b_1010_1010));
    trits.flip(&b8(0b_1010_0110));
    println!("{:?}", trits);
}
