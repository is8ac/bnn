#![feature(const_generics)]

use bitnn::bits::{b8, BitArray, BitArrayOPs, BitZipMap, Distance};
use bitnn::shape::{Element, MapMut, Shape, ZipFold, ZipMap};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_hc::Hc128Rng;

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

const SIGNS: [i8; 2] = [1, -1];

fn main() {
    let examples = vec![
        ([-1i32, 0, 1, 2, -3, 3, 2, -3], 0usize),
        ([-1, 2, 0, 2, -3, 3, 2, -3], 0),
        ([-2, 1, 1, 1, -1, 3, 2, -3], 0),
        ([-1i32, 0, 1, 2, -3, 3, 2, -3], 0),
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

    let new_weights = (0..30).fold(weights, |weights, i| {
        dbg!(i);
        let weight_grads = examples
            .iter()
            .filter_map(|(input, class)| {
                let acts = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&weights, &input);
                let (max_index, &max_val) = acts
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i != class)
                    .max_by_key(|(_, v)| *v)
                    .unwrap();
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
                    <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::grads(
                        &weights, input, &mut grads, up_index, down_index,
                    );
                    grads
                },
            );

        //dbg!(weight_grads);
        let new_weights = <[(); 3] as ZipMap<
            ([i8; 8], i8),
            (usize, ([u32; 8], u32)),
            ([i8; 8], i8),
        >>::zip_map(&weights, &weight_grads, |(w, b), (n, (wg, bg))| {
            let n = *n as u32 / 2;
            (
                <[(); 8] as ZipMap<i8, u32, i8>>::zip_map(w, wg, |w, &g| {
                    w.saturating_add(SIGNS[(g > n) as usize])
                }),
                b.saturating_add(SIGNS[(*bg > n) as usize]),
            )
        });
        let n_correct = examples
            .iter()
            .filter(|(input, class)| {
                let acts = <[([i8; 8], i8); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&weights, &input);
                let (_, &max_val) = acts
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i != class)
                    .max_by_key(|(_, v)| *v)
                    .unwrap();
                acts[*class] > max_val
            })
            .count();
        dbg!(n_correct);
        new_weights
    });
    dbg!(examples.len());
    dbg!(new_weights);

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
}
