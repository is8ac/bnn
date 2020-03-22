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
    (<i32 as Element<I::BitShape>>::Array, i32): Element<C>,
{
    fn iiivmm(&self, input: &<i32 as Element<I::BitShape>>::Array) -> <i32 as Element<C>>::Array;
    fn weights_grad(
        input: &<i32 as Element<I::BitShape>>::Array,
        grads: &mut <(<i32 as Element<I::BitShape>>::Array, i32) as Element<C>>::Array,
        up_index: usize,
        down_index: usize,
    );
    fn input_grads(&self, up_index: usize, down_index: usize) -> I;
}

impl<I: BitArray + BitArrayOPs, const C: usize> IIIVMM<I, [(); C]>
    for [(<i8 as Element<I::BitShape>>::Array, i32); C]
where
    I::BitShape: MapMut<i32, i32> + ZipMap<i8, i8, bool> + ZipFold<i32, i32, i8>,
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
            ) + self[c].1;
        }
        target
    }
    fn weights_grad(
        input: &<i32 as Element<I::BitShape>>::Array,
        grads: &mut [(<i32 as Element<I::BitShape>>::Array, i32); C],
        up_index: usize,
        down_index: usize,
    ) {
        for &(i, sign) in &[(up_index, 1i32), (down_index, -1i32)] {
            grads[i].1 += sign;
            <I::BitShape as MapMut<i32, i32>>::map_mut(&mut grads[i].0, &input, |grad, input| {
                *grad += (input * sign) as i32;
            });
        }
    }
    fn input_grads(&self, up_index: usize, down_index: usize) -> I {
        let grads = <I::BitShape as ZipMap<i8, i8, bool>>::zip_map(
            &self[up_index].0,
            &self[down_index].0,
            |&up, &down| down > up,
        );
        I::bitpack(&grads)
    }
}

fn main() {
    let input = [-1i32, 0, 1, 2, -3, 5, -2, -3];
    let weights = [
        ([-1i8, 0, 1, 2, 3, 5, -3, 1], 0i32),
        ([-1, 0, -1, 2, -3, -5, -2, -1], 0i32),
        ([1, 0, 1, -2, -3, 5, 2, -3], 0i32),
    ];
    let acts = <[([i8; 8], i32); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&weights, &input);
    dbg!(&acts);
    let grads = <[([i8; 8], i32); 3] as IIIVMM<b8, [(); 3]>>::input_grads(&weights, 1, 2);
    dbg!(&grads);
    let new_input =
        <b8 as BitZipMap<i32, i32>>::bit_zip_map(
            &grads,
            &input,
            |sign, i| if sign { i - 1 } else { i + 1 },
        );

    let acts = <[([i8; 8], i32); 3] as IIIVMM<b8, [(); 3]>>::iiivmm(&weights, &new_input);
    dbg!(&acts);
}
