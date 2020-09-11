#![feature(const_generics)]

use bnn::bits::{b1, b32, t1, BitArray, BitMap, BitMapPack, MaskedDistance, TritArray};
use bnn::datasets::cifar;
use bnn::image2d::{Conv2D, PatchFold, Poolable2D};
use bnn::shape::{Element, IndexGet, Map, Shape, ZipMap};
use bnn::unary::u8x3_to_b32;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::env;
use std::path::Path;

const N_EXAMPLES: usize = 10;

type Hidden = [b32; 32];
type Pixel = b32;
type Patch = [[Pixel; 3]; 3];
type AuxPatchShape = [[(); 5]; 5];
type HiddenPatch = <Hidden as Element<AuxPatchShape>>::Array;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let mut args = env::args();
    args.next();
    let base_path = args
        .next()
        .expect("you must give path to cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&Path::new(&base_path), N_EXAMPLES);
    let unary_examples: Vec<([[Pixel; 32]; 32], usize)> = int_examples_32
        .par_iter()
        .map(|(image, class)| {
            (
                <[[(); 32]; 32] as Map<[u8; 3], Pixel>>::map(image, |&pixel| u8x3_to_b32(pixel)),
                *class,
            )
        })
        .collect();

    let weights: <<Patch as BitArray>::TritArrayType as Element<<Hidden as BitArray>::BitShape>>::Array = rng.gen();
    let aux_weights: [<HiddenPatch as BitArray>::TritArrayType; 10] = rng.gen();

    dbg!(unary_examples[6].0);
    let l1: [[Hidden; 32]; 32] = unary_examples[6].0.conv2d(|patch: &Patch| {
        Hidden::bit_map_pack(&weights, |trits: &<Patch as BitArray>::TritArrayType| {
            trits.masked_distance(patch) < (<Patch as BitArray>::BitShape::N as u32 / 4)
        })
    });
    dbg!(l1);

    //let l2: [[[u32; 10]; 32]; 32] =
    //    l1.conv2d(|patch: &HiddenPatch| <[(); 10] as Map<_, u32>>::map(&aux_weights, |trits: &<HiddenPatch as BitArray>::TritArrayType| trits.masked_distance(patch)));
    ////dbg!(l2);

    let full_acts: [u32; 10] = l1.patch_fold([0u32; 10], |acc: [u32; 10], patch: &HiddenPatch| {
        <[(); 10] as ZipMap<u32, <HiddenPatch as BitArray>::TritArrayType, u32>>::zip_map(
            &acc,
            &aux_weights,
            |act, trits| act + trits.masked_distance(patch),
        )
    });
    dbg!(full_acts);

    let grads: Vec<_> = <Hidden as BitArray>::BitShape::indices()
        .map(|hidden_index| {
            let chan_weights = weights.index_get(hidden_index);

            // what weight movement will flip the act?
            let patch_grads: [[<Patch as BitArray>::TritArrayType; 32]; 32] =
                unary_examples[6].0.conv2d(|patch: &Patch| {
                    let threshold = <Patch as BitArray>::BitShape::N as u32 / 4;
                    let act = chan_weights.masked_distance(patch);

                    let is_above = act == threshold; // if act is equal to threshold, act is unset, but decreasing it by one will make it set.
                    let is_below = act == (threshold - 1); // if act is one below threshold, it is set, but increasing by one will make it unset.

                    if is_above | is_below {
                        //let patch_bits
                        <Patch as BitArray>::TritArrayType::default()
                    } else {
                        <Patch as BitArray>::TritArrayType::default()
                    }
                });

            let chan_grads: [[Option<bool>; 32]; 32] =
                unary_examples[6].0.conv2d(|patch: &Patch| {
                    let threshold = <Patch as BitArray>::BitShape::N as u32 / 4;
                    let act = chan_weights.masked_distance(patch);

                    let is_above = act == threshold; // if act is equal to threshold, act is unset, but decreasing it by one will make it set.
                    let is_below = act == (threshold - 1); // if act is one below threshold, it is set, but increasing by one will make it unset.

                    Some(is_below).filter(|_| is_above | is_below)
                });
            //dbg!(chan_grads);
            let n_alive: usize = chan_grads
                .iter()
                .flatten()
                .filter(|trit| trit.is_some())
                .count();
            dbg!(n_alive);

            // for each of the 900 patches, XOR with chan_weights as needed, and flip by that pixels chan grad to make a <<bool as Element<ImageShape>>::Array as Element<Patch::BitShape>>::Array.
            // Or just leave it as a <Patch as Element<ImageShape>>::Array and extract bits later?
            // for each, cached aux conv and loss.
            // How many will be duplicats?
            let chan_aux_weights: [<t1 as Element<AuxPatchShape>>::Array; 10] =
                <[(); 10] as Map<
                    <<Hidden as BitArray>::TritArrayType as Element<AuxPatchShape>>::Array,
                    <t1 as Element<AuxPatchShape>>::Array,
                >>::map(&aux_weights, |class_patch| {
                    <AuxPatchShape as Map<<Hidden as BitArray>::TritArrayType, t1>>::map(
                        class_patch,
                        |pixel| t1(pixel.get_trit(hidden_index)),
                    )
                });

            let chan_l1: [[b1; 32]; 32] = <[[(); 32]; 32] as Map<Hidden, b1>>::map(&l1, |pixel| {
                b1(pixel.get_bit(hidden_index))
            });
            let chan_acts: [u32; 10] = chan_l1.patch_fold(
                [0u32; 10],
                |acc: [u32; 10], patch: &<b1 as Element<AuxPatchShape>>::Array| {
                    <[(); 10] as ZipMap<
                        u32,
                        <<b1 as Element<AuxPatchShape>>::Array as BitArray>::TritArrayType,
                        u32,
                    >>::zip_map(&acc, &chan_aux_weights, |act, trits| {
                        act + trits.masked_distance(patch)
                    })
                },
            );
            dbg!(chan_acts);
            n_alive
        })
        .collect();
    let n_dead_chans: usize = grads.iter().filter(|&x| *x == 0usize).count();
    dbg!(n_dead_chans as f64 / <Hidden as BitArray>::BitShape::N as f64);
}

pub trait OrPool<const X: usize, const Y: usize>
where
    Self: Poolable2D<X, Y>,
{
    fn or_pool(&self) -> Self::Pooled;
}
