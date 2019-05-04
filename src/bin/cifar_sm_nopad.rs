extern crate bitnn;
extern crate rand;
extern crate rand_hc;
//extern crate rayon;
extern crate time;

use bitnn::datasets::cifar;
use bitnn::layers::{Apply, InvertibleDownsample, SaveLoad};
use bitnn::train::{CacheBatch, EmbeddingSplitCacheBatch, TrainConv, TrainFC};
use bitnn::{
    BitLen, ExtractPatches, FlipBit, FlipBitIndexed, GetBit, GetPatch, HammingDistance, SetBit,
};
//use rand::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
//use rayon::prelude::*;
use std::fs;
use std::iter;
use std::marker::PhantomData;
use std::path::Path;
use time::PreciseTime;

const N_EXAMPLES: usize = 50_00;
const B0: usize = 1;
const B1: usize = 1;
const B2: usize = 1;
const FC: usize = 8;

fn main() {
    let path_string = format!("params/cifar_sm_n{}_adaptive_updatethresh10", N_EXAMPLES);
    let base_path = Path::new(&path_string);
    fs::create_dir_all(base_path).unwrap();

    let mut rng = Hc128Rng::seed_from_u64(1);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let weights: [[[[u32; 3]; 3]; 3]; 1] = rng.gen();
    let chan3_images: Vec<([[[u8; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);
    let conv_images: Vec<([[[u32; 1]; 32]; 32], usize)> = chan3_images
        .iter()
        .map(|(image, class)| (weights.apply(image), *class))
        .collect();
    let images_subset: Vec<_> = conv_images
        .iter()
        //.filter(|(_, class)| *class == 0)
        .take(20)
        .collect();
    //for b in 0..128 {
    //    for y in 0..30 {
    //        for x in 0..30 {
    //            print!("{}", images_subset[6].0[x][y].bit(b) as u8);
    //        }
    //        println!("",);
    //    }
    //    println!("",);
    //}

    let start = PreciseTime::now();
    let images30: Vec<([[[u32; B0]; 30]; 30], usize)> =
        <[[[[[u32; 1]; 3]; 3]; 32]; B0] as TrainConv<
            _,
            _,
            [[[u32; 1]; 3]; 3],
            _,
            [u32; B0],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &conv_images, &base_path.join("i30"), 8, 10);
    println!("time: {}", start.to(PreciseTime::now()));
    let images_subset: Vec<_> = images30
        .iter()
        //.filter(|(_, class)| *class == 0)
        .take(20)
        .collect();
    for b in 0..32 {
        for y in 0..30 {
            for x in 0..30 {
                print!("{}", images_subset[6].0[x][y].bit(b) as u8);
            }
            println!("",);
        }
        println!("",);
    }

    let start = PreciseTime::now();
    let images28: Vec<([[[u32; B0]; 28]; 28], usize)> =
        <[[[[[u32; B0]; 3]; 3]; 32]; B0] as TrainConv<
            _,
            _,
            [[[u32; B0]; 3]; 3],
            _,
            [u32; B0],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &images30, &base_path.join("i28"), 8, 10);
    println!("time: {}", start.to(PreciseTime::now()));

    let images_subset: Vec<_> = images28
        .iter()
        //.filter(|(_, class)| *class == 0)
        .take(20)
        .collect();
    for b in 0..32 {
        for y in 0..28 {
            for x in 0..28 {
                print!("{}", images_subset[6].0[x][y].bit(b) as u8);
            }
            println!("",);
        }
        println!("",);
    }
    let start = PreciseTime::now();
    let images26: Vec<([[[u32; B0]; 26]; 26], usize)> =
        <[[[[[u32; B0]; 3]; 3]; 32]; B0] as TrainConv<
            _,
            _,
            [[[u32; B0]; 3]; 3],
            _,
            [u32; B0],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &images28, &base_path.join("i26"), 8, 10);
    println!("time: {}", start.to(PreciseTime::now()));

    let start = PreciseTime::now();
    let images24: Vec<([[[u32; B0]; 24]; 24], usize)> =
        <[[[[[u32; B0]; 3]; 3]; 32]; B0] as TrainConv<
            _,
            _,
            [[[u32; B0]; 3]; 3],
            _,
            [u32; B0],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &images26, &base_path.join("i24"), 8, 10);
    println!("time: {}", start.to(PreciseTime::now()));

    let start = PreciseTime::now();
    let images: Vec<([[[u32; B0]; 22]; 22], usize)> =
        <[[[[[u32; B0]; 3]; 3]; 32]; B0] as TrainConv<
            _,
            _,
            [[[u32; B0]; 3]; 3],
            _,
            [u32; B0],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &images24, &base_path.join("i22"), 8, 10);
    println!("time: {}", start.to(PreciseTime::now()));

    let start = PreciseTime::now();
    let images: Vec<([[[u32; B1]; 20]; 20], usize)> =
        <[[[[[u32; B0]; 3]; 3]; 32]; B1] as TrainConv<
            _,
            _,
            [[[u32; B0]; 3]; 3],
            _,
            [u32; B1],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &images, &base_path.join("i20"), 8, 10);
    println!("time: {}", start.to(PreciseTime::now()));

    let start = PreciseTime::now();
    let images: Vec<([[[u32; B1]; 18]; 18], usize)> =
        <[[[[[u32; B1]; 3]; 3]; 32]; B1] as TrainConv<
            _,
            _,
            [[[u32; B1]; 3]; 3],
            _,
            [u32; B1],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &images, &base_path.join("i18"), 5, 30);
    println!("time: {}", start.to(PreciseTime::now()));

    //let start = PreciseTime::now();
    //let images: Vec<([[[u32; B1]; 16]; 16], usize)> =
    //    <[[[[[u32; B1]; 3]; 3]; 32]; B1] as TrainConv<
    //        _,
    //        _,
    //        [[[u32; B1]; 3]; 3],
    //        _,
    //        [u32; B1],
    //        CacheBatch<_, _, _>,
    //    >>::train(&mut rng, &images, &base_path.join("i16"), 5, 30);
    //println!("time: {}", start.to(PreciseTime::now()));

    //let start = PreciseTime::now();
    //let images: Vec<([[[u32; B1]; 14]; 14], usize)> =
    //    <[[[[[u32; B1]; 3]; 3]; 32]; B1] as TrainConv<
    //        _,
    //        _,
    //        [[[u32; B1]; 3]; 3],
    //        _,
    //        [u32; B1],
    //        EmbeddingSplitCacheBatch<_, _, _>,
    //    >>::train(&mut rng, &images, &base_path.join("i14"), 5, 30);
    //println!("time: {}", start.to(PreciseTime::now()));

    //let start = PreciseTime::now();
    //let images: Vec<([[[u32; B1]; 12]; 12], usize)> =
    //    <[[[[[u32; B1]; 3]; 3]; 32]; B1] as TrainConv<
    //        _,
    //        _,
    //        [[[u32; B1]; 3]; 3],
    //        _,
    //        [u32; B1],
    //        EmbeddingSplitCacheBatch<_, _, _>,
    //    >>::train(&mut rng, &images, &base_path.join("i12"), 5, 30);
    //println!("time: {}", start.to(PreciseTime::now()));

    //let start = PreciseTime::now();
    //let images: Vec<([[[u32; B2]; 10]; 10], usize)> =
    //    <[[[[[u32; B1]; 3]; 3]; 32]; B2] as TrainConv<
    //        _,
    //        _,
    //        [[[u32; B1]; 3]; 3],
    //        _,
    //        [u32; B2],
    //        EmbeddingSplitCacheBatch<_, _, _>,
    //    >>::train(&mut rng, &images, &base_path.join("i10"), 5, 30);
    //println!("time: {}", start.to(PreciseTime::now()));

    //let start = PreciseTime::now();
    //let images: Vec<([[[u32; B2]; 8]; 8], usize)> =
    //    <[[[[[u32; B2]; 3]; 3]; 32]; B2] as TrainConv<
    //        _,
    //        _,
    //        [[[u32; B2]; 3]; 3],
    //        _,
    //        [u32; B2],
    //        EmbeddingSplitCacheBatch<_, _, _>,
    //    >>::train(&mut rng, &images, &base_path.join("i8"), 5, 30);
    //println!("time: {}", start.to(PreciseTime::now()));

    //let start = PreciseTime::now();
    //let images: Vec<([[[u32; B2]; 6]; 6], usize)> =
    //    <[[[[[u32; B2]; 3]; 3]; 32]; B2] as TrainConv<
    //        _,
    //        _,
    //        [[[u32; B2]; 3]; 3],
    //        _,
    //        [u32; B2],
    //        EmbeddingSplitCacheBatch<_, _, _>,
    //    >>::train(&mut rng, &images, &base_path.join("i6"), 5, 30);
    //println!("time: {}", start.to(PreciseTime::now()));

    //let embeddings: Vec<([u32; FC], usize)> = <[[[[[u32; B2]; 6]; 6]; 32]; FC] as TrainFC<
    //    _,
    //    _,
    //    _,
    //    CacheBatch<_, _, _>,
    //>>::train(
    //    &mut rng, &images, &base_path.join("fc_head"), 4, 3
    //);
}
