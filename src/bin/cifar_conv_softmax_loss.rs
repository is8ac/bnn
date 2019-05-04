extern crate bitnn;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;

use bitnn::datasets::cifar;
use bitnn::layers::{Apply, InvertibleDownsample, SaveLoad};
use bitnn::train::{CacheBatch, EmbeddingSplitCacheBatch, TrainConv, TrainFC};
use bitnn::{
    BitLen, ExtractPatches, FlipBit, FlipBitIndexed, GetBit, GetPatch, HammingDistance, SetBit,
};
use rand::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs;
use std::iter;
use std::marker::PhantomData;
use std::path::Path;
use time::PreciseTime;

//N=100: 72%

const N_EXAMPLES: usize = 500;
const B0: usize = 2;
const B1: usize = 4;
const B2: usize = 8;
const FC: usize = 8;

fn main() {
    let path_string = format!(
        "params/cifar_sm_n300_{}-{}-{}-{}_irev_test1",
        B0, B1, B2, FC
    );
    let base_path = Path::new(&path_string);
    fs::create_dir_all(base_path).unwrap();

    let mut rng = Hc128Rng::seed_from_u64(8);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let chan3_images: Vec<([[[u32; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);

    let start = PreciseTime::now();
    let mut b0_images: Vec<([[[u32; B0]; 32]; 32], usize)> =
        <[[[[[u32; 3]; 3]; 3]; 32]; B0] as TrainConv<
            _,
            _,
            [[[u32; 3]; 3]; 3],
            _,
            [u32; B0],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &chan3_images, &base_path.join("b0_trans"), 12, 2);

    println!("time: {}", start.to(PreciseTime::now()));

    for i in 0..3 {
        let start = PreciseTime::now();
        b0_images = <[[[[[u32; B0]; 3]; 3]; 32]; B0] as TrainConv<
            _,
            _,
            [[[u32; B0]; 3]; 3],
            _,
            [u32; B0],
            CacheBatch<_, _, _>,
        >>::train(
            &mut rng,
            &b0_images,
            &base_path.join(format!("b0_l{}", i)),
            8,
            2,
        );
        println!("time: {}", start.to(PreciseTime::now()));
    }
    //let images_subset: Vec<_> = b0_images.iter().filter(|(_, class)| *class == 0).take(20).collect();
    //for b in 0..64 {
    //    for x in 0..32 {
    //        for i in 0..11 {
    //            for y in 0..32 {
    //                print!("{}", images_subset[i].0[x][y].bit(b) as u8);
    //            }
    //            print!(" ");
    //        }
    //        println!("",);
    //    }
    //    println!("",);
    //}
    let start = PreciseTime::now();
    let images_16x16: Vec<([[[[[u32; B0]; 2]; 2]; 16]; 16], usize)> = b0_images
        .par_iter()
        .map(|(image, class)| (image.i_rev_pool(), *class))
        .collect();

    let mut b1_images: Vec<([[[u32; B1]; 16]; 16], usize)> =
        <[[[[[[[u32; B0]; 2]; 2]; 3]; 3]; 32]; B1] as TrainConv<
            _,
            _,
            [[[[[u32; B0]; 2]; 2]; 3]; 3],
            _,
            [u32; B1],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &images_16x16, &base_path.join("b1_trans"), 8, 2);
    println!("time: {}", start.to(PreciseTime::now()));

    let images_subset: Vec<_> = b1_images
        .iter()
        .filter(|(_, class)| *class == 0)
        .take(20)
        .collect();
    for b in 0..128 {
        for x in 0..16 {
            for i in 0..11 {
                for y in 0..16 {
                    print!("{}", images_subset[i].0[x][y].bit(b) as u8);
                }
                print!(" ");
            }
            println!("",);
        }
        println!("",);
    }

    for i in 0..7 {
        let start = PreciseTime::now();
        b1_images = <[[[[[u32; B1]; 3]; 3]; 32]; B1] as TrainConv<
            _,
            _,
            [[[u32; B1]; 3]; 3],
            _,
            [u32; B1],
            CacheBatch<_, _, _>,
        >>::train(
            &mut rng,
            &b1_images,
            &base_path.join(format!("b1_l{}", i)),
            8,
            2,
        );
        println!("time: {}", start.to(PreciseTime::now()));
    }

    let start = PreciseTime::now();
    let mut b2_images: Vec<([[[u32; B2]; 8]; 8], usize)> =
        <[[[[[u32; B1]; 2]; 2]; 32]; B2] as TrainConv<
            _,
            _,
            [[[u32; B1]; 2]; 2],
            _,
            [u32; B2],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &b1_images, &base_path.join("b2_trans"), 8, 2);
    println!("time: {}", start.to(PreciseTime::now()));

    for i in 0..6 {
        let start = PreciseTime::now();
        b2_images = <[[[[[u32; B2]; 3]; 3]; 32]; B2] as TrainConv<
            _,
            _,
            [[[u32; B2]; 3]; 3],
            _,
            [u32; B2],
            CacheBatch<_, _, _>,
        >>::train(
            &mut rng,
            &b2_images,
            &base_path.join(format!("b2_l{}", i)),
            5,
            2,
        );
        println!("time: {}", start.to(PreciseTime::now()));
    }
    let embeddings: Vec<([u32; FC], usize)> =
        <[[[[[u32; B2]; 8]; 8]; 32]; FC] as TrainFC<_, _, _, CacheBatch<_, _, _>>>::train(
            &mut rng,
            &b2_images,
            &base_path.join("fc_head"),
            4,
            3,
        );
}
