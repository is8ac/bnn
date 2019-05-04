extern crate bitnn;
extern crate image;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;
use image::{GenericImage, ImageBuffer, Rgb};

use bitnn::datasets::cifar;
use bitnn::layers::{Apply, InvertibleDownsample, SaveLoad};
use bitnn::train::{CacheBatch, EmbeddingSplitCacheBatch, TrainConv, TrainFC};
use bitnn::vec_concat_2_examples;
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
//const B0: usize = 1;
//const B1: usize = 2;
//const B2: usize = 4;
//const FC: usize = 8;

fn main() {
    let path_string = "params/cifar_sm_n500_irev_dense_test3";
    let base_path = Path::new(&path_string);
    fs::create_dir_all(base_path).unwrap();

    let mut rng = Hc128Rng::seed_from_u64(1);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let chan3_images: Vec<([[[u32; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);

    for (image_n, (image, class)) in chan3_images.iter().enumerate() {
        let mut target_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(32u32, 32u32);
        for x in 0..32 {
            for y in 0..32 {
                let mut pixel = [0u8; 3];
                for b in 0..3 {
                    pixel[b] = (image[x][y][b].count_ones() * 8) as u8;
                }
                target_image.get_pixel_mut(x as u32, y as u32).data = pixel;
            }
        }
        target_image
            .save(format!("images/c{}_{}.png", class, image_n))
            .unwrap();
    }

    let start = PreciseTime::now();
    let c0_images: Vec<([[[u32; 2]; 32]; 32], usize)> =
        <[[[[[u32; 3]; 3]; 3]; 32]; 2] as TrainConv<
            _,
            _,
            [[[u32; 3]; 3]; 3],
            _,
            [u32; 2],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &chan3_images, &base_path.join("conv0"), 12, 2);

    let irev_0_16x16_images: Vec<([[[[[u32; 3]; 2]; 2]; 16]; 16], usize)> = chan3_images
        .par_iter()
        .map(|(image, class)| (image.i_rev_pool(), *class))
        .collect();

    let b0_images: Vec<([[[u32; 2]; 16]; 16], usize)> =
        <[[[[[[[u32; 3]; 2]; 2]; 3]; 3]; 32]; 2] as TrainConv<
            _,
            _,
            [[[[[u32; 3]; 2]; 2]; 3]; 3],
            _,
            [u32; 2],
            CacheBatch<_, _, _>,
        >>::train(
            &mut rng,
            &irev_0_16x16_images,
            &base_path.join("irev0"),
            12,
            2,
        );
    let irev_1_16x16_images: Vec<([[[[[u32; 2]; 2]; 2]; 16]; 16], usize)> = c0_images
        .par_iter()
        .map(|(image, class)| (image.i_rev_pool(), *class))
        .collect();

    let c2_images: Vec<([[_; 16]; 16], usize)> =
        vec_concat_2_examples(&b0_images, &irev_1_16x16_images);

    let b1_images: Vec<([[[u32; 4]; 16]; 16], usize)> =
        <[[[[_; 3]; 3]; 32]; 4] as TrainConv<
            _,
            _,
            [[_; 3]; 3],
            _,
            [u32; 4],
            CacheBatch<_, _, _>,
        >>::train(&mut rng, &c2_images, &base_path.join("conv1"), 12, 2);

    //let irev_8x8: Vec<([[[[[u32; B0]; 2]; 2]; 8]; 8], usize)> = b0_images
    //    .par_iter()
    //    .map(|(image, class)| (image.i_rev_pool(), *class))
    //    .collect();

    //let mut b1_images: Vec<([[[u32; B1]; 8]; 8], usize)> =
    //    <[[[[[[[u32; B0]; 2]; 2]; 3]; 3]; 32]; B1] as TrainConv<
    //        _,
    //        _,
    //        [[[[[u32; B0]; 2]; 2]; 3]; 3],
    //        _,
    //        [u32; B1],
    //        CacheBatch<_, _, _>,
    //    >>::train(&mut rng, &irev_8x8, &base_path.join("irev1"), 11, 2);

    //let embeddings: Vec<([u32; FC], usize)> =
    //    <[[[[[u32; B1]; 8]; 8]; 32]; FC] as TrainFC<_, _, _, CacheBatch<_, _, _>>>::train(
    //        &mut rng,
    //        &b1_images,
    //        &base_path.join("fc_head"),
    //        4,
    //        3,
    //    );
    //println!("time: {}", start.to(PreciseTime::now()));
}
