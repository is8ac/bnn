extern crate bitnn;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::layers::Apply;
use bitnn::layers::SaveLoad;
use bitnn::train::{CacheBatch, TrainFC};
use bitnn::{BitLen, FlipBit, FlipBitIndexed, GetBit, GetPatch, HammingDistance, SetBit};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs;
use std::marker::PhantomData;
use std::path::Path;
use time::PreciseTime;

const N_EXAMPLES: usize = 60_000;

fn main() {
    let base_path = Path::new("params/fc_test_1");
    fs::create_dir_all(base_path).unwrap();

    let mut rng = Hc128Rng::seed_from_u64(8);
    let images = mnist::load_images_bitpacked(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"),
        N_EXAMPLES,
    );
    let classes = mnist::load_labels(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"),
        N_EXAMPLES,
    );
    let examples: Vec<([u64; 13], usize)> = images
        .iter()
        .cloned()
        .zip(classes.iter().map(|x| *x as usize))
        .collect();

    let start = PreciseTime::now();
    let embeddings: Vec<([u32; 2], usize)> = <[[[u64; 13]; 32]; 2] as TrainFC<
        _,
        _,
        _,
        CacheBatch<_, _, _>,
    >>::train(
        &mut rng, &examples, &base_path.join("l0"), 8, 1
    );

    println!("time: {}", start.to(PreciseTime::now()));

    //let start = PreciseTime::now();
    //let embeddings = <[[[u32; 8]; 32]; 4] as TrainFC<
    //    _,
    //    _,
    //    _,
    //    CacheBatch<_, _, _>,
    //>>::train(
    //    &mut rng,
    //    &embeddings,
    //    &base_path.join("l1"),
    //    11,
    //    6,
    //);

    //println!("time: {}", start.to(PreciseTime::now()));
    //let start = PreciseTime::now();
    //let embeddings = <[[[u32; 4]; 32]; 4] as TrainFC<
    //    _,
    //    _,
    //    _,
    //    CacheBatch<_, _, _>,
    //>>::train(
    //    &mut rng,
    //    &embeddings,
    //    &base_path.join("l2"),
    //    11,
    //    6,
    //);
    //println!("time: {}", start.to(PreciseTime::now()));
    //let start = PreciseTime::now();
    //let embeddings = <[[[u32; 4]; 32]; 4] as TrainFC<
    //    _,
    //    _,
    //    _,
    //    CacheBatch<_, _, _>,
    //>>::train(
    //    &mut rng,
    //    &embeddings,
    //    &base_path.join("l3"),
    //    11,
    //    6,
    //);
    //println!("time: {}", start.to(PreciseTime::now()));

    //let avg_loss = cache_batch.bit_loss(1);
    //println!("loss: {}", avg_loss);
    // 83.194%
    // full head: 83.294% PT496S
    //   no head: 81.27% PT66S

    // no head, 8 embedding
    // acc:  81.692%
    // time: PT158.867849050S

    // part_head
    // acc:  83%
    // time: PT166.743565312S
    // depth: 11 84.7
    // acc: 85.27333333333334%
    // time: PT279.716677584S
}
