extern crate bitnn;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::layers::{Conv2D, SaveLoad};
use bitnn::{BitLen, ExtractPatches, GetBit, TrainEncoder};
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs;
use std::path::Path;
use time::PreciseTime;

const N_EXAMPLES: usize = 10_000;
const EMBEDDING_LEN: usize = 1;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(26))
        .build_global()
        .unwrap();

    let base_path = Path::new("params/fc_test_1");
    fs::create_dir_all(base_path).unwrap();

    let mut rng = Hc128Rng::seed_from_u64(8);
    let images = mnist::load_images_u8_unary(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"),
        N_EXAMPLES,
    );
    let classes = mnist::load_labels(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"),
        N_EXAMPLES,
    );
    let examples: Vec<([[u8; 28]; 28], usize)> = images
        .iter()
        .cloned()
        .zip(classes.iter().map(|x| *x as usize))
        .collect();

    let patches: Vec<[[u8; 3]; 3]> = images
        .iter()
        .map(|image| {
            let patches: Vec<[[u8; 3]; 3]> = image.patches().iter().cloned().collect();
            patches
        })
        .flatten()
        .collect();
    dbg!(patches.len());

    let encoder = <[[[[u8; 3]; 3]; 32]; EMBEDDING_LEN] as TrainEncoder<
        _,
        _,
        [[[[u32; EMBEDDING_LEN]; 8]; 3]; 3],
    >>::train_encoder(&mut rng, &patches);

    let examples: Vec<([[[u32; EMBEDDING_LEN]; 28]; 28], usize)> = examples
        .par_iter()
        .map(|(image, class)| (encoder.conv2d(image), *class))
        .collect();
    let patches: Vec<[[[u32; EMBEDDING_LEN]; 3]; 3]> = examples
        .iter()
        .map(|(image, _)| {
            let patches: Vec<[[[u32; EMBEDDING_LEN]; 3]; 3]> =
                image.patches().iter().cloned().collect();
            patches
        })
        .flatten()
        .collect();

    let encoder = <[[[[[u32; EMBEDDING_LEN]; 3]; 3]; 32]; EMBEDDING_LEN] as TrainEncoder<
        _,
        _,
        [[[[[u32; EMBEDDING_LEN]; 32]; EMBEDDING_LEN]; 3]; 3],
    >>::train_encoder(&mut rng, &patches);

    let examples: Vec<([[[u32; EMBEDDING_LEN]; 28]; 28], usize)> = examples
        .par_iter()
        .map(|(image, class)| (encoder.conv2d(image), *class))
        .collect();

    let image = examples.iter().find(|(_, class)| *class == 0).unwrap().0;
    for b in 0..32 * EMBEDDING_LEN {
        for x in 0..28 {
            for y in 0..28 {
                print!("{}", if image[x][y].bit(b) { "*" } else { " " });
            }
            print!("\n",);
        }
    }
}
