extern crate bitnn;
extern crate rayon;
extern crate time;
use rayon::prelude::*;

use bitnn::datasets::cifar;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, unary, Patch, Pool2x2, SimplifyBits, WeightsMatrix};
use time::PreciseTime;

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let size: usize = 10000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images = Vec::with_capacity(size * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, size);
        images.append(&mut batch);
    }
    images
}

fn conv_8pixel<I: Copy, O: Default + Copy>(input: &[[I; 32]; 32], map_fn: &Fn(&[I; 8], &mut O)) -> [[O; 32]; 32] {
    let mut output = [[O::default(); 32]; 32];
    for x in 0..(32 - 2) {
        for y in 0..(32 - 2) {
            let patch = [
                input[x + 0][y + 0],
                input[x + 0][y + 1],
                input[x + 0][y + 2],
                input[x + 1][y + 0],
                input[x + 1][y + 1],
                input[x + 1][y + 2],
                input[x + 2][y + 0],
                input[x + 2][y + 1],
            ];
            map_fn(&patch, &mut output[x + 1][y + 1]);
        }
    }
    output
}

fn pixel_map<I, O: Copy + Default>(input: &[[I; 32]; 32], map_fn: &Fn(&I, &mut O)) -> [[O; 32]; 32] {
    let mut output = [[O::default(); 32]; 32];
    for x in 0..32 {
        for y in 0..32 {
            map_fn(&input[x][y], &mut output[x][y]);
        }
    }
    output
}

fn rgb_to_u16(pixels: [u8; 3]) -> u16 {
    unary::to_5(pixels[0]) as u16 | ((unary::to_5(pixels[1]) as u16) << 5) | ((unary::to_6(pixels[2]) as u16) << 10)
}

//let start = PreciseTime::now();
//println!("{} seconds just bitpack", start.to(PreciseTime::now()));

fn main() {
    let weights = [(0u128, [0u32; 4]); 16];
    let examples = load_data();
    let new_image = pixel_map::<[u8; 3], u16>(&examples[0].1, &|input, output| *output = rgb_to_u16(*input));
    let unary_examples: Vec<(usize, _)> = examples
        .par_iter()
        .map(|(class, image)| (*class, pixel_map::<[u8; 3], u16>(image, &|input, output| *output = rgb_to_u16(*input))))
        .collect();
    let conved_images: Vec<(usize, _)> = unary_examples
        .par_iter()
        .map(|(class, image)| {
            (
                *class,
                conv_8pixel::<u16, u64>(image, &|input, output| *output = weights.vecmul(&input.simplify()).simplify()).or_pool_2x2(),
            )
        }).collect();
}
