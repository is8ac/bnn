extern crate bitnn;
extern crate rayon;
extern crate time;
use rayon::prelude::*;

use bitnn::datasets::cifar;
//use bitnn::featuregen;
use bitnn::layers::unary;
use bitnn::layers::{NewFromSplit, ObjectiveHead, PatchMap, VecApply, Layer};
//use time::PreciseTime;

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let size: usize = 1000;
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

fn rgb_to_u16(pixels: [u8; 3]) -> u16 {
    unary::to_4(pixels[0]) as u16 | ((unary::to_5(pixels[1]) as u16) << 4) | ((unary::to_5(pixels[2]) as u16) << 9)
}

//let start = PreciseTime::now();
//println!("{} seconds just bitpack", start.to(PreciseTime::now()));

fn main() {
    let examples = load_data();
    let unary_examples: Vec<(usize, [[u16; 32]; 32])> = examples
        .iter()
        .map(|(class, image)| (*class, image.patch_map(&|input, output: &mut u16| *output = rgb_to_u16(*input))))
        .collect();
    let l0_patches: Vec<(usize, u128)> = ().vec_apply(&unary_examples);

    let mut layer1 = <[(u128, [u32; 4]); 16]>::new_from_split(&l0_patches);

    let l1_examples: Vec<(usize, [[u64; 32]; 32])> = layer1.vec_apply(&unary_examples);
    //let pooled_images: Vec<(usize, [[u64; 16]; 16])> = ().vec_apply(&l1_examples);
    let l2_patches: Vec<(usize, [u64; 8])> = ().vec_apply(&l1_examples);

    //let mut readout = <[[u64; 8]; 10]>::new_from_split(&l2_patches);
    let head = Layer::<[[u64; 32]; 32], [u64; 8], (), [[u64; 8]; 10]>::new_from_split(&l1_examples);
    //let head = Layer<[[u64; 32]; 32], [[u64; 16]; 16], (), Layer<[[u64; 16]; 16], [u64; 8], (), [[u64; 8]; 10]>>::new_from_split(&l1_examples);
    //let acc = l1_head.acc(&l1_examples);
}
// 32%
// PT40.8
// 153.9
