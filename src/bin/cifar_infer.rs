extern crate bitnn;
extern crate rayon;
extern crate time;
use time::PreciseTime;

use bitnn::datasets::cifar;
//use bitnn::featuregen;
use bitnn::layers::unary;
use bitnn::layers::{
    Accuracy, Infer, Layer, NewFromSplit, Objective, OptimizeHead, OrPool2x2, Patch3x3NotchedConv,
    PatchMap, PoolOrLayer, SaveLoad, VecApply,
};
use std::marker::PhantomData;
use std::path::Path;

fn rgb_to_u16(pixels: [u8; 3]) -> u16 {
    unary::to_5(pixels[0]) as u16
        | ((unary::to_5(pixels[1]) as u16) << 5)
        | ((unary::to_6(pixels[2]) as u16) << 10)
}

const TEST_N: usize = 10000;

fn main() {
    let test_images = cifar::load_images_10(
        &String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/test_batch.bin"),
        TEST_N,
    );

    let model = Layer::<
        [[u16; 32]; 32],
        Patch3x3NotchedConv<u16, u128, [u128; 16], u16>,
        [[u16; 32]; 32],
        Layer<
            [[u16; 32]; 32],
            OrPool2x2<u16>,
            [[u16; 16]; 16],
            Layer<
                [[u16; 16]; 16],
                Patch3x3NotchedConv<u16, u128, [u128; 16], u16>,
                [[u16; 16]; 16],
                Layer<
                    [[u16; 16]; 16],
                    OrPool2x2<_>,
                    [[u16; 8]; 8],
                    Layer<
                        [[u16; 8]; 8],
                        Patch3x3NotchedConv<u16, u128, [u128; 32], u32>,
                        [[u32; 8]; 8],
                        Layer<
                            [[u32; 8]; 8],
                            Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>,
                            [[u32; 8]; 8],
                            Layer<
                                [[u32; 8]; 8],
                                OrPool2x2<u32>,
                                [[u32; 4]; 4],
                                [[[u32; 4]; 4]; 10],
                            >,
                        >,
                    >,
                >,
            >,
        >,
    >::new_from_fs(&Path::new("params/full_model_e2e.prms"));

    let start = PreciseTime::now();
    let mut sum_correct = 0u64;
    for (class, image) in test_images {
        let correct = class
            == model
                .top1(&image.patch_map(&|input, output: &mut u16| *output = rgb_to_u16(*input)));
        sum_correct += correct as u64;
    }

    println!("time: {}", start.to(PreciseTime::now()));
    println!("correct: {:?}", sum_correct as f64 / TEST_N as f64);
}

// 34.7% train acc
