// if main overflows stack:
// ulimit -S -s 4194304

extern crate bitnn;
extern crate rayon;

use rayon::prelude::*;
use std::time::Instant;

use bitnn::bits::{b32, BitArray, BitArrayOPs, BitMul, BitWord, Classify, IncrementFracCounters};
use bitnn::block::BlockCode;
use bitnn::count::{CounterArray, ElementwiseAdd};
use bitnn::datasets::cifar;
use bitnn::image2d::StaticImage;
use bitnn::layer::{
    AvgPoolLayer, BitPoolLayer, ClassifyLayer, ConcatImages, CountBits, IntToEdges, Layer,
};
use bitnn::shape::{Element, ZipMap};
use bitnn::unary::NormalizeAndBitpack;
use bitnn::weight::{
    Objective, SimpleClassify, SupervisedWeightsGen, UnsupervisedClusterWeightsGen,
};
use rand::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::path::Path;

const N_EXAMPLES: usize = 50_000;
const N_CLASSES: usize = 10;
const K: usize = 26;
type InputType = [[b32; 3]; 3];
type TargetType = b32;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");

    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let examples = <()>::edges(&int_examples_32);

    let accumulator = <[[b32; 3]; 3] as CountBits<
        StaticImage<[[b32; 32]; 32]>,
        [[b32; 3]; 3],
        CounterArray<[[b32; 3]; 3], [(); K], { N_CLASSES }>,
    >>::count_bits(&examples);

    let start = Instant::now();
    let inputs: Vec<(u32, InputType, usize)> = accumulator
        .counters
        .par_iter()
        .enumerate()
        .map(|(class, inputs)| {
            inputs
                .par_iter()
                .enumerate()
                .filter(|(_, count)| **count > 500)
                .map(move |(index, count)| (*count, index, class))
                .map(|(count, index, class)| {
                    (
                        count,
                        <InputType>::reverse_block(&accumulator.bit_matrix, index),
                        class,
                    )
                })
        })
        .flatten()
        .collect();
    dbg!(start.elapsed());
    dbg!(inputs.len());
    let mut layer: <InputType as Element<<TargetType as BitArray>::BitShape>>::Array = rng.gen();

    let start = Instant::now();
    let hidden_inputs: Vec<(u32, TargetType, usize)> = inputs
        .par_iter()
        .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
        .collect();
    dbg!(start.elapsed());

    let start = Instant::now();
    let mut aux_weights = <[(TargetType, u32); N_CLASSES]>::generate(&hidden_inputs);
    dbg!(start.elapsed());

    let (n_correct, total) = inputs
        .iter()
        .map(|(count, input, class)| {
            (
                ((aux_weights.max_class(&layer.bit_mul(input)) == *class) as u64) * *count as u64,
                *count as usize,
            )
        })
        .fold((0u64, 0usize), |a: (u64, usize), b: (u64, usize)| {
            (a.0 + b.0, a.1 + b.1)
        });
    dbg!(total);
    println!("{}%", n_correct as f64 / total as f64 * 100f64);

    let mut cur_loss: u64 = inputs
        .par_iter()
        .map(|(count, input, class)| {
            aux_weights.loss(&layer.bit_mul(input), *class) as u64 * *count as u64
        })
        .sum();
    let loss_start = Instant::now();
    for e in 0..3 {
        let hidden_inputs: Vec<(u32, TargetType, usize)> = inputs
            .par_iter()
            .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
            .collect();
        aux_weights = <[(TargetType, u32); N_CLASSES]>::generate(&hidden_inputs);
        dbg!(e);
        for b in 0..<InputType as Element<<TargetType as BitArray>::BitShape>>::Array::BIT_LEN {
            layer.flip_bit(b);
            let new_loss: u64 = inputs
                .par_iter()
                .map(|(count, input, class)| {
                    aux_weights.loss(&layer.bit_mul(input), *class) as u64 * *count as u64
                })
                .sum();
            if new_loss < cur_loss {
                cur_loss = new_loss;
                dbg!(cur_loss as f64 / total as f64);
            } else {
                layer.flip_bit(b);
            }
        }
    }
    dbg!(loss_start.elapsed());

    let (n_correct, total) = inputs
        .iter()
        .map(|(count, input, class)| {
            (
                ((aux_weights.max_class(&layer.bit_mul(input)) == *class) as u64) * *count as u64,
                *count as usize,
            )
        })
        .fold((0u64, 0usize), |a: (u64, usize), b: (u64, usize)| {
            (a.0 + b.0, a.1 + b.1)
        });
    dbg!(total);
    println!("{}%", n_correct as f64 / total as f64 * 100f64);
}
