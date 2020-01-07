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
use bitnn::layer::CountBits;
use bitnn::shape::{Element, ZipMap};
use bitnn::unary::{Normalize, Unary};
use bitnn::weight::Objective;
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
    //let examples = <()>::edges(&int_examples_32);

    let accumulator = <[[b32; 3]; 3] as CountBits<
        StaticImage<[[[u8; 3]; 32]; 32]>,
        [[(); 3]; 3],
        //Normalize<Unary<b32>>,
        Unary<[[b32; 3]; 3]>,
        CounterArray<[[b32; 3]; 3], [(); K], { N_CLASSES }>,
    >>::count_bits(&int_examples_32);

    let start = Instant::now();
    let inputs: Vec<(u32, InputType, usize)> = accumulator
        .counters
        .par_iter()
        .enumerate()
        .map(|(class, inputs)| {
            inputs
                .par_iter()
                .enumerate()
                .filter(|(_, count)| **count > 1000)
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

    let total: u64 = inputs.par_iter().map(|(count, _, _)| *count as u64).sum();
    dbg!(total);

    let mut layer: <InputType as Element<<TargetType as BitArray>::BitShape>>::Array = rng.gen();
    let mut aux_weights = {
        let hidden_inputs: Vec<(u32, TargetType, usize)> = inputs
            .par_iter()
            .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
            .collect();
        <[(TargetType, u32); N_CLASSES]>::generate(&hidden_inputs)
    };

    let mut cur_loss: u64 = inputs
        .par_iter()
        .map(|(count, input, class)| {
            aux_weights.loss(&layer.bit_mul(input), *class) as u64 * *count as u64
        })
        .sum();
    let loss_start = Instant::now();
    for e in 0..7 {
        dbg!(e);
        for b in 0..<InputType as Element<<TargetType as BitArray>::BitShape>>::Array::BIT_LEN {
            if b % 11 == 0 {
                aux_weights = {
                    let hidden_inputs: Vec<(u32, TargetType, usize)> = inputs
                        .par_iter()
                        .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
                        .collect();
                    <[(TargetType, u32); N_CLASSES]>::generate(&hidden_inputs)
                };
                cur_loss = inputs
                    .par_iter()
                    .map(|(count, input, class)| {
                        aux_weights.loss(&layer.bit_mul(input), *class) as u64 * *count as u64
                    })
                    .sum();
            }
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
    {
        let hidden_inputs: Vec<(u32, TargetType, usize)> = inputs
            .par_iter()
            .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
            .collect();
        let aux_weights = <[(TargetType, u32); N_CLASSES]>::generate(&hidden_inputs);

        let (n_correct, total) = inputs
            .iter()
            .map(|(count, input, class)| {
                (
                    ((aux_weights.max_class(&layer.bit_mul(input)) == *class) as u64)
                        * *count as u64,
                    *count as usize,
                )
            })
            .fold((0u64, 0usize), |a: (u64, usize), b: (u64, usize)| {
                (a.0 + b.0, a.1 + b.1)
            });
        dbg!(total);
        println!("{}%", n_correct as f64 / total as f64 * 100f64);
    }
}
