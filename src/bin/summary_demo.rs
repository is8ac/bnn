// if main overflows stack:
// ulimit -S -s 4194304

extern crate bitnn;
extern crate rayon;

use rayon::prelude::*;
use std::time::Instant;

use bitnn::bits::{b32, BitArray, BitMul, BitWord, Classify, IndexedFlipBit};
use bitnn::block::BlockCode;
use bitnn::count::CounterArray;
use bitnn::datasets::cifar;
use bitnn::image2d::StaticImage;
use bitnn::layer::{Apply, CountBits};
use bitnn::shape::Element;
use bitnn::unary::{Identity, Normalize, Unary};
use bitnn::weight::Objective;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::path::Path;

const N_EXAMPLES: usize = 50_000;
const N_CLASSES: usize = 10;
const K: usize = 27;
const TN: usize = 1;
type InputType = [[b32; 3]; 3];
type TargetType = [b32; TN];
type PreprocessorType = Unary<[[b32; 3]; 3]>;
//type PreprocessorType = Normalize<Unary<b32>>;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");

    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    //let examples = <()>::edges(&int_examples_32);

    let accumulator = <[[b32; 3]; 3] as CountBits<
        StaticImage<[[[u8; 3]; 32]; 32]>,
        [[(); 3]; 3],
        PreprocessorType,
        CounterArray<[[b32; 3]; 3], [(); K], { N_CLASSES }>,
    >>::count_bits(&int_examples_32);
    let orig: u64 = accumulator
        .counters
        .iter()
        .map(|x| x.iter().sum::<u32>() as u64)
        .sum();
    dbg!(orig);

    let start = Instant::now();
    let inputs: Vec<(u32, InputType, usize)> = accumulator
        .counters
        .par_iter()
        .enumerate()
        .map(|(class, inputs)| {
            inputs
                .par_iter()
                .enumerate()
                .filter(|(_, count)| **count > 3000)
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
    dbg!(total as f64 / orig as f64);

    let mut layer: <InputType as Element<<TargetType as BitArray>::BitShape>>::Array = rng.gen();
    let mut aux_weights = {
        let hidden_inputs: Vec<(u32, TargetType, usize)> = inputs
            .par_iter()
            .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
            .collect();
        <[TargetType; N_CLASSES]>::generate(&hidden_inputs)
    };

    let mut cur_loss: u64 = inputs
        .par_iter()
        .map(|(count, input, class)| {
            aux_weights.loss(&layer.bit_mul(input), *class) as u64 * *count as u64
        })
        .sum();
    let loss_start = Instant::now();
    let head_loss_start = Instant::now();
    let hidden_inputs: Vec<(u32, TargetType, usize)> = inputs
        .par_iter()
        .map(|(count, input, class)| (*count, layer.bit_mul(input), *class))
        .collect();
    dbg!(cur_loss as f64 / total as f64);
    aux_weights.decend(&hidden_inputs, &mut cur_loss);
    dbg!(cur_loss as f64 / total as f64);
    dbg!(head_loss_start.elapsed());
    //20.4968
    //l: 3.201
    for e in 0..4 {
        dbg!(e);
        let start = Instant::now();
        for ib in 0..InputType::BIT_LEN {
            for ob in 0..TargetType::BIT_LEN {
                layer.indexed_flip_bit(ob, ib);
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
                    layer.indexed_flip_bit(ob, ib);
                }
            }
        }
        dbg!(start.elapsed());
        dbg!(loss_start.elapsed());
    }
    let acc_start = Instant::now();
    let n_correct: u64 = int_examples_32
        .par_iter()
        .map(|(image, class)| {
            let hidden = <[[[[b32; 3]; 3]; 32]; TN] as Apply<
                StaticImage<[[[u8; 3]; 32]; 32]>,
                [[(); 3]; 3],
                PreprocessorType,
                StaticImage<[[TargetType; 32]; 32]>,
            >>::apply(&layer, image);
            let max_class = <[TargetType; N_CLASSES] as Classify<
                StaticImage<[[TargetType; 32]; 32]>,
                (),
                [(); N_CLASSES],
            >>::max_class(&aux_weights, &hidden);
            (max_class == *class) as u64
        })
        .sum();
    dbg!(n_correct as f64 / int_examples_32.len() as f64);
    dbg!(acc_start.elapsed());
    //0.10128
    //50: 0.1522
    //500: 0.16752
    //1000: 0.15516
    //K24: 0.10548
    // 0.15516
    //TN2:0.1383
    //TN1:0.139

    // pow1: 0.12286
    // pow2: 0.0987
    // 100: 27000
    // 50: 0.1548 i: 103282
    // 0.1339
    // 0.1317
    // 1332
}
