#![feature(const_generics)]

use bitnn::bits::{b32, BitArray, BitMul, BitWord, Classify, IndexedFlipBit};
use bitnn::cluster::{ImageCountByCentroids, ImagePatchLloyds};
use bitnn::datasets::cifar;
use bitnn::image2d::StaticImage;
use bitnn::layer::Apply;
use bitnn::shape::Element;
use bitnn::unary::{Preprocess, Unary};
use bitnn::weight::Objective;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

fn sum_loss<I: Element<O::BitShape> + Sync, O: BitArray + Sync, const C: usize>(
    layer_weights: &<I as Element<O::BitShape>>::Array,
    aux_weights: &[O; C],
    counts: &[(I, [u32; C])],
) -> u64
where
    [O; C]: Objective<O, C>,
    <I as Element<O::BitShape>>::Array: BitMul<I, O> + Sync,
{
    counts
        .par_iter()
        .map(|(input, counts)| aux_weights.count_loss(&layer_weights.bit_mul(input), counts) as u64)
        .sum()
}

fn decend<
    RNG: Rng,
    I: BitArray + Element<O::BitShape> + Sync + BitWord,
    O: BitArray + Sync + BitWord + Send,
    const C: usize,
>(
    rng: &mut RNG,
    layer_weights: &mut <I as Element<O::BitShape>>::Array,
    aux_weights: &mut [O; C],
    counts: &[(I, [u32; C])],
    window_size: usize,
    depth: usize,
) where
    [O; C]: Objective<O, C>,
    <I as Element<O::BitShape>>::Array: BitMul<I, O> + Sync + IndexedFlipBit<I, O>,
{
    dbg!(depth);
    if depth > 0 {
        decend(
            rng,
            layer_weights,
            aux_weights,
            counts,
            (window_size * 4) / 5,
            depth - 1,
        );
    }
    dbg!(window_size);
    let n_examples: u64 = counts
        .iter()
        .map(|(_, c)| c.iter().sum::<u32>() as u64)
        .sum();
    let mut cur_sum_loss = sum_loss(&*layer_weights, &aux_weights, &counts);
    let hidden_counts: Vec<(O, [u32; C])> = counts
        .par_iter()
        .map(|(input, counts)| (layer_weights.bit_mul(input), *counts))
        .collect();
    aux_weights.count_decend(&hidden_counts, &mut cur_sum_loss);
    let indices = {
        let mut indices: Vec<usize> = (0..I::BIT_LEN)
            .map(|i| (i * (counts.len() - window_size)) / I::BIT_LEN)
            .collect();
        rng.shuffle(&mut indices);
        indices
    };
    for (ib, &index) in indices.iter().enumerate() {
        let minibatch = &counts[index..index + window_size];
        cur_sum_loss = sum_loss(&*layer_weights, &aux_weights, &minibatch);
        //dbg!(cur_sum_loss);
        for ob in 0..O::BIT_LEN {
            layer_weights.indexed_flip_bit(ob, ib);
            let new_loss: u64 = sum_loss(&*layer_weights, &aux_weights, &minibatch);
            if new_loss < cur_sum_loss {
                cur_sum_loss = new_loss;
            } else {
                layer_weights.indexed_flip_bit(ob, ib);
            }
        }
    }
    dbg!(cur_sum_loss as f64 / n_examples as f64);
}

fn avg_acc<
    I: BitArray + Element<O::BitShape> + Sync + BitWord,
    O: BitArray + Sync + BitWord + Send,
    const C: usize,
>(
    examples: &Vec<(StaticImage<[[[u8; 3]; 32]; 32]>, usize)>,
    layer_weights: &<I as Element<O::BitShape>>::Array,
    aux_weights: &[O; C],
) -> f64
where
    [O; C]: Objective<O, C> + Classify<StaticImage<[[O; 32]; 32]>, (), [(); 10]>,
    <I as Element<O::BitShape>>::Array: BitMul<I, O>
        + Sync
        + IndexedFlipBit<I, O>
        + Apply<StaticImage<[[[u8; 3]; 32]; 32]>, [[(); 3]; 3], Unary<I>, StaticImage<[[O; 32]; 32]>>,
    Unary<I>: Preprocess<[[[u8; 3]; 3]; 3]>,
{
    let acc_start = Instant::now();
    let n_correct: u64 = examples
        .par_iter()
        .map(|(image, class)| {
            let hidden_state = <<I as Element<<O as BitArray>::BitShape>>::Array as Apply<
                StaticImage<[[[u8; 3]; 32]; 32]>,
                [[(); 3]; 3],
                Unary<I>,
                StaticImage<[[O; 32]; 32]>,
            >>::apply(layer_weights, image);
            let max_class =
                <[O; C] as Classify<StaticImage<[[O; 32]; 32]>, (), [(); 10]>>::max_class(
                    aux_weights,
                    &hidden_state,
                );
            (max_class == *class) as u64
        })
        .sum();
    dbg!(acc_start.elapsed());
    n_correct as f64 / examples.len() as f64
}

// 1000: 0.199 acc
const N_EXAMPLES: usize = 50_000;
const K: usize = 300;
type PatchType = [[b32; 3]; 3];
type OutputType = [b32; 2];

fn main() {
    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let mut rng = Hc128Rng::seed_from_u64(0);
    let start = Instant::now();
    let centroids = <PatchType as ImagePatchLloyds<
        StaticImage<[[[u8; 3]; 32]; 32]>,
        [[(); 3]; 3],
        Unary<PatchType>,
    >>::lloyds(&mut rng, &int_examples_32, K, 4);
    dbg!(start.elapsed());

    let start = Instant::now();
    let counts = <PatchType as ImageCountByCentroids<
        StaticImage<[[[u8; 3]; 32]; 32]>,
        [[(); 3]; 3],
        Unary<PatchType>,
        [(); 10],
    >>::count_by_centroids(&int_examples_32, &centroids);

    dbg!(start.elapsed());
    //dbg!(&counts);
    dbg!(counts.len());
    let n_examples: u64 = counts
        .iter()
        .map(|(_, c)| c.iter().sum::<u32>() as u64)
        .sum();
    dbg!(n_examples);

    let mut seed_accs = vec![];
    for seed in 0..10 {
        dbg!(seed);
        let mut rng = Hc128Rng::seed_from_u64(seed);

        let mut layer_weights: <PatchType as Element<<OutputType as BitArray>::BitShape>>::Array =
            rng.gen();
        let mut aux_weights: [OutputType; 10] = rng.gen();

        let decend_start = Instant::now();
        decend(
            &mut rng,
            &mut layer_weights,
            &mut aux_weights,
            &counts,
            K,
            8,
        );
        dbg!(decend_start.elapsed());
        let acc = avg_acc(&int_examples_32, &layer_weights, &aux_weights);
        dbg!(acc);
        seed_accs.push((seed, acc));
    }
    for (seed, acc) in &seed_accs {
        println!("{}: {}", seed, acc);
    }
    let sum_acc: f64 = seed_accs.iter().map(|(_, acc)| *acc).sum();
    dbg!(sum_acc / seed_accs.len() as f64);
}
