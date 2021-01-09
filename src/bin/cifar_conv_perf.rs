use bnn::bits::{b32, BitPack};
use bnn::datasets::cifar;
use bnn::descend::Descend;
use bnn::image2d::{PixelMap, PixelPack};
use bnn::layers::{
    Conv2D, FcMSE, FusedConvSegmentedAvgPoolFcMSE, GlobalAvgPool, Model, SegmentedAvgPool, FC,
};
use bnn::shape::Shape;
use bnn::unary;
use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::env;
use std::iter;
use std::path::PathBuf;
use std::time::Instant;

type WeightType = bool;
//type WeightType = Option<bool>;

macro_rules! model_type {
    ($image_dim:expr, $input_shape:ty, $output_shape:ty, $segs:expr, $version:expr) => {
        FusedConvSegmentedAvgPoolFcMSE::<
            [[(); $image_dim]; $image_dim],
            $input_shape,
            $output_shape,
            WeightType,
            $version,
            $segs,
            $segs,
            3,
            3,
            10,
        >
    };
}

macro_rules! time_loss_deltas {
    ($image_dim:expr, $input_shape:ty, $output_shape:ty, $version:expr, $segs:expr, $n_examples:expr, $n_threads:expr) => {
        let pool = rayon::ThreadPoolBuilder::new().num_threads($n_threads).stack_size(2usize.pow(30)).build().unwrap();

        let mut rng = Hc128Rng::seed_from_u64(0);
        let unary_examples: Vec<([[<$input_shape as BitPack<bool>>::T; $image_dim]; $image_dim], usize)> = (0..$n_examples).map(|_| (rng.gen(), rng.gen_range(0, 10))).collect();

        let mut rng = Hc128Rng::seed_from_u64(0);
        let model =
            <model_type!($image_dim, $input_shape, $output_shape, $segs, $version) as Model<<[[(); $image_dim]; $image_dim] as PixelPack<<$input_shape as BitPack<bool>>::T>>::I, 10>>::rand(&mut rng);
        let start = Instant::now();
        let sum: usize = pool.install(|| {
            //let sum: usize = unary_examples
            unary_examples
                .par_iter()
                .map(|(image, class)| {
                    let deltas = model.loss_deltas(image, 0, *class);
                    deltas.len()
                })
                .sum()
        });
        let elapsed = start.elapsed();
        if sum == 0 {
            panic!();
        }
        println!(
            "| {} | {} | {} | {} | {} | {} | {:.3} | {:.3} | {:.3} |",
            $version,
            $segs,
            $n_threads,
            <$input_shape>::N,
            <$output_shape>::N,
            <model_type!($image_dim, $input_shape, $output_shape, $segs, $version)>::N_PARAMS,
            (elapsed.as_millis()) as f64 / $n_examples as f64,
            //(elapsed.as_millis() * $n_threads as u128) as f64 / $n_examples as f64,
            (elapsed.as_nanos() * $n_threads as u128) as f64 / ($n_examples * <$output_shape>::N) as f64,
            (elapsed.as_nanos() * $n_threads as u128) as f64 / ($n_examples * <model_type!($image_dim, $input_shape, $output_shape, $segs, $version)>::N_PARAMS) as f64,
            //(sum as f64 / $n_examples as f64) / <model_type!($image_dim, $input_shape, $output_shape, $segs, $version)>::N_PARAMS as f64,
        );
    };
}

fn main() {
    //rayon::ThreadPoolBuilder::new().num_threads(16).stack_size(2usize.pow(24)).build_global().unwrap();

    let mut rng = Hc128Rng::seed_from_u64(0);

    println!("| version | segs | threads | input pixel size | output pixel size | n params | ms per example | ns per channel | ns per parameter |");
    println!("| - | - | - | - | - | - | - | - | - |");

    //time_loss_deltas!(32, [[(); 32]; 4], [[(); 32]; 1], 8, 2, 25, 1);
    //time_loss_deltas!(32, [[(); 32]; 4], [[(); 32]; 2], 8, 2, 25, 1);
    //time_loss_deltas!(32, [[(); 32]; 4], [[(); 32]; 4], 8, 2, 25, 1);
    //time_loss_deltas!(32, [[(); 32]; 4], [[(); 32]; 8], 8, 2, 12, 1);
    //time_loss_deltas!(32, [[(); 32]; 4], [[(); 32]; 16], 8, 2, 16, 1);
    //time_loss_deltas!(32, [[(); 32]; 4], [[(); 32]; 32], 8, 2, 13, 1);
    //time_loss_deltas!(32, [[(); 32]; 1], [[(); 32]; 4], 8, 2, 25, 1);
    //time_loss_deltas!(32, [[(); 32]; 2], [[(); 32]; 4], 8, 2, 25, 1);
    //time_loss_deltas!(32, [[(); 32]; 4], [[(); 32]; 4], 8, 2, 25, 1);
    //time_loss_deltas!(32, [[(); 32]; 8], [[(); 32]; 4], 8, 2, 12, 1);
    //time_loss_deltas!(32, [[(); 32]; 16], [[(); 32]; 4], 8, 2, 16, 1);
    //time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 4], 8, 2, 13, 1);
    //time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 8, 1, 15, 1);
    //time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 8, 2, 15, 1);
    //time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 8, 3, 15, 1);
    //time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 8, 4, 15, 1);

    time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 7, 2, 15, 1);
    time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 7, 2, 15, 2);
    time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 7, 2, 15, 4);
    time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 7, 2, 15, 8);
    time_loss_deltas!(32, [[(); 32]; 32], [[(); 32]; 32], 7, 2, 15, 16);
}
