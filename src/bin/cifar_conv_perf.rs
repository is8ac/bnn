use bnn::bits::{b32, BitPack};
use bnn::datasets::cifar;
use bnn::descend::Descend;
use bnn::image2d::{PixelMap, PixelPack};
use bnn::layers::{Conv2D, FcMSE, GlobalAvgPool, Model, FC};
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
    ($input_shape:ty, $output_shape:ty, $version:expr) => {
        Conv2D<[[(); 32]; 32], $input_shape, $output_shape, WeightType, GlobalAvgPool<[[(); 32]; 32], $output_shape, FcMSE<$output_shape, WeightType, 10>, 3, 3>, $version, 3, 3, 10>
    }
}

macro_rules! time_loss_deltas {
    ($input_shape:ty, $output_shape:ty, $version:expr, $n_examples:expr, $n_threads:expr) => {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads($n_threads)
            .stack_size(2usize.pow(24))
            .build()
            .unwrap();

        let mut rng = Hc128Rng::seed_from_u64(0);
        let unary_examples: Vec<([[<$input_shape as BitPack<bool>>::T; 32]; 32], usize)> = (0
            ..$n_examples)
            .map(|_| (rng.gen(), rng.gen_range(0, 10)))
            .collect();

        let mut rng = Hc128Rng::seed_from_u64(0);
        let model = <model_type!($input_shape, $output_shape, $version) as Model<
            <[[(); 32]; 32] as PixelPack<<$input_shape as BitPack<bool>>::T>>::I,
            10,
        >>::rand(&mut rng);
        let start = Instant::now();
        let sum: usize = pool.install(|| {
            unary_examples
                .par_iter()
                .map(|(image, class)| {
                    let deltas = model.loss_deltas(image, 0, *class);
                    deltas.len()
                })
                .sum()
        });
        let elapsed = start.elapsed();
        println!(
            "| {} | {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3}",
            $version,
            $n_threads,
            <$input_shape>::N,
            <$output_shape>::N,
            <model_type!($input_shape, $output_shape, $version)>::N_PARAMS,
            elapsed.as_millis() as f64 / $n_examples as f64,
            (elapsed.as_millis() * $n_threads) as f64 / $n_examples as f64,
            (elapsed.as_nanos() * $n_threads) as f64 / ($n_examples * <$input_shape>::N) as f64,
            (elapsed.as_nanos() * $n_threads) as f64 / ($n_examples * <$output_shape>::N) as f64,
            (elapsed.as_nanos() * $n_threads) as f64
                / ($n_examples * <model_type!($input_shape, $output_shape, $version)>::N_PARAMS)
                    as f64,
        );
        if sum == 0 {
            panic!("no updates")
        }
    };
}

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    println!(
        "| version | threads | input pixel size | output pixel size | n params | total ms per example | ms per example | ns per pixel bit | ns per channel | ns per parameter |"
    );
    println!("| - | - | - | - | - | - | - | - | - | - |");

    time_loss_deltas!([[(); 32]; 1], [[(); 8]; 1], 4, 1024, 16);
    time_loss_deltas!([[(); 32]; 1], [[(); 16]; 1], 4, 1024, 16);
    time_loss_deltas!([[(); 32]; 1], [[(); 32]; 1], 4, 1024, 16);
    time_loss_deltas!([[(); 32]; 1], [[(); 32]; 2], 4, 1024, 16);
    time_loss_deltas!([[(); 32]; 1], [[(); 32]; 4], 4, 1024, 16);
    time_loss_deltas!([[(); 32]; 1], [[(); 32]; 8], 4, 1024, 16);
    time_loss_deltas!([[(); 32]; 1], [[(); 32]; 16], 4, 1024, 16);
    //time_loss_deltas!([[(); 8]; 1], [[(); 32]; 1], 4, 1024, 16);
    //time_loss_deltas!([[(); 16]; 1], [[(); 32]; 1], 4, 1024, 16);
    //time_loss_deltas!([[(); 32]; 1], [[(); 32]; 1], 4, 1024, 16);
    //time_loss_deltas!([[(); 32]; 2], [[(); 32]; 1], 4, 1024, 16);
    //time_loss_deltas!([[(); 32]; 4], [[(); 32]; 1], 4, 1024, 16);
    //time_loss_deltas!([[(); 32]; 8], [[(); 32]; 1], 4, 1024, 16);
    //time_loss_deltas!([[(); 32]; 16], [[(); 32]; 1], 4, 1024, 16);
    //time_loss_deltas!([[(); 32]; 32], [[(); 32]; 1], 4, 1024, 16);
}

//| 60 | 137.01666666666668 |
//| 120 | 3.3333333333333335 |
//| 1000 | 0.57 |
//| 1000 | 0.385 |

//| 60 | 562.7833333333333 |
//| 120 | 6.85 |
//| 1000 | 1.146 |
//| 1000 | 0.775 |
