use bnn::bits::{b32, BitPack};
use bnn::datasets::cifar;
use bnn::descend::Descend;
use bnn::image2d::{PixelMap, PixelPack};
use bnn::layers::{Conv2D, FcMSE, GlobalAvgPool, Model, FC};
use bnn::unary;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::env;
use std::iter;
use std::path::PathBuf;
use std::time::Instant;

const N_EXAMPLES: usize = 50_0;

type WeightType = bool;
//type WeightType = Option<bool>;

type HiddenShape = [[(); 32]; 1];
type PixelShape = [(); 32];

type ConvPoolFcMSEModel = Conv2D<
    [[(); 32]; 32],
    PixelShape,
    HiddenShape,
    WeightType,
    GlobalAvgPool<[[(); 32]; 32], HiddenShape, FcMSE<HiddenShape, WeightType, 10>, 3, 3>,
    1,
    3,
    3,
    10,
>;

type Pixel = <PixelShape as BitPack<bool>>::T;
type InputType = <[[(); 32]; 32] as PixelPack<Pixel>>::I;
type IndexType = <ConvPoolFcMSEModel as Model<InputType, 10>>::Index;

fn main() {
    rayon::ThreadPoolBuilder::new()
        //.num_threads(16)
        .build_global()
        .unwrap();

    let base_path = {
        let mut args = env::args();
        args.next();
        let path = args
            .next()
            .expect("you must give path to base of mnist dir");
        PathBuf::from(&path)
    };
    let start = Instant::now();

    let (train_examples, test_examples) =
        cifar::load_examples_from_base(&base_path, (N_EXAMPLES, 10_000));
    dbg!(train_examples.len(), test_examples.len());

    let unary_train_examples: Vec<([[b32; 32]; 32], usize)> = train_examples
        .par_iter()
        .map(|(image, class)| {
            (
                <[[(); 32]; 32] as PixelMap<[u8; 3], b32>>::map(image, |&pixel| {
                    unary::u8x3_to_b32(pixel)
                }),
                *class,
            )
        })
        .collect();

    let unary_test_examples: Vec<([[b32; 32]; 32], usize)> = test_examples
        .par_iter()
        .map(|(image, class)| {
            (
                <[[(); 32]; 32] as PixelMap<[u8; 3], b32>>::map(image, |&pixel| {
                    unary::u8x3_to_b32(pixel)
                }),
                *class,
            )
        })
        .collect();

    let threshold = 0;
    let trunc = 10_000;
    let updates = 5;
    let max_minibatch_size = 128;
    let scale = (1, 2);
    let n_epochs = 1;

    println!("| n epochs | n updates | threshold | max | scale | truncation | updates | train acc | test acc | train time |");
    println!("| - | - | - | - | - | - | - | - | - | - | - |");

    //for &max_minibatch_size in &[128, 256, 512, 1024, 2048] {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let model = <ConvPoolFcMSEModel as Model<InputType, 10>>::rand(&mut rng);
    let start = Instant::now();

    let (model, min_minibatch, n_updates) = model.train_n_epochs(
        &unary_train_examples,
        threshold,
        trunc,
        updates,
        n_epochs,
        max_minibatch_size,
        scale,
    );

    println!(
        "| {} | {} | {} | {} | {}/{} | {} | {} | {:.3}% | {:.3}% | {:?} |",
        n_epochs,
        n_updates,
        threshold,
        max_minibatch_size,
        scale.0,
        scale.1,
        trunc,
        updates,
        model.avg_acc(&unary_train_examples) * 100f64,
        model.avg_acc(&unary_test_examples) * 100f64,
        start.elapsed()
    );
    //}
}
