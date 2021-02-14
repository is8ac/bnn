use bnn::bits::{b32, BitPack};
use bnn::datasets::cifar;
use bnn::descend::Descend;
use bnn::image2d::{ImageShape, PixelFold, PixelMap, PixelPack};
use bnn::layers::{FusedConvSegmentedAvgPoolFcMSE, Model};
use bnn::shape::{Map, Shape, ZipMap};
use bnn::unary;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::env;
use std::iter;
use std::path::PathBuf;
use std::time::Instant;

const N_EXAMPLES: usize = 50_000;
const SEG: usize = 2;
const PATCH: usize = 3;

//type WeightType = bool;
type WeightType = Option<bool>;

type HiddenShape = [[(); 32]; 4];
type PixelShape = [[(); 32]; 3];

type ConvPoolFcMSEModel = FusedConvSegmentedAvgPoolFcMSE<
    [[(); 32]; 32],
    PixelShape,
    HiddenShape,
    WeightType,
    7,
    SEG,
    SEG,
    PATCH,
    PATCH,
    10,
>;
type L2ConvPoolFcMSEModel = FusedConvSegmentedAvgPoolFcMSE<
    [[(); 32]; 32],
    HiddenShape,
    HiddenShape,
    WeightType,
    7,
    SEG,
    SEG,
    PATCH,
    PATCH,
    10,
>;

type Pixel = <PixelShape as BitPack<bool>>::T;
type HiddenPixel = <HiddenShape as BitPack<bool>>::T;
type InputType = <[[(); 32]; 32] as PixelPack<Pixel>>::I;
type IndexType = <ConvPoolFcMSEModel as Model<InputType, 10>>::Index;

// scalea should be between 0 an 256, usually around 128.
fn normalize_image<
    IS: ImageShape
        + PixelMap<[u8; 3], [b32; 3], 0, 0>
        + PixelFold<(usize, [u32; 3]), [u8; 3], 0, 0>
        + PixelFold<[f32; 3], [u8; 3], 0, 0>,
>(
    image: &<IS as PixelPack<[u8; 3]>>::I,
) -> <IS as PixelPack<[b32; 3]>>::I {
    let (n_pixels, sums) = <IS as PixelFold<(usize, [u32; 3]), [u8; 3], 0, 0>>::pixel_fold(
        image,
        (0usize, [0u32; 3]),
        |counts, pixel| {
            (
                counts.0 + 1,
                <[(); 3] as ZipMap<u32, u8, u32>>::zip_map(&counts.1, pixel, |sum, &c| {
                    sum + c as u32
                }),
            )
        },
    );
    let n = n_pixels as f32;
    let means = <[(); 3] as Map<u32, f32>>::map(&sums, |&x| x as f32 / n);
    let squared_sdevs = <IS as PixelFold<[f32; 3], [u8; 3], 0, 0>>::pixel_fold(
        image,
        [0f32; 3],
        |counts, pixel| {
            let dev = <[(); 3] as ZipMap<f32, u8, f32>>::zip_map(&means, pixel, |mean, &c| {
                (c as f32 - mean)
            });
            <[(); 3] as ZipMap<f32, f32, f32>>::zip_map(&counts, &dev, |sum, &sd| sum + sd.powi(2))
        },
    );
    let sdevs = <[(); 3] as Map<f32, f32>>::map(&squared_sdevs, |&x| (x / n).sqrt());
    <IS as PixelMap<[u8; 3], [b32; 3], 0, 0>>::map(image, |pixel| {
        let centered =
            <[(); 3] as ZipMap<f32, u8, f32>>::zip_map(&means, pixel, |mean, &c| c as f32 - mean);
        <[(); 3] as ZipMap<f32, f32, b32>>::zip_map(&centered, &sdevs, |pixel, sdev| {
            unary::f32_to_b32((pixel / sdev).tanh())
        })
    })
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        //.num_threads(16)
        .stack_size(2usize.pow(30))
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

    let unary_train_examples: Vec<([[[b32; 3]; 32]; 32], usize)> = train_examples
        .par_iter()
        .map(|(image, class)| (normalize_image::<[[(); 32]; 32]>(image), *class))
        .collect();
    let unary_test_examples: Vec<([[[b32; 3]; 32]; 32], usize)> = test_examples
        .par_iter()
        .map(|(image, class)| (normalize_image::<[[(); 32]; 32]>(image), *class))
        .collect();

    let threshold = 0;
    let updates = HiddenShape::N / 2;
    let start_minibatch_size = 32;
    let scale = (3, 2);
    let n_epochs = 8;

    println!("| n epochs | n updates | threshold | max | scale | truncation | updates | train acc | test acc | train time |");
    println!("| - | - | - | - | - | - | - | - | - | - | - |");

    //for &start_minibatch_size in &[64, 128, 256, 512] {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let model = <ConvPoolFcMSEModel as Model<InputType, 10>>::rand(&mut rng);

    let start = Instant::now();
    let (model, min_minibatch, n_updates) = model.train_n_epochs(
        &unary_train_examples,
        updates,
        n_epochs,
        start_minibatch_size,
        scale,
    );

    println!(
        "| {} | {} | {} | {}/{} | {} | {} | {:.3}% | {:.3}% | {:?} |",
        n_epochs,
        n_updates,
        threshold,
        start_minibatch_size,
        scale.0,
        scale.1,
        updates,
        model.avg_acc(&unary_train_examples) * 100f64,
        model.avg_acc(&unary_test_examples) * 100f64,
        start.elapsed()
    );
    //}

    let l2_unary_train_examples: Vec<_> = unary_train_examples
        .par_iter()
        .map(|(image, class)| (model.apply(image), *class))
        .collect();
    let l2_unary_test_examples: Vec<_> = unary_test_examples
        .par_iter()
        .map(|(image, class)| (model.apply(image), *class))
        .collect();

    let mut rng = Hc128Rng::seed_from_u64(0);
    let model = <L2ConvPoolFcMSEModel as Model<
        <[[(); 32]; 32] as PixelPack<HiddenPixel>>::I,
        10,
    >>::rand(&mut rng);

    let start = Instant::now();
    let (model, min_minibatch, n_updates) = model.train_n_epochs(
        &l2_unary_train_examples,
        updates,
        n_epochs,
        start_minibatch_size,
        scale,
    );

    println!(
        "| {} | {} | {} | {}/{} | {} | {} | {:.3}% | {:.3}% | {:?} |",
        n_epochs,
        n_updates,
        threshold,
        start_minibatch_size,
        scale.0,
        scale.1,
        updates,
        model.avg_acc(&l2_unary_train_examples) * 100f64,
        model.avg_acc(&l2_unary_test_examples) * 100f64,
        start.elapsed()
    );

    let l3_unary_train_examples: Vec<_> = l2_unary_train_examples
        .par_iter()
        .map(|(image, class)| (model.apply(image), *class))
        .collect();
    let l3_unary_test_examples: Vec<_> = l2_unary_test_examples
        .par_iter()
        .map(|(image, class)| (model.apply(image), *class))
        .collect();

    let mut rng = Hc128Rng::seed_from_u64(0);
    let model = <L2ConvPoolFcMSEModel as Model<
        <[[(); 32]; 32] as PixelPack<HiddenPixel>>::I,
        10,
    >>::rand(&mut rng);

    let start = Instant::now();
    let (model, min_minibatch, n_updates) = model.train_n_epochs(
        &l3_unary_train_examples,
        updates,
        n_epochs,
        start_minibatch_size,
        scale,
    );

    println!(
        "| {} | {} | {} | {}/{} | {} | {} | {:.3}% | {:.3}% | {:?} |",
        n_epochs,
        n_updates,
        threshold,
        start_minibatch_size,
        scale.0,
        scale.1,
        updates,
        model.avg_acc(&l3_unary_train_examples) * 100f64,
        model.avg_acc(&l3_unary_test_examples) * 100f64,
        start.elapsed()
    );
}
