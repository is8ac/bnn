use bnn::bits::PackedElement;
use bnn::datasets::mnist;
use bnn::descend::Descend;
use bnn::layers::{FcMSE, Model, FC};
use bnn::shape::Map;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::iter;
use std::path::PathBuf;
use std::time::Instant;

const N_EXAMPLES: usize = 60_000;
const N_CLASSES: usize = 10;

type InputShape = [[(); 32]; 25];
type HiddenShape = [[(); 32]; 4];

//type WeightType = bool;
type WeightType = Option<bool>;

type InputType = <bool as PackedElement<InputShape>>::Array;

type H0Model = FcMSE<InputShape, WeightType, N_CLASSES>;
type H1Model =
    FC<InputShape, WeightType, HiddenShape, FcMSE<HiddenShape, WeightType, N_CLASSES>, N_CLASSES>;
type H2Model = FC<
    InputShape,
    WeightType,
    HiddenShape,
    FC<HiddenShape, WeightType, HiddenShape, FcMSE<HiddenShape, WeightType, N_CLASSES>, N_CLASSES>,
    N_CLASSES,
>;
type H3Model = FC<
    InputShape,
    WeightType,
    HiddenShape,
    FC<
        HiddenShape,
        WeightType,
        HiddenShape,
        FC<
            HiddenShape,
            WeightType,
            HiddenShape,
            FcMSE<HiddenShape, WeightType, N_CLASSES>,
            N_CLASSES,
        >,
        N_CLASSES,
    >,
    N_CLASSES,
>;

fn recursively_train<M: Model<InputType, N_CLASSES> + Descend<InputType, N_CLASSES> + Sync>(
    model: M,
    examples: &Vec<(InputType, usize)>,
    example_truncation: usize,
    n_updates: usize,
    minibatch_size: usize,
    min: usize,
    scale: (usize, usize),
) -> (M, usize, usize) {
    let (model, n_epochs, s) = if minibatch_size >= min {
        recursively_train::<M>(
            model,
            examples,
            example_truncation,
            n_updates,
            (minibatch_size * scale.0) / scale.1,
            min,
            scale,
        )
    } else {
        (model, 0, 0)
    };

    let (model, n) = model.train(&examples, example_truncation, n_updates, minibatch_size);
    (model, n_epochs + 1, s + n)
}

fn bench_model<M: Model<InputType, N_CLASSES> + Descend<InputType, N_CLASSES> + Sync>(
    examples: &Vec<(InputType, usize)>,
    test_examples: &Vec<(InputType, usize)>,
    example_truncation: usize,
    n_updates: usize,
    min: usize,
    scale: (usize, usize),
) {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let model = <M as Model<InputType, N_CLASSES>>::rand(&mut rng);

    let start = Instant::now();
    let (model, n_epochs, n_samples) = recursively_train::<M>(
        model,
        examples,
        example_truncation,
        n_updates,
        examples.len(),
        min,
        scale,
    );
    let sum_loss: u64 = examples
        .par_iter()
        .map(|(image, class)| model.loss(image, *class))
        .sum();
    println!(
        "| {} | {} | {} | {}/{} | {} | {} | {:.3}% | {:.3}% | {} |",
        n_epochs,
        n_samples,
        min,
        scale.0,
        scale.1,
        n_updates,
        example_truncation,
        model.avg_acc(&examples) * 100f64,
        model.avg_acc(&test_examples) * 100f64,
        start.elapsed().as_secs()
    );
}

fn print_header() {
    println!("| n epochs | n samples | min | scale | updates | truncation | train acc | test acc | train time |");
    println!("| - | - | - | - | - | - | - | - | - |");
}

fn bench_arch<M: Model<InputType, N_CLASSES> + Descend<InputType, N_CLASSES> + Sync>(
    examples: &Vec<(InputType, usize)>,
    test_examples: &Vec<(InputType, usize)>,
    trunc: usize,
    updates: usize,
    min: usize,
    scale: (usize, usize),
) {
    println!("\n#### min");
    print_header();
    for &min in &[20, 50, 70, 100, 200, 500, 1000] {
        bench_model::<M>(&examples, &test_examples, trunc, updates, min, scale);
    }
    println!("\n#### n updates");
    print_header();
    for &updates in &[1, 3, 7, 10, 20] {
        bench_model::<M>(&examples, &test_examples, trunc, updates, min, scale);
    }
    println!("\n#### scale");
    print_header();
    for &scale in &[(1, 4), (1, 3), (1, 2), (2, 3), (3, 4)] {
        bench_model::<M>(&examples, &test_examples, trunc, updates, min, scale);
    }
    println!("\n#### truncation");
    print_header();
    for &trunc in &[100, 200, 500, 1000, 2_000, 5_000] {
        bench_model::<M>(&examples, &test_examples, trunc, updates, min, scale);
    }
}

macro_rules! h1_model {
    ($w:ty, $h:ty) => {
        FC<InputShape, $w, $h, FcMSE<$h, $w, N_CLASSES>, N_CLASSES>
    }
}

macro_rules! h2_model {
    ($w:ty, $h:ty) => {
        FC<InputShape, $w, $h, FC<$h, $w, $h, FcMSE<$h, $w, N_CLASSES>, N_CLASSES>, N_CLASSES>
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(32)
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

    let images =
        mnist::load_images_bitpacked_u32(&base_path.join("train-images-idx3-ubyte"), N_EXAMPLES);
    let labels = mnist::load_labels(&base_path.join("train-labels-idx1-ubyte"), N_EXAMPLES);
    let examples: Vec<(_, usize)> = images
        .iter()
        .zip(labels.iter())
        .map(|(image, class)| (*image, *class))
        .collect();

    let test_images =
        mnist::load_images_bitpacked_u32(&base_path.join("train-images-idx3-ubyte"), 10_000);
    let test_labels = mnist::load_labels(&base_path.join("train-labels-idx1-ubyte"), 10_000);
    let test_examples: Vec<(_, usize)> = test_images
        .iter()
        .zip(test_labels.iter())
        .map(|(image, class)| (*image, *class))
        .collect();

    println!("### zero hidden bit");
    bench_arch::<FcMSE<InputShape, bool, N_CLASSES>>(
        &examples,
        &test_examples,
        1000,
        7,
        200,
        (2, 3),
    );
    println!("### zero hidden trit");
    bench_arch::<FcMSE<InputShape, Option<bool>, N_CLASSES>>(
        &examples,
        &test_examples,
        1000,
        7,
        200,
        (2, 3),
    );

    println!("\n### bit 1 x 32");
    bench_arch::<h1_model!(bool, [[(); 32]; 1])>(&examples, &test_examples, 1000, 7, 200, (2, 3));
    println!("\n### trit 1 x 32");
    bench_arch::<h1_model!(Option<bool>, [[(); 32]; 1])>(
        &examples,
        &test_examples,
        1000,
        7,
        200,
        (2, 3),
    );

    println!("\n### bit 1 x 64");
    bench_arch::<h1_model!(bool, [[(); 32]; 2])>(&examples, &test_examples, 1000, 7, 200, (2, 3));
    println!("\n### trit 1 x 64");
    bench_arch::<h1_model!(Option<bool>, [[(); 32]; 2])>(
        &examples,
        &test_examples,
        1000,
        7,
        200,
        (2, 3),
    );

    println!("\n### bit 1 x 128");
    bench_arch::<h1_model!(bool, [[(); 32]; 4])>(&examples, &test_examples, 1000, 7, 200, (2, 3));
    println!("\n### trit 1 x 128");
    bench_arch::<h1_model!(Option<bool>, [[(); 32]; 4])>(
        &examples,
        &test_examples,
        1000,
        7,
        200,
        (2, 3),
    );

    println!("\n### bit 1 x 256");
    bench_arch::<h1_model!(bool, [[(); 32]; 8])>(&examples, &test_examples, 1000, 7, 200, (2, 3));
    println!("\n### trit 1 x 256");
    bench_arch::<h1_model!(Option<bool>, [[(); 32]; 8])>(
        &examples,
        &test_examples,
        1000,
        7,
        200,
        (2, 3),
    );
}
