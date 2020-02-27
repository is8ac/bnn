use bitnn::bits::{BitMap, BitWord};
use bitnn::count::ElementwiseAdd;
use bitnn::datasets::cifar;
use bitnn::descend::{DescendFloat, Model, OneHiddenLayerConvPooledModel};
use bitnn::float::{FFFVMMtanh, FFFVMM};
use bitnn::image2d::{Conv2D, PixelFold, PixelMap, StaticImage};
use bitnn::layer::{FullFloatConvLayer, FullFloatConvLayerParams};
use bitnn::shape::{Element, Map};
use bitnn::unary::to_10;
use rand::Rng;
use rand::SeedableRng;
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;

const N_EXAMPLES: usize = 800;

fn float_channel_unary(input: [u8; 3]) -> [f32; 32] {
    (to_10(input[0]) | (to_10(input[1]) << 10) | (to_10(input[2]) << 20)).bit_map(|sign| {
        if sign {
            -1f32
        } else {
            1f32
        }
    })
}

type HiddenShape = [(); 32];

fn main() {
    let cifar_base_path = Path::new("../cifar-10-batches-bin");
    //let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let float_examples: Vec<_> = int_examples_32
        .par_iter()
        .map(|(image, class)| (image.pixel_map(|&p| float_channel_unary(p)), *class))
        .collect();

    let hyper_params = FullFloatConvLayerParams {
        seed: 0u64,
        n_workers: 8,
        n_iters: 2000,
        noise_sdev: 0.06,
        sdev_decay_rate: 0.998,
    };
    dbg!(&hyper_params);

    let (named_data, model) = <[[[f32; 32]; 3]; 3] as FullFloatConvLayer<
        StaticImage<[f32; 32], 32, 32>,
        [[(); 3]; 3],
        HiddenShape,
        StaticImage<[f32; 32], 32, 32>,
        [(); 10],
    >>::gen(
        &float_examples,
        &Path::new("params/float_unary"),
        &hyper_params,
    );

    let (examples, dataset_name) = (0..20).fold(named_data, |(examples, dataset_name), i| {
        dbg!(i);
        let (result, model) =
            <[[<f32 as Element<HiddenShape>>::Array; 3]; 3] as FullFloatConvLayer<
                StaticImage<<f32 as Element<HiddenShape>>::Array, 32, 32>,
                [[(); 3]; 3],
                HiddenShape,
                StaticImage<[f32; 32], 32, 32>,
                [(); 10],
            >>::gen(&examples, &dataset_name, &hyper_params);

        let n_correct: u64 = result
            .0
            .par_iter()
            .map(|(image, class)| {
                let pooled =
                    <StaticImage<<f32 as Element<HiddenShape>>::Array, 32, 32> as PixelFold<
                        (usize, <f32 as Element<HiddenShape>>::Array),
                        [[(); 3]; 3],
                    >>::pixel_fold(
                        &image,
                        (0usize, <f32 as Element<HiddenShape>>::Array::default()),
                        |mut acc, pixel| {
                            acc.0 += 1;
                            acc.1.elementwise_add(pixel);
                            acc
                        },
                    );
                let n = pooled.0 as f32;
                let avgs = <HiddenShape as Map<f32, f32>>::map(&pooled.1, |x| x / n);
                let acts = model.1.fffvmm(&avgs);
                let max_act = acts
                    .iter()
                    .enumerate()
                    .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                (max_act == *class) as u64
            })
            .sum();

        dbg!(n_correct);
        dbg!(n_correct as f64 / N_EXAMPLES as f64);
        let acc = n_correct as f64 / N_EXAMPLES as f64;
        dbg!(acc);

        result
    });
}
