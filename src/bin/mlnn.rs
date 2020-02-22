#![feature(const_generics)]

use bitnn::bits::{b32, BitArray, BitWord, Classify};
use bitnn::datasets::cifar;
use bitnn::image2d::{PixelMap, StaticImage};
use bitnn::layer::{ConvLayer, TrainParams};
use bitnn::shape::Element;
use bitnn::unary::{to_10, Identity};
use rayon::prelude::*;
use std::path::Path;

fn avg_acc<O: BitArray + Sync, Image: Send + Sync, const C: usize>(
    examples: &Vec<(Image, usize)>,
    aux_weights: &[<f32 as Element<O::BitShape>>::Array; C],
) -> f64
where
    [<f32 as Element<O::BitShape>>::Array; C]: Classify<Image, [(); 10]> + Sync,
    f32: Element<O::BitShape>,
{
    let n_correct: u64 = examples
        .par_iter()
        .map(|(image, class)| {
            let max_class = <[<f32 as Element<O::BitShape>>::Array; C] as Classify<
                Image,
                [(); 10],
            >>::max_class(aux_weights, &image);
            (max_class == *class) as u64
        })
        .sum();
    n_correct as f64 / examples.len() as f64
}

fn unary_32(input: &[u8; 3]) -> b32 {
    to_10(input[0]) | (to_10(input[1]) << 10) | (to_10(input[2]) << 20)
}

const N_EXAMPLES: usize = 1_000;

fn main() {
    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let unary_examples: Vec<_> = int_examples_32
        .par_iter()
        .map(|(image, class)| (image.pixel_map(|p| unary_32(p)), *class))
        .collect();

    let train_params = TrainParams {
        lloyds_seed: 0,
        k: 400,
        lloyds_iters: 7,

        weights_init_seed: 0,

        minibatch_shuff_seed: 0,
        descend_minibatch_max: 300,
        descend_minibatch_threshold: 3,

        descend_rate: 0.8,

        aux_seed: 0,
        aux_sdev: 0.3,
    };
    let (bit_examples_32, (layer_weights, aux_weights)) = <[[b32; 3]; 3] as ConvLayer<
        StaticImage<b32, 32, 32>,
        [[(); 3]; 3],
        [b32; 2],
        StaticImage<[b32; 2], 32, 32>,
        [(); 10],
    >>::gen(&unary_examples, train_params);
    let acc =
        avg_acc::<[b32; 2], StaticImage<[b32; 2], 32, 32>, 10>(&bit_examples_32, &aux_weights);
    dbg!(acc);

    let bit_examples_32 = (0..5).fold(bit_examples_32, |examples, l| {
        dbg!(l);
        let (examples, (layer_weights, aux_weights)) = <[[[b32; 2]; 3]; 3] as ConvLayer<
            StaticImage<[b32; 2], 32, 32>,
            [[(); 3]; 3],
            [b32; 2],
            StaticImage<[b32; 2], 32, 32>,
            [(); 10],
        >>::gen(&examples, train_params);
        let acc = avg_acc::<[b32; 2], StaticImage<[b32; 2], 32, 32>, 10>(&examples, &aux_weights);
        dbg!(acc);
        examples
    });
    /*
    let bitpooled_examples_16 =
        <() as BitPoolLayer<StaticImage<[[[[b32; 2]; 2]; 16]; 16]>>>::bit_pool(&bit_examples_32);
    let int_examples_16 =
        <() as AvgPoolLayer<StaticImage<[[[u8; 3]; 32]; 32]>>>::avg_pool(&int_examples_32);

    let (bit_examples_16, layer_weights, aux_weights) = <[[[b32; 3]; 3]; 3] as Layer<
        StaticImage<[[[u8; 3]; 16]; 16]>,
        [[(); 3]; 3],
        Normalize<Unary<[b32; 3]>>,
        [b32; 2],
        StaticImage<[[[b32; 2]; 16]; 16]>,
        [(); 10],
    >>::gen(&int_examples_16, train_params);
    let acc = avg_acc(&bit_examples_16, &aux_weights);
    dbg!(acc);

    let bit_examples_16 = <() as ConcatImages<
        StaticImage<[[[[b32; 2]; 2]; 16]; 16]>,
        StaticImage<[[[b32; 2]; 16]; 16]>,
        StaticImage<[[[b32; 6]; 16]; 16]>,
    >>::concat(&bitpooled_examples_16, &bit_examples_16);

    let (bit_examples_16, layer_weights, aux_weights) = <[[[b32; 6]; 3]; 3] as Layer<
        StaticImage<[[[b32; 6]; 16]; 16]>,
        [[(); 3]; 3],
        Identity,
        [b32; 4],
        StaticImage<[[[b32; 4]; 16]; 16]>,
        [(); 10],
    >>::gen(&bit_examples_16, train_params);
    let acc = avg_acc(&bit_examples_16, &aux_weights);
    dbg!(acc);

    let (bit_examples_16, layer_weights, aux_weights) = <[[[b32; 4]; 3]; 3] as Layer<
        StaticImage<[[[b32; 4]; 16]; 16]>,
        [[(); 3]; 3],
        Identity,
        [b32; 4],
        StaticImage<[[[b32; 4]; 16]; 16]>,
        [(); 10],
    >>::gen(&bit_examples_16, train_params);
    let acc = avg_acc(&bit_examples_16, &aux_weights);
    dbg!(acc);
    */
}
