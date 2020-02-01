#![feature(const_generics)]

use bitnn::bits::{b32, BitArray, BitWord, Classify};
use bitnn::datasets::cifar;
use bitnn::image2d::StaticImage;
use bitnn::layer::{Layer, TrainParams};
use bitnn::shape::Element;
use bitnn::unary::{Identity, Unary};
use rayon::prelude::*;
use std::path::Path;

fn avg_acc<O: BitArray + Sync + BitWord, Image: Send + Sync, const C: usize>(
    examples: &Vec<(Image, usize)>,
    aux_weights: &[(<f32 as Element<O::BitShape>>::Array, f32); C],
) -> f64
where
    [(<f32 as Element<O::BitShape>>::Array, f32); C]: Classify<Image, (), [(); 10]> + Sync,
    f32: Element<O::BitShape>,
{
    let n_correct: u64 = examples
        .par_iter()
        .map(|(image, class)| {
            let max_class = <[(<f32 as Element<O::BitShape>>::Array, f32); C] as Classify<
                Image,
                (),
                [(); 10],
            >>::max_class(aux_weights, &image);
            (max_class == *class) as u64
        })
        .sum();
    n_correct as f64 / examples.len() as f64
}

const N_EXAMPLES: usize = 5_000;

fn main() {
    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);

    let train_params = TrainParams {
        lloyds_seed: 0,
        k: 1000,
        lloyds_iters: 5,
        weights_seed: 0,
        noise_sdev: 0.002,
        decend_iters: 10_000,
    };
    let (bit_examples_32, (layer_weights, aux_weights)) =
        <[[[b32; 3]; 3]; 3] as Layer<
            StaticImage<[[[u8; 3]; 32]; 32]>,
            [[(); 3]; 3],
            //Normalize<Unary<[b32; 3]>>,
            Unary<[[[b32; 3]; 3]; 3]>,
            [b32; 2],
            StaticImage<[[[b32; 2]; 32]; 32]>,
            [(); 10],
        >>::gen(&int_examples_32, train_params);
    let acc =
        avg_acc::<[b32; 2], StaticImage<[[[b32; 2]; 32]; 32]>, 10>(&bit_examples_32, &aux_weights);
    dbg!(acc);

    let bit_examples_32 = (0..5).fold(bit_examples_32, |examples, l| {
        dbg!(l);
        let (examples, (layer_weights, aux_weights)) = <[[[b32; 2]; 3]; 3] as Layer<
            StaticImage<[[[b32; 2]; 32]; 32]>,
            [[(); 3]; 3],
            Identity,
            [b32; 2],
            StaticImage<[[[b32; 2]; 32]; 32]>,
            [(); 10],
        >>::gen(&examples, train_params);
        //println!("{:?}", examples[6].0);
        let acc =
            avg_acc::<[b32; 2], StaticImage<[[[b32; 2]; 32]; 32]>, 10>(&examples, &aux_weights);
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
