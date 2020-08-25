#![feature(const_generics)]
use bnn::bits::{b32, BitArray, BitMapPack, MaskedDistance};
use bnn::datasets::mnist;
use bnn::layers::{FClayer, FcAuxTrainParams};
use bnn::shape::{Map, Shape};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::env;
use std::path::PathBuf;

const N_EXAMPLES: usize = 60_000;

type ExampleType = [b32; 25];
type Hidden = [b32; 8];

fn main() {
    let aux_params = FcAuxTrainParams {
        min: 500,
        scale: (3, 4),
        k: 5,
    };
    let mut rng = Hc128Rng::seed_from_u64(0);

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

    //let weights: <<ExampleType as BitArray>::TritArrayType as Element<<Hidden as BitArray>::BitShape>>::Array = rng.gen();
    //let aux_weights: [<Hidden as BitArray>::TritArrayType; 10] = rng.gen();

    let (weights, aux_weights) = <() as FClayer<
        ExampleType,
        Hidden,
        [<Hidden as BitArray>::TritArrayType; 10],
    >>::descend_weights_minibatched(
        rng.gen(),
        rng.gen(),
        &aux_params,
        &images,
        &labels,
        5,
        200,
        (14, 15),
    );

    {
        let hidden_examples: Vec<Hidden> = images
            .par_iter()
            .map(|input| {
                <Hidden as BitMapPack<<ExampleType as BitArray>::TritArrayType>>::bit_map_pack(
                    &weights,
                    |trits| {
                        trits.masked_distance(&input)
                            > (<<ExampleType as BitArray>::BitShape as Shape>::N as u32 / 4)
                    },
                )
            })
            .collect();
        let n_correct_train: u64 = hidden_examples
            .iter()
            .zip(labels.iter())
            .map(|(image, &class)| {
                let acts: [u32; 10] =
                    <[(); 10] as Map<<Hidden as BitArray>::TritArrayType, u32>>::map(
                        &aux_weights,
                        |trits| trits.masked_distance(image),
                    );
                (acts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, act)| *act)
                    .unwrap()
                    .0
                    == class) as u64
            })
            .sum();
        println!(
            "train acc: {:.3}%",
            (n_correct_train as f64 / images.len() as f64) * 100f64
        );
    }

    {
        let test_images =
            mnist::load_images_bitpacked_u32(&base_path.join("t10k-images-idx3-ubyte"), 10_000);
        let test_labels = mnist::load_labels(&base_path.join("t10k-labels-idx1-ubyte"), 10_000);

        let test_hidden_examples: Vec<Hidden> = test_images
            .par_iter()
            .map(|input| {
                <Hidden as BitMapPack<<ExampleType as BitArray>::TritArrayType>>::bit_map_pack(
                    &weights,
                    |trits| {
                        trits.masked_distance(&input)
                            > (<<ExampleType as BitArray>::BitShape as Shape>::N as u32 / 4)
                    },
                )
            })
            .collect();

        let n_correct_test: u64 = test_hidden_examples
            .iter()
            .zip(test_labels.iter())
            .map(|(image, &class)| {
                let acts: [u32; 10] =
                    <[(); 10] as Map<<Hidden as BitArray>::TritArrayType, u32>>::map(
                        &aux_weights,
                        |trits| trits.masked_distance(image),
                    );
                (acts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, act)| *act)
                    .unwrap()
                    .0
                    == class) as u64
            })
            .sum();
        println!(
            "test acc: {:.3}%",
            (n_correct_test as f64 / test_images.len() as f64) * 100f64
        );
    }

    {
        let test_images =
            mnist::load_images_bitpacked_u32(&base_path.join("train-images-idx3-ubyte"), 10_000);
        let test_labels = mnist::load_labels(&base_path.join("train-labels-idx1-ubyte"), 10_000);

        let test_hidden_examples: Vec<Hidden> = test_images
            .par_iter()
            .map(|input| {
                <Hidden as BitMapPack<<ExampleType as BitArray>::TritArrayType>>::bit_map_pack(
                    &weights,
                    |trits| {
                        trits.masked_distance(&input)
                            > (<<ExampleType as BitArray>::BitShape as Shape>::N as u32 / 4)
                    },
                )
            })
            .collect();

        let n_correct_test: u64 = test_hidden_examples
            .iter()
            .zip(test_labels.iter())
            .map(|(image, &class)| {
                let acts: [u32; 10] =
                    <[(); 10] as Map<<Hidden as BitArray>::TritArrayType, u32>>::map(
                        &aux_weights,
                        |trits| trits.masked_distance(image),
                    );
                (acts
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, act)| *act)
                    .unwrap()
                    .0
                    == class) as u64
            })
            .sum();
        println!(
            "10,000 train acc: {:.3}%",
            (n_correct_test as f64 / test_images.len() as f64) * 100f64
        );
    }
}
