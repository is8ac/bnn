#![feature(const_generics)]
use bnn::bits::{b32, BitArray, MaskedDistance};
use bnn::datasets::mnist;
use bnn::layers::DescendFCaux;
use bnn::shape::{Element, Map, Shape};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::env;
use std::iter;
use std::path::PathBuf;

const N_EXAMPLES: usize = 60_000;

type ExampleType = [b32; 25];

fn main() {
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

    let weights: [<ExampleType as BitArray>::TritArrayType; 10] = rng.gen();

    let weights = <()>::minibatch_train(weights, &images, &labels, 50, (3, 4), 5);

    {
        let n_correct_train: u64 = images
            .iter()
            .zip(labels.iter())
            .map(|(image, &class)| {
                let acts: [u32; 10] =
                    <[(); 10] as Map<<ExampleType as BitArray>::TritArrayType, u32>>::map(
                        &weights,
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

        let n_correct_test: u64 = test_images
            .iter()
            .zip(test_labels.iter())
            .map(|(image, &class)| {
                let acts: [u32; 10] =
                    <[(); 10] as Map<<ExampleType as BitArray>::TritArrayType, u32>>::map(
                        &weights,
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
}

//train acc: 76.072%
//test acc: 77.530%
//
//real	0m6.987s
//user	3m22.683s
//sys	0m2.086s
