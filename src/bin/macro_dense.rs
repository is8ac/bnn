#[macro_use]
extern crate bitnn;
extern crate rand;

use std::time::SystemTime;
use bitnn::datasets::mnist;

//#[derive(PartialEq, Eq, Hash)]

const TRAINING_SIZE: usize = 60000;
const h1: usize = 2;
const h2: usize = 8;


fn main() {
    let test_size = 1000;
    let onval = 128u32;

    let images = mnist::load_images_bitpacked(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels_onehot(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE, onval);
    //let test_images = mnist::load_images_bitpacked(&String::from("mnist/train-images-idx3-ubyte"), test_size);
    //let test_labels = mnist::load_labels_onehot(&String::from("mnist/train-labels-idx1-ubyte"), test_size, onval);

    let mut params = ([[0u64; 13]; h1 * 64], [[0u64; h1]; h2 * 64], [[0u64; h2]; 10]);
    for o in 0..h1 * 64 {
        for i in 0..13 {
            params.0[o][i] = rand::random::<u64>()
        }
    }
    for o in 0..h2 * 64 {
        for i in 0..h1 {
            params.1[o][i] = rand::random::<u64>()
        }
    }
    for o in 0..10 {
        for i in 0..h2 {
            params.2[o][i] = rand::random::<u64>()
        }
    }
    let layer1 = dense_bits2bits!(13, h1);
    let layer2 = dense_bits2bits!(h1, h2);
    let layer3 = dense_bits2ints!(h2, 10);
    let loss = int_loss!(10);

    let layer1_loss = |input: &[u64; 13], targets: &[u32; 10], layer_params: &[[u64; 13]; h1 * 64], cache: &(&[u64; h1], &[u64; h2], &[u32; 10], u32)| -> u32 {
        let mut state1 = [0u64; h1];
        layer1(&mut state1, layer_params, &input);
        if state1 == *cache.0 {
            return cache.3;
        }
        let mut state2 = [0u64; h2];
        layer2(&mut state2, &params.1, &state1);
        let mut state3 = [0u32; 10];
        layer3(&mut state3, &params.2, &state2);
        loss(&state3, &targets)
    };
    //let layer2_loss = |input: &[u64; 2], targets: &[u32; 10], layer_params: &[[u64; 2]; 10]| -> u32 {
    //    let mut state2 = [0u32; 10];
    //    layer2(&mut state2, &layer_params, &input);
    //    loss(&state2, &targets)
    //};

    let mut state_cache: Vec<_> = vec![(&[0u64; h1], &[0u64; h2], &[0u32; 10], 0u32); TRAINING_SIZE];

    for e in 0..TRAINING_SIZE {
         layer1(&mut state_cache[e].0, &params.0, &images[e]);
         layer2(&mut state_cache[e].1, &params.1, &state_cache[e].0);
         layer3(&mut state_cache[e].2, &params.2, &state_cache[e].1);
         state_cache[e].3 = loss(&state_cache[e].2, &labels[e]);
    }
    let sum_nil_loss: u64 = images
        .iter()
        .zip(labels.iter()).zip(state_cache.iter())
        .map(|((image, target), cache)| layer1_loss(image, target, &params.0, cache) as u64)
        .sum();
    let nil_loss = sum_nil_loss as f64 / TRAINING_SIZE as f64;
    println!("avg nil loss: {:?}", nil_loss);
    let mut scratch_params0 = params.0;
    let mut losses0 = [[0f64; 13 * 64]; h1 * 64];
    for o in 0..h1 * 64 {
        for i in 0..13 {
            let start = SystemTime::now();
            for b in 0..64 {
                scratch_params0[o][i] = scratch_params0[o][i] ^ 0b1u64 << b;
                let sum_nil_loss: u64 = images
                    .iter()
                    .zip(labels.iter()).zip(state_cache.iter())
                    .map(|((image, target), cache)| layer1_loss(image, target, &scratch_params0, cache) as u64)
                    .sum();
                losses0[o][i * b] = nil_loss - (sum_nil_loss as f64 / TRAINING_SIZE as f64);
                scratch_params0[o][i] = params.0[o][i];
                println!("{:?} {:?} {:?} delta: {:?}", o, i, b, losses0[o][i * b]);
            }
            println!("time for 64: {:?}", start.elapsed().unwrap());
        }
    }
}
