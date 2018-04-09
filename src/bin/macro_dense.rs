#[macro_use]
extern crate bitnn;
extern crate rand;

use bitnn::datasets::mnist;
use std::time::SystemTime;

//#[derive(PartialEq, Eq, Hash)]

const TRAINING_SIZE: usize = 6000;
const h1: usize = 2;
const h2: usize = 1;

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
    let layer1_cached = dense_bits2bits_cached!(13, h1);
    let layer2 = dense_bits2bits!(h1, h2);
    let layer2_cached = dense_bits2bits_cached!(h1, h2);
    let layer3 = dense_bits2ints!(h2, 10);
    let layer3_cached = dense_bits2ints_cached!(h2, 10);
    let loss = int_loss!(10);

    let cached_loss = |&(input, c1, c2, c3, l): &([u64; 13], [u64; h1], [u64; h2], [u32; 10], u32),
                       params: &([[u64; 13]; h1 * 64], [[u64; h1]; h2 * 64], [[u64; h2]; 10]),
                       changed_layer: u32,
                       updated_output: usize,
                       targets: &[u32; 10]|
     -> u32 {
        let mut state1 = c1;
        if changed_layer == 1 {
            layer1_cached(&mut state1, &params.0, &input, updated_output);
            if state1[updated_output / 64] == c1[updated_output / 64] {
                return l;
            }
        }
        if changed_layer < 1 {
            layer1(&mut state1, &params.0, &input);
            if state1 == c1 {
                return l;
            }
        }
        let mut state2 = c2;
        if changed_layer == 2 {
            layer2_cached(&mut state2, &params.1, &state1, updated_output);
            if state2[updated_output / 64] == c2[updated_output / 64] {
                return l;
            }
        }
        if changed_layer < 2 {
            layer2(&mut state2, &params.1, &state1);
            if state2 == c2 {
                return l;
            }
        }
        let mut state3 = c3;
        if changed_layer == 3 {
            layer3_cached(&mut state3, &params.2, &state2, updated_output);
            if state3[updated_output / 64] == c3[updated_output / 64] {
                return l;
            }
        }
        if changed_layer < 3 {
            layer3(&mut state3, &params.2, &state2);
            if state3 == c3 {
                return l;
            }
        }
        loss(&state3, &targets)
    };

    let mut state_cache = vec![([0u64; 13], [0u64; h1], [0u64; h2], [0u32; 10], 0u32, 0usize); TRAINING_SIZE];

    for e in 0..TRAINING_SIZE {
        state_cache[e].0 = images[e];
        layer1(&mut state_cache[e].1, &params.0, &images[e]);
        let state0 = state_cache[e].1;
        layer2(&mut state_cache[e].2, &params.1, &state0);
        let state1 = state_cache[e].2;
        layer3(&mut state_cache[e].3, &params.2, &state1);
        state_cache[e].4 = loss(&state_cache[e].3, &labels[e]);
    }
    let mut sum_nil_loss: u64 = state_cache.iter().map(|&(_, _, _, _, loss)| loss as u64).sum();
    println!("sum loss: {:?}", sum_nil_loss);
    let nil_loss = sum_nil_loss as f64 / TRAINING_SIZE as f64;
    println!("avg nil loss: {:?}", nil_loss);
    println!("starting layer 3");
    for i in 0..h2 {
        for o in 0..10 {
            let start = SystemTime::now();
            let mut changed = false;
            for b in 0..64 {
                params.2[o][i] = params.2[o][i] ^ 0b1u64 << b;
                let sum_loss: u64 = images
                    .iter()
                    .zip(labels.iter())
                    .enumerate()
                    .map(|(e, (image, target))| cached_loss(&state_cache[e], &params, 3, o, target) as u64)
                    .sum();
                if sum_loss <= sum_nil_loss {
                    sum_nil_loss = sum_loss;
                    //println!("keeping");
                    changed = true;
                    let loss = sum_loss as f64 / TRAINING_SIZE as f64;
                    println!("{:?} loss: {:?}", b, loss);
                } else {
                    params.2[o][i] = params.2[o][i] ^ 0b1u64 << b; // revert
                }
            }
            println!("{:?} {:?} time: {:?}", o, i, start.elapsed().unwrap());
            if changed {
                for e in 0..TRAINING_SIZE {
                    let state1 = state_cache[e].2;
                    layer3(&mut state_cache[e].3, &params.2, &state1);
                    state_cache[e].4 = loss(&state_cache[e].3, &labels[e]);
                }
            }
        }
    }
}
