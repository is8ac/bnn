extern crate rand;
extern crate time;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::SystemTime;

const BATCH_SIZE: usize = 3;

const HIDDEN_SIZE: usize = 2;

fn load_labels() -> [usize; BATCH_SIZE] {
    let path = Path::new("mnist/train-labels-idx1-ubyte");
    let mut file = File::open(&path).expect("can't open images");
    let mut header: [u8; 8] = [0; 8];
    file.read_exact(&mut header).expect("can't read header");

    let mut bytes: [u8; BATCH_SIZE] = [0; BATCH_SIZE];
    file.read_exact(&mut bytes).expect("can't read label");
    let mut labels: [usize; BATCH_SIZE] = [0; BATCH_SIZE];
    for (i, byte) in bytes.iter().enumerate() {
        labels[i] = *byte as usize;
    }
    return labels;
}

fn load_images() -> Vec<[u64; 13]> {
    let path = Path::new("mnist/train-images-idx3-ubyte");
    let mut file = File::open(&path).expect("can't open images");
    let mut header: [u8; 16] = [0; 16];
    file.read_exact(&mut header).expect("can't read header");

    let mut images_bytes: [u8; 784] = [0; 784];

    // bitpack the image into 13 64 bit words.
    // There will be unused space in the last word, this is acceptable.
    // the bits of each words will be in revere order,
    // rev() the slice before use if you want them in the correct order.
    let mut images: Vec<[u64; 13]> = Vec::new();
    for _ in 0..BATCH_SIZE {
        file.read_exact(&mut images_bytes).expect("can't read images");
        let mut image_words: [u64; 13] = [0; 13];
        for p in 0..784 {
            let word_index = p / 64;
            image_words[word_index] = image_words[word_index] |
                                      (((images_bytes[p] > 128) as u64) << p % 64);
        }
        images.push(image_words);
    }
    return images;
}

fn main() {
    let start = SystemTime::now();
    let images = load_images();
    let labels = load_labels();
    println!("load: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();

    let mut weights1: [[u64; 13]; HIDDEN_SIZE * 64] = [[0; 13]; HIDDEN_SIZE * 64];
    let mut new_weights1: [[[u64; 64]; 13]; HIDDEN_SIZE * 64] = [[[0; 64]; 13]; HIDDEN_SIZE * 64];
    let mut goodness_deltas1: [[[i32; 64]; 13]; HIDDEN_SIZE * 64] = [[[0; 64]; 13]; HIDDEN_SIZE *
                                                                                    64];


    // outputs of the first layer for the second to read.
    let mut hidden_outputs: [[u64; HIDDEN_SIZE]; 10] = [[0; HIDDEN_SIZE]; 10];
    // inputs which the second layer expect from the first layer.
    let mut hidden_targets: [[bool; HIDDEN_SIZE * 64]; 10] = [[false; HIDDEN_SIZE * 64]; 10];

    // init first layer
    for o in 0..HIDDEN_SIZE * 64 {
        // for each of the hidden outputs,
        for i in 0..13 {
            // for each of the inputs,
            weights1[o][i] = rand::random::<u64>(); // 64 random bits
            for p in 0..64 {
                // for each permutation of the weight word,
                // xor the existing weight with the 1 bit update
                new_weights1[o][i][p] = weights1[o][i] ^ (1 << (p));
            }
        }
    }

    let mut weights2: [[u64; 2]; 10] = [[0; 2]; 10];
    let mut new_weights2: [[[u64; 64]; 2]; 10] = [[[0; 64]; 2]; 10];
    let mut goodness_deltas2: [[[i32; 64]; 2]; 10] = [[[0; 64]; 2]; 10];

    // init second layer
    for o in 0..10 {
        // for each of the final outputs,
        for i in 0..HIDDEN_SIZE {
            weights2[o][i] = rand::random::<u64>();
            // for each bit in the hidden state
            // random value for the hidden state
            hidden_targets[o][i] = rand::random::<bool>();
            // random weight
            for p in 0..64 {
                // for each permutation of the weight word,
                // xor the existing weight word with the 1 bit update.
                new_weights2[o][i][p] = weights2[o][i] ^ (1 << (p));
            }
        }
    }
    // calculate goodness deltas permutations in the first layer
    println!("init: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();
    // calculate the deltas for each permutation of each weight.
    for (s, &o) in labels.iter().enumerate() { // for each sample,
        for h in 0..HIDDEN_SIZE * 64 { // for each bit of hidden state,
            for i in 0..13 { // for each word of the input,
                // calculate the value of the weight with no changes.
                let nil_value = (weights1[h][i] ^ images[s][i]).count_ones() as i32;
                for p in 0..64 {
                    let new_value = (new_weights1[h][i][p] ^ images[s][i]).count_ones() as i32;
                    let delta = new_value - nil_value;
                    goodness_deltas1[o][i][p] += delta;
                }
            }
        }
    }
    println!("main: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();
    for o in 0..HIDDEN_SIZE * 64 { // for each bit of hidden state
        for i in 0..13 {
            for p in 0..64 {
                if goodness_deltas1[o][i][p] > 0 {
                    weights1[o][i] = weights1[o][i] ^ (1 << (p));
                }
            }
            hidden_outputs[o][i] = hidden_outputs[o][i] ^ weights1[o][i];
        }
    }
    println!("final: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();
    let mut correct: u32 = 0;
    let mut incorrect: u32 = 0;
    for (s, image) in images.iter().enumerate() {
        let mut hidden: [u64; 2] = [0; 2];
        let mut top_index: u32 = 0;
        let mut top_sum: u32 = 0;
        // Compute the first layer.
        for o in 0..HIDDEN_SIZE * 64 {
            let mut sum: u32 = 0;
            for i in 0..13 {
                sum += (weights1[o][i] ^ image[i]).count_ones()
            }
            let word_index = o / 64;
            // set a bit to the hidden state.
            hidden[word_index] = hidden[word_index] | (((sum > 400) as u64) << o % 64);
        }
        println!("{:064b}", hidden[0]);
        println!("{:064b}", hidden[1]);
        // Compute the second layer.
        for o in 0..10 {
            let mut sum: u32 = 0;
            for i in 0..2 {
                sum += (weights2[o][i] ^ hidden[i]).count_ones()
            }
            if sum > top_sum {
                top_index = o as u32;
                top_sum = sum;
            }
        }
        if top_index == labels[s] as u32 {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }
    println!("run: {:?}", start.elapsed().unwrap());
    println!("{:?}% correct",
             correct as f32 / (correct + incorrect) as f32);

    for word in weights1[0].iter().rev() {
        print!("{:064b}", word);
    }
}
