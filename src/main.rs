extern crate time;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::SystemTime;

const BATCH_SIZE: usize = 60000;

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

    let mut weights: [[u64; 13]; 10] = [[0; 13]; 10];
    let mut new_weights: [[[u64; 64]; 13]; 10] = [[[0; 64]; 13]; 10];
    let mut goodness_deltas: [[[i32; 64]; 13]; 10] = [[[0; 64]; 13]; 10];
    for o in 0..10 {
        for i in 0..13 {
            for p in 0..64 {
                new_weights[o][i][p] = weights[o][i] ^ (1 << (p));
            }
        }
    }
    println!("init: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();
    for (s, &o) in labels.iter().enumerate() {
        for i in 0..13 {
            let nil_goodness = (weights[o][i] ^ images[s][i]).count_ones() as i32;
            for p in 0..64 {
                let new_goodness = (new_weights[o][i][p] ^ images[s][i]).count_ones() as i32;
                let delta = new_goodness - nil_goodness;
                goodness_deltas[o][i][p] += delta;
            }
        }
    }
    println!("main: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();
    for o in 0..10 {
        for i in 0..13 {
            for p in 0..64 {
                if goodness_deltas[o][i][p] > 0 {
                    weights[o][i] = weights[o][i] ^ (1 << (p));
                }
            }
        }
    }
    println!("final: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();
    let mut correct: u32 = 0;
    let mut incorrect: u32 = 0;
    for (s, image) in images.iter().enumerate() {
        let mut top_index: u32 = 0;
        let mut top_sum: u32 = 0;
        for o in 0..10 {
            let mut sum: u32 = 0;
            for i in 0..13 {
                sum += (weights[o][i] ^ image[i]).count_ones()
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

    for word in weights[0].iter().rev() {
        print!("{:064b}", word);
    }
}
