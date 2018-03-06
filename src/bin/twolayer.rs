extern crate time;
extern crate rand;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::SystemTime;

const BATCH_SIZE: usize = 1000;

fn load_labels() -> Vec<usize> {
    let path = Path::new("mnist/train-labels-idx1-ubyte");
    let mut file = File::open(&path).expect("can't open images");
    let mut header: [u8; 8] = [0; 8];
    file.read_exact(&mut header).expect("can't read header");

    let mut bytes: [u8; BATCH_SIZE] = [0; BATCH_SIZE];
    file.read_exact(&mut bytes).expect("can't read label");
    let mut labels: Vec<usize> = Vec::new();
    for byte in bytes.iter() {
        labels.push(*byte as usize);
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

fn bit_perturbations() -> [u64; 64] {
    let mut words = [0u64; 64];
    for i in 0..64 {
        words[i] = 1 << i;
    }
    return words;
}

fn accuracy13_10(&(weights1, weights2): &([[u64; 13]; HIDDEN_SIZE * 64],
                                          [[u64; HIDDEN_SIZE]; 10]),
                 inputs: &Vec<[u64; 13]>,
                 targets: &Vec<usize>)
                 -> f32 {
    let mut correct: u32 = 0;
    let mut incorrect: u32 = 0;
    for e in 0..BATCH_SIZE {
        let mut hidden_layer = [0u64; 2];
        for o in 0..HIDDEN_SIZE * 64 {
            let mut sum = 0;
            for i in 0..13 {
                sum += (weights1[o][i] ^ inputs[e][i]).count_ones();
            }
            let word_index = o / 64;
            hidden_layer[word_index] = hidden_layer[word_index] | (((sum > 416) as u64) << o % 64);
            // println!("{:?}", sum);
            // println!("{:064b}", (((sum > 64) as u64) << o % 64));
        }
        let mut top_index: u32 = 0;
        let mut top_sum: u32 = 0;
        for o in 0..10 {
            let mut sum: u32 = 0;
            for i in 0..HIDDEN_SIZE {
                sum += (weights2[o][i] ^ hidden_layer[i]).count_ones();
            }
            if sum > top_sum {
                top_index = o as u32;
                top_sum = sum;
            }
        }
        if top_index == targets[e] as u32 {
            correct += 1;
        } else {
            incorrect += 1;
        }
    }
    return correct as f32 / (correct + incorrect) as f32;
}


fn goodness13_10(&(weights1, weights2): &([[u64; 13]; HIDDEN_SIZE * 64],
                                          [[u64; HIDDEN_SIZE]; 10]),
                 inputs: &Vec<[u64; 13]>,
                 targets: &Vec<usize>)
                 -> i64 {
    let mut goodness = 0i64;
    for e in 0..BATCH_SIZE {
        let mut hidden_layer = [0u64; 2];
        for o in 0..HIDDEN_SIZE * 64 {
            let mut sum = 0;
            for i in 0..13 {
                sum += (weights1[o][i] ^ inputs[e][i]).count_ones();
            }
            let word_index = o / 64;
            hidden_layer[word_index] = hidden_layer[word_index] | (((sum > 416) as u64) << o % 64);
            // println!("{:?}", sum);
            // println!("{:064b}", (((sum > 64) as u64) << o % 64));
        }
        for o in 0..10 {
            let mut sum = 0i64;
            for i in 0..HIDDEN_SIZE {
                sum += (weights2[o][i] ^ hidden_layer[i]).count_ones() as i64;
            }
            if o == targets[e] {
                // println!("+sum: {:?}", sum);
                goodness = goodness + sum;
            } else {
                // println!("-sum: {:?}", sum);
                //goodness = goodness - sum;
            }
        }
    }
    return goodness;
}

const HIDDEN_SIZE: usize = 2;

fn main() {
    let images = load_images();
    let labels = load_labels();
    let perturbations = bit_perturbations();

    let mut params = ([[0u64; 13]; HIDDEN_SIZE * 64], [[0u64; HIDDEN_SIZE]; 10]);
    for o in 0..HIDDEN_SIZE * 64 {
        for i in 0..13 {
            params.0[o][i] = rand::random::<u64>();
        }
    }
    for o in 0..10 {
        for i in 0..HIDDEN_SIZE {
            params.1[o][i] = rand::random::<u64>();
        }
    }
    println!("init goodness: {:?}",
             goodness13_10(&params, &images, &labels));
    for iter in 0..30 {
        let start = SystemTime::now();
        for o in 0..HIDDEN_SIZE * 64 {
            // for the outer dimention of the weights,
            for i in 0..13 {
                // and the inner dimention,
                // calculate the goodness of no change,
                let nil_goodness = goodness13_10(&params, &images, &labels);
                for p in 0..64 {
                    // println!("   {:064b}", params.0[o][i]);
                    params.0[o][i] = params.0[o][i] ^ perturbations[p];
                    // println!("vs {:064b}", params.0[o][i]);
                    let new_goodness = goodness13_10(&params, &images, &labels);
                    if !(new_goodness > nil_goodness) {
                        // revert
                        params.0[o][i] = params.0[o][i] ^ perturbations[p];
                        // println!("reverting {:?} {:?}", nil_goodness, new_goodness);
                    } else {
                        // println!("keeping {:?} {:?}", nil_goodness, new_goodness);
                    }
                }
            }
        }
        println!("layer one: {:?}", start.elapsed().unwrap());
        for o in 0..10 {
            // for the outer dimention of the weights,
            for i in 0..HIDDEN_SIZE {
                // and the inner dimention,
                // calculate the goodness of no change,
                let nil_goodness = goodness13_10(&params, &images, &labels);
                for p in 0..64 {
                    params.0[o][i] = params.0[o][i] ^ perturbations[p];
                    let new_goodness = goodness13_10(&params, &images, &labels);
                    if !(new_goodness > nil_goodness) {
                        // revert
                        params.1[o][i] = params.1[o][i] ^ perturbations[p];
                    }
                }
            }
        }

        // println!("new_goodness: {:?}", start.elapsed().unwrap());
        println!("goodness at {:?} {:?}",
                 iter,
                 goodness13_10(&params, &images, &labels));
        println!("accuracy {:?}%", accuracy13_10(&params, &images, &labels) * 100f32);
    }
}
