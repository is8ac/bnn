extern crate rand;
extern crate time;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::SystemTime;

const BATCH_SIZE: usize = 15625;

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

fn cheap_avg_96(words: &[u64; 96]) -> u64 {
    let mut l1 = [0u64; 32];
    for i in 0..32 {
        l1[i] = (words[i * 3 + 0] | words[i * 3 + 1]) & words[i * 3 + 2]
    }
    let mut l2 = [0u64; 16];
    for i in 0..16 {
        l2[i] = l1[i * 2 + 0] & l1[i * 2 + 1]
    }
    let mut l3 = [0u64; 8];
    for i in 0..8 {
        l3[i] = l2[i * 2 + 0] | l2[i * 2 + 1]
    }
    let mut l4 = [0u64; 4];
    for i in 0..4 {
        l4[i] = l3[i * 2 + 0] & l3[i * 2 + 1]
    }
    let avg = (l4[0] | l4[1]) & (l4[2] | l4[3]);
    return avg;
}

fn cheap_avg_192(words: &[u64; 192]) -> u64 {
    let mut l0 = [0u64; 64];
    for i in 0..64 {
        l0[i] = (words[i * 3 + 0] | words[i * 3 + 1]) & words[i * 3 + 2]
    }
    let mut l1 = [0u64; 32];
    for i in 0..32 {
        l1[i] = l0[i * 2 + 0] | l0[i * 2 + 1]
    }
    let mut l2 = [0u64; 16];
    for i in 0..16 {
        l2[i] = l1[i * 2 + 0] & l1[i * 2 + 1]
    }
    let mut l3 = [0u64; 8];
    for i in 0..8 {
        l3[i] = l2[i * 2 + 0] | l2[i * 2 + 1]
    }
    let mut l4 = [0u64; 4];
    for i in 0..4 {
        l4[i] = l3[i * 2 + 0] & l3[i * 2 + 1]
    }
    let avg = (l4[0] | l4[1]) & (l4[2] | l4[3]);
    return avg;
}

fn avg5(v0: u64, v1: u64, v2: u64, v3: u64, v4: u64) -> u64 {
    let d01 = v0 & v1;
    let d02 = v0 & v2;
    let d03 = v0 & v3;
    let d12 = v1 & v2;
    let d13 = v1 & v3;
    let d23 = v2 & v3;

    let t012 = d01 & v2;
    let t013 = d01 & v3;
    let t014 = d01 & v4;
    let t023 = d02 & v3;
    let t024 = d02 & v4;
    let t034 = d03 & v4;
    let t123 = d12 & v3;
    let t124 = d12 & v4;
    let t134 = d13 & v4;
    let t234 = d23 & v4;

    let avg = t012 | t013 | t014 | t023 | t024 | t034 | t123 | t124 | t134 | t234;
    return avg;
}

fn avg3125(words: &[u64; 3125]) -> u64 {
    let mut l4 = [0u64; 625];
    for i in 0..625 {
        l4[i] = avg5(words[i * 5 + 0],
                     words[i * 5 + 1],
                     words[i * 5 + 2],
                     words[i * 5 + 3],
                     words[i * 5 + 4])
    }
    let mut l3 = [0u64; 125];
    for i in 0..125 {
        l3[i] = avg5(l4[i * 5 + 0],
                     l4[i * 5 + 1],
                     l4[i * 5 + 2],
                     l4[i * 5 + 3],
                     l4[i * 5 + 4])
    }
    let mut l2 = [0u64; 25];
    for i in 0..25 {
        l2[i] = avg5(l3[i * 5 + 0],
                     l3[i * 5 + 1],
                     l3[i * 5 + 2],
                     l3[i * 5 + 3],
                     l3[i * 5 + 4])
    }
    let l1 = avg5(l2[0], l2[1], l2[2], l2[3], l2[4]);
    return l1;
}

fn avg15625(words: &[u64; 15625]) -> u64 {
    let mut l5 = [0u64; 3125];
    for i in 0..3125 {
        l5[i] = avg5(words[i * 5 + 0],
                     words[i * 5 + 1],
                     words[i * 5 + 2],
                     words[i * 5 + 3],
                     words[i * 5 + 4])
    }
    let mut l4 = [0u64; 625];
    for i in 0..625 {
        l4[i] = avg5(l5[i * 5 + 0],
                     l5[i * 5 + 1],
                     l5[i * 5 + 2],
                     l5[i * 5 + 3],
                     l5[i * 5 + 4])
    }
    let mut l3 = [0u64; 125];
    for i in 0..125 {
        l3[i] = avg5(l4[i * 5 + 0],
                     l4[i * 5 + 1],
                     l4[i * 5 + 2],
                     l4[i * 5 + 3],
                     l4[i * 5 + 4])
    }
    let mut l2 = [0u64; 25];
    for i in 0..25 {
        l2[i] = avg5(l3[i * 5 + 0],
                     l3[i * 5 + 1],
                     l3[i * 5 + 2],
                     l3[i * 5 + 3],
                     l3[i * 5 + 4])
    }
    let l1 = avg5(l2[0], l2[1], l2[2], l2[3], l2[4]);
    return l1;
}



fn main() {
    let start = SystemTime::now();
    let images = load_images();
    let labels = load_labels();
    println!("load: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();

    let mut weights: [[u64; 13]; 10] = rand::random::<[[u64; 13]; 10]>();
    // init the weights.
    for o in 0..10 {
        for i in 0..13 {
            weights[o][i] = rand::random::<u64>();
        }
    }
    println!("init: {:?}", start.elapsed().unwrap());
    let start = SystemTime::now();
    for i in 0..13 {
        for o in 0..10 {
            let mut new_weights = [0u64; BATCH_SIZE];
            for (s, &to) in labels.iter().enumerate() {
                // if this is the correct output, we want it to be large.
                if o == to {
                    new_weights[s] = !images[s][i];
                } else { // if not, make it small.
                    new_weights[s] = images[s][i];
                }
            }
            weights[o][i] = avg15625(&new_weights);
        }
    }
    println!("main: {:?}", start.elapsed().unwrap());

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

    for word in weights[7].iter().rev() {
        print!("{:064b}", word);
    }
}
