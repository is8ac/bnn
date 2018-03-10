extern crate rand;
extern crate time;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::SystemTime;

fn load_labels(path: &String, size: usize) -> Vec<usize> {
    let path = Path::new(path);
    let mut file = File::open(&path).expect("can't open images");
    let mut header: [u8; 8] = [0; 8];
    file.read_exact(&mut header).expect("can't read header");

    let mut byte: [u8; 1] = [0; 1];
    let mut labels: Vec<usize> = Vec::new();
    for _ in 0..size {
        file.read_exact(&mut byte).expect("can't read label");
        labels.push(byte[0] as usize);
    }
    return labels;
}

fn load_images(path: &String, size: usize) -> Vec<[[u64; 28]; 28]> {
    let path = Path::new(path);
    let mut file = File::open(&path).expect("can't open images");
    let mut header: [u8; 16] = [0; 16];
    file.read_exact(&mut header).expect("can't read header");

    let mut images_bytes: [u8; 784] = [0; 784];

    // bitpack the image into 13 64 bit words.
    // There will be unused space in the last word, this is acceptable.
    // the bits of each words will be in revere order,
    // rev() the slice before use if you want them in the correct order.
    let mut images: Vec<[[u64; 28]; 28]> = Vec::new();
    for _ in 0..size {
        file.read_exact(&mut images_bytes)
            .expect("can't read images");
        let mut image = [[0u64; 28]; 28];
        for p in 0..784 {
            let mut ones = 0u64;
            for i in 0..images_bytes[p] / 4 {
                ones = ones | 1 << i;
            }
            image[p / 28][p % 28] = ones;
        }
        images.push(image);
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

macro_rules! conv {
    ($x_size:expr, $y_size:expr, $strides:expr) => (
        |input: &[[u64; $x_size]; $y_size], weights: &[[u64; 9]; 64]| -> [[u64; ($x_size - 2) / $strides.0]; ($y_size - 2) / $strides.1] {
            let mut output = [[0u64; ($x_size - 2) / $strides.0]; ($y_size - 2) / $strides.1];
            for x in 0..($x_size - 2) / $strides.0 {
                for y in 0..($y_size - 2) / $strides.1 {
                    for chan in 0..64 {
                    let sum = (weights[chan][0] ^ input[x * $strides.0 + 0][y * $strides.1 + 0]).count_ones()
                            + (weights[chan][1] ^ input[x * $strides.0 + 1][y * $strides.1 + 0]).count_ones()
                            + (weights[chan][2] ^ input[x * $strides.0 + 2][y * $strides.1 + 0]).count_ones()
                            + (weights[chan][3] ^ input[x * $strides.0 + 0][y * $strides.1 + 1]).count_ones()
                            + (weights[chan][4] ^ input[x * $strides.0 + 1][y * $strides.1 + 1]).count_ones()
                            + (weights[chan][5] ^ input[x * $strides.0 + 2][y * $strides.1 + 1]).count_ones()
                            + (weights[chan][6] ^ input[x * $strides.0 + 0][y * $strides.1 + 2]).count_ones()
                            + (weights[chan][7] ^ input[x * $strides.0 + 1][y * $strides.1 + 2]).count_ones()
                            + (weights[chan][8] ^ input[x * $strides.0 + 2][y * $strides.1 + 2]).count_ones();
                        output[x][y] = output[x][y] | (((sum > 288) as u64) << chan);
                    }
                }
            }
            return output;
        }
    )
}

fn convnet(
    &(conv0, conv1, conv2, dense0): &(
        [[u64; 9]; 64],
        [[u64; 9]; 64],
        [[u64; 9]; 64],
        [[[u64; 5]; 5]; 10],
    ),
    inputs: &Vec<[[u64; 28]; 28]>,
    targets: &Vec<usize>,
) -> i64 {
    let mut goodness = 0i64;
    for e in 0..inputs.len() {
        let hidden1 = conv!(28, 28, (1, 1))(&inputs[0], &conv0);
        let hidden2 = conv!(26, 26, (2, 2))(&hidden1, &conv1);
        let hidden3 = conv!(12, 12, (2, 2))(&hidden2, &conv2);
        for o in 0..10 {
            let mut sum = 0i64;
            for x in 0..5 {
                for y in 0..5 {
                    sum += (dense0[o][x][y] ^ hidden3[x][y]).count_ones() as i64;
                }
            }
            if o == targets[e] {
                // println!("+sum: {:?}", sum);
                goodness = goodness + sum;
            } else {
                // println!("-sum: {:?}", sum);
                // goodness = goodness - sum;
            }
        }
    }
    return goodness;
}

fn main() {
    let training_size = 1000;
    let test_size = 1000;
    let images = load_images(
        &String::from("mnist/train-images-idx3-ubyte"),
        training_size,
    );
    let labels = load_labels(
        &String::from("mnist/train-labels-idx1-ubyte"),
        training_size,
    );
    let test_images = load_images(&String::from("mnist/train-images-idx3-ubyte"), test_size);
    let test_labels = load_labels(&String::from("mnist/train-labels-idx1-ubyte"), test_size);

    let perturbations = bit_perturbations();

    let mut params = ([[0u64; 9]; 64], [[0u64; 9]; 64], [[0u64; 9]; 64], [[[0u64; 5]; 5]; 10]);
    for o in 0..64 {
        for i in 0..9 {
            params.0[o][i] = rand::random::<u64>();
            params.1[o][i] = rand::random::<u64>();
            params.2[o][i] = rand::random::<u64>();
        }
    }
    for o in 0..10 {
        for x in 0..5 {
            for y in 0..5 {
                params.3[o][x][y] = rand::random::<u64>()
            }
        }
    }
    let mut nil_goodness = convnet(&params, &images, &labels);
    println!("nil_goodness {:?}", nil_goodness);
    for iter in 0..30 {
        let iter_start = SystemTime::now();
        for o in 0..64 {
            // for the outer dimention of the weights,
            for i in 0..9 {
                // and the inner dimention,
                // calculate the goodness of no change,
                let word_start = SystemTime::now();
                let mut updateed = false;
                let old_word = params.0[o][i];
                for p in 0..64 {
                    params.0[o][i] = params.0[o][i] ^ perturbations[p];
                    // println!("vs {:064b}", params.0[o][i]);
                    let new_goodness = convnet(&params, &images, &labels);
                    println!("delta: {:?}", new_goodness - nil_goodness);
                    if new_goodness > nil_goodness {
                        println!("{:?}", new_goodness);
                        // keep the change
                        nil_goodness = new_goodness;
                        updateed = true;
                    } else {
                        // revert the change.
                        params.0[o][i] = params.0[o][i] ^ perturbations[p];
                    }
                }
                if updateed {
                    println!(
                        "word {:?} {:?}: {:?} bits updated. {:?}",
                        o,
                        i,
                        (old_word ^ params.0[o][i]).count_ones(),
                        word_start.elapsed().unwrap()
                    );
                }
            }
        }
        println!("layer one: {:?}", iter_start.elapsed().unwrap());
        for o in 0..64 {
            // for the outer dimension of the weights,
            for i in 0..9 {
                // and the inner dimension,
                let word_start = SystemTime::now();
                let mut updateed = false;
                let old_word = params.0[o][i];
                for p in 0..64 {
                    params.1[o][i] = params.1[o][i] ^ perturbations[p];
                    let new_goodness = convnet(&params, &images, &labels);
                    if new_goodness > nil_goodness {
                        // keep the change
                        nil_goodness = new_goodness;
                        updateed = true;
                    } else {
                        // revert the change.
                        params.1[o][i] = params.1[o][i] ^ perturbations[p];
                    }
                }
                if updateed {
                    println!(
                        "word {:?} {:?}: {:?} bits updated. {:?}",
                        o,
                        i,
                        (old_word ^ params.1[o][i]).count_ones(),
                        word_start.elapsed().unwrap()
                    );
                }
            }
        }
    }
}
