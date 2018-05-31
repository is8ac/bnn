// 93%
use std::time::SystemTime;

use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::thread::sleep;

#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use std::time::Duration;

//const MINIBATCH_SIZE: usize = 200;
const TRAINING_SIZE: usize = 10000;
const TEST_SIZE: usize = 10000;
const NIL_LOSS_SIZE: usize = TRAINING_SIZE;

// channel sizes (will be multiplied by 8);
const C0: usize = 1;
const C1: usize = 2;
const C2: usize = 3;
const C3: usize = 4;

// Parameter sizes and max output ranges.
const K1_SIZE: usize = C0 * C1 * 3 * 3;
const K1_MAX: i32 = i32::max_value() / ((C1 * 8) * 3 * 3) as i32;

const K2_SIZE: usize = C1 * (C2 * 8) * 3 * 3;
const K2_MAX: i32 = i32::max_value() / ((C2 * 8) * 3 * 3) as i32;

const K3_SIZE: usize = C2 * (C3 * 8) * 3 * 3;
const K3_MAX: i32 = i32::max_value() / ((C3 * 8) * 5 * 5) as i32;

const D1_SIZE: usize = C3 * 5 * 5 * 10;

conv_3x3_u8_params_u32_activation_input!(conv1, 28, 28, C0, C1, K1_MAX, 0);
max_pool!(pool1, u32, 28, 28, C1 * 8);
conv_3x3_u8_params_u32_activation_output!(conv2, 14, 14, C1, C2 * 8, K2_MAX, 0);
max_pool!(pool2, u32, 14, 14, C2 * 8);
conv_3x3_u8_params_u32_activation_output!(conv3, 7, 7, C2, C3 * 8, K3_MAX, 0);
flatten3d!(flatten, u32, 7, 7, C3 * 8);
dense_ints2ints!(dense1, u32, i32, 5 * 5 * C3, 10);

#[derive(Clone, Copy)]
struct Parameters {
    ck1: [u8; K1_SIZE],
    ck2: [u8; K2_SIZE],
    ck3: [u8; K3_SIZE],
    dense1: [u8; D1_SIZE],
}

impl Parameters {
    fn new() -> Parameters {
        Parameters {
            ck1: random_byte_array!(K1_SIZE)(),
            ck2: random_byte_array!(K2_SIZE)(),
            ck3: random_byte_array!(K3_SIZE)(),
            dense1: random_byte_array!(D1_SIZE)(),
        }
    }
    fn write(&self, wtr: &mut Vec<u8>) {
        for &i in self.ck1.iter() {
            wtr.push(i);
        }
        for &i in self.ck2.iter() {
            wtr.push(i);
        }
        for &i in self.ck3.iter() {
            wtr.push(i);
        }
        for &i in self.dense1.iter() {
            wtr.push(i);
        }
    }
    fn read(mut data: &mut Vec<u8>) -> Parameters {
        data.reverse();
        Parameters {
            ck1: read_array!(K1_SIZE)(&mut data),
            ck2: read_array!(K2_SIZE)(&mut data),
            ck3: read_array!(K3_SIZE)(&mut data),
            dense1: read_array!(D1_SIZE)(&mut data),
        }
    }
}

fn model_image(image: &[[[u8; 1]; 28]; 28], params: &Parameters) -> [i32; 10] {
    let s1 = conv1(&image, &params.ck1);
    let pooled1 = pool1(&s1);
    model_s2(&pooled1, params)
}

fn model_s2(input: &[[[u32; C1 * 8]; 14]; 14], params: &Parameters) -> [i32; 10] {
    let s2 = conv2(&input, &params.ck2);
    let pooled2 = pool2(&s2);
    model_s3(&pooled2, &params)
}

fn model_s3(input: &[[[u32; C2 * 8]; 7]; 7], params: &Parameters) -> [i32; 10] {
    let s3 = conv3(&input, &params.ck3);
    let flat = flatten(&s3);
    model_d1(&flat, params)
}

fn model_d1(input: &[u32; C3 * 8 * 5 * 5], params: &Parameters) -> [i32; 10] {
    dense1(&input, &params.dense1)
}

fn output_loss(actuals: &[i32; 10], target: usize) -> i64 {
    actuals.iter().map(|o| ((o - actuals[target] + 3) as i64).max(0)).sum()
}

fn infer(image: &[[[u8; 1]; 28]; 28], params: &Parameters) -> usize {
    let output = model_image(&image, &params);
    let mut index: usize = 0;
    let mut max = 0i32;
    for i in 0..10 {
        if output[i] > max {
            max = output[i];
            index = i;
        }
    }
    index
}

fn avg_accuracy(examples: &Vec<([[[u8; 1]; 28]; 28], usize)>, params: &Parameters) -> f64 {
    let start = SystemTime::now();
    let total: u64 = examples.iter().map(|(image, label)| (infer(image, params) == *label) as u64).sum();
    println!("time per example: {:?}", start.elapsed().unwrap() / examples.len() as u32);
    total as f64 / examples.len() as f64
}

fn load_params() -> Parameters {
    let path = Path::new("mnist_conv.prms");
    let mut file = File::open(&path).expect("can't open images");
    let mut data = vec![];
    file.read_to_end(&mut data).expect("can't read params");
    Parameters::read(&mut data)
}

macro_rules! avg_loss {
    ($inputs:expr, $labels:expr, $model:expr, $params:expr) => {{
        let mut sum = 0i64;
        for e in 0..TRAINING_SIZE {
            sum += output_loss(&$model(&$inputs[e], &$params), $labels[e]);
        }
        sum as f64 / TRAINING_SIZE as f64
    }};
}

fn write_params(params: &Parameters) {
    let mut wtr = vec![];
    params.write(&mut wtr);
    let mut file = File::create("mnist_conv.prms").unwrap();
    file.write_all(&*wtr).unwrap();
}

macro_rules! optimize_layer {
    ($params:expr, $layer:expr, $model:expr, $num_bytes:expr, $inputs:expr, $targets:expr) => {
        println!("starting {:?} bytes", $num_bytes);
        let mut nil_loss = avg_loss!($inputs, $targets, $model, $params);
        for w in 0..$num_bytes {
            for b in 0..8 {
                $layer[w] = $layer[w] ^ 1u8 << b;
                let new_loss = avg_loss!($inputs, $targets, $model, $params);
                if new_loss <= nil_loss {
                    nil_loss = new_loss;
                    println!("{:?} keeping: {:?}", w, nil_loss);
                } else {
                    println!("{:?} reverting: {:?}", w, new_loss);
                    $layer[w] = $layer[w] ^ 1u8 << b;
                }
            }
        }
        println!("avg acc: {:?}%", avg_accuracy(&test_examples, &$params) * 100.0);
        write_params(&params);
    };
}

fn main() {
    let images = mnist::load_images_u8_1chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);
    //let examples: Vec<([[[u8; 1]; 28]; 28], usize)> = images.iter().zip(labels).map(|(&image, target)| (image, target)).collect();

    let test_images = mnist::load_images_u8_1chan(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);
    let test_examples: Vec<([[[u8; 1]; 28]; 28], usize)> = test_images.iter().zip(test_labels).map(|(&image, target)| (image, target)).collect();

    //let mut params = load_params();
    let mut params = Parameters::new();

    let mut nil_loss = avg_loss!(images, labels, model_image, params);
    println!("nil loss: {:?}", nil_loss);
    if true {
        println!("starting: {:?}", K1_SIZE);
        for w in 0..K1_SIZE {
            for b in 0..8 {
                params.ck1[w] = params.ck1[w] ^ 1u8 << b;
                let new_loss = avg_loss!(images, labels, model_image, params);
                if new_loss <= nil_loss {
                    nil_loss = new_loss;
                    println!("{:?} keeping: {:?}", w, nil_loss);
                } else {
                    println!("{:?} reverting: {:?}", w, new_loss);
                    params.ck1[w] = params.ck1[w] ^ 1u8 << b;
                }
            }
        }
        println!("avg acc: {:?}%", avg_accuracy(&test_examples, &params) * 100.0);
        write_params(&params);

        let mut s1 = vec![[[[0u32; C1 * 8]; 14]; 14]; TRAINING_SIZE];
        for i in 0..TRAINING_SIZE {
            s1[i] = pool1(&conv1(&images[i], &params.ck1));
        }
        println!("starting: {:?}", K2_SIZE);
        for w in 0..K2_SIZE {
            for b in 0..8 {
                params.ck2[w] = params.ck2[w] ^ 1u8 << b;
                let new_loss = avg_loss!(s1, labels, model_s2, params);
                if new_loss <= nil_loss {
                    nil_loss = new_loss;
                    println!("{:?} keeping: {:?}", w, nil_loss);
                } else {
                    println!("{:?} reverting: {:?}", w, new_loss);
                    params.ck2[w] = params.ck2[w] ^ 1u8 << b;
                }
            }
        }
        println!("avg acc: {:?}%", avg_accuracy(&test_examples, &params) * 100.0);
        write_params(&params);

        let mut s2 = vec![[[[0u32; C2 * 8]; 7]; 7]; TRAINING_SIZE];
        for i in 0..TRAINING_SIZE {
            s2[i] = pool2(&conv2(&s1[i], &params.ck2));
        }
        println!("starting: {:?}", K3_SIZE);
        for w in 0..K3_SIZE {
            for b in 0..8 {
                params.ck3[w] = params.ck3[w] ^ 1u8 << b;
                let new_loss = avg_loss!(s2, labels, model_s3, params);
                if new_loss <= nil_loss {
                    nil_loss = new_loss;
                    println!("{:?} keeping: {:?}", w, nil_loss);
                } else {
                    println!("{:?} reverting: {:?}", w, new_loss);
                    params.ck3[w] = params.ck3[w] ^ 1u8 << b;
                }
            }
        }
        println!("avg acc: {:?}%", avg_accuracy(&test_examples, &params) * 100.0);
        write_params(&params);

        let mut s3 = vec![[0u32; C3 * 8 * 5 * 5]; TRAINING_SIZE];
        for i in 0..TRAINING_SIZE {
            s3[i] = flatten(&conv3(&s2[i], &params.ck3));
        }
        println!("starting: {:?}", D1_SIZE);
        for w in 0..D1_SIZE {
            for b in 0..8 {
                params.dense1[w] = params.dense1[w] ^ 1u8 << b;
                let new_loss = avg_loss!(s3, labels, model_d1, params);
                if new_loss <= nil_loss {
                    nil_loss = new_loss;
                    println!("{:?} keeping: {:?}", w, nil_loss);
                } else {
                    println!("{:?} reverting: {:?}", w, new_loss);
                    params.dense1[w] = params.dense1[w] ^ 1u8 << b;
                }
            }
        }
        write_params(&params);
    }

    println!("avg acc: {:?}%", avg_accuracy(&test_examples, &params) * 100.0);
}
