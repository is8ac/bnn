// 49 acc: 18.3%
// 21 acc: 16.1%
// (loss)
// 182: acc: 18.3%
// acc 20%
extern crate byteorder;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::io::BufReader;

#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use rand::{thread_rng, Rng};
use bitnn::layers;
use std::sync::Arc;
use std::thread;

const MINIBATCH_SIZE: usize = 2000;
const TRAINING_SIZE: usize = 60000;
const TEST_SIZE: usize = 3000;
const NTHREADS: usize = 8;
const C0: usize = 1;
const C1: usize = 5;
const C2: usize = 4;
const C3: usize = 1;

u64_3d!(Kernel1, C1 * 64, 9, C0);
i16_1d!(Thresholds1, C1 * 64);
u64_3d!(Kernel2, C2 * 64, 9, C1);
i16_1d!(Thresholds2, C2 * 64);
u64_3d!(Kernel3, C3 * 64, 9, C2);
i16_1d!(Thresholds3, C3 * 64);
u64_2d!(Dense1, 10, 5 * 5 * C3);

conv3x3!(conv1, 28, 28, C0, C1);
pool_or2x2!(pool1, 28, 28, C1);
conv3x3!(conv2, 14, 14, C1, C2);
pool_or2x2!(pool2, 14, 14, C2);
conv3x3!(conv3, 7, 7, C2, C3);
flatten3d!(flatten, 7, 7, C3);
dense_bits2ints!(dense1, 5 * 5 * C3, 10);
softmax!(i16_softmax, 10, i16);
f32_sqr_diff_loss!(sqr_diff_loss, 10);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Parameters {
    ck1: Kernel1,
    ct1: Thresholds1,
    ck2: Kernel2,
    ct2: Thresholds2,
    ck3: Kernel3,
    ct3: Thresholds3,
    dense1: Dense1,
}

fn zero() -> i16 {
    0
}

impl Parameters {
    fn new() -> Parameters {
        Parameters {
            ck1: Kernel1::new_random(),
            ct1: Thresholds1::new_const((C0 * 64 * 9 / 2) as i16),
            ck2: Kernel2::new_random(),
            ct2: Thresholds2::new_const((C1 * 64 * 9 / 2) as i16),
            ck3: Kernel3::new_random(),
            ct3: Thresholds3::new_const((C2 * 64 * 9 / 2) as i16),
            dense1: Dense1::new_random(),
        }
    }
    fn child(&self) -> Parameters {
        let rand_bits_func = layers::random_bits9;
        Parameters {
            ck1: self.ck1.child(&rand_bits_func),
            //ct1: self.ct1.child(&layers::random_int_plusminus_one),
            ct1: self.ct1.child(&zero),
            ck2: self.ck2.child(&rand_bits_func),
            //ct2: self.ct2.child(&layers::random_int_plusminus_one),
            ct2: self.ct2.child(&zero),
            ck3: self.ck3.child(&rand_bits_func),
            //ct3: self.ct3.child(&layers::random_int_plusminus_one),
            ct3: self.ct3.child(&zero),
            dense1: self.dense1.child(&rand_bits_func),
        }
    }
    fn write(&self, wtr: &mut Vec<u8>) {
        self.ck1.write(wtr);
        self.ct1.write(wtr);
        self.ck2.write(wtr);
        self.ct2.write(wtr);
        self.ck3.write(wtr);
        self.ct3.write(wtr);
        self.dense1.write(wtr);
    }
    fn read(rdr: &mut std::io::Read) -> Parameters {
        Parameters {
            ck1: Kernel1::read(rdr),
            ct1: Thresholds1::read(rdr),
            ck2: Kernel2::read(rdr),
            ct2: Thresholds2::read(rdr),
            ck3: Kernel3::read(rdr),
            ct3: Thresholds3::read(rdr),
            dense1: Dense1::read(rdr),
        }
    }
}

fn model(image: &[[[u64; 1]; 28]; 28], params: &Parameters) -> [i16; 10] {
    let s1 = conv1(&image, &params.ck1.weights, &params.ct1.thresholds);
    let pooled1 = pool1(&s1);
    let s2 = conv2(&pooled1, &params.ck2.weights, &params.ct2.thresholds);
    let pooled2 = pool2(&s2);
    let s3 = conv3(&pooled2, &params.ck3.weights, &params.ct3.thresholds);
    //println!("s3: {:?}", s3);
    let flat = flatten(&s3);
    dense1(&flat, &params.dense1.weights)
    //i16_softmax(&outputs)
}

fn loss(image: &[[[u64; 1]; 28]; 28], target: usize, params: &Parameters) -> i32 {
    let actuals = model(&image, &params);
    actuals
        .iter()
        //.inspect(|o| println!("target: {:?}, o: {:?}", actuals[target], o))
        .map(|o| ((o - actuals[target] + 7) as i32).max(0))
        //.inspect(|o| println!("l: {:?}", o))
        .sum()
}

fn infer(image: &[[[u64; 1]; 28]; 28], params: &Parameters) -> usize {
    let output = model(&image, &params);
    let mut index: usize = 0;
    let mut max = 0i16;
    for i in 0..10 {
        if output[i] > max {
            max = output[i];
            index = i;
        }
    }
    index
}

fn avg_loss(examples: Arc<Vec<([[[u64; 1]; 28]; 28], usize)>>, params: &Parameters) -> f32 {
    let total: i64 = examples.iter().map(|(image, label)| loss(image, *label, params) as i64).sum();
    //println!("total: {:?}", total);
    (total as f64 / examples.len() as f64) as f32
}

fn avg_accuracy(examples: &Vec<([[[u64; 1]; 28]; 28], usize)>, params: &Parameters) -> f32 {
    let total: u64 = examples.iter().map(|(image, label)| (infer(image, params) == *label) as u64).sum();
    total as f32 / examples.len() as f32
}

fn load_params() -> Parameters {
    let path = Path::new("mnist_conv.prms");
    let prms_file = File::open(&path).expect("can't open parameters");
    let mut buff_reader = BufReader::new(prms_file);
    Parameters::read(&mut buff_reader)
}

fn load_training_set() -> Vec<([[[u64; 1]; 28]; 28], usize)> {
    let images = mnist::load_images_64chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);
    let mut examples_vec: Vec<([[[u64; 1]; 28]; 28], usize)> = images.iter().zip(labels).map(|(&image, target)| (image, target)).collect();
    thread_rng().shuffle(&mut examples_vec);
    examples_vec
}

fn main() {
    let mut examples_vec = load_training_set();
    let mut examples = examples_vec.iter().map(|&e| e).cycle();

    let test_images = mnist::load_images_64chan(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);
    let test_examples: Vec<([[[u64; 1]; 28]; 28], usize)> = test_images.iter().zip(test_labels).map(|(&image, target)| (image, target)).collect();
    println!("v0.8, using {:?} threads", NTHREADS);
    let mut params = load_params();
    //let mut params = Parameters::new();
    for i in 0..10000 {
        let mut nil_loss = 10000f32;
        let mut children = vec![];
        let minibatch: Arc<Vec<([[[u64; 1]; 28]; 28], usize)>> = Arc::new(examples.by_ref().take(MINIBATCH_SIZE).collect());

        let parent = params.clone();
        let examples_arc = Arc::clone(&minibatch);
        children.push(thread::spawn(move || {
            let child_loss = avg_loss(examples_arc, &parent);
            (child_loss, parent, 0)
        }));
        for i in 1..NTHREADS {
            // Spin up another thread
            let child = params.child();
            let examples_arc = Arc::clone(&minibatch);
            children.push(thread::spawn(move || {
                let child_loss = avg_loss(examples_arc, &child);
                (child_loss, child, i)
            }));
        }
        for child_thread in children {
            // Wait for the thread to finish. Returns a result.
            let (child_loss, child_params, index) = child_thread.join().unwrap();
            if child_loss < nil_loss {
                println!("keeping {:?} {:?}", index, child_loss);
                nil_loss = child_loss;
                params = child_params;
            }
        }
        println!("{:?} loss: {:?}", i, nil_loss);
        if i % 10 == 0 {
            println!("acc: {:?}%", avg_accuracy(&test_examples, &params) * 100.0);
            let mut wtr = vec![];
            params.write(&mut wtr);
            let mut file = File::create("mnist_conv.prms").unwrap();
            file.write_all(&*wtr).unwrap();
        }
    }
}
