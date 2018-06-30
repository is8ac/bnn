// 93%
use std::sync::mpsc::channel;
use std::time::SystemTime;

use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::thread;
use std::thread::sleep;

#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use rand::prelude::*;
use std::time::Duration;

const TRAINING_SIZE: usize = 60000;
const TEST_SIZE: usize = 10000;
const NTHREADS: usize = 8;
const CACHE_SIZE: usize = TRAINING_SIZE / NTHREADS;

// channel sizes (will be multiplied by 8);
const C0: usize = 1;
const C1: usize = 2;
const C2: usize = 4;
const C3: usize = 6;
const D1: usize = 40;

// Parameter sizes and max output ranges.
const K1_SIZE: usize = C0 * C1 * 3 * 3;
const K1_MAX: i32 = i32::max_value() / ((C1 * 8) * 3 * 3) as i32;

const K2_SIZE: usize = C1 * (C2 * 8) * 3 * 3;
const K2_MAX: i32 = i32::max_value() / ((C2 * 8) * 3 * 3) as i32;

const K3_SIZE: usize = C2 * (C3 * 8) * 3 * 3;
const K3_MAX: i32 = i32::max_value() / ((C3 * 8) * 5 * 5) as i32;

const D1_SIZE: usize = C3 * 5 * 5 * D1 * 8;
const D1_MAX: i32 = i32::max_value() / D1 as i32;

const D2_SIZE: usize = D1 * 10;

conv_3x3_u8_params_u32_activation_input!(conv1, 28, 28, C0, C1, K1_MAX, 0);
max_pool!(pool1, u32, 28, 28, C1 * 8);
conv_3x3_u8_params_u32_activation_output!(conv2, 14, 14, C1, C2 * 8, K2_MAX, 0);
max_pool!(pool2, u32, 14, 14, C2 * 8);
conv_3x3_u8_params_u32_activation_output!(conv3, 7, 7, C2, C3 * 8, K3_MAX, 0);
flatten3d!(flatten, u32, 7, 7, C3 * 8);
dense_ints2ints!(dense1, u32, i32, 5 * 5 * C3, D1 * 8);
max_1d!(relu1, D1 * 8, D1_MAX);
dense_ints2ints!(dense2, u32, i32, D1, 10);

struct Cache {
    images: Vec<[[[u8; 1]; 28]; 28]>,
    labels: Vec<usize>,
    s1: Vec<[[[u32; C1 * 8]; 14]; 14]>,
    s2: Vec<[[[u32; C2 * 8]; 7]; 7]>,
    s3: Vec<[u32; C3 * 8 * 5 * 5]>,
    s4: Vec<[u32; D1 * 8]>,
    actuals: Vec<[i32; 10]>,
    losses: Vec<i64>,
    correct: Vec<bool>,
    params: Parameters,
    clean: [bool; 9],
}

impl Cache {
    fn update_images(&mut self) {
        println!("I don't actualy know how to update the images yet");
        // some day it may make sens to load in the images in demand.
        self.clean[0] = true;
        panic!("don't try to invalidate the images.")
    }
    fn update_s1(&mut self) {
        if !self.clean[0] {
            self.update_images();
        }
        //println!("updating cache 1");
        for i in 0..CACHE_SIZE {
            //println!("{:?} {:?}", i, self.images.len());
            self.s1[i] = pool1(&conv1(&self.images[i], &self.params.ck1));
        }
        // s1 is now clean
        self.clean[1] = true;
    }
    fn update_s2(&mut self) {
        // if s1 is not clean, we must update the it before we can calculate s2.
        if !self.clean[1] {
            self.update_s1();
        }
        //println!("updating cache 2");
        for i in 0..CACHE_SIZE {
            self.s2[i] = pool2(&conv2(&self.s1[i], &self.params.ck2));
        }
        // s2 is now clean
        self.clean[2] = true;
    }
    fn update_s3(&mut self) {
        if !self.clean[2] {
            self.update_s2();
        }
        for i in 0..CACHE_SIZE {
            self.s3[i] = flatten(&conv3(&self.s2[i], &self.params.ck3));
        }
        // s3 is now clean.
        self.clean[3] = true;
    }
    fn update_s4(&mut self) {
        if !self.clean[3] {
            self.update_s3();
        }
        for i in 0..CACHE_SIZE {
            self.s4[i] = relu1(&dense1(&self.s3[i], &self.params.dense1));
        }
        // s4 is now clean.
        self.clean[4] = true;
    }
    fn update_actuals(&mut self) {
        if !self.clean[4] {
            self.update_s4();
        }
        //println!("updating actuals cache ");
        for i in 0..CACHE_SIZE {
            self.actuals[i] = dense2(&self.s4[i], &self.params.dense2);
        }
        // s3 is now clean.
        self.clean[5] = true;
    }
    fn update_losses(&mut self) {
        if !self.clean[5] {
            self.update_actuals();
        }
        for i in 0..CACHE_SIZE {
            let target = self.actuals[i][self.labels[i]];
            self.losses[i] = self.actuals[i].iter().map(|o| ((o - target).max(0) + 9) as i64).sum();
        }
        self.clean[6] = true;
    }
    fn avg_loss(&mut self) -> f64 {
        if !self.clean[6] {
            self.update_losses();
        }
        let sum: i64 = self.losses.iter().sum();
        sum as f64 / CACHE_SIZE as f64
    }
    fn update_correct(&mut self) {
        if !self.clean[5] {
            self.update_actuals();
        }
        for e in 0..CACHE_SIZE {
            let mut index: usize = 0;
            let mut max = 0i32;
            for i in 0..10 {
                if self.actuals[e][i] > max {
                    max = self.actuals[e][i];
                    index = i;
                }
            }
            self.correct[e] = index == self.labels[e];
        }
        self.clean[7] = true;
    }
    fn avg_accuracy(&mut self) -> f64 {
        if !self.clean[6] {
            self.update_correct();
        }
        let sum: u64 = self.correct.iter().map(|&c| c as u64).sum();
        sum as f64 / CACHE_SIZE as f64
    }
    fn mutate(&mut self, update: Update) {
        // invalidate the cache
        for i in update.layer..9 - 1 {
            self.clean[i] = false;
        }
        self.params.mutate(update.layer, update.bit)
    }
    fn new(images: Vec<[[[u8; 1]; 28]; 28]>, labels: Vec<usize>, params: Parameters) -> Cache {
        Cache {
            images: images,
            s1: vec![[[[0u32; C1 * 8]; 14]; 14]; CACHE_SIZE],
            s2: vec![[[[0u32; C2 * 8]; 7]; 7]; CACHE_SIZE],
            s3: vec![[0u32; C3 * 8 * 5 * 5]; CACHE_SIZE],
            s4: vec![[0u32; D1 * 8]; CACHE_SIZE],
            actuals: vec![[0i32; 10]; CACHE_SIZE],
            losses: vec![0i64; CACHE_SIZE],
            correct: vec![false; CACHE_SIZE],
            labels: labels,
            params: params,
            clean: [true, false, false, false, false, false, false, false, true], // the images and labels start out clean, but the other caches do not.
        }
    }
}

#[derive(Clone, Copy)]
struct Update {
    layer: usize,
    bit: usize,
}

#[derive(Clone, Copy)]
struct Parameters {
    ck1: [u8; K1_SIZE],
    ck2: [u8; K2_SIZE],
    ck3: [u8; K3_SIZE],
    dense1: [u8; D1_SIZE],
    dense2: [u8; D2_SIZE],
}

macro_rules! flip_bit {
    ($array:expr, $index:expr) => {
        $array[$index / 8] = $array[$index / 8] ^ 0b1u8 << ($index % 8);
    };
}

impl Parameters {
    fn mutate(&mut self, layer: usize, bit_index: usize) {
        if layer == 1 {
            flip_bit!(self.ck1, bit_index);
        } else if layer == 2 {
            flip_bit!(self.ck2, bit_index);
        } else if layer == 3 {
            flip_bit!(self.ck3, bit_index);
        } else if layer == 4 {
            flip_bit!(self.dense1, bit_index);
        } else if layer == 5 {
            flip_bit!(self.dense2, bit_index);
        } else {
            panic!("bad layer ID");
        }
    }
    fn new() -> Parameters {
        Parameters {
            ck1: random_byte_array!(K1_SIZE)(),
            ck2: random_byte_array!(K2_SIZE)(),
            ck3: random_byte_array!(K3_SIZE)(),
            dense1: random_byte_array!(D1_SIZE)(),
            dense2: random_byte_array!(D2_SIZE)(),
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
        for &i in self.dense2.iter() {
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
            dense2: read_array!(D2_SIZE)(&mut data),
        }
    }
}

fn model(image: &[[[u8; 1]; 28]; 28], params: &Parameters) -> [i32; 10] {
    let s1 = conv1(&image, &params.ck1);
    let pooled1 = pool1(&s1);
    let s2 = conv2(&pooled1, &params.ck2);
    let pooled2 = pool2(&s2);
    let s3 = conv3(&pooled2, &params.ck3);
    let flat = flatten(&s3);
    let dense = relu1(&dense1(&flat, &params.dense1));
    dense2(&dense, &params.dense2)
}

fn load_params() -> Parameters {
    println!("loading params");
    let path = Path::new("mnist_conv.prms");
    let mut file = File::open(&path).expect("can't open params");
    let mut data = vec![];
    file.read_to_end(&mut data).expect("can't read params");
    Parameters::read(&mut data)
}

fn write_params(params: &Parameters) {
    let mut wtr = vec![];
    params.write(&mut wtr);
    let mut file = File::create("mnist_conv.prms").unwrap();
    file.write_all(&*wtr).unwrap();
}

fn main() {
    println!("starting v0.1.4 with {:?} threads", NTHREADS);
    let mut rng = thread_rng();
    let mut images = mnist::load_images_u8_1chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let mut labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);
    //let examples: Vec<([[[u8; 1]; 28]; 28], usize)> = images.iter().zip(labels).map(|(&image, target)| (image, target)).collect();

    let test_images = mnist::load_images_u8_1chan(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);
    //let test_examples: Vec<([[[u8; 1]; 28]; 28], usize)> = test_images.iter().zip(test_labels).map(|(&image, target)| (image, target)).collect();

    let mut params = load_params();
    //let mut params = Parameters::new();

    let (loss_tx, loss_rx) = channel();
    let mut sender_chans = vec![];
    for t in 0..NTHREADS {
        let tx = loss_tx.clone();
        let (update_tx, update_rx) = channel();
        sender_chans.push(update_tx);
        // each worker needs its own shard of the training set.
        let examples_len = images.len();
        let images_shard = images.split_off(examples_len - CACHE_SIZE);
        let labels_shard = labels.split_off(examples_len - CACHE_SIZE);
        thread::spawn(move || {
            let mut cache = Cache::new(images_shard, labels_shard, params);
            loop {
                let (update, mutate, send_loss) = update_rx.recv().expect("can't receive update");
                if mutate {
                    cache.mutate(update);
                }
                if send_loss {
                    let loss = cache.avg_loss();
                    tx.send(loss).expect("can't send loss");
                }
            }
        });
    }
    let (eval_update_tx, eval_update_rx) = channel();
    thread::spawn(move || {
        let mut test_cache = Cache::new(test_images, test_labels, params);
        let mut last_save = SystemTime::now();
        loop {
            let update = eval_update_rx.recv().expect("eval thread can't receive update");
            test_cache.mutate(update);
            if (last_save + Duration::new(30, 0)) < SystemTime::now() {
                let avg_acc = test_cache.avg_accuracy();
                println!("avg acc: {:?}%", avg_acc * 100.0);
                write_params(&test_cache.params);
                last_save = SystemTime::now();
            }
        }
    });


    let layers = [0, K1_SIZE * 8, K2_SIZE * 8, K3_SIZE * 8, D1_SIZE * 8, D2_SIZE * 8];
    let train_order = [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5];

    for w in 0..NTHREADS {
        sender_chans[w]
            .send((Update { layer: 0, bit: 0 }, false, true))
            .expect("can't send update")
    }
    let mut sum_loss = 0f64;
    for w in 0..NTHREADS {
        sum_loss += loss_rx.recv().expect("can't receive loss");
    }
    let mut nil_loss = sum_loss / NTHREADS as f64;
    println!("nil loss: {:?}", nil_loss);
    loop {
        for &l in train_order.iter() {
            println!("begining layer {:?} with {:?} bits", l, layers[l]);
            for i in 0..100 {
                let b = rng.gen_range(0, layers[l]);
                let update = Update { layer: l, bit: b };
                for w in 0..NTHREADS {
                    sender_chans[w].send((update.clone(), true, true)).expect("can't send update")
                }
                let mut sum_loss = 0f64;
                for w in 0..NTHREADS {
                    sum_loss += loss_rx.recv().expect("can't receive loss");
                }
                let new_loss = sum_loss / NTHREADS as f64;
                if new_loss <= nil_loss {
                    // update the eval worker.
                    eval_update_tx.send(update);
                    nil_loss = new_loss;
                    println!("{:?} {:?}/{:?} keeping with loss: {:?}", l, i, layers[l], new_loss);
                } else {
                    println!("reverting", );
                    // revert
                    for w in 0..NTHREADS {
                        sender_chans[w].send((update, true, false)).expect("can't send update")
                    }
                }
            }
        }
    }
}
