use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::SystemTime;

#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use rand::prelude::*;

// channel sizes (all but C0 will be multiplied by 8);
const C0: usize = 1;
const C1: usize = 8;
const C2: usize = 16;
const C3: usize = 16;
const D1: usize = 64;

// Parameter sizes and max output ranges.
const K1_SIZE: usize = C0 * C1 * 3 * 3;
const K1_MAX: i32 = i32::max_value() / ((C1 * 8) * 3 * 3) as i32;

const K2_SIZE: usize = C1 * (C2 * 8) * 3 * 3;
const K2_MAX: i32 = i32::max_value() / ((C2 * 8) * 3 * 3) as i32;

const K3_SIZE: usize = C2 * (C3 * 8) * 3 * 3;
const K3_MAX: i32 = i32::max_value() / ((C2 * 8) * 3 * 3) as i32;

const D1_SIZE: usize = C3 * 3 * 3 * D1 * 8;
const D1_MAX: i32 = i32::max_value() / D1 as i32;

const D2_SIZE: usize = D1 * 10;

conv!(conv1, i32, 28, 28, C0, u8, C1, u32, 3, 3, K1_MAX, 0);
max_pool!(pool1, u32, 28, 28, C1 * 8);

conv!(conv2, i32, 14, 14, C1 * 8, u32, C2, u32, 3, 3, K2_MAX, 0);
max_pool!(pool2, u32, 14, 14, C2 * 8);

conv!(conv3, i32, 7, 7, C2 * 8, u32, C3, u32, 3, 3, K3_MAX, 0);
max_pool!(pool3, u32, 7, 7, C3 * 8);

flatten3d!(flatten, u32, 3, 3, C3 * 8);

dense_ints2ints!(dense1, u32, i32, 3 * 3 * C3, D1 * 8);
max_1d!(relu1, D1 * 8, D1_MAX);
dense_ints2ints!(dense2, u32, i32, D1, 10);
softmax_loss!(lossfn, i32, 10, 0.0000005f64);

fn is_correct(actual: &[i32; 10], target: usize) -> bool {
    for &o in actual.iter() {
        if o > actual[target] {
            return false;
        }
    }
    true
}

macro_rules! flip_bit {
    ($array:expr, $index:expr) => {
        $array[$index / 8] = $array[$index / 8] ^ 0b1u8 << ($index % 8);
    };
}

struct Cache {
    params: Box<Parameters>,
    images: Vec<[[[u8; 1]; 28]; 28]>,
    labels: Vec<usize>,
    s1: (bool, Vec<[[[u32; C1 * 8]; 14]; 14]>), // output of first conv and maxpool
    s2: (bool, Vec<[[[u32; C2 * 8]; 7]; 7]>),   // output of second conv and maxpool
    s3: (bool, Vec<[u32; C3 * 8 * 3 * 3]>),     // output of third conv and maxpool
    s4: (bool, Vec<[u32; D1 * 8]>),             // output of first FC layer
    cache_size: usize,
}

impl Cache {
    fn calc_s1(&self, i: usize) -> [[[u32; C1 * 8]; 14]; 14] {
        pool1(&conv1(&self.images[i], &self.params.ck1))
    }
    fn calc_s2(&self, i: usize) -> [[[u32; C2 * 8]; 7]; 7] {
        if self.s1.0 {
            pool2(&conv2(&self.s1.1[i], &self.params.ck2))
        } else {
            pool2(&conv2(&self.calc_s1(i), &self.params.ck2))
        }
    }
    fn calc_s3(&self, i: usize) -> [u32; C3 * 8 * 3 * 3] {
        if self.s2.0 {
            flatten(&pool3(&conv3(&self.s2.1[i], &self.params.ck3)))
        } else {
            flatten(&pool3(&conv3(&self.calc_s2(i), &self.params.ck3)))
        }
    }
    fn calc_s4(&self, i: usize) -> [u32; D1 * 8] {
        if self.s3.0 {
            relu1(&dense1(&self.s3.1[i], &self.params.d1))
        } else {
            relu1(&dense1(&self.calc_s3(i), &self.params.d1))
        }
    }
    fn calc_actual(&self, i: usize) -> [i32; 10] {
        if self.s4.0 {
            dense2(&self.s4.1[i], &self.params.d2)
        } else {
            dense2(&self.calc_s4(i), &self.params.d2)
        }
    }
    fn update_s1(&mut self) {
        for i in 0..self.cache_size {
            self.s1.1[i] = self.calc_s1(i);
        }
        self.s1.0 = true;
    }
    fn invalidate_s1(&mut self) {
        self.s1.0 = false;
        self.invalidate_s2()
    }
    fn update_s2(&mut self) {
        for i in 0..self.cache_size {
            self.s2.1[i] = self.calc_s2(i);
        }
        self.s2.0 = true;
    }
    fn invalidate_s2(&mut self) {
        self.s2.0 = false;
        self.invalidate_s3()
    }
    fn update_s3(&mut self) {
        for i in 0..self.cache_size {
            self.s3.1[i] = self.calc_s3(i);
        }
        self.s3.0 = true;
    }
    fn invalidate_s3(&mut self) {
        self.s3.0 = false;
        self.invalidate_s4()
    }

    fn update_s4(&mut self) {
        for i in 0..self.cache_size {
            self.s4.1[i] = self.calc_s4(i);
        }
        self.s4.0 = true;
    }
    fn invalidate_s4(&mut self) {
        self.s4.0 = false;
    }
    fn avg_loss(&mut self, batch_size: usize, seed: [u8; 32]) -> f64 {
        //let start = SystemTime::now();
        let mut vec: Vec<usize> = (0..self.cache_size).collect();
        let slice: &mut [usize] = &mut vec;

        let mut rng = rand::prng::ChaChaRng::from_seed(seed);
        rng.shuffle(slice);
        //println!("rng gen time: {:?}", start.elapsed().unwrap());
        let mut sum = 0f64;

        for &e in slice[0..batch_size].iter() {
            let actual = self.calc_actual(e);
            sum += lossfn(&actual, self.labels[e]);
        }
        sum / batch_size as f64
    }
    fn avg_accuracy(&mut self, batch_size: usize) -> f64 {
        let mut sum: u64 = 0;
        for e in 0..batch_size {
            let actual = self.calc_actual(e);
            sum += is_correct(&actual, self.labels[e]) as u64;
        }
        sum as f64 / batch_size as f64
    }
    fn update_cache(&mut self, cache_index: usize) {
        if cache_index == 0 {
            // do nothing
        } else if cache_index == 1 {
            self.update_s1();
        } else if cache_index == 2 {
            self.update_s2();
        } else if cache_index == 3 {
            self.update_s3();
        } else if cache_index == 4 {
            self.update_s4();
        } else {
            panic!("bad cache index");
        }
    }
    fn mutate(&mut self, layer: usize, bit: usize) {
        self.params.mutate(layer, bit);
        if layer == 1 {
            self.invalidate_s1();
        } else if layer == 2 {
            self.invalidate_s2();
        } else if layer == 3 {
            self.invalidate_s3();
        } else if layer == 4 {
            self.invalidate_s4();
        } else if layer == 5 {
        } else {
            panic!("bad layer ID");
        }
    }
    fn new(images: Vec<[[[u8; 1]; 28]; 28]>, labels: Vec<usize>, params: Box<Parameters>) -> Cache {
        let cache_size = images.len();
        Cache {
            params: params,
            images: images,
            labels: labels,
            s1: (false, vec![[[[0u32; C1 * 8]; 14]; 14]; cache_size]),
            s2: (false, vec![[[[0u32; C2 * 8]; 7]; 7]; cache_size]),
            s3: (false, vec![[0u32; C3 * 8 * 3 * 3]; cache_size]),
            s4: (false, vec![[0u32; D1 * 8]; cache_size]),
            cache_size: cache_size,
        }
    }
}

#[derive(Clone, Copy)]
struct Message {
    mutate: bool,       // true if the worker is to mutate its parameters
    layer: usize,       // the layer to be mutated.
    bit: usize,         // the bit to be flipped.
    update_cache: bool, // true if the worker is to update its cache
    cache_index: usize, // the layer of the cache to be updated
    loss: bool,         // true if the worker is to send back loss
    seed: u64,          // seed when selecting the minibatches
    accuracy: bool,
    minibatch_size: usize, // number of samples to use when computing loss.
}

#[derive(Clone, Copy)]
struct Parameters {
    ck1: [u8; K1_SIZE],
    ck2: [u8; K2_SIZE],
    ck3: [u8; K3_SIZE],
    d1: [u8; D1_SIZE],
    d2: [u8; D2_SIZE],
}

impl Parameters {
    fn new() -> Parameters {
        Parameters {
            ck1: random_byte_array!(K1_SIZE)(),
            ck2: random_byte_array!(K2_SIZE)(),
            ck3: random_byte_array!(K3_SIZE)(),
            d1: random_byte_array!(D1_SIZE)(),
            d2: random_byte_array!(D2_SIZE)(),
        }
    }
    fn mutate(&mut self, layer: usize, bit: usize) {
        if layer == 1 {
            flip_bit!(self.ck1, bit);
        } else if layer == 2 {
            flip_bit!(self.ck2, bit);
        } else if layer == 3 {
            flip_bit!(self.ck3, bit);
        } else if layer == 4 {
            flip_bit!(self.d1, bit);
        } else if layer == 5 {
            flip_bit!(self.d2, bit);
        } else {
            panic!("bad layer ID");
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
        for &i in self.d1.iter() {
            wtr.push(i);
        }
        for &i in self.d2.iter() {
            wtr.push(i);
        }
    }
    fn read(mut data: &mut Vec<u8>) -> Parameters {
        data.reverse();
        Parameters {
            ck1: read_array!(K1_SIZE)(&mut data),
            ck2: read_array!(K2_SIZE)(&mut data),
            ck3: read_array!(K3_SIZE)(&mut data),
            d1: read_array!(D1_SIZE)(&mut data),
            d2: read_array!(D2_SIZE)(&mut data),
        }
    }
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

struct WorkerPool {
    send_chans: Vec<Sender<Message>>,
    recv_chan: Receiver<f64>,
    nthreads: usize,
}

impl WorkerPool {
    fn new(mut images: Vec<[[[u8; 1]; 28]; 28]>, mut labels: Vec<usize>, params: Parameters, nthreads: usize) -> WorkerPool {
        let (loss_tx, loss_rx) = channel();
        let mut sender_chans = vec![];
        let cache_size = images.len() / nthreads;
        for _ in 0..nthreads {
            let tx = loss_tx.clone();
            let (update_tx, update_rx) = channel();
            sender_chans.push(update_tx);
            // each worker needs its own shard of the training set.
            let examples_len = images.len();
            let images_shard = images.split_off(examples_len - cache_size);
            let labels_shard = labels.split_off(examples_len - cache_size);
            let mut worker_params = Box::new(params);
            thread::spawn(move || {
                let mut cache = Cache::new(images_shard, labels_shard, worker_params);
                loop {
                    let msg: Message = update_rx.recv().expect("can't receive update");
                    if msg.update_cache {
                        cache.update_cache(msg.cache_index);
                    }
                    if msg.mutate {
                        cache.mutate(msg.layer, msg.bit);
                    }
                    if msg.loss {
                        let mut seed = [0u8; 32];
                        seed[0] = (msg.seed >> 0) as u8;
                        seed[1] = (msg.seed >> 8) as u8;
                        seed[2] = (msg.seed >> 16) as u8;
                        seed[3] = (msg.seed >> 24) as u8;
                        let loss = cache.avg_loss(msg.minibatch_size, seed);
                        tx.send(loss).expect("can't send loss");
                    }
                    if msg.accuracy {
                        let acc = cache.avg_accuracy(msg.minibatch_size);
                        tx.send(acc).expect("can't send accuracy");
                    }
                }
            });
        }
        WorkerPool {
            send_chans: sender_chans,
            recv_chan: loss_rx,
            nthreads: nthreads,
        }
    }
    fn mutate(&self, l: usize, b: usize) {
        let update = Message {
            mutate: true,
            layer: l,
            bit: b,
            update_cache: false,
            cache_index: 0,
            loss: false,
            seed: 0,
            accuracy: false,
            minibatch_size: 0,
        };
        for w in 0..self.nthreads {
            self.send_chans[w].send(update.clone()).expect("can't send update")
        }
    }
    fn update_cache(&self, i: usize) {
        let update = Message {
            mutate: false,
            layer: 0,
            bit: 0,
            update_cache: true,
            cache_index: i,
            loss: false,
            seed: 0,
            accuracy: false,
            minibatch_size: 0,
        };
        for w in 0..self.nthreads {
            self.send_chans[w].send(update.clone()).expect("can't send update")
        }
    }
    fn loss(&self, n: usize, seed: u64) -> f64 {
        let update = Message {
            mutate: false,
            layer: 0,
            bit: 0,
            update_cache: false,
            cache_index: 0,
            loss: true,
            seed: seed,
            accuracy: false,
            minibatch_size: n / self.nthreads,
        };
        for w in 0..self.nthreads {
            self.send_chans[w].send(update.clone()).expect("can't send update")
        }
        let mut sum_loss = 0f64;
        for _ in 0..self.nthreads {
            sum_loss += self.recv_chan.recv().expect("can't receive loss");
        }
        sum_loss / self.nthreads as f64
    }
    fn accuracy(&self, n: usize) -> f64 {
        let update = Message {
            mutate: false,
            layer: 0,
            bit: 0,
            update_cache: false,
            cache_index: 0,
            loss: false,
            seed: 0,
            accuracy: true,
            minibatch_size: n / self.nthreads,
        };
        for w in 0..self.nthreads {
            self.send_chans[w].send(update.clone()).expect("can't send update")
        }
        let mut sum_loss = 0f64;
        for _ in 0..self.nthreads {
            sum_loss += self.recv_chan.recv().expect("can't receive loss");
        }
        sum_loss / self.nthreads as f64
    }
}

fn main() {
    const TRAINING_SIZE: usize = 60_000;
    const TEST_SIZE: usize = 10_000;
    const NTHREADS: usize = 8;
    const MINIBATCH_SIZE: usize = 64;
    println!("starting v0.1.4 with {:?} threads", NTHREADS);

    let images = mnist::load_images_u8_1chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);

    let test_images = mnist::load_images_u8_1chan(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);

    let mut params = load_params();
    //let mut params = Parameters::new();

    let pool = WorkerPool::new(images, labels, params, NTHREADS);
    let test_pool = WorkerPool::new(test_images, test_labels, params, NTHREADS);

    let layer_sizes = [0, K1_SIZE * 8, K2_SIZE * 8, K3_SIZE * 8, D1_SIZE * 8, D2_SIZE * 8]; // lookup table of layer sizes
    println!("layer bits: {:?}", layer_sizes);

    let mut seed: u64 = 0;
    let mut nil_loss: f64 = 1234.56;
    for i in 0.. {
        for l in 1..6 {
            println!("begining layer {:?} with {:?} bits", l, layer_sizes[l]);
            pool.update_cache(l - 1);
            test_pool.update_cache(l - 1);
            for b in 0..layer_sizes[l] {
                if b % 17 == 0 {
                    seed += 1;
                    println!("{:?} new seed: {:?}", i, seed);
                    nil_loss = pool.loss(MINIBATCH_SIZE, seed);
                    let acc = test_pool.accuracy(100);
                    println!("{:?} test acc: {:?}%", i, acc * 100f64);
                }
                pool.mutate(l, b);
                //let start = SystemTime::now();
                let new_loss = pool.loss(MINIBATCH_SIZE, seed);
                //println!("time per example: {:?}", start.elapsed().unwrap() / (MINIBATCH_SIZE / NTHREADS) as u32);
                if new_loss < nil_loss {
                    nil_loss = new_loss;
                    // update the eval worker.
                    test_pool.mutate(l, b);
                    params.mutate(l, b);
                    println!("{:?} {:?}/{:?} loss: {:?}", l, b, layer_sizes[l], new_loss);
                } else {
                    //println!("{:?} {:?}/{:?} reverting: {:?}", l, b, layer_sizes[l], new_loss);
                    pool.mutate(l, b);
                }
            }
            write_params(&params);
            let acc = test_pool.accuracy(TEST_SIZE);
            println!("{:?} test acc: {:?}%", i, acc * 100f64);
        }
    }
}
