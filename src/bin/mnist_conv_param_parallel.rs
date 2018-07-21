use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::sync::mpsc::channel;
use std::sync::mpsc::{Receiver, Sender};
use std::time::SystemTime;
use std::{thread, time};

use std::collections::HashSet;

#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use rand::prelude::*;

// channel sizes (all but C0 will be multiplied by 8);
const C0: usize = 1;
const C1: usize = 2;
const C2: usize = 2;
const C3: usize = 2;
const D1: usize = 3;

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
    fn set_bit(&mut self, layer: usize, b: usize, val: bool) {
        self.params.set_bit(layer, b, val);
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

macro_rules! read_bit {
    ($array:expr, $index:expr) => {
        (($array[$index / 8] >> ($index % 8)) & 0b1u8) == 0b1u8
    };
}

macro_rules! set_bit {
    ($array:expr, $index:expr, $val:expr) => {
        let mask = 0b1u8 << ($index % 8);
        $array[$index / 8] = ($array[$index / 8] & !mask) | (($val as u8) << ($index % 8));
    };
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
    fn read_bit(&self, layer: usize, bit: usize) -> bool {
        if layer == 1 {
            read_bit!(self.ck1, bit)
        } else if layer == 2 {
            read_bit!(self.ck2, bit)
        } else if layer == 3 {
            read_bit!(self.ck3, bit)
        } else if layer == 4 {
            read_bit!(self.d1, bit)
        } else if layer == 5 {
            read_bit!(self.d2, bit)
        } else {
            panic!("bad layer ID");
        }
    }
    fn set_bit(&mut self, layer: usize, b: usize, val: bool) {
        if layer == 1 {
            set_bit!(self.ck1, b, val);
        } else if layer == 2 {
            set_bit!(self.ck2, b, val);
        } else if layer == 3 {
            set_bit!(self.ck3, b, val);
        } else if layer == 4 {
            set_bit!(self.d1, b, val);
        } else if layer == 5 {
            set_bit!(self.d2, b, val);
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
                let seed: u64 = 0;
                loop {
                    let msg: Message = update_rx.recv().expect("can't receive update");
                    if msg.update_cache {
                        cache.update_cache(msg.cache_index);
                    }
                    if msg.mutate {
                        cache.mutate(msg.layer, msg.bit);
                    }
                    if msg.loss {
                        let mut seed_bytes = [0u8; 32];
                        seed_bytes[0] = (seed >> 0) as u8;
                        seed_bytes[1] = (seed >> 8) as u8;
                        seed_bytes[2] = (seed >> 16) as u8;
                        seed_bytes[3] = (seed >> 24) as u8;
                        let loss = cache.avg_loss(msg.minibatch_size, seed_bytes);
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

#[derive(Clone, Copy)]
struct Update {
    layer: usize,
    bit: usize,
    state: bool,
}

fn convert_to_seed(input: u64) -> [u8; 32] {
    let mut seed_bytes = [0u8; 32];
    seed_bytes[0] = (input >> 0) as u8;
    seed_bytes[1] = (input >> 8) as u8;
    seed_bytes[2] = (input >> 16) as u8;
    seed_bytes[3] = (input >> 24) as u8;
    seed_bytes
}

fn new_layer_worker(
    images: Vec<[[[u8; 1]; 28]; 28]>,
    labels: Vec<usize>,
    mut params: Box<Parameters>,
    layer_sizes: &'static [usize],
    update_tx: Sender<Update>,
    minibatch_size: usize,
) -> Sender<Update> {
    let (tx, rx) = channel::<Update>();

    let mut cache = Cache::new(images, labels, params);
    let mut seed: u64 = 0;
    thread::spawn(move || {
        let mut rng = rand::thread_rng();
        for layer in 1..layer_sizes.len() {
            println!("now on layer {:?} of size {:?}", layer, layer_sizes[layer]);
            let mut unoptimized_bits: HashSet<usize> = (0..layer_sizes[layer]).collect();
            let mut nil_loss = cache.avg_loss(minibatch_size, convert_to_seed(seed));
            while unoptimized_bits.len() != 0 {
                // get the index of random bit from the set of bits which have not yet been optimized.
                let bit = *unoptimized_bits.iter().nth(rng.gen_range(0, unoptimized_bits.len())).unwrap();
                cache.mutate(layer, bit);
                let new_loss = cache.avg_loss(minibatch_size, convert_to_seed(seed));
                let state = cache.params.read_bit(layer, bit);
                cache.mutate(layer, bit);
                if new_loss < nil_loss {
                    update_tx
                        .send(Update {
                            bit: bit,
                            state: state,
                            layer: layer,
                        })
                        .expect("worker: can't send update to master");
                }
                // now get all updates from the other workers
                let mut done = false;
                while !done {
                    let resp = rx.try_recv();
                    done = !resp.is_ok();
                    if !done {
                        let msg = resp.expect("can't unwrap msg");
                        cache.update_cache(layer - 1);
                        // if the update is for the same layer as we are working on,
                        cache.set_bit(msg.layer, msg.bit, msg.state); // apply it,
                        unoptimized_bits.remove(&msg.bit); // and remove it from the set of unoptimized_bits.
                    }
                }
            }
        }
        println!("loop is done");
    });
    tx
}

fn spawn_workers(
    mut images: Vec<[[[u8; 1]; 28]; 28]>,
    mut labels: Vec<usize>,
    params: Parameters,
    update_tx: Sender<Update>,
    minibatch_size: usize,
    n: usize,
    layer_sizes: &'static [usize],
) -> Vec<Sender<Update>> {
    let mut senders = vec![];
    for i in 0..n {
        let tx = update_tx.clone();
        let examples_len = images.len();
        let images_shard = images.split_off(examples_len - minibatch_size);
        let labels_shard = labels.split_off(examples_len - minibatch_size);
        let mut worker_params = Box::new(params);
        let worker_tx = new_layer_worker(images_shard, labels_shard, worker_params, layer_sizes, tx, minibatch_size);
        senders.push(worker_tx);
    }
    senders
}

fn start_broker(rx: Receiver<Update>, txs: Vec<Sender<Update>>) {
    thread::spawn(move || loop {
        let msg = rx.recv().expect("broker: can't receive msg");
        for tx in txs.iter() {
            tx.send(msg.clone()).expect("broker: can't send to worker");
        }
    });
}

fn main() {
    const TRAINING_SIZE: usize = 60_000;
    const TEST_SIZE: usize = 10_000;
    const NTHREADS: usize = 8;
    const MINIBATCH_SIZE: usize = 30;
    println!("starting v0.1.4 with {:?} threads", NTHREADS);

    let images = mnist::load_images_u8_1chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);

    let test_images = mnist::load_images_u8_1chan(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);

    //let mut params = load_params();
    let mut params = Parameters::new();

    let layer_sizes: &[usize] = &[0, K1_SIZE * 8, K2_SIZE * 8, K3_SIZE * 8, D1_SIZE * 8, D2_SIZE * 8]; // lookup table of layer sizes
    println!("layer bits: {:?}", layer_sizes);

    let mut seed: u64 = 0;
    let mut nil_loss: f64 = 1234.56;
    let (broker_tx, broker_rx) = channel::<Update>();
    let (eval_tx, eval_rx) = channel::<Update>();
    let mut worker_txs = spawn_workers(images, labels, params, broker_tx, MINIBATCH_SIZE, NTHREADS, &layer_sizes);
    worker_txs.push(eval_tx);
    start_broker(broker_rx, worker_txs);

    let mut cache = Cache::new(test_images, test_labels, Box::new(params));
    for layer in 1..layer_sizes.len() {
        println!("now on layer {:?} of size {:?}", layer, layer_sizes[layer]);
        cache.update_cache(layer - 1);
        let mut unoptimized_bits: HashSet<usize> = (0..layer_sizes[layer]).collect();
        while unoptimized_bits.len() != 0 {
            let msg = eval_rx.recv().expect("master: can't unwrap msg");
            //cache.update_cache(msg.layer - 1);
            // if the update is for the same layer as we are working on,
            cache.set_bit(msg.layer, msg.bit, msg.state); // apply it,
            unoptimized_bits.remove(&msg.bit); // and remove it from the set of unoptimized_bits.
            if msg.layer != layer {
                // but if it is for a different layer, complain about it.
                println!("master: not committing: update has layer {:?} but I'm on layer {:?}", msg.layer, layer);
            }
            println!("{:?} bits left", unoptimized_bits.len());
        }
        let avg_acc = cache.avg_accuracy(100);
        //let avg_loss = cache.avg_loss(100, convert_to_seed(42));
        println!("{:?} bits left, acc: {:?}%", unoptimized_bits.len(), avg_acc * 100f64);
    }
}
