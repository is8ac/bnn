use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::sync::mpsc::channel;
use std::thread;
use std::thread::sleep;
use std::time;
use std::time::SystemTime;

#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use rand::prelude::*;
use std::time::Duration;

const TRAINING_SIZE: usize = 60_000;
const TEST_SIZE: usize = 10_000;
const NTHREADS: usize = 8;
const CACHE_SIZE: usize = TRAINING_SIZE / NTHREADS;
const MINIBATCH_SIZE: usize = 1000;

// channel sizes (all but C0 will be multiplied by 8);
const C0: usize = 1;
const C1: usize = 4;
const C2: usize = 8;
const D1: usize = 128;

// Parameter sizes and max output ranges.
const K1_SIZE: usize = C0 * C1 * 5 * 5;
const K1_MAX: i32 = i32::max_value() / ((C1 * 8) * 5 * 5) as i32;

const K2_SIZE: usize = C1 * (C2 * 8) * 5 * 5;
const K2_MAX: i32 = i32::max_value() / ((C2 * 8) * 5 * 5) as i32;

const D1_SIZE: usize = C2 * 7 * 7 * D1 * 8;
const D1_MAX: i32 = i32::max_value() / D1 as i32;

const D2_SIZE: usize = D1 * 10;

conv!(conv1, i32, 28, 28, C0, u8, C1, u32, 5, 5, K1_MAX, 0);
max_pool!(pool1, u32, 28, 28, C1 * 8);

conv!(conv2, i32, 14, 14, C1 * 8, u32, C2, u32, 5, 5, K2_MAX, 0);
max_pool!(pool2, u32, 14, 14, C2 * 8);

flatten3d!(flatten, u32, 7, 7, C2 * 8);

dense_ints2ints!(dense1, u32, i32, 7 * 7 * C2, D1 * 8);
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
    s2: (bool, Vec<[u32; C2 * 8 * 7 * 7]>),     // output of second conv and maxpool
    s3: (bool, Vec<[u32; D1 * 8]>),             // output of first FC layer
    cache_size: usize,
}

impl Cache {
    fn calc_s1(&self, i: usize) -> [[[u32; C1 * 8]; 14]; 14] {
        pool1(&conv1(&self.images[i], &self.params.ck1))
    }
    fn calc_s2(&self, i: usize) -> [u32; C2 * 8 * 7 * 7] {
        if self.s1.0 {
            flatten(&pool2(&conv2(&self.s1.1[i], &self.params.ck2)))
        } else {
            flatten(&pool2(&conv2(&self.calc_s1(i), &self.params.ck2)))
        }
    }
    fn calc_s3(&self, i: usize) -> [u32; D1 * 8] {
        if self.s2.0 {
            relu1(&dense1(&self.s2.1[i], &self.params.d1))
        } else {
            relu1(&dense1(&self.calc_s2(i), &self.params.d1))
        }
    }
    fn calc_actual(&self, i: usize) -> [i32; 10] {
        if self.s3.0 {
            dense2(&self.s3.1[i], &self.params.d2)
        } else {
            dense2(&self.calc_s3(i), &self.params.d2)
        }
    }
    fn update_s1(&mut self) {
        println!("updating cache 1");
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
        println!("updating cache 2");
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
        println!("updating cache 3");
        for i in 0..self.cache_size {
            self.s3.1[i] = self.calc_s3(i);
        }
        self.s3.0 = true;
    }
    fn invalidate_s3(&mut self) {
        self.s3.0 = false;
    }
    fn avg_loss(&mut self, batch_size: usize) -> f64 {
        let mut vec: Vec<usize> = (0..self.cache_size).collect();
        let slice: &mut [usize] = &mut vec;

        thread_rng().shuffle(slice);
        let mut sum = 0f64;

        for &e in slice[0..batch_size].iter() {
            let actual = self.calc_actual(e);
            sum += lossfn(&actual, self.labels[e]);
        }
        sum / batch_size as f64
    }
    fn avg_accuracy(&mut self) -> f64 {
        let mut sum: u64 = 0;
        for e in 0..self.cache_size {
            let actual = self.calc_actual(e);
            sum += is_correct(&actual, self.labels[e]) as u64;
        }
        sum as f64 / self.cache_size as f64
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
        } else {
            panic!("bad cache index");
        }
    }
    fn mutate(&mut self, update: Update) {
        if update.layer == 1 {
            flip_bit!(self.params.ck1, update.bit);
            self.invalidate_s1();
        } else if update.layer == 2 {
            flip_bit!(self.params.ck2, update.bit);
            self.invalidate_s2();
        } else if update.layer == 3 {
            flip_bit!(self.params.d1, update.bit);
            self.invalidate_s3();
        } else if update.layer == 4 {
            flip_bit!(self.params.d2, update.bit);
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
            s2: (false, vec![[0u32; C2 * 8 * 7 * 7]; cache_size]),
            s3: (false, vec![[0u32; D1 * 8]; cache_size]),
            cache_size: cache_size,
        }
    }
}

#[derive(Clone, Copy)]
struct Update {
    cache_index: usize,
    layer: usize,
    bit: usize,
    minibatch_size: usize,
}

#[derive(Clone, Copy)]
struct Parameters {
    ck1: [u8; K1_SIZE],
    ck2: [u8; K2_SIZE],
    d1: [u8; D1_SIZE],
    d2: [u8; D2_SIZE],
}

impl Parameters {
    fn new() -> Parameters {
        Parameters {
            ck1: random_byte_array!(K1_SIZE)(),
            ck2: random_byte_array!(K2_SIZE)(),
            d1: random_byte_array!(D1_SIZE)(),
            d2: random_byte_array!(D2_SIZE)(),
        }
    }
    fn write(&self, wtr: &mut Vec<u8>) {
        for &i in self.ck1.iter() {
            wtr.push(i);
        }
        for &i in self.ck2.iter() {
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

fn main() {
    println!("starting v0.1.4 with {:?} threads", NTHREADS);
    let mut rng = thread_rng();
    let mut images = mnist::load_images_u8_1chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let mut labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);
    //let examples: Vec<([[[u8; 1]; 28]; 28], usize)> = images.iter().zip(labels).map(|(&image, target)| (image, target)).collect();

    let test_images = mnist::load_images_u8_1chan(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);
    //let test_examples: Vec<([[[u8; 1]; 28]; 28], usize)> = test_images.iter().zip(test_labels).map(|(&image, target)| (image, target)).collect();

    //let mut params = load_params();
    let mut params = Parameters::new();

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
        let mut worker_params = Box::new(params);
        thread::spawn(move || {
            let mut cache = Cache::new(images_shard, labels_shard, worker_params);
            loop {
                let (update, mutate, send_loss) = update_rx.recv().expect("can't receive update");
                if mutate {
                    cache.mutate(update);
                }
                if send_loss {
                    let loss = cache.avg_loss(update.minibatch_size);
                    tx.send(loss).expect("can't send loss");
                }
            }
        });
    }
    let (eval_update_tx, eval_update_rx) = channel();
    let mut eval_params = Box::new(params);
    thread::spawn(move || {
        let mut test_cache = Cache::new(test_images, test_labels, eval_params);
        let mut last_save = SystemTime::now();
        loop {
            let update: Update = eval_update_rx.recv().expect("eval thread can't receive update");
            test_cache.update_cache(update.cache_index);
            test_cache.mutate(update);
            if (last_save + Duration::new(10, 0)) < SystemTime::now() {
                let avg_acc = test_cache.avg_accuracy();
                println!("avg acc: {:?}%", avg_acc * 100.0);
                write_params(&test_cache.params);
                last_save = SystemTime::now();
            }
        }
    });
    let ten_millis = time::Duration::from_millis(1000);

    thread::sleep(ten_millis);

    let layers = [0, K1_SIZE * 8, K2_SIZE * 8, D1_SIZE * 8, D2_SIZE * 8];
    let train_order = [1, 2, 3, 4, 2, 3, 4, 3, 4, 4];

    for w in 0..NTHREADS {
        sender_chans[w]
            .send((
                Update {
                    layer: 0,
                    bit: 0,
                    cache_index: 0,
                    minibatch_size: MINIBATCH_SIZE / NTHREADS,
                },
                false,
                true,
            ))
            .expect("can't send update")
    }
    println!("sent updates");
    let mut sum_loss = 0f64;
    for w in 0..NTHREADS {
        println!("getting loss",);
        sum_loss += loss_rx.recv().expect("can't receive loss");
    }
    let mut nil_loss = sum_loss / NTHREADS as f64;
    println!("nil loss: {:?}", nil_loss);
    loop {
        for &l in train_order.iter() {
            println!("begining layer {:?} with {:?} bits", l, layers[l]);
            for i in 0..20 {
                let b = rng.gen_range(0, layers[l]);
                let update = Update {
                    layer: l,
                    bit: b,
                    cache_index: 0,
                    minibatch_size: 100,
                };
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
                    eval_update_tx.send(update).expect("can't send update");
                    let update = Update {
                        layer: 0,
                        bit: 0,
                        cache_index: 0,
                        minibatch_size: CACHE_SIZE,
                    };
                    for w in 0..NTHREADS {
                        sender_chans[w].send((update.clone(), false, true)).expect("can't send update")
                    }
                    let mut sum_loss = 0f64;
                    for w in 0..NTHREADS {
                        sum_loss += loss_rx.recv().expect("can't receive loss");
                    }
                    nil_loss = sum_loss / NTHREADS as f64;
                    println!("{:?} {:?}/{:?} put loss: {:?} real loss: {:?}", l, i, layers[l], new_loss, nil_loss);
                } else {
                    println!("reverting",);
                    // revert
                    for w in 0..NTHREADS {
                        sender_chans[w].send((update, true, false)).expect("can't send update")
                    }
                }
            }
        }
    }
}
