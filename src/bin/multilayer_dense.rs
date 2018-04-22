#[macro_use]
extern crate bitnn;
extern crate rand;

use bitnn::datasets::mnist;
use std::time::SystemTime;

const TRAINING_SIZE: usize = 60000;
const TEST_SIZE: usize = 10000;
const H1: usize = 3;
const H2: usize = 2;

dense_bits2bits!(layer1, 13, H1);
dense_bits2bits_oneoutput!(layer1_oneoutput, 13, H1);
dense_bits2bits!(layer2, H1, H2);
dense_bits2bits_oneoutput!(layer2_oneoutput, H1, H2);
dense_bits2ints!(layer3, H2, 10);
dense_bits2ints_oneoutput!(layer3_oneoutput, H2, 10);
int_loss!(loss, 10);

fn avg_loss(cache: &mut Vec<ExampleCache>, params: &Parameters, loss_func: fn(&ExampleCache, &Parameters, usize) -> u32, o: usize) -> f64 {
    let sum: u64 = cache.iter().by_ref().map(|example| loss_func(&example, &params, o) as u64).sum();
    sum as f64 / TRAINING_SIZE as f64
}

fn nocache_loss(cache: &ExampleCache, params: &Parameters, updated_output: usize) -> u32 {
    let mut s1 = [0u64; H1];
    layer1(&mut s1, &params.l1, &cache.input);
    let mut s2 = [0u64; H2];
    layer2(&mut s2, &params.l2, &s1);
    let mut s3 = [0u32; 10];
    layer3(&mut s3, &params.l3, &s2);
    loss(&s3, &cache.target)
}

fn l1_loss(cache: &ExampleCache, params: &Parameters, updated_output: usize) -> u32 {
    let mut s1 = cache.states.0;
    layer1_oneoutput(&mut s1, &params.l1, &cache.input, updated_output);
    if s1[updated_output / 64] == cache.states.0[updated_output / 64] {
        cache.loss
    } else {
        let mut s2 = [0u64; H2];
        layer2(&mut s2, &params.l2, &s1);
        if s2 == cache.states.1 {
            cache.loss
        } else {
            let mut s3 = [0u32; 10];
            layer3(&mut s3, &params.l3, &s2);
            if s3 == cache.states.2 {
                cache.loss
            } else {
                loss(&s3, &cache.target)
            }
        }
    }
}

fn l2_loss(cache: &ExampleCache, params: &Parameters, updated_output: usize) -> u32 {
    let mut s2 = cache.states.1;
    layer2_oneoutput(&mut s2, &params.l2, &cache.states.0, updated_output);
    if s2 == cache.states.1 {
        cache.loss
    } else {
        let mut s3 = [0u32; 10];
        layer3(&mut s3, &params.l3, &s2);
        if s3 == cache.states.2 {
            cache.loss
        } else {
            loss(&s3, &cache.target)
        }
    }
}

fn l3_loss(cache: &ExampleCache, params: &Parameters, updated_output: usize) -> u32 {
    let mut s3 = cache.states.2;
    layer3_oneoutput(&mut s3, &params.l3, &cache.states.1, updated_output);
    if s3 == cache.states.2 {
        cache.loss
    } else {
        loss(&s3, &cache.target)
    }
}

struct ExampleCache {
    input: [u64; 13],
    target: [u32; 10],
    loss: u32,
    states: ([u64; H1], [u64; H2], [u32; 10]),
}

impl ExampleCache {
    fn new(input: [u64; 13], target: [u32; 10], params: &Parameters) -> ExampleCache {
        let mut example = ExampleCache {
            input: input,
            target: target,
            loss: 0u32,
            states: ([0u64; H1], [0u64; H2], [0u32; 10]),
        };
        example.refresh(&params);
        example
    }
    fn refresh(&mut self, params: &Parameters) {
        layer1(&mut self.states.0, &params.l1, &self.input);
        layer2(&mut self.states.1, &params.l2, &self.states.0);
        layer3(&mut self.states.2, &params.l3, &self.states.1);
        self.loss = loss(&self.states.2, &self.target);
    }
}

struct Parameters {
    l1: [[u64; 13]; H1 * 64],
    l2: [[u64; H1]; H2 * 64],
    l3: [[u64; H2]; 10],
}

impl Parameters {
    fn random() -> Parameters {
        let mut params = Parameters {
            l1: [[0u64; 13]; H1 * 64],
            l2: [[0u64; H1]; H2 * 64],
            l3: [[0u64; H2]; 10],
        };
        for o in 0..H1 * 64 {
            for i in 0..13 {
                params.l1[o][i] = rand::random::<u64>()
            }
        }
        for o in 0..H2 * 64 {
            for i in 0..H1 {
                params.l2[o][i] = rand::random::<u64>()
            }
        }
        for o in 0..10 {
            for i in 0..H2 {
                params.l3[o][i] = rand::random::<u64>()
            }
        }
        params
    }
}

fn optimize_layer3(cache: &mut Vec<ExampleCache>, params: &mut Parameters) {
    for e in 0..TRAINING_SIZE {
        cache[e].refresh(&params);
    }
    let mut nil_avg_loss = avg_loss(cache, &params, nocache_loss, 0);
    println!("avg nil loss: {:?}", nil_avg_loss);
    println!("starting layer 3");
    for i in 0..H2 {
        for o in 0..10 {
            let start = SystemTime::now();
            let mut changed = false;
            for b in 0..64 {
                params.l3[o][i] = params.l3[o][i] ^ (0b1u64 << b);
                let avg_loss = avg_loss(cache, &params, l3_loss, o);
                if avg_loss < nil_avg_loss {
                    nil_avg_loss = avg_loss;
                    changed = true;
                //println!("{:?} loss: {:?}", b, avg_loss);
                } else {
                    params.l3[o][i] = params.l3[o][i] ^ (0b1u64 << b); // revert
                }
            }
            if changed {
                for e in 0..TRAINING_SIZE {
                    cache[e].refresh(&params);
                }
            }
            //println!("{:?} {:?} time: {:?}", o, i, start.elapsed().unwrap());
        }
    }
}

fn optimize_layer2(cache: &mut Vec<ExampleCache>, params: &mut Parameters) {
    for e in 0..TRAINING_SIZE {
        cache[e].refresh(&params);
    }
    let mut nil_avg_loss = avg_loss(cache, &params, nocache_loss, 0);
    println!("avg nil loss: {:?}", nil_avg_loss);
    println!("starting layer 2");
    for i in 0..H1 {
        for o in 0..H2 {
            let start = SystemTime::now();
            let mut changed = false;
            for b in 0..64 {
                params.l2[o][i] = params.l2[o][i] ^ (0b1u64 << b);
                let avg_loss = avg_loss(cache, &params, l2_loss, o);
                if avg_loss < nil_avg_loss {
                    nil_avg_loss = avg_loss;
                    changed = true;
                //println!("{:?} loss: {:?}", b, avg_loss);
                } else {
                    params.l2[o][i] = params.l2[o][i] ^ (0b1u64 << b); // revert
                }
            }
            if changed {
                for e in 0..TRAINING_SIZE {
                    cache[e].refresh(&params);
                }
            }
            //println!("{:?} {:?} time: {:?}", o, i, start.elapsed().unwrap());
        }
    }
}

fn optimize_layer1(cache: &mut Vec<ExampleCache>, params: &mut Parameters) {
    for e in 0..TRAINING_SIZE {
        cache[e].refresh(&params);
    }
    let mut nil_avg_loss = avg_loss(cache, &params, nocache_loss, 0);
    println!("avg nil loss: {:?}", nil_avg_loss);
    println!("starting layer 1");
    for i in 0..13 {
        let mut changed = false;
        for o in 0..H1 {
            let start = SystemTime::now();
            let start_64 = SystemTime::now();
            for b in 0..64 {
                params.l1[o][i] = params.l1[o][i] ^ (0b1u64 << b);
                let avg_loss = avg_loss(cache, &params, l1_loss, o);
                if avg_loss < nil_avg_loss {
                    nil_avg_loss = avg_loss;
                    changed = true;
                //println!("{:?} loss: {:?}", b, avg_loss);
                } else {
                    params.l1[o][i] = params.l1[o][i] ^ (0b1u64 << b); // revert
                }
            }
            println!("{:?} {:?} time: {:?}", o, i, start.elapsed().unwrap());
        }
        if changed {
            let refresh_start = SystemTime::now();
            for e in 0..TRAINING_SIZE {
                cache[e].refresh(&params);
            }
        }
    }
}

fn infer(image: &[u64; 13], params: &Parameters) -> usize {
    let mut s1 = [0u64; H1];
    layer1(&mut s1, &params.l1, &image);
    let mut s2 = [0u64; H2];
    layer2(&mut s2, &params.l2, &s1);
    let mut s3 = [0u32; 10];
    layer3(&mut s3, &params.l3, &s2);
    s3.iter().enumerate().max_by_key(|&(_, item)| item).unwrap().0
}

fn avg_accuracy(images: &Vec<[u64; 13]>, labels: &Vec<usize>, params: &Parameters) -> f32 {
    if images.len() != labels.len() {
        panic!("images.len != labels.len");
    }
    let total: u64 = images
        .iter()
        .zip(labels.iter())
        .map(|(image, &label)| (infer(image, params) == label) as u64)
        .sum();
    total as f32 / images.len() as f32
}

fn main() {
    let onval = (H2 as u32 * 64) / 2;

    let images = mnist::load_images_bitpacked(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels_onehot(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE, onval);
    let test_images = mnist::load_images_bitpacked(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);

    let mut params = Parameters::random();

    let mut state_cache: Vec<_> = images
        .iter()
        .zip(labels.iter())
        .map(|(&image, &label)| ExampleCache::new(image, label, &params))
        .collect();

    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer3(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer2(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer3(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer2(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer1(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer3(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer2(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer1(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer3(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer2(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
    optimize_layer3(&mut state_cache, &mut params);
    println!("avg_accuracy: {:?}%", avg_accuracy(&test_images, &test_labels, &params) * 100 as f32);
}
