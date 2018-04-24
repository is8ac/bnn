#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use rand::Rng;
use std::sync::Arc;
use std::thread;

const TRAINING_SIZE: usize = 60000;
const TEST_SIZE: usize = 10000;
const H1: usize = 4;
const H2: usize = 1;
const NTHREADS: usize = 8;

dense_bits_fused_threshold!(layer1, 13, H1);
dense_bits_fused_threshold!(layer2, H1, H2);
dense_bits2ints!(layer3, H2, 10);

struct Parameters {
    l1_weights: [[u64; 13]; H1 * 64],
    l1_thresholds: [i16; H1 * 64],
    l2_weights: [[u64; H1]; H2 * 64],
    l2_thresholds: [i16; H2 * 64],
    l3_weights: [[u64; H2]; 10],
}

macro_rules! random_bits {
    () => {
        rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
            & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()  & rand::random::<u64>()
    };
}

impl Parameters {
    fn new_nil() -> Parameters {
        Parameters {
            l1_weights: [[0u64; 13]; H1 * 64],
            l1_thresholds: [0i16; H1 * 64],
            l2_weights: [[0u64; H1]; H2 * 64],
            l2_thresholds: [0i16; H2 * 64],
            l3_weights: [[0u64; H2]; 10],
        }
    }
    fn random() -> Parameters {
        let mut params = Parameters {
            l1_weights: [[0u64; 13]; H1 * 64],
            l1_thresholds: [(13 * 64 / 2) as i16; H1 * 64],
            l2_weights: [[0u64; H1]; H2 * 64],
            l2_thresholds: [(H1 * 64 / 2) as i16; H2 * 64],
            l3_weights: [[0u64; H2]; 10],
        };
        for o in 0..H1 * 64 {
            for i in 0..13 {
                params.l1_weights[o][i] = rand::random::<u64>();
            }
        }
        for o in 0..H2 * 64 {
            for i in 0..H1 {
                params.l2_weights[o][i] = rand::random::<u64>();
            }
        }
        for o in 0..10 {
            for i in 0..H2 {
                params.l3_weights[o][i] = rand::random::<u64>();
            }
        }
        params
    }
    fn child(&self) -> Parameters {
        let mut child_params = Parameters::new_nil();
        for o in 0..H1 * 64 {
            for i in 0..13 {
                child_params.l1_weights[o][i] = self.l1_weights[o][i] ^ random_bits!();
            }
        }
        for t in 0..H1 * 64 {
            child_params.l1_thresholds[t] = self.l1_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
        }
        for o in 0..H2 * 64 {
            for i in 0..H1 {
                child_params.l2_weights[o][i] = self.l2_weights[o][i] ^ random_bits!();
            }
        }
        for t in 0..H2 * 64 {
            child_params.l2_thresholds[t] = self.l2_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
        }
        for o in 0..10 {
            for i in 0..H2 {
                child_params.l3_weights[o][i] = self.l3_weights[o][i] ^ random_bits!();
            }
        }
        child_params
    }
}

fn infer(image: &[u64; 13], params: &Parameters) -> usize {
    let mut s1 = [0u64; H1];
    layer1(&mut s1, &params.l1_weights, &params.l1_thresholds, &image);
    let mut s2 = [0u64; H2];
    layer2(&mut s2, &params.l2_weights, &params.l2_thresholds, &s1);
    let mut s3 = [0u16; 10];
    layer3(&mut s3, &params.l3_weights, &s2);
    s3.iter().enumerate().max_by_key(|&(_, item)| item).unwrap().0
}

fn avg_accuracy(examples: &Arc<Vec<([u64; 13], usize)>>, params: &Parameters) -> f32 {
    let total: u64 = examples.iter().map(|(image, label)| (infer(image, params) == *label) as u64).sum();
    total as f32 / examples.len() as f32
}

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);
    let examples: Arc<Vec<([u64; 13], usize)>> = Arc::new(images.iter().zip(labels).map(|(&image, target)| (image, target)).collect());
    //let test_images = mnist::load_images_bitpacked(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    //let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);

    let builder = thread::Builder::new();

    let mut params = Parameters::random();
    //println!("init_acc: {:?}%", avg_accuracy(&images, &labels, &params) * 100.0);
    let mut acc = 0f32;
    for i in 0..10000 {
        let mut children = vec![];
        for i in 0..NTHREADS {
            // Spin up another thread
            let child = params.child();
            let examples_arc = Arc::clone(&examples);
            children.push(thread::spawn(move || {
                let child_acc = avg_accuracy(&examples_arc, &child);
                (child_acc, child)
            }));
        }
        for child_thread in children {
            // Wait for the thread to finish. Returns a result.
            let (child_acc, child_params) = child_thread.join().unwrap();
            if child_acc > acc {
                acc = child_acc;
                params = child_params;
            }
        }
        println!("{:?} acc: {:?}%", i, acc * 100.0);
        println!("thresholds: {:?}", (params.l1_thresholds[0], params.l1_thresholds[1], params.l1_thresholds[2], params.l1_thresholds[3]));
    }
}
