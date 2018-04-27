// 200 acc: 19.88%
#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use rand::Rng;
use std::sync::Arc;
use std::thread;

const TRAINING_SIZE: usize = 10000;
const TEST_SIZE: usize = 10000;
const NTHREADS: usize = 96;

conv2d!(conv1, 28, 28, (1, 1));
conv2d!(conv2, 26, 26, (2, 2));
conv2d!(conv3, 12, 12, (2, 2));
conv2d!(conv4, 5, 5, (1, 1));
squash!(squash_layer, 3, 3);
dense_bits2ints!(dense_5, 9, 10);

struct Parameters {
    l1_weights: [[u64; 9]; 64],
    l1_thresholds: [i16; 64],
    l2_weights: [[u64; 9]; 64],
    l2_thresholds: [i16; 64],
    l3_weights: [[u64; 9]; 64],
    l3_thresholds: [i16; 64],
    l4_weights: [[u64; 9]; 64],
    l4_thresholds: [i16; 64],
    l5_weights: [[u64; 9]; 10],
}

macro_rules! random_bits {
    () => {
        rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
            & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
    };
}

impl Parameters {
    fn new_nil() -> Parameters {
        Parameters {
            l1_weights: [[0u64; 9]; 64],
            l1_thresholds: [(9 * 64 / 2) as i16; 64],
            l2_weights: [[0u64; 9]; 64],
            l2_thresholds: [(9 * 64 / 2) as i16; 64],
            l3_weights: [[0u64; 9]; 64],
            l3_thresholds: [(9 * 64 / 2) as i16; 64],
            l4_weights: [[0u64; 9]; 64],
            l4_thresholds: [(9 * 64 / 2) as i16; 64],
            l5_weights: [[0u64; 9]; 10],
        }
    }
    fn random() -> Parameters {
        let mut params = Parameters {
            l1_weights: [[0u64; 9]; 64],
            l1_thresholds: [(9 * 64 / 2) as i16; 64],
            l2_weights: [[0u64; 9]; 64],
            l2_thresholds: [(9 * 64 / 2) as i16; 64],
            l3_weights: [[0u64; 9]; 64],
            l3_thresholds: [(9 * 64 / 2) as i16; 64],
            l4_weights: [[0u64; 9]; 64],
            l4_thresholds: [(9 * 64 / 2) as i16; 64],
            l5_weights: [[0u64; 9]; 10],
        };
        for o in 0..64 {
            for i in 0..9 {
                params.l1_weights[o][i] = rand::random::<u64>();
                params.l2_weights[o][i] = rand::random::<u64>();
                params.l3_weights[o][i] = rand::random::<u64>();
                params.l4_weights[o][i] = rand::random::<u64>();
            }
        }
        for o in 0..10 {
            for i in 0..9 {
                params.l5_weights[o][i] = rand::random::<u64>();
            }
        }
        params
    }
    fn child(&self) -> Parameters {
        let mut child_params = Parameters::new_nil();
        for o in 0..64 {
            for i in 0..9 {
                child_params.l1_weights[o][i] = self.l1_weights[o][i] ^ random_bits!();
                child_params.l2_weights[o][i] = self.l2_weights[o][i] ^ random_bits!();
                child_params.l3_weights[o][i] = self.l3_weights[o][i] ^ random_bits!();
                child_params.l4_weights[o][i] = self.l4_weights[o][i] ^ random_bits!();
            }
        }
        for t in 0..64 {
            child_params.l1_thresholds[t] = self.l1_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
            child_params.l2_thresholds[t] = self.l2_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
            child_params.l3_thresholds[t] = self.l3_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
            child_params.l4_thresholds[t] = self.l4_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
        }
        for o in 0..10 {
            for i in 0..9 {
                child_params.l5_weights[o][i] = self.l5_weights[o][i] ^ random_bits!();
            }
        }
        child_params
    }
}

fn infer(image: &[[u64; 28]; 28], params: &Parameters) -> usize {
    let s1 = conv1(&image, &params.l1_weights, &params.l1_thresholds);
    let s2 = conv2(&s1, &params.l2_weights, &params.l2_thresholds);
    let s3 = conv3(&s2, &params.l3_weights, &params.l3_thresholds);
    let s4 = conv4(&s3, &params.l4_weights, &params.l4_thresholds);
    let squashed = squash_layer(&s4);
    //println!("squashed: {:?}", squashed);
    //println!("l5 weights {:?}", params.l5_weights);
    let output = dense_5(&squashed, &params.l5_weights);
    //println!("output {:?}", output);
    output.iter().enumerate().max_by_key(|&(_, item)| item).unwrap().0
}

fn avg_accuracy(examples: &Arc<Vec<([[u64; 28]; 28], usize)>>, params: &Parameters) -> f32 {
    let total: u64 = examples.iter().map(|(image, label)| (infer(image, params) == *label) as u64).sum();
    total as f32 / examples.len() as f32
}

fn main() {
    let images = mnist::load_images_64chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);
    let examples: Arc<Vec<([[u64; 28]; 28], usize)>> = Arc::new(images.iter().zip(labels).map(|(&image, target)| (image, target)).collect());
    //let test_images = mnist::load_images_bitpacked(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    //let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);
    //let test_examples: Arc<Vec<([u64; 784], usize)>> = Arc::new(test_images.iter().zip(test_labels).map(|(&image, target)| (image, target)).collect());
    println!("v0.2, using {:?} threads", NTHREADS);
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
            //println!("child acc: {:?}", child_acc * 100.0);
            if child_acc > acc {
                acc = child_acc;
                params = child_params;
            }
        }
        println!("{:?} acc: {:?}%", i, acc * 100.0);
    }
}
