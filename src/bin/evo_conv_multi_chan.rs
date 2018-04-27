// 76%
#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use rand::Rng;
use std::sync::Arc;
use std::thread;

const TRAINING_SIZE: usize = 6000;
const TEST_SIZE: usize = 10000;
const NTHREADS: usize = 96;
const C0: usize = 1;
const C1: usize = 3;
const C2: usize = 2;
const C3: usize = 1;

fn random_bits() -> u64 {
    rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
        & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
        & rand::random::<u64>()
}

fn random_int() -> i16 {
    rand::thread_rng().gen_range(-1, 2)
}


conv3x3!(Conv1, 28, 28, C0, C1);
pool_or2x2!(Pool1, 28, 28, C1);
conv3x3!(Conv2, 14, 14, C1, C2);
pool_or2x2!(Pool2, 14, 14, C2);
conv3x3!(Conv3, 7, 7, C2, C3);
flatten3d!(Flatten, 7, 7, C3);
dense_bits2ints!(Dense1, 9, 10);

struct Layers {
    conv1: Conv1,
    pool1: Pool1
    conv2: Conv2,
    conv3: Conv3,
    dense: Dense,
}

impl Layers {
    fn new_nil() -> Layers {
        Layers {
            conv1: Conv1,
            conv2_weights: [[[0u64; C1]; 9]; C2 * 64],
            conv3_weights: [[[0u64; C2]; 9]; C3 * 64],
            dense_weights: [[0u64; 7 * 7 * C3]; 10],
        }
    }
    fn random() -> Parameters {
        let mut params = Parameters {
            conv1_weights: [[[0u64; C0]; 9]; C1 * 64],
            conv1_thresholds: [(9 * 64 / 2) as i16; C1 * 64],
            conv2_weights: [[[0u64; C1]; 9]; C2 * 64],
            conv2_thresholds: [(9 * 64 / 2) as i16; C2 * 64],
            conv3_weights: [[[0u64; C2]; 9]; C3 * 64],
            conv3_thresholds: [(C3 * 64 / 2) as i16; C3 * 64],
            dense_weights: [[0u64; 7 * 7 * C3]; 10],
        };
        for o in 0..C1 * 64 {
            for i in 0..9 {
                for c in 0..C0 {
                    params.conv1_weights[o][i][c] = rand::random::<u64>();
                }
            }
        }
        for o in 0..C2 * 64 {
            for i in 0..9 {
                for c in 0..C1 {
                    params.conv2_weights[o][i][c] = rand::random::<u64>();
                }
            }
        }
        for o in 0..C3 * 64 {
            for i in 0..9 {
                for c in 0..C2 {
                    params.conv3_weights[o][i][c] = rand::random::<u64>();
                }
            }
        }
        for o in 0..10 {
            for i in 0..7 * 7 * C3 {
                params.dense_weights[o][i] = rand::random::<u64>();
            }
        }
        params
    }
    fn child(&self) -> Parameters {
        let mut child_params = Parameters::new_nil();
        for o in 0..64 {
            for i in 0..9 {
                child_params.conv1_weights[o][i] = self.conv1_weights[o][i] ^ random_bits!();
                child_params.conv2_weights[o][i] = self.conv2_weights[o][i] ^ random_bits!();
                child_params.conv3_weights[o][i] = self.conv3_weights[o][i] ^ random_bits!();
            }
        }
        for t in 0..64 {
            child_params.conv1_thresholds[t] = self.conv1_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
            child_params.conv2_thresholds[t] = self.conv2_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
            child_params.conv3_thresholds[t] = self.conv3_thresholds[t] + rand::thread_rng().gen_range(-1, 2);
        }
        for o in 0..10 {
            for i in 0..9 {
                child_params.dense_weights[o][i] = self.dense_weights[o][i] ^ random_bits!();
            }
        }
        child_params
    }
}

fn infer(image: &[[[u64; 1]; 28]; 28], params: &Parameters) -> usize {
    let s1 = conv1(&image, &params.l1_weights, &params.l1_thresholds);
    let pooled1 = pool1(s1);
    let s2 = conv2(&s1, &params.l2_weights, &params.l2_thresholds);
    let pooled2 = pool2(s2);
    let s3 = conv3(&s2, &params.l3_weights, &params.l3_thresholds);
    let flat = flatten(&s3);
    let output = dense_1(&flat, &params.dense1);
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
