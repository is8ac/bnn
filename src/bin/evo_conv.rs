// 49 acc: 18.3%
// 21 acc: 16.1%
#[macro_use]
extern crate bitnn;
extern crate rand;
use bitnn::datasets::mnist;
use bitnn::layers;
use std::sync::Arc;
use std::thread;

const TRAINING_SIZE: usize = 3000;
const TEST_SIZE: usize = 10000;
const NTHREADS: usize = 8;
const C0: usize = 1;
const C1: usize = 4;
const C2: usize = 3;
const C3: usize = 2;

u64_3d!(Kernel1, C1 * 64, 9, C0);
i16_1d!(Thresholds1, C1 * 64);
u64_3d!(Kernel2, C2 * 64, 9, C1);
i16_1d!(Thresholds2, C2 * 64);
u64_3d!(Kernel3, C3 * 64, 9, C2);
i16_1d!(Thresholds3, C3 * 64);
u64_2d!(Dense1, 10, 7 * 7 * C3);

conv3x3!(conv1, 28, 28, C0, C1);
pool_or2x2!(pool1, 28, 28, C1);
conv3x3!(conv2, 14, 14, C1, C2);
pool_or2x2!(pool2, 14, 14, C2);
conv3x3!(conv3, 7, 7, C2, C3);
flatten3d!(flatten, 7, 7, C3);
dense_bits2ints!(dense1, 7 * 7 * C3, 10);
int_loss!(sqr_diff_loss, 10);

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
        Parameters {
            ck1: self.ck1.child(&layers::random_bits14),
            //ct1: self.ct1.child(&layers::random_int_plusminus_one),
            ct1: self.ct1.child(&zero),
            ck2: self.ck2.child(&layers::random_bits14),
            //ct2: self.ct2.child(&layers::random_int_plusminus_one),
            ct2: self.ct2.child(&zero),
            ck3: self.ck3.child(&layers::random_bits14),
            //ct3: self.ct3.child(&layers::random_int_plusminus_one),
            ct3: self.ct3.child(&zero),
            dense1: self.dense1.child(&layers::random_bits14),
        }
    }
}

fn model(image: &[[[u64; 1]; 28]; 28], params: &Parameters) -> [i16; 10] {
    let s1 = conv1(&image, &params.ck1.weights, &params.ct1.thresholds);
    let pooled1 = pool1(&s1);
    let s2 = conv2(&pooled1, &params.ck2.weights, &params.ct2.thresholds);
    let pooled2 = pool2(&s2);
    let s3 = conv3(&pooled2, &params.ck3.weights, &params.ct3.thresholds);
    let flat = flatten(&s3);
    dense1(&flat, &params.dense1.weights)
}

fn loss(image: &[[[u64; 1]; 28]; 28], target: usize, params: &Parameters) -> i64 {
    let output = model(&image, &params);
    let sum: i16 = output.iter().map(|o| o - output[target]).sum();
    sum as i64
}

fn infer(image: &[[[u64; 1]; 28]; 28], params: &Parameters) -> usize {
    let output = model(&image, &params);
    output.iter().enumerate().max_by_key(|&(_, item)| item).unwrap().0
}

fn avg_loss(examples: &Arc<Vec<([[[u64; 1]; 28]; 28], usize)>>, params: &Parameters) -> f64 {
    let total: i64 = examples.iter().map(|(image, label)| loss(image, *label, params)).sum();
    //println!("total: {:?}", total);
    total as f64 / examples.len() as f64
}


fn avg_accuracy(examples: &Arc<Vec<([[[u64; 1]; 28]; 28], usize)>>, params: &Parameters) -> f32 {
    let total: u64 = examples.iter().map(|(image, label)| (infer(image, params) == *label) as u64).sum();
    total as f32 / examples.len() as f32
}

fn main() {
    let images = mnist::load_images_64chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);
    let examples: Arc<Vec<([[[u64; 1]; 28]; 28], usize)>> = Arc::new(images.iter().zip(labels).map(|(&image, target)| (image, target)).collect());
    //let test_images = mnist::load_images_bitpacked(&String::from("mnist/t10k-images-idx3-ubyte"), TEST_SIZE);
    //let test_labels = mnist::load_labels(&String::from("mnist/t10k-labels-idx1-ubyte"), TEST_SIZE);
    //let test_examples: Arc<Vec<([u64; 784], usize)>> = Arc::new(test_images.iter().zip(test_labels).map(|(&image, target)| (image, target)).collect());
    println!("v0.8, using {:?} threads", NTHREADS);

    let mut params = Parameters::new();
    //println!("init_acc: {:?}%", avg_accuracy(&images, &labels, &params) * 100.0);
    let mut nil_loss = 100000000000000f64;
    for i in 0..10000 {
        let mut children = vec![];
        for i in 0..NTHREADS {
            // Spin up another thread
            let child = params.child();
            let examples_arc = Arc::clone(&examples);
            children.push(thread::spawn(move || {
                let child_loss = avg_loss(&examples_arc, &child);
                (child_loss, child)
            }));
        }
        for child_thread in children {
            // Wait for the thread to finish. Returns a result.
            let (child_loss, child_params) = child_thread.join().unwrap();
            if child_loss <= nil_loss {
                println!("keeping");
                nil_loss = child_loss;
                params = child_params;
            }
        }
        println!("{:?} loss: {:?}", i, nil_loss);
        if i % 10 == 0 {
            println!("acc: {:?}%", avg_accuracy(&examples, &params) * 100.0);
        }
    }
}
