#[macro_use]
extern crate bitnn;
extern crate rand;
extern crate time;

use bitnn::datasets::mnist;

fn dense_goodness(
    &params: &([[u64; 13]; 2 * 64], [[u64; 2]; 10]),
    &(input, target): &([u64; 13], usize),
) -> i64 {
    let mut hidden = [0u64; 2];
    dense_bits2bits!(13, 2)(&mut hidden, &params.0, &input);
    let mut output = [0u32; 10];
    dense_bits2ints!(2, 10)(&mut output, &params.1, &hidden);
    output[target] as i64
}

fn dense_acc(
    &params: &([[u64; 13]; 2 * 64], [[u64; 2]; 10]),
    &(input, target): &([u64; 13], usize),
) -> i64 {
    let mut hidden = [0u64; 2];
    dense_bits2bits!(13, 2)(&mut hidden, &params.0, &input);
    let mut output = [0u32; 10];
    dense_bits2ints!(2, 10)(&mut output, &params.1, &hidden);
    correct!(output, target) as i64
}

const TRAINING_SIZE: usize = 60000;

fn main() {
    let training_size = TRAINING_SIZE;
    let test_size = 10000;
    let goodness_size = 1000;

    println!("loading data");
    let images = mnist::load_images_bitpacked(
        &String::from("mnist/train-images-idx3-ubyte"),
        training_size,
    );
    let labels = mnist::load_labels(
        &String::from("mnist/train-labels-idx1-ubyte"),
        training_size,
    );
    //let test_images =
    //    mnist::load_images_bitpacked(&String::from("mnist/train-images-idx3-ubyte"), test_size);
    //let test_labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), test_size);
    //let test_examples: Vec<([u64; 13], usize)> = test_images
    //    .iter()
    //    .zip(test_labels.iter())
    //    .map(|x| (*x.0, *x.1))
    //    .collect();
    //let mut test_set = test_examples.iter().cycle();

    //let mut examples: Vec<([u64; 13], usize)> = images
    //    .iter()
    //    .zip(labels.iter())
    //    .map(|x| (*x.0, *x.1))
    //    .collect();
    //let mut rng = rand::thread_rng();
    //rng.shuffle(&mut examples);
    //let mut training_set = examples.iter().cycle();

    println!("initializing params"); //goodness!(outputs, target)

    let mut params = ([[0u64; 13]; 2 * 64], [[0u64; 2]; 10]);
    for o in 0..2 * 64 {
        for i in 0..13 {
            params.0[o][i] = rand::random::<u64>()
        }
    }
    for o in 0..10 {
        for i in 0..2 {
            params.1[o][i] = rand::random::<u64>()
        }
    }
    let layer_1 = dense_bits2bits!(13, 2);
    let layer1_grad = dense_bits2bits_grad!(13, 2);
    //let state_1 = [u64; 2];
    let layer_2 = dense_bits2ints!(2, 10);
    let layer2_grad = dense_bits2ints_grad!(2, 10);
    //let state_2 = [u32; 10];
    //println!("images.len(): {:?}", images.len());
    println!("allocing");
    let mut outputs_weights = [0i32; 13 * 64];
    //for e in 0..TRAINING_SIZE {
    let e = 0;
    let mut grads = [0i8; 10];
    grads[labels[e]] = 1;
    let mut state1 = [0u64; 2];
    layer_1(&mut state1, &params.0, &images[e]);
    let grads2 = layer2_grad(&grads, &params.1, &state1);
    let grads1 = layer1_grad(&grads2, &params.0, &images[e]);
    println!("grads1: {:?}", grads1[0]);
    println!("grads2: {:?}", grads2[0]);
    //}
}
