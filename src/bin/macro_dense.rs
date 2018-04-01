#[macro_use]
extern crate bitnn;
extern crate rand;
extern crate time;

use rand::distributions::{IndependentSample, Range};
use rand::Rng;
use bitnn::datasets::mnist;

fn dense_goodness(&params: &[[u64; 13]; 10], &(input, target): &([u64; 13], usize)) -> i64 {
    let mut outputs = [0u32; 10];
    dense_bits2ints!(13, 10)(&mut outputs, &params, &input);
    outputs[target] as i64
}

fn dense_acc(&params: &[[u64; 13]; 10], &(input, target): &([u64; 13], usize)) -> i64 {
    let mut output = [0u32; 10];
    dense_bits2ints!(13, 10)(&mut output, &params, &input);
    correct!(output, target) as i64
}

fn main() {
    let training_size = 60000;
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
    let test_images =
        mnist::load_images_bitpacked(&String::from("mnist/train-images-idx3-ubyte"), test_size);
    let test_labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), test_size);
    let test_examples: Vec<([u64; 13], usize)> = test_images
        .iter()
        .zip(test_labels.iter())
        .map(|x| (*x.0, *x.1))
        .collect();
    let mut test_set = test_examples.iter().cycle();

    let mut examples: Vec<([u64; 13], usize)> = images
        .iter()
        .zip(labels.iter())
        .map(|x| (*x.0, *x.1))
        .collect();
    let mut rng = rand::thread_rng();
    rng.shuffle(&mut examples);
    let mut training_set = examples.iter().cycle();

    println!("initializing params"); //goodness!(outputs, target)

    let mut params = [[0u64; 13]; 10];
    for o in 0..10 {
        for i in 0..13 {
            params[o][i] = rand::random::<u64>()
        }
    }
    println!("data loaded, starting first avg");
    let mut avg_goodness = average!(
        training_set,
        wrap_params!(&params, dense_goodness, &([u64; 13], usize)),
        goodness_size
    );
    println!("avg goodness: {:?}", avg_goodness);

    //let accuracy = test_set.by_ref().take(test_size).fold(0u64, |sum, x| {
    //    sum + (dense_infer(&params, &x.0) == x.1) as u64
    //}) as f64 / test_size as f64;
    //println!("accuracy: {:?}%", accuracy * 100.0);

    let image_input_range = Range::new(0, 13);
    let target_range = Range::new(0, 10);
    for iter in 0..10000 {
        let i: usize = image_input_range.ind_sample(&mut rng);
        let o: usize = target_range.ind_sample(&mut rng);
        optimize_word_bits!(
            params[o][i],
            avg_goodness,
            average!(
                training_set,
                wrap_params!(&params, dense_goodness, &([u64; 13], usize)),
                goodness_size
            )
        );
        if iter % 10 == 0 {
            println!("iter: {:?} {:?}", iter, avg_goodness);
            let accuracy = average!(
                test_set,
                wrap_params!(&params, dense_acc, &([u64; 13], usize)),
                goodness_size
            );
            println!("accuracy: {:?}%", accuracy * 100.0);
        }
    }
}
