extern crate bitnn;
extern crate rayon;
use rayon::prelude::*;

use bitnn::datasets::mnist;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, pack_3x3, pixelmap, unary, Patch};

fn eval_acc<T: Patch>(readout_features: &Vec<T>, test_inputs: &Vec<Vec<T>>, biases: &Vec<i32>) -> f64 {
    let num_examples: usize = test_inputs.iter().map(|x| x.len()).sum();
    let test_correct: u64 = test_inputs
        .iter()
        .enumerate()
        .map(|x| {
            x.1.iter()
                .map(move |input| {
                    x.0 == readout_features
                        .iter()
                        .zip(biases.iter())
                        .map(|(base_point, bias)| input.hamming_distance(&base_point) as i32 - bias)
                        .enumerate()
                        .max_by_key(|(_, dist)| *dist)
                        .unwrap()
                        .0
                }).map(|x| x as u64)
        }).flatten()
        .sum();
    test_correct as f64 / num_examples as f64
}

fn confusion_matrix<T: Patch>(readout_features: &Vec<T>, test_inputs: &Vec<Vec<T>>) {
    let matrix: Vec<(usize, Vec<u64>)> = test_inputs
        .iter()
        .map(|examples| {
            examples.iter().fold((0, vec![0u64; 10]), |mut acc, input| {
                let actual = readout_features
                    .iter()
                    .map(|base_point| input.hamming_distance(&base_point))
                    .enumerate()
                    .max_by_key(|(_, dist)| *dist)
                    .unwrap()
                    .0;
                acc.1[actual] += 1;
                acc.0 += 1;
                acc
            })
        }).collect();
    for (i, (len, class)) in matrix.iter().enumerate() {
        print!("{:}: ", i);
        for &num in class {
            print!("{:.4} | ", num as f64 / *len as f64);
        }
        print!("\n",);
    }
    //println!("{:?}", matrix);
}

fn bias<T: Patch>(readout_features: &Vec<T>, inputs: &Vec<Vec<T>>) -> Vec<i32> {
    let example_len: usize = inputs.iter().map(|x| x.len()).sum();
    let avg_activations: Vec<i32> = inputs
        .iter()
        .flatten()
        .fold(vec![0u32; 10], |acc, input| {
            readout_features
                .iter()
                .zip(acc.iter())
                .map(|(base_point, class_sum)| class_sum + input.hamming_distance(&base_point))
                .collect()
        }).iter()
        .map(|&x| (x as f64 / example_len as f64) as i32)
        .collect();

    avg_activations
}

fn apply_features_fc<T: Patch + Sync + Send, O: Patch + Sync + Send>(
    inputs: &Vec<Vec<T>>,
    features_vec: &Vec<T>,
    thresholds: &Vec<Vec<u32>>,
) -> Vec<Vec<O>> {
    inputs
        .par_iter()
        .map(|x| {
            x.iter()
                .map(|input| featuregen::apply_unary(input, &features_vec, thresholds))
                .collect()
        }).collect()
}

fn false_positives_by_class<T: Patch>(readout_features: &Vec<T>, inputs: &Vec<Vec<T>>, biases: &Vec<i32>) -> Vec<f64> {
    inputs
        .iter()
        .enumerate()
        .map(|x| {
            let sum: u64 =
                x.1.iter()
                    .map(|input| {
                        (x.0 == readout_features
                            .iter()
                            .zip(biases.iter())
                            .map(|(base_point, bias)| input.hamming_distance(&base_point) as i32 - bias)
                            .enumerate()
                            .max_by_key(|(_, dist)| *dist)
                            .unwrap()
                            .0) as u64
                    }).sum();
            sum as f64 / x.1.len() as f64
        }).collect()
}


const TRAIN_SIZE: usize = 60_000;

fn nzip<X: Clone, Y: Clone>(a_vec: &Vec<Vec<X>>, b_vec: &Vec<Vec<Y>>) -> Vec<Vec<(X, Y)>> {
    a_vec
        .iter()
        .zip(b_vec.iter())
        .map(|(a, b)| a.iter().cloned().zip(b.iter().cloned()).collect())
        .collect()
}

fn main() {
    let images = mnist::load_images(
        &String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"),
        TRAIN_SIZE,
    );
    let images: Vec<[u128; 49]> = images
        .iter()
        .map(|image| pack_3x3::array_pack_u8_u128_49(&pack_3x3::flatten_28x28(&pixelmap::pm_28(&image, &unary::to_8))))
        .collect();
    let labels = mnist::load_labels(
        &String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"),
        TRAIN_SIZE,
    );
    let examples: Vec<(usize, [u128; 49])> = labels
        .iter()
        .zip(images.iter())
        .map(|(&label, &image)| (label as usize, image))
        .collect();
    let l0_train_inputs = featuregen::split_by_label(&examples, 10);
    let l0_readout_features = featuregen::gen_readout_features(&l0_train_inputs, 0.0);
    let biases = bias(&l0_readout_features, &l0_train_inputs);
    let acc = eval_acc(&l0_readout_features, &l0_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);

    let (l1_features_vec, l1_thresholds) = featuregen::gen_hidden_features(&l0_train_inputs, 8, 7, 2);
    let l1_train_inputs = apply_features_fc::<_, [u128; 3]>(&l0_train_inputs, &l1_features_vec, &l1_thresholds);
    let l1z_train_inputs = nzip(&l0_train_inputs, &l1_train_inputs);
    let l1_readout_features = featuregen::gen_readout_features(&l1z_train_inputs, 0.0);
    let biases = bias(&l1_readout_features, &l1z_train_inputs);
    let acc = eval_acc(&l1_readout_features, &l1z_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);

    let (l2_features_vec, l2_thresholds) = featuregen::gen_hidden_features(&l1z_train_inputs, 8, 6, 2);
    let l2_train_inputs = apply_features_fc::<_, [u128; 8]>(&l1z_train_inputs, &l2_features_vec, &l2_thresholds);
    let l2z_train_inputs = nzip(&l1z_train_inputs, &l2_train_inputs);
    let l2_readout_features = featuregen::gen_readout_features(&l2z_train_inputs, 0.0);
    let biases = bias(&l2_readout_features, &l2z_train_inputs);
    let acc = eval_acc(&l2_readout_features, &l2z_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);

    let (l3_features_vec, l3_thresholds) = featuregen::gen_hidden_features(&l2z_train_inputs, 8, 6, 2);
    let l3_train_inputs = apply_features_fc::<_, [u128; 8]>(&l2z_train_inputs, &l3_features_vec, &l3_thresholds);
    let l3z_train_inputs = nzip(&l2_train_inputs, &l3_train_inputs);
    let l3_readout_features = featuregen::gen_readout_features(&l3z_train_inputs, 0.0);
    let biases = bias(&l3_readout_features, &l3z_train_inputs);
    let acc = eval_acc(&l3_readout_features, &l3z_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);
}
