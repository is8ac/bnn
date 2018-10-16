extern crate bitnn;
extern crate rayon;
use rayon::prelude::*;

use bitnn::datasets::mnist;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, Patch};

fn split_by_label<T: Copy>(examples: &Vec<(usize, T)>, len: usize) -> Vec<Vec<T>> {
    let mut by_label: Vec<Vec<T>> = (0..len).map(|_| Vec::new()).collect();
    for (label, example) in examples {
        by_label[*label].push(*example);
    }
    let _: Vec<_> = by_label.iter_mut().map(|x| x.shrink_to_fit()).collect();
    by_label
}

fn apply_unary<T: Patch, O: Patch>(input: &T, features_vec: &Vec<(T, T)>, thresholds: &Vec<Vec<u32>>) -> O {
    let distances = bitvecmul::vmbvm(&features_vec, &input);
    let mut bools = vec![false; O::bit_len()];
    for c in 0..distances.len() {
        for i in 0..thresholds[0].len() {
            bools[(c * thresholds[0].len()) + i] = distances[c] > thresholds[c][i];
        }
    }
    O::bitpack(&bools.as_slice())
}

fn gen_readout_features<T: Patch + Sync + Clone + Default>(by_class: &Vec<Vec<T>>) -> Vec<(T, T)> {
    (0..by_class.len())
        .map(|class| {
            let grads = featuregen::grads_one_shard(by_class, class);
            let (base_point, mask) = featuregen::grads_to_bits(&grads, 0.0);
            (base_point, mask)
        }).collect()
}

fn eval_acc<T: Patch>(readout_features: &Vec<(T, T)>, test_inputs: &Vec<Vec<T>>) -> f64 {
    let num_examples: usize = test_inputs.iter().map(|x| x.len()).sum();
    let test_correct: u64 = test_inputs
        .iter()
        .enumerate()
        .map(|x| {
            x.1.iter()
                .map(move |input| {
                    x.0 == readout_features
                        .iter()
                        .map(|(base_point, mask)| input.masked_hamming_distance(&base_point, &mask))
                        .enumerate()
                        .max_by_key(|(_, dist)| *dist)
                        .unwrap()
                        .0
                }).map(|x| x as u64)
        }).flatten()
        .sum();
    test_correct as f64 / num_examples as f64
}

// - (Vec<usize>, Vec<T>)
// - Vec<(usize, T)>
// - Vec<Vec<T>>
// - Vec<Vec<Vec<T>>>

fn gen_hidden_features<T: Patch + Copy + Sync + Default + Send, O: Patch + Copy + Sync + Send + Default>(
    train_inputs: &Vec<Vec<T>>,
    n_features_iters: usize,
    unary_size: usize,
    test_inputs: &Vec<Vec<T>>,
) -> (Vec<Vec<O>>, Vec<Vec<O>>) {
    let flat_inputs: Vec<T> = train_inputs.iter().flatten().cloned().collect();
    let mut shards: Vec<Vec<Vec<T>>> = vec![train_inputs.clone().to_owned()];
    let mut features_vec: Vec<(T, T)> = vec![];
    for i in 0..n_features_iters {
        for class in 0..10 {
            let (base_point, mask) = featuregen::gen_basepoint(&shards, 0.0, class);
            //let mask = nil_mask;
            features_vec.push((base_point, mask));
            let split_threshold = featuregen::gen_threshold_masked(&flat_inputs, &base_point, &mask);
            shards = featuregen::split_labels_set_by_distance(&shards, &base_point, &mask, split_threshold, 3);
            //println!("{:?} \t {:?}", shards.len(), class);
        }
    }
    let thresholds: Vec<Vec<u32>> = features_vec
        .par_iter()
        .map(|(base, mask)| featuregen::vec_threshold(&flat_inputs, &base, &mask, unary_size))
        .collect();
    let new_inputs: Vec<Vec<O>> = train_inputs
        .par_iter()
        .map(|x| x.iter().map(|input| apply_unary(input, &features_vec, &thresholds)).collect())
        .collect();
    let readout_features = gen_readout_features(&new_inputs);

    let new_test_inputs: Vec<Vec<O>> = test_inputs
        .par_iter()
        .map(|x| x.iter().map(|input| apply_unary(input, &features_vec, &thresholds)).collect())
        .collect();
    let acc = eval_acc(&readout_features, &new_test_inputs);
    println!("acc: {:?}%", acc * 100.0);
    (new_inputs, new_test_inputs)
}

const TRAIN_SIZE: usize = 60_000;

fn nzip<X: Clone, Y: Clone>(a_vec: &Vec<Vec<X>>, b_vec: &Vec<Vec<Y>>) -> Vec<Vec<(X, Y)>> {
    a_vec.iter().zip(b_vec.iter()).map(|(a, b)| a.iter().cloned().zip(b.iter().cloned()).collect()).collect()
}

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, [u64; 13])> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let l0_train_inputs = split_by_label(&examples, 10);

    let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
    let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
    let test_examples: Vec<(usize, [u64; 13])> = test_labels.iter().zip(test_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let l0_test_inputs = split_by_label(&test_examples, 10);

    let (l1_train_inputs, l1_test_inputs) = gen_hidden_features::<_, [u128; 2]>(&l0_train_inputs, 4, 6, &l0_test_inputs);
    let l1_train_inputs = nzip(&l0_train_inputs, &l1_train_inputs);
    let l1_test_inputs = nzip(&l0_test_inputs, &l1_test_inputs);
    let (l2_train_inputs, l2_test_inputs) = gen_hidden_features::<_, [u128; 3]>(&l1_train_inputs, 3, 5, &l1_test_inputs);
    let l2_train_inputs = nzip(&l1_train_inputs, &l2_train_inputs);
    let l2_test_inputs = nzip(&l1_test_inputs, &l2_test_inputs);
    let (_l3_train_inputs, _l3_test_inputs) = gen_hidden_features::<_, [u128; 3]>(&l2_train_inputs, 3, 5, &l2_test_inputs);
}
