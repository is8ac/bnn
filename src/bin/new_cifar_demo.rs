extern crate bitnn;
extern crate rayon;
extern crate time;
use rayon::prelude::*;

use bitnn::datasets::cifar;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, pack_3x3, pixelmap, unary, Layer2d, Patch};
use time::PreciseTime;

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

fn nzip<X: Clone, Y: Clone>(a_vec: &Vec<Vec<X>>, b_vec: &Vec<Vec<Y>>) -> Vec<Vec<(X, Y)>> {
    a_vec
        .iter()
        .zip(b_vec.iter())
        .map(|(a, b)| a.iter().cloned().zip(b.iter().cloned()).collect())
        .collect()
}

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let size: usize = 10000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images = Vec::with_capacity(size * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, size);
        images.append(&mut batch);
    }
    images
}

fn conv_layer<
    A: Patch + Copy + Default + Send + Sync,
    I: Layer2d<A> + Send + Sync,
    B: Patch + Copy + Default,
    O: Layer2d<B> + Send + Sync,
>(
    images: &Vec<Vec<I>>,
    n_features_iters: usize,
    unary_size: usize,
    culling_threshold: usize,
) -> Vec<Vec<O>> {
    let start = PreciseTime::now();
    let patches: Vec<Vec<_>> = images
        .par_iter()
        .map(|imgs| imgs.iter().map(|image| image.to_3x3_patches()).flatten().collect())
        .collect();

    let (features_vec, thresholds) = featuregen::gen_hidden_features(&patches, n_features_iters, unary_size, culling_threshold);
    println!("{:?} * {:?} = {:?}", features_vec.len(), A::bit_len(), features_vec.len() * A::bit_len());
    let output = images
        .par_iter()
        .map(|class| {
            class
                .iter()
                .map(|image| {
                    O::from_pixels_1_padding(
                        &image
                            .to_3x3_patches()
                            .iter()
                            .map(|&pixels| featuregen::apply_unary(&pixels, &features_vec, &thresholds))
                            .collect(),
                    )
                }).collect()
        }).collect();
    println!("{}", start.to(PreciseTime::now()));
    output
}

//let start = PreciseTime::now();
//println!("{} seconds just bitpack", start.to(PreciseTime::now()));

fn main() {
    let examples = load_data();
    let start = PreciseTime::now();
    let u14_packed_images: Vec<(usize, [[u16; 32]; 32])> = examples
        .par_iter()
        .map(|(label, image)| (*label, pixelmap::pm_32(&image, &unary::rgb_to_u14)))
        .collect();
    let l0_images = featuregen::split_by_label(&u14_packed_images, 10);

    let l1_images = conv_layer::<_, _, u128, _>(&l0_images, 3, 4, 20);
    let l1_pooled_images: Vec<Vec<[[_; 16]; 16]>> = l1_images
        .par_iter()
        .map(|class| class.iter().map(|image| pixelmap::pool_or_32(image)).collect())
        .collect();

    let l2_images = conv_layer::<_, _, [u128; 2], [[_; 16]; 16]>(&l1_pooled_images, 4, 6, 7);
    //let l2_images = conv_layer::<_, _, [u128; 3], [[_; 16]; 16]>(&l2_images, 5, 6, 7);
    //let l2_images = conv_layer::<_, _, [u128; 3], _>(&l2_images, 6, 6, 7);
    let l2_pooled_images: Vec<Vec<[[_; 8]; 8]>> = l2_images
        .par_iter()
        .map(|class| class.iter().map(|image| pixelmap::pool_or_16(image)).collect())
        .collect();

    let l3_images = conv_layer::<_, _, [u128; 3], [[_; 8]; 8]>(&l2_pooled_images, 5, 5, 3);
    //let l3_images = conv_layer::<_, _, [u128; 4], _>(&l3_images, 6, 6, 3);
    let l3_pooled_images: Vec<Vec<[[_; 4]; 4]>> = l3_images
        .par_iter()
        .map(|class| class.iter().map(|image| pixelmap::pool_or_8(image)).collect())
        .collect();
    let l3_flat_images: Vec<Vec<[_; 16]>> = l3_pooled_images
        .par_iter()
        .map(|class| class.iter().map(|image| pack_3x3::flatten_4x4(&image)).collect())
        .collect();

    let l3_readout_features = featuregen::gen_readout_features(&l3_flat_images);
    let biases = bias(&l3_readout_features, &l3_flat_images);
    let acc = eval_acc(&l3_readout_features, &l3_flat_images, &biases);
    println!("l3 acc: {:?}%", acc * 100.0);

    let featuregen_start = PreciseTime::now();
    let (l4_features_vec, l4_thresholds) = featuregen::gen_hidden_features(&l3_flat_images, 5, 6, 2);
    println!("{:?} * {:?} = {:?}", l4_features_vec.len(), <[[u128; 3]; 16]>::bit_len(), l4_features_vec.len() * <[[u128; 3]; 16]>::bit_len());
    println!("l4 featuregen: {}", featuregen_start.to(PreciseTime::now()));

    let l4_train_inputs = apply_features_fc::<_, [u128; 3]>(&l3_flat_images, &l4_features_vec, &l4_thresholds);
    let l4_readout_features = featuregen::gen_readout_features(&l4_train_inputs);
    let biases = bias(&l4_readout_features, &l4_train_inputs);
    let acc = eval_acc(&l4_readout_features, &l4_train_inputs, &biases);
    println!("acc: {:?}%", acc * 100.0);
    println!("{} seconds all", start.to(PreciseTime::now()));
}
// 32%
// PT40.8
// 153.9
