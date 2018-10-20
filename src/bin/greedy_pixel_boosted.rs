#[macro_use]
extern crate bitnn;
extern crate time;

use bitnn::datasets::cifar;
use time::PreciseTime;

const TRAINING_SIZE: usize = 10000;
const IMAGE_SIZE: usize = 32;
const CHAN_0: usize = 1;
const CHAN_1: usize = 2;
const NUM_CLASSES: usize = 100;

boosted_grads_3x3!(l1_grads, IMAGE_SIZE, IMAGE_SIZE, CHAN_0, NUM_CLASSES, 20);
bitpack_u64_3d!(l1_bitpack_params, f32, 3, 3, CHAN_0, 0f32);

xor_conv3x3_activations!(l1_activations, IMAGE_SIZE, IMAGE_SIZE, CHAN_0, CHAN_1);
median_activations!(l1_median_activations, IMAGE_SIZE, IMAGE_SIZE, CHAN_0, CHAN_1);
threshold_and_bitpack_image!(l1_image_bitpack, IMAGE_SIZE, IMAGE_SIZE, CHAN_1);

#[macro_export]
macro_rules! eval_acc {
    ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $n_labels:expr) => {
        fn $name(images: &Vec<(u8, [[[u64; $in_chans]; 32]; 32])>, filters: &[[[[u64; $in_chans]; 3]; 3]; $n_labels]) -> f32 {
            xor_conv3x3_onechan_pooled!(conv_onechan, IMAGE_SIZE, IMAGE_SIZE, $in_chans);
            let num_correct: u64 = images
                .iter()
                .map(|(label, image)| {
                    (filters
                        .iter()
                        .map(|filter| conv_onechan(&image, &filter))
                        .enumerate()
                        .max_by_key(|(_, activation)| *activation)
                        .unwrap()
                        .0 as u8
                        == *label) as u64
                }).sum();
            num_correct as f32 / images.len() as f32
        }
    };
}

eval_acc!(l1_eval, 32, 32, CHAN_0, NUM_CLASSES);

macro_rules! bitpack_filter_set {
    ($name:ident, $in_chans:expr, $n_labels:expr) => {
        fn $name(grads: &[[[[[f32; 64]; $in_chans]; 3]; 3]; $n_labels]) -> [[[[u64; $in_chans]; 3]; 3]; $n_labels] {
            let mut filters = [[[[0u64; $in_chans]; 3]; 3]; $n_labels];
            for l in 0..$n_labels {
                for x in 0..3 {
                    for y in 0..3 {
                        for iw in 0..$in_chans {
                            for b in 0..64 {
                                let bit = grads[l][x][y][iw][b] > 0f32;
                                filters[l][x][y][iw] = filters[l][x][y][iw] | ((bit as u64) << b);
                            }
                        }
                    }
                }
            }
            filters
        }
    };
}

bitpack_filter_set!(l1_bitpack, CHAN_0, NUM_CLASSES);

//let start = PreciseTime::now();
//println!("{} seconds just bitpack", start.to(PreciseTime::now()));

fn main() {
    let test_data_path = String::from("/home/isaac/big/cache/datasets/cifar-100-binary/test.bin");
    let test_images = cifar::load_images_64chan_100(&test_data_path, 10000, true);

    let data_path = String::from("/home/isaac/big/cache/datasets/cifar-100-binary/train.bin");
    let images = cifar::load_images_64chan_100(&data_path, TRAINING_SIZE, true);

    let mut all_filter_sets: Vec<[[[[u64; CHAN_0]; 3]; 3]; 100]> = Vec::new();
    for b in 0..3 {
        // genorate grads boosted with past filter sets.
        let grads = l1_grads(&images, &all_filter_sets);
        // then bitpack the grads down to a filter set.
        let filters = l1_bitpack(&grads);
        println!("{:?} acc: {:?}%", b, l1_eval(&test_images, &filters) * 100f32);
        // push the filter set into the vec of all filter sets so that it will boost the next filter set.
        all_filter_sets.push(filters);
    }
    // we need to flatten the Vec of filter sets down to a vec of filters.
    let mut filters: Vec<[[[u64; CHAN_0]; 3]; 3]> = Vec::new();
    for filter_set in all_filter_sets {
        // convert the array of 100 filters to a Vec of filters.
        let mut filters_vec = filter_set.iter().map(|x| *x).collect();
        filters.append(&mut filters_vec);
    }
    // now that we have the filters in a flat Vec, we can sort and deduplicate.
    filters.sort();
    filters.dedup();
    println!("unique filters: {:?}", filters.len());
    // now put them back into an array of filters words, this time with as many as we have channels, not classes.
    let mut l1_filters = [[[[[0u64; CHAN_0]; 3]; 3]; 64]; CHAN_1];
    for cw in 0..CHAN_1 {
        for b in 0..64 {
            l1_filters[cw][b] = filters[(cw * 64) + b];
        }
    }
    // now we must calculate the medians of each channel.
    let activations: Vec<[[[[u32; 64]; CHAN_1]; IMAGE_SIZE]; IMAGE_SIZE]> = images.iter().map(|x| l1_activations(&l1_filters, &x.1)).collect();
    println!("done with activations");

    let thresholds = l1_median_activations(&activations);
    let mut l1_images: Vec<(u8, [[[u64; CHAN_1]; IMAGE_SIZE]; IMAGE_SIZE])> = activations
        .iter()
        .map(|x| l1_image_bitpack(&x, &thresholds))
        .zip(images.iter().map(|(label, _)| label))
        .map(|(image, &label)| (label, image))
        .collect();

    let mut l1_test_images: Vec<(u8, [[[u64; CHAN_1]; IMAGE_SIZE]; IMAGE_SIZE])> = test_images
        .iter()
        .map(|(label, image)| (*label, l1_image_bitpack(&l1_activations(&l1_filters, &image), &thresholds)))
        .collect();
    println!("{:?}", l1_images.len());
    println!("{:?}", l1_test_images.len());
}
