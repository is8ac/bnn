#[macro_use]
extern crate bitnn;

use bitnn::datasets::cifar;

const TRAINING_SIZE: usize = 5000;
const IMAGE_SIZE: usize = 32;
const CHAN_0: usize = 1;
const CHAN_1: usize = 3;
const NUM_CLASSES: usize = 100;

//xor_conv3x3_means!(l1_means, IMAGE_SIZE, IMAGE_SIZE, CHAN_0, CHAN_1);

//xor_conv3x3_onechan_pooled!(l1_conv_onechan, IMAGE_SIZE, IMAGE_SIZE, CHAN_0);
xor_conv3x3_bitpacked!(
    l1_conv_bitpacked,
    IMAGE_SIZE,
    IMAGE_SIZE,
    CHAN_0,
    CHAN_1,
    ((CHAN_0 * 64 * 9) / 2) as u32
);
boosted_grads_3x3!(l1_grads, IMAGE_SIZE, IMAGE_SIZE, CHAN_0, NUM_CLASSES, 20);
bitpack_u64_3d!(l1_bitpack_params, f32, 3, 3, CHAN_0, 0f32);
//xor_conv3x3_max!(l1_max_label, 1, NUM_CLASSES);

//xor_conv3x3_onechan_pooled!(l2_conv_onechan, IMAGE_SIZE, IMAGE_SIZE, CHAN_1);
//xor_conv3x3_bitpacked!(
//    l2_conv_bitpacked,
//    IMAGE_SIZE,
//    IMAGE_SIZE,
//    CHAN_1,
//    CHAN_1,
//    ((CHAN_1 * 64 * 9) / 2) as u32
//);
//boosted_grads_3x3!(l2_grads, IMAGE_SIZE, IMAGE_SIZE, CHAN_1, NUM_CLASSES, 20);
//bitpack_u64_3d!(l2_bitpack_params, f32, 3, 3, CHAN_1, 0f32);
//xor_conv3x3_max!(l2_max_label, 1, NUM_CLASSES);

#[macro_export]
macro_rules! eval_acc {
    ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $n_labels:expr) => {
        fn $name(
            images: &Vec<(u8, [[[u64; $in_chans]; 32]; 32])>,
            filters: &[[[[u64; $in_chans]; 3]; 3]; $n_labels],
        ) -> f32 {
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
                        .0 as u8 == *label) as u64
                })
                .sum();
            num_correct as f32 / images.len() as f32
        }
    };
}

eval_acc!(l1_eval, 32, 32, CHAN_0, NUM_CLASSES);
//eval_acc!(l2_eval, 32, 32, CHAN_1, NUM_CLASSES);

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
//bitpack_filter_set!(l2_bitpack, CHAN_1, NUM_CLASSES);

fn main() {
    let test_data_path = String::from("/home/isaac/big/cache/datasets/cifar-100-binary/test.bin");
    let test_images = cifar::load_images_64chan_100(&test_data_path, 500, true);

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


    //let mut l1_images: Vec<(u8, [[[u64; CHAN_1]; 32]; 32])> = images
    //    .iter()
    //    .map(|(label, image)| (*label, l1_conv_bitpacked(&image, &l1_filters)))
    //    .collect();

    //let mut l1_test_images: Vec<(u8, [[[u64; CHAN_1]; 32]; 32])> = test_images
    //    .iter()
    //    .map(|(label, image)| (*label, l1_conv_bitpacked(&image, &l1_filters)))
    //    .collect();

    //for l in 0..20 {
    //    let mut all_filter_sets: Vec<[[[[u64; CHAN_1]; 3]; 3]; 100]> = Vec::new();
    //    for b in 0..9 {
    //        let grads = l2_grads(&l1_images, &all_filter_sets);
    //        let filters = l2_bitpack(&grads);
    //        println!(
    //            "{:?} {:?} l2 acc: {:?}%",
    //            l,
    //            b,
    //            l2_eval(&l1_test_images, &filters) * 100f32
    //        );
    //        all_filter_sets.push(filters);
    //    }
    //    let mut filters: Vec<[[[u64; CHAN_1]; 3]; 3]> = Vec::new();
    //    for filter_set in all_filter_sets {
    //        let mut filters_vec = filter_set.iter().map(|x| *x).collect();
    //        filters.append(&mut filters_vec);
    //    }
    //    filters.sort();
    //    filters.dedup();
    //    println!("unique filters: {:?}", filters.len());
    //    let mut l1_filters = [[[[[0u64; CHAN_1]; 3]; 3]; 64]; CHAN_1];
    //    for cw in 0..CHAN_1 {
    //        for b in 0..64 {
    //            l1_filters[cw][b] = filters[(cw * 64) + b];
    //        }
    //    }
    //    let mut total_ones = 0;
    //    let mut total_words = 0;
    //    for cw in 0..CHAN_1 {
    //        for b in 0..64 {
    //            for x in 0..3 {
    //                for y in 0..3 {
    //                    for c in 0..CHAN_1 {
    //                        total_words += 1;
    //                        total_ones += l1_filters[cw][b][x][y][c].count_ones();
    //                        //println!("{:064b}", l1_filters[cw][b][x][y][c]);
    //                    }
    //                }
    //            }
    //        }
    //    }
    //    println!("filters: avg_ones: {:?}", total_ones as f32 / total_words as f32);

    //    l1_images = l1_images
    //        .iter()
    //        .map(|(label, image)| (*label, l2_conv_bitpacked(image, &l1_filters)))
    //        .collect();

    //    l1_test_images = l1_test_images
    //        .iter()
    //        .map(|(label, image)| (*label, l2_conv_bitpacked(image, &l1_filters)))
    //        .collect();

    //    let mut total_ones = 0;
    //    let mut total_words = 0;
    //    for i in 0..1000 {
    //        for x in 1..31 {
    //            for y in 1..31 {
    //                for c in 0..CHAN_1 {
    //                    total_words += 1;
    //                    total_ones += l1_images[i].1[x][y][c].count_ones();
    //                    //println!("{:064b}", l1_images[i].1[x][y][c]);
    //                }
    //            }
    //        }
    //    }
    //    println!("images: avg_ones: {:?}", total_ones as f32 / total_words as f32);
    //}
}
