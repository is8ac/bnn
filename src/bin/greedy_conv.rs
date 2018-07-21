#[macro_use]
extern crate bitnn;

use bitnn::datasets::cifar;

const TRAINING_SIZE: usize = 5000;
const IMAGE_SIZE: usize = 32;
const CHAN_0: usize = 1;
const CHAN_1: usize = 2;
const NUM_CLASSES: usize = 100;

xor_conv3x3_onechan_pooled!(l1_conv_onechan, IMAGE_SIZE, IMAGE_SIZE, 1);
xor_conv3x3_bitpacked!(l1_conv_bitpacked, IMAGE_SIZE, IMAGE_SIZE, CHAN_0, CHAN_1, 64 * 9 / 2);
bitpack_u64_3d!(l1_bitpack_params, f32, 3, 3, 1, 0f32);

xor_conv3x3_onechan_pooled!(conv_onechan, IMAGE_SIZE, IMAGE_SIZE, 2);
xor_conv3x3_bitpacked!(conv_bitpacked, IMAGE_SIZE, IMAGE_SIZE, CHAN_1, CHAN_1, 2 * 64 * 9 / 2);
bitpack_u64_3d!(bitpack_params, f32, 3, 3, 2, 0f32);

#[macro_export]
macro_rules! grads_no_boost {
    ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr) => {
        fn $name(
            examples: &Vec<(u8, [[[u64; $in_chans]; $y_size]; $x_size])>,
        ) -> (
            [[[[f32; 64]; $in_chans]; 3]; 3],
            Vec<[[[[f32; 64]; $in_chans]; 3]; 3]>,
        ) {
            let mut grads = vec![[[[[0u32; 64]; $in_chans]; 3]; 3]; NUM_CLASSES];
            let mut lens = vec![0u64; NUM_CLASSES];
            for (label, image) in examples.iter() {
                for iw in 0..$in_chans {
                    for ib in 0..64 {
                        for px in 0..3 {
                            for py in 0..3 {
                                let mut sum: u32 = 0;
                                // for each pixel,
                                for x in 0..$x_size - 2 {
                                    for y in 0..$y_size - 2 {
                                        sum += (image[x + px][y + py][iw] as u32 >> ib) & 0b1u32;
                                    }
                                }
                                // now add to the label grads,
                                grads[*label as usize][px][py][iw][ib] += sum;
                                lens[*label as usize] += 1;
                            }
                        }
                    }
                }
            }
            // Now we must scale down.
            let mut scaled_grads = vec![[[[[0f32; 64]; $in_chans]; 3]; 3]; NUM_CLASSES];
            let mut global_scaled_grads = [[[[0f32; 64]; $in_chans]; 3]; 3];
            let sum_len: u64 = lens.iter().sum();
            for px in 0..3 {
                for py in 0..3 {
                    for ic in 0..$in_chans {
                        for b in 0..64 {
                            let mut sum_grad = 0;
                            for l in 0..NUM_CLASSES {
                                sum_grad += grads[l][px][py][ic][b];
                                scaled_grads[l][px][py][ic][b] = grads[l][px][py][ic][b] as f32 / lens[l] as f32;
                            }
                            global_scaled_grads[px][py][ic][b] = sum_grad as f32 / sum_len as f32;
                        }
                    }
                }
            }
            (global_scaled_grads, scaled_grads)
        }
    };
}

grads_no_boost!(l1_avg_grads, IMAGE_SIZE, IMAGE_SIZE, CHAN_0);
grads_no_boost!(avg_grads, IMAGE_SIZE, IMAGE_SIZE, CHAN_1);

#[macro_export]
macro_rules! eval_acc {
    ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr) => {
        fn $name(images: &Vec<(u8, [[[u64; $in_chans]; 32]; 32])>, filters: &Vec<[[[u64; $in_chans]; 3]; 3]>) -> f32 {
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

eval_acc!(l1_eval, 32, 32, 1);
eval_acc!(eval, 32, 32, 2);

fn main() {
    let test_data_path = String::from("/home/isaac/big/cache/datasets/cifar-100-binary/test.bin");
    let test_images = cifar::load_images_64chan_100(&test_data_path, 1000, true);

    let data_path = String::from("/home/isaac/big/cache/datasets/cifar-100-binary/train.bin");
    let images = cifar::load_images_64chan_100(&data_path, TRAINING_SIZE, true);

    let (global_avg_grads, class_avg_grads) = l1_avg_grads(&images);

    let mut filters: Vec<[[[u64; 1]; 3]; 3]> = class_avg_grads
        .iter()
        .map(|grads| l1_bitpack_params(&sub_i32_4d!(f32, 3, 3, 1, 64)(&global_avg_grads, &grads)))
        .collect();

    println!("acc: {:?}%", l1_eval(&test_images, &filters) * 100f32);
    filters.sort();
    filters.dedup();
    println!("len: {:?}", filters.len());
    let mut packed_filters = [[[[[0u64; CHAN_0]; 3]; 3]; 64]; CHAN_1];
    for b in 0..filters.len() {
        packed_filters[b / 64][b % 64] = filters[b];
    }
    let mut images: Vec<(u8, [[[u64; 2]; 32]; 32])> = images
        .iter()
        .map(|(label, image)| (*label, l1_conv_bitpacked(image, &packed_filters)))
        .collect();

    let mut test_images: Vec<(u8, [[[u64; 2]; 32]; 32])> = test_images
        .iter()
        .map(|(label, image)| (*label, l1_conv_bitpacked(image, &packed_filters)))
        .collect();

    for l in 0..8 {
        let (global_avg_grads, class_avg_grads) = avg_grads(&images);

        let mut filters: Vec<[[[u64; 2]; 3]; 3]> = class_avg_grads
            .iter()
            .map(|grads| bitpack_params(&sub_i32_4d!(f32, 3, 3, 2, 64)(&global_avg_grads, &grads)))
            .collect();

        println!("{:?} acc: {:?}%", l, eval(&test_images, &filters) * 100f32);
        filters.sort();
        filters.dedup();
        for filter in filters.iter() {
            println!("");
            for px in 0..3 {
                for py in 0..3 {
                    println!("{:?}x{:?}\t{:064b}", px, py, filter[px][py][0]);
                }
            }
        }
        println!("len: {:?}", filters.len());
        let mut packed_filters = [[[[[0u64; CHAN_1]; 3]; 3]; 64]; CHAN_1];
        for b in 0..filters.len() {
            packed_filters[b / 64][b % 64] = filters[b];
        }
        images = images
            .iter()
            .map(|(label, image)| (*label, conv_bitpacked(&image, &packed_filters)))
            .collect();

        test_images = test_images
            .iter()
            .map(|(label, image)| (*label, conv_bitpacked(&image, &packed_filters)))
            .collect();
    }
}
