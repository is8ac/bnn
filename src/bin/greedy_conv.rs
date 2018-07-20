#[macro_use]
extern crate bitnn;
extern crate rand;

use bitnn::datasets::cifar;

const TRAINING_SIZE: usize = 5000;
const IMAGE_SIZE: usize = 32;
const CHAN_0: usize = 1;
const NUM_CLASSES: usize = 100;

xor_conv3x3_onechan_pooled!(conv_onechan, IMAGE_SIZE, IMAGE_SIZE, 1);
bitpack_u64_3d!(bitpack, 3, 3, 1, 0f32);

#[macro_export]
macro_rules! grads_no_boost {
    ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr) => {
        fn $name(examples: &Vec<(u8, [[[u64; $in_chans]; $y_size]; $x_size])>) -> ([[[[f32; 64]; 1]; 3]; 3], Vec<[[[[f32; 64]; 1]; 3]; 3]>) {
            let mut grads = vec![[[[[0u32; 64]; 1]; 3]; 3]; NUM_CLASSES];
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
            let mut scaled_grads = vec![[[[[0f32; 64]; 1]; 3]; 3]; NUM_CLASSES];
            let mut global_scaled_grads = [[[[0f32; 64]; 1]; 3]; 3];
            let sum_len: u64 = lens.iter().sum();
            for px in 0..3 {
                for py in 0..3 {
                    for ic in 0..1 {
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

grads_no_boost!(l1_grads, IMAGE_SIZE, IMAGE_SIZE, CHAN_0);

fn eval(filters: &Vec<[[[u64; 1]; 3]; 3]>) -> f32 {
    let data_path = String::from("/home/isaac/big/cache/datasets/cifar-100-binary/test.bin");
    let images = cifar::load_images_64chan_100(&data_path, 1000, true);
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

fn main() {
    let data_path = String::from("/home/isaac/big/cache/datasets/cifar-100-binary/train.bin");
    let images = cifar::load_images_64chan_100(&data_path, TRAINING_SIZE, true);

    let (global_avg_grads, avg_grads) = l1_grads(&images);

    let mut filters: Vec<[[[u64; 1]; 3]; 3]> = avg_grads
        .iter()
        .map(|grads| bitpack(&sub_i32_4d!(f32, 3, 3, 1, 64)(&global_avg_grads, &grads)))
        .collect();

    //filters.sort();
    //filters.dedup();

    for filter in filters.iter() {
        println!("");
        for px in 0..3 {
            for py in 0..3 {
                println!("{:?}x{:?} {:064b}", px, py, filter[px][py][0]);
            }
        }
    }
    println!("len: {:?}", filters.len());
    println!("acc: {:?}%", eval(&filters) * 100f32);
}
