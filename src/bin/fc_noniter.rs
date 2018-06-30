#[macro_use]
extern crate bitnn;
extern crate rand;

use bitnn::datasets::mnist;
use std::time::SystemTime;

const TRAINING_SIZE: usize = 1000;

binary_conv3x3!(conv, 28, 28, 1, 1);
binary_conv3x3_partial!(conv_update, 28, 28, 1, 1);
//binary_conv3x3_dummy!(conv_dummy, 28, 28, 1);
fc_3dbits2ints!(fc, 26, 26, 1, 10);
fc_3dbits2ints_grads2params!(grads2params, 26, 26, 1, 10);
fc_3dbits2ints_grads!(grads, 26, 26, 1, 10);
fc_3dbits2ints_grads_update!(update_grads, 26, 26, 1, 10);
partial_fc_3dbits2ints_grads_zero!(zero_grads, 26, 26, 1, 10);
fc_3dbits2ints_grads_loss!(grads_loss, 26, 26, 1, 10);

fn max_index(input: &[u32; 10]) -> usize {
    let mut index = 0;
    let mut max = 0;
    for i in 0..10 {
        if input[i] > max {
            max = input[i];
            index = i;
        }
    }
    index
}

fn main() {
    let images = mnist::load_images_64chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);

    let mut conv_params = random_prms_3d!(1 * 64, 9, 1);

    let start = SystemTime::now();
    let mut convolved_images: Vec<[[[u64; 1]; 26]; 26]> = images.iter().map(|x| conv(&x, &conv_params)).collect();
    //let mut convolved_images: Vec<[[[u64; 1]; 26]; 26]> = images.iter().map(|x| conv_dummy(&x)).collect();
    println!("conv time: {:?}", start.elapsed().unwrap());

    let mut pos_grads = [[[[[0i32; 64]; 1]; 26]; 26]; 10];
    let mut neg_grads = [[[[[0i32; 64]; 1]; 26]; 26]; 10];

    let start = SystemTime::now();
    for e in 0..TRAINING_SIZE {
        grads(&convolved_images[e], &mut pos_grads, &mut neg_grads, labels[e]);
    }
    let mut nil_min_loss = grads_loss(&pos_grads, &neg_grads);
    println!("nil min loss: {:?}", nil_min_loss as f32 / TRAINING_SIZE as f32);

    for ow in 0..1 {
        for ob in 0..64 {
            let start = SystemTime::now();
            let o = ow * 64 + ob;
            for i in 0..1 {
                for ib in 0..64 {
                    for p in 0..9 {
                        // mutate a bit of the params.
                        conv_params[o][p][i] = conv_params[o][p][i] ^ (0b1u64 << ib);
                        // zero the relevant part of the fc grads.
                        zero_grads(&mut pos_grads, ow, ob);
                        zero_grads(&mut neg_grads, ow, ob);
                        for e in 0..TRAINING_SIZE {
                            // for each example,
                            // first update the convolved_images with the new params,
                            conv_update(&images[e], &conv_params, &mut convolved_images[e], ow, ob);
                            // then update the grads
                            update_grads(&convolved_images[e], &mut pos_grads, &mut neg_grads, labels[e], ow, ob);
                        }
                        // once the grads have been updated, we can calculate the loss.
                        let new_min_loss = grads_loss(&pos_grads, &neg_grads);
                        if new_min_loss <= nil_min_loss {
                            nil_min_loss = new_min_loss;
                            println!("{:?} keeping: {:?} < {:?}", ob, nil_min_loss as f32 / TRAINING_SIZE as f32, nil_min_loss as f32 / TRAINING_SIZE as f32);
                        } else {
                            println!(
                                "{:?} rejecting: {:?} !< {:?}",
                                ob,
                                new_min_loss as f32 / TRAINING_SIZE as f32,
                                nil_min_loss as f32 / TRAINING_SIZE as f32
                            );
                            // if the change increased loss, revert it.
                            conv_params[o][p][i] = conv_params[o][p][i] ^ (0b1u64 << ib);
                        }
                    }
                }
            }
            println!("one bit time: {:?}", start.elapsed().unwrap());
            // accuracy
            let fc_params = grads2params(&pos_grads, &neg_grads);
            //println!("fc params: {:640b}", fc_params[5][5][7][0]);
            let sum_correct: i64 = convolved_images
                .iter()
                .map(|x| fc(x, &fc_params))
                .zip(labels.iter())
                .map(|(actual, &target)| (max_index(&actual) == target) as i64)
                .sum();
            println!("avg_acc: {:?}%", sum_correct as f32 / TRAINING_SIZE as f32 * 100f32);
        }
    }
    //println!("min loss: {:?}", new_min_loss as f32 / TRAINING_SIZE as f32);
}
