#[macro_use]
extern crate bitnn;
extern crate rand;

use bitnn::datasets::cifar;

const TRAINING_SIZE: usize = 10000;
const IMAGE_SIZE: usize = 32;
const CHAN_0: usize = 1;

xor_conv3x3_onechan_pooled!(conv_onechan, IMAGE_SIZE, IMAGE_SIZE, 1);
xor_conv3x3_onechan_pooled_grads_update!(grads_update, IMAGE_SIZE, IMAGE_SIZE, 1);
bitpack_u64_3d!(bitpack, 3, 3, 1, 0f32);

fn main() {
    let data_path = String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin");
    let images = cifar::load_images_64chan(&data_path, TRAINING_SIZE);
    let split_images: Vec<Vec<&(u8, [[[u64; 1]; 32]; 32])>> = (0u8..10)
        .map(|target_label| images.iter().filter(|(label, _)| *label == target_label).collect())
        .collect();

    let avg_grads: Vec<[[[[f32; 64]; 1]; 3]; 3]> = split_images
        .iter()
        .map(|images| {
            let (grad_sum, len) = images.iter().map(|(_, sub_image)| sub_image).fold(
                ([[[[0f32; 64]; CHAN_0]; 3]; 3], 0),
                |mut grads, e| {
                    grads_update(&e, &mut grads.0);
                    grads.1 += 1;
                    grads
                },
            );
            div_4d!(len as f32, 3, 3, 1, 64)(&grad_sum)
        })
        .collect();

    let global_avg_grads = div_4d!(10f32, 3, 3, 1, 64)(&sum_4d!(f32, 3, 3, 1, 64)(&avg_grads));

    let filters: Vec<[[[u64; 1]; 3]; 3]> = avg_grads
        .iter()
        .map(|grads| bitpack(&sub_i32_4d!(f32, 3, 3, 1, 64)(&grads, &global_avg_grads)))
        .collect();

    let data_path = String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/test_batch.bin");
    let images = cifar::load_images_64chan(&data_path, TRAINING_SIZE);
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
    println!("acc: {:?}%", (num_correct as f32 / images.len() as f32) * 100f32);
}
