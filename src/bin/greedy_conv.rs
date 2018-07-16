#[macro_use]
extern crate bitnn;
extern crate rand;

use bitnn::datasets::cifar;

const TRAINING_SIZE: usize = 1000;
const IMAGE_SIZE: usize = 32;
const CHAN_0: usize = 1;

xor_conv3x3_onechan_pooled!(conv_onechan, IMAGE_SIZE, IMAGE_SIZE, 1);
xor_conv3x3_onechan_pooled_grads_update!(grads_update, IMAGE_SIZE, IMAGE_SIZE, 1);
bitpack_u64_3d!(bitpack, 3, 3, 1, 0f32);

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

//let start = SystemTime::now();
//println!("conv time: {:?}", start.elapsed().unwrap());

fn calc_params_diffs(images: &Vec<(u8, [[[u64; 1]; IMAGE_SIZE]; IMAGE_SIZE])>, target_label: u8) -> [[[u64; 1]; 3]; 3] {
    let zero_grads = {
        let (grad_sum, len) = images
            .iter()
            .filter(|(label, _)| *label == target_label)
            .map(|(_, image)| image)
            .fold(([[[[0f32; 64]; CHAN_0]; 3]; 3], 0), |mut grads, e| {
                grads_update(&e, &mut grads.0);
                grads.1 += 1;
                grads
            });
        div_4d!(len as f32, 3, 3, 1, 64)(&grad_sum)
    };

    let non_zero_grads = {
        let (grad_sum, len) = images
            .iter()
            .filter(|(label, _)| *label != target_label)
            .map(|(_, image)| image)
            .fold(([[[[0f32; 64]; CHAN_0]; 3]; 3], 0), |mut grads, e| {
                grads_update(&e, &mut grads.0);
                grads.1 += 1;
                grads
            });
        div_4d!(len as f32, 3, 3, 1, 64)(&grad_sum)
    };

    let diffs = sub_i32_4d!(f32, 3, 3, 1, 64)(&zero_grads, &non_zero_grads);
    bitpack(&diffs)
}

fn main() {
    //const CHAN_1: usize = 10;
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
    println!("diffs:   {:?}", filters);

    let filters2: Vec<[[[u64; 1]; 3]; 3]> = (0..10)
        .map(|target_label| calc_params_diffs(&images, target_label))
        .collect();

    //println!("params: {:?}", params);
    //println!("{:064b}", params[0][0][0]);
    //println!("{:064b}", params[0][1][0]);
    let sum_activations: u32 = images.iter().map(|(_, image)| conv_onechan(image, &filters[0])).sum();
    println!("avg activation: {:?}", sum_activations as f32 / images.len() as f32);
    println!("are equal: {:?}", filters == filters2);
    for l in 0..10 {
        println!("{:?}", l);
        for px in 0..3 {
            for py in 0..3 {
                if filters[l][px][py][0] != filters2[l][px][py][0] {
                    println!("diff:", );
                    println!("1: {:064b}", filters[l][px][py][0]);
                    println!("2: {:064b}", filters2[l][px][py][0]);
                }
            }
        }
    }

    //let sum_activations = images
    //    .iter()
    //    .zip(labels.iter())
    //    .filter(|(_, &label)| label == 0);

    //let params: Vec<[[[u64; 1]; 3]; 3]> = grads
    //    .iter()
    //    .map(|label_grads| bitpack(&sub_i32_4d!(f32, 3, 3, 1, 64)(label_grads, &avg_grads)))
    //    .collect();

    //let filtered_images: Vec<&[[[u64; 1]; 28]; 28]> = images
    //    .iter()
    //    .zip(labels.iter())
    //    .filter(|(image, &label)| conv_onechan(&image, &params[0]) > 50000)
    //    .map(|(image, _)| image)
    //    .collect();

    //println!("activation: {:?}", conv_onechan(&images[0], &params[0]));
    //println!("fill len: {:?}", images.len());
    //println!("filtered len: {:?}", filtered_images.len());

    //println!("zeros params: {:?}", zeros_params);
    //println!("{:064b}", zeros_params[0][0][0]);

    //let ones_diffs = sub_i32_4d!(3, 3, 1, 64)(&zeros_grads, &ones_grads);
    //let ones_params = bitpack(&ones_diffs);
    //println!("ones params: {:?}", ones_params);

    //println!("zeros zeros activation: {:?}", conv_onechan(&zeros[0], &zeros_params));
    //println!("zeros ones  activation: {:?}", conv_onechan(&zeros[0], &ones_params));
    //println!("ones  ones  activation: {:?}", conv_onechan(&ones[0], &ones_params));
    //println!("ones  zeros activation: {:?}", conv_onechan(&ones[0], &zeros_params));
}
