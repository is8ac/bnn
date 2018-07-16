#[macro_use]
extern crate bitnn;
extern crate rand;

use bitnn::datasets::mnist;

const TRAINING_SIZE: usize = 1000;
const image_size: usize = 28;

xor_conv3x3_onechan_pooled!(conv_onechan, image_size, image_size, 1);
xor_conv3x3_onechan_pooled_grads_update!(grads_update, image_size, image_size, 1);
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

fn main() {
    const CHAN_0: usize = 1;
    //const CHAN_1: usize = 10;
    let images = mnist::load_images_64chan(&String::from("mnist/train-images-idx3-ubyte"), TRAINING_SIZE);
    let labels = mnist::load_labels(&String::from("mnist/train-labels-idx1-ubyte"), TRAINING_SIZE);

    let split_examples: Vec<Vec<&[[[u64; 1]; 28]; 28]>> = (0..10)
        .map(|i| {
            images
                .iter()
                .zip(labels.iter())
                .filter(|(_, &label)| label == i)
                .map(|(image, _)| image)
                .collect()
        })
        .collect();

    let grads: Vec<[[[[f32; 64]; CHAN_0]; 3]; 3]> = split_examples
        .iter()
        .map(|l| {
            div_4d!(l.len() as f32, 3, 3, 1, 64)(&l.iter().fold([[[[0f32; 64]; CHAN_0]; 3]; 3], |mut grads, e| {
                grads_update(&e, &mut grads);
                grads
            }))
        })
        .collect();
    println!("grads: {:?}", grads[5][2][2][0][50]);
    let avg_grads = div_4d!(grads.len() as f32, 3, 3, 1, 64)(&sum_4d!(f32, 3, 3, 1, 64)(&grads));

    let params: Vec<[[[u64; 1]; 3]; 3]> = grads
        .iter()
        .map(|label_grads| bitpack(&sub_i32_4d!(f32, 3, 3, 1, 64)(label_grads, &avg_grads)))
        .collect();

    let filtered_images: Vec<&[[[u64; 1]; 28]; 28]> = images
        .iter()
        .zip(labels.iter())
        .filter(|(image, &label)| conv_onechan(&image, &params[0]) > 50000)
        .map(|(image, _)| image)
        .collect();

    println!("activation: {:?}", conv_onechan(&images[0], &params[0]));
    println!("fill len: {:?}", images.len());
    println!("filtered len: {:?}", filtered_images.len());

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
