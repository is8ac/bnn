extern crate rand;
extern crate time;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::time::SystemTime;
use rand::distributions::{IndependentSample, Range};
use rand::Rng;

fn load_labels(path: &String, size: usize) -> Vec<usize> {
    let path = Path::new(path);
    let mut file = File::open(&path).expect("can't open images");
    let mut header: [u8; 8] = [0; 8];
    file.read_exact(&mut header).expect("can't read header");

    let mut byte: [u8; 1] = [0; 1];
    let mut labels: Vec<usize> = Vec::new();
    for _ in 0..size {
        file.read_exact(&mut byte).expect("can't read label");
        labels.push(byte[0] as usize);
    }
    return labels;
}

fn load_images(path: &String, size: usize) -> Vec<[[u64; 28]; 28]> {
    let path = Path::new(path);
    let mut file = File::open(&path).expect("can't open images");
    let mut header: [u8; 16] = [0; 16];
    file.read_exact(&mut header).expect("can't read header");

    let mut images_bytes: [u8; 784] = [0; 784];

    // bitpack the image into 13 64 bit words.
    // There will be unused space in the last word, this is acceptable.
    // the bits of each words will be in revere order,
    // rev() the slice before use if you want them in the correct order.
    let mut images: Vec<[[u64; 28]; 28]> = Vec::new();
    for _ in 0..size {
        file.read_exact(&mut images_bytes)
            .expect("can't read images");
        let mut image = [[0u64; 28]; 28];
        for p in 0..784 {
            let mut ones = 0u64;
            for i in 0..images_bytes[p] / 4 {
                ones = ones | 1 << i;
            }
            image[p / 28][p % 28] = ones;
        }
        images.push(image);
    }
    return images;
}

fn bit_perturbations() -> [u64; 64] {
    let mut words = [0u64; 64];
    for i in 0..64 {
        words[i] = 1 << i;
    }
    return words;
}

macro_rules! conv {
    ($x_size:expr, $y_size:expr, $strides:expr) => (
        |input: &[[u64; $x_size]; $y_size],
        weights: &[[u64; 9]; 64]| ->
        [[u64; ($x_size - 2) / $strides.0]; ($y_size - 2) / $strides.1] {
            let mut output = [[0u64; ($x_size - 2) / $strides.0]; ($y_size - 2) / $strides.1];
            for x in 0..($x_size - 2) / $strides.0 {
                for y in 0..($y_size - 2) / $strides.1 {
                    for chan in 0..64 {
        let sum = (weights[chan][0] ^ input[x * $strides.0 + 0][y * $strides.1 + 0]).count_ones()
                + (weights[chan][1] ^ input[x * $strides.0 + 1][y * $strides.1 + 0]).count_ones()
                + (weights[chan][2] ^ input[x * $strides.0 + 2][y * $strides.1 + 0]).count_ones()
                + (weights[chan][3] ^ input[x * $strides.0 + 0][y * $strides.1 + 1]).count_ones()
                + (weights[chan][4] ^ input[x * $strides.0 + 1][y * $strides.1 + 1]).count_ones()
                + (weights[chan][5] ^ input[x * $strides.0 + 2][y * $strides.1 + 1]).count_ones()
                + (weights[chan][6] ^ input[x * $strides.0 + 0][y * $strides.1 + 2]).count_ones()
                + (weights[chan][7] ^ input[x * $strides.0 + 1][y * $strides.1 + 2]).count_ones()
                + (weights[chan][8] ^ input[x * $strides.0 + 2][y * $strides.1 + 2]).count_ones();
                        output[x][y] = output[x][y] | (((sum > 288) as u64) << chan);
                    }
                }
            }
            return output;
        }
    )
}

macro_rules! optimize {
    ($word:expr, $goodness:expr, $goodness_size:expr, $avg_goodness:expr) => (
        for p in 0..64 {
            // for each bit of the word,
            let perturbation = 1 << p;
            $word = $word ^ perturbation;
            let mut total_goodness = 0i64;
            for num_samples in 1..500 {
                let new_goodness = $goodness();
                //println!("avg: {:?}, new: {:?}", avg_goodness, new_goodness);
                total_goodness += new_goodness;
                let delta = (total_goodness as f64 / num_samples as f64) - $avg_goodness;
                let is_good = delta / (1.0 / num_samples as f64);
                //println!("num_samples: {:?}, new_goodness: {:?}, delta: {:?}, is_good {:?}", num_samples, new_goodness, delta, is_good);
                if is_good < -100.0 {
                    $word = $word ^ perturbation; // revert
                    //println!("reverting");
                    break;
                }
                if is_good > 100.0 {
                    $avg_goodness = ($avg_goodness * $goodness_size as f64 + total_goodness as f64)
                        / ($goodness_size as f64 + num_samples as f64);
                    println!("keeping");
                    break;
                }
            }
        }
    )
}

fn convnet(
    &(conv0, conv1, conv2, dense0): &(
        [[u64; 9]; 64],
        [[u64; 9]; 64],
        [[u64; 9]; 64],
        [[[u64; 5]; 5]; 10],
    ),
    &(input, target): &([[u64; 28]; 28], usize),
) -> i64 {
    let mut goodness = 0i64;
    let hidden1 = conv!(28, 28, (1, 1))(&input, &conv0);
    let hidden2 = conv!(26, 26, (2, 2))(&hidden1, &conv1);
    let hidden3 = conv!(12, 12, (2, 2))(&hidden2, &conv2);
    for o in 0..10 {
        let mut sum = 0i64;
        if o == target {
            for x in 0..5 {
                for y in 0..5 {
                    sum += (dense0[o][x][y] ^ hidden3[x][y]).count_ones() as i64;
                }
            }
            // println!("+sum: {:?}", sum);
            goodness = goodness + sum;
        }
    }
    //println!("goodness: {:?}", goodness);
    return goodness;
}

fn convnet_infer(
    &(conv0, conv1, conv2, dense0): &(
        [[u64; 9]; 64],
        [[u64; 9]; 64],
        [[u64; 9]; 64],
        [[[u64; 5]; 5]; 10],
    ),
    &input: &[[u64; 28]; 28],
) -> usize {
    let hidden1 = conv!(28, 28, (1, 1))(&input, &conv0);
    let hidden2 = conv!(26, 26, (2, 2))(&hidden1, &conv1);
    let hidden3 = conv!(12, 12, (2, 2))(&hidden2, &conv2);
    let mut top_index: usize = 0;
    let mut top_sum: i32 = 0;
    for o in 0..10 {
        let mut sum = 0i32;
        for x in 0..5 {
            for y in 0..5 {
                sum += (dense0[o][x][y] ^ hidden3[x][y]).count_ones() as i32;
            }
        }
        if sum > top_sum {
            top_index = o;
            top_sum = sum;
        }
    }
    return top_index as usize;
}

fn main() {
    let training_size = 100;
    let test_size = 100;
    //let acc_size = 1000;
    let goodness_size = 500;
    println!("loading data");
    let images = load_images(
        &String::from("mnist/train-images-idx3-ubyte"),
        training_size,
    );
    let labels = load_labels(
        &String::from("mnist/train-labels-idx1-ubyte"),
        training_size,
    );
    let test_images = load_images(&String::from("mnist/train-images-idx3-ubyte"), test_size);
    let test_labels = load_labels(&String::from("mnist/train-labels-idx1-ubyte"), test_size);
    //let mut test_set = test_images.iter().zip(test_labels.iter()).cycle();
    let mut test_set = images.iter().zip(labels.iter()).cycle();

    let mut examples: Vec<([[u64; 28]; 28], usize)> = images
        .iter()
        .zip(labels.iter())
        .map(|x| (*x.0, *x.1))
        .collect();
    let mut rng = rand::thread_rng();
    rng.shuffle(&mut examples);
    let mut training_set = examples.iter().cycle();
    println!("initializing params");
    let mut params = (
        [[0u64; 9]; 64],
        [[0u64; 9]; 64],
        [[0u64; 9]; 64],
        [[[0u64; 5]; 5]; 10],
    );
    for o in 0..64 {
        for i in 0..9 {
            params.0[o][i] = rand::random::<u64>();
            params.1[o][i] = rand::random::<u64>();
            params.2[o][i] = rand::random::<u64>();
        }
    }
    for o in 0..10 {
        for x in 0..5 {
            for y in 0..5 {
                params.3[o][x][y] = rand::random::<u64>()
            }
        }
    }
    println!("data loaded, starting first avg");
    let mut avg_goodness = training_set
        .by_ref()
        .take(goodness_size)
        .fold(0i64, |sum, x| sum + convnet(&params, x)) as f64
        / goodness_size as f64;
    println!("avg goodness: {:?}", avg_goodness);

    let example = test_set.next().unwrap();
    println!(
        "actual: {:?} target: {:?}",
        convnet_infer(&params, &example.0),
        example.1
    );
    let example = test_set.next().unwrap();
    println!(
        "actual: {:?} target: {:?}",
        convnet_infer(&params, &example.0),
        example.1
    );

    let accuracy = test_set.by_ref().take(test_size).fold(0u64, |sum, x| {
        sum + (convnet_infer(&params, &x.0) == *x.1) as u64
    }) as f64 / test_size as f64;
    println!("accuracy: {:?}%", accuracy * 100.0);

    let output_range = Range::new(0, 64);
    let input_range = Range::new(0, 9);
    let image_side_range = Range::new(0, 5);
    let target_range = Range::new(0, 10);
    for mi in 0..100 {
        println!("starting meta iter {:?}", mi);
        for iter in 0..100 {
            let o: usize = output_range.ind_sample(&mut rng);
            let i: usize = input_range.ind_sample(&mut rng);
            optimize!(
                params.0[o][i],
                || convnet(
                    &params,
                    training_set.next().expect("can't get next example"),
                ),
                goodness_size,
                avg_goodness
            );
            println!("iter: {:?} {:?}", iter, avg_goodness);
        }
        let accuracy = test_set.by_ref().take(test_size).fold(0u64, |sum, x| {
            sum + (convnet_infer(&params, &x.0) == *x.1) as u64
        }) as f64 / test_size as f64;
        println!("accuracy: {:?}%", accuracy * 100.0);
        println!("layer 2");

        for iter in 0..100 {
            let o: usize = output_range.ind_sample(&mut rng);
            let i: usize = input_range.ind_sample(&mut rng);
            optimize!(
                params.1[o][i],
                || convnet(
                    &params,
                    training_set.next().expect("can't get next example"),
                ),
                goodness_size,
                avg_goodness
            );
            println!("iter: {:?} {:?}", iter, avg_goodness);
        }
        let accuracy = test_set.by_ref().take(test_size).fold(0u64, |sum, x| {
            sum + (convnet_infer(&params, &x.0) == *x.1) as u64
        }) as f64 / test_size as f64;
        println!("accuracy: {:?}%", accuracy * 100.0);
        println!("layer 3");

        for iter in 0..100 {
            let o: usize = output_range.ind_sample(&mut rng);
            let i: usize = input_range.ind_sample(&mut rng);
            optimize!(
                params.2[o][i],
                || convnet(
                    &params,
                    training_set.next().expect("can't get next example"),
                ),
                goodness_size,
                avg_goodness
            );
            println!("iter: {:?} {:?}", iter, avg_goodness);
        }
        let accuracy = test_set.by_ref().take(test_size).fold(0u64, |sum, x| {
            sum + (convnet_infer(&params, &x.0) == *x.1) as u64
        }) as f64 / test_size as f64;
        println!("accuracy: {:?}%", accuracy * 100.0);

        println!("layer 4");
        for iter in 0..100 {
            let x: usize = image_side_range.ind_sample(&mut rng);
            let y: usize = image_side_range.ind_sample(&mut rng);
            let o: usize = target_range.ind_sample(&mut rng);
            optimize!(
                params.3[o][x][y],
                || convnet(
                    &params,
                    training_set.next().expect("can't get next example"),
                ),
                goodness_size,
                avg_goodness
            );
            println!("iter: {:?} {:?}", iter, avg_goodness);
        }
        let accuracy = test_set.by_ref().take(test_size).fold(0u64, |sum, x| {
            sum + (convnet_infer(&params, &x.0) == *x.1) as u64
        }) as f64 / test_size as f64;
        println!("accuracy: {:?}%", accuracy * 100.0);
    }
}
