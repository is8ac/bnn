#[macro_use]
pub mod optimize {
    /// Optimizes one 64bit word of the params.
    /// $word is a pointer to one word of the params.
    /// $avg_goodness should be a mutable f64.
    /// $goodness should be a function which depends on $word and returns a f64.
    #[macro_export]
    macro_rules! optimize_word_bits {
        ($word:expr, $avg_goodness:expr, $goodness:expr) => (
            for p in 0..64 {
                // for each bit of the word,
                let perturbation = 1 << p;
                $word = $word ^ perturbation;
                let new_goodness =  $goodness;
                if new_goodness > $avg_goodness {
                    //println!("keeping");
                    $avg_goodness = new_goodness;
                } else {
                    //println!("reverting");
                    $word = $word ^ perturbation; // revert
                }
            }
        )
    }

    /// Expands a function which takes a single example and
    /// returns one value into a code which returns the average of n calls.
    #[macro_export]
    macro_rules! average {
        ($training_set:expr, $sample:expr, $n:expr) => (
            $training_set
                .by_ref()
                .take($n)
                .fold(0i64, |sum, x| sum + $sample(x)) as f64
                / $n as f64
        )
    }

    /// Converts a function which takes params and an example and
    /// returns a goodness into a function which just takes the example.
    #[macro_export]
    macro_rules! wrap_params {
        ($params:expr, $value_func:expr, $data_type:ty) => (
            |example: $data_type| -> i64 {
                $value_func($params, example)
            }
        )
    }
}

pub mod datasets {
    pub mod mnist {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;
        pub fn load_labels(path: &String, size: usize) -> Vec<usize> {
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
        pub fn load_images_bitpacked(path: &String, size: usize) -> Vec<[u64; 13]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[u64; 13]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image_words: [u64; 13] = [0; 13];
                for p in 0..784 {
                    let word_index = p / 64;
                    image_words[word_index] =
                        image_words[word_index] | (((images_bytes[p] > 128) as u64) << p % 64);
                }
                images.push(image_words);
            }
            return images;
        }

        pub fn load_images_64chan(path: &String, size: usize) -> Vec<[[u64; 28]; 28]> {
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
    }
    pub mod layers {
        #[macro_export]
        macro_rules! conv2d {
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
        #[macro_export]
        macro_rules! dense_bits2bits {
            ($input_size:expr, $output_size:expr) => (
                |output: &mut [u64; $output_size], weights: &[[u64; $input_size]; $output_size * 64], input: &[u64; $input_size]| {
                    for o in 0..$output_size * 64 {
                        let mut sum = 0;
                        for i in 0..$input_size {
                            sum += (weights[o][i] ^ input[i]).count_ones();
                        }
                        let word_index = o / 64;
                        output[word_index] = output[word_index] |
                        (((sum > $input_size * 64 / 2) as u64) << o % 64);
                    }
                }
            )
        }
        #[macro_export]
        macro_rules! dense_bits2bits_grad {
            ($input_size:expr, $output_size:expr) => (
                |target_grad: &[i8; $output_size * 64], weights: &[[u64; $input_size]; $output_size * 64], input: &[u64; $input_size]| -> [i8; $input_size * 64] {
                    let threshold = ($input_size * 64) / 2;
                    let mut input_sums = [0i32; $input_size * 64];
                    for output_word in 0..$output_size {
                        for output_word_bit in 0..64 {
                            let mut xored_bits = [0u64; $input_size];
                            let mut sum = 0;
                            for i in 0..$input_size {
                                xored_bits[i] = input[i] ^ weights[output_word * output_word_bit][i];
                                sum += xored_bits[i].count_ones();
                            }
                            if target_grad[output_word * output_word_bit] > 0 {
                                if sum == threshold {
                                    for input_word in 0..$input_size {
                                        for input_word_bit in 0..64 {
                                            if ((xored_bits[input_word] >> input_word_bit) & 0b1) == 0b1 {
                                                input_sums[input_word * input_word_bit] += 1;
                                            }
                                        }
                                    }
                                }
                            }
                            if sum == (threshold + 1) {
                                for input_word in 0..$input_size {
                                    for input_word_bit in 0..64 {
                                        if ((xored_bits[input_word] >> input_word_bit) & 0b1) == 0b1 {
                                            input_sums[input_word * input_word_bit] += -1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let mut grads = [0i8; $input_size * 64];
                    for (index, &sum) in input_sums.iter().enumerate() {
                        if sum > 0 {
                            grads[index] = 1;
                        }
                        if sum < 0 {
                            grads[index] = -1;
                        }
                    }
                    grads
                }
            )
        }
        #[macro_export]
        macro_rules! dense_bits2ints {
            ($input_size:expr, $output_size:expr) => (
                |output: &mut [u32; $output_size], params: &[[u64; $input_size]; $output_size], input: &[u64; $input_size]| {
                    for o in 0..$output_size {
                        output[o] = input.iter().
                        zip(params[o].iter()).
                        fold(0, |acc, x| acc + (x.0 ^ x.1).count_ones());
                    }
                }
            )
        }
        #[macro_export]
        macro_rules! dense_bits2ints_grad {
            ($input_size:expr, $output_size:expr) => (
                |target_grad: &[i8; $output_size], weights: &[[u64; $input_size]; $output_size], input: &[u64; $input_size]| -> [i8; $input_size * 64] {
                    let threshold = ($input_size * 64) / 2;
                    let mut input_sums = [0i32; $input_size * 64];
                    for output_index in 0..$output_size {
                        let mut xored_bits = [0u64; $input_size];
                        let mut sum = 0;
                        for i in 0..$input_size {
                            xored_bits[i] = input[i] ^ weights[output_index][i];
                            sum += xored_bits[i].count_ones();
                        }
                        if sum == threshold {
                            for input_word in 0..$input_size {
                                for input_word_bit in 0..64 {
                                    if ((xored_bits[input_word] >> input_word_bit) & 0b1) == 0b1 {
                                        input_sums[input_word * input_word_bit] += 1;
                                    }
                                }
                            }
                        }
                        if sum == (threshold + 1) {
                            for input_word in 0..$input_size {
                                for input_word_bit in 0..64 {
                                    if ((xored_bits[input_word] >> input_word_bit) & 0b1) == 0b1 {
                                        input_sums[input_word * input_word_bit] += -1;
                                    }
                                }
                            }
                        }
                    }
                    let mut grads = [0i8; $input_size * 64];
                    for (index, &sum) in input_sums.iter().enumerate() {
                        if sum > 0 {
                            grads[index] = 1;
                        }
                        if sum < 0 {
                            grads[index] = -1;
                        }
                    }
                    grads
                }
            )
        }
        #[macro_export]
        macro_rules! correct {
            ($input:expr, $target:expr) => (
                ($input.iter().enumerate().
                max_by_key(|&(_, item)| item).unwrap().0 == $target)
            )
        }
        #[macro_export]
        macro_rules! goodness {
            ($input:expr, $target:expr) => (
                $input.iter().enumerate().
                fold(0, |acc, (index, value)| if index == $target {acc + value} else {acc})
            )
        }
    }
}
