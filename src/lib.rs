extern crate rand;
#[macro_use]
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
        pub fn load_labels_onehot(path: &String, size: usize, onval: u16) -> Vec<[u16; 10]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 8] = [0; 8];
            file.read_exact(&mut header).expect("can't read header");

            let mut byte: [u8; 1] = [0; 1];
            let mut labels: Vec<[u16; 10]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut byte).expect("can't read label");
                let mut label = [0u16; 10];
                label[byte[0] as usize] = onval;
                labels.push(label);
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
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image_words: [u64; 13] = [0; 13];
                for p in 0..784 {
                    let word_index = p / 64;
                    image_words[word_index] = image_words[word_index] | (((images_bytes[p] > 128) as u64) << p % 64);
                }
                images.push(image_words);
            }
            return images;
        }

        pub fn load_images_64chan(path: &String, size: usize) -> Vec<[[[u64; 1]; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[[[u64; 1]; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image = [[[0u64; 1]; 28]; 28];
                for p in 0..784 {
                    let mut ones = 0u64;
                    for i in 0..images_bytes[p] / 4 {
                        ones = ones | 1 << i;
                    }
                    image[p / 28][p % 28][0] = ones;
                }
                images.push(image);
            }
            return images;
        }
        pub fn load_images_64chan_flat(path: &String, size: usize) -> Vec<[u64; 28 * 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[u64; 28 * 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image = [0u64; 28 * 28];
                for p in 0..784 {
                    let mut ones = 0u64;
                    for i in 0..images_bytes[p] / 4 {
                        ones = ones | 1 << i;
                    }
                    image[p] = ones;
                }
                images.push(image);
            }
            return images;
        }
    }
}

pub mod layers {
    extern crate rand;
    use rand::Rng;
    pub fn random_bits12() -> u64 {
        rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
            & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
            & rand::random::<u64>()
    }
    pub fn random_bits11() -> u64 {
        rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
            & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>()
    }


    pub fn random_int_plusminus_one() -> i16 {
        rand::thread_rng().gen_range(-1, 2)
    }

    #[macro_export]
    macro_rules! conv3x3 {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
            struct $name {
                weights: [[[u64; $in_chans]; 9]; $out_chans * 64],
                thresholds: [i16; $out_chans * 64],
            }
            impl $name {
                fn layer(&self, input: &[[[u64; $in_chans]; $y_size]; $x_size]) -> [[[u64; $out_chans]; $y_size]; $x_size] {
                    let mut output = [[[0u64; $out_chans]; $y_size]; $x_size];
                    for x in 1..$x_size - 1 {
                        for y in 1..$y_size - 1 {
                            for ow in 0..$out_chans {
                                for ob in 0..64 {
                                    let mut sum = 0;
                                    for iw in 0..$in_chans {
                                        sum += (self.weights[ow * 64 + ob][0][iw] ^ input[x + 0][y + 0][iw]).count_ones()
                                            + (self.weights[ow * 64 + ob][1][iw] ^ input[x + 1][y + 0][iw]).count_ones()
                                            + (self.weights[ow * 64 + ob][2][iw] ^ input[x + 2][y + 0][iw]).count_ones()
                                            + (self.weights[ow * 64 + ob][3][iw] ^ input[x + 0][y + 1][iw]).count_ones()
                                            + (self.weights[ow * 64 + ob][4][iw] ^ input[x + 1][y + 1][iw]).count_ones()
                                            + (self.weights[ow * 64 + ob][5][iw] ^ input[x + 2][y + 1][iw]).count_ones()
                                            + (self.weights[ow * 64 + ob][6][iw] ^ input[x + 0][y + 2][iw]).count_ones()
                                            + (self.weights[ow * 64 + ob][7][iw] ^ input[x + 1][y + 2][iw]).count_ones()
                                            + (self.weights[ow * 64 + ob][8][iw] ^ input[x + 2][y + 2][iw]).count_ones();
                                    }
                                    output[x][y][ow] = output[x][y][ow] | (((sum as i16 > self.thresholds[ow * 64 + ob]) as u64) << ob);
                                }
                            }
                        }
                    }
                    output
                }
                fn new() -> $name {
                    let mut new_layer = $name {
                        weights: [[[0u64; $in_chans]; 9]; $out_chans * 64],
                        thresholds: [(($in_chans * 64 * 9) / 2) as i16; $out_chans * 64],
                    };
                    for o in 0..$out_chans * 64 {
                        for i in 0..9 {
                            for c in 0..$in_chans {
                                new_layer.weights[o][i][c] = rand::random::<u64>();
                            }
                        }
                    }
                    new_layer
                }
                fn child(&self) -> $name {
                    let mut child = $name::new();
                    for o in 0..$out_chans * 64 {
                        for i in 0..9 {
                            for c in 0..$in_chans {
                                child.weights[o][i][c] =
                                    self.weights[o][i][c] ^ (rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>());
                            }
                        }
                    }
                    for t in 0..$out_chans * 64 {
                        child.thresholds[t] = self.thresholds[t] + rand::thread_rng().gen_range(-1, 2);
                    }
                    child
                }
            }
        };
    }
    #[macro_export]
    macro_rules! conv1x1 {
            ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
                struct $name {
                    weights: [[u64; $in_chans]; $out_chans * 64],
                    thresholds: [i16; $out_chans * 64],
                }
                impl $name {
                    fn layer(
                        input: &[[[u64; $in_chans]; $x_size]; $y_size],
                        weights: &[[u64; $in_chans]; 64 * $out_chans],
                        thresholds: &[i16; $out_chans * 64],
                    ) -> [[[u64 $num_chans]; $x_size]; $y_size] {
                        let mut output = [[[0u64; $out_chans]; $x_size]; $y_size];
                        for x in 0..$x_size {
                            for y in 0..$y_size {
                                for ow in 0..$out_chans {
                                    for ob in 0..64 {
                                        let mut sum = 0;
                                        for iw in 0..$in_chans {
                                            sum += (weights[ow * 64 + ob][0][iw] ^ input[x][y][iw]).count_ones()
                                        }
                                        output[x][y][ow] = output[x][y][ow] | (((sum as i16 > thresholds[ow * 64 + ob]) as u64) << ob);
                                    }
                                }
                            }
                        }
                        output
                    }
                    fn new() -> $name {
                        let mut new_layer = $name {
                            weights: [[[0u64; $in_chans]; 9]; $out_chans * 64],
                            thresholds: [($in_chans * 64 / 2) as i16; $out_chans * 64],
                        };
                        for o in 0..$out_chans * 64 {
                            for c in 0..$in_chans {
                                new_layer.weights[o][c] = rand::random::<u64>();
                            }
                        }
                        new_layer
                    }
                    fn child(&self) -> $name {
                        let mut child = $name::new();
                        for o in 0..$out_chans * 64 {
                            for i in 0..9 {
                                for c in 0..$in_chans {
                                    child.weights[o][i][c] =
                                        self.weights[o][i][c] ^ (rand::random::<u64>() & rand::random::<u64>() & rand::random::<u64>());
                                    }
                                }
                            }
                            for t in 0..$out_chans * 64 {
                                child.thresholds[t] = self.thresholds[t] + rand::thread_rng().gen_range(-1, 2);
                            }
                            child
                        }
                }
            };
        }
    #[macro_export]
    macro_rules! pool_or2x2 {
        ($name:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[u64; $num_chans]; $x_size]; $y_size]) -> [[[u64; $num_chans]; $y_size / 2]; $x_size / 2] {
                let mut output = [[[0u64; $num_chans]; $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    for y in 0..$y_size / 2 {
                        for chan in 0..$num_chans {
                            output[x][y][chan] = input[x * 2 + 0][y * 2 + 0][chan] | input[x * 2 + 1][y * 2 + 0][chan]
                                | input[x * 2 + 0][y * 2 + 1][chan] | input[x * 2 + 1][y * 2 + 1][chan]
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! pool_and2x2 {
            ($name:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
                fn $name(input: &[[[u64; $num_chans]; $x_size]; $y_size]) -> [[[u64 $num_chans]; $y_size]; $x_size] {
                    let mut output = [[[0u64; $num_chans]; $y_size / 2]; $x_size / 2];
                    for x in 0..$x_size / 2 {
                        for y in 0..$y_size / 2 {
                            for chan in 0..$num_chans {
                                output[x][y][chan] =
                                input[x * 2 + 0][y * 2 + 0][chan] &
                                input[x * 2 + 1][y * 2 + 0][chan] &
                                input[x * 2 + 0][y * 2 + 1][chan] &
                                input[x * 2 + 1][y * 2 + 1][chan]
                            }
                        }
                    }
                    output
                }
            };
        }
    /// takes [x, y, chan] (or any other 3 dimentional vector)
    #[macro_export]
    macro_rules! flatten3d {
        ($name:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[u64; $num_chans]; $x_size]; $y_size]) -> [u64; $x_size * $y_size * $num_chans] {
                let mut output = [0u64; $x_size * $y_size];
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        for c in $num_chans {
                            output[x * $y_size * $num_chans + y * $num_chans + c] = input[x][y][c];
                        }
                    }
                }
                //for o in 0..$x_size * $y_size {
                //    output[o] = input[o / $y_size][o % $y_size];
                //}
                //println!("squash output: {:?}", output);
                output
            }
        };
    }
    #[macro_export]
    macro_rules! i16_1d {
        ($name:ident, $size:expr) => {
            struct $name {
                thresholds: [i16; $size],
            }
            impl $name {
                fn new_nil() -> $name {
                    $name { thresholds: [0i16; $size] }
                }
                fn new_const(const_val: i16) -> $name {
                    $name {
                        thresholds: [const_val; $size],
                    }
                }
                fn child(&self, random_ints: &Fn() -> i16) -> $name {
                    let mut child = $name::new_nil();
                    for i in 0..$size {
                        child.thresholds[i] = self.thresholds[i] + random_ints();
                    }
                    child
                }
            }
        };
    }
    #[macro_export]
    macro_rules! u64_2d {
        ($name:ident, $dim0:expr, $dim1:expr) => {
            struct $name {
                weights: [[u64; $dim1]; $dim0],
            }
            impl $name {
                fn new_nil() -> $name {
                    $name {
                        weights: [[0u64; $dim1]; $dim0],
                    }
                }
                fn new_random() -> $name {
                    let mut new_params = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            new_params.weights[d0][d1] = rand::random::<u64>();
                        }
                    }
                    new_params
                }
                fn child(&self, random_bits: &Fn() -> u64) -> $name {
                    let mut child = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            child.weights[d0][d1] = self.weights[d0][d1] ^ random_bits();
                        }
                    }
                    child
                }
            }
        };
    }
    #[macro_export]
    macro_rules! u64_3d {
        ($name:ident, $dim0:expr, $dim1:expr, $dim2:expr) => {
            struct $name {
                weights: [[[u64; $dim2]; $dim1]; $dim0],
            }
            impl $name {
                fn new_nil() -> $name {
                    $name {
                        weights: [[[0u64; $dim2]; $dim1]; $dim0],
                    }
                }
                fn new_random() -> $name {
                    let mut new_params = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                new_params.weights[d0][d1][d2] = rand::random::<u64>();
                            }
                        }
                    }
                    new_params
                }
                fn child(&self, random_bits: &Fn() -> u64) -> $name {
                    let mut child = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                child.weights[d0][d1][d2] = self.weights[d0][d1][d2] ^ random_bits();
                            }
                        }
                    }
                    child
                }
            }
        };
    }
    #[macro_export]
    macro_rules! dense_bits2bits {
        ($name:ident, $input_size:expr, $output_size:expr) => {
            struct $name {
                weights: [[u64; $input_size]; $output_size * 64],
                thresholds: [i16; $output_size * 64],
            }
            impl $name {
                fn layer(&self, input: &[u64; $input_size]) -> [u64; $output_size] {
                    let mut output = [0u64; $output_size];
                    for o in 0..$output_size {
                        for b in 0..64 {
                            let mut sum = 0u32;
                            for i in 0..$input_size {
                                sum += (self.weights[o * 64 + b][i] ^ input[i]).count_ones();
                            }
                            output[o] = output[o] | (((sum as i16 > self.thresholds[o * 64 + b]) as u64) << b);
                        }
                    }
                    output
                }
                fn new() -> $name {
                    let mut new_layer = $name {
                        weights: [[0u64; $input_size]; $output_size * 64],
                        thresholds: [($input_size * 64 / 2) as i16; $output_size * 64],
                    };
                    for o in 0..$output_size * 64 {
                        for i in 0..$input_size {
                            new_layer.weights[o][i] = rand::random::<u64>();
                        }
                    }
                    new_layer
                }
                fn child(&self, random_bits: &Fn() -> u64, random_int: &Fn() -> i16) -> $name {
                    let mut child = $name::new();
                    for o in 0..$output_size * 64 {
                        for i in 0..$input_size {
                            child.weights[o][i] = self.weights[o][i] ^ random_bits();
                        }
                    }
                    for o in 0..$output_size * 64 {
                        child.thresholds[o] = self.thresholds[o] + random_int();
                    }
                    child
                }
            }
        };
    }
    #[macro_export]
    macro_rules! dense_bits2ints {
        ($name:ident, $input_size:expr, $output_size:expr) => {
            struct $name {
                weights: [[u64; $input_size]; $output_size],
            }
            impl $name {
                fn $name(&self, input: &[u64; $input_size]) -> [i16; $output_size] {
                    let mut output = [0i16; $output_size];
                    for i in 0..$input_size {
                        for o in 0..$output_size {
                            output[o] += (input[i] ^ self.params[o][i]).count_ones() as i16;
                        }
                    }
                    output
                }
                fn new() -> $name {
                    let mut new_layer = $name {
                        weights: [[0u64; $input_size]; $output_size * 64],
                    };
                    for o in 0..$output_size * 64 {
                        for i in 0..$input_size {
                            new_layer.weights[o][i] = rand::random::<u64>();
                        }
                    }
                    new_layer
                }
                fn child(&self, random_bits: &Fn() -> u64) -> $name {
                    let mut child = $name::new();
                    for o in 0..$output_size * 64 {
                        for i in 0..$input_size {
                            child.weights[o][i] = self.weights[o][i] ^ random_bits();
                        }
                    }
                    child
                }
            }
        };
    }
    #[macro_export]
    macro_rules! threshold_and_bitpack {
        ($name:ident, $size:expr) => {
            fn $name(output: &mut [u64; $size], params: &[u16; $size], input: &[u16; $size * 64]) {
                for o in 0..$size {
                    for b in 0..64 {
                        let i = o * b;
                        output[o] = output[o] | (((input[i] > params[i]) as u64) << b);
                    }
                }
            }
        };
    }
    #[macro_export]
    macro_rules! correct {
        ($size:expr) => {
            |input, target| -> bool { (input.iter().enumerate().max_by_key(|&(_, item)| item).unwrap().0 == target) }
        };
    }
    #[macro_export]
    macro_rules! int_loss {
        ($name:ident, $size:expr) => {
            fn $name(actual: &[u16; $size], target: &[u16; $size]) -> i32 {
                actual
                    .iter()
                    .zip(target.iter())
                    .map(|(&a, &t)| {
                        let diff = a as i32 - t as i32;
                        diff * diff
                    })
                    .sum()
            }
        };
    }
}
