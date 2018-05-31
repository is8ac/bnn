#![feature(test)]
extern crate test;

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
        pub fn load_images_u8_1chan(path: &String, size: usize) -> Vec<[[[u8; 1]; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[[[u8; 1]; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image = [[[0u8; 1]; 28]; 28];
                for p in 0..784 {
                    image[p / 28][p % 28][0] = images_bytes[p];
                }
                images.push(image);
            }
            return images;
        }
    }
}

#[macro_use]
pub mod params {
    #[macro_export]
    macro_rules! make_0val {
        (u8) => {
            0u8
        };
        (u16) => {
            0u16
        };
        (u32) => {
            0u32
        };
        (u64) => {
            0u64
        };
        (i8) => {
            0i8
        };
        (i16) => {
            0i16
        };
        (i32) => {
            0i32
        };
        (i64) => {
            0i64
        };
    }
    #[macro_export]
    macro_rules! read_type {
        ($rdr:expr,u8) => {
            $rdr.read_u8::<BigEndian>()
        };
        ($rdr:expr,u16) => {
            $rdr.read_u16::<BigEndian>()
        };
        ($rdr:expr,u32) => {
            $rdr.read_u32::<BigEndian>()
        };
        ($rdr:expr,u64) => {
            $rdr.read_u64::<BigEndian>()
        };
    }
    #[macro_export]
    macro_rules! write_type {
        ($wtr:expr, $val:expr,u8) => {
            $wtr.write_u8::<BigEndian>($val)
        };
        ($wtr:expr, $val:expr,u16) => {
            $wtr.write_u16::<BigEndian>($val)
        };
        ($wtr:expr, $val:expr,u32) => {
            $wtr.write_u32::<BigEndian>($val)
        };
        ($wtr:expr, $val:expr,u64) => {
            $wtr.write_u64::<BigEndian>($val)
        };
    }
}

#[macro_use]
pub mod layers {
    #[macro_export]
    macro_rules! log_dist_1d {
        ($type:ty, $prefix:expr, $size:expr) => {
            |input: &[$type; $size]| {
                let mut sum = 0;
                let mut max = 0;
                let mut min = 0;
                for i in 0..$size {
                    if input[i] > max {
                        max = input[i];
                    }
                    if input[i] < min {
                        min = input[i];
                    }
                    sum += input[i];
                }
                let avg = sum as f64 / $size as f64;
                println!("{:?} {:?} {:?} {:?}", $prefix, min, avg, max);
            }
        };
    }

    #[macro_export]
    macro_rules! log_dist {
        ($type:ty, $prefix:expr, $dim0:expr, $dim1:expr, $dim2:expr) => {
            |input: &[[[$type; $dim2]; $dim1]; $dim0]| {
                let mut sum = 0;
                let mut max = 0;
                let mut min = 0;
                for d0 in 0..$dim0 {
                    for d1 in 0..$dim1 {
                        for d2 in 0..$dim2 {
                            if input[d0][d1][d2] > max {
                                max = input[d0][d1][d2];
                            }
                            if input[d0][d1][d2] < min {
                                min = input[d0][d1][d2];
                            }
                            sum += input[d0][d1][d2];
                        }
                    }
                }
                let avg = sum as f64 / ($dim0 * $dim1 * $dim2) as f64;
                println!("{:?} {:?} {:?} {:?}", $prefix, min, avg, max);
            }
        };
    }
    #[macro_export]
    macro_rules! random_byte_array {
        ($size:expr) => {
            || -> [u8; $size] {
                let mut output = [0u8; $size];
                for i in 0..$size {
                    output[i] = rand::random::<u8>();
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! read_array {
        ($size:expr) => {
            |vector: &mut Vec<u8>| -> [u8; $size] {
                let mut output = [0u8; $size];
                for i in 0..$size {
                    output[i] = vector.pop().expect("not enough bytes of params to read");
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! conv_3x3_u8_params_u32_activation_output {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr, $max:expr, $shift:expr) => {
            fn $name(
                input: &[[[u32; $in_chans * 8]; $y_size]; $x_size],
                filter: &[u8; $in_chans * $out_chans * 3 * 3],
            ) -> [[[u32; $out_chans]; $y_size]; $x_size] {
                let mut output = [[[0u32; $out_chans]; $y_size]; $x_size];
                for x in 1..$x_size - 1 {
                    // for all the pixels in the output, inset by one
                    for y in 1..$y_size - 1 {
                        for o in 0..$out_chans {
                            let mut sum = 0i32;
                            let o_offset = o * $in_chans * 3 * 3;
                            for iw in 0..$in_chans {
                                let i_offset = o_offset + iw * 3 * 3;
                                for ib in 0..8 {
                                    let mask = 0b1u8 << ib;
                                    let i = iw * 8 + ib;
                                    for ix in 0..3 {
                                        let ix_offset = i_offset + ix * 3;
                                        for iy in 0..3 {
                                            let value = input[x + ix - 1][y + iy - 1][i] as i32;
                                            if (filter[ix_offset + iy] & mask) == mask {
                                                sum += value;
                                            } else {
                                                sum -= value;
                                            }
                                        }
                                    }
                                }
                            }
                            output[x][y][o] = (sum >> $shift).max(0).min($max) as u32;
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! conv_3x3_u8_params_u32_activation_input {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr, $max:expr, $shift:expr) => {
            fn $name(
                input: &[[[u8; $in_chans]; $y_size]; $x_size],
                filter: &[u8; $in_chans * $out_chans * 3 * 3],
            ) -> [[[u32; $out_chans * 8]; $y_size]; $x_size] {
                let mut output = [[[0u32; $out_chans * 8]; $y_size]; $x_size];
                for x in 1..$x_size - 1 {
                    // for all the pixels in the output, inset by one
                    for y in 1..$y_size - 1 {
                        for ow in 0..$out_chans {
                            for ob in 0..8 {
                                let mask = 0b1u8 << ob;
                                let o = ow * 8 + ob;
                                let mut sum = 0i32;
                                let o_offset = ow * $in_chans * 3 * 3;
                                for i in 0..$in_chans {
                                    let i_offset = o_offset + i * 3 * 3;
                                    for ix in 0..3 {
                                        let ix_offset = i_offset + ix * 3;
                                        for iy in 0..3 {
                                            let val = input[x + ix - 1][y + iy - 1][i] as i32;
                                            if (filter[ix_offset + iy] & mask) == mask {
                                                sum += val;
                                            } else {
                                                sum -= val;
                                            }
                                        }
                                    }
                                }
                                output[x][y][o] = (sum >> $shift).max(0).min($max) as u32;
                            }
                        }
                    }
                }
                output
            }
        };
    }

    /// 2 by 2 or pooling. Takes 2x2 patches, `or`s the 4 bits together. Reduces image size by a factor of 2 in each dimention.
    #[macro_export]
    macro_rules! max_pool {
        ($name:ident, $type:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[$type; $num_chans]; $x_size]; $y_size]) -> [[[$type; $num_chans]; $y_size / 2]; $x_size / 2] {
                let mut output = [[[make_0val!($type); $num_chans]; $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let ix = x * 2;
                    for y in 0..$y_size / 2 {
                        let iy = y * 2;
                        for chan in 0..$num_chans {
                            output[x][y][chan] = input[ix + 0][iy + 0][chan]
                                .max(input[ix + 1][iy + 0][chan])
                                .max(input[ix + 0][iy + 1][chan])
                                .max(input[ix + 1][iy + 1][chan]);
                        }
                    }
                }
                output
            }
        };
    }
    /// takes [x, y, chan] (or any other 3 dimentional vector)
    // assumes that the outer one pixels are empty.
    #[macro_export]
    macro_rules! flatten3d {
        ($name:ident, $type:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[$type; $num_chans]; $x_size]; $y_size]) -> [$type; ($x_size - 2) * ($y_size - 2) * $num_chans] {
                let mut output = [make_0val!($type); ($x_size - 2) * ($y_size - 2) * $num_chans];
                let mut index = 0;
                for x in 1..$x_size - 1 {
                    for y in 1..$y_size - 1 {
                        for c in 0..$num_chans {
                            output[index] = input[x][y][c];
                            index += 1;
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! dense_ints2ints {
        ($name:ident, $in_type:ident, $out_type:ident, $input_size:expr, $output_size:expr) => {
            fn $name(input: &[$in_type; $input_size * 8], weights: &[u8; $input_size * $output_size]) -> [$out_type; $output_size] {
                let mut output = [make_0val!($out_type); $output_size];
                for i in 0..$input_size {
                    let i_offset = i * $output_size;
                    for b in 0..8 {
                        let index = i * 8 + b;
                        let mask = 0b1u8 << b;
                        for o in 0..$output_size {
                            let value = input[index] as $out_type;
                            if (weights[i_offset + o] & mask) == mask {
                                output[o] += value;
                            } else {
                                output[o] -= value;
                            }
                        }
                    }
                }
                output
            }
        };
    }
}
