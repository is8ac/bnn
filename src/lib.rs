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
        pub fn load_images_bitpacked(path: &String, size: usize) -> Vec<[u64; 13]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in reverse order,
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
    }
}

#[macro_use]
pub mod params {
    #[macro_export]
    macro_rules! random_prms_3d {
        ($dim0:expr, $dim1:expr, $dim2:expr) => {{
            let mut new_params = [[[0u64; $dim2]; $dim1]; $dim0];
            for d0 in 0..$dim0 {
                for d1 in 0..$dim1 {
                    for d2 in 0..$dim2 {
                        new_params[d0][d1][d2] = rand::random::<u64>();
                    }
                }
            }
            new_params
        }};
    }
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
                                            let val = input[x + ix - 1][y + iy - 1][i] as i32;
                                            //let bit = (filter[ix_offset + iy] >> ib) & 0b1u8;
                                            //sum += (val ^ (0u8 - bit) as i32) + bit as i32;
                                            if (filter[ix_offset + iy] & mask) == mask {
                                                sum += val;
                                            } else {
                                                sum -= val;
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
                                            //let bit = (filter[ix_offset + iy] >> ob) & 0b1u8;
                                            //sum += (val ^ (0u8 - bit) as i32) + bit as i32;
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

    #[macro_export]
    macro_rules! conv {
        (
            $name:ident,
            $internal_type:expr,
            $x_size:expr,
            $y_size:expr,
            $in_chans:expr,
            $in_type:ident,
            $out_chans:expr,
            $out_type:ident,
            $patch_x:expr,
            $patch_y:expr,
            $max:expr,
            $shift:expr
        ) => {
            fn $name(
                input: &[[[$in_type; $in_chans]; $y_size]; $x_size],
                filter: &[u8; $in_chans * $out_chans * 3 * 3],
            ) -> [[[$out_type; $out_chans * 8]; $y_size]; $x_size] {
                let mut output = [[[make_0val!($out_type); $out_chans * 8]; $y_size]; $x_size];
                for x in ($patch_x / 2)..$x_size - ($patch_x / 2) {
                    // for all the pixels in the output, inset by one
                    for y in ($patch_y / 2)..$y_size - ($patch_y / 2) {
                        for ow in 0..$out_chans {
                            for ob in 0..8 {
                                let mask = 0b1u8 << ob;
                                let o = ow * 8 + ob;
                                let mut sum = make_0val!($internal_type);
                                let o_offset = ow * $in_chans * $patch_x * $patch_y;
                                for i in 0..$in_chans {
                                    let i_offset = o_offset + i * $patch_x * $patch_y;
                                    for ix in 0..$patch_x {
                                        let ix_offset = i_offset + ix * $patch_y;
                                        for iy in 0..$patch_y {
                                            let val = input[x + ix - ($patch_x / 2)][y + iy - ($patch_y / 2)][i] as i32;
                                            //let bit = (filter[ix_offset + iy] >> ob) & 0b1u8;
                                            //sum += (val ^ (0u8 - bit) as i32) + bit as i32;
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
    #[macro_export]
    macro_rules! max_1d {
        ($name:ident, $size:expr, $max:expr) => {
            fn $name(input: &[i32; $size]) -> [u32; $size] {
                let mut output = [0u32; $size];
                for i in 0..$size {
                    output[i] = input[i].max(0).min($max) as u32;
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! rnn_cell {
        ($name:ident, $x_size:expr, $h_size:expr, $range:expr) => {
            fn $name(x: [i32; x_size], h: [i32; $h_size], weights: [u8; ($x_size + $h_size * 8) * $h_size]) -> [i32; $h_size * 8] {
                let mut output = [i32; $h_size];
                for ow in 0..$h_size {
                    let o_offset = ow * ($x_size * $h_size * 8)
                    for ob in 0..8 {
                        let o = ow * ob;
                        let mask = 0b1u8 << b;
                        let mut value;
                        for i in 0..$x_size {
                            if (weights[o_offset + i] & mask) == mask {
                                value += x[i];
                            } else {
                                value -= x[i];
                            }
                        }
                        for i in 0..$h_size {
                            if (weights[o_offset + $x_size + i] & mask) == mask {
                                value += h[i];
                            } else {
                                value -= h[i];
                            }
                        }
                        output[o] = value.min($range).max(-$range); // tanh
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! onehot {
        ($name:ident, $size:expr, $out_type:ident, $off:expr, $on:expr) => {
            fn $name(index: u8) -> [$size; $out_type] {
                let mut output = [$size; $off];
                output[index] = $on;
                output
            }
        };
    }

    #[macro_export]
    macro_rules! binary_conv3x3_dummy {
        ($name:ident, $x_size:expr, $y_size:expr, $chans:expr) => {
            fn $name(input: &[[[u64; $chans]; $y_size]; $x_size]) -> [[[u64; $chans]; $y_size - 2]; $x_size - 2] {
                let mut output = [[[0u64; $chans]; $y_size - 2]; $x_size - 2];
                for x in 0..$x_size - 2 {
                    // for all the pixels in the output, inset by one
                    for y in 0..$y_size - 2 {
                        output[x][y] = input[x + 1][y + 1];
                    }
                }
                output
            }
        };
    }

    #[macro_export]
    macro_rules! binary_conv3x3 {
      ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
          fn $name(
              input: &[[[u64; $in_chans]; $y_size]; $x_size],
              weights: &[[[u64; $in_chans]; 9]; $out_chans * 64],
          ) -> [[[u64; $out_chans]; $y_size - 2]; $x_size - 2] {
              let mut output = [[[0u64; $out_chans]; $y_size - 2]; $x_size - 2];
              let threshold = (9 * 32 * $in_chans) as u32;
              for x in 0..$x_size - 2 { // for all the pixels in the output, inset by one
                  for y in 0..$y_size - 2 {
                      for ow in 0..$out_chans { // for each word of the output channels,
                          for ob in 0..64 { // for each of the 64 bits of that output word,
                              let wi = ow * 64 + ob;
                              let mut sum = 0; // sum holds all the 3 * 3 * input_chans * 64 bits.
                              for iw in 0..$in_chans { // for each word of the input,
                                  // we take each of the 9 pixels of the input patch and xor the weight for that [output_chan, pixel, input_chan], with the input
                                 sum += (weights[wi][0][iw] ^ input[x + 0][y + 0][iw]).count_ones()
                                      + (weights[wi][1][iw] ^ input[x + 1][y + 0][iw]).count_ones()
                                      + (weights[wi][2][iw] ^ input[x + 2][y + 0][iw]).count_ones()
                                      + (weights[wi][3][iw] ^ input[x + 0][y + 1][iw]).count_ones()
                                      + (weights[wi][4][iw] ^ input[x + 1][y + 1][iw]).count_ones()
                                      + (weights[wi][5][iw] ^ input[x + 2][y + 1][iw]).count_ones()
                                      + (weights[wi][6][iw] ^ input[x + 0][y + 2][iw]).count_ones()
                                      + (weights[wi][7][iw] ^ input[x + 1][y + 2][iw]).count_ones()
                                      + (weights[wi][8][iw] ^ input[x + 2][y + 2][iw]).count_ones();
                              }
                              output[x][y][ow] = output[x][y][ow] | (((sum > threshold) as u64) << ob);
                          }
                      }
                  }
              }
              output
            }
        };
    }
    #[macro_export]
    macro_rules! binary_conv3x3_partial {
      ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
          fn $name(
              input: &[[[u64; $in_chans]; $y_size]; $x_size],
              weights: &[[[u64; $in_chans]; 9]; $out_chans * 64],
              output: &mut [[[u64; $out_chans]; $y_size - 2]; $x_size - 2],
              ow: usize,
              ob: usize,
          ) {
              let threshold = (9 * 32 * $in_chans) as u32;
              let wi = ow * 64 + ob;
              let mask = !(0b1u64 << ob);
              for x in 0..$x_size - 2 { // for all the pixels in the output, inset by one
                  for y in 0..$y_size - 2 {
                      let mut sum = 0; // sum holds all the 3 * 3 * input_chans * 64 bits.
                      for iw in 0..$in_chans { // for each word of the input,
                          // we take each of the 9 pixels of the input patch and xor the weight for that [output_chan, pixel, input_chan], with the input
                         sum += (weights[wi][0][iw] ^ input[x + 0][y + 0][iw]).count_ones()
                              + (weights[wi][1][iw] ^ input[x + 1][y + 0][iw]).count_ones()
                              + (weights[wi][2][iw] ^ input[x + 2][y + 0][iw]).count_ones()
                              + (weights[wi][3][iw] ^ input[x + 0][y + 1][iw]).count_ones()
                              + (weights[wi][4][iw] ^ input[x + 1][y + 1][iw]).count_ones()
                              + (weights[wi][5][iw] ^ input[x + 2][y + 1][iw]).count_ones()
                              + (weights[wi][6][iw] ^ input[x + 0][y + 2][iw]).count_ones()
                              + (weights[wi][7][iw] ^ input[x + 1][y + 2][iw]).count_ones()
                              + (weights[wi][8][iw] ^ input[x + 2][y + 2][iw]).count_ones();
                      }
                      output[x][y][ow] = output[x][y][ow] & mask; // set the bit to 0
                      output[x][y][ow] = output[x][y][ow] | (((sum > threshold) as u64) << ob);
                  }
              }
            }
        };
    }
    #[macro_export]
    macro_rules! fc_3dbits2ints {
        ($name:ident, $x_size:expr, $y_size:expr, $z_size:expr, $output_size:expr) => {
            fn $name(
                input: &[[[u64; $z_size]; $y_size]; $x_size],
                weights: &[[[[u64; $z_size]; $y_size]; $x_size]; $output_size],
            ) -> [u32; $output_size] {
                let mut output = [0u32; $output_size];
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        for z in 0..$z_size {
                            for o in 0..$output_size {
                                output[o] += (input[x][y][z] ^ weights[o][x][y][z]).count_ones();
                            }
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! fc_3dbits2ints_goodness {
        ($name:ident, $x_size:expr, $y_size:expr, $z_size:expr, $output_size:expr) => {
            fn $name(
                input: &[[[u64; $z_size]; $y_size]; $x_size],
                weights: &[[[[u64; $z_size]; $y_size]; $x_size]; $output_size],
                actual: usize,
            ) -> u32 {
                let mut goodness = 0;
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        for z in 0..$z_size {
                            goodness += (input[x][y][z] ^ weights[actual][x][y][z]).count_ones();
                        }
                    }
                }
                output
            }
        };
    }

    #[macro_export]
    macro_rules! fc_3dbits2ints_grads {
        ($name:ident, $x_size:expr, $y_size:expr, $i_size:expr, $output_size:expr) => {
            fn $name(
                input: &[[[u64; $i_size]; $y_size]; $x_size],
                pos_grads: &mut [[[[[i32; 64]; $i_size]; $y_size]; $x_size]; $output_size],
                neg_grads: &mut [[[[[i32; 64]; $i_size]; $y_size]; $x_size]; $output_size],
                actual: usize,
            ) {
                for o in 0..$output_size {
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            for i in 0..$i_size {
                                for b in 0..64 {
                                    let mask = 0b1u64 << b;
                                    let set = (input[x][y][i] & mask) == mask;
                                    if o == actual {
                                        if set {
                                            neg_grads[o][x][y][i][b] += 1;
                                        } else {
                                            pos_grads[o][x][y][i][b] += 1;
                                        }
                                    } else {
                                        if set {
                                            neg_grads[o][x][y][i][b] -= 1;
                                        } else {
                                            pos_grads[o][x][y][i][b] -= 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
    }
    #[macro_export]
    macro_rules! fc_3dbits2ints_grads_update {
        ($name:ident, $x_size:expr, $y_size:expr, $i_size:expr, $output_size:expr) => {
            fn $name(
                input: &[[[u64; $i_size]; $y_size]; $x_size],
                pos_grads: &mut [[[[[i32; 64]; $i_size]; $y_size]; $x_size]; $output_size],
                neg_grads: &mut [[[[[i32; 64]; $i_size]; $y_size]; $x_size]; $output_size],
                actual: usize,
                i: usize,
                b: usize,
            ) {
                let mask = 0b1u64 << b;
                for o in 0..$output_size {
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            let set = (input[x][y][i] & mask) == mask;
                            if o == actual {
                                if set {
                                    neg_grads[o][x][y][i][b] += 1;
                                } else {
                                    pos_grads[o][x][y][i][b] += 1;
                                }
                            } else {
                                if set {
                                    neg_grads[o][x][y][i][b] -= 1;
                                } else {
                                    pos_grads[o][x][y][i][b] -= 1;
                                }
                            }
                        }
                    }
                }
            }
        };
    }
    #[macro_export]
    macro_rules! partial_fc_3dbits2ints_grads_zero {
        ($name:ident, $x_size:expr, $y_size:expr, $i_size:expr, $output_size:expr) => {
            fn $name(grads: &mut [[[[[i32; 64]; $i_size]; $y_size]; $x_size]; $output_size], i: usize, b: usize) {
                for o in 0..$output_size {
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            grads[o][x][y][i][b] = 0;
                        }
                    }
                }
            }
        };
    }
    #[macro_export]
    macro_rules! fc_3dbits2ints_grads_loss {
        ($name:ident, $x_size:expr, $y_size:expr, $z_size:expr, $output_size:expr) => {
            fn $name(
                pos_grads: &[[[[[i32; 64]; $z_size]; $y_size]; $x_size]; $output_size],
                neg_grads: &[[[[[i32; 64]; $z_size]; $y_size]; $x_size]; $output_size],
            ) -> i32 {
                let mut loss = 0;
                for o in 0..$output_size {
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            for z in 0..$z_size {
                                for b in 0..64 {
                                    loss += pos_grads[o][x][y][z][b].min(neg_grads[o][x][y][z][b]);
                                }
                            }
                        }
                    }
                }
                loss
            }
        };
    }
    #[macro_export]
    macro_rules! fc_3dbits2ints_grads2params {
        ($name:ident, $x_size:expr, $y_size:expr, $z_size:expr, $output_size:expr) => {
            fn $name(
                pos_grads: &[[[[[i32; 64]; $z_size]; $y_size]; $x_size]; $output_size],
                neg_grads: &[[[[[i32; 64]; $z_size]; $y_size]; $x_size]; $output_size],
            ) -> [[[[u64; $z_size]; $y_size]; $x_size]; $output_size] {
                let mut params = [[[[0u64; $z_size]; $y_size]; $x_size]; $output_size];
                for o in 0..$output_size {
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            for z in 0..$z_size {
                                for b in 0..64 {
                                    let bit = pos_grads[o][x][y][z][b] > neg_grads[o][x][y][z][b];
                                    params[o][x][y][z] = params[o][x][y][z] | ((bit as u64) << b);
                                }
                            }
                        }
                    }
                }
                params
            }
        };
    }
}
