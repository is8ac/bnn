extern crate image;
extern crate rand;
extern crate rayon;

#[macro_use]
pub mod datasets {
    pub mod fake {
        extern crate image;
        use image::{ImageBuffer, Rgb};

        pub fn parse_rgb_u8(bits: u8) -> [u8; 3] {
            let mut bytes = [0u8; 3];
            for color in 0..3 {
                bytes[color] = ((bits >> color) & 0b1u8) * 255;
            }
            bytes
        }
        pub fn diag_rgb_u8_packed() -> [[u8; 32]; 32] {
            let mut image = [[0u8; 32]; 32];
            for x in 0..32 {
                for y in 0..32 {
                    image[x][y] = ((((((x + (y % 3) + 0) as u8) % 3) == 0) as u8) << 0)
                        | ((((((x + (y % 3) + 1) as u8) % 3) == 0) as u8) << 1)
                        | ((((((x + (y % 3) + 2) as u8) % 3) == 0) as u8) << 2);
                }
            }
            image
        }
        pub fn write_u8_packed_image(data: [[u8; 32]; 32], path: &String) {
            let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(32u32, 32u32);
            for x in 0..32 {
                for y in 0..32 {
                    let bits = data[x][y];
                    let pixel_bytes = parse_rgb_u8(bits);
                    image.get_pixel_mut(x as u32, y as u32).data = pixel_bytes;
                }
            }
            image.save(path).unwrap();
        }
    }
    pub mod cifar {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;
        extern crate image;
        use image::{ImageBuffer, Rgb};

        pub fn write_image(data: [[[u64; 1]; 32]; 32], path: &String) {
            let mut image = ImageBuffer::<Rgb<u8>, Vec<u8>>::new(32u32, 32u32);
            for x in 0..32 {
                for y in 0..32 {
                    let bits = data[x][y][0];
                    let pixel_bytes = parse_rgb_u64(bits);
                    image.get_pixel_mut(x as u32, y as u32).data = pixel_bytes;
                }
            }
            image.save(path).unwrap();
        }
        pub fn encode_unary_rgb(bytes: [u8; 3]) -> u64 {
            let mut ones = 0u64;
            for color in 0..3 {
                for i in 0..bytes[color] / 12 {
                    ones = ones | 0b1u64 << (color * 21 + i as usize);
                }
            }
            ones
        }
        pub fn parse_rgb_u64(bits: u64) -> [u8; 3] {
            let mut bytes = [0u8; 3];
            for color in 0..3 {
                bytes[color] = (((bits >> (color * 21)) & 0b111111111111111111111u64).count_ones() * 12) as u8; // 21 bit mask
            }
            bytes
        }
        pub fn load_images_64chan_10(path: &String, size: usize) -> Vec<(u8, [[[u64; 1]; 32]; 32])> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open data");

            let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
            let mut label: [u8; 1] = [0; 1];
            let mut images: Vec<(u8, [[[u64; 1]; 32]; 32])> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut label).expect("can't read label");
                file.read_exact(&mut image_bytes).expect("can't read images");
                let mut image = [[[0u64; 1]; 32]; 32];
                for x in 0..32 {
                    for y in 0..32 {
                        image[x][y][0] = encode_unary_rgb([
                            image_bytes[(0 * 1024) + (y * 32) + x],
                            image_bytes[(1 * 1024) + (y * 32) + x],
                            image_bytes[(2 * 1024) + (y * 32) + x],
                        ]);
                    }
                }
                images.push((label[0], image));
            }
            return images;
        }
        pub fn load_images_64chan_100(path: &String, size: usize, fine: bool) -> Vec<(u8, [[[u64; 1]; 32]; 32])> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open data");

            let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
            let mut label: [u8; 2] = [0; 2];
            let mut images: Vec<(u8, [[[u64; 1]; 32]; 32])> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut label).expect("can't read label");
                file.read_exact(&mut image_bytes).expect("can't read images");
                let mut image = [[[0u64; 1]; 32]; 32];
                for x in 0..32 {
                    for y in 0..32 {
                        image[x][y][0] = encode_unary_rgb([
                            image_bytes[(0 * 1024) + (y * 32) + x],
                            image_bytes[(1 * 1024) + (y * 32) + x],
                            image_bytes[(2 * 1024) + (y * 32) + x],
                        ]);
                    }
                }
                images.push((label[fine as usize], image));
            }
            return images;
        }
    }
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
        pub fn load_images(path: &String, size: usize) -> Vec<[[u8; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[[u8; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image = [[0u8; 28]; 28];
                for p in 0..784 {
                    image[p / 28][p % 28] = images_bytes[p];
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
pub mod layers {
    use rayon::prelude::*;

    pub trait Patch {
        fn hamming_distance(&self, &Self) -> u32;
        fn bit_increment(&self, &mut [u32]);
        fn bit_len() -> usize;
        fn bitpack(&[bool]) -> Self;
    }

    macro_rules! primitive_patch {
        ($type:ty, $len:expr) => {
            impl Patch for $type {
                fn hamming_distance(&self, other: &$type) -> u32 {
                    (self ^ other).count_ones()
                }
                fn bit_increment(&self, counters: &mut [u32]) {
                    if counters.len() != $len {
                        panic!("counters is the wrong len");
                    }
                    for i in 0..$len {
                        counters[i] += ((self >> i) & 0b1 as $type) as u32;
                    }
                }
                fn bit_len() -> usize {
                    $len
                }
                fn bitpack(bools: &[bool]) -> $type {
                    if bools.len() != $len {
                        panic!("counters is the wrong len");
                    }
                    let mut val = 0 as $type;
                    for i in 0..$len {
                        val = val | ((bools[i] as $type) << i);
                    }
                    val
                }
            }
        };
    }

    primitive_patch!(u8, 8);
    primitive_patch!(u16, 16);
    primitive_patch!(u32, 32);
    primitive_patch!(u64, 64);
    primitive_patch!(u128, 128);

    macro_rules! array_patch {
        ($len:expr) => {
            impl<T: Patch + Copy + Default> Patch for [T; $len] {
                fn hamming_distance(&self, other: &[T; $len]) -> u32 {
                    let mut distance = 0;
                    for i in 0..$len {
                        distance += self[i].hamming_distance(&other[i]);
                    }
                    distance
                }
                fn bit_increment(&self, counters: &mut [u32]) {
                    if counters.len() != ($len * T::bit_len()) {
                        panic!("counters is the wrong len");
                    }
                    for i in 0..$len {
                        self[i].bit_increment(&mut counters[i * T::bit_len()..(i + 1) * T::bit_len()]);
                    }
                }
                fn bit_len() -> usize {
                    $len * T::bit_len()
                }
                fn bitpack(bools: &[bool]) -> [T; $len] {
                    if bools.len() != ($len * T::bit_len()) {
                        panic!("counters is the wrong len");
                    }
                    let mut val = [T::default(); $len];
                    for i in 0..$len {
                        val[i] = T::bitpack(&bools[i * T::bit_len()..(i + 1) * T::bit_len()]);
                    }
                    val
                }
            }
        };
    }

    array_patch!(1);
    array_patch!(2);
    array_patch!(3);
    array_patch!(4);
    array_patch!(9);


    pub mod bitvecmul {
        macro_rules! primitive_bit_vecmul {
            ($name:ident, $type:ty, $len:expr) => {
                pub fn $name<T: super::Patch>(weights: &[(T, u32); $len], input: &T) -> $type {
                    let mut output = 0 as $type;
                    for b in 0..$len {
                        output = output | ((((weights[b].0).hamming_distance(input) >= weights[b].1) as $type) << b);
                    }
                    output
                }
            };
        }

        primitive_bit_vecmul!(bvm_u3, u8, 3);
        primitive_bit_vecmul!(bvm_u7, u8, 7);
        primitive_bit_vecmul!(bvm_u8, u8, 8);
        primitive_bit_vecmul!(bvm_u14, u16, 14);
        primitive_bit_vecmul!(bvm_u16, u16, 16);
        primitive_bit_vecmul!(bvm_u32, u32, 32);
        primitive_bit_vecmul!(bvm_u64, u64, 64);
        primitive_bit_vecmul!(bvm_u128, u128, 128);
    }

    pub mod unary {
        macro_rules! to_unary {
            ($name:ident, $type:ty, $len:expr) => {
                pub fn $name(input: u8) -> $type {
                    !((!0) << (input / (255 / $len)))
                }
            };
        }

        to_unary!(to_3, u8, 3);
        to_unary!(to_4, u8, 4);
        to_unary!(to_5, u8, 5);
        to_unary!(to_7, u8, 7);
        to_unary!(to_14, u16, 14);
    }

    pub mod pack_3x3 {
        macro_rules! pack_3x3 {
            ($name:ident, $in_type:ty, $out_type:ty, $out_len:expr) => {
                pub fn $name(pixels: [$in_type; 9]) -> $out_type {
                    let mut word = 0 as $out_type;
                    for i in 0..9 {
                        word = word | ((pixels[i] as $out_type) << (i * ($out_len / 9)))
                    }
                    word
                }
            };
        }

        pack_3x3!(p3, u8, u32, 32);
        pack_3x3!(p7, u8, u64, 64);
        pack_3x3!(p14, u16, u128, 128);
    }

    pub trait Extract2dPatches<T> {
        fn extract_3x3_patches(&self) -> Vec<[T; 9]>;
    }

    macro_rules! extract_3x3_patches {
        ($x_size:expr, $y_size:expr) => {
            impl<T: Copy> Extract2dPatches<T> for [[T; $y_size]; $x_size] {
                fn extract_3x3_patches(&self) -> Vec<[T; 9]> {
                    let mut patches = Vec::with_capacity(($x_size - 2) * ($y_size - 2));
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            patches.push([
                                self[x + 0][y + 0],
                                self[x + 1][y + 0],
                                self[x + 2][y + 0],
                                self[x + 0][y + 1],
                                self[x + 1][y + 1],
                                self[x + 2][y + 1],
                                self[x + 0][y + 2],
                                self[x + 1][y + 2],
                                self[x + 2][y + 2],
                            ]);
                        }
                    }
                    patches
                }
            }
        };
    }
    extract_3x3_patches!(28, 28);
    extract_3x3_patches!(32, 32);
    extract_3x3_patches!(64, 64);

    pub mod pixelmap {
        macro_rules! pixel_map_2d {
            ($name:ident, $x_size:expr, $y_size:expr) => {
                pub fn $name<I: Copy, O: Copy + Default>(input: &[[I; $y_size]; $x_size], map_fn: &Fn(I) -> O) -> [[O; $y_size]; $x_size] {
                    let mut output = [[O::default(); $y_size]; $x_size];
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            output[x][y] = map_fn(input[x][y]);
                        }
                    }
                    output
                }
            };
        }

        pixel_map_2d!(pm_28, 28, 28);
        pixel_map_2d!(pm_32, 32, 32);
        pixel_map_2d!(pm_64, 64, 64);
    }

    pub fn count_bits<T: Patch + Sync>(patches: &Vec<T>) -> Vec<u32> {
        patches
            .par_iter()
            .fold(
                || vec![0u32; T::bit_len()],
                |mut counts, example| {
                    example.bit_increment(&mut counts);
                    counts
                },
            )
            .reduce(|| vec![0u32; T::bit_len()], |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect())
    }


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
    macro_rules! dist_1d {
        ($type:ty, $prefix:expr, $size:expr) => {
            |input: &[$type; $size]| -> ($type, $type, f64) {
                let mut sum = 0 as $type;
                let mut max = 0 as $type;
                let mut min = 0 as $type;
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
                (min, avg, max)
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
    macro_rules! xor_conv3x3_max {
        ($name:ident, $in_chans:expr, $n_labels:expr) => {
            fn $name(input: &[[[u64; $in_chans]; 3]; 3], filters: &[[[[u64; $in_chans]; 3]; 3]; $n_labels]) -> usize {
                let mut max_index = 0;
                let mut max_val = 0;
                for l in 0..$n_labels {
                    let mut sum: u32 = 0;
                    for px in 0..3 {
                        for py in 0..3 {
                            for iw in 0..$in_chans {
                                sum += (filters[l][px][py][iw] ^ input[px][py][iw]).count_ones();
                            }
                        }
                    }
                    if sum > max_val {
                        max_val = sum;
                        max_index = l;
                    }
                }
                max_index
            }
        };
    }
    #[macro_export]
    macro_rules! xor_conv3x3_infer {
        ($in_chans:expr, $filters:expr, $input:expr) => {{
            let mut sum: u32 = 0;
            for px in 0..3 {
                for py in 0..3 {
                    for iw in 0..$in_chans {
                        sum += ($filters[px][py][iw] ^ $input[px][py][iw]).count_ones();
                    }
                }
            }
            sum
        }};
    }

    #[macro_export]
    macro_rules! xor_conv3x3_isin_topn {
        ($name:ident, $in_chans:expr, $n_labels:expr, $n:expr) => {
            fn $name(input: &[[[u64; $in_chans]; 3]; 3], filters: &[[[[u64; $in_chans]; 3]; 3]; $n_labels], label: usize) -> bool {
                let target_sum = xor_conv3x3_infer!($in_chans, filters[label], input);
                let mut n_wrong = 0;
                for target in 0..$n_labels {
                    let actual_sum = xor_conv3x3_infer!($in_chans, filters[target], input);
                    if actual_sum > target_sum {
                        n_wrong += 1;
                        if n_wrong >= $n {
                            return false;
                        }
                    }
                }
                true
            }
        };
    }
    #[macro_export]
    macro_rules! bitpack_filter_set {
        ($name:ident, $in_chans:expr, $n_labels:expr) => {
            fn $name(grads: &[[[[[f32; 64]; $in_chans]; 3]; 3]; $n_labels]) -> [[[[u64; $in_chans]; 3]; 3]; $n_labels] {
                let mut filters = [[[[0u64; $in_chans]; 3]; 3]; $n_labels];
                for l in 0..$n_labels {
                    for x in 0..3 {
                        for y in 0..3 {
                            for iw in 0..$in_chans {
                                for b in 0..64 {
                                    let bit = grads[l][x][y][iw][b] > 0f32;
                                    filters[l][x][y][iw] = filters[l][x][y][iw] | ((bit as u64) << b);
                                }
                            }
                        }
                    }
                }
                filters
            }
        };
    }

    #[macro_export]
    macro_rules! par_xor_conv3x3_median_activations {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
            fn $name(filters: &[[[[[u64; $in_chans]; 3]; 3]; 64]; $out_chans], inputs: &Vec<&[[[u64; $in_chans]; $y_size]; $x_size]>) -> [[u32; 64]; $out_chans] {
                let thresholds_vec: Vec<[u32; 64]> = filters
                    .par_iter()
                    .map(|filter_word| {
                        let thresholds_word_vec: Vec<u32> = filter_word
                            .par_iter()
                            .map(|filter| {
                                let mut chan_activations: Vec<u32> = Vec::with_capacity(inputs.len() * $x_size * $y_size);
                                for input in inputs {
                                    for x in 0..$x_size - 2 {
                                        for y in 0..$y_size - 2 {
                                            let mut sum = 0;
                                            for px in 0..3 {
                                                for py in 0..3 {
                                                    for iw in 0..$in_chans {
                                                        sum += (filter[px][py][iw] ^ input[x + px][y + py][iw]).count_ones();
                                                    }
                                                }
                                            }
                                            chan_activations.push(sum);
                                        }
                                    }
                                }
                                chan_activations.sort_unstable();
                                chan_activations[(inputs.len() * $x_size * $y_size) / 2]
                            })
                            .collect();
                        let mut thresholds_word = [0u32; 64];
                        for i in 0..64 {
                            thresholds_word[i] = thresholds_word_vec[i];
                        }
                        thresholds_word
                    })
                    .collect();
                let mut thresholds = [[0u32; 64]; $out_chans];
                for i in 0..$out_chans {
                    thresholds[i] = thresholds_vec[i];
                }
                thresholds
            }
        };
    }

    #[macro_export]
    macro_rules! xor_conv3x3_median_activations {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
            fn $name(filters: &[[[[[u64; $in_chans]; 3]; 3]; 64]; $out_chans], inputs: &Vec<&[[[u64; $in_chans]; $y_size]; $x_size]>) -> [[u32; 64]; $out_chans] {
                let mut thresholds = [[0u32; 64]; $out_chans];
                for ow in 0..$out_chans {
                    for ob in 0..64 {
                        let mut chan_activations: Vec<u32> = Vec::with_capacity(inputs.len() * $x_size * $y_size);
                        for input in inputs {
                            for x in 0..$x_size - 2 {
                                for y in 0..$y_size - 2 {
                                    let mut sum = 0;
                                    for px in 0..3 {
                                        for py in 0..3 {
                                            for iw in 0..$in_chans {
                                                sum += (filters[ow][ob][px][py][iw] ^ input[x + px][y + py][iw]).count_ones();
                                            }
                                        }
                                    }
                                    chan_activations.push(sum);
                                }
                            }
                        }
                        chan_activations.sort_unstable();
                        thresholds[ow][ob] = chan_activations[(inputs.len() * $x_size * $y_size) / 2];
                    }
                }
                thresholds
            }
        };
    }

    #[macro_export]
    macro_rules! fused_xor_conv3x3_and_bitpack {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
            fn $name(
                input: &[[[u64; $in_chans]; $y_size]; $x_size],
                filters: &[[[[[u64; $in_chans]; 3]; 3]; 64]; $out_chans],
                thresholds: &[[u32; 64]; $out_chans],
            ) -> [[[u64; $out_chans]; $y_size]; $x_size] {
                let mut output = [[[0u64; $out_chans]; $y_size]; $x_size];
                for x in 0..$x_size - 2 {
                    for y in 0..$y_size - 2 {
                        for ow in 0..$out_chans {
                            for ob in 0..64 {
                                let mut sum = 0;
                                for px in 0..3 {
                                    for py in 0..3 {
                                        for iw in 0..$in_chans {
                                            sum += (filters[ow][ob][px][py][iw] ^ input[x + px][y + py][iw]).count_ones();
                                        }
                                    }
                                }
                                output[x][y][ow] = output[x][y][ow] | (((sum > thresholds[ow][ob]) as u64) << ob);
                            }
                        }
                    }
                }
                output
            }
        };
    }

    #[macro_export]
    macro_rules! xor_conv3x3_activations {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
            fn $name(filters: &[[[[[u64; $in_chans]; 3]; 3]; 64]; $out_chans], input: &[[[u64; $in_chans]; $y_size]; $x_size]) -> [[[[u32; 64]; $out_chans]; $y_size]; $x_size] {
                let mut activations = [[[[0u32; 64]; $out_chans]; $y_size]; $x_size];
                for x in 0..$x_size - 2 {
                    for y in 0..$y_size - 2 {
                        for ow in 0..$out_chans {
                            for ob in 0..64 {
                                for px in 0..3 {
                                    for py in 0..3 {
                                        for iw in 0..$in_chans {
                                            activations[x + 1][y + 1][ow][ob] += (filters[ow][ob][px][py][iw] ^ input[x + px][y + py][iw]).count_ones();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                activations
            }
        };
    }
    #[macro_export]
    macro_rules! threshold_and_bitpack_image {
        ($name:ident, $x_size:expr, $y_size:expr, $out_chans:expr) => {
            fn $name(input: &[[[[u32; 64]; $out_chans]; $y_size]; $x_size], thresholds: &[[u32; 64]; $out_chans]) -> [[[u64; $out_chans]; $y_size]; $x_size] {
                let mut output = [[[0u64; $out_chans]; $y_size]; $x_size];
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        for c in 0..$out_chans {
                            for b in 0..64 {
                                output[x][y][c] = output[x][y][c] | (((input[x][y][c][b] > thresholds[c][b]) as u64) << b);
                            }
                        }
                    }
                }
                output
            }
        };
    }

    #[macro_export]
    macro_rules! xor_conv3x3_onechan_pooled {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr) => {
            fn $name(input: &[[[u64; $in_chans]; $y_size]; $x_size], weights: &[[[u64; $in_chans]; 3]; 3]) -> u32 {
                let mut sum: u32 = 0;
                for x in 0..$x_size - 2 {
                    // for all the pixels in the output, inset by one
                    for y in 0..$y_size - 2 {
                        for px in 0..3 {
                            for py in 0..3 {
                                for iw in 0..$in_chans {
                                    sum += (weights[px][py][iw] ^ input[x + px][y + py][iw]).count_ones();
                                }
                            }
                        }
                    }
                }
                sum
            }
        };
    }

    #[macro_export]
    macro_rules! bitpack_u64_3d {
        ($name:ident, $type:ty, $a_size:expr, $b_size:expr, $c_size:expr, $thresh:expr) => {
            fn $name(grads: &[[[[$type; 64]; $c_size]; $b_size]; $a_size]) -> [[[u64; $c_size]; $b_size]; $a_size] {
                let mut params = [[[0u64; $c_size]; $b_size]; $a_size];
                for a in 0..$a_size {
                    for b in 0..$b_size {
                        for c in 0..$c_size {
                            for i in 0..64 {
                                let bit = grads[a][b][c][i] > $thresh;
                                params[a][b][c] = params[a][b][c] | ((bit as u64) << i);
                            }
                        }
                    }
                }
                params
            }
        };
    }
    #[macro_export]
    macro_rules! or_pooling {
        ($name:ident, $x_size:expr, $y_size:expr, $chans:expr) => {
            fn $name(image: &[[[u64; $chans]; $y_size]; $x_size]) -> [[[u64; $chans]; $y_size / 2]; $x_size / 2] {
                let mut pooled = [[[0u64; $chans]; $y_size / 2]; $x_size / 2];
                for x in 0..($x_size / 2) {
                    let x_base = x * 2;
                    for y in 0..($y_size / 2) {
                        let y_base = y * 2;
                        for c in 0..$chans {
                            //println!("x: {:?}, y: {:?}", x, y);
                            pooled[x][y][c] =
                                image[x_base + 0][y_base + 0][c] | image[x_base + 0][y_base + 1][c] | image[x_base + 1][y_base + 0][c] | image[x_base + 1][y_base + 1][c];
                            //println!("{:064b}", pooled[x][y][c]);
                        }
                    }
                }
                pooled
            }
        };
    }
    #[macro_export]
    macro_rules! boosted_grads_3x3 {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $n_labels:expr, $n:expr) => {
            fn $name(
                examples: &Vec<(u8, [[[u64; $in_chans]; $y_size]; $x_size])>,
                filter_sets: &Vec<[[[[u64; $in_chans]; 3]; 3]; $n_labels]>,
            ) -> [[[[[f32; 64]; $in_chans]; 3]; 3]; $n_labels] {
                fn is_hard(patch: &[[[u64; $in_chans]; 3]; 3], filter_sets: &Vec<[[[[u64; $in_chans]; 3]; 3]; $n_labels]>, label: u8) -> bool {
                    xor_conv3x3_isin_topn!(is_correct, $in_chans, $n_labels, $n);
                    for filter_set in filter_sets {
                        if is_correct(&patch, &filter_set, label as usize) {
                            return false;
                        }
                    }
                    true
                };
                // for each bit of each (hard) patch, how many bits of the image are set for the corresponding label?
                let mut label_sums = [[[[[0u32; 64]; $in_chans]; 3]; 3]; $n_labels];
                // for each label, how many pixels did we look at?
                let mut label_lens = [0u64; $n_labels];
                for (label, image) in examples.iter() {
                    // for each pixel of the image
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            // extract a patch
                            let patch = [
                                [image[x + 0][y + 0], image[x + 0][y + 1], image[x + 0][y + 2]],
                                [image[x + 1][y + 0], image[x + 1][y + 1], image[x + 1][y + 2]],
                                [image[x + 2][y + 0], image[x + 2][y + 1], image[x + 2][y + 2]],
                            ];
                            // if none of the existing filters can correctly label the patch,
                            if is_hard(&patch, &filter_sets, *label) {
                                // increment that labels pixel counter.
                                label_lens[*label as usize] += 1;
                                // for each of the 9 pixles in the patch,
                                for px in 0..3 {
                                    for py in 0..3 {
                                        // for each word,
                                        for iw in 0..$in_chans {
                                            // and for each bit,
                                            for ib in 0..64 {
                                                // increment the counter for the bit of that label
                                                label_sums[*label as usize][px][py][iw][ib] += ((patch[px][py][iw] >> ib) & 0b1u64) as u32;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Now we must scale down.
                // How much more often was each bit of the patch set in true labels then in average?
                let mut scaled_grads = [[[[[0f32; 64]; $in_chans]; 3]; 3]; $n_labels];
                // how many pixels in total did we look at?
                let global_len: u64 = label_lens.iter().sum();
                println!("total pixels: {:?}", global_len);
                for px in 0..3 {
                    for py in 0..3 {
                        for ic in 0..$in_chans {
                            // for each bit, of the patch
                            for b in 0..64 {
                                let mut global_sum = 0u32; // how many times was this bit set in any label?
                                let mut scaled_label_sums = [0f32; $n_labels]; // what fraction of each label had this bit set?
                                for l in 0..$n_labels {
                                    global_sum += label_sums[l][px][py][ic][b];
                                    scaled_label_sums[l] = label_sums[l][px][py][ic][b] as f32 / label_lens[l] as f32;
                                }
                                let scaled_global_sum = global_sum as f32 / global_len as f32;
                                for l in 0..$n_labels {
                                    scaled_grads[l][px][py][ic][b] = scaled_global_sum - scaled_label_sums[l];
                                }
                            }
                        }
                    }
                }
                scaled_grads
            }
        };
    }
    #[macro_export]
    macro_rules! fixed_dim_extract_3x3_patchs {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr) => {
            fn $name(input: &[[[u64; $in_chans]; $y_size]; $x_size], patches: &mut Vec<[u64; 3 * 3 * $in_chans]>) {
                patches.reserve(($x_size - 2) * ($y_size - 2));
                for x in 0..$x_size - 2 {
                    for y in 0..$y_size - 2 {
                        let mut patch = [0u64; 3 * 3 * $in_chans];
                        for px in 0..3 {
                            let px_offset = px * 3 * $in_chans;
                            for py in 0..3 {
                                let py_offset = px_offset + (py * $in_chans);
                                for i in 0..$in_chans {
                                    patch[py_offset + i] = input[x + px][y + py][i];
                                }
                            }
                        }
                        patches.push(patch);
                    }
                }
            }
        };
    }
    #[macro_export]
    macro_rules! avg_bits {
        ($examples:expr, $in_size:expr) => {{
            let sums: Vec<u32> = $examples
                .par_iter()
                .fold(
                    || vec![0u32; $in_size * 64],
                    |mut counts, example| {
                        for i in 0..$in_size {
                            let offset = i * 64;
                            for b in 0..64 {
                                counts[offset + b] += ((example[i] >> b) & 0b1u64) as u32;
                            }
                        }
                        counts
                    },
                )
                .reduce(
                    || vec![0u32; $in_size * 64],
                    |mut a, b| {
                        for i in 0..$in_size * 64 {
                            a[i] += b[i];
                        }
                        a
                    },
                );
            let len = $examples.len() as f64;
            let mut avgs = [0f64; $in_size * 64];
            for i in 0..$in_size * 64 {
                avgs[i] = sums[i] as f64 / len;
            }
            avgs
        }};
    }
    #[macro_export]
    macro_rules! sum_bits {
        ($examples:expr, $in_size:expr) => {{
            let sums: Vec<u32> = $examples
                .par_iter()
                .fold(
                    || vec![0u32; $in_size * 64],
                    |mut counts, example| {
                        for i in 0..$in_size {
                            let offset = i * 64;
                            for b in 0..64 {
                                counts[offset + b] += ((example[i] >> b) & 0b1u64) as u32;
                            }
                        }
                        counts
                    },
                )
                .reduce(
                    || vec![0u32; $in_size * 64],
                    |mut a, b| {
                        for i in 0..$in_size * 64 {
                            a[i] += b[i];
                        }
                        a
                    },
                );
            ($examples.len(), sums)
        }};
    }
    #[macro_export]
    macro_rules! vec2array {
        ($vector:expr, $size:expr, $type:expr) => {{
            let mut array = [$type; $size];
            for i in 0..$size {
                array[i] = $vector[i];
            }
            array
        }};
    }
}

#[cfg(test)]
mod tests {
    use super::layers::{bitvecmul, pack_3x3, unary};
    use super::layers::{Extract2dPatches, Patch};
    #[test]
    fn patch_count() {
        let mut counters = vec![0u32; 32];
        [0b1011_1111u8, 0b0100_0000u8, 0b0100_0000u8, 0b1000_1010u8].bit_increment(&mut counters);
        let bools: Vec<_> = counters.iter().map(|&x| x != 0).collect();
        let avg = <[u16; 2]>::bitpack(&bools);

        let mut counters2 = vec![0u32; 32];
        avg.bit_increment(&mut counters2);
        println!("{:?}", counters);
        assert_eq!(counters, counters2)
    }
    #[test]
    fn patch_dist() {
        assert_eq!(123u8.hamming_distance(&123u8), 0);
        assert_eq!(0b1010_1000u8.hamming_distance(&0b1010_0111u8), 4);
        assert_eq!([0b1111_0000u8, 0b1111_0000u8].hamming_distance(&[0b0000_1100u8, 0b1111_1111u8]), 6 + 4);
    }
    #[test]
    fn bit_vecmul() {
        let input = [0u128; 3];
        let weights = [([0u128; 3], 0u32); 7];
        let output = bitvecmul::bvm_u7(&weights, &input);
        assert_eq!(output, 0b0111_1111u8);
    }
    #[test]
    fn unary() {
        assert_eq!(unary::to_3(128), 0b0000_0001u8);
        assert_eq!(unary::to_3(255), 0b0000_0111u8);
        assert_eq!(unary::to_7(255), 0b0111_1111u8);
        assert_eq!(unary::to_7(0), 0b0000_0000u8);
    }
    #[test]
    fn pack() {
        assert_eq!(
            pack_3x3::p3([0b11u8, 0b1u8, 0b1u8, 0b1u8, 0b1u8, 0b1u8, 0b1u8, 0b111u8, 0b1u8]),
            0b001111001_001001001_001001011u32
        );
    }
    #[test]
    fn extract_patches() {
        let value = 0b1011_0111u8;
        let layer = [[value; 28]; 28];
        let patches: Vec<_> = layer.extract_3x3_patches().iter().map(|&x| x).collect();
        assert_eq!(patches.len(), 676);
        assert_eq!(patches[0], [value; 9])
    }
}
