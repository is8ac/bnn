#![feature(const_generics)]

extern crate bincode;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate serde_derive;
extern crate time;
use rayon::prelude::*;

pub mod datasets {
    pub mod mnist {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;
        pub fn load_labels(path: &Path, size: usize) -> Vec<usize> {
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
        pub fn load_images_bitpacked(path: &Path, size: usize) -> Vec<[u64; 13]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

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
        pub fn load_images_bitpacked_u32(path: &Path, size: usize) -> Vec<[u32; 25]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[u32; 25]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image_words: [u32; 25] = [0; 25];
                for p in 0..784 {
                    let word_index = p / 32;
                    image_words[word_index] =
                        image_words[word_index] | (((images_bytes[p] > 128) as u32) << p % 32);
                }
                images.push(image_words);
            }
            return images;
        }

        pub fn load_images_u8_unary(path: &Path, size: usize) -> Vec<[[u8; 28]; 28]> {
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[[u8; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image = [[0u8; 28]; 28];
                for p in 0..784 {
                    image[p / 28][p % 28] = images_bytes[p];
                }
                images.push(image);
            }
            return images;
        }
    }

    pub mod cifar {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;

        macro_rules! to_unary {
            ($name:ident, $type:ty, $len:expr) => {
                fn $name(input: u8) -> $type {
                    !((!0) << (input / (256 / $len) as u8))
                }
            };
        }

        to_unary!(to_3, u8, 3);
        to_unary!(to_10, u32, 10);
        to_unary!(to_11, u32, 11);
        to_unary!(to_32, u32, 32);

        pub trait ConvertPixel {
            fn convert(pixel: [u8; 3]) -> Self;
        }
        impl ConvertPixel for [u8; 3] {
            fn convert(pixel: [u8; 3]) -> [u8; 3] {
                pixel
            }
        }
        impl ConvertPixel for [u32; 1] {
            fn convert(pixel: [u8; 3]) -> [u32; 1] {
                [to_11(pixel[0]) as u32
                    | ((to_11(pixel[1]) as u32) << 11)
                    | ((to_10(pixel[2]) as u32) << 22)]
            }
        }
        impl ConvertPixel for u32 {
            fn convert(pixel: [u8; 3]) -> u32 {
                to_10(pixel[0]) as u32
                    | ((to_10(pixel[1]) as u32) << 10)
                    | ((to_10(pixel[2]) as u32) << 20)
            }
        }
        impl ConvertPixel for u8 {
            fn convert(pixel: [u8; 3]) -> u8 {
                to_3(pixel[0]) | ((to_3(pixel[1])) << 3) | ((to_3(pixel[2])) << 6)
            }
        }

        impl ConvertPixel for [u32; 3] {
            fn convert(pixel: [u8; 3]) -> [u32; 3] {
                [to_32(pixel[0]), to_32(pixel[1]), to_32(pixel[2])]
            }
        }

        pub fn load_images_from_base<T: Default + Copy + ConvertPixel>(
            base_path: &Path,
            n: usize,
        ) -> Vec<([[T; 32]; 32], usize)> {
            if n > 50000 {
                panic!("n must be <= 50,000");
            }
            (1..6)
                .map(|i| {
                    let mut file = File::open(&base_path.join(format!("data_batch_{}.bin", i)))
                        .expect("can't open data");

                    let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
                    let mut label: [u8; 1] = [0; 1];
                    let mut images: Vec<([[T; 32]; 32], usize)> = Vec::new();
                    for _ in 0..10000 {
                        file.read_exact(&mut label).expect("can't read label");
                        file.read_exact(&mut image_bytes)
                            .expect("can't read images");
                        let mut image = [[T::default(); 32]; 32];
                        for x in 0..32 {
                            for y in 0..32 {
                                let pixel = [
                                    image_bytes[(0 * 1024) + (y * 32) + x],
                                    image_bytes[(1 * 1024) + (y * 32) + x],
                                    image_bytes[(2 * 1024) + (y * 32) + x],
                                ];
                                image[x][y] = T::convert(pixel);
                            }
                        }
                        images.push((image, label[0] as usize));
                    }
                    images
                })
                .flatten()
                .take(n)
                .collect()
        }
    }
}

pub mod bits {
    pub trait HammingDistance {
        fn hamming_distance(&self, other: &Self) -> u32;
    }

    impl<T: HammingDistance, const L: usize> HammingDistance for [T; L] {
        fn hamming_distance(&self, other: &[T; L]) -> u32 {
            let mut distance = 0u32;
            for i in 0..L {
                distance += self[i].hamming_distance(&other[i]);
            }
            distance
        }
    }

    impl<A: HammingDistance, B: HammingDistance> HammingDistance for (A, B) {
        fn hamming_distance(&self, other: &(A, B)) -> u32 {
            self.0.hamming_distance(&other.0) + self.1.hamming_distance(&other.1)
        }
    }

    pub trait BitOr {
        fn bit_or(&self, other: &Self) -> Self;
    }

    impl<T: BitOr, const L: usize> BitOr for [T; L]
    where
        [T; L]: Default,
    {
        fn bit_or(&self, other: &Self) -> Self {
            let mut target = <[T; L]>::default();
            for i in 0..L {
                target[i] = self[i].bit_or(&other[i]);
            }
            target
        }
    }

    pub trait BitLen: Sized {
        const BIT_LEN: usize;
    }

    impl<A: BitLen, B: BitLen> BitLen for (A, B) {
        const BIT_LEN: usize = A::BIT_LEN + B::BIT_LEN;
    }

    macro_rules! array_bit_len {
        ($len:expr) => {
            impl<T: BitLen> BitLen for [T; $len] {
                const BIT_LEN: usize = $len * T::BIT_LEN;
            }
        };
    }

    array_bit_len!(1);
    array_bit_len!(2);
    array_bit_len!(3);
    array_bit_len!(4);
    array_bit_len!(5);
    array_bit_len!(6);
    array_bit_len!(7);
    array_bit_len!(8);
    array_bit_len!(13);
    array_bit_len!(16);
    array_bit_len!(25);
    array_bit_len!(32);

    pub trait FlipBit {
        fn flip_bit(&mut self, b: usize);
    }

    impl<T: BitLen + FlipBit, const L: usize> FlipBit for [T; L] {
        fn flip_bit(&mut self, index: usize) {
            self[index / T::BIT_LEN].flip_bit(index % T::BIT_LEN);
        }
    }

    pub trait GetBit {
        fn bit(&self, i: usize) -> bool;
    }

    impl<T: GetBit + BitLen, const L: usize> GetBit for [T; L] {
        fn bit(&self, i: usize) -> bool {
            self[i / T::BIT_LEN].bit(i % T::BIT_LEN)
        }
    }

    macro_rules! impl_for_uint {
        ($type:ty, $len:expr) => {
            impl BitLen for $type {
                const BIT_LEN: usize = $len;
            }
            impl FlipBit for $type {
                fn flip_bit(&mut self, index: usize) {
                    *self ^= 1 << index
                }
            }
            impl BitOr for $type {
                fn bit_or(&self, other: &Self) -> $type {
                    self | other
                }
            }
            impl GetBit for $type {
                #[inline(always)]
                fn bit(&self, i: usize) -> bool {
                    ((self >> i) & 1) == 1
                }
            }
            impl<I: HammingDistance + BitLen> BitMul<I, $type> for [(I, u32); $len] {
                fn bit_mul(&self, input: &I) -> $type {
                    let mut target = <$type>::default();
                    for i in 0..$len {
                        target |= ((self[i].0.hamming_distance(input) < self[i].1) as $type) << i;
                    }
                    target
                }
            }
            impl<I: HammingDistance + BitLen> BitMul<I, $type> for [I; $len] {
                fn bit_mul(&self, input: &I) -> $type {
                    let mut target = <$type>::default();
                    for i in 0..$len {
                        target |= ((self[i].hamming_distance(input) < (I::BIT_LEN as u32 / 2))
                            as $type)
                            << i;
                    }
                    target
                }
            }
            impl HammingDistance for $type {
                #[inline(always)]
                fn hamming_distance(&self, other: &$type) -> u32 {
                    (self ^ other).count_ones()
                }
            }
        };
    }

    impl_for_uint!(u32, 32);
    impl_for_uint!(u16, 16);

    pub trait BitMul<I, O> {
        fn bit_mul(&self, input: &I) -> O;
    }

    impl<I, O: Default + Copy, T: BitMul<I, O>, const L: usize> BitMul<I, [O; L]> for [T; L]
    where
        [O; L]: Default,
    {
        fn bit_mul(&self, input: &I) -> [O; L] {
            let mut target = <[O; L]>::default();
            for i in 0..L {
                target[i] = self[i].bit_mul(input);
            }
            target
        }
    }
}

pub mod count {
    use crate::bits::BitLen;

    pub trait Counters {
        fn elementwise_add(&mut self, other: &Self);
    }

    impl<A: Counters, B: Counters> Counters for (A, B) {
        fn elementwise_add(&mut self, other: &Self) {
            self.0.elementwise_add(&other.0);
            self.1.elementwise_add(&other.1);
        }
    }

    impl Counters for u32 {
        fn elementwise_add(&mut self, other: &u32) {
            *self += other;
        }
    }

    impl<T: Counters, const L: usize> Counters for [T; L] {
        fn elementwise_add(&mut self, other: &[T; L]) {
            for i in 0..L {
                self[i].elementwise_add(&other[i]);
            }
        }
    }

    pub trait IncrementCounters {
        type BitCounterType;
        fn increment_counters(&self, counters: &mut Self::BitCounterType);
        fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self;
        fn compare_and_bitpack(
            counters_0: &Self::BitCounterType,
            n_0: usize,
            counters_1: &Self::BitCounterType,
            n_1: usize,
        ) -> Self;
    }

    impl<A: IncrementCounters, B: IncrementCounters> IncrementCounters for (A, B) {
        type BitCounterType = (A::BitCounterType, B::BitCounterType);
        fn increment_counters(&self, counters: &mut Self::BitCounterType) {
            self.0.increment_counters(&mut counters.0);
            self.1.increment_counters(&mut counters.1);
        }
        fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
            (
                A::threshold_and_bitpack(&counters.0, threshold),
                B::threshold_and_bitpack(&counters.1, threshold),
            )
        }
        fn compare_and_bitpack(
            counters_0: &Self::BitCounterType,
            n_0: usize,
            counters_1: &Self::BitCounterType,
            n_1: usize,
        ) -> Self {
            (
                A::compare_and_bitpack(&counters_0.0, n_0, &counters_1.0, n_1),
                B::compare_and_bitpack(&counters_0.1, n_0, &counters_1.1, n_1),
            )
        }
    }

    impl<T: IncrementCounters, const L: usize> IncrementCounters for [T; L]
    where
        Self: Default,
    {
        type BitCounterType = [T::BitCounterType; L];
        fn increment_counters(&self, counters: &mut [T::BitCounterType; L]) {
            for i in 0..L {
                self[i].increment_counters(&mut counters[i]);
            }
        }
        fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
            let mut target = <[T; L]>::default();
            for i in 0..L {
                target[i] = T::threshold_and_bitpack(&counters[i], threshold);
            }
            target
        }
        fn compare_and_bitpack(
            counters_0: &Self::BitCounterType,
            n_0: usize,
            counters_1: &Self::BitCounterType,
            n_1: usize,
        ) -> Self {
            let mut target = <[T; L]>::default();
            for i in 0..L {
                target[i] = T::compare_and_bitpack(&counters_0[i], n_0, &counters_1[i], n_1);
            }
            target
        }
    }
    macro_rules! impl_for_uint {
        ($type:ty, $len:expr) => {
            impl IncrementCounters for $type {
                type BitCounterType = [u32; <$type>::BIT_LEN];
                fn increment_counters(&self, counters: &mut Self::BitCounterType) {
                    for b in 0..<$type>::BIT_LEN {
                        counters[b] += ((self >> b) & 1) as u32
                    }
                }
                fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
                    let mut target = <$type>::default();
                    for i in 0..$len {
                        target |= (counters[i] > threshold) as $type << i;
                    }
                    target
                }
                fn compare_and_bitpack(counters_0: &Self::BitCounterType, n_0: usize, counters_1: &Self::BitCounterType, n_1: usize) -> Self {
                    let n0f = n_0 as f64;
                    let n1f = n_1 as f64;
                    let mut target = <$type>::default();
                    for i in 0..$len {
                        target |= (counters_0[i] as f64 / n0f > counters_1[i] as f64 / n1f) as $type << i;
                    }
                    target
                }
            }
        }
    }
    impl_for_uint!(u32, 32);
    impl_for_uint!(u16, 16);
}

pub mod image2d {
    use crate::bits::BitOr;

    pub trait OrPool<Output> {
        fn or_pool(&self) -> Output;
    }

    macro_rules! impl_orpool {
        ($x_size:expr, $y_size:expr) => {
            impl<Pixel: BitOr + Default + Copy> OrPool<[[Pixel; $y_size / 2]; $x_size / 2]>
                for [[Pixel; $y_size]; $x_size]
            {
                fn or_pool(&self) -> [[Pixel; $y_size / 2]; $x_size / 2] {
                    let mut target = [[Pixel::default(); $y_size / 2]; $x_size / 2];
                    for x in 0..$x_size / 2 {
                        let x_index = x * 2;
                        for y in 0..$y_size / 2 {
                            let y_index = y * 2;
                            target[x][y] = self[x_index + 0][y_index + 0]
                                .bit_or(&self[x_index + 0][y_index + 1])
                                .bit_or(&self[x_index + 1][y_index + 0])
                                .bit_or(&self[x_index + 1][y_index + 1]);
                        }
                    }
                    target
                }
            }
        };
    }

    impl_orpool!(32, 32);
    impl_orpool!(16, 16);
    impl_orpool!(8, 8);
    impl_orpool!(4, 4);

    pub trait AvgPool {
        type OutputImage;
        fn avg_pool(&self) -> Self::OutputImage;
    }

    macro_rules! impl_avgpool {
        ($x_size:expr, $y_size:expr) => {
            impl AvgPool for [[[u8; 3]; $y_size]; $x_size] {
                type OutputImage = [[[u8; 3]; $y_size / 2]; $x_size / 2];
                fn avg_pool(&self) -> Self::OutputImage {
                    let mut target = [[[0u8; 3]; $y_size / 2]; $x_size / 2];
                    for x in 0..$x_size / 2 {
                        let x_index = x * 2;
                        for y in 0..$y_size / 2 {
                            let y_index = y * 2;
                            for c in 0..3 {
                                let sum = self[x_index + 0][y_index + 0][c] as u16
                                    + self[x_index + 0][y_index + 1][c] as u16
                                    + self[x_index + 1][y_index + 0][c] as u16
                                    + self[x_index + 1][y_index + 1][c] as u16;
                                target[x][y][c] = (sum / 4) as u8;
                            }
                        }
                    }
                    target
                }
            }
        };
    }

    impl_avgpool!(32, 32);
    impl_avgpool!(16, 16);
    impl_avgpool!(8, 8);
    impl_avgpool!(4, 4);

    pub trait Normalize2D<P> {
        type OutputImage;
        fn normalize_2d(&self) -> Self::OutputImage;
    }

    // slide the min to 0
    impl<const X: usize, const Y: usize> Normalize2D<[u8; 3]> for [[[u8; 3]; Y]; X]
    where
        [[[u8; 3]; Y]; X]: Default,
    {
        type OutputImage = [[[u8; 3]; Y]; X];
        fn normalize_2d(&self) -> Self::OutputImage {
            let mut mins = [255u8; 3];
            for x in 0..X {
                for y in 0..Y {
                    for c in 0..3 {
                        mins[c] = self[x][y][c].min(mins[c]);
                    }
                }
            }
            let mut target = Self::OutputImage::default();
            for x in 0..X {
                for y in 0..Y {
                    for c in 0..3 {
                        target[x][y][c] = self[x][y][c] - mins[c];
                    }
                }
            }
            target
        }
    }

    pub trait ExtractPixels<P> {
        fn extract_pixels(&self, pixels: &mut Vec<P>);
    }

    impl<P: Copy, const X: usize, const Y: usize> ExtractPixels<P> for [[P; Y]; X] {
        fn extract_pixels(&self, pixels: &mut Vec<P>) {
            for x in 0..X {
                for y in 0..Y {
                    pixels.push(self[x][y]);
                }
            }
        }
    }

    pub trait PixelMap2D<I, O> {
        type OutputImage;
        fn map_2d<F: Fn(&I) -> O>(&self, map_fn: F) -> Self::OutputImage;
    }

    impl<I, O, const X: usize, const Y: usize> PixelMap2D<I, O> for [[I; Y]; X]
    where
        [[O; Y]; X]: Default,
    {
        type OutputImage = [[O; Y]; X];
        fn map_2d<F: Fn(&I) -> O>(&self, map_fn: F) -> Self::OutputImage {
            let mut target = <[[O; Y]; X]>::default();
            for x in 0..X {
                for y in 0..Y {
                    target[x][y] = map_fn(&self[x][y]);
                }
            }
            target
        }
    }

    pub trait Concat2D<A, B> {
        fn concat_2d(a: &A, b: &B) -> Self;
    }

    impl<A: Copy, B: Copy, const X: usize, const Y: usize> Concat2D<[[A; Y]; X], [[B; Y]; X]>
        for [[(A, B); Y]; X]
    where
        Self: Default,
    {
        fn concat_2d(a: &[[A; Y]; X], b: &[[B; Y]; X]) -> Self {
            let mut target = <Self>::default();
            for x in 0..X {
                for y in 0..Y {
                    target[x][y] = (a[x][y], b[x][y]);
                }
            }
            target
        }
    }

    // extracts patches and puts them in the pixels of the output image.
    pub trait Conv2D<OutputImage> {
        fn conv2d(&self) -> OutputImage;
    }

    macro_rules! impl_conv2d_2x2 {
        ($x:expr, $y:expr) => {
            impl<P: Copy + Default> Conv2D<[[[[P; 2]; 2]; $y / 2]; $x / 2]> for [[P; $y]; $x] {
                fn conv2d(&self) -> [[[[P; 2]; 2]; $y / 2]; $x / 2] {
                    let mut target = <[[[[P; 2]; 2]; $y / 2]; $x / 2]>::default();
                    for x in 0..$x / 2 {
                        let x_offset = x * 2;
                        for y in 0..$y / 2 {
                            let y_offset = y * 2;
                            for fx in 0..2 {
                                for fy in 0..2 {
                                    target[x][y][fx][fy] = self[x_offset + fx][y_offset + fy];
                                }
                            }
                        }
                    }
                    target
                }
            }
        };
    }

    impl_conv2d_2x2!(32, 32);
    impl_conv2d_2x2!(16, 16);
    impl_conv2d_2x2!(8, 8);
    impl_conv2d_2x2!(4, 4);

    impl<P: Copy, const X: usize, const Y: usize> Conv2D<[[[[P; 3]; 3]; Y]; X]> for [[P; Y]; X]
    where
        [[[[P; 3]; 3]; Y]; X]: Default,
    {
        fn conv2d(&self) -> [[[[P; 3]; 3]; Y]; X] {
            let mut target = <[[[[P; 3]; 3]; Y]; X]>::default();

            for fx in 1..3 {
                for fy in 1..3 {
                    target[0][0][fx][fy] = self[0 + fx][0 + fy];
                }
            }
            for y in 0..Y - 2 {
                for fx in 1..3 {
                    for fy in 0..3 {
                        target[0][y + 1][fx][fy] = self[0 + fx][y + fy];
                    }
                }
            }
            for fx in 1..3 {
                for fy in 0..2 {
                    target[0][Y - 1][fx][fy] = self[0 + fx][Y - 2 + fy];
                }
            }

            // begin center
            for x in 0..X - 2 {
                for fx in 0..3 {
                    for fy in 1..3 {
                        target[x + 1][0][fx][fy] = self[x + fx][0 + fy];
                    }
                }
                for y in 0..Y - 2 {
                    for fx in 0..3 {
                        for fy in 0..3 {
                            target[x + 1][y + 1][fx][fy] = self[x + fx][y + fy];
                        }
                    }
                }
                for fx in 0..3 {
                    for fy in 0..2 {
                        target[x + 1][Y - 1][fx][fy] = self[x + fx][Y - 2 + fy];
                    }
                }
            }
            // end center

            for fx in 0..2 {
                for fy in 1..3 {
                    target[X - 1][0][fx][fy] = self[X - 2 + fx][0 + fy];
                }
            }
            for y in 0..Y - 2 {
                for fx in 0..2 {
                    for fy in 0..3 {
                        target[X - 1][y + 1][fx][fy] = self[X - 2 + fx][y + fy];
                    }
                }
            }
            for fx in 0..2 {
                for fy in 0..2 {
                    target[X - 1][Y - 1][fx][fy] = self[X - 2 + fx][Y - 2 + fy];
                }
            }

            target
        }
    }
}
