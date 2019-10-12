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
        labels
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
            for (p, &pixel) in images_bytes.iter().enumerate() {
                let word_index = p / 32;
                image_words[word_index] |= ((pixel > 128) as u32) << (p % 32);
            }
            images.push(image_words);
        }
        images
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
        images
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
                                image_bytes[(y * 32) + x],
                                image_bytes[1024 + (y * 32) + x],
                                image_bytes[2048 + (y * 32) + x],
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
