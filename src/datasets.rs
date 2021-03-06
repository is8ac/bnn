pub mod mnist {
    use crate::bits::b32;
    //use crate::image2d::StaticImage;
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
    pub fn load_images_bitpacked_u32(path: &Path, size: usize) -> Vec<[b32; 25]> {
        let path = Path::new(path);
        let mut file = File::open(&path).expect("can't open images");
        let mut header: [u8; 16] = [0; 16];
        file.read_exact(&mut header).expect("can't read header");

        let mut images_bytes: [u8; 784] = [0; 784];

        let mut images: Vec<[b32; 25]> = Vec::new();
        for _ in 0..size {
            file.read_exact(&mut images_bytes)
                .expect("can't read images");
            let mut image_words = <[b32; 25]>::default();
            for (p, &pixel) in images_bytes.iter().enumerate() {
                let word_index = p / 32;
                image_words[word_index].0 |= ((pixel > 128) as u32) << (p % 32);
            }
            images.push(image_words);
        }
        images
    }

    //pub fn load_images_u8(path: &Path, size: usize) -> Vec<StaticImage<u8, 28, 28>> {
    //    let mut file = File::open(&path).expect("can't open images");
    //    let mut header: [u8; 16] = [0; 16];
    //    file.read_exact(&mut header).expect("can't read header");

    //    let mut images_bytes: [u8; 784] = [0; 784];

    //    let mut images: Vec<StaticImage<u8, 28, 28>> = Vec::new();
    //    for _ in 0..size {
    //        file.read_exact(&mut images_bytes).expect("can't read images");
    //        let mut image = StaticImage::default();
    //        for p in 0..784 {
    //            image.image[p / 28][p % 28] = images_bytes[p];
    //        }
    //        images.push(image);
    //    }
    //    images
    //}
}

pub mod cifar {
    //use crate::image2d::StaticImage;
    use std::fs::File;
    use std::io::prelude::*;
    use std::path::Path;

    /// returns pair of (train, test)
    pub fn load_examples_from_base(
        base_path: &Path,
        (train_n, test_n): (usize, usize),
    ) -> (
        Vec<([[[u8; 3]; 32]; 32], usize)>,
        Vec<([[[u8; 3]; 32]; 32], usize)>,
    ) {
        if train_n > 50_000 {
            panic!("train n must be <= 50,000");
        }
        if test_n > 10_000 {
            panic!("test n must be <= 10,000");
        }
        let train_set: Vec<_> = (1..6)
            .map(|i| load_examples(&base_path.join(format!("data_batch_{}.bin", i)), 10000))
            .flatten()
            .take(train_n)
            .collect();
        let test_set: Vec<_> = load_examples(&base_path.join("test_batch.bin"), test_n);
        (train_set, test_set)
    }

    fn load_examples(path: &Path, n: usize) -> Vec<([[[u8; 3]; 32]; 32], usize)> {
        let mut file = File::open(path).expect("can't open data");

        let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
        let mut label: [u8; 1] = [0; 1];
        (0..n)
            .map(|_| {
                file.read_exact(&mut label).expect("can't read label");
                file.read_exact(&mut image_bytes)
                    .expect("can't read images");
                let mut image = [[[0_u8; 3]; 32]; 32];
                for x in 0..32 {
                    for y in 0..32 {
                        image[x][y] = [
                            image_bytes[0 + (y * 32) + x],
                            image_bytes[1024 + (y * 32) + x],
                            image_bytes[2048 + (y * 32) + x],
                        ];
                    }
                }
                (image, label[0] as usize)
            })
            .collect()
    }
}
