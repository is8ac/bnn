extern crate bitnn;
extern crate rayon;
extern crate serde;
extern crate serde_json;
extern crate time;
use bitnn::datasets::cifar;
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use time::PreciseTime;

use bitnn::{layers, featuregen};
use bitnn::layers::{bitvecmul, pack_3x3, pixelmap, unary, Layer2d, Patch};

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images = Vec::with_capacity(10000 * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, 10000);
        images.append(&mut batch);
    }
    images
}

fn split_by_label<T: Copy>(examples: &Vec<(usize, T)>, len: usize) -> Vec<Vec<T>> {
    let mut by_label: Vec<Vec<T>> = (0..len).map(|_| Vec::new()).collect();
    for (label, example) in examples {
        by_label[*label].push(*example);
    }
    let _: Vec<_> = by_label.iter_mut().map(|x| x.shrink_to_fit()).collect();
    by_label
}



fn main() {
    let start = PreciseTime::now();
    let images = load_data();
    let u14_packed_images: Vec<(usize, [[u16; 32]; 32])> = images
        .par_iter()
        .map(|(label, image)| (*label, pixelmap::pm_32(&image, &unary::rgb_to_u14)))
        .collect();

    let by_label = split_by_label(&u14_packed_images, 10);
    let l1_basepoints: [(u128, u32); 32] = {
        let path = Path::new("l1_base_points.prms");
        let mut file_opt = File::open(&path);
        if file_opt.is_err() {
            // if we can't open the file, we will need to compute the l1_basepoints from scratch.
            let mut shards: Vec<Vec<Vec<u128>>> = vec![
                by_label
                    .par_iter()
                    .map(|imgs| {
                        imgs.iter()
                            .map(|image| image.to_3x3_patches())
                            .flatten()
                            .map(|pixels| pack_3x3::p14(pixels))
                            .collect()
                    })
                    .collect(),
            ];
            println!("load + data preparation time: {}", start.to(PreciseTime::now()));

            let mut features_vec = vec![];

            let seed = 0b11111111111111111111111111111111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000000000u128;
            let mut bit_distances: Vec<u32> = shards.iter().flatten().flatten().map(|y| y.hamming_distance(&seed)).collect();
            bit_distances.par_sort();
            let threshold = bit_distances[bit_distances.len() / 2];
            features_vec.push((seed, threshold));
            shards = featuregen::split_labels_set_by_distance(&shards, &seed, threshold);
            println!("{:0128b} {:} \t {:?}", seed, threshold, shards.len());

            for i in 0..3 {
                for class in 0..10 {
                    let (base_point, threshold) = featuregen::gen_basepoint(&shards, class);
                    features_vec.push((base_point, threshold));
                    shards = featuregen::split_labels_set_by_distance(&shards, &base_point, threshold);
                    println!("{:0128b} {:} \t {:?} \t {:?}", base_point, threshold, shards.len(), class);
                    //let lens: Vec<Vec<usize>> = shards.iter().map(|x| x.iter().map(|y| y.len()).collect()).collect();
                    //println!("{:?}", lens);
                }
            }
            let mut features = [(0u128, 0u32); 32];
            for (i, feature) in features_vec.iter().enumerate() {
                features[i] = *feature;
            }
            let text = serde_json::to_string(&features).unwrap();
            let mut file = File::create(&"l1_base_points.prms".to_string()).unwrap();
            file.write_all(text.as_bytes()).unwrap();
            features
        } else {
            let mut file = file_opt.unwrap();
            let mut text = String::new();
            file.read_to_string(&mut text).unwrap();
            let features: [(u128, u32); 32] = serde_json::from_str(&text).unwrap();
            features
        }
    };
    let start = PreciseTime::now();

    let l2_by_labels: Vec<Vec<[[u32; 16]; 16]>> = by_label
        .par_iter()
        .map(|class| {
            class
                .par_iter()
                .map(|image| {
                    let l2_pixels: Vec<_> = image
                        .to_3x3_patches()
                        .iter()
                        .map(|pixels| bitvecmul::bvm_u32(&l1_basepoints, &pack_3x3::p14(*pixels)))
                        .collect();
                    let image32 = <[[u32; 32]; 32]>::from_pixels_1_padding(&l2_pixels);
                    pixelmap::pool_or_32(&image32)
                })
                .collect()
        })
        .collect();
    println!("l2 time: {}", start.to(PreciseTime::now()));
    let fc_patches: Vec<Vec<[u128; 64]>> = l2_by_labels
        .iter()
        .map(|x| {
            x.iter()
                .map(|image| pack_3x3::array_pack_u32_u128_64(&pack_3x3::flatten_16x16(image)))
                .collect()
        })
        .collect();
    for x in fc_patches[0][0].iter() {
        println!("{:0128b}", x);
    }
    let shards = vec![l2_by_labels];
    //let mut features_vec = vec![];
    //for class in 0..10 {
    //    let (base_point, threshold) = gen_basepoint(&shards, class);
    //    features_vec.push((base_point, threshold));
    //}
}
