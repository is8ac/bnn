extern crate rand;
extern crate time;

extern crate bincode;
extern crate bitnn;
extern crate image;
extern crate rand_hc;
extern crate rayon;
extern crate serde_derive;
use bitnn::datasets::cifar;
use bitnn::layers::{Apply, SaveLoad};
use bitnn::objective_eval::VulkanFastCacheObjectiveEvalCreator;
use bitnn::optimize::TrainLayer;
use bitnn::vec_concat_2_examples;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::fs;
use std::path::Path;
use time::PreciseTime;

const HEAD_UPDATE_FREQ: usize = 60;

fn main() {
    let log_file_path = Path::new("vk_train_log.txt");
    let base_path = Path::new("params/vk_dense_1");
    fs::create_dir_all(base_path).unwrap();
    let mut rng = Hc128Rng::seed_from_u64(42);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let mut c1_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        cifar::load_images_from_base(cifar_base_path, 50_000);
    let eval_creator = VulkanFastCacheObjectiveEvalCreator::new();

    let start = PreciseTime::now();
    let l1_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        <[[[[[u32; 1]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c1_images,
            &base_path.join("b1_l0_c3_1-1"),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            2,
        );

    let c2_images: Vec<(usize, [[[u32; 2]; 32]; 32])> =
        vec_concat_2_examples(&c1_images, &l1_images);
    let l2_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        <[[[[[u32; 2]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c2_images,
            &base_path.join("b1_l1_c3_2-1"),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            2,
        );

    let c3_images: Vec<(usize, [[[u32; 3]; 32]; 32])> =
        vec_concat_2_examples(&c2_images, &l2_images);
    let l3_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        <[[[[[u32; 3]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c3_images,
            &base_path.join("b1_l2_c3_3-1"),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            2,
        );

    let c4_images: Vec<(usize, [[[u32; 4]; 32]; 32])> =
        vec_concat_2_examples(&c3_images, &l3_images);
    let l4_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        <[[[[[u32; 4]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c4_images,
            &base_path.join("b1_l3_c3_4-1"),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            2,
        );

    let c5_images: Vec<(usize, [[[u32; 5]; 32]; 32])> =
        vec_concat_2_examples(&c4_images, &l4_images);
    let l5_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        <[[[[[u32; 5]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c5_images,
            &base_path.join("b1_l4_c3_5-1"),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            2,
        );

    let c6_images: Vec<(usize, [[[u32; 6]; 32]; 32])> =
        vec_concat_2_examples(&c5_images, &l5_images);
    let l6_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        <[[[[[u32; 6]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c6_images,
            &base_path.join("b1_l5_c3_6-1"),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            2,
        );

    println!("full time: {}", start.to(PreciseTime::now()));
}
