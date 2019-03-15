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
use bitnn::objective_eval::VulkanObjectiveEvalCreator;
use bitnn::optimize::TrainLayer;
use bitnn::vec_concat_2_examples;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::fs;
use std::path::Path;
use time::PreciseTime;

const HEAD_UPDATE_FREQ: usize = 17;
// reduce sum batch size should have strictly no effect on obj.
const RS_BATCH: usize = 400;
// depth shoud have approximately no effect on run time.
const DEPTH: usize = 8;

fn main() {
    let log_file_path = Path::new("vk_train_log.txt");
    let base_path = Path::new("params/vk_mirror_test");
    fs::create_dir_all(base_path).unwrap();
    let mut rng = Hc128Rng::seed_from_u64(42);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let mut l0_images = cifar::load_images_from_base(cifar_base_path, 50_000);
    let eval_creator = VulkanObjectiveEvalCreator::new(RS_BATCH);

    let weights = [[[[[0u32; 1]; 3]; 3]; 16]; 1];
    let output: [[[u32; 1]; 32]; 32] = weights.apply(&l0_images[0].1);

    let start = PreciseTime::now();
    let mut l1_images: Vec<(usize, [[[u32; 1]; 32]; 32])> = <[[[[[u32; 1]; 3]; 3]; 16]; 1]>::train_from_images(
        &mut rng,
        &eval_creator,
        &l0_images,
        &base_path.join(format!("l{}_c3_32-32", 0)),
        DEPTH,
        HEAD_UPDATE_FREQ,
        &log_file_path,
    );

    for l in 1..7 {
        let c0_1_images: Vec<(usize, [[[u32; 2]; 32]; 32])> = vec_concat_2_examples(&l0_images, &l1_images);
        l0_images = l1_images;
        l1_images = <[[[[[u32; 2]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c0_1_images,
            &base_path.join(format!("l{}_c3_64-32", l)),
            DEPTH,
            HEAD_UPDATE_FREQ,
            &log_file_path,
        );
    }

    let c0_1_images: Vec<(usize, [[[u32; 2]; 32]; 32])> = vec_concat_2_examples(&l0_images, &l1_images);

    let mut l7_images: Vec<(usize, [[[u32; 2]; 16]; 16])> = <[[[[[u32; 2]; 2]; 2]; 32]; 2]>::train_from_images(
        &mut rng,
        &eval_creator,
        &c0_1_images,
        &base_path.join(format!("l{}_2x2_2-2", 7)),
        DEPTH,
        HEAD_UPDATE_FREQ,
        &log_file_path,
    );

    let mut l8_images = <[[[[[u32; 2]; 3]; 3]; 16]; 2]>::train_from_images(
        &mut rng,
        &eval_creator,
        &l7_images,
        &base_path.join(format!("l{}_3x2_2-2", 8)),
        DEPTH,
        HEAD_UPDATE_FREQ,
        &log_file_path,
    );
    for l in 9..30 {
        let c7_8_images: Vec<(usize, [[[u32; 4]; 16]; 16])> = vec_concat_2_examples(&l7_images, &l8_images);
        l7_images = l8_images;
        l8_images = <[[[[[u32; 4]; 3]; 3]; 16]; 2]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c7_8_images,
            &base_path.join(format!("l{}_3x2_4-2", l)),
            DEPTH,
            HEAD_UPDATE_FREQ,
            &log_file_path,
        );
    }

    println!("full time: {}", start.to(PreciseTime::now()));
}
