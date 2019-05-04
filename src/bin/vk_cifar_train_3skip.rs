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
use bitnn::vec_concat_3_examples;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::fs;
use std::path::Path;
use time::PreciseTime;

const HEAD_UPDATE_FREQ: usize = 60;

fn main() {
    let log_file_path = Path::new("vk_train_log.txt");
    let base_path = Path::new("params/vk_3skip_b");
    fs::create_dir_all(base_path).unwrap();
    let mut rng = Hc128Rng::seed_from_u64(42);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let chan3_images: Vec<(usize, [[[u32; 3]; 32]; 32])> =
        cifar::load_images_from_base(cifar_base_path, 50_000);
    let mut l0_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        cifar::load_images_from_base(cifar_base_path, 50_000);
    let eval_creator = VulkanFastCacheObjectiveEvalCreator::new();

    let start = PreciseTime::now();
    let mut l1_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        <[[[[[u32; 3]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &chan3_images,
            &base_path.join("b1_l0_c3_3-1"),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            2,
        );
    let c0_1_images: Vec<(usize, [[[u32; 2]; 32]; 32])> =
        vec_concat_2_examples(&l0_images, &l1_images);

    let mut l2_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
        <[[[[[u32; 2]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c0_1_images,
            &base_path.join("b1_l1_c3_2-1"),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            2,
        );

    for l in 2..20 {
        let c0_1_2_images: Vec<(usize, [[[u32; 3]; 32]; 32])> =
            vec_concat_3_examples(&l0_images, &l1_images, &l2_images);
        l0_images = l1_images;
        l1_images = l2_images;
        l2_images = <[[[[[u32; 3]; 3]; 3]; 16]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c0_1_2_images,
            &base_path.join(format!("b1_l{}_c3_3-1", l)),
            11,
            HEAD_UPDATE_FREQ,
            &log_file_path,
            1,
        );
    }

    //let c0_1_images: Vec<(usize, [[[u32; 2]; 32]; 32])> = vec_concat_2_examples(&l0_images, &l1_images);

    //let mut l0_images: Vec<(usize, [[[u32; 2]; 16]; 16])> = <[[[[[u32; 3]; 2]; 2]; 16]; 2]>::train_from_images(
    //    &mut rng,
    //    &eval_creator,
    //    &chan3_images,
    //    &base_path.join(format!("b2_l{}_2x2_3-2", 0)),
    //    9,
    //    HEAD_UPDATE_FREQ,
    //    &log_file_path,
    //    2,
    //);

    //let mut l1_images: Vec<(usize, [[[u32; 2]; 16]; 16])> = <[[[[[u32; 2]; 2]; 2]; 16]; 2]>::train_from_images(
    //    &mut rng,
    //    &eval_creator,
    //    &c0_1_images,
    //    &base_path.join(format!("b2_l{}_2x2_2-2", 1)),
    //    9,
    //    HEAD_UPDATE_FREQ,
    //    &log_file_path,
    //    2,
    //);

    //for l in 2..20 {
    //    let c0_1_images: Vec<(usize, [[[u32; 4]; 16]; 16])> = vec_concat_2_examples(&l0_images, &l1_images);
    //    l0_images = l1_images;
    //    l1_images = <[[[[[u32; 4]; 3]; 3]; 16]; 2]>::train_from_images(
    //        &mut rng,
    //        &eval_creator,
    //        &c0_1_images,
    //        &base_path.join(format!("b2_l{}_3x3_4-2", l)),
    //        9,
    //        HEAD_UPDATE_FREQ,
    //        &log_file_path,
    //        2,
    //    );
    //}

    //let c0_1_images: Vec<(usize, [[[u32; 4]; 16]; 16])> = vec_concat_2_examples(&l0_images, &l1_images);
    //let mut l0_images: Vec<(usize, [[[u32; 4]; 8]; 8])> = <[[[[[u32; 4]; 2]; 2]; 16]; 4]>::train_from_images(
    //    &mut rng,
    //    &eval_creator,
    //    &c0_1_images,
    //    &base_path.join(format!("b3_l{}_2x2_4-4", 0)),
    //    9,
    //    HEAD_UPDATE_FREQ,
    //    &log_file_path,
    //    2,
    //);

    //let mut l1_images = <[[[[[u32; 4]; 3]; 3]; 16]; 4]>::train_from_images(
    //    &mut rng,
    //    &eval_creator,
    //    &l0_images,
    //    &base_path.join(format!("b3_l{}_3x3_4-4", 1)),
    //    7,
    //    HEAD_UPDATE_FREQ,
    //    &log_file_path,
    //    3,
    //);
    //for l in 2..30 {
    //    let c0_1_images: Vec<(usize, [[[u32; 8]; 8]; 8])> = vec_concat_2_examples(&l0_images, &l1_images);
    //    l0_images = l1_images;
    //    l1_images = <[[[[[u32; 8]; 3]; 3]; 16]; 4]>::train_from_images(
    //        &mut rng,
    //        &eval_creator,
    //        &c0_1_images,
    //        &base_path.join(format!("b3_l{}_3x3_8-4", l)),
    //        7,
    //        HEAD_UPDATE_FREQ,
    //        &log_file_path,
    //        3,
    //    );
    //}

    println!("full time: {}", start.to(PreciseTime::now()));
}
