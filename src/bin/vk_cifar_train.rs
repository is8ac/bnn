extern crate rand;
extern crate time;

extern crate bincode;
extern crate bitnn;
extern crate image;
extern crate rand_hc;
extern crate rayon;
extern crate serde_derive;
use bitnn::datasets::cifar;
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
    let base_path = Path::new("params/vk_array_test");
    fs::create_dir_all(base_path).unwrap();
    let mut rng = Hc128Rng::seed_from_u64(42);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let mut l0_images = cifar::load_images_from_base(cifar_base_path, 50_000);
    let eval_creator = VulkanObjectiveEvalCreator::new(RS_BATCH);

    let start = PreciseTime::now();
    let mut l1_images: Vec<(usize, [[[u32; 1]; 32]; 32])> = <[[[[[u32; 1]; 3]; 3]; 32]; 1]>::train_from_images(
        &mut rng,
        &eval_creator,
        &l0_images,
        &base_path.join(format!("l{}_c3_32-32", 0)),
        DEPTH,
        HEAD_UPDATE_FREQ,
        &log_file_path,
    );

    for l in 1..20 {
        let c0_1_images: Vec<(usize, [[[u32; 2]; 32]; 32])> = vec_concat_2_examples(&l0_images, &l1_images);
        l0_images = l1_images;
        l1_images = <[[[[[u32; 2]; 3]; 3]; 32]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c0_1_images,
            &base_path.join(format!("l{}_c3_64-32", l)),
            DEPTH,
            HEAD_UPDATE_FREQ,
            &log_file_path,
        );
    }

    println!("full time: {}", start.to(PreciseTime::now()));
}
