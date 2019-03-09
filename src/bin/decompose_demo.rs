extern crate rand;
extern crate time;

extern crate bincode;
extern crate bitnn;
extern crate image;
extern crate rand_hc;
extern crate rayon;
extern crate serde_derive;
use bitnn::objective_eval::{VulkanObjectiveEvalCreator, TestCPUObjectiveEvalCreator, ObjectiveEvalCreator, ObjectiveEval};
use rand::prelude::*;
use rand_hc::Hc128Rng;
use bitnn::datasets::cifar;
use bitnn::layers::Extract3x3Patches;
use rand::SeedableRng;
use std::path::Path;
use time::PreciseTime;
use std::iter;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(42);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let examples = cifar::load_images_from_base(cifar_base_path, 50000);
    let patches = {
        // decompose the images into patches,
        let mut patches: Vec<(u8, [[[u32; 1]; 3]; 3])> = examples
            .iter()
            .map(|(class, image)| iter::repeat(*class as u8).zip(image.patches()))
            .flatten()
            .collect();
        // and shuffle them
        patches.shuffle(&mut rng);
        patches
    };

    let mut weights: [[[[u32; 1]; 3]; 3]; 32] = rng.gen();
    let mut head: [u32; 10] = rng.gen();

    let gpu_eval_creator = VulkanObjectiveEvalCreator::new();
    let mut gpu_obj_eval = gpu_eval_creator.new_obj_eval(&weights, &head, &patches);
    let gpu_obj = gpu_obj_eval.obj();

    let cpu_eval_creator = TestCPUObjectiveEvalCreator::new();
    let mut cpu_obj_eval = cpu_eval_creator.new_obj_eval(&weights, &head, &patches);
    let cpu_obj = cpu_obj_eval.obj();

    assert_eq!(gpu_obj, cpu_obj);
}
