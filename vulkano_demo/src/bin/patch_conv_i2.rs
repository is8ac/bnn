extern crate rand;
extern crate rayon;
extern crate time;
extern crate vulkano;
extern crate vulkano_demo;
extern crate vulkano_shaders;
use rand::prelude::*;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;
use time::PreciseTime;
use vulkano_demo::{
    Apply, FlipBit, ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEval,
    TestCPUObjectiveEvalCreator, VulkanObjectiveEval, VulkanObjectiveEvalCreator,
};

const N_EXAMPLES: usize = 50000 * 900;
const INPUT_LEN: usize = 2;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(43);

    let mut weights: [[[[u32; INPUT_LEN]; 3]; 3]; 32] = rng.gen();
    let mut head: [u32; 10] = rng.gen();

    let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..N_EXAMPLES)
        .map(|_| (rng.gen_range(0, 10), rng.gen()))
        .collect();

    let gpu_eval_creator = VulkanObjectiveEvalCreator::new();
    let mut gpu_obj_eval = gpu_eval_creator.new_obj_eval(&weights, &head, &examples);

    let cpu_eval_creator = TestCPUObjectiveEvalCreator::new();
    let mut cpu_obj_eval = cpu_eval_creator.new_obj_eval(&weights, &head, &examples);

    let start = PreciseTime::now();
    let gpu_obj = gpu_obj_eval.obj();
    println!(
        "GPU mills: {:?}",
        start.to(PreciseTime::now()).num_milliseconds()
    );

    for i in 0..32 {
        gpu_obj_eval.flip_weights_bit(2, i);
        cpu_obj_eval.flip_weights_bit(2, i);
    }

    let start = PreciseTime::now();
    let gpu_obj = gpu_obj_eval.obj();
    println!(
        "GPU mills: {:?}",
        start.to(PreciseTime::now()).num_milliseconds()
    );

    let start = PreciseTime::now();
    let gpu_obj = gpu_obj_eval.obj();
    println!(
        "GPU mills: {:?}",
        start.to(PreciseTime::now()).num_milliseconds()
    );

    dbg!(gpu_obj);

    let cpu_obj: u64 = cpu_obj_eval.obj();
    assert_eq!(gpu_obj, cpu_obj);
}
