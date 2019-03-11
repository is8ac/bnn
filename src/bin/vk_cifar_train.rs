extern crate rand;
extern crate time;

extern crate bincode;
extern crate bitnn;
extern crate image;
extern crate rand_hc;
extern crate rayon;
extern crate serde_derive;
use bitnn::datasets::cifar;
use bitnn::layers::Extract3x3Patches;
use bitnn::objective_eval::{ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEvalCreator, VulkanObjectiveEvalCreator};
use bitnn::{BitLen, FlipBit};
use rand::prelude::*;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::iter;
use std::path::Path;
use time::PreciseTime;

const HEAD_UPDATE_FREQ: usize = 50;

trait Train<EvalCreator, InputPatch, Weights, Embedding> {
    fn train_pass(eval_creator: &EvalCreator, weights: &mut Weights, head: &mut [Embedding; 10], patches: &[(u8, InputPatch)]) -> u64;
}

macro_rules! impl_train_for_3x3patch {
    ($len:expr) => {
        impl<EvalCreator: ObjectiveEvalCreator<[[[u32; $len]; 3]; 3], [[[[u32; $len]; 3]; 3]; 32], u32>>
            Train<EvalCreator, [[[u32; $len]; 3]; 3], [[[[u32; $len]; 3]; 3]; 32], u32> for [[[u32; $len]; 3]; 3]
        where
            EvalCreator::ObjectiveEvalType: ObjectiveEval<[[[u32; $len]; 3]; 3], [[[[u32; $len]; 3]; 3]; 32], u32>,
        {
            fn train_pass(
                eval_creator: &EvalCreator,
                weights: &mut [[[[u32; $len]; 3]; 3]; 32],
                head: &mut [u32; 10],
                patches: &[(u8, [[[u32; $len]; 3]; 3])],
            ) -> u64 {
                //dbg!(patches.len());

                let mut gpu_obj_eval = eval_creator.new_obj_eval(&weights, &head, &patches);

                let start = PreciseTime::now();
                let mut cur_obj = gpu_obj_eval.obj();
                let mut iter = 0;
                for o in 0..32 {
                    for i in 0..<[[[u32; 1]; 3]; 3]>::BIT_LEN {
                        if iter % HEAD_UPDATE_FREQ == 0 {
                            for hi in 0..32 {
                                for ho in o..10 {
                                    gpu_obj_eval.flip_head_bit(ho, hi);
                                    let new_obj = gpu_obj_eval.obj();
                                    if new_obj > cur_obj {
                                        cur_obj = new_obj;
                                        //println!("head: {} {}: {} {}", o, i, new_obj, new_obj as f64 / patches.len() as f64);
                                        head[o].flip_bit(i);
                                    } else {
                                        gpu_obj_eval.flip_head_bit(ho, hi);
                                    }
                                }
                            }
                        }
                        gpu_obj_eval.flip_weights_bit(o, i);
                        let new_obj = gpu_obj_eval.obj();
                        if new_obj > cur_obj {
                            iter += 1;
                            cur_obj = new_obj;
                            //println!("{} {}: {} {}", o, i, new_obj, new_obj as f64 / patches.len() as f64);
                            weights[o].flip_bit(i);
                        } else {
                            gpu_obj_eval.flip_weights_bit(o, i);
                        }
                    }
                }
                //println!("{} {}", patches.len(), start.to(PreciseTime::now()));
                cur_obj
            }
        }
    };
}
impl_train_for_3x3patch!(1);

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(42);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let examples = cifar::load_images_from_base(cifar_base_path, 10_000);
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

    for rs_batch_size in 74..154 {
        let eval_creator = VulkanObjectiveEvalCreator::new(rs_batch_size);

        let start = PreciseTime::now();
        //let obj = <[[[u32; 1]; 3]; 3]>::train_pass(&eval_creator, &mut weights, &mut head, &patches[0..patches.len() / 32]);
        //let obj = <[[[u32; 1]; 3]; 3]>::train_pass(&eval_creator, &mut weights, &mut head, &patches[patches.len() / 32..patches.len() / 16]);
        //let obj = <[[[u32; 1]; 3]; 3]>::train_pass(&eval_creator, &mut weights, &mut head, &patches[patches.len() / 16..patches.len() / 8]);
        //let obj = <[[[u32; 1]; 3]; 3]>::train_pass(&eval_creator, &mut weights, &mut head, &patches[patches.len() / 8..patches.len() / 4]);
        //let obj = <[[[u32; 1]; 3]; 3]>::train_pass(&eval_creator, &mut weights, &mut head, &patches[patches.len() / 4..patches.len() / 2]);
        let obj = <[[[u32; 1]; 3]; 3]>::train_pass(&eval_creator, &mut weights, &mut head, &patches[patches.len() / 2..patches.len()]);
        println!("rs batch: {} time: {}", rs_batch_size, start.to(PreciseTime::now()));
    }
    let l2_images: Vec<(usize, [[[u32; 1]; 3]; 3])> = examples.par_iter().map(|(class, image)| weights.apply(image)).collect();
}
