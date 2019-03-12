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
use bitnn::objective_eval::{ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEvalCreator, VulkanObjectiveEvalCreator};
use bitnn::{vec_concat_2_examples, vec_extract_patches, BitLen, ConcatImages, Extract3x3Patches, FlipBit};
use rand::prelude::*;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufWriter;
use std::iter;
use std::path::Path;
use time::PreciseTime;

trait TrainLayer<EvalCreator, InputImage, OutputImage> {
    fn train_from_images<RNG: rand::Rng>(
        rng: &mut RNG,
        eval_creator: &EvalCreator,
        images: &Vec<(usize, InputImage)>,
        fs_path: &Path,
        depth: usize,
    ) -> Vec<(usize, OutputImage)>;
}

macro_rules! impl_trainlayer_for_weights {
    ($input_len:expr, $output_type:ty) => {
        impl<
                EvalCreator: ObjectiveEvalCreator<[[[u32; $input_len]; 3]; 3], [[[[u32; $input_len]; 3]; 3]; 32], $output_type>,
                InputImage: Sync + Extract3x3Patches<[u32; $input_len]>,
                OutputImage: Sync + Send,
            > TrainLayer<EvalCreator, InputImage, OutputImage> for [[[[u32; $input_len]; 3]; 3]; 32]
        where
            Self: Apply<InputImage, OutputImage>,
            <EvalCreator as ObjectiveEvalCreator<[[[u32; $input_len]; 3]; 3], [[[[u32; $input_len]; 3]; 3]; 32], u32>>::ObjectiveEvalType:
                ObjectiveEval<[[[u32; $input_len]; 3]; 3], [[[[u32; $input_len]; 3]; 3]; 32], u32>,
        {
            fn train_from_images<RNG: rand::Rng>(
                rng: &mut RNG,
                eval_creator: &EvalCreator,
                images: &Vec<(usize, InputImage)>,
                fs_path: &Path,
                depth: usize,
            ) -> Vec<(usize, OutputImage)> {
                let weights = Self::new_from_fs(fs_path).unwrap_or_else(|| {
                    println!("{} not found, training", &fs_path.to_str().unwrap());
                    let patches: Vec<(u8, [[[u32; $input_len]; 3]; 3])> = vec_extract_patches(rng, &images);

                    let mut weights: [[[[u32; $input_len]; 3]; 3]; 32] = rng.gen();
                    let mut head: [u32; 10] = rng.gen();

                    let start = PreciseTime::now();
                    let obj = <[[[u32; $input_len]; 3]; 3]>::recurs_train(eval_creator, &mut weights, &mut head, &patches, depth);
                    println!("obj: {}, time: {}", obj, start.to(PreciseTime::now()));
                    write_to_log_event(fs_path, start.to(PreciseTime::now()), obj, depth, images.len());
                    weights.write_to_fs(&fs_path);
                    weights
                });

                images.par_iter().map(|(class, image)| (*class, weights.apply(image))).collect()
            }
        }
    };
}

impl_trainlayer_for_weights!(1, u32);
impl_trainlayer_for_weights!(2, u32);

trait Train<EvalCreator, InputPatch, Weights, Embedding> {
    fn train_pass(eval_creator: &EvalCreator, weights: &mut Weights, head: &mut [Embedding; 10], patches: &[(u8, InputPatch)]) -> u64;
    fn recurs_train(eval_creator: &EvalCreator, weights: &mut Weights, head: &mut [Embedding; 10], patches: &[(u8, InputPatch)], depth: usize)
        -> f64;
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
                dbg!(patches.len());

                let mut gpu_obj_eval = eval_creator.new_obj_eval(&weights, &head, &patches);

                let start = PreciseTime::now();
                let mut cur_obj = gpu_obj_eval.obj();
                for e in 0..32 {
                    let mut iter = 0;
                    for i in 0..<[[[u32; $len]; 3]; 3]>::BIT_LEN {
                        if iter % HEAD_UPDATE_FREQ == 0 {
                            for o in 0..10 {
                                gpu_obj_eval.flip_head_bit(o, e);
                                let new_obj = gpu_obj_eval.obj();
                                if new_obj >= cur_obj {
                                    cur_obj = new_obj;
                                    println!("head: {} {}: {} {}", e, o, new_obj, new_obj as f64 / patches.len() as f64);
                                    head[o].flip_bit(e);
                                } else {
                                    gpu_obj_eval.flip_head_bit(o, e);
                                }
                            }
                            iter += 1;
                        }
                        gpu_obj_eval.flip_weights_bit(e, i);
                        let new_obj = gpu_obj_eval.obj();
                        if new_obj >= cur_obj {
                            iter += 1;
                            cur_obj = new_obj;
                            println!("{} {}: {} {}", e, i, new_obj, new_obj as f64 / patches.len() as f64);
                            weights[e].flip_bit(i);
                        } else {
                            gpu_obj_eval.flip_weights_bit(e, i);
                        }
                    }
                }
                println!("{} {}", patches.len(), start.to(PreciseTime::now()));
                cur_obj
            }
            fn recurs_train(
                eval_creator: &EvalCreator,
                weights: &mut [[[[u32; $len]; 3]; 3]; 32],
                head: &mut [u32; 10],
                patches: &[(u8, [[[u32; $len]; 3]; 3])],
                depth: usize,
            ) -> f64 {
                if depth == 0 {
                    Self::train_pass(eval_creator, weights, head, &patches[0..patches.len() / 2]);
                } else {
                    Self::recurs_train(eval_creator, weights, head, &patches[0..patches.len() / 2], depth - 1);
                }
                Self::train_pass(eval_creator, weights, head, &patches[patches.len() / 2..]) as f64 / (patches.len() / 2) as f64
            }
        }
    };
}
impl_train_for_3x3patch!(1);
impl_train_for_3x3patch!(2);

fn write_to_log_event(layer_name: &Path, duration: time::Duration, obj: f64, depth: usize, n: usize) {
    let mut file = OpenOptions::new().write(true).append(true).open("vk_train_log.txt").unwrap();
    writeln!(
        file,
        "{} depth: {}, obj: {}, rs_batch: {}, head_update_freq: {}, n: {}, {}",
        layer_name.to_str().unwrap(),
        depth,
        obj,
        RS_BATCH,
        HEAD_UPDATE_FREQ,
        n,
        duration,
    )
    .unwrap();
}

const HEAD_UPDATE_FREQ: usize = 17;
// reduce sum batch size should have strictly no effect on obj.
const RS_BATCH: usize = 400;
// depth shoud have approximately no effect on run time.
const DEPTH: usize = 8;

fn main() {
    let base_path = Path::new("params/vk_test_greateq");
    fs::create_dir_all(base_path).unwrap();
    let mut rng = Hc128Rng::seed_from_u64(42);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let mut l0_images = cifar::load_images_from_base(cifar_base_path, 50_000);

    let start = PreciseTime::now();
    let mut l1_images = {
        let eval_creator = VulkanObjectiveEvalCreator::new(RS_BATCH);
        let mut l1_images: Vec<(usize, [[[u32; 1]; 32]; 32])> =
            <[[[[u32; 1]; 3]; 3]; 32]>::train_from_images(&mut rng, &eval_creator, &l0_images, &base_path.join(format!("l{}_c3_32-32", 0)), DEPTH);
        l1_images
    };

    for l in 1..100 {
        let c0_1_images: Vec<(usize, [[[u32; 2]; 32]; 32])> = vec_concat_2_examples(&l0_images, &l1_images);
        l0_images = l1_images;
        let eval_creator = VulkanObjectiveEvalCreator::new(RS_BATCH);
        l1_images =
            <[[[[u32; 2]; 3]; 3]; 32]>::train_from_images(&mut rng, &eval_creator, &c0_1_images, &base_path.join(format!("l{}_c3_64-32", l)), DEPTH);
    }

    println!("full time: {}", start.to(PreciseTime::now()));
}
