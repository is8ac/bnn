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
use bitnn::{vec_concat_2_examples, vec_extract_patches, BitLen, ConcatImages, Extract3x3Patches, FlipBit, FlipBitIndexed};
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

trait TrainLayer<EvalCreator, Pixel, Embedding, InputImage, OutputImage> {
    fn train_from_images<RNG: rand::Rng>(
        rng: &mut RNG,
        eval_creator: &EvalCreator,
        images: &Vec<(usize, InputImage)>,
        fs_path: &Path,
        depth: usize,
    ) -> Vec<(usize, OutputImage)>;
}

impl<
        Pixel,
        Weights: SaveLoad + Sync,
        Embedding,
        EvalCreator: ObjectiveEvalCreator<[[Pixel; 3]; 3], Weights, Embedding>,
        InputImage: Sync + Extract3x3Patches<Pixel>,
        OutputImage: Sync + Send,
    > TrainLayer<EvalCreator, Pixel, Embedding, InputImage, OutputImage> for Weights
where
    Self: Apply<InputImage, OutputImage>,
    <EvalCreator as ObjectiveEvalCreator<[[Pixel; 3]; 3], Weights, Embedding>>::ObjectiveEvalType: ObjectiveEval<[[Pixel; 3]; 3], Weights, Embedding>,
    rand::distributions::Standard: rand::distributions::Distribution<Weights>,
    rand::distributions::Standard: rand::distributions::Distribution<Embedding>,
    [[Pixel; 3]; 3]: Train<EvalCreator, [[Pixel; 3]; 3], Weights, Embedding>,
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
            let patches: Vec<(u8, [[Pixel; 3]; 3])> = vec_extract_patches(rng, &images);

            let mut weights: Weights = rng.gen();
            let mut head: [Embedding; 10] = rng.gen();

            let start = PreciseTime::now();
            let obj = <[[Pixel; 3]; 3]>::recurs_train(eval_creator, &mut weights, &mut head, &patches, depth);
            println!("obj: {}, time: {}", obj, start.to(PreciseTime::now()));
            write_to_log_event(fs_path, start.to(PreciseTime::now()), obj, depth, images.len());
            weights.write_to_fs(&fs_path);
            weights
        });

        images.par_iter().map(|(class, image)| (*class, weights.apply(image))).collect()
    }
}

trait Train<EvalCreator, InputPatch, Weights, Embedding> {
    fn train_pass(eval_creator: &EvalCreator, weights: &mut Weights, head: &mut [Embedding; 10], patches: &[(u8, InputPatch)]) -> u64;
    fn recurs_train(eval_creator: &EvalCreator, weights: &mut Weights, head: &mut [Embedding; 10], patches: &[(u8, InputPatch)], depth: usize)
        -> f64;
}

impl<Patch: BitLen, Weights: FlipBitIndexed, Embedding: FlipBit, EvalCreator: ObjectiveEvalCreator<Patch, Weights, Embedding>>
    Train<EvalCreator, Patch, Weights, Embedding> for Patch
where
    EvalCreator::ObjectiveEvalType: ObjectiveEval<Patch, Weights, Embedding>,
{
    fn train_pass(eval_creator: &EvalCreator, weights: &mut Weights, head: &mut [Embedding; 10], patches: &[(u8, Patch)]) -> u64 {
        dbg!(patches.len());

        let mut gpu_obj_eval = eval_creator.new_obj_eval(&weights, &head, &patches);

        let start = PreciseTime::now();
        let mut cur_obj = gpu_obj_eval.obj();
        for e in 0..32 {
            let mut iter = 0;
            for i in 0..<Patch>::BIT_LEN {
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
                    weights.flip_bit_indexed(e, i);
                } else {
                    gpu_obj_eval.flip_weights_bit(e, i);
                }
            }
        }
        println!("{} {}", patches.len(), start.to(PreciseTime::now()));
        cur_obj
    }
    fn recurs_train(eval_creator: &EvalCreator, weights: &mut Weights, head: &mut [Embedding; 10], patches: &[(u8, Patch)], depth: usize) -> f64 {
        if depth == 0 {
            Self::train_pass(eval_creator, weights, head, &patches[0..patches.len() / 2]);
        } else {
            Self::recurs_train(eval_creator, weights, head, &patches[0..patches.len() / 2], depth - 1);
        }
        Self::train_pass(eval_creator, weights, head, &patches[patches.len() / 2..]) as f64 / (patches.len() / 2) as f64
    }
}

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
    let base_path = Path::new("params/vk_array_test");
    fs::create_dir_all(base_path).unwrap();
    let mut rng = Hc128Rng::seed_from_u64(42);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let mut l0_images = cifar::load_images_from_base(cifar_base_path, 50_000);

    let start = PreciseTime::now();
    let mut l1_images = {
        let eval_creator = VulkanObjectiveEvalCreator::new(RS_BATCH);
        let mut l1_images: Vec<(usize, [[[u32; 1]; 32]; 32])> = <[[[[[u32; 1]; 3]; 3]; 32]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &l0_images,
            &base_path.join(format!("l{}_c3_32-32", 0)),
            DEPTH,
        );
        l1_images
    };

    for l in 1..100 {
        let c0_1_images: Vec<(usize, [[[u32; 2]; 32]; 32])> = vec_concat_2_examples(&l0_images, &l1_images);
        l0_images = l1_images;
        let eval_creator = VulkanObjectiveEvalCreator::new(RS_BATCH);
        l1_images = <[[[[[u32; 2]; 3]; 3]; 32]; 1]>::train_from_images(
            &mut rng,
            &eval_creator,
            &c0_1_images,
            &base_path.join(format!("l{}_c3_64-32", l)),
            DEPTH,
        );
    }

    println!("full time: {}", start.to(PreciseTime::now()));
}
