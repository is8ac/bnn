extern crate bincode;
extern crate bitnn;
extern crate rayon;
extern crate time;
use time::PreciseTime;

use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::{
    Accuracy, Layer, NewFromSplit, OptimizeHead, OptimizeLayer, OrPool2x2, Patch3x3NotchedConv,
    PatchMap, SaveLoad, VecApply,
};
use std::path::Path;

fn load_data() -> Vec<(usize, [[u32; 32]; 32])> {
    let size: usize = 10000;
    let paths = vec![
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        String::from("cifar-10-batches-bin/data_batch_1.bin"),
        String::from("cifar-10-batches-bin/data_batch_2.bin"),
        String::from("cifar-10-batches-bin/data_batch_3.bin"),
        String::from("cifar-10-batches-bin/data_batch_4.bin"),
        String::from("cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images: Vec<(usize, [[[u8; 3]; 32]; 32])> = Vec::with_capacity(size * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, size);
        images.append(&mut batch);
    }
    images
        .iter()
        .map(|(class, image)| {
            (
                *class,
                image.patch_map(&|input, output: &mut u32| *output = unary::rgb_to_u32(*input)),
            )
        })
        .collect()
}

type Layer1 = Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>;
type ExpandLayer1 = Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 64], u64>;
type Layer2 = Patch3x3NotchedConv<u64, [u128; 4], [[u128; 4]; 64], u64>;

const MOD: usize = 30;
const ITERS: usize = 5;

macro_rules! train_layer {
    ($layer_type:ty, $fs_path:expr, $examples:expr) => {{
        let l1 = <$layer_type>::new_from_fs(Path::new($fs_path)).unwrap_or_else(|| {
            println!("{} not found, training", $fs_path);
            let params = <$layer_type>::train(&$examples, MOD, ITERS);
            params.write_to_fs(Path::new($fs_path));
            params
        });
        l1.vec_apply(&$examples)
    }};
}

fn main() {
    let examples = load_data();

    let examples = train_layer!(Layer1, "params/convlayer0.vecprms", examples);
    let examples = train_layer!(Layer1, "params/convlayer1.vecprms", examples);
    let examples = train_layer!(Layer1, "params/convlayer2.vecprms", examples);
    let examples: Vec<(usize, [[u64; 32]; 32])> = train_layer!(ExpandLayer1, "params/convlayer_expand.vecprms", examples);
    let pool_layer = OrPool2x2::default();
    let examples = pool_layer.vec_apply(&examples);
    let examples = train_layer!(Layer2, "params/convlayer3.vecprms", examples);
    let examples = train_layer!(Layer2, "params/convlayer4.vecprms", examples);
    let examples: Vec<(usize, [[u64; 16]; 16])> = train_layer!(Layer2, "params/convlayer5.vecprms", examples);
    // pass: 4, obj: 0.2972111111111111
    let examples = train_layer!(Layer2, "params/convlayer6.vecprms", examples);
    let examples = train_layer!(Layer2, "params/convlayer7.vecprms", examples);
    let examples = train_layer!(Layer2, "params/convlayer8.vecprms", examples);
}
