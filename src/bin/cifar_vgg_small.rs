extern crate bincode;
extern crate bitnn;
extern crate rayon;
extern crate time;
use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::{
    Conv2x2Stride2, Conv3x3, MakePixelHead, NewFromSplit, OptimizeLayer, Patch3x3NotchedConv,
    PixelHead10, PixelMap, SaveLoad, VecApply,
};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;
use time::PreciseTime;

fn load_data() -> Vec<(usize, [[u32; 32]; 32])> {
    let size: usize = 10_000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images: Vec<(usize, [[[u8; 3]; 32]; 32])> = Vec::with_capacity(size * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, size);
        images.append(&mut batch);
    }
    images
        .iter()
        .map(|(class, image)| (*class, image.pixel_map(&|input| unary::rgb_to_u32(*input))))
        .collect()
}

fn write_to_log_event(
    duration: time::Duration,
    obj: f64,
    layer_name: &str,
    iter: usize,
    updates: u64,
) {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("train_log_vgg_small.txt")
        .unwrap();
    writeln!(
        file,
        "{}, {}, obj: {}, updates: {}, {}",
        layer_name, iter, obj, updates, duration
    );
}

type Conv16 = Patch3x3NotchedConv<u16, u128, [u128; 16], u16>;
type Pool2x2_16_32 = Conv2x2Stride2<u16, u64, [u64; 32], u32>;
type Conv32 = Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>;
type Pool2x2_32_64 = Conv2x2Stride2<u32, u128, [u128; 64], u64>;
type Conv64 = Patch3x3NotchedConv<u64, [u128; 4], [[u128; 4]; 64], u64>;
//type Conv128 = Conv3x3<u128, [[[u128; 3]; 3]; 128], u128>;

macro_rules! train_layer {
    ($layer_type:ty, $fs_path:expr, $examples:expr) => {{
        let layer = <$layer_type>::new_from_fs(Path::new($fs_path)).unwrap_or_else(|| {
            println!("{} not found, training", $fs_path);
            let mut layer = <$layer_type>::new_from_split(&$examples);
            let mut head = layer.make_pixel_head(&$examples);
            let start = PreciseTime::now();
            for i in 0..ITERS {
                let iter_start = PreciseTime::now();
                let (obj, updates) = layer.optimize_layer(&mut head, &$examples, MOD);
                write_to_log_event(iter_start.to(PreciseTime::now()), obj, $fs_path, i, updates);
                println!("{} obj: {}, updates: {}", i, obj, updates);
                if updates == 0 {
                    dbg!("Layer is Pareto optimal");
                    break;
                }
            }
            println!("optimize time: {:?}", start.to(PreciseTime::now()));
            layer.write_to_fs(Path::new($fs_path));
            layer
        });
        layer.vec_apply(&$examples)
    }};
}

const MOD: usize = 13;
const ITERS: usize = 20;

fn main() {
    write_to_log_event(
        PreciseTime::now().to(PreciseTime::now()),
        0f64,
        &"starting train",
        0,
        0,
    );
    let examples = load_data();
    dbg!(examples.len());
    let examples = train_layer!(Conv32, "params/chan32_base/conv1.vecprms", examples);
    let examples = train_layer!(Conv32, "params/chan32_base/conv2.vecprms", examples);
    let examples = train_layer!(Conv32, "params/chan32_base/conv3.vecprms", examples);
    let examples = train_layer!(Conv32, "params/chan32_base/conv4.vecprms", examples);
    let examples = train_layer!(Conv32, "params/chan32_base/conv5.vecprms", examples);
    //let examples = train_layer!(Pool2x2_16_32, "params/small/convpool4.vecprms", examples);
    //let examples = train_layer!(Conv32, "params/small/conv5.vecprms", examples);
    //let examples = train_layer!(Conv32, "params/small/conv6.vecprms", examples);
    //let examples = train_layer!(Pool2x2_32_64, "params/small/convpool9.vecprms", examples);
    //let examples = train_layer!(Conv64, "params/small/conv10.vecprms", examples);
    //let examples = train_layer!(Conv64, "params/small/conv11.vecprms", examples);
    //let examples = train_layer!(Conv64, "params/small/conv11.vecprms", examples);
}
