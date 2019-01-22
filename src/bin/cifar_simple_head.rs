extern crate bincode;
extern crate bitnn;
extern crate rayon;
extern crate time;
use time::PreciseTime;

use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::{
    Conv2x2Stride2, NewFromSplit, OptimizeLayer, OrPool2x2, Patch3x3NotchedConv, PixelHead10,
    PixelMap, SaveLoad, VecApply,
};
use std::path::Path;

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

type Layer1 = Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>;

macro_rules! train_layer {
    ($layer_type:ty, $fs_path:expr, $examples:expr) => {{
        let layer = <$layer_type>::new_from_fs(Path::new($fs_path)).unwrap_or_else(|| {
            println!("{} not found, training", $fs_path);
            let mut layer = <$layer_type>::new_from_split(&$examples);
            let new_examples = layer.vec_apply(&$examples);
            let mut head = PixelHead10::<u32>::new_from_split(&new_examples);
            let start = PreciseTime::now();
            for i in 0..ITERS {
                let obj = layer.optimize_layer(&mut head, &$examples, MOD);
                println!("{} obj: {}", i, obj);
            }
            println!("optimize time: {:?}", start.to(PreciseTime::now()));
            layer.write_to_fs(Path::new($fs_path));
            layer
        });
        layer.vec_apply(&$examples)
    }};
}

const MOD: usize = 20;
const ITERS: usize = 5;

fn main() {
    let examples = load_data();
    let examples = train_layer!(Layer1, "params/simple_conv1.vecprms", examples);
    let examples = train_layer!(Layer1, "params/simple_conv2.vecprms", examples);
    let examples = train_layer!(Layer1, "params/simple_conv3.vecprms", examples);
    let mut pool_layer = Conv2x2Stride2::<u32, u128, [u128; 64], u64>::new_from_split(&examples);
    let new_examples = pool_layer.vec_apply(&examples);
    let mut head = PixelHead10::<u64>::new_from_split(&new_examples);
    for i in 0..ITERS {
        let obj = pool_layer.optimize_layer(&mut head, &examples, MOD);
        println!("{} obj: {:?}", i, obj);
    }
}

// 16: l1: 2 obj: 0.17929422222222224
// 32: l1: 1 obj: 0.1674630888888889
// 32: l1: 2 obj: 0.17097233333333334
// 32: l2: 0 obj: 0.1839521777777778
// 32: l2: 1 obj: 0.19115926666666666
// 32: l2: 2 obj: 0.19462657777777778
// 32: l2: 3 obj: 0.1971473111111111
// 32: l2: 4 obj: 0.20118255555555556
// 32: l3: 0 obj: 0.1968487111111111
// 32: l3: 4 obj: 0.2119747777777778

// 64: p4: 2 obj: 0.21677163265306118
// 64: p4: 3 obj: 0.21739704081632644
// 64: p4: 4 obj: 0.21751316326530604
