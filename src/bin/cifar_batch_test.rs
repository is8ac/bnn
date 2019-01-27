extern crate bincode;
extern crate bitnn;
extern crate rayon;
extern crate time;
use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::{
    Conv2x2Stride2, Conv3x3, ExtractPixels, Image2D, MakePixelHead, NewFromSplit, Objective,
    OptimizeHead, OptimizeLayer, Patch3x3NotchedConv, PixelHead10, PixelMap, SaveLoad, VecApply,
};
use bitnn::Patch;
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
    )
    .unwrap();
}

type Conv16 = Patch3x3NotchedConv<u16, u128, [u128; 16], u16>;
type Pool2x2_16_32 = Conv2x2Stride2<u16, u64, [u64; 32], u32>;
type Conv32 = Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>;
type Pool2x2_32_64 = Conv2x2Stride2<u32, u128, [u128; 64], u64>;
type Conv64 = Patch3x3NotchedConv<u64, [u128; 4], [[u128; 4]; 64], u64>;
type Pool2x2_64_128 = Conv2x2Stride2<u64, [u128; 2], [[u128; 2]; 128], u128>;
type Conv128 = Conv3x3<u128, [[[u128; 3]; 3]; 128], u128>;

trait Train<I, O, H> {
    fn train_new_layer(
        examples: &Vec<(usize, I)>,
        fs_path: &Path,
        head_update_freq: usize,
        depth: usize,
    ) -> Vec<(usize, O)>;
    fn train(
        layer: &mut Self,
        head: &mut H,
        examples_batch: &[(usize, I)],
        head_update_freq: usize,
        depth: usize,
    );
}

macro_rules! train_layer {
    ($layer_type:ty, $fs_path:expr, $examples:expr) => {{
        let layer = <$layer_type>::new_from_fs(Path::new($fs_path)).unwrap_or_else(|| {
            println!("{} not found, training", $fs_path);
            let mut layer = <$layer_type>::new_from_split(&$examples);
            let mut head = layer.make_pixel_head(&$examples);
            let start = PreciseTime::now();
            for i in 0..ITERS {
                let iter_start = PreciseTime::now();
                let obj = layer.optimize_layer(&mut head, &$examples, MOD);
                write_to_log_event(iter_start.to(PreciseTime::now()), obj, $fs_path, i);
                println!("{} obj: {}", i, obj);
            }
            println!("optimize time: {:?}", start.to(PreciseTime::now()));
            layer.write_to_fs(Path::new($fs_path));
            layer
        });
        layer.vec_apply(&$examples)
    }};
}

impl<
        I: Sync,
        O: Sync,
        P: Copy + Default + Sync,
        L: OptimizeLayer<I, O> + NewFromSplit<I> + MakePixelHead<I, O, P> + VecApply<I, O> + SaveLoad,
    > Train<I, O, PixelHead10<P>> for L
where
    PixelHead10<P>: OptimizeHead<O> + Objective<O> + Sync,
{
    fn train_new_layer(
        examples: &Vec<(usize, I)>,
        fs_path: &Path,
        head_update_freq: usize,
        depth: usize,
    ) -> Vec<(usize, O)> {
        let layer = Self::new_from_fs(fs_path).unwrap_or_else(|| {
            let mut layer = Self::new_from_split(examples);
            let mut head = layer.make_pixel_head(examples);
            let start = PreciseTime::now();
            Self::train(&mut layer, &mut head, examples, head_update_freq, depth);
            //layer.optimize_layer::<PixelHead10<P>>(&mut head, &examples, head_update_freq);
            let (obj, updates) =
                layer.optimize_layer::<PixelHead10<P>>(&mut head, &examples, head_update_freq);
            println!(
                "obj: {}, depth: {}, {}",
                obj,
                depth,
                start.to(PreciseTime::now())
            );
            layer
        });
        layer.vec_apply(&examples)
    }
    fn train(
        layer: &mut Self,
        head: &mut PixelHead10<P>,
        examples: &[(usize, I)],
        head_update_freq: usize,
        depth: usize,
    ) {
        dbg!(depth);
        if depth > 0 {
            Self::train(
                layer,
                head,
                &examples[0..examples.len() / 2],
                head_update_freq,
                depth - 1,
            );
        } else {
            layer.optimize_layer::<PixelHead10<P>>(
                head,
                &examples[0..examples.len() / 2],
                head_update_freq,
            );
        }
        dbg!(examples.len());
        let (obj, updates) = layer.optimize_layer::<PixelHead10<P>>(
            head,
            &examples[(examples.len() / 2)..],
            head_update_freq,
        );
        println!("depth: {}, obj: {}, updates: {:?}", depth, obj, updates);
    }
}
const MOD: usize = 7;
//const ITERS: usize = 20;

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
    let examples = Conv32::train_new_layer(&examples, Path::new("params/chan32_base/conv1_32.vecprms"), MOD, 6);
    //let examples = train_layer!(Conv16, "params/batch_test/conv1.vecprms", examples);
    //let examples = train_layer!(Conv16, "params/batch_test/conv2.vecprms", examples);
    //let examples = train_layer!(Conv16, "params/batch_test/conv3.vecprms", examples);
    //let examples = train_layer!(Conv16, "params/batch_test/conv4.vecprms", examples);
    //let examples = train_layer!(Conv16, "params/batch_test/conv5.vecprms", examples);
}
// MOD: 20
// obj: 0.1795872222222222,  depth: 3, PT965.950173838S
// obj: 0.18120546666666665, depth: 4, PT961.309522735S
// obj: 0.18230419999999997, depth: 5, PT961.848137223S
// obj: 0.18377104444444442, depth: 6, PT1017.590428990S
// obj: 0.17532048888888888, depth: 7, PT954.682125976S

// MOD 7:
// obj: 0.18389286666666663, depth: 6, PT1022.072527604S
// obj: 0.1842907111111111, depth: 6, PT1504.277525121S

// obj: 0.19333304444444444, depth: 6, PT5386.694501952S
