extern crate bincode;
extern crate bitnn;
extern crate rayon;
extern crate time;
use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::{vec_concat_2_examples, vec_concat_4_examples};
use bitnn::layers::{
    Conv1x1, Conv2x2Stride2, Conv3x3, ExtractPixels, Image2D, MakePixelHead, NewFromSplit,
    Objective, OptimizeHead, OptimizeLayer, Patch3x3NotchedConv, PixelHead10, PixelMap, SaveLoad,
    VecApply,
};
use bitnn::Patch;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;
use time::PreciseTime;

fn load_data() -> Vec<(usize, [[u16; 32]; 32])> {
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
        .map(|(class, image)| (*class, image.pixel_map(&|input| unary::rgb_to_u16(*input))))
        .collect()
}

fn write_to_log_event(duration: time::Duration, obj: f64, depth: usize, layer_name: &Path) {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("train_log_vgg_small.txt")
        .unwrap();
    writeln!(
        file,
        "{}, {}, obj: {}, {}",
        layer_name.to_str().unwrap(),
        depth,
        obj,
        duration
    )
    .unwrap();
}

type Conv16 = Patch3x3NotchedConv<u16, u128, [u128; 16], u16>;
type Conv32_16 = Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 16], u16>;
type Conv64_16 = Patch3x3NotchedConv<u64, [u128; 4], [[u128; 4]; 16], u16>;

type Pool2x2_16_32 = Conv2x2Stride2<u16, u64, [u64; 32], u32>;
type Conv32 = Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>;
type Pool2x2_32_64 = Conv2x2Stride2<u32, u128, [u128; 64], u64>;
type Conv64 = Patch3x3NotchedConv<u64, [u128; 4], [[u128; 4]; 64], u64>;
type Pool2x2_64_128 = Conv2x2Stride2<u64, [u128; 2], [[u128; 2]; 128], u128>;
type Conv128 = Conv3x3<u128, [[[u128; 3]; 3]; 128], u128>;
type DR64_32 = Conv1x1<u64, [u64; 32], u32>;

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
            println!("{} not found, training", &fs_path.to_str().unwrap());
            let mut layer = Self::new_from_split(examples);
            let mut head = layer.make_pixel_head(examples);
            let start = PreciseTime::now();
            Self::train(&mut layer, &mut head, examples, head_update_freq, depth);
            //layer.optimize_layer::<PixelHead10<P>>(&mut head, &examples, head_update_freq);
            //let new_examples: Vec<(usize, O)> = layer.vec_apply(examples);
            //let obj = head.avg_objective(&new_examples);
            let (obj, updates) =
                layer.optimize_layer::<PixelHead10<P>>(&mut head, &examples, head_update_freq);
            println!(
                "obj: {}, depth: {}, {}",
                obj,
                depth,
                start.to(PreciseTime::now())
            );
            write_to_log_event(start.to(PreciseTime::now()), obj, depth, &fs_path);
            layer.write_to_fs(&fs_path);
            layer
        });
        layer.vec_apply(&examples)
    }
    // run time is ~constant with depth.
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
const DEPTH: usize = 6;

fn main() {
    let base_path = Path::new("params/densenet_small");
    write_to_log_event(
        PreciseTime::now().to(PreciseTime::now()),
        0f64,
        0,
        &base_path.join("start"),
    );
    let s0 = load_data();
    dbg!(s0.len());
    let s1 = Conv16::train_new_layer(&s0, &base_path.join("l1_c3_16-16"), MOD, DEPTH);

    let cc0_1: Vec<(usize, [[u32; 32]; 32])> = vec_concat_2_examples(&s0, &s1);
    let s2 = Conv32_16::train_new_layer(&cc0_1, &base_path.join("l2_c3_32-16"), MOD, DEPTH);

    let cc1_2: Vec<(usize, [[u32; 32]; 32])> = vec_concat_2_examples(&s1, &s2);
    let s3 = Conv32_16::train_new_layer(&cc1_2, &base_path.join("l3_c3_32-16"), MOD, DEPTH);

    let cc2_3: Vec<(usize, [[u32; 32]; 32])> = vec_concat_2_examples(&s2, &s3);
    let s4 = Conv32_16::train_new_layer(&cc2_3, &base_path.join("l4_c3_32-16"), MOD, DEPTH);

    let cc0_3: Vec<(usize, [[u64; 32]; 32])> = vec_concat_4_examples(&s0, &s1, &s2, &s3);
    let s4 = Conv64_16::train_new_layer(&cc0_3, &base_path.join("l4_c3_64-16"), MOD, DEPTH);
}
