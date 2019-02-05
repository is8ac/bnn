extern crate bincode;
extern crate bitnn;
extern crate rand;
extern crate rayon;
extern crate time;
use bitnn::datasets::cifar;
use bitnn::layers::unary;
use bitnn::layers::{vec_concat_2_examples, vec_concat_4_examples};
use bitnn::layers::{
    Conv1x1, Conv2x2Stride2, Conv3x3, ExtractPixels, Image2D, MakePixelHead, NewFromRng,
    NewFromSplit, Objective, OptimizeHead, OptimizeLayer, Patch3x3NotchedConv, PixelHead10,
    PixelMap, SaveLoad, VecApply,
};
use bitnn::Patch;
use rand::isaac::Isaac64Rng;
use rand::rngs::ThreadRng;
use rand::{Rng, SeedableRng};
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::Path;
use time::PreciseTime;

fn load_data() -> Vec<(usize, [[u32; 32]; 32])> {
    let size: usize = 10_000;
    let paths = vec![
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
        .map(|(class, image)| (*class, image.pixel_map(&|input| unary::rgb_to_u32(*input))))
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
type Conv32_32 = Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>;
type Conv64_32 = Patch3x3NotchedConv<u64, [u128; 4], [[u128; 4]; 32], u32>;

trait Train<I, O, H> {
    fn train_new_layer<RNG: rand::Rng>(
        examples: &Vec<(usize, I)>,
        fs_path: &Path,
        rng: &mut RNG,
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
        L: OptimizeLayer<I, O>
            + NewFromSplit<I>
            + MakePixelHead<I, O, P>
            + VecApply<I, O>
            + SaveLoad
            + NewFromRng,
    > Train<I, O, PixelHead10<P>> for L
where
    PixelHead10<P>: OptimizeHead<O> + Objective<O> + Sync + NewFromRng,
{
    fn train_new_layer<RNG: rand::Rng>(
        examples: &Vec<(usize, I)>,
        fs_path: &Path,
        rng: &mut RNG,
        head_update_freq: usize,
        depth: usize,
    ) -> Vec<(usize, O)> {
        let layer = Self::new_from_fs(fs_path).unwrap_or_else(|| {
            println!("{} not found, training", &fs_path.to_str().unwrap());
            let mut layer: Self = Self::new_from_rng(rng);
            let mut head: PixelHead10<P> = PixelHead10::<P>::new_from_rng(rng);

            let start = PreciseTime::now();
            Self::train(&mut layer, &mut head, examples, head_update_freq, depth);
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

const SEED: u64 = 42;
const MOD: usize = 20;
const DEPTH: usize = 6;

fn main() {
    let mut rng = Isaac64Rng::seed_from_u64(SEED);
    let base_path = Path::new("params/1skip32");
    write_to_log_event(
        PreciseTime::now().to(PreciseTime::now()),
        0f64,
        0,
        &base_path.join("start"),
    );
    let start = PreciseTime::now();

    let mut s0 = load_data();
    dbg!(s0.len());
    let mut s1 =
        Conv32_32::train_new_layer(&s0, &base_path.join("l0_c3_32-32"), &mut rng, MOD, DEPTH);

    for i in 1..100 {
        let cc: Vec<(usize, [[u64; 32]; 32])> = vec_concat_2_examples(&s1, &s0);
        s0 = s1;
        s1 = Conv64_32::train_new_layer(
            &cc,
            &base_path.join(format!("l{}_c3_64-32", i)),
            &mut rng,
            MOD,
            DEPTH,
        );
    }

    println!("full duration: {:?}", start.to(PreciseTime::now()));
}
