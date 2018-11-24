extern crate bitnn;
extern crate rayon;
extern crate time;
use rayon::prelude::*;

use bitnn::datasets::cifar;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, unary, ExtractPatches, ObjectiveHeadFC, Patch, PatchMap, SimplifyBits, WeightsMatrix};
use std::marker::PhantomData;
use time::PreciseTime;

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let size: usize = 1000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images = Vec::with_capacity(size * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, size);
        images.append(&mut batch);
    }
    images
}

fn rgb_to_u16(pixels: [u8; 3]) -> u16 {
    unary::to_5(pixels[0]) as u16 | ((unary::to_5(pixels[1]) as u16) << 5) | ((unary::to_6(pixels[2]) as u16) << 10)
}

trait PoolOR<T: Patch, O> {
    fn pool_or(&self) -> O;
}

macro_rules! patch_map_trait_2x2_pool {
    ($x_size:expr, $y_size:expr) => {
        impl<T: Patch + Default + Copy> PoolOR<T, [[T; $y_size / 2]; $y_size / 2]> for [[T; $y_size]; $x_size] {
            fn pool_or(&self) -> [[T; $y_size / 2]; $x_size / 2] {
                let mut output = [[T::default(); $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let x_base = x * 2;
                    for y in 0..$y_size / 2 {
                        let y_base = y * 2;
                        output[x][y] = self[x_base + 0][y_base + 0]
                            .bit_or(&self[x_base + 0][y_base + 1])
                            .bit_or(&self[x_base + 1][y_base + 0])
                            .bit_or(&self[x_base + 1][y_base + 1]);
                    }
                }
                output
            }
        }
    };
}

patch_map_trait_2x2_pool!(32, 32);
//patch_map_trait_2x2_pool!(16, 16);

//trait ObjectiveHead<I> {
//    fn acc(&self, &Vec<(usize, I)>) -> f64;
//    fn update(&mut self, &Vec<(usize, I)>);
//    fn new_from_split(&Vec<(usize, I)>) -> Self;
//}
//
//struct ORPooledObjectiveHead<O, R: ObjectiveHead<O>> {
//    readout: R,
//}
//
//impl<O, P: Copy + Default + Patch, I: PoolOR<P, O>, R: ObjectiveHead<O>> ORPooledObjectiveHead<O, R> {
//    fn apply(inputs: &Vec<(usize, I)>) -> &Vec<(usize, O)> {
//        &inputs.iter().map(|(class, input)| (*class, input.pool_or())).collect::<Vec<(usize, O)>>()
//    }
//}
//
//impl<O, P: Copy + Default + Patch, I: PoolOR<P, O>, R: ObjectiveHead<O>> ObjectiveHead<I> for ORPooledObjectiveHead<O, R> {
//    fn acc(&self, examples: &Vec<(usize, I)>) -> f64 {
//        let pooled_examples = Self::apply(&examples);
//        self.readout.acc(&pooled_examples)
//    }
//    fn update(&mut self, examples: &Vec<(usize, I)>) {
//        let pooled_examples = Self::apply(&examples);
//        self.readout.update(&pooled_examples);
//    }
//    fn new_from_split(examples: &Vec<(usize, I)>) -> Self {
//        let pooled_examples = Self::apply(&examples);
//        ORPooledObjectiveHead {
//            readout: R::new_from_split(&pooled_examples),
//        }
//    }
//}
//
//trait ConvWeightsMatrix<InputPixel: Patch, OutputPixel: Patch, InputPatch: Patch, InputImage: PatchMap<InputPixel, InputPatch, OutputImage, OutputPixel>, OutputImage> {
//    fn update_output(&self, &InputImage, &mut OutputImage, usize);
//}

fn main() {
    // load the data.
    let examples = load_data();

    let start = PreciseTime::now();
    // convert each pixels from 8bit rgb to to unary.
    let unary_examples: Vec<(usize, [[u16; 32]; 32])> = examples
        .par_iter()
        .map(|(class, image)| (*class, image.patch_map(&|input, output| *output = rgb_to_u16(*input))))
        .collect();

    // extract 8 pixel patches from all the images, simplify to u128, and flatten to a single Vec of labeled patches.
    let l1_patches: Vec<(usize, u128)> = unary_examples
        .par_iter()
        .map(|(class, image): &(usize, [[u16; 32]; 32])| image.extract_patches().iter().map(&|x: &[u16; 8]| (*class, x.simplify())).collect::<Vec<(usize, u128)>>())
        .flatten()
        .collect();

    // with these patches we can generate good intial weights.
    let mut hidden_layer = <[(u128, [u32; 4]); 32]>::new_from_split(&l1_patches);

    // now we can apply the weights to the images apply 2x2 OR pooling to get a 16 pixels per image.
    let l1_conved_patches: Vec<(usize, _)> = unary_examples
        .par_iter()
        .map(|(class, image)| {
            let patches: Vec<(usize, u128)> = image
                // first we map over 8 pixel patches to apply the conv. Then we apply 2x2 OR pooling.
                .patch_map(&|input: &[u16; 8], output| *output = hidden_layer.vecmul(&input.simplify()).simplify())
                .patch_map(&|i: &[u128; 4], t: &mut u128| {
                    *t = i[0] | i[1] | i[2] | i[3];
                }).extract_patches()
                .iter()
                .map(|x| (*class, *x))
                .collect();
            patches
        }).flatten()
        .collect();

    let mut readout_layer = <[_; 10]>::new_from_split(&l1_conved_patches);
    readout_layer.update(&l1_conved_patches);
    readout_layer.update(&l1_conved_patches);
    println!("acc: {}%", readout_layer.acc(&l1_conved_patches) * 100f64);

    //let n_correct: usize = unary_examples
    //    .par_iter()
    //    .map(|(class, image)| {
    //        let correct = image
    //            // first we map over 8 pixel patches to apply the conv. Then we apply 2x2 OR pooling.
    //            .patch_map(&|input: &[u16; 8], output| *output = hidden_layer.vecmul(&input.simplify()).simplify())
    //            .patch_map(&|i: &[_; 4], t| {
    //                *t = i[0] | i[1] | i[2] | i[3];
    //            }).extract_patches()
    //            .iter()
    //            .map();
    //    }).sum();
    println!("{}", start.to(PreciseTime::now()));
}
