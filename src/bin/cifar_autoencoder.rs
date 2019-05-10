extern crate bitnn;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;

use bitnn::datasets::cifar;
use bitnn::layers::{Conv2D, SaveLoad};
use bitnn::train::{CacheBatch, TrainConv};
use bitnn::{
    Apply, BitLen, ExtractPatches, GetBit, Image2D, TrainAutoencoder, TrainEncoderSupervised,
};
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs;
use std::iter;
use std::path::Path;
use time::PreciseTime;

pub trait TrainConvLayerAutoencoder<InputImage, Patch, OutputImage, Embedding, Decoder> {
    fn conv_autoencoder<RNG: rand::Rng>(
        rng: &mut RNG,
        input: &Vec<(InputImage, usize)>,
        params_path: &Path,
    ) -> Vec<(OutputImage, usize)>;
}

impl<
        InputImage: Sync + Image2D + ExtractPatches<Patch>,
        Patch: Copy + Sync + Send,
        Encoder: TrainAutoencoder<Patch, OutputImage::Pixel, Decoder>
            + SaveLoad
            + Conv2D<InputImage, OutputImage>
            + Sync,
        Decoder,
        OutputImage: Sync + Image2D + Send,
    > TrainConvLayerAutoencoder<InputImage, Patch, OutputImage, OutputImage::Pixel, Decoder>
    for Encoder
{
    fn conv_autoencoder<RNG: rand::Rng>(
        rng: &mut RNG,
        examples: &Vec<(InputImage, usize)>,
        params_path: &Path,
    ) -> Vec<(OutputImage, usize)> {
        let encoder = Self::new_from_fs(params_path).unwrap_or_else(|| {
            println!("{} not found, training", &params_path.to_str().unwrap());

            let patches: Vec<Patch> = examples
                .iter()
                .map(|(image, _)| {
                    let patches: Vec<Patch> = image.patches().iter().cloned().collect();
                    patches
                })
                .flatten()
                .collect();
            dbg!(patches.len());

            let encoder =
                <Self as TrainAutoencoder<Patch, OutputImage::Pixel, Decoder>>::train_autoencoder(
                    rng, &patches,
                );
            encoder.write_to_fs(&params_path);
            encoder
        });

        let output_examples: Vec<(OutputImage, usize)> = examples
            .par_iter()
            .map(|(image, class)| (encoder.conv2d(image), *class))
            .collect();
        output_examples
    }
}

const N_EXAMPLES: usize = 10_00;
const EMBEDDING_LEN: usize = 1;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(28))
        .build_global()
        .unwrap();
    let path_string = format!("params/cifar_autoencoder_test_2_n{}", N_EXAMPLES);
    let base_path = Path::new(&path_string);
    fs::create_dir_all(base_path).unwrap();

    let mut rng = Hc128Rng::seed_from_u64(1);
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");

    let chan3_images: Vec<([[[u32; 1]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);

    let mut examples = <[[[[[u32; 1]; 3]; 3]; 32]; EMBEDDING_LEN] as TrainConvLayerAutoencoder<
        _,
        _,
        [[[u32; 1]; 32]; 32],
        _,
        [[[[[u32; EMBEDDING_LEN]; 32]; 1]; 3]; 3],
    >>::conv_autoencoder(&mut rng, &chan3_images, &base_path.join("l1"));

    examples = <[[[[[u32; 1]; 3]; 3]; 32]; EMBEDDING_LEN] as TrainConvLayerAutoencoder<
        _,
        _,
        [[[u32; 1]; 32]; 32],
        _,
        [[[[[u32; EMBEDDING_LEN]; 32]; 1]; 3]; 3],
    >>::conv_autoencoder(&mut rng, &examples, &base_path.join("l2"));

    for l in 0..20 {
        let start = PreciseTime::now();
        examples = <[[[[[u32; 1]; 3]; 3]; 32]; 1] as TrainConv<
            [[[u32; 1]; 32]; 32],
            [[[u32; 1]; 32]; 32],
            [[[u32; 1]; 3]; 3],
            _,
            [u32; 1],
            CacheBatch<_, _, _>,
        >>::train(
            &mut rng,
            &examples,
            &base_path.join(format!("b0_l{}", l)),
            9,
            20,
        );
        println!("time: {}", start.to(PreciseTime::now()));
    }
}
