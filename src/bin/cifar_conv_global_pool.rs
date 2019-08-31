#![feature(const_generics)]

use bitnn::datasets::cifar;
use bitnn::image2d::{Conv2D, Normalize2D, PixelMap2D};
use bitnn::layer::{GlobalPoolingBayes, SupervisedLayer, UnsupervisedLayer};
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs::create_dir_all;
use std::path::Path;
use time::PreciseTime;

trait Classes<const C: usize> {
    fn max_index(&self) -> usize;
    fn min_index(&self) -> usize;
}
impl<const C: usize> Classes<{ C }> for [u32; C] {
    fn max_index(&self) -> usize {
        let mut max_val = 0;
        let mut max_index = 0;
        for i in 0..C {
            if self[i] > max_val {
                max_val = self[i];
                max_index = i;
            }
        }
        max_index
    }
    fn min_index(&self) -> usize {
        let mut max_val = !0;
        let mut max_index = 0;
        for i in 0..C {
            if self[i] < max_val {
                max_val = self[i];
                max_index = i;
            }
        }
        max_index
    }
}

trait Unary<Bits> {
    fn unary_encode(&self) -> Bits;
}

impl Unary<u32> for [u8; 3] {
    fn unary_encode(&self) -> u32 {
        let mut bits = 0b0u32;
        for c in 0..3 {
            bits |= (!0b0u32 << (self[c] / 23)) << (c * 10);
            bits &= !0b0u32 >> (32 - ((c + 1) * 10));
        }
        !bits
    }
}

fn print_patch<const X: usize, const Y: usize>(patch: &[[u32; Y]; X]) {
    for x in 0..X {
        for y in 0..Y {
            print!("{:032b} | ", patch[x][y]);
        }
        print!("\n");
    }
    println!("-----------",);
}

trait PrintBits {
    fn print_bits(&self);
}

impl PrintBits for u32 {
    fn print_bits(&self) {
        print!("{:032b}", self);
    }
}

impl<T: PrintBits, const L: usize> PrintBits for [T; L] {
    fn print_bits(&self) {
        print!("[");
        for i in 0..L - 1 {
            self[i].print_bits();
            print!(",");
        }
        self[L - 1].print_bits();
        print!("]");
    }
}

trait PrintWeights {
    fn weight_bits(&self);
}

impl<Pixel: PrintBits, const C: usize> PrintWeights for [[(Pixel, u32); 32]; C] {
    fn weight_bits(&self) {
        for c in 0..C {
            for b in 0..32 {
                println!("{}: ", self[c][b].1);
                self[c][b].0.print_bits();
                print!("\n");
            }
        }
    }
}

trait PrintImage {
    fn image_pixels(&self);
    fn image_channels(&self);
}

impl<const C: usize, const Y: usize, const X: usize> PrintImage for [[[u32; C]; Y]; X] {
    fn image_pixels(&self) {
        for x in 0..X {
            for y in 0..Y {
                for c in 0..C {
                    print!("{:032b}", self[x][y][c]);
                }
                print!("\n",);
            }
            println!("----------");
        }
    }
    fn image_channels(&self) {
        for w in 0..C {
            for b in 0..32 {
                for x in 0..X {
                    for y in 0..Y {
                        let bit = ((self[x][y][w] >> b) & 1) == 1;
                        print!("{}", if bit { 1 } else { 0 });
                    }
                    print!("\n",);
                }
                println!("---------");
            }
        }
    }
}

fn split_examples<Image: Sync + Clone + Send + Copy>(
    examples: &Vec<(Image, usize)>,
) -> (Vec<Image>, Vec<usize>) {
    (
        examples.par_iter().map(|(image, _)| *image).collect(),
        examples.par_iter().map(|(_, label)| *label).collect(),
    )
}

fn zip_examples<Image: Sync + Clone + Send + Copy>(
    images: &Vec<Image>,
    labels: &Vec<usize>,
) -> Vec<(Image, usize)> {
    images
        .par_iter()
        .cloned()
        .zip(labels.par_iter().cloned())
        .collect()
}

fn conv_normalize_raw_images<const X: usize, const Y: usize>(
    raw_images: &Vec<[[[u8; 3]; Y]; X]>,
) -> Vec<[[[[u32; 3]; 3]; Y]; X]>
where
    [[[u8; 3]; Y]; X]: Conv2D<[[[[[u8; 3]; 3]; 3]; Y]; X]>,
    [[[[[u8; 3]; 3]; 3]; Y]; X]:
        PixelMap2D<[[[u8; 3]; 3]; 3], [[u32; 3]; 3], OutputImage = [[[[u32; 3]; 3]; Y]; X]>,
{
    raw_images
        .par_iter()
        .map(|image| {
            <[[[u8; 3]; Y]; X] as Conv2D<[[[[[u8; 3]; 3]; 3]; Y]; X]>>::conv2d(&image)
                .map_2d(|patch| patch.normalize_2d().map_2d(|pixel| pixel.unary_encode()))
        })
        .collect()
}

const N_EXAMPLES: usize = 50_000;
const B0_CHANS: usize = 1;
const B1_CHANS: usize = 2;
//const B2_CHANS: usize = 3;
//const B3_CHANS: usize = 4;

fn main() {
    let start = PreciseTime::now();
    let mut rng = Hc128Rng::seed_from_u64(0);

    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(24))
        //.num_threads(4)
        .build_global()
        .unwrap();

    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    let params_path = Path::new("params/cluster_split_5");
    create_dir_all(&params_path).unwrap();
    println!("init time: {}", start.to(PreciseTime::now()));

    let examples: Vec<([[[u8; 3]; 32]; 32], usize)> =
        cifar::load_images_from_base(cifar_base_path, N_EXAMPLES);
    let (raw_images, labels) = split_examples(&examples);
    let normalized_images = conv_normalize_raw_images(&raw_images);

    let unary_images: Vec<_> = raw_images
        .iter()
        .map(|image| image.map_2d(|pixel| pixel.unary_encode()))
        .collect();

    {
        let examples = zip_examples(&unary_images, &labels);
        let _: [(u32, u32); 10] = <[[u32; 32]; 32]>::global_pooling_bayes(&examples);
    }

    let (images, weights) = <[[([[u32; 3]; 3], u32); 32]; B0_CHANS]>::unsupervised_cluster(
        &mut rng,
        &normalized_images,
        &params_path.join("b0_l0.prms"),
    );
    //weights.weight_bits();
    //images[6].image_channels();

    //let conved_images: Vec<[[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32]> = images
    //    .par_iter()
    //    .map(|image| {
    //        <[[[u32; B0_CHANS]; 32]; 32] as Conv2D<[[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32]>>::conv2d(
    //            image,
    //        )
    //    })
    //    .collect();
    //let (images, weights) = <[[([[[u32; 1]; 3]; 3], u32); 32]; B0_CHANS]>::unsupervised_cluster(
    //    &mut rng,
    //    &conved_images,
    //    &params_path.join("b0_l1.prms"),
    //);
    //weights.weight_bits();
    //images[6].image_channels();

    //let conved_images: Vec<[[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32]> = images
    //    .par_iter()
    //    .map(|image| {
    //        <[[[u32; B0_CHANS]; 32]; 32] as Conv2D<[[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32]>>::conv2d(
    //            image,
    //        )
    //    })
    //    .collect();
    //let examples: Vec<([[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32], usize)> = conved_images
    //    .par_iter()
    //    .cloned()
    //    .zip(labels.par_iter().cloned())
    //    .collect();
    //let (split_images, split_weights): (Vec<([[[u32; B0_CHANS]; 32]; 32], usize)>, _) =
    //    <[[([[[u32; B0_CHANS]; 3]; 3], u32); 32]; B0_CHANS] as SupervisedLayer<
    //        [[[[[u32; B0_CHANS]; 3]; 3]; 32]; 32],
    //        [[[u32; B0_CHANS]; 32]; 32],
    //    >>::supervised_split(&examples, 10, &params_path.join("b0_l0_split.prms"));
    //split_weights.weight_bits();
    //split_images[6].0.image_channels();

    //let weights: [([u32; B0_CHANS], u32); 10] =
    //    <[[[u32; B0_CHANS]; 32]; 32]>::global_pooling_bayes(&split_images);
    ////dbg!(&weights);
    //let n_correct: usize = split_images
    //    .iter()
    //    .map(|(image, class)| {
    //        let activations = image.infer(&weights);
    //        //println!("{:?}", activations);
    //        let actual_class = activations.min_index();
    //        (actual_class == *class) as usize
    //    })
    //    .sum();
    //println!(
    //    "{:?}%",
    //    n_correct as f64 / split_images.len() as f64 * 100.0
    //);
}
