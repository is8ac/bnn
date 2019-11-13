extern crate bitnn;
extern crate num_cpus;
extern crate rayon;

use bitnn::bits::b32;
use bitnn::datasets::cifar;
use bitnn::image2d::{AvgPool, BitPool, StaticImage};
use bitnn::layer;
use bitnn::shape::ZipMap;
use rayon::prelude::*;
use std::path::Path;

const N_EXAMPLES: usize = 10_00;
const N_CLASSES: usize = 10;

macro_rules! conv {
    ($examples:expr, $dim:expr, $in_size:expr, $out_size:expr) => {{
        let (_, new_examples) =
            <[[([[[(b32, b32); $in_size]; 3]; 3], u32); 32]; $out_size] as layer::Layer<
                StaticImage<[b32; $in_size], $dim, $dim>,
                [[[b32; $in_size]; 3]; 3],
                [[[b32; $in_size]; 3]; 3],
                [(); N_CLASSES],
            >>::gen(&$examples);
        new_examples
    }};
}

macro_rules! patch_norm_conv {
    ($examples:expr, $dim:expr) => {{
        let (_, bit_examples) = <[[([[[(b32, b32); 3]; 3]; 3], u32); 32]; 1] as layer::Layer<
            StaticImage<[u8; 3], $dim, $dim>,
            [[[u8; 3]; 3]; 3],
            [[[b32; 3]; 3]; 3],
            [(); N_CLASSES],
        >>::gen(&$examples);
        bit_examples
    }};
}

fn main() {
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(28))
        //.num_threads(2)
        .build_global()
        .unwrap();

    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let bit_examples_32 = patch_norm_conv!(int_examples_32, 32);
    let foo = conv!(bit_examples_32, 32, 1, 1);

    //let int_examples_16 =
    //    <[[(); 32]; 32] as layer::AvgPoolLayer<[u8; 3]>>::avg_pool(&int_examples_32);
    //let bit_examples_16 = patch_norm_conv!(int_examples_16, 16);
    //let foo = conv!(bit_examples_16, 16, 1, 1);

    //let int_examples_8 =
    //    <[[(); 16]; 16] as layer::AvgPoolLayer<[u8; 3]>>::avg_pool(&int_examples_16);
    //let bit_examples_8 = patch_norm_conv!(int_examples_8, 8);
    //let foo = conv!(bit_examples_8, 8, 1, 1);

    //let int_examples_4 = <[[(); 8]; 8] as layer::AvgPoolLayer<[u8; 3]>>::avg_pool(&int_examples_8);
    //let bit_examples_4 = patch_norm_conv!(int_examples_4, 4);
    //let foo = conv!(bit_examples_4, 4, 1, 1);

    //let examples = conv!(bit_examples_32, 32, 1, 1);

    //let examples = <[[(); 32]; 32] as layer::BitPoolLayer<[b32; 1], [b32; 2]>>::bit_pool(&examples);
    //let examples = conv!(examples, 16, 2, 1);

    //let examples = <[[(); 16]; 16] as layer::ConcatLayer<[b32; 1], [b32; 1], [b32; 2]>>::concat(
    //    &examples,
    //    &bit_examples_16,
    //);
    //let examples = conv!(examples, 16, 2, 2);
    ////let examples = conv!(examples, 16, 2, 2);
    ////let examples = conv!(examples, 16, 2, 1);

    ////let bit_pooled_examples_8 = <[[(); 16]; 16] as layer::BitPoolLayer<[b32; 1], [b32; 2]>>::bit_pool(&examples);
}
