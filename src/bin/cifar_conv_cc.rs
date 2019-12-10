// if main overflows stack:
// ulimit -S -s 4194304

extern crate bitnn;
extern crate num_cpus;
extern crate rayon;

use bitnn::bits::{b32, BitArray};
use bitnn::datasets::cifar;
use bitnn::image2d::StaticImage;
use bitnn::layer::{AvgPoolLayer, BitPoolLayer, ClassifyLayer, ConcatImages, CountBits, Layer};
use bitnn::shape::Element;
use bitnn::unary::NormalizeAndBitpack;
use bitnn::weight::{SimpleClassify, SupervisedWeightsGen, UnsupervisedClusterWeightsGen};
use std::path::Path;

const N_EXAMPLES: usize = 2000;
const N_CLASSES: usize = 10;

macro_rules! cluster_conv {
    ($examples:expr, $dim:expr, $in_type:ty, $out_size:expr) => {{
        let new_examples = <[[(
            [[<(
                <$in_type as BitArray>::WordType,
                <$in_type as BitArray>::WordType,
            ) as Element<<$in_type as BitArray>::WordShape>>::Array; 3]; 3],
            u32,
        ); 32]; $out_size] as Layer<
            StaticImage<[[$in_type; $dim]; $dim]>,
            [[$in_type; 3]; 3],
            [[$in_type; 3]; 3],
            UnsupervisedClusterWeightsGen<[[[b32; $out_size]; 3]; 3], [b32; $out_size], N_CLASSES>,
            [b32; $out_size],
            [(); N_CLASSES],
        >>::gen(&$examples);
        new_examples
    }};
}


macro_rules! conv {
    ($examples:expr, $dim:expr, $in_type:ty, $out_size:expr) => {{
        let new_examples = <[[(
            [[<(
                <$in_type as BitArray>::WordType,
                <$in_type as BitArray>::WordType,
            ) as Element<<$in_type as BitArray>::WordShape>>::Array; 3]; 3],
            u32,
        ); 32]; $out_size] as Layer<
            StaticImage<[[$in_type; $dim]; $dim]>,
            [[$in_type; 3]; 3],
            [[$in_type; 3]; 3],
            SupervisedWeightsGen<[[[b32; $out_size]; 3]; 3], [b32; $out_size], N_CLASSES>,
            [b32; $out_size],
            [(); N_CLASSES],
        >>::gen(&$examples);
        new_examples
    }};
}

macro_rules! patch_norm_conv {
    ($examples:expr, $dim:expr) => {{
        let bit_examples = <[[([[[(b32, b32); 3]; 3]; 3], u32); 32]; 1] as Layer<
            StaticImage<[[[u8; 3]; $dim]; $dim]>,
            [[[u8; 3]; 3]; 3],
            [[[b32; 3]; 3]; 3],
            SupervisedWeightsGen<[[[b32; 3]; 3]; 3], [b32; 1], N_CLASSES>,
            [b32; 1],
            [(); N_CLASSES],
        >>::gen(&$examples);
        bit_examples
    }};
}

macro_rules! cluster_patch_norm_conv {
    ($examples:expr, $dim:expr) => {{
        let bit_examples = <[[([[[(b32, b32); 3]; 3]; 3], u32); 32]; 1] as Layer<
            StaticImage<[[[u8; 3]; $dim]; $dim]>,
            [[[u8; 3]; 3]; 3],
            [[[b32; 3]; 3]; 3],
            UnsupervisedClusterWeightsGen<[[[b32; 3]; 3]; 3], [b32; 1], N_CLASSES>,
            [b32; 1],
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
    let examples = cluster_patch_norm_conv!(int_examples_32, 32);
    let examples = cluster_conv!(examples, 32, [b32; 1], 1);
    let examples = cluster_conv!(examples, 32, [b32; 1], 1);
    let examples = cluster_conv!(examples, 32, [b32; 1], 1);
    let examples = conv!(examples, 32, [b32; 1], 1);
    //println!("{}", examples[0].0);
    // 19.2


    let acc = <[([[[(b32, b32); 1]; 3]; 3], u32); 10] as ClassifyLayer<
        StaticImage<[[[b32; 1]; 32]; 32]>,
        [[[b32; 1]; 3]; 3],
        SimpleClassify<[[[b32; 1]; 3]; 3], { 10 }>,
        [(); 10],
    >>::gen_classify(&examples);
    println!("{}%", acc * 100f64);

    //let int_examples_16 = <()>::avg_pool(&int_examples_32);
    ////let bit_examples_16 = patch_norm_conv!(int_examples_16, 16);
    ////let examples = <()>::bit_pool(&examples);
    ////let examples = conv!(examples, 16, [[b32; 1]; 2], 2);
    ////let examples = <()>::concat(&examples, &bit_examples_16);
    ////let examples = conv!(examples, 16, [b32; 3], 2);
    ////let examples = conv!(examples, 16, [b32; 2], 2);
    //////let examples = conv!(examples, 16, [b32; 2], 2);

    //let int_examples_8 = <()>::avg_pool(&int_examples_16);
    //let bit_examples_8 = patch_norm_conv!(int_examples_8, 8);
    ////let examples = <()>::bit_pool(&examples);
    ////let examples = conv!(examples, 8, [[b32; 2]; 2], 2);
    ////let examples = <()>::concat(&examples, &bit_examples_8);
    //let examples = conv!(bit_examples_8, 8, [b32; 1], 2);
    //let examples = conv!(examples, 8, [b32; 2], 2);
    //let examples = conv!(examples, 8, [b32; 2], 2);
    ////let examples = conv!(examples, 8, [b32; 3], 3);
    ////let examples = conv!(examples, 8, [b32; 3], 3);

    ////let int_examples_4 = <()>::avg_pool(&int_examples_8);
    ////let bit_examples_4 = patch_norm_conv!(int_examples_4, 4);
    ////let examples = <()>::bit_pool(&examples);
    ////let examples = conv!(examples, 4, [[b32; 3]; 2], 2);
    ////let examples = <()>::concat(&examples, &bit_examples_4);
    ////let examples = conv!(examples, 4, [b32; 3], 3);

    //let (_, acc) = <[([[[(b32, b32); 2]; 3]; 3], u32); 10] as ClassifyLayer<
    //    StaticImage<[[[b32; 2]; 8]; 8]>,
    //    [[[b32; 2]; 3]; 3],
    //    [(); 10],
    //>>::gen_classify(&examples);
    //println!("{}%", acc * 100f64);
    // 31.1%
    // 23%
}
