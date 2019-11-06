extern crate bitnn;
extern crate num_cpus;
extern crate rayon;

use bitnn::bits::{b32, BitArray};
use bitnn::datasets::cifar;
use bitnn::image2d::{AvgPool, OrPool};
use bitnn::layer;
use bitnn::shape::{Element, ZipMap};
use rayon::prelude::*;
use std::path::Path;

const N_EXAMPLES: usize = 2000;
const N_CLASSES: usize = 10;
type InputType = [[b32; 3]; 3];

fn main() {
    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(28))
        //.num_threads(2)
        .build_global()
        .unwrap();

    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);

    let (_, bit_examples_32) = <[[([[(b32, b32); 3]; 3], u32); 32]; 1] as layer::Conv2D3x3<
        [[[u8; 3]; 32]; 32],
        [[b32; 3]; 3],
        [b32; 1],
        [(); N_CLASSES],
    >>::apply("foo1", &int_examples_32);

    let (_, bit_examples_32) = <[[([[[(b32, b32); 1]; 3]; 3], u32); 32]; 1] as layer::Conv2D3x3<
        [[[b32; 1]; 32]; 32],
        [[[b32; 1]; 3]; 3],
        [b32; 1],
        [(); N_CLASSES],
    >>::apply("foo2", &bit_examples_32);

    let int_examples_16: Vec<(_, usize)> = int_examples_32
        .par_iter()
        .map(|(image, class)| (image.avg_pool(), *class))
        .collect();
    let (_, bit_examples_16) = <[([[(b32, b32); 3]; 3], u32); 32] as layer::Conv2D3x3<
        [[[u8; 3]; 16]; 16],
        [[b32; 3]; 3],
        b32,
        [(); N_CLASSES],
    >>::apply("foo3", &int_examples_16);
    let bit_examples_16: Vec<(_, usize)> = bit_examples_32
        .par_iter()
        .zip(bit_examples_16.par_iter())
        .map(|(a, b)| {
            assert_eq!(a.1, b.1);
            (
                <[[(); 16]; 16] as ZipMap<[b32; 1], b32, [b32; 2]>>::zip_map(
                    &a.0.or_pool(),
                    &b.0,
                    |a, &b| [a[0], b],
                ),
                a.1,
            )
        })
        .collect();
    let (_, bit_examples_16) = <[[([[[(b32, b32); 2]; 3]; 3], u32); 32]; 2] as layer::Conv2D3x3<
        [[[b32; 2]; 16]; 16],
        [[[b32; 2]; 3]; 3],
        [b32; 2],
        [(); N_CLASSES],
    >>::apply("foo3", &bit_examples_16);

    let (_, bit_examples_16) = <[[([[[(b32, b32); 2]; 3]; 3], u32); 32]; 2] as layer::Conv2D3x3<
        [[[b32; 2]; 16]; 16],
        [[[b32; 2]; 3]; 3],
        [b32; 2],
        [(); N_CLASSES],
    >>::apply("foo3", &bit_examples_16);
    let (_, bit_examples_16) = <[[([[[(b32, b32); 2]; 3]; 3], u32); 32]; 2] as layer::Conv2D3x3<
        [[[b32; 2]; 16]; 16],
        [[[b32; 2]; 3]; 3],
        [b32; 2],
        [(); N_CLASSES],
    >>::apply("foo3", &bit_examples_16);

}
