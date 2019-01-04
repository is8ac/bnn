extern crate bitnn;
extern crate rayon;
extern crate time;
use time::PreciseTime;

use bitnn::datasets::cifar;
//use bitnn::featuregen;
use bitnn::layers::unary;
use bitnn::layers::{
    ExtractPatchesNotched3x3, ExtractPixels, ExtractPixelsPadded, Layer, NewFromSplit,
    ObjectiveHead, OrPool2x2, PatchMap, PixelMap, PoolOrLayer, VecApply,
};

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
    unary::to_5(pixels[0]) as u16
        | ((unary::to_5(pixels[1]) as u16) << 5)
        | ((unary::to_6(pixels[2]) as u16) << 10)
}

fn main() {
    //let start = PreciseTime::now();
    let examples = load_data();
    let unary_examples: Vec<(usize, [[u16; 32]; 32])> = examples
        .iter()
        .map(|(class, image)| {
            (
                *class,
                image.patch_map(&|input, output: &mut u16| *output = rgb_to_u16(*input)),
            )
        })
        .collect();

    let mut model =
        Layer::<[[u16; 32]; 32], ExtractPatchesNotched3x3<u128>, u128, [u128; 10]>::new_from_split(
            &unary_examples,
        );
    let acc = model.optimize(&unary_examples, 3);
    println!("acc: {:?}%", acc * 100f64);

    let mut model = Layer::<
        [[u16; 32]; 32],
        OrPool2x2<u16>,
        [[u16; 16]; 16],
        Layer<[[u16; 16]; 16], ExtractPatchesNotched3x3<u128>, u128, [u128; 10]>,
    >::new_from_split(&unary_examples);
    let acc = model.optimize(&unary_examples, 3);
    println!("acc: {:?}%", acc * 100f64);

    let mut model = Layer::<
        [[u16; 32]; 32],
        OrPool2x2<u16>,
        [[u16; 16]; 16],
        Layer<[[u16; 16]; 16], ExtractPixels<u16>, u16, [u16; 10]>,
    >::new_from_split(&unary_examples);
    let acc = model.optimize(&unary_examples, 3);
    println!("acc: {:?}%", acc * 100f64);

    let mut model = Layer::<[[u16; 32]; 32], ExtractPixels<u16>, u16, [u16; 10]>::new_from_split(
        &unary_examples,
    );

    let acc = model.acc(&unary_examples);

    let pxlmap = PixelMap::<u16, [u16; 32], u32>::new_from_split(&unary_examples);
    let nls = pxlmap.vec_apply(&unary_examples);
    let mut model = Layer::<
        [[u16; 32]; 32],
        PixelMap<u16, [u16; 32], u32>,
        [[u32; 32]; 32],
        Layer<[[u32; 32]; 32], ExtractPixels<u32>, u32, [u32; 10]>,
    >::new_from_split(&unary_examples);

    let acc = model.optimize(&unary_examples, 3);
    //println!("acc: {:?}%", acc * 100f64);

    //let acc = model.optimize(&unary_examples, 3);
    //println!("acc: {:?}%", acc * 100f64);
    //let acc = model.optimize(&unary_examples, 3);
    //println!("acc: {:?}%", acc * 100f64);
}
// 32%

// 1/50 mod123 21.64% all: PT57.16S
// 1/50 mod30  20.98% PT217.460382764S
// 1/50 mod50  21.47% PT 60.99S
// 1/5  mod50  21.13% PT677.77S
// 1/5  mod50 20.96% PT319.77S

//acc: 20.949489795918367%
//all: PT36.504409270S
//acc: 21.725%
//all: PT20.222965598S

// foo
