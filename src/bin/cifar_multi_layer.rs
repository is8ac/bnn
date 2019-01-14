extern crate bitnn;
extern crate rayon;
extern crate time;
use time::PreciseTime;

use bitnn::datasets::cifar;
//use bitnn::featuregen;
use bitnn::layers::unary;
use bitnn::layers::{
    Accuracy, Layer, NewFromSplit, OptimizeHead, OrPool2x2, Patch3x3NotchedConv, PatchMap,
    PoolOrLayer, SaveLoad, VecApply,
};
use std::marker::PhantomData;
use std::path::Path;

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let size: usize = 10000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
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

type Layer1Head = Layer<
    [[u16; 32]; 32],
    Patch3x3NotchedConv<u16, u128, [u128; 16], u16>,
    [[u16; 32]; 32],
    Layer<
        [[u16; 32]; 32],
        OrPool2x2<u16>,
        [[u16; 16]; 16],
        Patch3x3NotchedConv<u16, u128, [u128; 10], f64>,
    >,
>;

//let mut model = Layer::<
//    [[u16; 8]; 8],
//    Patch3x3NotchedConv<u16, u128, [u128; 32], u32>,
//    [[u32; 8]; 8],
//    Layer<
//        [[u32; 8]; 8],
//        OrPool2x2<u32>,
//        [[u32; 4]; 4],
//        Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 10], f64>,
//    >,
//>::new_from_split(&examples);

const MOD: usize = 20;
// 281
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

    let conv1 = Patch3x3NotchedConv::<u16, u128, [u128; 16], u16>::new_from_fs(&Path::new(
        "params/l1.prms",
    ));
    //let examples = conv1.vec_apply(&unary_examples);
    let pool1 = OrPool2x2::default();
    //let examples = pool1.vec_apply(&examples);
    let conv2 = Patch3x3NotchedConv::<u16, u128, [u128; 16], u16>::new_from_fs(&Path::new(
        "params/l2.prms",
    ));
    //let examples = conv2.vec_apply(&examples);
    let pool2 = OrPool2x2::default();
    //let examples = pool2.vec_apply(&examples);
    let conv3 = Patch3x3NotchedConv::<u16, u128, [u128; 32], u32>::new_from_fs(&Path::new(
        "params/l3.prms",
    ));
    //let examples = conv3.vec_apply(&examples);

    let final_head = Layer::<
        [[u32; 8]; 8],
        Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>,
        [[u32; 8]; 8],
        Layer<[[u32; 8]; 8], OrPool2x2<u32>, [[u32; 4]; 4], [[[u32; 4]; 4]; 10]>,
    >::new_from_fs(&Path::new("params/model4.prms"));

    let head = Layer::new_from_parts(conv3, final_head);
    let head = Layer::new_from_parts(pool2, head);
    let head = Layer::new_from_parts(conv2, head);
    let head = Layer::new_from_parts(pool1, head);
    let mut model: Layer<
        [[u16; 32]; 32],
        Patch3x3NotchedConv<u16, u128, [u128; 16], u16>,
        [[u16; 32]; 32],
        Layer<
            [[u16; 32]; 32],
            OrPool2x2<u16>,
            [[u16; 16]; 16],
            Layer<
                [[u16; 16]; 16],
                Patch3x3NotchedConv<u16, u128, [u128; 16], u16>,
                [[u16; 16]; 16],
                Layer<
                    [[u16; 16]; 16],
                    OrPool2x2<_>,
                    [[u16; 8]; 8],
                    Layer<
                        [[u16; 8]; 8],
                        Patch3x3NotchedConv<u16, u128, [u128; 32], u32>,
                        [[u32; 8]; 8],
                        Layer<
                            [[u32; 8]; 8],
                            Patch3x3NotchedConv<u32, [u128; 2], [[u128; 2]; 32], u32>,
                            [[u32; 8]; 8],
                            Layer<
                                [[u32; 8]; 8],
                                OrPool2x2<u32>,
                                [[u32; 4]; 4],
                                [[[u32; 4]; 4]; 10],
                            >,
                        >,
                    >,
                >,
            >,
        >,
    > = Layer::new_from_parts(conv1, head);

    println!("starting optimize");
    let start = PreciseTime::now();
    for i in 0..3 {
        let obj = model.optimize_head(&unary_examples, MOD);
        println!(
            "mod: {}, obj: {:?}%, time: {}",
            MOD,
            obj * 100f64,
            start.to(PreciseTime::now())
        );
        println!("writing parameters to disk");
        //model.data.write_to_fs(&Path::new("params/l4.prms"));
        model.write_to_fs(&Path::new("params/full_model_e2e.prms"));
    }
}

// 34.7% train acc
