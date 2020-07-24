#![feature(move_ref_pattern)]
#![feature(const_generics)]
use bitnn::bits::{b32, BitArray, BitArrayOPs, Distance, IncrementFracCounters, MaskedDistance, TritArray, TritPack};
use bitnn::cluster::{self, patch_count_lloyds, sparsify_centroid_count, CentroidCount, ImagePatchTritLloyds};
use bitnn::count::ElementwiseAdd;
use bitnn::datasets::cifar;
use bitnn::descend::{sum_loss_correct, TrainWeights, BTBVMM, IIIVMM};
use bitnn::image2d::{Conv2D, Image2D, PatchFold, PixelFold, PixelMap, StaticImage};
use bitnn::shape::{Element, Flatten, Map, Shape, ZipFold, ZipMap};
use bitnn::unary::{edges_from_patch, to_10, to_32};
use rand::distributions;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

fn unary(input: [u8; 3]) -> b32 {
    to_10(input[0]) | (to_10(input[1]) << 10) | (to_10(input[2]) << 20)
}

fn unary_3_chan(input: [u8; 3]) -> [b32; 3] {
    let mut target = [b32::default(); 3];
    for c in 0..3 {
        target[c] = to_32(input[c]);
    }
    target
}

const N_EXAMPLES: usize = 5_000;
type ImageType = StaticImage<[b32; 3], 32, 32>;
type PatchType = [[[b32; 3]; 3]; 3];
type HiddenType = [b32; 4];

fn main() {
    rayon::ThreadPoolBuilder::new().stack_size(2usize.pow(24)).build_global().unwrap();

    let mut rng = Hc128Rng::seed_from_u64(0);

    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let examples: Vec<_> = int_examples_32.par_iter().map(|(image, class)| (image.pixel_map(|&p| unary_3_chan(p)), *class)).collect();
    let centroids = <[[[b32; 3]; 3]; 3] as ImagePatchTritLloyds<ImageType, [[(); 3]; 3]>>::trit_lloyds(&mut rng, &examples, 1000, 3, 100);
    dbg!(centroids.len());
    let mut centroid_values: Vec<(u64, <PatchType as BitArray>::TritArrayType, u32)> = centroids
        .par_iter()
        .map(|centroid| {
            let acts: Vec<u32> = examples
                .iter()
                .map(|(image, _)| {
                    <ImageType as PatchFold<Vec<u32>, [[(); 3]; 3]>>::patch_fold(image, Vec::new(), |mut acts, patch| {
                        acts.push(centroid.masked_distance(patch));
                        acts
                    })
                })
                .flatten()
                .collect();
            let threshold = acts[acts.len() / 2];
            let class_values: [i64; 10] = examples.iter().fold([0i64; 10], |mut acc, (image, class)| {
                let n_acts = <ImageType as PatchFold<_, [[(); 3]; 3]>>::patch_fold(image, 0u32, |count, patch| count + (centroid.masked_distance(patch) > threshold) as u32);
                acc[*class] += n_acts as i64 - (<<ImageType as Image2D>::ImageShape as Shape>::N / 2) as i64;
                acc
            });
            let sum_value: u64 = class_values.iter().map(|x| x.pow(2) as u64).sum();
            (sum_value, *centroid, threshold)
        })
        .collect();
    dbg!();
    centroid_values.sort_by_key(|(v, _, _)| *v);
    centroid_values.reverse();
    for i in 0..1 {
        dbg!(centroid_values[i]);
    }
    let weights_vec: Vec<_> = centroid_values
        .iter()
        .take(<<HiddenType as BitArray>::BitShape as Shape>::N)
        .map(|(_, centroid, threshold)| ((*centroid, *threshold)))
        .collect();
    let weights = <<HiddenType as BitArray>::BitShape as Flatten<(<PatchType as BitArray>::TritArrayType, u32)>>::from_vec(&weights_vec);
    dbg!();
    let sum_class_acts: [(usize, <u64 as Element<<HiddenType as BitArray>::BitShape>>::Array); 10] = examples
        .par_iter()
        .fold(
            || <[(usize, <u64 as Element<<HiddenType as BitArray>::BitShape>>::Array); 10]>::default(),
            |mut acc, (image, class)| {
                acc[*class] = <ImageType as PatchFold<_, [[(); 3]; 3]>>::patch_fold(image, acc[*class], |(n, acc), patch| {
                    (
                        n + 1,
                        <<HiddenType as BitArray>::BitShape as ZipMap<u64, (<PatchType as BitArray>::TritArrayType, u32), u64>>::zip_map(&acc, &weights, |acc, (trits, threshold)| {
                            acc + (trits.masked_distance(patch) > *threshold) as u64
                        }),
                    )
                });
                acc
            },
        )
        .reduce_with(|mut a, b| {
            a.elementwise_add(&b);
            a
        })
        .unwrap();
    let sums = sum_class_acts
        .iter()
        .fold(<(usize, <u64 as Element<<HiddenType as BitArray>::BitShape>>::Array)>::default(), |mut a, b| {
            a.elementwise_add(b);
            a
        });

    let aux_weights =
        <[(); 10] as Map<(usize, <u64 as Element<<HiddenType as BitArray>::BitShape>>::Array), <u64 as Element<<HiddenType as BitArray>::BitShape>>::Array>>::map(&sum_class_acts, |(n, counts)| {
            <<HiddenType as BitArray>::BitShape as ZipMap<u64, u64, u64>>::zip_map(counts, &sums.1, |&class, &sum_class| (class * *n as u64) / sum_class)
        });
    //dbg!(aux_weights);
    let n_correct: u64 = examples
        .par_iter()
        .map(|(image, class)| {
            let acts = <ImageType as PatchFold<_, [[(); 3]; 3]>>::patch_fold(image, <u64 as Element<<HiddenType as BitArray>::BitShape>>::Array::default(), |acc, patch| {
                <<HiddenType as BitArray>::BitShape as ZipMap<u64, (<PatchType as BitArray>::TritArrayType, u32), u64>>::zip_map(&acc, &weights, |acc, (trits, threshold)| {
                    acc + (trits.masked_distance(patch) > *threshold) as u64
                })
            });
            let max_act: usize = aux_weights
                .iter()
                .map(|class_weights| <<HiddenType as BitArray>::BitShape as ZipFold<u64, u64, u64>>::zip_fold(&acts, class_weights, 0, |s, a, b| s + a * b))
                .enumerate()
                .max_by_key(|(_, act)| *act)
                .unwrap()
                .0;
            (max_act == *class) as u64
        })
        .sum();
    dbg!(n_correct as f64 / examples.len() as f64);
}
// pab = (pba*pa)/pb
