#![feature(move_ref_pattern)]
#![feature(const_generics)]
use bitnn::bits::{b32, BitArray, IncrementFracCounters};
use bitnn::cluster::{
    self, patch_count_lloyds, sparsify_centroid_count, CentroidCount, ImagePatchLloyds,
};
use bitnn::datasets::cifar;
use bitnn::descend::{sum_loss_correct, TrainWeights, BTBVMM, IIIVMM};
use bitnn::image2d::{Conv2D, Image2D, PixelFold, PixelMap, StaticImage};
use bitnn::shape::{Element, Map, Shape};
use bitnn::unary::{edges_from_patch, to_10, to_32};
use rand::distributions;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Copy, Clone)]
pub struct ClusterParams {
    /// seed for patch level clustering. Example: 0
    patch_lloyds_seed: u64,
    /// initial number of patch centroids. Actual number after pruning will generally be smaller. Increase for accurecy, decrese for performance. Example: 500
    patch_lloyds_k: usize,
    /// number of clustering iterations. Increase for accuracy, decrease for performance. Example: 3
    patch_lloyds_i: usize,
    /// Patch level clustering prune threshold. Increase for performance, decrese for accurecy. Set to 0 to effectively disable pruning. Example: 1
    patch_lloyds_prune_threshold: usize,
    /// image / patch bag level clustering seed. Example: 0
    image_lloyds_seed: u64,
    /// initial number of image centroids. Actual number after pruning will generally be smaller. Increase for accurecy, decrese for performance. Example: 500
    image_lloyds_k: usize,
    /// Number of image cluster iters. Example: 3
    image_lloyds_i: usize,
    /// image cluster prune threshold. Increase for performance, decrese for accuracy. Set to 0 to effectively disable. Example: 1
    image_lloyds_prune_threshold: usize,
    /// Prune patches within patch bags. Increase for performance, decrease for accuracy. Example: 1
    sparsify_centroid_count_filter_threshold: u32,
}

// This need only called once per layer.
// return is (patch_centroids, (image_patch_bags, class_counts))
// where image_patch_bags is a Vec<(sparse_patch_bag, center)>
fn cluster_patches_and_images<
    P: CentroidCount<ImageType, [[(); 3]; 3], C>
        + BitArray
        + Sync
        + Send
        + ImagePatchLloyds<ImageType, [[(); 3]; 3]>,
    ImageType: Image2D + Sync + Send,
    const C: usize,
>(
    examples: &Vec<(ImageType, usize)>,
    params: &ClusterParams,
) -> (Vec<P>, (Vec<(Vec<(usize, u32)>, i32)>, Vec<[u32; C]>))
where
    distributions::Standard: distributions::Distribution<P>,
    [u32; C]: Default + Copy + Sync + Send,
{
    let mut rng = Hc128Rng::seed_from_u64(params.patch_lloyds_seed);
    let patch_centroids = <P as ImagePatchLloyds<_, [[(); 3]; 3]>>::lloyds(
        &mut rng,
        &examples,
        params.patch_lloyds_k,
        params.patch_lloyds_i,
        params.patch_lloyds_prune_threshold,
    );
    let patch_dists: Vec<(Vec<u32>, usize)> = examples
        .par_iter()
        .map(|(image, class)| {
            (
                <P as CentroidCount<ImageType, [[(); 3]; 3], C>>::centroid_count(
                    image,
                    &patch_centroids,
                ),
                *class,
            )
        })
        .collect();
    dbg!();
    let patch_dist_centroids: Vec<Vec<u32>> = patch_count_lloyds(
        &patch_dists,
        params.image_lloyds_seed,
        params.image_lloyds_i,
        patch_centroids.len(),
        params.image_lloyds_k,
        params.image_lloyds_prune_threshold,
    );
    dbg!();
    let patch_bag_cluster_class_dists: Vec<[u32; C]> =
        cluster::class_dist::<C>(&patch_dists, &patch_dist_centroids);
    //{
    //    // sanity checks
    //    assert_eq!(patch_bag_cluster_class_dists.len(), patch_dist_centroids.len());
    //    let sum: u32 = patch_bag_cluster_class_dists.iter().flatten().sum();
    //    assert_eq!(sum as usize, N_EXAMPLES);
    //}

    // the i32 is 1/2 the sum of the counts
    let sparse_patch_bags: Vec<(Vec<(usize, u32)>, i32)> = patch_dist_centroids
        .par_iter()
        .map(|patch_counts| {
            let bag = sparsify_centroid_count(
                patch_counts,
                params.sparsify_centroid_count_filter_threshold,
            );
            let n: u32 = bag.iter().map(|(_, c)| *c).sum();
            (bag, n as i32 / 2)
        })
        .collect();
    (
        patch_centroids,
        (sparse_patch_bags, patch_bag_cluster_class_dists),
    )
}

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

trait TrainLayer<ImageShape, PatchShape, Pixel, H, const C: usize>
where
    ImageShape: Shape,
    H: Element<ImageShape>,
    Pixel: Element<ImageShape>,
{
    fn train_layer(
        examples: &Vec<(<Pixel as Element<ImageShape>>::Array, usize)>,
        cluster_params: &ClusterParams,
    ) -> Vec<(<H as Element<ImageShape>>::Array, usize)>;
}

impl<ImageShape, PatchShape, Pixel, H, const C: usize>
    TrainLayer<ImageShape, PatchShape, Pixel, H, C> for ()
where
    ImageShape: Shape,
    PatchShape: Shape,
    H: Element<ImageShape> + BitArray + Sync + IncrementFracCounters,
    Pixel: Element<ImageShape> + Element<PatchShape>,
    u32: Element<H::BitShape>,
    i8: Element<H::BitShape>,
    i32: Element<H::BitShape>,
    (): TrainWeights<<Pixel as Element<PatchShape>>::Array, H, C>,
    distributions::Standard: distributions::Distribution<<Pixel as Element<PatchShape>>::Array>,
    H::BitShape: Map<u32, i32>,
    [u32; C]: Default,
    [(<i8 as Element<H::BitShape>>::Array, i8); C]: IIIVMM<H, C>,
    <u32 as Element<H::BitShape>>::Array: Default,
    <i8 as Element<<H as BitArray>::BitShape>>::Array: Sync,
    <H as Element<ImageShape>>::Array: Image2D<PixelType = H>
        + PixelFold<(usize, <u32 as Element<H::BitShape>>::Array), PatchShape>
        + Send,
    <Pixel as Element<ImageShape>>::Array: Image2D<PixelType = Pixel>
        + Sync
        + Conv2D<PatchShape, H, OutputType = <H as Element<ImageShape>>::Array>,
    <Pixel as Element<PatchShape>>::Array: BitArray
        + ImagePatchLloyds<<Pixel as Element<ImageShape>>::Array, PatchShape>
        + BitArray
        + Sync
        + CentroidCount<<Pixel as Element<ImageShape>>::Array, PatchShape, C>,
    (
        <<Pixel as Element<PatchShape>>::Array as BitArray>::TritArrayType,
        u32,
    ): Element<H::BitShape>,
    <(
        <<Pixel as Element<PatchShape>>::Array as BitArray>::TritArrayType,
        u32,
    ) as Element<H::BitShape>>::Array: BTBVMM<<Pixel as Element<PatchShape>>::Array, H> + Sync,
{
    fn train_layer(
        examples: &Vec<(<Pixel as Element<ImageShape>>::Array, usize)>,
        cluster_params: &ClusterParams,
    ) -> Vec<(<H as Element<ImageShape>>::Array, usize)> {
        let (patch_centroids, (patch_bags, class_counts)): (
            Vec<<Pixel as Element<PatchShape>>::Array>,
            (Vec<(Vec<(usize, u32)>, i32)>, Vec<[u32; C]>),
        ) = {
            let mut rng = Hc128Rng::seed_from_u64(cluster_params.patch_lloyds_seed);
            let patch_centroids =
                <<Pixel as Element<PatchShape>>::Array as ImagePatchLloyds<_, PatchShape>>::lloyds(
                    &mut rng,
                    &examples,
                    cluster_params.patch_lloyds_k,
                    cluster_params.patch_lloyds_i,
                    cluster_params.patch_lloyds_prune_threshold,
                );
            let patch_dists: Vec<(Vec<u32>, usize)> = examples
                .par_iter()
                .map(|(image, class)| {
                    (
                        <<Pixel as Element<PatchShape>>::Array as CentroidCount<
                            <Pixel as Element<ImageShape>>::Array,
                            PatchShape,
                            C,
                        >>::centroid_count(image, &patch_centroids),
                        *class,
                    )
                })
                .collect();
            dbg!();
            let patch_dist_centroids: Vec<Vec<u32>> = patch_count_lloyds(
                &patch_dists,
                cluster_params.image_lloyds_seed,
                cluster_params.image_lloyds_i,
                patch_centroids.len(),
                cluster_params.image_lloyds_k,
                cluster_params.image_lloyds_prune_threshold,
            );
            dbg!();
            let patch_bag_cluster_class_dists: Vec<[u32; C]> =
                cluster::class_dist::<C>(&patch_dists, &patch_dist_centroids);
            //{
            //    // sanity checks
            //    assert_eq!(patch_bag_cluster_class_dists.len(), patch_dist_centroids.len());
            //    let sum: u32 = patch_bag_cluster_class_dists.iter().flatten().sum();
            //    assert_eq!(sum as usize, N_EXAMPLES);
            //}

            // the i32 is 1/2 the sum of the counts
            let sparse_patch_bags: Vec<(Vec<(usize, u32)>, i32)> = patch_dist_centroids
                .par_iter()
                .map(|patch_counts| {
                    let bag = sparsify_centroid_count(
                        patch_counts,
                        cluster_params.sparsify_centroid_count_filter_threshold,
                    );
                    let n: u32 = bag.iter().map(|(_, c)| *c).sum();
                    (bag, n as i32 / 2)
                })
                .collect();
            (
                patch_centroids,
                (sparse_patch_bags, patch_bag_cluster_class_dists),
            )
        };

        dbg!(patch_centroids.len());
        dbg!(patch_bags.len());

        let (patch_weights, aux_weights, sum_loss) =
            <() as TrainWeights<<Pixel as Element<PatchShape>>::Array, H, C>>::train_weights(
                &patch_centroids,
                &patch_bags,
                &class_counts,
                0,
                7,
                5,
            );

        {
            let patch_acts: Vec<H> = patch_centroids
                .iter()
                .map(|patch| patch_weights.btbvmm(patch))
                .collect();
            let (true_sum_loss, n_correct, n) =
                sum_loss_correct::<H, C>(&patch_acts, &patch_bags, &class_counts, &aux_weights);
            dbg!(n_correct as f64 / n as f64);
            assert_eq!(sum_loss, true_sum_loss);
        }
        let new_examples: Vec<(_, usize)> = examples
            .par_iter()
            .map(|(image, class)| {
                (
                    Conv2D::<PatchShape, H>::conv2d(image, |patch| patch_weights.btbvmm(patch)),
                    *class,
                )
            })
            .collect();
        let n_correct: usize = new_examples
            .iter()
            .filter(|(image, class)| {
                let (n_pixels, hidden_act_counts) =
                    <<H as Element<ImageShape>>::Array as PixelFold<
                        (usize, <u32 as Element<<H as BitArray>::BitShape>>::Array),
                        PatchShape,
                    >>::pixel_fold(
                        image,
                        <(usize, <u32 as Element<<H as BitArray>::BitShape>>::Array)>::default(),
                        |mut acc, hidden_acts| {
                            hidden_acts.increment_frac_counters(&mut acc);
                            acc
                        },
                    );
                let n = (n_pixels / 2) as i32;
                let sum_acts = <<H as BitArray>::BitShape as Map<u32, i32>>::map(
                    &hidden_act_counts,
                    |count| *count as i32 - n,
                );
                let class_acts: [i32; C] = IIIVMM::<H, C>::iiivmm(&aux_weights, &sum_acts);
                let (_, max_act) = class_acts
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != *class)
                    .max_by_key(|(_, v)| *v)
                    .unwrap();
                class_acts[*class] > *max_act
            })
            .count();
        dbg!(n_correct as f64 / examples.len() as f64);
        new_examples
    }
}

//type PixelType = b32;
type PixelType = [b32; 3];
type PatchType = [[PixelType; 3]; 3];
type HiddenType = [b32; 4];

const N_EXAMPLES: usize = 2_000;

fn main() {
    dbg!(std::any::type_name::<PatchType>());
    dbg!(std::any::type_name::<HiddenType>());
    //rayon::ThreadPoolBuilder::new().stack_size(2usize.pow(22)).num_threads(16).build_global().unwrap();

    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let examples: Vec<_> = int_examples_32
        .par_iter()
        .map(|(image, class)| (image.pixel_map(|&p| unary_3_chan(p)), *class))
        .collect();

    //let examples: Vec<(_, usize)> = int_examples_32
    //    .par_iter()
    //    .map(|(image, class)| (Conv2D::<[[(); 3]; 3], b32>::conv2d(image, |patch| edges_from_patch(patch)), *class))
    //    .collect();

    //println!("{}", examples[6].0);

    dbg!(examples.len());
    let cluster_params = ClusterParams {
        patch_lloyds_seed: 0,
        patch_lloyds_k: 3_000,
        patch_lloyds_i: 5,
        patch_lloyds_prune_threshold: 30,
        image_lloyds_seed: 0,
        image_lloyds_i: 1,
        image_lloyds_k: 2_000,
        image_lloyds_prune_threshold: 0,
        sparsify_centroid_count_filter_threshold: 0,
    };
    dbg!(&cluster_params);
    let examples = <() as TrainLayer<
        StaticImage<(), 32, 32>,
        [[(); 3]; 3],
        PixelType,
        HiddenType,
        10,
    >>::train_layer(&examples, &cluster_params);
    let examples = <() as TrainLayer<
        StaticImage<(), 32, 32>,
        [[(); 3]; 3],
        HiddenType,
        HiddenType,
        10,
    >>::train_layer(&examples, &cluster_params);
    let examples = <() as TrainLayer<
        StaticImage<(), 32, 32>,
        [[(); 3]; 3],
        HiddenType,
        HiddenType,
        10,
    >>::train_layer(&examples, &cluster_params);
    //let examples = <() as TrainLayer<StaticImage<(), 32, 32>, [[(); 3]; 3], HiddenType, HiddenType, 10>>::train_layer(&examples, &cluster_params);
    //println!("{}", examples[6].0);
}
