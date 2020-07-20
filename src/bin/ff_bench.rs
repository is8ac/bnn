#![feature(move_ref_pattern)]
#![feature(const_generics)]
use bitnn::bits::{b32, BitArray};
use bitnn::cluster::{
    self, patch_count_lloyds, sparsify_centroid_count, CentroidCount, ImagePatchLloyds,
};
use bitnn::datasets::cifar;
use bitnn::descend::{sum_loss_correct, TrainWeights, BTBVMM};
use bitnn::image2d::{Image2D, PixelMap, StaticImage};
use bitnn::unary::to_10;
use rand::distributions;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;
use std::time::Instant;

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
    P: CentroidCount<ImageType, [[(); 3]; 3], [(); C]>
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
                <P as CentroidCount<ImageType, [[(); 3]; 3], [(); C]>>::centroid_count(
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

type PatchType = [[b32; 3]; 3];
type HiddenType = [b32; 4];

const N_EXAMPLES: usize = 1_000;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(2usize.pow(22))
        //.num_threads(16)
        .build_global()
        .unwrap();

    let mut rng = Hc128Rng::seed_from_u64(0);

    let cifar_base_path = Path::new("/big/cache/datasets/cifar-10-batches-bin");
    let int_examples_32 = cifar::load_images_from_base(&cifar_base_path, N_EXAMPLES);
    let examples: Vec<_> = int_examples_32
        .par_iter()
        .map(|(image, class)| (image.pixel_map(|&p| unary(p)), *class))
        .collect();

    let cluster_params = ClusterParams {
        patch_lloyds_seed: 0,
        patch_lloyds_k: 500,
        patch_lloyds_i: 2,
        patch_lloyds_prune_threshold: 1,
        image_lloyds_seed: 0,
        image_lloyds_i: 2,
        image_lloyds_k: 500,
        image_lloyds_prune_threshold: 1,
        sparsify_centroid_count_filter_threshold: 1,
    };
    dbg!();
    let cluster_start = Instant::now();
    let (patch_centroids, (patch_bags, class_counts)): (
        Vec<PatchType>,
        (Vec<(Vec<(usize, u32)>, i32)>, Vec<[u32; 10]>),
    ) = cluster_patches_and_images::<PatchType, StaticImage<b32, 32usize, 32usize>, 10>(
        &examples,
        &cluster_params,
    );
    dbg!(patch_centroids.len());
    dbg!(patch_bags.len());
    println!("cluster time: {:?}", cluster_start.elapsed());

    let (patch_weights, aux_weights, sum_loss) =
        <() as TrainWeights<PatchType, HiddenType, 10>>::train_weights(
            &patch_centroids,
            &patch_bags,
            &class_counts,
            0,
            5,
            3,
        );

    {
        let patch_acts: Vec<HiddenType> = patch_centroids
            .iter()
            .map(|patch| patch_weights.btbvmm(patch))
            .collect();
        let (true_sum_loss, n_correct, n) = sum_loss_correct::<HiddenType, 10>(
            &patch_acts,
            &patch_bags,
            &class_counts,
            &aux_weights,
        );
        assert_eq!(sum_loss, true_sum_loss);
        dbg!(n_correct as f64 / n as f64);
    }
}
