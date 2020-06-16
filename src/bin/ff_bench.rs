#![feature(const_generics)]
use bitnn::bits::{b32, BitArray, BitArrayOPs, BitMapPack, BitWord, IncrementFracCounters, MaskedDistance, TritArray};
use bitnn::cluster::{self, patch_count_lloyds, sparsify_centroid_count, CentroidCount, ImagePatchLloyds};
use bitnn::datasets::cifar;
//use bitnn::descend::Descend;
use bitnn::image2d::{PixelMap, StaticImage};
use bitnn::shape::{Element, IndexGet, Map, MapMut, Shape, ZipFold, ZipMap};
use bitnn::unary::to_10;
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::path::Path;
use std::time::{Duration, Instant};

trait IIIVMM<I: BitArray, C: Shape>
where
    i32: Element<C> + Element<I::BitShape>,
    (usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])): Element<C>,
    [u32; 2]: Element<I::BitShape>,
    Option<bool>: Element<I::BitShape>,
    u32: Element<C>,
{
    fn iiivmm(&self, input: &<i32 as Element<I::BitShape>>::Array) -> <i32 as Element<C>>::Array;
    fn update_iiivmm_acts(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        acts: &<i32 as Element<C>>::Array,
        index: &<I::BitShape as Shape>::Index,
    ) -> <i32 as Element<C>>::Array;
    fn subtract_input_from_iiivmm_acts(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        acts: &<i32 as Element<C>>::Array,
        index: &<I::BitShape as Shape>::Index,
    ) -> <i32 as Element<C>>::Array;
    fn increment_grads(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        class_dist: &<u32 as Element<C>>::Array,
        grads: &mut <(usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])) as Element<C>>::Array,
        input_threshold: i32,
    ) -> u64;
    fn update(&self, grads: &<(usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])) as Element<C>>::Array, update_threshold: u32) -> Self;
    fn descend(
        &self,
        patch_acts: &Vec<I>,
        sparse_patch_count_class_dists: &Vec<(Vec<(usize, u32)>, <u32 as Element<C>>::Array)>,
        i: usize,
        it: i32,
        ut: u32,
    ) -> Self;
}

impl<I, const C: usize> IIIVMM<I, [(); C]> for [(<i8 as Element<I::BitShape>>::Array, i8); C]
where
    I: BitArray + BitArrayOPs + IncrementFracCounters,
    I::BitShape: MapMut<i32, [u32; 2]> + ZipFold<i32, i32, i8> + ZipMap<i8, [u32; 2], i8>,
    [i32; C]: Default,
    [(); C]: ZipMap<
        (<i8 as Element<I::BitShape>>::Array, i8),
        (usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])),
        (<i8 as Element<I::BitShape>>::Array, i8),
    >,
    i8: Element<I::BitShape>,
    i32: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    bool: Element<I::BitShape>,
    Option<bool>: Element<I::BitShape>,
    [u32; 2]: Element<I::BitShape>,
    [(usize, (<[u32; 2] as Element<<I as BitArray>::BitShape>>::Array, [u32; 2])); C]: Default,
    [(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8); C]: Default,
    <I as BitArray>::BitShape: Map<u32, i32>,
    <i8 as Element<I::BitShape>>::Array: IndexGet<<I::BitShape as Shape>::Index, Element = i8>,
    <i32 as Element<I::BitShape>>::Array: IndexGet<<I::BitShape as Shape>::Index, Element = i32>,
    <u32 as Element<<I as BitArray>::BitShape>>::Array: Default,
{
    fn iiivmm(&self, input: &<i32 as Element<I::BitShape>>::Array) -> [i32; C] {
        <[(); C] as Map<(<i8 as Element<I::BitShape>>::Array, i8), i32>>::map(self, |(weights, bias)| {
            <I::BitShape as ZipFold<i32, i32, i8>>::zip_fold(&input, weights, 0, |sum, i, &w| sum + i * w as i32) + *bias as i32
        })
    }
    fn update_iiivmm_acts(&self, input: &<i32 as Element<I::BitShape>>::Array, partial_acts: &[i32; C], index: &<I::BitShape as Shape>::Index) -> [i32; C] {
        <[(); C] as ZipMap<i32, (<i8 as Element<I::BitShape>>::Array, i8), i32>>::zip_map(partial_acts, self, |partial_act, (weights, _)| {
            partial_act + *weights.index_get(index) as i32 * input.index_get(index)
        })
    }
    fn subtract_input_from_iiivmm_acts(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        partial_acts: &[i32; C],
        index: &<I::BitShape as Shape>::Index,
    ) -> [i32; C] {
        <[(); C] as ZipMap<i32, (<i8 as Element<I::BitShape>>::Array, i8), i32>>::zip_map(partial_acts, self, |partial_act, (weights, _)| {
            partial_act - *weights.index_get(index) as i32 * input.index_get(index)
        })
    }
    fn increment_grads(
        &self,
        input: &<i32 as Element<I::BitShape>>::Array,
        class_dist: &[u32; C],
        grads: &mut [(usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])); C],
        input_threshold: i32,
    ) -> u64 {
        let class_acts = <[(<i8 as Element<I::BitShape>>::Array, i8); C] as IIIVMM<I, [(); C]>>::iiivmm(self, input);
        class_dist
            .iter()
            .enumerate()
            .map(|(class, &count)| {
                let (max_index, &max_val) = class_acts.iter().enumerate().filter(|&(i, _)| i != class).max_by_key(|(_, v)| *v).unwrap();
                if class_acts[class] <= max_val {
                    for &(i, sign) in &[(class, false), (max_index, true)] {
                        (grads[i].1).1[sign as usize] += count;
                        grads[i].0 += count as usize;
                        <I::BitShape as MapMut<i32, [u32; 2]>>::map_mut(&mut (grads[i].1).0, &input, |grad, &input| {
                            grad[((input.is_positive()) ^ sign) as usize] += (input.abs() > input_threshold) as u32 * count;
                        });
                    }
                    0
                } else {
                    count as u64
                }
            })
            .sum()
    }
    fn update(&self, grads: &[(usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])); C], update_threshold: u32) -> Self {
        <[(); C] as ZipMap<
            (<i8 as Element<I::BitShape>>::Array, i8),
            (usize, (<[u32; 2] as Element<I::BitShape>>::Array, [u32; 2])),
            (<i8 as Element<I::BitShape>>::Array, i8),
        >>::zip_map(self, grads, |(w, b), (c, (wg, bg))| {
            (
                <I::BitShape as ZipMap<i8, [u32; 2], i8>>::zip_map(w, wg, |&w, &g| w.saturating_add(grad_counts_to_update(g, update_threshold))),
                b.saturating_add(grad_counts_to_update(*bg, update_threshold)),
            )
        })
    }
    fn descend(&self, patch_acts: &Vec<I>, sparse_patch_count_class_dists: &Vec<(Vec<(usize, u32)>, [u32; C])>, n_iters: usize, it: i32, ut: u32) -> Self {
        let total_n: u32 = sparse_patch_count_class_dists.iter().map(|(_, c)| c.iter()).flatten().sum();
        (0..n_iters).fold(<[(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8); C]>::default(), |aux_weights, i| {
            let (n_correct, aux_grads) = sparse_patch_count_class_dists.iter().fold(
                <(u64, [(usize, (<[u32; 2] as Element<<I as BitArray>::BitShape>>::Array, [u32; 2])); C])>::default(),
                |mut grads, (patch_bag, class)| {
                    let (hidden_act_n, hidden_act_counts) = patch_bag.iter().fold(
                        <(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array)>::default(),
                        |mut acc, &(index, count)| {
                            patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                            acc
                        },
                    );
                    let n = hidden_act_n as i32 / 2;
                    let hidden_acts = <<I as BitArray>::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| *count as i32 - n);
                    //dbg!(&hidden_acts);
                    let is_correct = <[(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<I, [(); C]>>::increment_grads(
                        &aux_weights,
                        &hidden_acts,
                        class,
                        &mut grads.1,
                        it,
                    );
                    //dbg!(loss);
                    grads.0 += is_correct as u64;
                    grads
                },
            );
            println!("{}: {}", i, n_correct as f64 / total_n as f64);
            <[(<i8 as Element<<I as BitArray>::BitShape>>::Array, i8); C] as IIIVMM<I, [(); C]>>::update(&aux_weights, &aux_grads, ut)
        })
    }
}

fn grad_counts_to_update(g: [u32; 2], threshold: u32) -> i8 {
    SIGNS[((g[0].saturating_sub(g[1]) | g[1].saturating_sub(g[0])) > threshold) as usize][(g[0] > g[1]) as usize]
}

const SIGNS: [[i8; 2]; 2] = [[0, 0], [1, -1]];

// Bit Trit Bit Vector Matrix Multipply
pub trait BTBVMM<I: BitArray, O: BitArray> {
    fn btbvmm(&self, input: &I) -> O;
    fn btbvmm_one_bit(&self, input: &I, index: &<O::BitShape as Shape>::Index) -> bool;
}

impl<I, O> BTBVMM<I, O> for <(I::TritArrayType, u32) as Element<O::BitShape>>::Array
where
    bool: Element<I::BitShape>,
    I: BitArray,
    O: BitArray + BitMapPack<(I::TritArrayType, u32)>,
    I::TritArrayType: MaskedDistance + TritArray<BitArrayType = I>,
    (I::TritArrayType, u32): Element<O::BitShape>,
    (I, u32): Element<O::BitShape>,
    <(I::TritArrayType, u32) as Element<O::BitShape>>::Array: IndexGet<<O::BitShape as Shape>::Index, Element = (I::TritArrayType, u32)>,
{
    fn btbvmm(&self, input: &I) -> O {
        <O as BitMapPack<(I::TritArrayType, u32)>>::bit_map_pack(self, |(weights, threshold)| weights.masked_distance(input) > *threshold)
    }
    fn btbvmm_one_bit(&self, input: &I, index: &<O::BitShape as Shape>::Index) -> bool {
        let (weights, threshold) = self.index_get(index);
        weights.masked_distance(input) > *threshold
    }
}

fn unary(input: [u8; 3]) -> b32 {
    to_10(input[0]) | (to_10(input[1]) << 10) | (to_10(input[2]) << 20)
}

fn act_loss(acts: &[i32; 10], class: usize) -> u64 {
    let (max_index, max_act) = acts.iter().enumerate().filter(|&(i, _)| i != class).max_by_key(|(_, v)| *v).unwrap();
    (max_act - acts[class]).abs() as u64
}

type PatchType = [[b32; 3]; 3];
type HiddenType = [b32; 4];

const N_EXAMPLES: usize = 1_000;

fn main() {
    let start = Instant::now();
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

    println!("init time: {:?}", start.elapsed());
    let centroid_start = Instant::now();
    let centroids = <PatchType as ImagePatchLloyds<_, [[(); 3]; 3]>>::lloyds(&mut rng, &examples, 500, 3, 0);
    println!("centroid time: {:?}", centroid_start.elapsed());
    dbg!(centroids.len());
    //dbg!(&centroids);

    let dist_centroid_start = Instant::now();
    let patch_dists: Vec<(Vec<u32>, usize)> = examples
        .par_iter()
        .map(|(image, class)| {
            (
                <[[b32; 3]; 3] as CentroidCount<StaticImage<b32, 32usize, 32usize>, [[(); 3]; 3], [(); 10]>>::centroid_count(image, &centroids),
                *class,
            )
        })
        .collect();
    dbg!();
    let patch_count_lloyds_time = Instant::now();
    let patch_dist_centroids = patch_count_lloyds(&patch_dists, 0, 3, centroids.len(), 300, 0);
    println!("patch count lloyds time: {:?}", patch_count_lloyds_time.elapsed());
    println!("dist centroid time: {:?}", dist_centroid_start.elapsed());
    dbg!(&patch_dist_centroids.len());

    let dist_centroid_start = Instant::now();
    // the class distributions of clusters of patch bags. Same len as patch_dist_centroids.
    let patch_bag_cluster_class_dists = cluster::class_dist::<10usize>(&patch_dists, &patch_dist_centroids);
    //dbg!(&patch_bag_cluster_class_dists);
    {
        // sanity checks
        assert_eq!(patch_bag_cluster_class_dists.len(), patch_dist_centroids.len());
        let sum: u32 = patch_bag_cluster_class_dists.iter().flatten().sum();
        assert_eq!(sum as usize, N_EXAMPLES);
    }
    let sparse_patch_count_class_dists: Vec<(Vec<(usize, u32)>, [u32; 10])> = patch_dist_centroids
        .iter()
        .zip(patch_bag_cluster_class_dists.iter())
        .map(|(patch_counts, class_dist)| (sparsify_centroid_count(patch_counts, 0), *class_dist))
        .collect();

    let sparse_patch_bags: Vec<Vec<(usize, u32)>> = patch_dist_centroids
        .par_iter()
        .map(|patch_counts| sparsify_centroid_count(patch_counts, 0))
        .collect();

    println!("dist centroid time: {:?}", dist_centroid_start.elapsed());

    let mut patch_weights = {
        let trits: <<PatchType as BitArray>::TritArrayType as Element<<HiddenType as BitArray>::BitShape>>::Array = rng.gen();

        <<HiddenType as BitArray>::BitShape as Map<<PatchType as BitArray>::TritArrayType, (<PatchType as BitArray>::TritArrayType, u32)>>::map(
            &trits,
            |&weights| {
                let mut target = <(<PatchType as BitArray>::TritArrayType, u32)>::default();
                target.0 = weights;
                target.1 = (<<PatchType as BitArray>::BitShape as Shape>::N as u32 - target.0.mask_zeros()) / 2;
                target
            },
        )
    };
    //dbg!(&patch_weights);
    let aux_weights = <[(<i8 as Element<<HiddenType as BitArray>::BitShape>>::Array, i8); 10]>::default();

    let patch_start = Instant::now();
    let patch_acts: Vec<HiddenType> = centroids.iter().map(|patch| patch_weights.btbvmm(patch)).collect();
    println!("patch act time: {:?}", patch_start.elapsed());
    //dbg!(patch_acts);

    let aux_start = Instant::now();
    let aux_weights = aux_weights.descend(&patch_acts, &sparse_patch_count_class_dists, 3, 300, 300);
    //dbg!(&aux_weights);
    println!("aux time: {:?}", aux_start.elapsed());

    let bit_index = (2, (7, ()));
    let partial_class_acts: Vec<[i32; 10]> = sparse_patch_bags
        .par_iter()
        .map(|patch_bag| {
            let (hidden_act_n, hidden_act_counts) = patch_bag.iter().fold(
                <(usize, <u32 as Element<<HiddenType as BitArray>::BitShape>>::Array)>::default(),
                |mut acc, &(index, count)| {
                    patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
                    //patch_acts[index].increment_frac_counters(&mut acc);
                    acc
                },
            );
            let n = hidden_act_n as i32 / 2;
            let hidden_acts = <<HiddenType as BitArray>::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| *count as i32 - n);
            let class_acts =
                <[(<i8 as Element<<HiddenType as BitArray>::BitShape>>::Array, i8); 10] as IIIVMM<HiddenType, [(); 10]>>::iiivmm(&aux_weights, &hidden_acts);
            <[(<i8 as Element<<HiddenType as BitArray>::BitShape>>::Array, i8); 10] as IIIVMM<HiddenType, [(); 10]>>::subtract_input_from_iiivmm_acts(
                &aux_weights,
                &hidden_acts,
                &class_acts,
                &bit_index,
            )
        })
        .collect();

    let n_iters = 1;
    let loss_start = Instant::now();
    for _ in 0..n_iters {
        let patch_acts: Vec<HiddenType> = centroids.par_iter().map(|patch| patch_weights.btbvmm(patch)).collect();
        //let n_bits: u32 = patch_acts.iter().map(|x| x.iter().map(|x| x.count_ones()).sum::<u32>()).sum();
        //dbg!(n_bits as f64 / (patch_acts.len() * 4 * 32) as f64);
        //dbg!(&patch_acts);
        //let sum_loss_start = Instant::now();

        for b in 0..32 * 4 {
            let (sum_n, sum_act) = sparse_patch_count_class_dists
                .iter()
                .map(|(patch_bag, class_counts)| {
                    let new_act_count: u32 = patch_bag.iter().map(|&(index, count)| patch_acts[index].bit(b) as u32 * count).sum();
                    let new_act_n: u32 = patch_bag.iter().map(|&(_, count)| count).sum();
                    (new_act_n, new_act_count)
                })
                .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

            //dbg!(sum_n);
            //dbg!(sum_act);
            dbg!(sum_act as f64 / sum_n as f64);
        }

        //sparse_patch_count_class_dists

        //let sum_loss: u64 = sparse_patch_count_class_dists
        //    .par_iter()
        //    .map(|(patch_bag, class_counts)| {
        //        let (hidden_act_n, hidden_act_counts) = patch_bag.iter().fold(
        //            <(usize, <u32 as Element<<HiddenType as BitArray>::BitShape>>::Array)>::default(),
        //            |mut acc, &(index, count)| {
        //                //patch_acts[index].weighted_increment_frac_counters(count, &mut acc);
        //                patch_acts[index].increment_frac_counters(&mut acc);
        //                acc
        //            },
        //        );
        //        let n = hidden_act_n as i32 / 2;
        //        let bit_index = 5;
        //        let new_act_count: u32 = patch_bag.iter().map(|&(index, count)| patch_acts[index].bit(bit_index) as u32).sum();
        //        //let new_act_n: u32 = patch_bag.iter().map(|&(_, count)| 1).sum();
        //        assert_eq!(new_act_n, hidden_act_n as u32);
        //        assert_eq!(new_act_count, hidden_act_counts[0][5]);
        //        let new_hidden_act = new_act_count as i32 - n;
        //        //println!("{} {} {}", n > hidden_act_counts[0][5] as i32, n, hidden_act_counts[0][5]);
        //        //let (hidden_act_n, hidden_act_counts) = <(usize, _)>::default();
        //        let hidden_acts = <<HiddenType as BitArray>::BitShape as Map<u32, i32>>::map(&hidden_act_counts, |count| *count as i32 - n);
        //        let class_acts = <[(<i8 as Element<<HiddenType as BitArray>::BitShape>>::Array, i8); 10] as IIIVMM<HiddenType, [(); 10]>>::iiivmm(
        //            &aux_weights,
        //            &hidden_acts,
        //        );
        //        class_counts
        //            .iter()
        //            .enumerate()
        //            .map(|(class, &count)| act_loss(&class_acts, class) * count as u64)
        //            .sum::<u64>()
        //    })
        //    .sum();
        ////println!("sum loss time: {:?}", sum_loss_start.elapsed());
        ////dbg!(sum_loss);
    }
    println!("avg loss time: {:?}", loss_start.elapsed() / n_iters);
}
