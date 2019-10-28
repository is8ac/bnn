#![feature(const_generics)]
#![feature(test)]

extern crate bincode;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate serde_derive;
extern crate time;

pub mod bits;
pub mod count;
pub mod datasets;
pub mod shape;
pub mod weight;

pub mod image2d {
    //use crate::bits::BitOr;

    pub trait OrPool<Output> {
        fn or_pool(&self) -> Output;
    }

    macro_rules! impl_orpool {
        ($x_size:expr, $y_size:expr) => {
            impl<Pixel: BitOr + Default + Copy> OrPool<[[Pixel; $y_size / 2]; $x_size / 2]>
                for [[Pixel; $y_size]; $x_size]
            {
                fn or_pool(&self) -> [[Pixel; $y_size / 2]; $x_size / 2] {
                    let mut target = [[Pixel::default(); $y_size / 2]; $x_size / 2];
                    for x in 0..$x_size / 2 {
                        let x_index = x * 2;
                        for y in 0..$y_size / 2 {
                            let y_index = y * 2;
                            target[x][y] = self[x_index + 0][y_index + 0]
                                .bit_or(&self[x_index + 0][y_index + 1])
                                .bit_or(&self[x_index + 1][y_index + 0])
                                .bit_or(&self[x_index + 1][y_index + 1]);
                        }
                    }
                    target
                }
            }
        };
    }

    //impl_orpool!(32, 32);
    //impl_orpool!(16, 16);
    //impl_orpool!(8, 8);
    //impl_orpool!(4, 4);

    pub trait AvgPool {
        type OutputImage;
        fn avg_pool(&self) -> Self::OutputImage;
    }

    macro_rules! impl_avgpool {
        ($x_size:expr, $y_size:expr) => {
            impl AvgPool for [[[u8; 3]; $y_size]; $x_size] {
                type OutputImage = [[[u8; 3]; $y_size / 2]; $x_size / 2];
                fn avg_pool(&self) -> Self::OutputImage {
                    let mut target = [[[0u8; 3]; $y_size / 2]; $x_size / 2];
                    for x in 0..$x_size / 2 {
                        let x_index = x * 2;
                        for y in 0..$y_size / 2 {
                            let y_index = y * 2;
                            for c in 0..3 {
                                let sum = self[x_index + 0][y_index + 0][c] as u16
                                    + self[x_index + 0][y_index + 1][c] as u16
                                    + self[x_index + 1][y_index + 0][c] as u16
                                    + self[x_index + 1][y_index + 1][c] as u16;
                                target[x][y][c] = (sum / 4) as u8;
                            }
                        }
                    }
                    target
                }
            }
        };
    }

    impl_avgpool!(32, 32);
    impl_avgpool!(16, 16);
    impl_avgpool!(8, 8);
    impl_avgpool!(4, 4);

    pub trait Normalize2D<P> {
        type OutputImage;
        fn normalize_2d(&self) -> Self::OutputImage;
    }

    // slide the min to 0
    impl<const X: usize, const Y: usize> Normalize2D<[u8; 3]> for [[[u8; 3]; Y]; X]
    where
        [[[u8; 3]; Y]; X]: Default,
    {
        type OutputImage = [[[u8; 3]; Y]; X];
        fn normalize_2d(&self) -> Self::OutputImage {
            let mut mins = [255u8; 3];
            for x in 0..X {
                for y in 0..Y {
                    for c in 0..3 {
                        mins[c] = self[x][y][c].min(mins[c]);
                    }
                }
            }
            let mut target = Self::OutputImage::default();
            for x in 0..X {
                for y in 0..Y {
                    for c in 0..3 {
                        target[x][y][c] = self[x][y][c] - mins[c];
                    }
                }
            }
            target
        }
    }

    pub trait ExtractPixels<P> {
        fn extract_pixels(&self, pixels: &mut Vec<P>);
    }

    impl<P: Copy, const X: usize, const Y: usize> ExtractPixels<P> for [[P; Y]; X] {
        fn extract_pixels(&self, pixels: &mut Vec<P>) {
            for x in 0..X {
                for y in 0..Y {
                    pixels.push(self[x][y]);
                }
            }
        }
    }

    pub trait PixelFold2D<P, C> {
        fn fold_2d<F: Fn(C, &P) -> C>(&self, acc: C, fold_fn: F) -> C;
    }

    impl<P, C, const X: usize, const Y: usize> PixelFold2D<P, C> for [[P; Y]; X] {
        // this is faster then `image.iter().flatten().sum()`
        fn fold_2d<F: Fn(C, &P) -> C>(&self, mut acc: C, fold_fn: F) -> C {
            for x in 0..X {
                for y in 0..Y {
                    acc = fold_fn(acc, &self[x][y]);
                }
            }
            acc
        }
    }

    pub trait PixelMap2D<I, O> {
        type OutputImage;
        fn map_2d<F: Fn(&I) -> O>(&self, map_fn: F) -> Self::OutputImage;
    }

    impl<I, O, const X: usize, const Y: usize> PixelMap2D<I, O> for [[I; Y]; X]
    where
        [[O; Y]; X]: Default,
    {
        type OutputImage = [[O; Y]; X];
        fn map_2d<F: Fn(&I) -> O>(&self, map_fn: F) -> Self::OutputImage {
            let mut target = <[[O; Y]; X]>::default();
            for x in 0..X {
                for y in 0..Y {
                    target[x][y] = map_fn(&self[x][y]);
                }
            }
            target
        }
    }

    pub trait Concat2D<A, B> {
        fn concat_2d(a: &A, b: &B) -> Self;
    }

    impl<A: Copy, B: Copy, const X: usize, const Y: usize> Concat2D<[[A; Y]; X], [[B; Y]; X]>
        for [[(A, B); Y]; X]
    where
        Self: Default,
    {
        fn concat_2d(a: &[[A; Y]; X], b: &[[B; Y]; X]) -> Self {
            let mut target = <Self>::default();
            for x in 0..X {
                for y in 0..Y {
                    target[x][y] = (a[x][y], b[x][y]);
                }
            }
            target
        }
    }

    // extracts patches and puts them in the pixels of the output image.
    pub trait Conv2D<OutputImage> {
        fn conv2d(&self) -> OutputImage;
    }

    macro_rules! impl_conv2d_2x2 {
        ($x:expr, $y:expr) => {
            impl<P: Copy + Default> Conv2D<[[[[P; 2]; 2]; $y / 2]; $x / 2]> for [[P; $y]; $x] {
                fn conv2d(&self) -> [[[[P; 2]; 2]; $y / 2]; $x / 2] {
                    let mut target = <[[[[P; 2]; 2]; $y / 2]; $x / 2]>::default();
                    for x in 0..$x / 2 {
                        let x_offset = x * 2;
                        for y in 0..$y / 2 {
                            let y_offset = y * 2;
                            for fx in 0..2 {
                                for fy in 0..2 {
                                    target[x][y][fx][fy] = self[x_offset + fx][y_offset + fy];
                                }
                            }
                        }
                    }
                    target
                }
            }
        };
    }

    impl_conv2d_2x2!(32, 32);
    impl_conv2d_2x2!(16, 16);
    impl_conv2d_2x2!(8, 8);
    impl_conv2d_2x2!(4, 4);

    impl<P: Copy, const X: usize, const Y: usize> Conv2D<[[[[P; 3]; 3]; Y]; X]> for [[P; Y]; X]
    where
        [[[[P; 3]; 3]; Y]; X]: Default,
    {
        fn conv2d(&self) -> [[[[P; 3]; 3]; Y]; X] {
            let mut target = <[[[[P; 3]; 3]; Y]; X]>::default();

            for fx in 1..3 {
                for fy in 1..3 {
                    target[0][0][fx][fy] = self[0 + fx][0 + fy];
                }
            }
            for y in 0..Y - 2 {
                for fx in 1..3 {
                    for fy in 0..3 {
                        target[0][y + 1][fx][fy] = self[0 + fx][y + fy];
                    }
                }
            }
            for fx in 1..3 {
                for fy in 0..2 {
                    target[0][Y - 1][fx][fy] = self[0 + fx][Y - 2 + fy];
                }
            }

            // begin center
            for x in 0..X - 2 {
                for fx in 0..3 {
                    for fy in 1..3 {
                        target[x + 1][0][fx][fy] = self[x + fx][0 + fy];
                    }
                }
                for y in 0..Y - 2 {
                    for fx in 0..3 {
                        for fy in 0..3 {
                            target[x + 1][y + 1][fx][fy] = self[x + fx][y + fy];
                        }
                    }
                }
                for fx in 0..3 {
                    for fy in 0..2 {
                        target[x + 1][Y - 1][fx][fy] = self[x + fx][Y - 2 + fy];
                    }
                }
            }
            // end center

            for fx in 0..2 {
                for fy in 1..3 {
                    target[X - 1][0][fx][fy] = self[X - 2 + fx][0 + fy];
                }
            }
            for y in 0..Y - 2 {
                for fx in 0..2 {
                    for fy in 0..3 {
                        target[X - 1][y + 1][fx][fy] = self[X - 2 + fx][y + fy];
                    }
                }
            }
            for fx in 0..2 {
                for fy in 0..2 {
                    target[X - 1][Y - 1][fx][fy] = self[X - 2 + fx][Y - 2 + fy];
                }
            }

            target
        }
    }
}

extern crate test;
#[cfg(test)]
mod tests {
    use super::image2d::PixelFold2D;
    use test::Bencher;

    const image8: [[u32; 8]; 8] = [[0u32; 8]; 8];
    const image32: [[u32; 32]; 32] = [[0u32; 32]; 32];

    #[bench]
    fn bench_image_fold_2d_8x8(b: &mut Bencher) {
        let image2 = image8; // stop the compile const propagating the image.
        b.iter(|| {
            let sum: u32 = image2.fold_2d(0u32, |acc, pixel| acc + pixel);
            sum
        });
    }
    #[bench]
    fn bench_iter_fold_2d_8x8(b: &mut Bencher) {
        let image2 = image8; // stop the compile const propagating the image.
        b.iter(|| {
            let sum: u32 = image2.iter().flatten().sum();
            sum
        });
    }

    #[bench]
    fn bench_image_fold_2d_32x32(b: &mut Bencher) {
        let image2 = image32; // stop the compile const propagating the image.
        b.iter(|| {
            let sum: u32 = image2.fold_2d(0u32, |acc, pixel| acc + pixel);
            sum
        });
    }
    #[bench]
    fn bench_iter_fold_2d_32x32(b: &mut Bencher) {
        let image2 = image32; // stop the compile const propagating the image.
        b.iter(|| {
            let sum: u32 = image2.iter().flatten().sum();
            sum
        });
    }
}

pub mod layer {
    use crate::bits::{BitMul, BitWord, HammingDistance};
    use crate::count::{Counters, IncrementCounters, IncrementFracCounters};
    use crate::image2d::{ExtractPixels, PixelMap2D};
    use bincode::{deserialize_from, serialize_into};
    use rayon::prelude::*;
    use std::collections::HashSet;
    use std::fs::File;
    use std::io::BufWriter;
    use std::path::Path;
    use time::PreciseTime;

    pub trait SupervisedLayer<InputImage, OutputImage> {
        fn supervised_split(
            examples: &Vec<(InputImage, usize)>,
            n_classes: usize,
            path: &Path,
        ) -> (Vec<(OutputImage, usize)>, Self);
    }

    impl<
            P: Send
                + Sync
                + HammingDistance
                + Lloyds
                + Copy
                + IncrementFracCounters
                + std::fmt::Debug
                + BitWord,
            InputImage: Sync + PixelMap2D<P, [u32; C], OutputImage = OutputImage> + ExtractPixels<P>,
            OutputImage: Sync + Send,
            const C: usize,
        > SupervisedLayer<InputImage, OutputImage> for [[(P, u32); 32]; C]
    where
        for<'de> Self: serde::Deserialize<'de>,
        Self: BitMul<P, [u32; C]> + Default + serde::Serialize + Default,
        P::BitCounterType: Default + Send + Counters + Clone,
        [[u32; C]; 10]: Default,
    {
        fn supervised_split(
            examples: &Vec<(InputImage, usize)>,
            n_classes: usize,
            path: &Path,
        ) -> (Vec<(OutputImage, usize)>, Self) {
            let weights: Self = File::open(&path)
                .map(|f| deserialize_from(f).unwrap())
                .ok()
                .unwrap_or_else(|| {
                    println!("no params found, training.");
                    let counters: Vec<(usize, <P as IncrementCounters>::BitCounterType)> = examples
                        .par_iter()
                        .fold(
                            || (0..n_classes).map(|_| (0, Default::default())).collect(),
                            |mut acc: Vec<(usize, <P as IncrementCounters>::BitCounterType)>,
                             (image, class)| {
                                for pixel in &{
                                    let mut pixels = Vec::new();
                                    image.extract_pixels(&mut pixels);
                                    pixels
                                } {
                                    pixel.increment_frac_counters(&mut acc[*class]);
                                }
                                acc
                            },
                        )
                        .reduce(
                            || (0..n_classes).map(|_| (0, Default::default())).collect(),
                            |a, b| {
                                a.iter()
                                    .cloned()
                                    .zip(b.iter())
                                    .map(|(a, b)| P::add_fracs(&a, b))
                                    .collect()
                            },
                        );

                    let partitions = gen_partitions(n_classes);
                    //partitions.sort_by_key(|x| x.len());
                    //partitions.reverse();
                    let partitions = &partitions[0..(C * 32)];
                    println!("{:?}", partitions);

                    let filters: Vec<P> = partitions
                        .iter()
                        .map(|elems| filter_from_split(&counters, elems))
                        .collect();
                    dbg!(filters.len());
                    //dbg!(&filters);

                    let pixels: Vec<P> = examples
                        .par_iter()
                        .fold(
                            || vec![],
                            |mut pixels, (image, _)| {
                                image.extract_pixels(&mut pixels);
                                pixels
                            },
                        )
                        .reduce(
                            || vec![],
                            |mut a, mut b| {
                                a.append(&mut b);
                                a
                            },
                        );
                    let features: Vec<(P, u32)> = filters
                        .iter()
                        .map(|filter| {
                            let mut activations: Vec<u32> = pixels
                                .par_iter()
                                .map(|pixel| pixel.hamming_distance(filter))
                                .collect();
                            activations.par_sort();
                            let threshold = activations[activations.len() / 2];
                            (*filter, threshold)
                        })
                        .collect();
                    let mut weights = <Self>::default();
                    for w in 0..C {
                        for b in 0..32 {
                            weights[w][b] = features[(w * 32) + b];
                        }
                    }
                    weights
                });
            println!("got params");
            let start = PreciseTime::now();
            let examples: Vec<(OutputImage, usize)> = examples
                .par_iter()
                .map(|(image, class)| {
                    (
                        image.map_2d(|x| {
                            let pixel = weights.bit_mul(x);
                            //dbg!(&pixel[0]);
                            pixel
                        }),
                        *class,
                    )
                })
                .collect();
            println!("time: {}", start.to(PreciseTime::now()));
            (examples, weights)
        }
    }

    pub trait UnsupervisedLayer<InputImage, OutputImage> {
        fn unsupervised_cluster<RNG: rand::Rng>(
            rng: &mut RNG,
            examples: &Vec<InputImage>,
            path: &Path,
        ) -> (Vec<OutputImage>, Self);
    }

    impl<
            P: Send + Sync + HammingDistance + Lloyds + Copy,
            InputImage: Sync + PixelMap2D<P, [u32; C], OutputImage = OutputImage> + ExtractPixels<P>,
            OutputImage: Sync + Send,
            const C: usize,
        > UnsupervisedLayer<InputImage, OutputImage> for [[(P, u32); 32]; C]
    where
        for<'de> Self: serde::Deserialize<'de>,
        Self: BitMul<P, [u32; C]> + Default + serde::Serialize,
    {
        fn unsupervised_cluster<RNG: rand::Rng>(
            rng: &mut RNG,
            images: &Vec<InputImage>,
            path: &Path,
        ) -> (Vec<OutputImage>, Self) {
            let weights: Self = File::open(&path)
                .map(|f| deserialize_from(f).unwrap())
                .ok()
                .unwrap_or_else(|| {
                    println!("no params found, training.");
                    let examples: Vec<P> = images
                        .par_iter()
                        .fold(
                            || vec![],
                            |mut pixels, image| {
                                image.extract_pixels(&mut pixels);
                                pixels
                            },
                        )
                        .reduce(
                            || vec![],
                            |mut a, mut b| {
                                a.append(&mut b);
                                a
                            },
                        );
                    dbg!(examples.len());
                    let clusters = <P>::lloyds(rng, &examples, C * 32);

                    let mut weights = Self::default();
                    for (i, filter) in clusters.iter().enumerate() {
                        weights[i / 32][i % 32] = *filter;
                    }
                    let mut f = BufWriter::new(File::create(path).unwrap());
                    serialize_into(&mut f, &weights).unwrap();
                    weights
                });
            println!("got params");
            //let start = PreciseTime::now();
            let images: Vec<OutputImage> = images
                .par_iter()
                .map(|image| image.map_2d(|x| weights.bit_mul(x)))
                .collect();
            //println!("time: {}", start.to(PreciseTime::now()));
            // PT2.04
            (images, weights)
        }
    }

    fn filter_from_split<T: IncrementCounters + IncrementFracCounters>(
        class_counts: &Vec<(usize, T::BitCounterType)>,
        indices: &HashSet<usize>,
    ) -> T
    where
        T::BitCounterType: Default + Counters,
    {
        let counters_0 = class_counts
            .iter()
            .enumerate()
            .filter(|(i, _)| indices.contains(i))
            .fold((0usize, T::BitCounterType::default()), |a, (_, b)| {
                T::add_fracs(&a, &b)
            });
        let counters_1 = class_counts
            .iter()
            .enumerate()
            .filter(|(i, _)| !indices.contains(i))
            .fold((0usize, T::BitCounterType::default()), |a, (_, b)| {
                T::add_fracs(&a, &b)
            });
        T::compare_fracs_and_bitpack(&counters_0, &counters_1)
    }

    fn gen_partitions(depth: usize) -> Vec<HashSet<usize>> {
        assert_ne!(depth, 0);
        if depth == 1 {
            vec![HashSet::new()]
        } else {
            let a = gen_partitions(depth - 1);
            a.iter()
                .cloned()
                .chain(a.iter().cloned().map(|mut x| {
                    x.insert(depth - 1);
                    x
                }))
                .collect()
        }
    }

    pub trait Lloyds
    where
        Self: IncrementCounters + Sized,
    {
        fn update_centers(example: &Vec<Self>, centers: &Vec<Self>) -> Vec<Self>;
        fn lloyds<RNG: rand::Rng>(
            rng: &mut RNG,
            examples: &Vec<Self>,
            k: usize,
        ) -> Vec<(Self, u32)>;
    }

    impl<T: Send + Sync + Copy + HammingDistance + BitWord + Eq + IncrementFracCounters> Lloyds for T
    where
        <Self as IncrementCounters>::BitCounterType:
            Default + Send + Sync + Counters + std::fmt::Debug + Clone,
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        fn update_centers(examples: &Vec<Self>, centers: &Vec<Self>) -> Vec<Self> {
            let counters: Vec<(usize, Self::BitCounterType)> = examples
                .par_iter()
                .fold(
                    || {
                        centers
                            .iter()
                            .map(|_| {
                                (
                                    0usize,
                                    <Self as IncrementCounters>::BitCounterType::default(),
                                )
                            })
                            .collect()
                    },
                    |mut acc: Vec<(usize, Self::BitCounterType)>, example| {
                        let cell_index = centers
                            .iter()
                            .map(|center| center.hamming_distance(example))
                            .enumerate()
                            .min_by_key(|(_, hd)| *hd)
                            .unwrap()
                            .0;
                        example.increment_frac_counters(&mut acc[cell_index]);
                        acc
                    },
                )
                .reduce(
                    || {
                        centers
                            .iter()
                            .map(|_| {
                                (
                                    0usize,
                                    <Self as IncrementCounters>::BitCounterType::default(),
                                )
                            })
                            .collect()
                    },
                    |a, b| {
                        a.iter()
                            .cloned()
                            .zip(b.iter())
                            .map(|(a, b)| T::add_fracs(&a, &b))
                            .collect()
                    },
                );
            counters
                .iter()
                .map(|(n_examples, counters)| {
                    <Self>::threshold_and_bitpack(&counters, *n_examples as u32 / 2)
                })
                .collect()
        }
        fn lloyds<RNG: rand::Rng>(
            rng: &mut RNG,
            examples: &Vec<Self>,
            k: usize,
        ) -> Vec<(Self, u32)> {
            let mut centroids: Vec<Self> = (0..k).map(|_| rng.gen()).collect();
            for i in 0..1000 {
                dbg!(i);
                let new_centroids = <Self>::update_centers(examples, &centroids);
                if new_centroids == centroids {
                    break;
                }
                centroids = new_centroids;
            }
            centroids
                .iter()
                .map(|centroid| {
                    let mut activations: Vec<u32> = examples
                        .par_iter()
                        .map(|example| example.hamming_distance(centroid))
                        .collect();
                    activations.par_sort();
                    (*centroid, activations[examples.len() / 2])
                    //(*centroid, Self::BIT_LEN as u32 / 2)
                })
                .collect()
        }
    }

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

    pub trait GlobalPoolingBayes<Pixel, const K: usize>
    where
        Self: Sized,
        Pixel: Sized,
    {
        fn global_pooling_bayes(examples: &Vec<(Self, usize)>) -> [(Pixel, u32); K];
        fn infer(&self, weights: &[(Pixel, u32); K]) -> [u32; K];
    }

    impl<
            Pixel: Sized
                + IncrementCounters
                + HammingDistance
                + IncrementFracCounters
                + Send
                + Sync
                + std::fmt::Debug
                + Copy,
            const X: usize,
            const Y: usize,
            const K: usize,
        > GlobalPoolingBayes<Pixel, { K }> for [[Pixel; Y]; X]
    where
        [u32; K]: Default + std::fmt::Debug,
        Pixel::BitCounterType: Counters + Send + Sync,
        [(usize, Pixel::BitCounterType); K]: Default,
        [(Pixel, u32); K]: Default + std::fmt::Debug,
        [[Pixel; Y]; X]: Send + Sync + ExtractPixels<Pixel>,
    {
        fn global_pooling_bayes(examples: &Vec<([[Pixel; Y]; X], usize)>) -> [(Pixel, u32); K] {
            let counters: [(usize, Pixel::BitCounterType); K] = examples
                .par_iter()
                .fold(
                    || <[(usize, Pixel::BitCounterType); K]>::default(),
                    |acc, (example, class)| {
                        {
                            let mut pixels: Vec<Pixel> = Vec::new();
                            example.extract_pixels(&mut pixels);
                            pixels
                        }
                        .iter()
                        .fold(acc, |mut acc, pixel| {
                            pixel.increment_frac_counters(&mut acc[*class]);
                            acc
                        })
                    },
                )
                .reduce(
                    || <[(usize, Pixel::BitCounterType); K]>::default(),
                    |mut a, b| {
                        for c in 0..K {
                            Pixel::add_assign_fracs(&mut a[c], &b[c]);
                        }
                        a
                    },
                );
            let filters: Vec<Pixel> = counters
                .par_iter()
                .map(|(n, bit_counter)| Pixel::threshold_and_bitpack(bit_counter, *n as u32 / 2))
                .collect();
            let dist: Vec<u64> = examples
                .par_iter()
                .fold(
                    || vec![0u64; 10],
                    |mut acc, (image, _)| {
                        for pixel in {
                            let mut pixels: Vec<Pixel> = Vec::new();
                            image.extract_pixels(&mut pixels);
                            pixels
                        }
                        .iter()
                        {
                            for c in 0..10 {
                                acc[c] += filters[c].hamming_distance(pixel) as u64;
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![0u64; 10],
                    |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect(),
                );

            let max_avg_act: f64 = *dist.iter().max().unwrap() as f64 / examples.len() as f64;
            let biases: Vec<u32> = dist
                .iter()
                .map(|&x| (max_avg_act - (x as f64 / examples.len() as f64)) as u32)
                .collect();

            let mut weights = <[(Pixel, u32); K]>::default();
            for c in 0..K {
                weights[c] = (filters[c], biases[c]);
            }
            let n_correct: usize = examples
                .par_iter()
                .map(|(image, class)| {
                    let activations = image.infer(&weights);
                    let index = activations.min_index();
                    (index == *class) as usize
                })
                .sum();
            println!(
                "global pool: {:?}%",
                n_correct as f64 / examples.len() as f64 * 100.0
            );

            weights
        }
        fn infer(&self, filters: &[(Pixel, u32); K]) -> [u32; K] {
            {
                let mut pixels: Vec<Pixel> = Vec::new();
                self.extract_pixels(&mut pixels);
                pixels
            }
            .iter()
            .fold(<[u32; K]>::default(), |mut acc, pixel| {
                for c in 0..K {
                    acc[c] += filters[c].0.hamming_distance(pixel);
                    //acc[c] += filters[c].0.hamming_distance(pixel) + filters[c].1;
                }
                acc
            })
        }
    }

    pub trait NaiveBayes<const K: usize>
    where
        Self: Sized,
    {
        fn naive_bayes_classify(examples: &Vec<(Self, usize)>) -> [(Self, u32); K];
    }

    impl<
            T: HammingDistance
                + IncrementCounters
                + IncrementFracCounters
                + Sync
                + Send
                + Default
                + Copy,
            const K: usize,
        > NaiveBayes<{ K }> for T
    where
        [(usize, T::BitCounterType); K]: Default + Sync + Send,
        T::BitCounterType: Counters + Send + Sync + Default,
        [(T, u32); K]: Default,
    {
        fn naive_bayes_classify(examples: &Vec<(T, usize)>) -> [(T, u32); K] {
            let counters: Vec<(usize, T::BitCounterType)> = examples
                .par_iter()
                .fold(
                    || {
                        (0..K)
                            .map(|_| <(usize, T::BitCounterType)>::default())
                            .collect::<Vec<_>>()
                    },
                    |mut acc, (example, class)| {
                        acc[*class].0 += 1;
                        example.increment_counters(&mut acc[*class].1);
                        acc
                    },
                )
                .reduce(
                    || {
                        (0..K)
                            .map(|_| <(usize, T::BitCounterType)>::default())
                            .collect::<Vec<_>>()
                    },
                    |mut a, b| {
                        for c in 0..K {
                            T::add_assign_fracs(&mut a[c], &b[c]);
                        }
                        a
                    },
                );
            let filters: Vec<T> = counters
                .par_iter()
                .map(|(n, bit_counter)| T::threshold_and_bitpack(bit_counter, *n as u32 / 2))
                .collect();

            let dist: Vec<u64> = examples
                .par_iter()
                .fold(
                    || vec![0u64; 10],
                    |mut acc, (image, _)| {
                        for c in 0..10 {
                            acc[c] += filters[c].hamming_distance(image) as u64;
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![0u64; 10],
                    |a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect(),
                );
            let max_avg_act: f64 = *dist.iter().max().unwrap() as f64 / examples.len() as f64;
            let biases: Vec<u32> = dist
                .iter()
                .map(|&x| (max_avg_act - (x as f64 / examples.len() as f64)) as u32)
                .collect();

            let mut target = <[(T, u32); K]>::default();
            for c in 0..K {
                target[c] = (filters[c], biases[c]);
            }
            let n_correct: u64 = examples
                .par_iter()
                .map(|(image, class)| {
                    let actual_class: usize = target
                        .iter()
                        .enumerate()
                        .min_by_key(|(_i, (filter, _bias))| filter.hamming_distance(image))
                        //.min_by_key(|(i, (filter, bias))| filter.hamming_distance(image) + bias)
                        .unwrap()
                        .0;
                    (actual_class == *class) as u64
                })
                .sum();
            println!(
                "Fully connected: {:?}%",
                n_correct as f64 / examples.len() as f64 * 100f64
            );

            target
        }
    }
}
