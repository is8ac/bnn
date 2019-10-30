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
    //use crate::bits::{BitMul, BitWord, HammingDistance};
    //use crate::count::Counters;
    //use crate::image2d::{ExtractPixels, PixelMap2D};
    //use bincode::{deserialize_from, serialize_into};
    //use rayon::prelude::*;
    use std::collections::HashSet;
    //use std::fs::File;
    //use std::io::BufWriter;
    //use std::path::Path;
    //use time::PreciseTime;

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
}
