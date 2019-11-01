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
pub mod image2d;
pub mod shape;
pub mod unary;
pub mod weight;

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
    //use std::collections::HashSet;
    //use std::fs::File;
    //use std::io::BufWriter;
    //use std::path::Path;
    //use time::PreciseTime;

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
