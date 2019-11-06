use crate::bits::{BitArray, BitArrayOPs, IncrementFracCounters, IncrementHammingDistanceMatrix};
use crate::image2d::{ConvIncrementCounters, Image2D};
use crate::shape::Element;
use crate::unary::Normalize2D;
use rayon::prelude::*;
use std::boxed::Box;

pub trait Counters {
    fn elementwise_add(&mut self, other: &Self);
}

impl<A: Counters, B: Counters> Counters for (A, B) {
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
    }
}

impl Counters for u32 {
    fn elementwise_add(&mut self, other: &u32) {
        *self += other;
    }
}

impl Counters for usize {
    fn elementwise_add(&mut self, other: &usize) {
        *self += other;
    }
}
impl<T: Counters> Counters for Box<T> {
    fn elementwise_add(&mut self, other: &Self) {
        <T as Counters>::elementwise_add(self, other);
    }
}

impl<T: Counters, const L: usize> Counters for [T; L] {
    fn elementwise_add(&mut self, other: &[T; L]) {
        for i in 0..L {
            self[i].elementwise_add(&other[i]);
        }
    }
}

// fully connected examples
pub trait CountBits<const C: usize>
where
    Self: BitArray,
    u32: Element<Self::BitShape>,
    bool: Element<Self::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<Self::BitShape>,
{
    fn count_bits(
        examples: &Vec<(Self, usize)>,
    ) -> (
        Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>,
        Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>,
        usize,
    );
}

impl<
        T: BitArray
            + BitArrayOPs
            + Send
            + Sync
            + IncrementFracCounters
            + IncrementHammingDistanceMatrix<T>,
        const C: usize,
    > CountBits<{ C }> for T
where
    u32: Element<Self::BitShape>,
    bool: Element<Self::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<Self::BitShape>,
    Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>: Default + Send + Sync + Counters,
    Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>:
        Default + Send + Sync + Counters,
    //(
    //    Box<[(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array); C]>,
    //    Box<<<u32 as Element<T::BitShape>>::Array as Element<T::BitShape>>::Array>,
    //    usize,
    //): Counters,
{
    fn count_bits(
        examples: &Vec<(Self, usize)>,
    ) -> (
        Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>,
        Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>,
        usize,
    ) {
        examples
            .par_chunks(examples.len() / num_cpus::get_physical())
            .map(|chunk| {
                chunk.iter().fold(
                    (
                        Box::<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>::default(),
                        Box::<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>::default(),
                        0usize,
                    ),
                    |mut acc, (image, class)| {
                        image.increment_frac_counters(&mut acc.0[*class]);
                        image.increment_hamming_distance_matrix(&mut *acc.1, image);
                        acc.2 +=1;
                        acc
                    },
                )
            })
            .reduce(
                || {
                    (
                        Box::<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>::default(),
                        Box::<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>::default(),
                        0usize,
                    )
                },
                |mut a, b| {
                    (a.0).elementwise_add(&b.0);
                    (a.1).elementwise_add(&b.1);
                    (a.2).elementwise_add(&b.2);
                    a
                },
            )
    }
}

pub trait CountBitsConv<Image, const C: usize>
where
    Self: BitArray,
    u32: Element<Self::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<Self::BitShape>,
{
    fn count_bits_conv(
        examples: &Vec<(Image, usize)>,
    ) -> (
        Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>,
        Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>,
        usize,
    );
}

impl<T: Copy + BitArray, IP: Copy, const X: usize, const Y: usize, const C: usize>
    CountBitsConv<[[IP; Y]; X], { C }> for [[T; 3]; 3]
where
    Self: BitArray,
    [[IP; Y]; X]: ConvIncrementCounters<Self, { C }> + Image2D<PixelType = IP>,
    [[IP; 3]; 3]: Normalize2D<Self>,
    ([[IP; Y]; X], usize): Send + Sync,
    u32: Element<Self::BitShape>,
    bool: Element<Self::BitShape>,
    u32: Element<T::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<Self::BitShape>,
    Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>: Default + Sync + Send + Counters,
    Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>:
        Default + Send + Sync + Counters,
{
    fn count_bits_conv(
        examples: &Vec<([[IP; Y]; X], usize)>,
    ) -> (
        Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>,
        Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>,
        usize,
    ) {
        examples
            .par_chunks(examples.len() / num_cpus::get_physical())
            .map(|chunk| {
                chunk.iter().fold(
                    (
                        Box::<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>::default(),
                        Box::<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>::default(),
                        0usize,
                    ),
                    |mut acc, (image, class)| {
                        image.conv_increment_counters(*class, &mut acc.0, &mut acc.1, &mut acc.2);
                        acc
                    },
                )
            })
            .reduce(
                || {
                    (
                        Box::<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>::default(),
                        Box::<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>::default(),
                        0usize,
                    )
                },
                |mut a, b| {
                    (a.0).elementwise_add(&b.0);
                    (a.1).elementwise_add(&b.1);
                    (a.2).elementwise_add(&b.2);
                    a
                },
            )
    }
}
