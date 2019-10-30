use crate::bits::{BitArray, IncrementFracCounters, IncrementHammingDistanceMatrix};
use crate::shape::{Element, Shape};
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
    );
}

impl<
        T: BitArray + Send + Sync + IncrementFracCounters + IncrementHammingDistanceMatrix<T>,
        const C: usize,
    > CountBits<{ C }> for T
where
    u32: Element<Self::BitShape>,
    bool: Element<Self::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<Self::BitShape>,
    Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>: Default + Send + Sync + Counters,
    Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>:
        Default + Send + Sync + Counters,
{
    fn count_bits(
        examples: &Vec<(Self, usize)>,
    ) -> (
        Box<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>,
        Box<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>,
    ) {
        examples
            .par_chunks(examples.len() / num_cpus::get_physical())
            .map(|chunk| {
                chunk.iter().fold(
                    (
                        Box::<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>::default(),
                        Box::<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>::default(),
                    ),
                    |mut acc, (image, class)| {
                        image.increment_frac_counters(&mut acc.0[*class]);
                        image.increment_hamming_distance_matrix(&mut *acc.1, image);
                        acc
                    },
                )
            })
            .reduce(
                || {
                    (
                        Box::<[(usize, <u32 as Element<Self::BitShape>>::Array); C]>::default(),
                        Box::<<<u32 as Element<Self::BitShape>>::Array as Element<Self::BitShape>>::Array>::default(),
                    )
                },
                |mut a, b| {
                    (a.0).elementwise_add(&b.0);
                    (a.1).elementwise_add(&b.1);
                    a
                },
            )
    }
}
