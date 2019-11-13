use crate::bits::{BitArray, BitArrayOPs, IncrementFracCounters, IncrementHammingDistanceMatrix};
use crate::image2d::Image2D;
use crate::shape::Element;
//use crate::unary::NormalizeAndBitpack;
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

/// There are two levels of indirection.
/// The first to change the shape by extracting many patches from it,
/// the second to change the type.
/// We take a 'Self', extract a 'Patch` from it, normalise it to `T`,
/// and use it to increment the counters.
pub trait IncrementCounters<Patch, T: BitArray, const C: usize>
where
    u32: Element<T::BitShape>,
    u32: Element<<T as BitArray>::BitShape>,
    <u32 as Element<T::BitShape>>::Array: Element<T::BitShape>,
{
    fn increment_counters(
        &self,
        class: usize,
        value_counters: &mut [(usize, <u32 as Element<T::BitShape>>::Array); C],
        counters_matrix: &mut <<u32 as Element<T::BitShape>>::Array as Element<T::BitShape>>::Array,
        n_examples: &mut usize,
    );
}
