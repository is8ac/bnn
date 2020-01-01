use crate::bits::{
    BitArray, IncrementCooccurrenceMatrix, IncrementFracCounters, IncrementHammingDistanceMatrix,
};
use crate::shape::Element;
use crate::unary::NormalizeAndBitpack;
use std::boxed::Box;

pub trait ElementwiseAdd {
    fn elementwise_add(&mut self, other: &Self);
}

impl<A: ElementwiseAdd, B: ElementwiseAdd> ElementwiseAdd for (A, B) {
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
    }
}
impl<A: ElementwiseAdd, B: ElementwiseAdd, C: ElementwiseAdd> ElementwiseAdd for (A, B, C) {
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
        self.2.elementwise_add(&other.2);
    }
}

impl ElementwiseAdd for Vec<u32> {
    fn elementwise_add(&mut self, other: &Vec<u32>) {
        self.iter_mut().zip(other.iter()).for_each(|(a, b)| *a += b);
    }
}

impl ElementwiseAdd for u32 {
    fn elementwise_add(&mut self, other: &u32) {
        *self += other;
    }
}

impl ElementwiseAdd for usize {
    fn elementwise_add(&mut self, other: &usize) {
        *self += other;
    }
}
impl<T: ElementwiseAdd> ElementwiseAdd for Box<T> {
    fn elementwise_add(&mut self, other: &Self) {
        <T as ElementwiseAdd>::elementwise_add(self, other);
    }
}

impl<T: ElementwiseAdd, const L: usize> ElementwiseAdd for [T; L] {
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
/// and use T to increment the counters.
pub trait IncrementCounters<Patch, T, Counters> {
    fn increment_counters(&self, class: usize, counters: &mut Counters);
}

impl<
        Input,
        T: BitArray + IncrementFracCounters + IncrementHammingDistanceMatrix<T>,
        const C: usize,
    >
    IncrementCounters<
        Input,
        T,
        Box<(
            [(usize, <u32 as Element<T::BitShape>>::Array); C],
            <<u32 as Element<T::BitShape>>::Array as Element<T::BitShape>>::Array,
            usize,
        )>,
    > for Input
where
    Input: NormalizeAndBitpack<T>,
    u32: Element<T::BitShape>,
    <u32 as Element<T::BitShape>>::Array: Element<T::BitShape>,
{
    fn increment_counters(
        &self,
        class: usize,
        counters: &mut Box<(
            [(usize, <u32 as Element<T::BitShape>>::Array); C],
            <<u32 as Element<T::BitShape>>::Array as Element<T::BitShape>>::Array,
            usize,
        )>,
    ) {
        let normalized = self.normalize_and_bitpack();
        normalized.increment_frac_counters(&mut counters.0[class]);
        normalized.increment_hamming_distance_matrix(&mut counters.1, &normalized);
        counters.2 += 1;
    }
}

impl<Input, T: BitArray + IncrementCooccurrenceMatrix<T>>
    IncrementCounters<
        Input,
        T,
        Box<(
            <[(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array); 2] as Element<
                <T as BitArray>::BitShape,
            >>::Array,
            usize,
        )>,
    > for Input
where
    [(usize, <u32 as Element<T::BitShape>>::Array); 2]: Element<T::BitShape>,
    Self: NormalizeAndBitpack<T>,
    u32: Element<T::BitShape>,
    <u32 as Element<T::BitShape>>::Array: Element<T::BitShape>,
{
    fn increment_counters(
        &self,
        _: usize,
        counters: &mut Box<(
            <[(usize, <u32 as Element<<T as BitArray>::BitShape>>::Array); 2] as Element<
                <T as BitArray>::BitShape,
            >>::Array,
            usize,
        )>,
    ) {
        let normalized = self.normalize_and_bitpack();
        normalized.increment_cooccurrence_matrix(&mut counters.0, &normalized);
        counters.1 += 1;
    }
}
