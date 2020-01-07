use crate::block::BlockCode;
use crate::shape::{Element, Shape};
use crate::unary::Preprocess;
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
/// If Patch is (), we use Self directly.
pub trait IncrementCounters<Patch, Preprocessor, Counters> {
    fn increment_counters(&self, class: usize, counters: &mut Counters);
}

impl<T, Preprocessor: Preprocess<T>, const K: usize, const C: usize>
    IncrementCounters<(), Preprocessor, CounterArray<Preprocessor::Output, [(); K], { C }>> for T
where
    Preprocessor::Output: BlockCode<[(); K]>,
{
    fn increment_counters(
        &self,
        class: usize,
        counters: &mut CounterArray<Preprocessor::Output, [(); K], { C }>,
    ) {
        let index = Preprocessor::preprocess(self).apply_block(&counters.bit_matrix);
        counters.counters[class][index] += 1;
    }
}

pub struct CounterArray<T: Element<K>, K: Shape, const C: usize> {
    pub bit_matrix: <T as Element<K>>::Array,
    pub counters: [Vec<u32>; C],
}

impl<T: BlockCode<[(); K]>, const K: usize, const C: usize> Default
    for CounterArray<T, [(); K], { C }>
where
    [Vec<u32>; C]: Default,
{
    fn default() -> Self {
        let mut counters = CounterArray {
            bit_matrix: T::encoder(),
            counters: <[Vec<u32>; C]>::default(),
        };
        for i in 0..C {
            counters.counters[i] = vec![0u32; 2usize.pow(K as u32)];
        }
        counters
    }
}

impl<T, const K: usize, const C: usize> ElementwiseAdd for CounterArray<T, [(); K], { C }>
where
    <T as Element<[(); K]>>::Array: Eq + std::fmt::Debug,
{
    fn elementwise_add(&mut self, other: &Self) {
        assert_eq!(self.bit_matrix, other.bit_matrix);
        for c in 0..C {
            assert_eq!(self.counters[c].len(), 2usize.pow(K as u32));
            assert_eq!(other.counters[c].len(), 2usize.pow(K as u32));
            self.counters[c]
                .iter_mut()
                .zip(other.counters[c].iter())
                .for_each(|(a, b)| *a += b);
        }
    }
}
