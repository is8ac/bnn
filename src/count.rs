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

impl<T: ElementwiseAdd> ElementwiseAdd for Vec<T> {
    fn elementwise_add(&mut self, other: &Vec<T>) {
        assert_eq!(self.len(), other.len());
        for i in 0..self.len() {
            self[i].elementwise_add(&other[i]);
        }
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
