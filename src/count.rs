pub trait ElementwiseAdd {
    fn elementwise_add(&mut self, other: &Self);
    fn weighted_elementwise_add(&mut self, other: &Self, weight: u32);
}

impl<A: ElementwiseAdd, B: ElementwiseAdd> ElementwiseAdd for (A, B) {
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
    }
    fn weighted_elementwise_add(&mut self, other: &Self, weight: u32) {
        self.0.weighted_elementwise_add(&other.0, weight);
        self.1.weighted_elementwise_add(&other.1, weight);
    }
}
impl<A: ElementwiseAdd, B: ElementwiseAdd, C: ElementwiseAdd> ElementwiseAdd for (A, B, C) {
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
        self.2.elementwise_add(&other.2);
    }
    fn weighted_elementwise_add(&mut self, other: &Self, weight: u32) {
        self.0.weighted_elementwise_add(&other.0, weight);
        self.1.weighted_elementwise_add(&other.1, weight);
        self.2.weighted_elementwise_add(&other.2, weight);
    }
}

impl<T: ElementwiseAdd> ElementwiseAdd for Vec<T> {
    fn elementwise_add(&mut self, other: &Vec<T>) {
        assert_eq!(self.len(), other.len());
        for i in 0..self.len() {
            self[i].elementwise_add(&other[i]);
        }
    }
    fn weighted_elementwise_add(&mut self, other: &Vec<T>, weight: u32) {
        assert_eq!(self.len(), other.len());
        for i in 0..self.len() {
            self[i].weighted_elementwise_add(&other[i], weight);
        }
    }
}

impl ElementwiseAdd for f32 {
    fn elementwise_add(&mut self, other: &f32) {
        *self += other;
    }
    fn weighted_elementwise_add(&mut self, other: &Self, weight: u32) {
        *self += other * weight as f32;
    }
}

impl ElementwiseAdd for f64 {
    fn elementwise_add(&mut self, other: &f64) {
        *self += other;
    }
    fn weighted_elementwise_add(&mut self, other: &Self, weight: u32) {
        *self += other * weight as f64;
    }
}

impl ElementwiseAdd for u32 {
    fn elementwise_add(&mut self, other: &u32) {
        *self += other;
    }
    fn weighted_elementwise_add(&mut self, other: &Self, weight: u32) {
        *self += other * weight;
    }
}

impl ElementwiseAdd for usize {
    fn elementwise_add(&mut self, other: &usize) {
        *self += other;
    }
    fn weighted_elementwise_add(&mut self, other: &Self, weight: u32) {
        *self += other * weight as usize;
    }
}

//impl<T: ElementwiseAdd> ElementwiseAdd for Box<T> {
//    fn elementwise_add(&mut self, other: &Self) {
//        <T as ElementwiseAdd>::elementwise_add(self, other);
//    }
//}

impl<T: ElementwiseAdd, const L: usize> ElementwiseAdd for [T; L] {
    fn elementwise_add(&mut self, other: &[T; L]) {
        for i in 0..L {
            self[i].elementwise_add(&other[i]);
        }
    }
    fn weighted_elementwise_add(&mut self, other: &[T; L], weight: u32) {
        for i in 0..L {
            self[i].weighted_elementwise_add(&other[i], weight);
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
