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

impl<T: Counters, const L: usize> Counters for [T; L] {
    fn elementwise_add(&mut self, other: &[T; L]) {
        for i in 0..L {
            self[i].elementwise_add(&other[i]);
        }
    }
}
