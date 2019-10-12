use crate::bits::BitLen;

pub trait Counters {
    type FloatRatioType;
    fn elementwise_add(&mut self, other: &Self);
    fn divide(&self, count: f64) -> Self::FloatRatioType;
    // for each bit independently, probability that input is member of self set given bit.
    fn bayes(&self, n_self: f64, other: &Self, n: f64, p_self: f64) -> Self::FloatRatioType;
}

impl<A: Counters, B: Counters> Counters for (A, B) {
    type FloatRatioType = (A::FloatRatioType, B::FloatRatioType);
    fn elementwise_add(&mut self, other: &Self) {
        self.0.elementwise_add(&other.0);
        self.1.elementwise_add(&other.1);
    }
    fn divide(&self, count: f64) -> Self::FloatRatioType {
        (self.0.divide(count), self.1.divide(count))
    }
    fn bayes(&self, n_self: f64, other: &Self, n: f64, p_self: f64) -> Self::FloatRatioType {
        (
            self.0.bayes(n_self, &other.0, n, p_self),
            self.1.bayes(n_self, &other.1, n, p_self),
        )
    }
}

impl Counters for u32 {
    type FloatRatioType = f64;
    fn elementwise_add(&mut self, other: &u32) {
        *self += other;
    }
    fn divide(&self, count: f64) -> Self::FloatRatioType {
        *self as f64 / count
    }
    fn bayes(&self, n_self: f64, &other: &u32, n: f64, p_self: f64) -> Self::FloatRatioType {
        let pb = (*self + other) as f64 / n;
        let pba = *self as f64 / n_self;
        (pba * p_self) / pb
    }
}

impl<T: Counters, const L: usize> Counters for [T; L]
where
    [T::FloatRatioType; L]: Default,
{
    type FloatRatioType = [T::FloatRatioType; L];
    fn elementwise_add(&mut self, other: &[T; L]) {
        for i in 0..L {
            self[i].elementwise_add(&other[i]);
        }
    }
    fn divide(&self, count: f64) -> Self::FloatRatioType {
        let mut target = <[T::FloatRatioType; L]>::default();
        for b in 0..L {
            target[b] = self[b].divide(count);
        }
        target
    }
    fn bayes(&self, n_self: f64, other: &Self, n: f64, p_self: f64) -> Self::FloatRatioType {
        let mut target = <[T::FloatRatioType; L]>::default();
        for b in 0..L {
            target[b] = self[b].bayes(n_self, &other[b], n, p_self);
        }
        target
    }
}

pub trait IncrementFracCounters
where
    Self: IncrementCounters,
{
    fn increment_frac_counters(&self, counters: &mut (usize, Self::BitCounterType));
    fn add_fracs(
        a: &(usize, Self::BitCounterType),
        b: &(usize, Self::BitCounterType),
    ) -> (usize, Self::BitCounterType);
    fn add_assign_fracs(
        a: &mut (usize, Self::BitCounterType),
        b: &(usize, Self::BitCounterType),
    );
}

//impl<T: IncrementCounters> IncrementFracCounters for T
//where
//    T::BitCounterType: Counters + Clone,
//{
//    fn increment_frac_counters(&self, counters: &mut (usize, Self::BitCounterType)) {
//        counters.0 += 1;
//        self.increment_counters(&mut counters.1);
//    }
//    fn add_fracs(
//        a: &(usize, Self::BitCounterType),
//        b: &(usize, Self::BitCounterType),
//    ) -> (usize, Self::BitCounterType) {
//        let mut result = (*a).clone();
//        result.0 += b.0;
//        result.1.elementwise_add(&b.1);
//        result
//    }
//    fn add_assign_fracs(
//        a: &mut (usize, Self::BitCounterType),
//        b: &(usize, Self::BitCounterType),
//    ) {
//        a.0 += b.0;
//        a.1.elementwise_add(&b.1);
//    }
//}

pub trait IncrementCounters
where
    Self: Sized,
{
    type BitCounterType;
    fn increment_counters(&self, counters: &mut Self::BitCounterType);
    fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self;
    fn compare_and_bitpack(
        counters_0: &Self::BitCounterType,
        n_0: f64,
        counters_1: &Self::BitCounterType,
        n_1: f64,
    ) -> Self;
    fn compare_fracs_and_bitpack(
        a: &(usize, Self::BitCounterType),
        b: &(usize, Self::BitCounterType),
    ) -> Self {
        Self::compare_and_bitpack(&a.1, a.0 as f64, &b.1, b.0 as f64)
    }
}

//impl<A: IncrementCounters, B: IncrementCounters> IncrementCounters for (A, B) {
//    type BitCounterType = (A::BitCounterType, B::BitCounterType);
//    fn increment_counters(&self, counters: &mut Self::BitCounterType) {
//        self.0.increment_counters(&mut counters.0);
//        self.1.increment_counters(&mut counters.1);
//    }
//    fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
//        (
//            A::threshold_and_bitpack(&counters.0, threshold),
//            B::threshold_and_bitpack(&counters.1, threshold),
//        )
//    }
//    fn compare_and_bitpack(
//        counters_0: &Self::BitCounterType,
//        n_0: f64,
//        counters_1: &Self::BitCounterType,
//        n_1: f64,
//    ) -> Self {
//        (
//            A::compare_and_bitpack(&counters_0.0, n_0, &counters_1.0, n_1),
//            B::compare_and_bitpack(&counters_0.1, n_0, &counters_1.1, n_1),
//        )
//    }
//}

//impl<T: IncrementCounters, const L: usize> IncrementCounters for [T; L]
//where
//    Self: Default,
//{
//    type BitCounterType = [T::BitCounterType; L];
//    fn increment_counters(&self, counters: &mut [T::BitCounterType; L]) {
//        for i in 0..L {
//            self[i].increment_counters(&mut counters[i]);
//        }
//    }
//    fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
//        let mut target = <[T; L]>::default();
//        for i in 0..L {
//            target[i] = T::threshold_and_bitpack(&counters[i], threshold);
//        }
//        target
//    }
//    fn compare_and_bitpack(
//        counters_0: &Self::BitCounterType,
//        n_0: f64,
//        counters_1: &Self::BitCounterType,
//        n_1: f64,
//    ) -> Self {
//        let mut target = <[T; L]>::default();
//        for i in 0..L {
//            target[i] = T::compare_and_bitpack(&counters_0[i], n_0, &counters_1[i], n_1);
//        }
//        target
//    }
//}
//macro_rules! impl_for_uint {
//    ($type:ty, $len:expr) => {
//        impl IncrementCounters for $type {
//            type BitCounterType = [u32; <$type>::BIT_LEN];
//            fn increment_counters(&self, counters: &mut Self::BitCounterType) {
//                for b in 0..<$type>::BIT_LEN {
//                    counters[b] += ((self >> b) & 1) as u32
//                }
//            }
//            fn threshold_and_bitpack(counters: &Self::BitCounterType, threshold: u32) -> Self {
//                let mut target = <$type>::default();
//                for i in 0..$len {
//                    target |= (counters[i] > threshold) as $type << i;
//                }
//                target
//            }
//            fn compare_and_bitpack(counters_0: &Self::BitCounterType, n_0: f64, counters_1: &Self::BitCounterType, n_1: f64) -> Self {
//                let mut target = <$type>::default();
//                for i in 0..$len {
//                    target |= (counters_0[i] as f64 / n_0 > counters_1[i] as f64 / n_1) as $type << i;
//                }
//                target
//            }
//        }
//    }
//}
//impl_for_uint!(u32, 32);
//impl_for_uint!(u16, 16);
//impl_for_uint!(u8, 8);
