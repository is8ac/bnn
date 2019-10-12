pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
}

impl<T: HammingDistance, const L: usize> HammingDistance for [T; L] {
    fn hamming_distance(&self, other: &[T; L]) -> u32 {
        let mut distance = 0u32;
        for i in 0..L {
            distance += self[i].hamming_distance(&other[i]);
        }
        distance
    }
}

//impl<A: HammingDistance, B: HammingDistance> HammingDistance for (A, B) {
//    fn hamming_distance(&self, other: &(A, B)) -> u32 {
//        self.0.hamming_distance(&other.0) + self.1.hamming_distance(&other.1)
//    }
//}

pub trait BitOr {
    fn bit_or(&self, other: &Self) -> Self;
}

impl<T: BitOr, const L: usize> BitOr for [T; L]
where
    [T; L]: Default,
{
    fn bit_or(&self, other: &Self) -> Self {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_or(&other[i]);
        }
        target
    }
}

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

impl<A: BitLen, B: BitLen> BitLen for (A, B) {
    const BIT_LEN: usize = A::BIT_LEN + B::BIT_LEN;
}

macro_rules! array_bit_len {
    ($len:expr) => {
        impl<T: BitLen> BitLen for [T; $len] {
            const BIT_LEN: usize = $len * T::BIT_LEN;
        }
    };
}

array_bit_len!(1);
array_bit_len!(2);
array_bit_len!(3);
array_bit_len!(4);
array_bit_len!(5);
array_bit_len!(6);
array_bit_len!(7);
array_bit_len!(8);
array_bit_len!(13);
array_bit_len!(16);
array_bit_len!(25);
array_bit_len!(32);

pub trait FlipBit {
    fn flip_bit(&mut self, b: usize);
}

impl<T: BitLen + FlipBit, const L: usize> FlipBit for [T; L] {
    fn flip_bit(&mut self, index: usize) {
        self[index / T::BIT_LEN].flip_bit(index % T::BIT_LEN);
    }
}

pub trait GetBit {
    fn bit(&self, i: usize) -> bool;
}

impl<T: GetBit + BitLen, const L: usize> GetBit for [T; L] {
    fn bit(&self, i: usize) -> bool {
        self[i / T::BIT_LEN].bit(i % T::BIT_LEN)
    }
}

macro_rules! impl_for_uint {
    ($type:ty, $len:expr) => {
        impl BitLen for $type {
            const BIT_LEN: usize = $len;
        }
        impl FlipBit for $type {
            fn flip_bit(&mut self, index: usize) {
                *self ^= 1 << index
            }
        }
        impl BitOr for $type {
            fn bit_or(&self, other: &Self) -> $type {
                self | other
            }
        }
        impl GetBit for $type {
            #[inline(always)]
            fn bit(&self, i: usize) -> bool {
                ((self >> i) & 1) == 1
            }
        }
        impl<I: HammingDistance + BitLen> BitMul<I, $type> for [(I, u32); $len] {
            fn bit_mul(&self, input: &I) -> $type {
                let mut target = <$type>::default();
                for i in 0..$len {
                    target |= ((self[i].0.hamming_distance(input) < self[i].1) as $type) << i;
                }
                target
            }
        }
        //impl<I: HammingDistance + BitLen> BitMul<I, $type> for [I; $len] {
        //    fn bit_mul(&self, input: &I) -> $type {
        //        let mut target = <$type>::default();
        //        for i in 0..$len {
        //            target |= ((self[i].hamming_distance(input) < (I::BIT_LEN as u32 / 2)) as $type) << i;
        //        }
        //        target
        //    }
        //}
        impl HammingDistance for $type {
            #[inline(always)]
            fn hamming_distance(&self, other: &$type) -> u32 {
                (self ^ other).count_ones()
            }
        }
    };
}

impl_for_uint!(u32, 32);
impl_for_uint!(u16, 16);
impl_for_uint!(u8, 8);

pub trait BitMul<I, O> {
    fn bit_mul(&self, input: &I) -> O;
}

impl<I, O: Default + Copy, T: BitMul<I, O>, const L: usize> BitMul<I, [O; L]> for [T; L]
where
    [O; L]: Default,
{
    fn bit_mul(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_mul(input);
        }
        target
    }
}
