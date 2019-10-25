use crate::shape::{Element, Shape};
use std::fmt;
use std::num::Wrapping;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};

pub trait IncrementHammingDistanceMatrix<T: BitArray>
where
    Self: BitArray,
    bool: Element<Self::BitShape>,
    u32: Element<Self::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<T::BitShape>,
    bool: Element<T::BitShape>,
    u32: Element<T::BitShape>,
{
    fn increment_hamming_distance_matrix(
        &self,
        counters_matrix: &mut <<u32 as Element<Self::BitShape>>::Array as Element<
            T::BitShape,
        >>::Array,
        target: &T,
    );
}

impl<I: IncrementHammingDistanceMatrix<T> + BitArray, T: BitArray, const L: usize>
    IncrementHammingDistanceMatrix<[T; L]> for I
where
    bool: Element<T::BitShape>,
    u32: Element<T::BitShape>,
    bool: Element<I::BitShape>,
    u32: Element<I::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<T::BitShape>,
    [T; L]: Default,
{
    fn increment_hamming_distance_matrix(
        &self,
        counters_matrix: &mut [<<u32 as Element<Self::BitShape>>::Array as Element<T::BitShape>>::Array;
                 L],
        target: &[T; L],
    ) {
        for w in 0..L {
            self.increment_hamming_distance_matrix(&mut counters_matrix[w], &target[w]);
        }
    }
}

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

// A collection of bits which has a shape.
pub trait BitArray
where
    Self::BitShape: Shape,
    u32: Element<Self::BitShape>,
    bool: Element<Self::BitShape>,
{
    type BitShape;
    fn bitpack(bools: &<bool as Element<Self::BitShape>>::Array) -> Self;
    fn increment_counters(&self, counters: &mut <u32 as Element<Self::BitShape>>::Array);
    fn flipped_increment_counters(
        &self,
        sign: bool,
        counters: &mut <u32 as Element<Self::BitShape>>::Array,
    );
}

impl<T: BitArray, const L: usize> BitArray for [T; L]
where
    [T; L]: Default,
    T::BitShape: Shape,
    u32: Element<T::BitShape>,
    bool: Element<T::BitShape>,
{
    type BitShape = [T::BitShape; L];
    fn bitpack(bools: &<bool as Element<Self::BitShape>>::Array) -> Self {
        let mut target = Self::default();
        for i in 0..L {
            target[i] = T::bitpack(&bools[i]);
        }
        target
    }
    fn increment_counters(&self, counters: &mut <u32 as Element<Self::BitShape>>::Array) {
        for i in 0..L {
            self[i].increment_counters(&mut counters[i]);
        }
    }
    fn flipped_increment_counters(
        &self,
        sign: bool,
        counters: &mut <u32 as Element<Self::BitShape>>::Array,
    ) {
        for i in 0..L {
            self[i].flipped_increment_counters(sign, &mut counters[i]);
        }
    }
}
pub trait IncrementFracCounters
where
    Self: BitArray,
    bool: Element<Self::BitShape>,
    u32: Element<Self::BitShape>,
{
    fn increment_frac_counters(
        &self,
        counters: &mut (usize, <u32 as Element<Self::BitShape>>::Array),
    );
}

impl<B: BitArray> IncrementFracCounters for B
where
    bool: Element<Self::BitShape>,
    u32: Element<Self::BitShape>,
{
    fn increment_frac_counters(
        &self,
        counters: &mut (usize, <u32 as Element<B::BitShape>>::Array),
    ) {
        counters.0 += 1;
        self.increment_counters(&mut counters.1);
    }
}

// A collection of bits which does not need to have a shape
pub trait HammingDistance {
    fn hamming_distance(&self, rhs: &Self) -> u32;
}

impl<T: HammingDistance, const L: usize> HammingDistance for [T; L] {
    fn hamming_distance(&self, rhs: &Self) -> u32 {
        let mut sum = 0u32;
        for i in 0..L {
            sum += self[i].hamming_distance(&rhs[i]);
        }
        sum
    }
}

pub trait BitWord {
    const BIT_LEN: usize;
    fn splat(sign: bool) -> Self;
    fn bit(&self, i: usize) -> bool;
}

macro_rules! for_uints {
    ($b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone)]
        pub struct $b_type(pub $u_type);

        impl $b_type {
            pub fn count_ones(self) -> u32 {
                self.0.count_ones()
            }
        }

        impl BitWord for $b_type {
            const BIT_LEN: usize = $len;
            fn splat(sign: bool) -> Self {
                $b_type((Wrapping(0 as $u_type) - Wrapping(sign as $u_type)).0)
            }
            fn bit(&self, i: usize) -> bool {
                ((self.0 >> i) & 1) == 1
            }
        }
        impl BitArray for $b_type {
            type BitShape = [(); $len];
            fn bitpack(bools: &<bool as Element<Self::BitShape>>::Array) -> Self {
                let mut bits = <$u_type>::default();
                for b in 0..$len {
                    bits |= (bools[b] as $u_type) << b;
                }
                $b_type(bits)
            }
            fn increment_counters(&self, counters: &mut [u32; $len]) {
                for b in 0..$len {
                    counters[b] += ((self.0 >> b) & 1) as u32
                }
            }
            fn flipped_increment_counters(
                &self,
                sign: bool,
                counters: &mut <u32 as Element<Self::BitShape>>::Array,
            ) {
                let word = *self ^ Self::splat(sign);
                word.increment_counters(counters);
            }
        }
        impl HammingDistance for $b_type {
            fn hamming_distance(&self, rhs: &$b_type) -> u32 {
                (self.0 ^ rhs.0).count_ones()
            }
        }
        impl<I: HammingDistance> BitMul<I, $b_type> for [(I, u32); $len] {
            fn bit_mul(&self, input: &I) -> $b_type {
                let mut target = $b_type(0);
                for b in 0..$len {
                    target |=
                        $b_type(((self[b].0.hamming_distance(input) < self[b].1) as $u_type) << b);
                }
                target
            }
        }
        impl Default for $b_type {
            fn default() -> Self {
                $b_type(0)
            }
        }
        impl BitXor for $b_type {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self::Output {
                $b_type(self.0 ^ rhs.0)
            }
        }
        impl BitXorAssign for $b_type {
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }
        impl BitOr for $b_type {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                $b_type(self.0 | rhs.0)
            }
        }
        impl BitOrAssign for $b_type {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }
        impl BitAnd for $b_type {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self::Output {
                $b_type(self.0 & rhs.0)
            }
        }
        impl BitAndAssign for $b_type {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }
        impl fmt::Display for $b_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, $format_string, self.0)
            }
        }

        impl fmt::Debug for $b_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, $format_string, self.0)
            }
        }
        impl<I: BitArray> IncrementHammingDistanceMatrix<$b_type> for I
        where
            bool: Element<I::BitShape>,
            u32: Element<I::BitShape>,
        {
            fn increment_hamming_distance_matrix(
                &self,
                counters_matrix: &mut [<u32 as Element<Self::BitShape>>::Array; <$b_type>::BIT_LEN],
                target: &$b_type,
            ) {
                for i in 0..<$b_type>::BIT_LEN {
                    self.flipped_increment_counters(target.bit(i), &mut counters_matrix[i]);
                }
            }
        }
    };
}

for_uints!(b8, u8, 8, "{:08b}");
for_uints!(b16, u16, 16, "{:016b}");
for_uints!(b32, u32, 32, "{:032b}");
