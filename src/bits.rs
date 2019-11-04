/// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use crate::shape::{Element, Shape};
use std::fmt;
use std::num::Wrapping;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign};

/// Increment the elements of a matrix of counters to count the number of times that the bits are different.
pub trait IncrementHammingDistanceMatrix<T: BitArray>
where
    Self: BitArray,
    u32: Element<Self::BitShape>,
    <u32 as Element<Self::BitShape>>::Array: Element<T::BitShape>,
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

impl<
        I: IncrementHammingDistanceMatrix<T> + BitArray,
        T: BitArray + BitArrayOPs,
        const L: usize,
    > IncrementHammingDistanceMatrix<[T; L]> for I
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

pub trait Classify
where
    u32: Element<Self::ClassesShape>,
    Self::ClassesShape: Shape,
{
    type Input;
    const N_CLASSES: usize;
    type ClassesShape;
    fn activations(&self, input: &Self::Input) -> <u32 as Element<Self::ClassesShape>>::Array;
    fn max_class(&self, input: &Self::Input) -> usize;
}

impl<I: Distance, const C: usize> Classify for [(I, u32); C]
where
    [u32; C]: Default,
{
    type Input = I::Rhs;
    const N_CLASSES: usize = C;
    type ClassesShape = [(); C];
    fn activations(&self, input: &I::Rhs) -> [u32; C] {
        let mut target = <[u32; C]>::default();
        for c in 0..C {
            target[c] = self[c].0.distance(input) + self[c].1;
        }
        target
    }
    fn max_class(&self, input: &I::Rhs) -> usize {
        let mut max_act = 0u32;
        let mut max_class = 0usize;
        for c in 0..C {
            let act = self[c].0.distance(input) + self[c].1;
            if act >= max_act {
                max_act = act;
                max_class = c;
            }
        }
        max_class
    }
}

pub trait BitMul<O> {
    type Input;
    fn bit_mul(&self, input: &Self::Input) -> O;
}

impl<O: Default + Copy, T: BitMul<O>, const L: usize> BitMul<[O; L]> for [T; L]
where
    [O; L]: Default,
{
    type Input = T::Input;
    fn bit_mul(&self, input: &Self::Input) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_mul(input);
        }
        target
    }
}

/// A collection of bits which has a shape.
pub trait BitArray
where
    Self: Sized,
    Self::WordShape: Shape,
    Self::BitShape: Shape,
    Self::WordType: Element<Self::WordShape, Array = Self>,
{
    /// The shape of the bits.
    /// Note that this is not the shape of the array with word as elements,
    /// but rather the shape of the array with bits as elements.
    type BitShape;
    /// The type of the bitword inside the shape.
    type WordType;
    /// The shape where words are elements.
    type WordShape;
}

impl<T: BitArray, const L: usize> BitArray for [T; L] {
    type BitShape = [T::BitShape; L];
    type WordType = T::WordType;
    type WordShape = [T::WordShape; L];
}

/// A collection of bits which has a shape.
pub trait BitArrayOPs
where
    Self: Sized + BitArray,
    u32: Element<Self::BitShape>,
    bool: Element<Self::BitShape>,
{
    /// bitpacks some bools into a `Self` of the same BitShape.
    fn bitpack(bools: &<bool as Element<Self::BitShape>>::Array) -> Self;
    /// For each bit that is set, increment the corresponding counter.
    fn increment_counters(&self, counters: &mut <u32 as Element<Self::BitShape>>::Array);
    /// For each bit that is the value of `sign`, increment the corresponding counter.
    fn flipped_increment_counters(
        &self,
        sign: bool,
        counters: &mut <u32 as Element<Self::BitShape>>::Array,
    );
}

impl<T: BitArrayOPs + BitArray, const L: usize> BitArrayOPs for [T; L]
where
    T::BitShape: Shape,
    u32: Element<T::BitShape>,
    bool: Element<T::BitShape>,
    [T; L]: Default + BitArray<BitShape = [T::BitShape; L]>,
{
    fn bitpack(bools: &[<bool as Element<T::BitShape>>::Array; L]) -> Self {
        let mut target = Self::default();
        for i in 0..L {
            target[i] = T::bitpack(&bools[i]);
        }
        target
    }
    fn increment_counters(&self, counters: &mut [<u32 as Element<T::BitShape>>::Array; L]) {
        for i in 0..L {
            self[i].increment_counters(&mut counters[i]);
        }
    }
    fn flipped_increment_counters(
        &self,
        sign: bool,
        counters: &mut [<u32 as Element<T::BitShape>>::Array; L],
    ) {
        for i in 0..L {
            self[i].flipped_increment_counters(sign, &mut counters[i]);
        }
    }
}

pub trait IncrementFracCounters
where
    Self: BitArray,
    u32: Element<Self::BitShape>,
{
    fn increment_frac_counters(
        &self,
        counters: &mut (usize, <u32 as Element<Self::BitShape>>::Array),
    );
}

impl<B: BitArray + BitArrayOPs> IncrementFracCounters for B
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

pub trait ArrayBitOr {
    fn bit_or(&self, rhs: &Self) -> Self;
}

impl<T: ArrayBitOr, const L: usize> ArrayBitOr for [T; L]
where
    [T; L]: Default,
{
    fn bit_or(&self, rhs: &Self) -> Self {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_or(&rhs[i]);
        }
        target
    }
}

/// Hamming distance betwene two collections of bits of the same shape.
pub trait Distance {
    type Rhs;
    /// Returns the number of bits that are different
    fn distance(&self, rhs: &Self::Rhs) -> u32;
}

impl<T: Distance, const L: usize> Distance for [T; L] {
    type Rhs = [T::Rhs; L];
    fn distance(&self, rhs: &Self::Rhs) -> u32 {
        let mut sum = 0u32;
        for i in 0..L {
            sum += self[i].distance(&rhs[i]);
        }
        sum
    }
}

/// A single word of bits.
pub trait BitWord {
    /// The number of bits in the word.
    const BIT_LEN: usize;
    /// Returns a word where all bits of set to the value of sign.
    fn splat(sign: bool) -> Self;
    /// Returns the value of the `i`th bit in `self`.
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
            type WordType = $b_type;
            type WordShape = ();
        }
        impl BitArrayOPs for $b_type {
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
        impl Distance for $b_type {
            type Rhs = $b_type;
            fn distance(&self, rhs: &Self::Rhs) -> u32 {
                (self.0 ^ rhs.0).count_ones()
            }
        }
        impl Distance for ($b_type, $b_type) {
            type Rhs = $b_type;
            fn distance(&self, &rhs: &Self::Rhs) -> u32 {
                ((self.0 ^ rhs) & self.1).count_ones()
            }
        }
        impl<I: Distance> BitMul<$b_type> for [(I, u32); $len] {
            type Input = I::Rhs;
            fn bit_mul(&self, input: &I::Rhs) -> $b_type {
                let mut target = $b_type(0);
                for b in 0..$len {
                    target |= $b_type(((self[b].0.distance(input) < self[b].1) as $u_type) << b);
                }
                target
            }
        }
        impl ArrayBitOr for $b_type {
            fn bit_or(&self, other: &$b_type) -> $b_type {
                *self | *other
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
        impl<I: BitArray + BitArrayOPs> IncrementHammingDistanceMatrix<$b_type> for I
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
for_uints!(b64, u64, 64, "{:064b}");
for_uints!(b128, u128, 128, "{:0128b}");
