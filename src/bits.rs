/// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use crate::layer::Apply;
use crate::shape::{Element, Shape, ZipMap};
use crate::unary::Preprocess;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::Wrapping;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

pub trait IndexedFlipBit<I, O> {
    fn indexed_flip_bit(&mut self, o: usize, i: usize);
}

impl<I, O, T: IndexedFlipBit<I, O>, const L: usize> IndexedFlipBit<I, [O; L]> for [T; L] {
    fn indexed_flip_bit(&mut self, o: usize, i: usize) {
        self[o % L].indexed_flip_bit(o / L, i)
    }
}

pub trait Classify<Example, Patch, ClassesShape>
where
    f32: Element<ClassesShape>,
    ClassesShape: Shape,
{
    fn activations(&self, input: &Example) -> <f32 as Element<ClassesShape>>::Array;
    fn max_class(&self, input: &Example) -> usize;
}

impl<I: BitArray + BFMA, const C: usize> Classify<I, (), [(); C]>
    for [(<f32 as Element<I::BitShape>>::Array, f32); C]
where
    [f32; C]: Default,
    f32: Element<I::BitShape>,
{
    fn activations(&self, example: &I) -> [f32; C] {
        let mut target = <[f32; C]>::default();
        for c in 0..C {
            target[c] = example.bfma(&self[c].0) + self[c].1;
        }
        target
    }
    fn max_class(&self, input: &I) -> usize {
        let activations = <Self as Classify<I, (), [(); C]>>::activations(self, input);
        let mut max_act = 0_f32;
        let mut max_class = 0_usize;
        for c in 0..C {
            if activations[c] >= max_act {
                max_act = activations[c];
                max_class = c;
            }
        }
        max_class
    }
}

//impl<T: BitMul<Preprocessor::Output, O>, I, O, Preprocessor: Preprocess<I>>
//    Apply<I, (), Preprocessor, O> for T
//{
//    fn apply(&self, input: &I) -> O {
//        self.bit_mul(&Preprocessor::preprocess(input))
//    }
//}

/// Bits input, bits matrix, and bits output
pub trait BitMul<I, O> {
    fn bit_mul(&self, input: &I) -> O;
}

impl<I, O, T: BitMul<I, O>, const L: usize> BitMul<I, [O; L]> for [T; L]
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

/// A collection of bits which has a shape.
pub trait BitArray
where
    Self: Sized,
    Self::WordShape: Shape,
    Self::BitShape: Shape,
    Self::WordType: Element<Self::WordShape, Array = Self>,
{
    /// The shape of the bits.
    /// Note that this is not the shape of the array with words as elements,
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

pub trait BitStates {
    const ONES: Self;
    const ZEROS: Self;
}

macro_rules! impl_bitstats_for_array {
    ($len:expr) => {
        impl<T: BitStates> BitStates for [T; $len] {
            const ONES: Self = [T::ONES; $len];
            const ZEROS: Self = [T::ZEROS; $len];
        }
    };
}

impl_bitstats_for_array!(1);
impl_bitstats_for_array!(2);
impl_bitstats_for_array!(3);
impl_bitstats_for_array!(4);

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
    fn weighted_increment_counters(
        &self,
        weight: u32,
        counters: &mut <u32 as Element<Self::BitShape>>::Array,
    );
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
    fn weighted_increment_counters(
        &self,
        weight: u32,
        counters: &mut <u32 as Element<Self::BitShape>>::Array,
    ) {
        for i in 0..L {
            self[i].weighted_increment_counters(weight, &mut counters[i]);
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
    fn bitpack_fracs(
        a: &(usize, <u32 as Element<Self::BitShape>>::Array),
        b: &(usize, <u32 as Element<Self::BitShape>>::Array),
    ) -> Self;
    fn increment_frac_counters(
        &self,
        counters: &mut (usize, <u32 as Element<Self::BitShape>>::Array),
    );
    fn weighted_increment_frac_counters(
        &self,
        weight: u32,
        counters: &mut (usize, <u32 as Element<Self::BitShape>>::Array),
    );
}

impl<B: BitArray + BitArrayOPs> IncrementFracCounters for B
where
    bool: Element<Self::BitShape>,
    u32: Element<Self::BitShape>,
    Self::BitShape: ZipMap<u32, u32, bool>,
{
    fn bitpack_fracs(
        a: &(usize, <u32 as Element<Self::BitShape>>::Array),
        b: &(usize, <u32 as Element<Self::BitShape>>::Array),
    ) -> Self {
        let ac = a.0 as u64;
        let bc = b.0 as u64;
        let diffs = <<Self as BitArray>::BitShape as ZipMap<u32, u32, bool>>::zip_map(
            &a.1,
            &b.1,
            |&a, &b| (a as u64 * bc) > (b as u64 * ac),
        );
        Self::bitpack(&diffs)
    }
    fn increment_frac_counters(
        &self,
        counters: &mut (usize, <u32 as Element<B::BitShape>>::Array),
    ) {
        counters.0 += 1;
        self.increment_counters(&mut counters.1);
    }
    fn weighted_increment_frac_counters(
        &self,
        weight: u32,
        counters: &mut (usize, <u32 as Element<B::BitShape>>::Array),
    ) {
        counters.0 += weight as usize;
        self.weighted_increment_counters(weight, &mut counters.1);
    }
}

pub trait ArrayBitAnd {
    fn bit_and(&self, rhs: &Self) -> Self;
}

impl<T: ArrayBitAnd, const L: usize> ArrayBitAnd for [T; L]
where
    [T; L]: Default,
{
    fn bit_and(&self, rhs: &Self) -> Self {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_and(&rhs[i]);
        }
        target
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

impl<T: BFBVM<Preprocessor::Output, O>, I, O, Preprocessor: Preprocess<I>>
    Apply<I, (), Preprocessor, O> for T
{
    fn apply(&self, input: &I) -> O {
        self.bfbvm(&Preprocessor::preprocess(input))
    }
}

/// Bit Float Bit Vector Multiply
/// Takes bits input, float matrix, and returns bit array output.
pub trait BFBVM<I, O> {
    fn bfbvm(&self, input: &I) -> O;
}

impl<I, T: BFBVM<I, O>, O, const L: usize> BFBVM<I, [O; L]> for [T; L]
where
    [O; L]: Default,
{
    fn bfbvm(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].bfbvm(input);
        }
        target
    }
}

/// Bit Float Multiply Accumulate
pub trait BFMA
where
    Self: BitArray,
    f32: Element<Self::BitShape>,
{
    fn bfma(&self, weights: &<f32 as Element<Self::BitShape>>::Array) -> f32;
}

impl<T: BitArray + BFMA, const L: usize> BFMA for [T; L]
where
    f32: Element<T::BitShape>,
{
    fn bfma(&self, weights: &[<f32 as Element<T::BitShape>>::Array; L]) -> f32 {
        let mut sum = 0f32;
        for i in 0..L {
            sum += self[i].bfma(&weights[i]);
        }
        sum
    }
}

/// Hamming distance between two collections of bits of the same shape.
pub trait Distance {
    /// Returns the number of bits that are different
    fn distance(&self, rhs: &Self) -> u32;
}

impl Distance for usize {
    fn distance(&self, rhs: &usize) -> u32 {
        (self ^ rhs).count_ones()
    }
}

impl<T: Distance, const L: usize> Distance for [T; L] {
    fn distance(&self, rhs: &Self) -> u32 {
        let mut sum = 0_u32;
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
    fn flip_bit(&mut self, i: usize);
}

impl<T: BitWord, const L: usize> BitWord for [T; L]
where
    Self: Default,
{
    const BIT_LEN: usize = T::BIT_LEN * L;
    fn splat(sign: bool) -> Self {
        let mut target = Self::default();
        for i in 0..L {
            target[i] = T::splat(sign);
        }
        target
    }
    fn bit(&self, i: usize) -> bool {
        self[i / T::BIT_LEN].bit(i % T::BIT_LEN)
    }
    fn flip_bit(&mut self, i: usize) {
        self[i / T::BIT_LEN].flip_bit(i % T::BIT_LEN)
    }
}

macro_rules! for_uints {
    ($b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
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
            fn flip_bit(&mut self, i: usize) {
                *self ^= $b_type(1 as $u_type) << i;
            }
        }
        impl BitArray for $b_type {
            type BitShape = [(); $len];
            type WordType = $b_type;
            type WordShape = ();
        }
        impl BitStates for $b_type {
            const ONES: $b_type = $b_type(!0);
            const ZEROS: $b_type = $b_type(0);
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
                    counters[b] += ((self.0 >> b) & 1) as u32;
                }
            }
            fn weighted_increment_counters(&self, weight: u32, counters: &mut [u32; $len]) {
                for b in 0..$len {
                    counters[b] += ((self.0 >> b) & 1) as u32 * weight;
                }
            }
            fn flipped_increment_counters(&self, sign: bool, counters: &mut [u32; $len]) {
                let word = *self ^ Self::splat(sign);
                word.increment_counters(counters);
            }
        }
        impl<I: BitWord> IndexedFlipBit<I, $b_type> for [I; $len] {
            fn indexed_flip_bit(&mut self, o: usize, i: usize) {
                self[o].flip_bit(i);
            }
        }
        impl Distance for $b_type {
            fn distance(&self, rhs: &Self) -> u32 {
                (self.0 ^ rhs.0).count_ones()
            }
        }
        impl BFMA for $b_type {
            fn bfma(&self, weights: &[f32; $len]) -> f32 {
                const SIGNS: [f32; 2] = [1f32, -1f32];
                let mut sum = 0f32;
                for b in 0..$len {
                    sum = weights[b].mul_add(SIGNS[self.bit(b) as usize], sum);
                }
                sum
            }
        }
        impl<I: BitArray + BFMA> BFBVM<I, $b_type>
            for [(<f32 as Element<I::BitShape>>::Array, f32); $len]
        where
            f32: Element<I::BitShape>,
        {
            fn bfbvm(&self, input: &I) -> $b_type {
                let mut target = $b_type(0);
                for b in 0..$len {
                    target |= $b_type((input.bfma(&self[b].0) + self[b].1 > 0f32) as $u_type) << b;
                }
                target
            }
        }
        impl<I: Distance + BitWord> BitMul<I, $b_type> for [I; $len] {
            fn bit_mul(&self, input: &I) -> $b_type {
                let mut target = $b_type(0);
                for b in 0..$len {
                    target |= $b_type(
                        ((self[b].distance(input) < (I::BIT_LEN as u32 / 2)) as $u_type) << b,
                    );
                }
                target
            }
        }
        impl Distribution<$b_type> for Standard {
            #[inline]
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $b_type {
                $b_type(rng.gen())
            }
        }
        impl Not for $b_type {
            type Output = $b_type;

            fn not(self) -> $b_type {
                $b_type(!(self.0))
            }
        }

        impl ArrayBitOr for $b_type {
            fn bit_or(&self, other: &$b_type) -> $b_type {
                *self | *other
            }
        }
        impl ArrayBitAnd for $b_type {
            fn bit_and(&self, other: &$b_type) -> $b_type {
                *self & *other
            }
        }
        impl Default for $b_type {
            fn default() -> Self {
                $b_type(0)
            }
        }
        impl PartialEq for $b_type {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl Eq for $b_type {}
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
        impl Shl<usize> for $b_type {
            type Output = Self;

            fn shl(self, rhs: usize) -> $b_type {
                $b_type(self.0 << rhs)
            }
        }
        impl Shr<usize> for $b_type {
            type Output = Self;

            fn shr(self, rhs: usize) -> $b_type {
                $b_type(self.0 >> rhs)
            }
        }
        impl Hash for $b_type {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.0.hash(state);
            }
        }
    };
}

for_uints!(b8, u8, 8, "{:08b}");
for_uints!(b16, u16, 16, "{:016b}");
for_uints!(b32, u32, 32, "{:032b}");
//for_uints!(b64, u64, 64, "{:064b}");
//for_uints!(b128, u128, 128, "{:0128b}");

pub trait AndOr {
    type Val;
    const IDENTITY: Self;
    fn andor(&self, val: &Self::Val) -> Self;
}

impl<T: BitArray + BitStates + ArrayBitAnd + ArrayBitOr> AndOr for [T; 2] {
    type Val = T;
    const IDENTITY: Self = [T::ONES, T::ZEROS];
    fn andor(&self, val: &Self::Val) -> Self {
        [self[0].bit_and(val), self[1].bit_or(val)]
    }
}
