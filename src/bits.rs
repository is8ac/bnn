// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use crate::shape::{Element, Shape, ZipMap};
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

pub trait Classify<Input, C: Shape> {
    fn max_class(&self, input: &Input) -> usize;
}

//impl<T: BitMul<I, O>, I, O> Apply<I, O> for <I as Element<O>>::Array {
//    fn apply(&self, input: &I) -> O {
//        self.bit_mul(&input)
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
    Self::TritArrayType: TritArray<TritShape = Self::BitShape, BitArrayType = Self>,
{
    /// The shape of the bits.
    /// Note that this is not the shape of the array with words as elements,
    /// but rather the shape of the array with bits as elements.
    type BitShape;
    /// The type of the bitword inside the shape.
    type WordType;
    /// The shape where words are elements.
    type WordShape;
    type TritArrayType;
}

impl<T: BitArray, const L: usize> BitArray for [T; L] {
    type BitShape = [T::BitShape; L];
    type WordType = T::WordType;
    type WordShape = [T::WordShape; L];
    type TritArrayType = [T::TritArrayType; L];
}

pub trait BitMapPack<E: Element<Self::BitShape>>
where
    Self: BitArray,
{
    fn bit_map_pack<F: Fn(&E) -> bool>(
        input: &<E as Element<Self::BitShape>>::Array,
        map_fn: F,
    ) -> Self;
}

impl<E: Element<T::BitShape>, T: BitArray + BitMapPack<E>, const L: usize> BitMapPack<E> for [T; L]
where
    [T; L]: Default,
{
    fn bit_map_pack<F: Fn(&E) -> bool>(
        input: &[<E as Element<T::BitShape>>::Array; L],
        map_fn: F,
    ) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::bit_map_pack(&input[i], &map_fn);
        }
        target
    }
}

pub trait BitMap<E: Element<Self::BitShape>>
where
    Self: BitArray,
{
    fn bit_map<F: Fn(bool) -> E>(&self, map_fn: F) -> <E as Element<Self::BitShape>>::Array;
    fn bit_map_mut<F: Fn(&mut E, bool)>(
        &self,
        target: &mut <E as Element<Self::BitShape>>::Array,
        map_fn: F,
    );
}

impl<T: BitMap<E>, E: Element<T::BitShape>, const L: usize> BitMap<E> for [T; L]
where
    [<E as Element<T::BitShape>>::Array; L]: Default,
{
    fn bit_map<F: Fn(bool) -> E>(&self, map_fn: F) -> [<E as Element<T::BitShape>>::Array; L] {
        let mut target = <[<E as Element<T::BitShape>>::Array; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_map(&map_fn);
        }
        target
    }
    fn bit_map_mut<F: Fn(&mut E, bool)>(
        &self,
        target: &mut [<E as Element<T::BitShape>>::Array; L],
        map_fn: F,
    ) {
        for i in 0..L {
            self[i].bit_map_mut(&mut target[i], &map_fn);
        }
    }
}

pub trait TritPack
where
    Self: TritArray,
    Option<bool>: Element<Self::TritShape>,
{
    fn trit_pack(trits: &<Option<bool> as Element<Self::TritShape>>::Array) -> Self;
    fn trit_expand(&self) -> <Option<bool> as Element<Self::TritShape>>::Array;
}

impl<T: TritPack, const L: usize> TritPack for [T; L]
where
    Option<bool>: Element<T::TritShape>,
    [T; L]: Default,
    <Option<bool> as Element<Self::TritShape>>::Array: Default,
{
    fn trit_pack(trits: &[<Option<bool> as Element<T::TritShape>>::Array; L]) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::trit_pack(&trits[i]);
        }
        target
    }
    fn trit_expand(&self) -> <Option<bool> as Element<Self::TritShape>>::Array {
        let mut target = <Option<bool> as Element<Self::TritShape>>::Array::default();
        for i in 0..L {
            target[i] = self[i].trit_expand();
        }
        target
    }
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

pub trait BitZipMap<E: Element<Self::BitShape>, O: Element<Self::BitShape>>
where
    Self: BitArray,
{
    fn bit_zip_map<F: Fn(bool, E) -> O>(
        &self,
        vals: &<E as Element<Self::BitShape>>::Array,
        map_fn: F,
    ) -> <O as Element<Self::BitShape>>::Array;
    fn bit_zip_map_mut<F: Fn(&mut O, bool, E)>(
        &self,
        target: &mut <O as Element<Self::BitShape>>::Array,
        vals: &<E as Element<Self::BitShape>>::Array,
        map_fn: F,
    );
}

impl<
        T: BitArray + BitZipMap<E, O>,
        E: Element<T::BitShape>,
        O: Element<T::BitShape>,
        const L: usize,
    > BitZipMap<E, O> for [T; L]
where
    [<O as Element<T::BitShape>>::Array; L]: Default,
{
    fn bit_zip_map<F: Fn(bool, E) -> O>(
        &self,
        vals: &[<E as Element<T::BitShape>>::Array; L],
        map_fn: F,
    ) -> [<O as Element<T::BitShape>>::Array; L] {
        let mut target = <[<O as Element<T::BitShape>>::Array; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_zip_map(&vals[i], &map_fn);
        }
        target
    }
    fn bit_zip_map_mut<F: Fn(&mut O, bool, E)>(
        &self,
        target: &mut [<O as Element<T::BitShape>>::Array; L],
        vals: &[<E as Element<T::BitShape>>::Array; L],
        map_fn: F,
    ) {
        for i in 0..L {
            self[i].bit_zip_map_mut(&mut target[i], &vals[i], &map_fn);
        }
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

/// Bit Float Bit Vector Multiply
/// Takes bits input, float matrix, and returns bit array output.
pub trait FFBVM<I, O> {
    fn ffbvm(&self, input: &I) -> O;
}

impl<I, T: FFBVM<I, O>, O, const L: usize> FFBVM<I, [O; L]> for [T; L]
where
    [O; L]: Default,
{
    fn ffbvm(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].ffbvm(input);
        }
        target
    }
}

/// Bit Float Bit Vector Multiply
/// Takes bits input, float matrix, and returns bit array output.
pub trait BFBVMM<I, O> {
    fn bfbvmm(&self, input: &I) -> O;
}

impl<I, T: BFBVMM<I, O>, O, const L: usize> BFBVMM<I, [O; L]> for [T; L]
where
    [O; L]: Default,
{
    fn bfbvmm(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].bfbvmm(input);
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

/// Masked Hamming distance between two collections of bits of the same shape.
pub trait MaskedDistance
where
    Self: TritArray,
{
    /// Returns the number of bits that are different and mask bit is set.
    /// Note that this is not adjusted for mask count. You should probably add mask_zeros() / 2 to the result.
    fn masked_distance(&self, bits: &Self::BitArrayType) -> u32;
}

impl<T: MaskedDistance, const L: usize> MaskedDistance for [T; L] {
    fn masked_distance(&self, bits: &[T::BitArrayType; L]) -> u32 {
        let mut sum = 0_u32;
        for i in 0..L {
            sum += self[i].masked_distance(&bits[i]);
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

pub trait TritArray
where
    Self: Sized,
    Self::TritShape: Shape,
    Self::BitArrayType: BitArray<BitShape = Self::TritShape, TritArrayType = Self>,
{
    const N: usize;
    type BitArrayType;
    type TritShape;
    fn mask_zeros(&self) -> u32;
    fn flip(&mut self, signs: &Self::BitArrayType);
    //fn trit_flip(&mut self, trits: &Self);
    fn get_trit(
        &self,
        index: &<<Self::BitArrayType as BitArray>::BitShape as Shape>::Index,
    ) -> Option<bool>;
}

impl<T: TritArray, const L: usize> TritArray for [T; L] {
    const N: usize = T::N * L;
    type BitArrayType = [T::BitArrayType; L];
    type TritShape = [T::TritShape; L];
    fn mask_zeros(&self) -> u32 {
        self.iter().map(|x| x.mask_zeros()).sum()
    }
    fn flip(&mut self, signs: &[T::BitArrayType; L]) {
        for i in 0..L {
            self[i].flip(&signs[i]);
        }
    }
    //fn trit_flip(&mut self, trits: &[T; L]) {
    //    for i in 0..L {
    //        self[i].trit_flip(&trits[i]);
    //    }
    //}
    fn get_trit(
        &self,
        (head, tail): &(
            usize,
            <<T::BitArrayType as BitArray>::BitShape as Shape>::Index,
        ),
    ) -> Option<bool> {
        self[*head].get_trit(tail)
    }
}

pub trait SetTrit
where
    Self: TritArray,
{
    fn set_trit(
        &self,
        trit: Option<bool>,
        index: &<<Self::BitArrayType as BitArray>::BitShape as Shape>::Index,
    ) -> Self;
}

impl<T: SetTrit + TritArray, const L: usize> SetTrit for [T; L]
where
    Self: Copy,
{
    fn set_trit(
        &self,
        trit: Option<bool>,
        (head, tail): &(
            usize,
            <<T::BitArrayType as BitArray>::BitShape as Shape>::Index,
        ),
    ) -> Self {
        let mut target = *self;
        target[*head] = self[*head].set_trit(trit, tail);
        target
    }
}

/// random signs, all ones mask.
trait RandInitTrit {
    fn rand_init() -> Self;
}

macro_rules! for_uints {
    ($t_type:ident, $b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        /// A word of trits. The 0 element is the signs, the 1 element is the magnitudes.
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $t_type(pub $u_type, pub $u_type);

        //impl std::ops::Add for $t_type {
        //    type Output = $t_type;
        //    fn add(self, other: Self) -> $t_type {
        //        let a0 = !self.0 & self.1;
        //        let a1 = self.0 & self.1;

        //        let b0 = !other.0 & other.1;
        //        let b1 = other.0 & other.1;

        //        let ones = (a1 & !b0) | (b1 & !a0);
        //        let zeros = (a0 & !b1) | (b0 & !a1);

        //        $t_type(ones, ones | zeros)
        //    }
        //}
        impl TritPack for $t_type {
            fn trit_pack(trits: &[Option<bool>; $len]) -> $t_type {
                let mut signs = <$u_type>::default();
                let mut mask = <$u_type>::default();
                for b in 0..$len {
                    signs |= (trits[b].unwrap_or(false) as $u_type) << b;
                    mask |= (trits[b].is_some() as $u_type) << b;
                }
                $t_type(signs, mask)
            }
            fn trit_expand(&self) -> [Option<bool>; $len] {
                let mut target = <[Option<bool>; $len]>::default();
                for b in 0..$len {
                    target[b] = if ((self.1 >> b) & 1) == 1 {
                        Some(((self.0 >> b) & 1) == 1)
                    } else {
                        None
                    };
                }
                target
            }
        }

        impl PartialEq for $t_type {
            fn eq(&self, other: &Self) -> bool {
                ((self.0 & self.1) == (other.0 & self.1)) & (self.1 == other.1)
            }
        }
        impl Eq for $t_type {}

        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $b_type(pub $u_type);

        impl TritArray for $t_type {
            const N: usize = $len;
            type BitArrayType = $b_type;
            type TritShape = [(); $len];
            fn mask_zeros(&self) -> u32 {
                self.1.count_zeros()
            }
            fn flip(&mut self, &$b_type(grads): &$b_type) {
                let signs = self.0;
                let magns = self.1;
                self.0 = grads;
                self.1 = (magns & !(grads ^ signs)) | !magns;
            }
            //fn trit_flip(&mut self, &trits: &$t_type) {
            //    *self = self.add(trits);
            //}
            fn get_trit(&self, &(index, _): &(usize, ())) -> Option<bool> {
                let sign = (self.0 >> index) & 1 == 1;
                let magn = (self.1 >> index) & 1 == 1;
                Some(sign).filter(|_| magn)
            }
        }

        impl SetTrit for $t_type {
            fn set_trit(&self, trit: Option<bool>, &(index, _): &(usize, ())) -> Self {
                let signs =
                    (self.0 & !(1 << index)) | ((trit.unwrap_or(false) as $u_type) << index);
                let magns = (self.1 & !(1 << index)) | ((trit.is_some() as $u_type) << index);
                $t_type(signs, magns)
            }
        }

        impl $b_type {
            pub fn count_ones(self) -> u32 {
                self.0.count_ones()
            }
        }
        impl<E: Copy, O: Copy + Default> BitZipMap<E, O> for $b_type {
            fn bit_zip_map<F: Fn(bool, E) -> O>(&self, vals: &[E; $len], map_fn: F) -> [O; $len] {
                let mut target = [O::default(); $len];
                for b in 0..$len {
                    target[b] = map_fn(self.bit(b), vals[b]);
                }
                target
            }
            fn bit_zip_map_mut<F: Fn(&mut O, bool, E)>(
                &self,
                target: &mut [O; $len],
                vals: &[E; $len],
                map_fn: F,
            ) {
                for b in 0..$len {
                    map_fn(&mut target[b], self.bit(b), vals[b]);
                }
            }
        }
        impl<E> BitMapPack<E> for $b_type {
            fn bit_map_pack<F: Fn(&E) -> bool>(input: &[E; $len], map_fn: F) -> $b_type {
                let mut target = <$u_type>::default();
                for b in 0..$len {
                    target |= (map_fn(&input[b]) as $u_type) << b;
                }
                $b_type(target)
            }
        }
        impl<E> BitMap<E> for $b_type
        where
            [E; $len]: Default,
        {
            fn bit_map<F: Fn(bool) -> E>(&self, map_fn: F) -> [E; $len] {
                let mut target = <[E; $len]>::default();
                for b in 0..$len {
                    target[b] = map_fn(self.bit(b));
                }
                target
            }
            fn bit_map_mut<F: Fn(&mut E, bool)>(&self, target: &mut [E; $len], map_fn: F) {
                for b in 0..$len {
                    map_fn(&mut target[b], self.bit(b));
                }
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
            type TritArrayType = $t_type;
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
        impl MaskedDistance for $t_type {
            fn masked_distance(&self, bits: &$b_type) -> u32 {
                ((bits.0 ^ self.0) & self.1).count_ones()
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
        impl<I: BitArray + BFMA> BFBVMM<I, $b_type>
            for [(<f32 as Element<I::BitShape>>::Array, f32); $len]
        where
            f32: Element<I::BitShape>,
        {
            fn bfbvmm(&self, input: &I) -> $b_type {
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
        impl Distribution<$t_type> for Standard {
            #[inline]
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $t_type {
                //$t_type(rng.gen(), rng.gen())
                $t_type(rng.gen(), !0)
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
        impl Default for $t_type {
            fn default() -> Self {
                $t_type(0, 0)
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
        impl fmt::Display for $t_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, $format_string, self.0)?;
                write!(f, $format_string, self.1)
            }
        }
        impl fmt::Debug for $t_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, $format_string, self.0)?;
                write!(f, $format_string, self.1)
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

for_uints!(t8, b8, u8, 8, "{:08b}");
for_uints!(t16, b16, u16, 16, "{:016b}");
for_uints!(t32, b32, u32, 32, "{:032b}");
//for_uints!(b64, u64, 64, "{:064b}");
//for_uints!(b128, u128, 128, "{:0128b}");

#[cfg(test)]
mod tests {
    use crate::bits::{b16, b8, t16, t8, BitWord, TritArray, TritPack};
    use std::ops::Add;

    fn unpack_t16(trits: t16) -> [Option<bool>; 16] {
        let mut options = <[Option<bool>; 16]>::default();
        for b in 0..16 {
            options[b] = if b16(trits.1).bit(b) {
                Some(b16(trits.0).bit(b))
            } else {
                None
            }
        }
        options
    }
    fn unpack_b16(bits: b16) -> [bool; 16] {
        let mut bools = <[bool; 16]>::default();
        for b in 0..16 {
            bools[b] = bits.bit(b);
        }
        bools
    }

    fn trit_bit_update(weight: Option<bool>, grad: bool) -> Option<bool> {
        if let Some(weight_sign) = weight {
            if weight_sign ^ grad {
                None
            } else {
                Some(grad)
            }
        } else {
            Some(grad)
        }
    }
    fn trit_to_i8(trit: Option<bool>) -> i8 {
        if let Some(sign) = trit {
            if sign {
                -1
            } else {
                1
            }
        } else {
            0
        }
    }

    fn u16_shift(shift: usize) -> u16 {
        let mut target = 0u16;
        for b in 0..16 {
            target |= (((b >> shift) as u16) & 1u16) << b;
        }
        target
    }

    #[test]
    fn bit_flip_test() {
        let signs = b16(u16_shift(0));
        let mut trits = t16(u16_shift(1), u16_shift(2));
        let before = unpack_t16(trits);
        trits.flip(&signs);
        for ((&t, &b), &o) in before
            .iter()
            .zip(unpack_b16(signs).iter())
            .zip(unpack_t16(trits).iter())
        {
            assert_eq!(trit_bit_update(t, b), o);
        }
    }

    fn trit_trit_add_i8(weight: Option<bool>, grad: Option<bool>) -> Option<bool> {
        let sum = (trit_to_i8(weight) + trit_to_i8(grad)).max(-1).min(1);
        match sum {
            -1 => Some(true),
            0 => None,
            1 => Some(false),
            _ => panic!(),
        }
    }

    fn trit_trit_add_option(weight: Option<bool>, grad: Option<bool>) -> Option<bool> {
        if let Some(grad_sign) = grad {
            trit_bit_update(weight, grad_sign)
        } else {
            weight
        }
    }

    #[test]
    fn trit_eq_test() {
        assert_eq!(t8(0b_0_u8, 0b_1_u8), t8(0b_0_u8, 0b_1_u8));
        assert_eq!(t8(0b_0_u8, 0b_0_u8), t8(0b_0_u8, 0b_0_u8));
        assert_eq!(t8(0b_1_u8, 0b_1_u8), t8(0b_1_u8, 0b_1_u8));
        assert_eq!(t8(0b_1_u8, 0b_0_u8), t8(0b_1_u8, 0b_0_u8));

        assert_eq!(t8(0b_1_u8, 0b_0_u8), t8(0b_0_u8, 0b_0_u8));
        assert_eq!(t8(0b_0_u8, 0b_0_u8), t8(0b_1_u8, 0b_0_u8));

        assert_ne!(t8(0b_1_u8, 0b_1_u8), t8(0b_0_u8, 0b_1_u8));

        assert_ne!(t8(0b_1_u8, 0b_0_u8), t8(0b_1_u8, 0b_1_u8));
        assert_ne!(t8(0b_1_u8, 0b_0_u8), t8(0b_0_u8, 0b_1_u8));
    }

    #[test]
    fn trit_flip_test() {
        let grads = t16(u16_shift(0), u16_shift(1));
        let trits = t16(u16_shift(2), u16_shift(3));
        for (i, ((&t, &b), &o)) in unpack_t16(trits)
            .iter()
            .zip(unpack_t16(grads).iter())
            .zip(unpack_t16(trits + grads).iter())
            .enumerate()
        {
            dbg!(i);
            let add_result_option = trit_trit_add_option(t, b);
            let add_result_i8 = trit_trit_add_i8(t, b);
            assert_eq!(add_result_i8, add_result_option);
            if add_result_option != o {
                println!("{:?} + {:?} = {:?} ! {:?}", t, b, add_result_option, o);
            }
            assert_eq!(add_result_option, o);
        }
    }
    #[test]
    fn gen_mod_u16_test() {
        assert_eq!(b16(u16_shift(2)), b16(0b_1111_0000_1111_0000_u16));
        assert_eq!(b16(u16_shift(3)), b16(0b_1111_1111_0000_0000_u16));
    }
    #[test]
    fn trit_expand() {
        let trits = t16(u16_shift(0), u16_shift(1));
        let option_bools = trits.trit_expand();
        assert_eq!(<t16>::trit_pack(&option_bools), trits);
    }
}
