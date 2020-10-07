// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use crate::shape::{Element, IndexMap, Shape};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::num::Wrapping;
use std::ops;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

pub trait Weight<B: BitArray>
where
    //Self::Weights: WeightArray<Self>,
    Self: Sized,
{
    type Weights;
}

impl<B: BitArray> Weight<B> for bool {
    type Weights = B;
}

impl<B: BitArray> Weight<B> for Option<bool>
//where
//    B::TritArrayType: WeightArray<Option<bool>>,
{
    type Weights = B::TritArrayType;
}

pub trait WeightArray<W>
where
    W: Copy,
    Self: Sized,
    Self::Bits: BitArray + BitPack,
    bool: Element<<Self::Bits as BitArray>::BitShape>,
    <Self::Bits as BitArray>::BitShape: IndexMap<bool, ()> + Shape,
    <<Self::Bits as BitArray>::BitShape as Element<()>>::Array:
        Shape<Index = <<Self::Bits as BitArray>::BitShape as Shape>::Index>,
    Self::Bits: Copy,
{
    type Bits;
    type Index;
    const RANGE: usize;
    fn bma(&self, input: &Self::Bits) -> u32;
    fn mutate(&self, value: W, index: <<Self::Bits as BitArray>::BitShape as Shape>::Index)
        -> Self;
    // which indices, when set to value, would the act match target?
    fn grads(&self, input: &Self::Bits, value: W, cur_act: u32, target_act: u32) -> Self::Bits;
    /// grad_slow does the same thing as grads, but it is a lot slower.
    fn grads_slow(
        &self,
        input: &Self::Bits,
        value: W,
        cur_act: u32,
        target_act: u32,
    ) -> Self::Bits {
        Self::Bits::bit_pack(&<<Self::Bits as BitArray>::BitShape as IndexMap<
            bool,
            (),
        >>::index_map((), |index| {
            self.mutate(value, index).bma(input) > target_act
        }))
    }
    fn states() -> Vec<W>;
}

//impl<T: TritArray + MaskedDistance + Copy> WeightArray<Option<bool>> for (T, u32) {
//    type Bits = T::BitArrayType;
//    type Index = <<T::BitArrayType as BitArray>::BitShape as Shape>::Index;
//    const RANGE: usize = 2;
//    fn bma(&self, input: &T::BitArrayType) -> u32 {
//        self.0.masked_distance(input) * 2 + self.1
//    }
//    fn mutate(&self, index: <<Self::Bits as BitArray>::BitShape as Shape>::Index, value: Option<bool>) -> Self {
//        let new_trits = self.0.set_trit(value, index);
//        (new_trits, new_trits.mask_zeros())
//    }
//fn states() -> Vec<Option<bool>> {
//    vec![Some(false), None, Some(true)]
//}
//}

impl<B: BitPack + BitArray + Distance + Copy> WeightArray<bool> for B
where
    B::BitShape: IndexMap<bool, ()> + Shape,
    <B::BitShape as Element<()>>::Array: Shape<Index = <B::BitShape as Shape>::Index>,
    bool: Element<B::BitShape>,
{
    type Bits = B;
    type Index = <<Self::Bits as BitArray>::BitShape as Shape>::Index;
    const RANGE: usize = 2;
    fn bma(&self, input: &B) -> u32 {
        self.distance(input) * 2
    }
    fn mutate(
        &self,
        value: bool,
        index: <<Self::Bits as BitArray>::BitShape as Shape>::Index,
    ) -> Self {
        self.set_bit(value, index)
    }
    fn grads(&self, input: &Self::Bits, value: bool, cur_act: u32, target_act: u32) -> Self::Bits {
        *input
    }
    fn states() -> Vec<bool> {
        vec![true, false]
    }
}

//impl<W, T: WeightArray<W>> WeightArray<W> for (T, u32) {
//    type Input = T::Input;
//    fn bma(&self, input: &T::Input) -> u32 {
//        self.0.bma(input) + self.1
//    }
//}

pub trait Classify<Input, C: Shape> {
    fn max_class(&self, input: &Input) -> usize;
}

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
    //Self::WordShape: Shape,
    Self::BitShape: Shape,
    //Self::WordType: Element<Self::WordShape, Array = Self>,
    Self::TritArrayType: TritArray<TritShape = Self::BitShape, BitArrayType = Self>,
{
    /// The shape of the bits.
    /// Note that this is not the shape of the array with words as elements,
    /// but rather the shape of the array with bits as elements.
    type BitShape;
    /// The type of the bitword inside the shape.
    //type WordType;
    /// The shape where words are elements.
    //type WordShape;
    type TritArrayType;
    fn get_bit(&self, i: <Self::BitShape as Shape>::Index) -> bool;
    fn set_bit_in_place(&mut self, bit: bool, i: <Self::BitShape as Shape>::Index);
    fn set_bit(mut self, bit: bool, i: <Self::BitShape as Shape>::Index) -> Self {
        self.set_bit_in_place(bit, i);
        self
    }
}

impl<T: BitArray, const L: usize> BitArray for [T; L] {
    type BitShape = [T::BitShape; L];
    //type WordType = T::WordType;
    //type WordShape = [T::WordShape; L];
    type TritArrayType = [T::TritArrayType; L];
    fn get_bit(&self, (i, tail): (u8, <T::BitShape as Shape>::Index)) -> bool {
        self[i as usize].get_bit(tail)
    }
    fn set_bit_in_place(&mut self, bit: bool, (i, tail): (u8, <T::BitShape as Shape>::Index)) {
        self[i as usize].set_bit_in_place(bit, tail)
    }
}

pub trait BitMap
where
    Self::WordType: ops::BitAnd<Output = Self::WordType>
        + ops::BitOr<Output = Self::WordType>
        + ops::BitXor<Output = Self::WordType>
        + ops::Not<Output = Self::WordType>
        + Copy,
{
    type WordType;
    fn map<F: Fn(Self::WordType, Self::WordType) -> Self::WordType>(
        &self,
        rhs: &Self,
        map_fn: F,
    ) -> Self;
    fn to_bit(sign: bool) -> Self::WordType;
}

impl<T: BitMap, const L: usize> BitMap for [T; L]
where
    [T; L]: Default,
{
    type WordType = T::WordType;
    fn map<F: Fn(Self::WordType, Self::WordType) -> Self::WordType>(
        &self,
        rhs: &Self,
        map_fn: F,
    ) -> Self {
        let mut target = <[T; L]>::default();
        target
    }
    fn to_bit(sign: bool) -> Self::WordType {
        T::to_bit(sign)
    }
}

pub trait TritMap
where
    Self::WordType: ops::BitAnd<Output = Self::WordType>
        + ops::BitOr<Output = Self::WordType>
        + ops::BitXor<Output = Self::WordType>
        + ops::Not<Output = Self::WordType>
        + Copy,
    Self: TritArray,
{
    type WordType;
    fn map<F: Fn(Self::WordType, Self::WordType, Self::WordType) -> Self::WordType>(
        &self,
        bits: &Self::BitArrayType,
        map_fn: F,
    ) -> Self::BitArrayType;
    fn to_bit(sign: bool) -> Self::WordType;
}

impl<T: TritArray + TritMap, const L: usize> TritMap for [T; L]
where
    [<T as TritArray>::BitArrayType; L]: Default,
{
    type WordType = T::WordType;
    fn map<F: Fn(Self::WordType, Self::WordType, Self::WordType) -> Self::WordType>(
        &self,
        bits: &Self::BitArrayType,
        map_fn: F,
    ) -> [<T as TritArray>::BitArrayType; L] {
        let mut target = <[<T as TritArray>::BitArrayType; L]>::default();
        for i in 0..L {
            target[i] = self[i].map(&bits[i], &map_fn);
        }
        target
    }
    fn to_bit(sign: bool) -> Self::WordType {
        T::to_bit(sign)
    }
}

pub trait BitMapPack<E: Element<Self::BitShape>>
where
    Self: BitArray,
{
    fn bit_map_pack<F: Fn(&E) -> bool>(
        input: &<E as Element<Self::BitShape>>::Array,
        map_fn: F,
    ) -> Self;
    fn bit_map_pack_mut<F: Fn(&E) -> bool>(
        &mut self,
        input: &<E as Element<Self::BitShape>>::Array,
        index: <Self::BitShape as Shape>::Index,
        map_fn: F,
    );
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
    fn bit_map_pack_mut<F: Fn(&E) -> bool>(
        &mut self,
        input: &[<E as Element<T::BitShape>>::Array; L],
        (index, tail): <Self::BitShape as Shape>::Index,
        map_fn: F,
    ) {
        self[index as usize].bit_map_pack_mut(&input[index as usize], tail, &map_fn);
    }
}

/*
pub trait BitMap<E: Element<Self::BitShape>>
where
    Self: BitArray,
{
    fn bit_map<F: Fn(bool) -> E>(&self, map_fn: F) -> <E as Element<Self::BitShape>>::Array;
    fn bit_map_mut<F: Fn(&mut E, bool)>(&self, target: &mut <E as Element<Self::BitShape>>::Array, map_fn: F);
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
    fn bit_map_mut<F: Fn(&mut E, bool)>(&self, target: &mut [<E as Element<T::BitShape>>::Array; L], map_fn: F) {
        for i in 0..L {
            self[i].bit_map_mut(&mut target[i], &map_fn);
        }
    }
}
*/

pub trait BitPack
where
    Self: BitArray,
    bool: Element<Self::BitShape>,
{
    fn bit_pack(trits: &<bool as Element<Self::BitShape>>::Array) -> Self;
    fn bit_expand(&self) -> <bool as Element<Self::BitShape>>::Array;
}

impl<T: BitPack, const L: usize> BitPack for [T; L]
where
    bool: Element<T::BitShape>,
    [T; L]: Default,
    <bool as Element<Self::BitShape>>::Array: Default,
{
    fn bit_pack(trits: &[<bool as Element<T::BitShape>>::Array; L]) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::bit_pack(&trits[i]);
        }
        target
    }
    fn bit_expand(&self) -> <bool as Element<Self::BitShape>>::Array {
        let mut target = <bool as Element<Self::BitShape>>::Array::default();
        for i in 0..L {
            target[i] = self[i].bit_expand();
        }
        target
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

pub trait TritArray
where
    Self: Sized,
    Self::TritShape: Shape,
    Self::BitArrayType: BitArray<BitShape = Self::TritShape, TritArrayType = Self> + Sized,
{
    type BitArrayType;
    type TritShape;
    fn mask_zeros(&self) -> u32;
    fn get_trit(
        &self,
        index: <<Self::BitArrayType as BitArray>::BitShape as Shape>::Index,
    ) -> Option<bool>;
    fn set_trit_in_place(
        &mut self,
        trit: Option<bool>,
        index: <<Self::BitArrayType as BitArray>::BitShape as Shape>::Index,
    );
    fn set_trit(
        mut self,
        trit: Option<bool>,
        index: <<Self::BitArrayType as BitArray>::BitShape as Shape>::Index,
    ) -> Self {
        self.set_trit_in_place(trit, index);
        self
    }
}

impl<T: TritArray, const L: usize> TritArray for [T; L] {
    type BitArrayType = [T::BitArrayType; L];
    type TritShape = [T::TritShape; L];
    fn mask_zeros(&self) -> u32 {
        self.iter().map(|x| x.mask_zeros()).sum()
    }
    fn get_trit(
        &self,
        (head, tail): (
            u8,
            <<T::BitArrayType as BitArray>::BitShape as Shape>::Index,
        ),
    ) -> Option<bool> {
        self[head as usize].get_trit(tail)
    }
    fn set_trit_in_place(
        &mut self,
        trit: Option<bool>,
        (head, tail): (
            u8,
            <<T::BitArrayType as BitArray>::BitShape as Shape>::Index,
        ),
    ) {
        self[head as usize].set_trit_in_place(trit, tail);
    }
}

pub trait TritGrads
where
    Self: TritArray,
{
    /// which trits have the potential to move the distance in the sign direction, and if so, which direction.
    fn grads(&self, input: &Self::BitArrayType, sign: bool) -> Self;
}

impl<T: TritGrads + TritArray, const L: usize> TritGrads for [T; L]
where
    Self: Default + TritArray<TritShape = [T::TritShape; L], BitArrayType = [T::BitArrayType; L]>,
{
    fn grads(&self, input: &Self::BitArrayType, sign: bool) -> [T; L] {
        let mut target = Self::default();
        for i in 0..L {
            target[i] = self[i].grads(&input[i], sign);
        }
        target
    }
}

macro_rules! for_uints {
    ($t_type:ident, $b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        /// A word of trits. The 0 element is the signs, the 1 element is the magnitudes.
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $t_type(pub $u_type, pub $u_type);

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

        impl BitPack for $b_type {
            fn bit_pack(trits: &[bool; $len]) -> $b_type {
                let mut signs = <$u_type>::default();
                for b in 0..$len {
                    signs |= (trits[b] as $u_type) << b;
                }
                $b_type(signs)
            }
            fn bit_expand(&self) -> [bool; $len] {
                let mut target = <[bool; $len]>::default();
                for b in 0..$len {
                    target[b] = ((self.0 >> b) & 1) == 1;
                }
                target
            }
        }

        impl TritGrads for $t_type {
            #[inline(always)]
            fn grads(&self, &input: &$b_type, sign: bool) -> Self {
                let sign = (Wrapping(0) - Wrapping(sign as $u_type)).0;
                let acts = self.0 ^ input.0;
                let mask = !((acts ^ sign) & self.1);

                $t_type(input.0 ^ sign, mask)
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
            type BitArrayType = $b_type;
            type TritShape = [(); $len];
            fn mask_zeros(&self) -> u32 {
                self.1.count_zeros()
            }
            fn get_trit(&self, (index, _): (u8, ())) -> Option<bool> {
                let sign = (self.0 >> index) & 1 == 1;
                let magn = (self.1 >> index) & 1 == 1;
                Some(sign).filter(|_| magn)
            }
            fn set_trit_in_place(&mut self, trit: Option<bool>, (index, _): (u8, ())) {
                self.0 &= !(1 << index);
                self.0 |= ((trit.unwrap_or(false) as $u_type) << index);

                self.1 &= !(1 << index);
                self.1 |= ((trit.is_some() as $u_type) << index);
            }
        }

        impl $b_type {
            pub fn count_ones(self) -> u32 {
                self.0.count_ones()
            }
            pub fn bit(self, index: usize) -> bool {
                ((self.0 >> index) & 1) == 1
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
            fn bit_map_pack_mut<F: Fn(&E) -> bool>(
                &mut self,
                input: &[E; $len],
                (index, ()): (u8, ()),
                map_fn: F,
            ) {
                self.0 &= !(1 << index);
                self.0 |= (map_fn(&input[index as usize]) as $u_type) << index;
            }
        }
        /*
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
        */

        impl BitMap for $b_type {
            type WordType = $u_type;
            fn map<F: Fn($u_type, $u_type) -> $u_type>(
                &self,
                &rhs: &$b_type,
                map_fn: F,
            ) -> $b_type {
                $b_type(map_fn(self.0, rhs.0))
            }
            fn to_bit(sign: bool) -> Self::WordType {
                (Wrapping(0) - Wrapping(sign as $u_type)).0
            }
        }

        impl TritMap for $t_type {
            type WordType = $u_type;
            fn map<F: Fn($u_type, $u_type, $u_type) -> $u_type>(
                &self,
                &rhs: &$b_type,
                map_fn: F,
            ) -> $b_type {
                $b_type(map_fn(self.0, self.1, rhs.0))
            }
            fn to_bit(sign: bool) -> Self::WordType {
                (Wrapping(0) - Wrapping(sign as $u_type)).0
            }
        }

        impl BitArray for $b_type {
            type BitShape = [(); $len];
            //type WordType = $b_type;
            //type WordShape = ();
            type TritArrayType = $t_type;
            fn get_bit(&self, (i, _): (u8, ())) -> bool {
                ((self.0 >> i) & 1) == 1
            }
            fn set_bit_in_place(&mut self, bit: bool, (i, _): (u8, ())) {
                self.0 &= !(1 << i);
                self.0 |= ((bit as $u_type) << i) as $u_type;
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
        impl Distribution<$b_type> for Standard {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $b_type {
                $b_type(rng.gen())
            }
        }
        impl Distribution<$t_type> for Standard {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $t_type {
                $t_type(rng.gen(), rng.gen())
                //$t_type(rng.gen(), !0)
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
//for_uints!(t64, b64, u64, 64, "{:064b}");
//for_uints!(t128, b128, u128, 128, "{:0128b}");

/// A word of 1 trits
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Serialize, Deserialize, Default, Debug, PartialEq, Eq)]
pub struct t1(u8);

impl t1 {
    pub fn new_from_option_bool(trit: Option<bool>) -> Self {
        if let Some(sign) = trit {
            t1(sign as u8 | (1u8 << 1))
        } else {
            t1(0)
        }
    }
}

impl TritArray for t1 {
    type BitArrayType = b1;
    type TritShape = [(); 1];
    fn mask_zeros(&self) -> u32 {
        (!((self.0 >> 1) & 1u8)) as u32
    }
    fn get_trit(&self, _: (u8, ())) -> Option<bool> {
        None
    }
    fn set_trit_in_place(&mut self, trit: Option<bool>, _: (u8, ())) {
        *self = Self::new_from_option_bool(trit)
    }
}

impl MaskedDistance for t1 {
    #[inline(always)]
    fn masked_distance(&self, &bits: &b1) -> u32 {
        (((bits.0 as u8 ^ self.0) & (self.0 >> 1)) & 1u8) as u32
    }
}

/// A word of 1 bits
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct b1(pub bool);

impl BitArray for b1 {
    type BitShape = [(); 1];
    type TritArrayType = t1;
    fn get_bit(&self, _: (u8, ())) -> bool {
        self.0
    }
    fn set_bit_in_place(&mut self, bit: bool, _: (u8, ())) {
        self.0 = bit;
    }
}
impl Distance for b1 {
    fn distance(&self, rhs: &Self) -> u32 {
        (self.0 ^ rhs.0) as u32
    }
}

impl fmt::Display for b1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:01b}", self.0 as u8)
    }
}
impl fmt::Debug for b1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:01b}", self.0 as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::{b32, WeightArray};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;
    use test::Bencher;

    type InputType = [b32; 2];

    #[test]
    fn rand_binary_act_grads() {
        let mut rng = Hc128Rng::seed_from_u64(0);
        (0..100).for_each(|_| {
            let weights: InputType = rng.gen();
            let input: InputType = rng.gen();
            let act = weights.bma(&input);
            dbg!(act);

            for target_act in act.saturating_sub(InputType::RANGE as u32)
                ..=act.saturating_add(InputType::RANGE as u32)
            {
                dbg!(target_act);
                for &val in InputType::states().iter() {
                    let grads = weights.grads(&input, val, act, target_act);
                    let true_grads = weights.grads_slow(&input, val, act, target_act);
                    assert_eq!(grads, true_grads);
                }
            }
        })
    }
}
