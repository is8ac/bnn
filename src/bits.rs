// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use crate::shape::{Element, IndexMap, LongDefault, Shape};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::marker::PhantomData;
use std::num::Wrapping;
use std::ops;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};
use std::slice;

pub trait PackedElement<S: Shape>
where
    Self::Array: PackedArray<Shape = S, Weight = Self>,
{
    type Array;
}

impl<W: PackedElement<S> + Weight, S: Shape, const L: usize> PackedElement<[S; L]> for W
where
    Self::Array: PackedArray<Shape = [S; L], Weight = Self>,
{
    type Array = [<W as PackedElement<S>>::Array; L];
}

pub trait PackedArray
where
    Self::Shape: Shape,
    Self::Weight: Weight,
    Self: Sized,
{
    type Weight;
    type Shape;
    fn get_weight(&self, index: <Self::Shape as Shape>::Index) -> Self::Weight;
    fn set_weight_in_place(&mut self, index: <Self::Shape as Shape>::Index, value: Self::Weight);
    fn set_weight(mut self, index: <Self::Shape as Shape>::Index, value: Self::Weight) -> Self {
        self.set_weight_in_place(index, value);
        self
    }
}

impl<T: PackedArray, const L: usize> PackedArray for [T; L]
where
    Self::Shape: Shape<Index = (u8, <T::Shape as Shape>::Index)>,
    Self::Weight: Weight,
{
    type Weight = T::Weight;
    type Shape = [T::Shape; L];
    fn get_weight(&self, (head, tail): (u8, <T::Shape as Shape>::Index)) -> Self::Weight {
        self[head as usize].get_weight(tail)
    }
    fn set_weight_in_place(
        &mut self,
        (head, tail): (u8, <T::Shape as Shape>::Index),
        value: Self::Weight,
    ) {
        self[head as usize].set_weight_in_place(tail, value)
    }
}

pub trait Weight
where
    Self: Sized + Copy,
{
    const N: usize;
    fn states() -> iter::Cloned<slice::Iter<'static, Self>>;
}

impl Weight for bool {
    const N: usize = 2;
    fn states() -> iter::Cloned<slice::Iter<'static, bool>> {
        [true, false].iter().cloned()
    }
}

impl Weight for Option<bool> {
    const N: usize = 3;
    fn states() -> iter::Cloned<slice::Iter<'static, Option<bool>>> {
        [Some(true), None, Some(false)].iter().cloned()
    }
}

impl Weight for (bool, bool) {
    const N: usize = 4;
    fn states() -> iter::Cloned<slice::Iter<'static, (bool, bool)>> {
        [(true, true), (true, false), (false, false), (false, true)]
            .iter()
            .cloned()
    }
}

pub trait WeightArray<W: 'static + Weight>
where
    Self: Sized + Copy,
    Self::Bits: BitMapPack<<<<Self::Bits as Activations>::Shape as Element<()>>::Array as Shape>::Index>
        + Activations,
    W: Weight + Copy,
    <<Self::Bits as Activations>::Shape as Shape>::Index:
        Element<<Self::Bits as Activations>::Shape> + Copy,
    <<Self::Bits as Activations>::Shape as Element<()>>::Array: Shape,
    <Self::Bits as Activations>::Shape: IndexMap<<<Self::Bits as Activations>::Shape as Shape>::Index, ()>
        + Shape<Index = <<<Self::Bits as Activations>::Shape as Element<()>>::Array as Shape>::Index>,
{
    type Bits;
    const RANGE: usize;
    const THRESHOLD: u32;
    /// multiply accumulate
    fn bma(&self, input: &Self::Bits) -> u32;
    /// thresholded activation
    fn act(&self, input: &Self::Bits) -> bool {
        self.bma(input) > Self::THRESHOLD
    }
    fn mutate(self, value: W, index: <<Self::Bits as Activations>::Shape as Shape>::Index) -> Self;
    /// losses if we were to set each of the elements to each different value. Is permited to prune null acts.
    fn losses<F: Fn(u32) -> u64>(
        &self,
        input: &Self::Bits,
        loss_fn: F,
    ) -> Vec<(<<Self::Bits as Activations>::Shape as Shape>::Index, W, u64)>;
    /// Does the same thing as losses but is a lot slower.
    fn losses_slow<F: Fn(u32) -> u64>(
        &self,
        input: &Self::Bits,
        loss_fn: F,
    ) -> Vec<(<<Self::Bits as Activations>::Shape as Shape>::Index, W, u64)> {
        let null_loss = loss_fn(self.bma(input));
        <W as Weight>::states()
            .map(|value| {
                <<Self::Bits as Activations>::Shape as Shape>::indices()
                    .map(|index| (index, value, loss_fn(self.mutate(value, index).bma(&input))))
                    .filter(|(_, _, l)| *l != null_loss)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
    /// Acts if we were to each element to value
    fn acts(&self, input: &Self::Bits, value: W) -> Self::Bits;
    // grad_slow does the same thing as grads, but it is a lot slower.
    fn acts_slow(&self, input: &Self::Bits, value: W) -> Self::Bits {
        let indices = <<Self::Bits as Activations>::Shape as IndexMap<
            <<Self::Bits as Activations>::Shape as Shape>::Index,
            (),
        >>::index_map((), |index| index);
        <Self::Bits as BitMapPack<<<Self::Bits as Activations>::Shape as Shape>::Index>>::bit_map_pack(&indices, |&index| self.mutate(value, index).act(input))
    }
}

/*
impl<T> WeightArray<Option<bool>> for (T, u32)
where
    T: TritArray + MaskedDistance + Copy + TritMap,
    T::BitArrayType: BitArray + Copy + BitPack + LongDefault,
    bool: Element<<T::BitArrayType as BitArray>::Shape>,
    T::Shape: IndexMap<bool, ()>,
    //<Self::Bits as BitArray>::Shape: IndexMap<bool, ()> + Shape,
    <<T::BitArrayType as BitArray>::Shape as Element<()>>::Array: Shape<Index = <<T::BitArrayType as BitArray>::Shape as Shape>::Index>,
    <u32 as Element<W::StatesShape>>::Array: Element<<Self::Bits as BitArray>::Shape>,
    u32: Element<W::StatesShape>,
{
    type Bits = T::BitArrayType;
    type Index = <<T::BitArrayType as BitArray>::Shape as Shape>::Index;
    const RANGE: usize = 2;
    fn bma(&self, input: &T::BitArrayType) -> u32 {
        self.0.masked_distance(input) * 2 + self.1
    }
    fn mutate(&self, value: Option<bool>, index: <<Self::Bits as BitArray>::Shape as Shape>::Index) -> Self {
        let new_trits = self.0.set_trit(value, index);
        (new_trits, new_trits.mask_zeros())
    }
    fn grads(&self, input: &Self::Bits, value: Option<bool>, cur_act: u32, target_act: u32) -> Self::Bits {
        let value_sign = T::to_bit(value.unwrap_or(false));

        let big_up = cur_act + 2 == target_act;
        let small_up = cur_act + 1 == target_act;
        let eq = target_act == cur_act;
        let small_down = target_act + 1 == cur_act;
        let big_down = target_act + 2 == cur_act;

        if big_down {
            if value.is_some() {
                self.0.map(&input, |trit_sign, trit_mask, input| ((input ^ trit_sign) & trit_mask) & ((trit_sign & trit_mask) ^ value_sign))
            } else {
                Self::Bits::long_default()
            }
        } else if big_up {
            if value.is_some() {
                self.0.map(&input, |trit_sign, trit_mask, input| ((!input ^ trit_sign) & trit_mask) & ((trit_sign & trit_mask) ^ value_sign))
            } else {
                Self::Bits::long_default()
            }
        } else if small_down {
            if value.is_some() {
                self.0.map(&input, |trit_sign, trit_mask, input| (input ^ !value_sign) & !trit_mask)
            } else {
                self.0.map(&input, |trit_sign, trit_mask, input| (input ^ trit_sign) & trit_mask)
            }
        } else if small_up {
            if value.is_some() {
                self.0.map(&input, |trit_sign, trit_mask, input| (input ^ value_sign) & !trit_mask)
            } else {
                self.0.map(&input, |trit_sign, trit_mask, input| (input ^ !trit_sign) & trit_mask)
            }
        } else if eq {
            if value.is_some() {
                self.0.map(&input, |trit_sign, trit_mask, input| (trit_sign ^ !value_sign) & trit_mask)
            } else {
                self.0.map(&input, |trit_sign, trit_mask, input| !trit_mask)
            }
        } else {
            Self::Bits::long_default()
        }
    }
    fn states() -> Vec<Option<bool>> {
        vec![Some(false), None, Some(true)]
    }
}

impl<B> WeightArray<bool> for B
where
    B: BitMap + BitArray + Distance + Copy + LongDefault + Activations + BitMapPack<<<<B as Activations>::Shape as Element<()>>::Array as Shape>::Index>,
    <B as BitArray>::Shape: Shape + IndexMap<<<B as Activations>::Shape as Shape>::Index, ()>,
    bool: Element<<B as BitArray>::Shape>,
    <B as Activations>::Shape: IndexMap<<<B as Activations>::Shape as Shape>::Index, ()> + Shape<Index = <<<B as Activations>::Shape as Element<()>>::Array as Shape>::Index>,
    <<B as Activations>::Shape as Element<()>>::Array: Shape,
    <<<B as Activations>::Shape as Element<()>>::Array as Shape>::Index: Element<<B as Activations>::Shape>,
    <<B as Activations>::Shape as Shape>::Index: Element<<B as Activations>::Shape>,
{
    type Bits = B;
    const RANGE: usize = 2;
    const THRESHOLD: u32 = B::Shape::N as u32 / 2;
    fn bma(&self, input: &B) -> u32 {
        self.distance(input)
    }
    fn mutate(self, value: bool, index: <<Self::Bits as Activations>::Shape as Shape>::Index) -> Self {
        self.set_bit(value, index)
    }
    fn losses<F: Fn(u32) -> u64>(&self, input: &Self::Bits, loss_fn: F) -> Vec<(<<Self::Bits as Activations>::Shape as Shape>::Index, bool, u64)> {
        self.losses_slow(input, loss_fn)
    }
    //fn acts(&self, input: &Self::Bits, value: bool) -> <[u32; 2] as Element<<B as BitArray>::Shape>>::Array {}
    //fn grads(&self, input: &Self::Bits, value: bool, cur_act: u32, target_act: u32) -> Self::Bits {
    //    let up = cur_act + 2 == target_act;
    //    let down = target_act + 2 == cur_act;
    //    let eq = target_act == cur_act;

    //    if up {
    //        if value {
    //            self.map(&input, |weight, input| !(weight ^ input) & !weight)
    //        } else {
    //            self.map(&input, |weight, input| !(weight ^ input) & weight)
    //        }
    //    } else if eq {
    //        if value {
    //            self.map(&input, |weight, _| weight)
    //        } else {
    //            self.map(&input, |weight, _| !weight)
    //        }
    //    } else if down {
    //        if value {
    //            self.map(&input, |weight, input| (weight ^ input) & !weight)
    //        } else {
    //            self.map(&input, |weight, input| (weight ^ input) & weight)
    //        }
    //    } else {
    //        Self::Bits::long_default()
    //    }
    //}
    fn acts(&self, input: &Self::Bits, value: bool) -> Self::Bits {
        self.acts_slow(input, value)
    }
}
*/

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
    [O; L]: LongDefault,
{
    fn bit_mul(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::long_default();
        for i in 0..L {
            target[i] = self[i].bit_mul(input);
        }
        target
    }
}

/// An array of binary sctivations
pub trait Activations
where
    Self: Sized,
    Self::Shape: Shape,
{
    /// The shape of the bits.
    /// Note that this is not the shape of the array with words as elements,
    /// but rather the shape of the array with bits as elements.
    type Shape;
    type Index;
    fn get_act(&self, index: Self::Index) -> bool;
    fn set_act(mut self, index: Self::Index, value: bool) -> Self {
        self.set_act_in_place(index, value);
        self
    }
    fn set_act_in_place(&mut self, index: Self::Index, value: bool);
}

impl<T: Activations, const L: usize> Activations for [T; L]
where
    [T; L]: LongDefault,
{
    type Shape = [T::Shape; L];
    type Index = (u8, T::Index);
    fn get_act(&self, (head, tail): (u8, T::Index)) -> bool {
        self[head as usize].get_act(tail)
    }
    fn set_act_in_place(&mut self, (head, tail): (u8, T::Index), value: bool) {
        self[head as usize].set_act_in_place(tail, value);
    }
}

trait WeightElement<A> {
    type Weights;
}

impl<W: WeightElement<A>, A, const L: usize> WeightElement<[A; L]> for W {
    type Weights = [W::Weights; L];
}

/// A collection of bits which has a shape.
pub trait BitArray
where
    Self: Sized,
    Self::Shape: Shape,
{
    /// The shape of the bits.
    /// Note that this is not the shape of the array with words as elements,
    /// but rather the shape of the array with bits as elements.
    type Shape;
    fn get_bit(&self, i: <Self::Shape as Shape>::Index) -> bool;
    fn set_bit_in_place(&mut self, bit: bool, i: <Self::Shape as Shape>::Index);
    fn set_bit(mut self, bit: bool, i: <Self::Shape as Shape>::Index) -> Self {
        self.set_bit_in_place(bit, i);
        self
    }
}

impl<T: BitArray, const L: usize> BitArray for [T; L] {
    type Shape = [T::Shape; L];
    fn get_bit(&self, (i, tail): (u8, <T::Shape as Shape>::Index)) -> bool {
        self[i as usize].get_bit(tail)
    }
    fn set_bit_in_place(&mut self, bit: bool, (i, tail): (u8, <T::Shape as Shape>::Index)) {
        self[i as usize].set_bit_in_place(bit, tail)
    }
}

pub trait BitMap
where
    Self::WordType: ops::BitAnd<Output = Self::WordType>
        + ops::BitOr<Output = Self::WordType>
        + ops::BitXor<Output = Self::WordType>
        + ops::Not<Output = Self::WordType>
        + Copy
        + fmt::Binary,
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
    [T; L]: LongDefault,
{
    type WordType = T::WordType;
    fn map<F: Fn(Self::WordType, Self::WordType) -> Self::WordType>(
        &self,
        rhs: &Self,
        map_fn: F,
    ) -> Self {
        let mut target = <[T; L]>::long_default();
        for i in 0..L {
            target[i] = self[i].map(&rhs[i], &map_fn);
        }
        target
    }
    fn to_bit(sign: bool) -> Self::WordType {
        T::to_bit(sign)
    }
}

/*
pub trait TritMap
where
    Self::WordType: ops::BitAnd<Output = Self::WordType> + ops::BitOr<Output = Self::WordType> + ops::BitXor<Output = Self::WordType> + ops::Not<Output = Self::WordType> + Copy + fmt::Binary,
    Self: TritArray,
{
    type WordType;
    fn map<F: Fn(Self::WordType, Self::WordType, Self::WordType) -> Self::WordType>(&self, bits: &Self::BitArrayType, map_fn: F) -> Self::BitArrayType;
    fn map_trit<F: Fn(Self::WordType, Self::WordType, Self::WordType) -> (Self::WordType, Self::WordType)>(&self, bits: &Self::BitArrayType, map_fn: F) -> Self;
    fn to_bit(sign: bool) -> Self::WordType;
}

impl<T: TritArray + TritMap, const L: usize> TritMap for [T; L]
where
    [T; L]: LongDefault,
    [<T as TritArray>::BitArrayType; L]: LongDefault,
{
    type WordType = T::WordType;
    fn map<F: Fn(Self::WordType, Self::WordType, Self::WordType) -> Self::WordType>(&self, bits: &Self::BitArrayType, map_fn: F) -> [<T as TritArray>::BitArrayType; L] {
        let mut target = <[<T as TritArray>::BitArrayType; L]>::long_default();
        for i in 0..L {
            target[i] = self[i].map(&bits[i], &map_fn);
        }
        target
    }
    fn map_trit<F: Fn(Self::WordType, Self::WordType, Self::WordType) -> (Self::WordType, Self::WordType)>(&self, bits: &Self::BitArrayType, map_fn: F) -> Self {
        let mut target = <[T; L]>::long_default();
        for i in 0..L {
            target[i] = self[i].map_trit(&bits[i], &map_fn);
        }
        target
    }
    fn to_bit(sign: bool) -> Self::WordType {
        T::to_bit(sign)
    }
}
*/

pub trait BitMapPack<E: Element<Self::Shape>>
where
    Self: Activations,
{
    fn bit_map_pack<F: Fn(&E) -> bool>(
        input: &<E as Element<Self::Shape>>::Array,
        map_fn: F,
    ) -> Self;
}

impl<E: Element<T::Shape>, T: BitMapPack<E>, const L: usize> BitMapPack<E> for [T; L]
where
    [T; L]: LongDefault,
{
    fn bit_map_pack<F: Fn(&E) -> bool>(
        input: &[<E as Element<T::Shape>>::Array; L],
        map_fn: F,
    ) -> [T; L] {
        let mut target = <[T; L]>::long_default();
        for i in 0..L {
            target[i] = T::bit_map_pack(&input[i], &map_fn);
        }
        target
    }
}

pub trait BitZipMap<E: Element<Self::Shape>, O: Element<Self::Shape>>
where
    Self: BitArray,
{
    fn bit_zip_map<F: Fn(bool, E) -> O>(
        &self,
        vals: &<E as Element<Self::Shape>>::Array,
        map_fn: F,
    ) -> <O as Element<Self::Shape>>::Array;
    fn bit_zip_map_mut<F: Fn(&mut O, bool, E)>(
        &self,
        target: &mut <O as Element<Self::Shape>>::Array,
        vals: &<E as Element<Self::Shape>>::Array,
        map_fn: F,
    );
}

impl<T: BitArray + BitZipMap<E, O>, E: Element<T::Shape>, O: Element<T::Shape>, const L: usize>
    BitZipMap<E, O> for [T; L]
where
    [<O as Element<T::Shape>>::Array; L]: LongDefault,
{
    fn bit_zip_map<F: Fn(bool, E) -> O>(
        &self,
        vals: &[<E as Element<T::Shape>>::Array; L],
        map_fn: F,
    ) -> [<O as Element<T::Shape>>::Array; L] {
        let mut target = <[<O as Element<T::Shape>>::Array; L]>::long_default();
        for i in 0..L {
            target[i] = self[i].bit_zip_map(&vals[i], &map_fn);
        }
        target
    }
    fn bit_zip_map_mut<F: Fn(&mut O, bool, E)>(
        &self,
        target: &mut [<O as Element<T::Shape>>::Array; L],
        vals: &[<E as Element<T::Shape>>::Array; L],
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
    [T; L]: LongDefault,
{
    fn bit_and(&self, rhs: &Self) -> Self {
        let mut target = <[T; L]>::long_default();
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
    [T; L]: LongDefault,
{
    fn bit_or(&self, rhs: &Self) -> Self {
        let mut target = <[T; L]>::long_default();
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
    [O; L]: LongDefault,
{
    fn ffbvm(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::long_default();
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
    [O; L]: LongDefault,
{
    fn bfbvmm(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::long_default();
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
    f32: Element<Self::Shape>,
{
    fn bfma(&self, weights: &<f32 as Element<Self::Shape>>::Array) -> f32;
}

impl<T: BitArray + BFMA, const L: usize> BFMA for [T; L]
where
    f32: Element<T::Shape>,
{
    fn bfma(&self, weights: &[<f32 as Element<T::Shape>>::Array; L]) -> f32 {
        let mut sum = 0f32;
        for i in 0..L {
            sum += self[i].bfma(&weights[i]);
        }
        sum
    }
}

/*
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
*/

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
    Self::Shape: Shape,
{
    type Shape;
    fn mask_zeros(&self) -> u32;
    //fn get_trit(&self, index: <Self::Shape as Shape>::Index) -> Option<bool>;
    //fn set_trit_in_place(&mut self, index: <Self::Shape as Shape>::Index, trit: Option<bool>);
    //fn set_trit(mut self, index: <Self::Shape as Shape>::Index, trit: Option<bool>) -> Self {
    //    self.set_trit_in_place(index, trit);
    //    self
    //}
}

impl<T: TritArray, const L: usize> TritArray for [T; L] {
    type Shape = [T::Shape; L];
    fn mask_zeros(&self) -> u32 {
        self.iter().map(|x| x.mask_zeros()).sum()
    }
    //fn get_trit(&self, (head, tail): (u8, <T::Shape as Shape>::Index)) -> Option<bool> {
    //    self[head as usize].get_trit(tail)
    //}
    //fn set_trit_in_place(&mut self, (head, tail): (u8, <T::Shape as Shape>::Index), trit: Option<bool>) {
    //    self[head as usize].set_trit_in_place(tail, trit);
    //}
}

macro_rules! impl_long_default_for_type {
    ($type:ty) => {
        impl LongDefault for $type {
            fn long_default() -> $type {
                <$type as Default>::default()
            }
        }
    };
}

impl_long_default_for_type!(Option<bool>);
impl_long_default_for_type!(bool);

macro_rules! for_uints {
    ($q_type:ident, $t_type:ident, $b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        impl_long_default_for_type!($u_type);

        /// A word of trits. The 0 element is the signs, the 1 element is the magnitudes.
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $t_type(pub $u_type, pub $u_type);

        impl $t_type {
            fn get_trit(&self, index: usize) -> Option<bool> {
                let sign = (self.0 >> index) & 1 == 1;
                let magn = (self.1 >> index) & 1 == 1;
                Some(sign).filter(|_| magn)
            }
            fn set_trit_in_place(&mut self, index: usize, trit: Option<bool>) {
                self.0 &= !(1 << index);
                self.0 |= ((trit.unwrap_or(false) as $u_type) << index);

                self.1 &= !(1 << index);
                self.1 |= ((trit.is_some() as $u_type) << index);
            }
        }

        impl_long_default_for_type!($t_type);

        /*
        impl TritGrads for $t_type {
            #[inline(always)]
            fn grads(&self, &input: &$b_type, sign: bool) -> Self {
                let sign = (Wrapping(0) - Wrapping(sign as $u_type)).0;
                let acts = self.0 ^ input.0;
                let mask = !((acts ^ sign) & self.1);

                $t_type(input.0 ^ sign, mask)
            }
        }
        */
        impl PartialEq for $t_type {
            fn eq(&self, other: &Self) -> bool {
                ((self.0 & self.1) == (other.0 & self.1)) & (self.1 == other.1)
            }
        }
        impl Eq for $t_type {}

        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $b_type(pub $u_type);

        impl_long_default_for_type!($b_type);

        /*
        impl TritArray for $t_type {
            type Shape = [(); $len];
            fn mask_zeros(&self) -> u32 {
                self.1.count_zeros()
            }
        }
        */

        impl PackedElement<[(); $len]> for Option<bool> {
            type Array = $t_type;
        }

        impl PackedArray for $t_type {
            type Weight = Option<bool>;
            type Shape = [(); $len];
            fn get_weight(&self, (b, _): (u8, ())) -> Option<bool> {
                self.get_trit(b as usize)
            }
            fn set_weight_in_place(&mut self, (b, _): (u8, ()), value: Option<bool>) {
                self.set_trit_in_place(b as usize, value);
            }
        }

        impl $b_type {
            pub fn count_ones(self) -> u32 {
                self.0.count_ones()
            }
            pub fn get_bit(self, index: usize) -> bool {
                ((self.0 >> index) & 1) == 1
            }
            pub fn set_bit_in_place(&mut self, index: usize, value: bool) {
                self.0 &= (1 << index);
                self.0 |= ((value as $u_type) << index);
            }
        }
        impl Activations for $b_type {
            type Shape = [(); $len];
            type Index = (u8, ());
            fn get_act(&self, (head, _): (u8, ())) -> bool {
                self.get_bit(head as usize)
            }
            fn set_act_in_place(&mut self, (head, _): (u8, ()), value: bool) {
                self.set_bit_in_place(head as usize, value);
            }
        }

        impl PackedElement<[(); $len]> for bool {
            type Array = $b_type;
        }

        impl PackedArray for $b_type {
            type Weight = bool;
            type Shape = [(); $len];
            fn get_weight(&self, (b, _): (u8, ())) -> bool {
                self.get_bit(b as usize)
            }
            fn set_weight_in_place(&mut self, (b, _): (u8, ()), value: bool) {
                self.set_bit_in_place(b as usize, value);
            }
        }

        /// A word of trits. The 0 element is the sign, the 1 element is the magnitudes.
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $q_type(pub $u_type, pub $u_type);

        impl_long_default_for_type!($q_type);

        impl PackedElement<[(); $len]> for (bool, bool) {
            type Array = $q_type;
        }

        impl PackedArray for $q_type {
            type Weight = (bool, bool);
            type Shape = [(); $len];
            fn get_weight(&self, (index, _): (u8, ())) -> (bool, bool) {
                (((self.0 >> index) & 1) == 1, ((self.1 >> index) & 1) == 1)
            }
            fn set_weight_in_place(&mut self, (index, _): (u8, ()), value: (bool, bool)) {
                self.0 &= (1 << index);
                self.0 |= ((value.0 as $u_type) << index);

                self.1 &= (1 << index);
                self.1 |= ((value.1 as $u_type) << index);
            }
        }

        impl WeightElement<$b_type> for bool {
            type Weights = $b_type;
        }

        impl WeightElement<$b_type> for Option<bool> {
            type Weights = $t_type;
        }

        impl WeightElement<$b_type> for (bool, bool) {
            type Weights = $q_type;
        }

        /*
        impl<E: Copy, O: Copy + LongDefault> BitZipMap<E, O> for $b_type {
            fn bit_zip_map<F: Fn(bool, E) -> O>(&self, vals: &[E; $len], map_fn: F) -> [O; $len] {
                let mut target = [O::long_default(); $len];
                for b in 0..$len {
                    target[b] = map_fn(self.get_bit(b), vals[b]);
                }
                target
            }
            fn bit_zip_map_mut<F: Fn(&mut O, bool, E)>(&self, target: &mut [O; $len], vals: &[E; $len], map_fn: F) {
                for b in 0..$len {
                    map_fn(&mut target[b], self.get_bit(b), vals[b]);
                }
            }
        }
        */
        impl<E> BitMapPack<E> for $b_type {
            fn bit_map_pack<F: Fn(&E) -> bool>(input: &[E; $len], map_fn: F) -> $b_type {
                let mut target = <$u_type>::long_default();
                for b in 0..$len {
                    target |= (map_fn(&input[b]) as $u_type) << b;
                }
                $b_type(target)
            }
        }
        /*
        impl<E> BitMap<E> for $b_type
        where
            [E; $len]: LongDefault,
        {
            fn bit_map<F: Fn(bool) -> E>(&self, map_fn: F) -> [E; $len] {
                let mut target = <[E; $len]>::long_default();
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

        /*
        impl TritMap for $t_type {
            type WordType = $u_type;
            #[inline(always)]
            fn map<F: Fn($u_type, $u_type, $u_type) -> $u_type>(&self, &rhs: &$b_type, map_fn: F) -> $b_type {
                $b_type(map_fn(self.0, self.1, rhs.0))
            }
            #[inline(always)]
            fn map_trit<F: Fn(Self::WordType, Self::WordType, Self::WordType) -> (Self::WordType, Self::WordType)>(&self, rhs: &Self::BitArrayType, map_fn: F) -> $t_type {
                let (sign, mask) = map_fn(self.0, self.1, rhs.0);
                $t_type(sign, mask)
            }

            fn to_bit(sign: bool) -> Self::WordType {
                (Wrapping(0) - Wrapping(sign as $u_type)).0
            }
        }
        impl BitArray for $b_type {
            type Shape = [(); $len];
            //type WordType = $b_type;
            //type WordShape = ();
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
        */
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
        impl Default for $q_type {
            fn default() -> Self {
                $q_type(0, 0)
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
        impl fmt::Display for $q_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, $format_string, self.0)?;
                write!(f, $format_string, self.1)
            }
        }
        impl fmt::Debug for $q_type {
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

for_uints!(q8, t8, b8, u8, 8, "{:08b}");
for_uints!(q16, t16, b16, u16, 16, "{:016b}");
for_uints!(q32, t32, b32, u32, 32, "{:032b}");
for_uints!(q64, t64, b64, u64, 64, "{:064b}");
for_uints!(q128, t128, b128, u128, 128, "{:0128b}");

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

/*
impl TritArray for t1 {
    type Shape = [(); 1];
    fn mask_zeros(&self) -> u32 {
        (!((self.0 >> 1) & 1u8)) as u32
    }
    fn get_trit(&self, _: (u8, ())) -> Option<bool> {
        None
    }
    fn set_trit_in_place(&mut self, _: (u8, ()), trit: Option<bool>) {
        *self = Self::new_from_option_bool(trit)
    }
}

impl MaskedDistance for t1 {
    #[inline(always)]
    fn masked_distance(&self, &bits: &b1) -> u32 {
        (((bits.0 as u8 ^ self.0) & (self.0 >> 1)) & 1u8) as u32
    }
}
*/

/// A word of 1 bits
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct b1(pub bool);

impl BitArray for b1 {
    type Shape = [(); 1];
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
    use super::{b128, b16, b32, b64, b8, BitArray, TritArray, TritGrads, TritMap, WeightArray};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;
    use test::Bencher;

    //type InputType = [b128; 2];
    type InputType = [[[b64; 2]; 3]; 3];
    //type InputType = [[[[[b8; 2]; 2]; 2]; 2]; 2];
    //type InputType = [[b16; 32]; 32];
    //type InputType = [b64; 32];
    //type InputType = [[[[b32; 3]; 3]; 3]; 3];

    type TritType = <InputType as BitArray>::TritArrayType;

    #[test]
    fn rand_bit_act_grads() {
        let mut rng = Hc128Rng::seed_from_u64(0);
        (0..50).for_each(|_| {
            let weights: InputType = rng.gen();
            let input: InputType = rng.gen();
            let act = weights.bma(&input);

            for target_act in act.saturating_sub(InputType::RANGE as u32)
                ..=act.saturating_add(InputType::RANGE as u32)
            {
                for &val in InputType::states().iter() {
                    let grads = weights.grads(&input, val, act, target_act);
                    let true_grads = weights.grads_slow(&input, val, act, target_act);
                    assert_eq!(grads, true_grads);
                }
            }
        })
    }
    #[test]
    fn rand_trit_act_grads() {
        let mut rng = Hc128Rng::seed_from_u64(0);
        (0..100).for_each(|_| {
            let weights: TritType = rng.gen();
            let weights = (weights, weights.mask_zeros());
            let input: InputType = rng.gen();
            let act = weights.bma(&input);

            dbg!(act);
            for target_act in act.saturating_sub(<(TritType, u32)>::RANGE as u32)
                ..=act.saturating_add(<(TritType, u32)>::RANGE as u32)
            {
                for &val in <(TritType, u32)>::states().iter() {
                    dbg!(val);
                    dbg!(target_act);
                    let grads = weights.grads(&input, val, act, target_act);
                    let true_grads = weights.grads_slow(&input, val, act, target_act);
                    assert_eq!(grads, true_grads);
                }
            }
        })
    }
    #[test]
    fn grads_map_test() {
        let mut rng = Hc128Rng::seed_from_u64(0);

        (0..100).for_each(|_| {
            let bits: InputType = rng.gen();
            let trits: TritType = rng.gen();
            let sign: bool = rng.gen();

            let grads_a = trits.grads(&bits, sign);

            let sign = TritType::to_bit(sign);
            let grads_b = trits.map_trit(&bits, |trit_sign, trit_mask, input_bit| {
                let acts = trit_sign ^ input_bit;
                let mask = !((acts ^ sign) & trit_mask);

                (input_bit ^ sign, mask)
            });
            assert_eq!(grads_a, grads_b);
        });
    }

    #[bench]
    fn bench_grads(b: &mut Bencher) {
        let mut rng = Hc128Rng::seed_from_u64(0);

        let bits: InputType = rng.gen();
        let trits: TritType = rng.gen();
        let sign = test::black_box(false);
        let bits = test::black_box(bits);
        let trits = test::black_box(trits);

        b.iter(|| trits.grads(&bits, sign));
    }
    #[bench]
    fn bench_map_grads(b: &mut Bencher) {
        let mut rng = Hc128Rng::seed_from_u64(0);

        let bits: InputType = rng.gen();
        let trits: TritType = rng.gen();
        let sign = test::black_box(false);
        let bits = test::black_box(bits);
        let trits = test::black_box(trits);

        b.iter(|| {
            let sign = TritType::to_bit(sign);
            trits.map_trit(&bits, |trit_sign, trit_mask, input_bit| {
                let acts = trit_sign ^ input_bit;
                let mask = !((acts ^ sign) & trit_mask);

                (input_bit ^ sign, mask)
            })
        });
    }
    #[bench]
    fn bench_bit_map_acts(b: &mut Bencher) {
        let mut rng = Hc128Rng::seed_from_u64(0);

        let weights: InputType = rng.gen();
        let input: InputType = rng.gen();
        let act = weights.bma(&input);

        let weights = test::black_box(weights);
        let input = test::black_box(input);
        let act = test::black_box(act);

        b.iter(|| {
            let mut sum = 0usize;
            for target_act in act.saturating_sub(InputType::RANGE as u32)
                ..=act.saturating_add(InputType::RANGE as u32)
            {
                for &val in InputType::states().iter() {
                    let grads = weights.grads(&input, val, act, target_act);
                    sum += grads.len();
                }
            }
            sum
        });
    }
    #[bench]
    fn bench_trit_map_acts(b: &mut Bencher) {
        let mut rng = Hc128Rng::seed_from_u64(0);

        let weights: TritType = rng.gen();
        let weights = (weights, weights.mask_zeros());
        let input: InputType = rng.gen();
        let act = weights.bma(&input);

        let weights = test::black_box(weights);
        let input = test::black_box(input);
        let act = test::black_box(act);

        b.iter(|| {
            let mut sum = 0usize;
            for target_act in act.saturating_sub(<(TritType, u32)>::RANGE as u32)
                ..=act.saturating_add(<(TritType, u32)>::RANGE as u32)
            {
                for &val in <(TritType, u32)>::states().iter() {
                    let grads = weights.grads(&input, val, act, target_act);
                    sum += grads.len();
                }
            }
            sum
        });
    }
}
