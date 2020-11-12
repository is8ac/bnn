// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use crate::shape::{Element, LongDefault, Shape, Wrap};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter;
use std::num::Wrapping;
use std::ops;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};
use std::slice;

pub trait PackedElement<S: Shape> {
    type Array;
}

impl<W: PackedElement<S> + Weight, S: Shape, const L: usize> PackedElement<[S; L]> for W {
    type Array = [<W as PackedElement<S>>::Array; L];
}

pub trait PackedArray
where
    Self::Shape: Shape,
    Self::Weight: Weight,
    Self: Sized,
    Self::UWord: ops::BitAnd<Output = Self::UWord>
        + ops::BitOr<Output = Self::UWord>
        + ops::BitXor<Output = Self::UWord>
        + ops::Not<Output = Self::UWord>
        + ops::Not<Output = Self::UWord>
        + Copy
        + fmt::Binary,
    Self::Word: Copy + PackedWord<Weight = Self::Weight>,
{
    type Weight;
    type Word;
    type UWord;
    type Shape;
    fn blit(val: Self::Weight) -> Self;
    fn get_weight(&self, index: <Self::Shape as Shape>::Index) -> Self::Weight;
    fn set_weight_in_place(&mut self, index: <Self::Shape as Shape>::Index, value: Self::Weight);
    fn set_weight(mut self, index: <Self::Shape as Shape>::Index, value: Self::Weight) -> Self {
        self.set_weight_in_place(index, value);
        self
    }
}

impl<T: Copy + PackedArray, const L: usize> PackedArray for [T; L]
where
    Self::Shape: Shape<Index = (u8, <T::Shape as Shape>::Index)>,
    Self::Weight: Weight,
{
    type Weight = T::Weight;
    type Word = T::Word;
    type UWord = T::UWord;
    type Shape = [T::Shape; L];
    fn blit(val: Self::Weight) -> Self {
        [T::blit(val); L]
    }
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
    const RANGE: u32;
    const N: usize;
    fn states() -> iter::Cloned<slice::Iter<'static, Self>>;
    fn bma(&self, input: bool) -> u32;
}

impl Weight for bool {
    const RANGE: u32 = 1;
    const N: usize = 2;
    fn states() -> iter::Cloned<slice::Iter<'static, bool>> {
        [true, false].iter().cloned()
    }
    fn bma(&self, input: bool) -> u32 {
        (self ^ input) as u32
    }
}

impl Weight for Option<bool> {
    const RANGE: u32 = 2;
    const N: usize = 3;
    fn states() -> iter::Cloned<slice::Iter<'static, Option<bool>>> {
        [Some(true), None, Some(false)].iter().cloned()
    }
    fn bma(&self, input: bool) -> u32 {
        if let Some(sign) = self {
            (sign ^ input) as u32 * 2
        } else {
            1
        }
    }
}

impl Weight for (bool, bool) {
    const RANGE: u32 = 3;
    const N: usize = 4;
    fn states() -> iter::Cloned<slice::Iter<'static, (bool, bool)>> {
        [(true, true), (true, false), (false, false), (false, true)]
            .iter()
            .cloned()
    }
    fn bma(&self, input: bool) -> u32 {
        let input: i8 = if input { -1 } else { 1 };
        let weight: i8 = (if self.0 { -1 } else { 1 }) * (self.1 as i8 + 1);
        ((weight * input) + 3) as u32
    }
}

impl Weight for Option<(bool, bool)> {
    const RANGE: u32 = 3;
    const N: usize = 5;
    fn states() -> iter::Cloned<slice::Iter<'static, Option<(bool, bool)>>> {
        [
            Some((true, true)),
            Some((true, false)),
            None,
            Some((false, false)),
            Some((false, true)),
        ]
        .iter()
        .cloned()
    }
    fn bma(&self, input: bool) -> u32 {
        if let Some(weight) = self {
            let input: i8 = if input { -1 } else { 1 };
            let weight: i8 = (if weight.0 { -1 } else { 1 }) * (weight.1 as i8 + 1);
            ((weight * input) + 3) as u32
        } else {
            3
        }
    }
}

pub trait WeightArray
where
    Self: Copy,
    bool: PackedElement<Self::Shape>,
    Self::Shape: Shape + PackedIndexMap<bool>,
    Self::Weight: 'static + Weight,
    <bool as PackedElement<Self::Shape>>::Array:
        PackedArray<Weight = bool, Shape = Self::Shape> + Copy,
{
    type Shape;
    type Weight;
    const MAX: u32;
    const THRESHOLD: u32;
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn get_weight(&self, index: <Self::Shape as Shape>::Index) -> Self::Weight;
    /// multiply accumulate
    fn bma(&self, input: &<bool as PackedElement<Self::Shape>>::Array) -> u32;
    fn bma_slow(&self, input: &<bool as PackedElement<Self::Shape>>::Array) -> u32 {
        <Self::Shape as Shape>::indices()
            .map(|i| self.get_weight(i).bma(input.get_weight(i)))
            .sum()
    }
    /// thresholded activation
    fn act(&self, input: &<bool as PackedElement<Self::Shape>>::Array) -> bool {
        self.bma(input) > Self::THRESHOLD
    }
    fn act_is_alive(&self, input: &<bool as PackedElement<Self::Shape>>::Array) -> bool {
        let act = self.bma(input);
        ((act + Self::Weight::RANGE) > Self::THRESHOLD)
            | ((Self::THRESHOLD + Self::Weight::RANGE) < act)
    }
    fn mutate_in_place(&mut self, index: <Self::Shape as Shape>::Index, value: Self::Weight);
    fn mutate(mut self, index: <Self::Shape as Shape>::Index, value: Self::Weight) -> Self {
        self.mutate_in_place(index, value);
        self
    }
    /// loss delta if we were to set each of the elements to each different value. Is permited to prune null deltas.
    fn loss_deltas<F: Fn(u32) -> i64>(
        &self,
        input: &<bool as PackedElement<Self::Shape>>::Array,
        loss_delta_fn: F,
    ) -> Vec<(<Self::Shape as Shape>::Index, Self::Weight, i64)>;
    /// Does the same thing as loss_deltas but is a lot slower.
    fn loss_deltas_slow<F: Fn(u32) -> i64>(
        &self,
        input: &<bool as PackedElement<Self::Shape>>::Array,
        loss_delta_fn: F,
    ) -> Vec<(<Self::Shape as Shape>::Index, Self::Weight, i64)> {
        <Self::Weight as Weight>::states()
            .map(|value| {
                <Self::Shape as Shape>::indices()
                    .map(|index| {
                        (
                            index,
                            value,
                            loss_delta_fn(self.mutate(index, value).bma(&input)),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .filter(|(_, _, l)| *l != 0)
            .collect()
    }
    /// Acts if we were to set each element to value
    fn acts(
        &self,
        input: &<bool as PackedElement<Self::Shape>>::Array,
        value: Self::Weight,
    ) -> <bool as PackedElement<Self::Shape>>::Array;
    /// acts_slow does the same thing as grads, but it is a lot slower.
    fn acts_slow(
        &self,
        input: &<bool as PackedElement<Self::Shape>>::Array,
        value: Self::Weight,
    ) -> <bool as PackedElement<Self::Shape>>::Array {
        <Self::Shape as PackedIndexMap<bool>>::index_map(|index| {
            self.mutate(index, value).act(input)
        })
    }
    /// Act if each input is flipped.
    fn input_acts(
        &self,
        input: &<bool as PackedElement<Self::Shape>>::Array,
    ) -> <bool as PackedElement<Self::Shape>>::Array;
    /// input_acts_slow does the same thing as input_acts, but it is a lot slower.
    fn input_acts_slow(
        &self,
        input: &<bool as PackedElement<Self::Shape>>::Array,
    ) -> <bool as PackedElement<Self::Shape>>::Array {
        <Self::Shape as PackedIndexMap<bool>>::index_map(|index| {
            self.act(&input.set_weight(index, !input.get_weight(index)))
        })
    }
}

impl<B> WeightArray for B
where
    B: Copy + WBBZM + PackedArray<Weight = bool> + Distance + std::fmt::Debug,
    bool: PackedElement<B::Shape, Array = B>,
    Standard: Distribution<B>,
    B::Shape: PackedIndexMap<bool>,
    <B as PackedArray>::Word: BitWord<Word = <B as PackedArray>::UWord>,
    <B::Shape as Shape>::Index: std::fmt::Debug,
    <B::Shape as Shape>::IndexIter: Iterator<Item = <B::Shape as Shape>::Index>,
    <bool as PackedElement<B::Shape>>::Array: PackedArray<Weight = bool> + Copy,
    <<B as PackedArray>::Shape as Shape>::Index: Element<<B as PackedArray>::Shape>,
{
    type Shape = B::Shape;
    type Weight = bool;
    const MAX: u32 = B::Shape::N as u32;
    const THRESHOLD: u32 = B::Shape::N as u32 / 2;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        rng.gen()
    }
    fn get_weight(&self, index: <B::Shape as Shape>::Index) -> bool {
        <B as PackedArray>::get_weight(self, index)
    }
    fn bma(&self, input: &B) -> u32 {
        self.distance(input)
    }
    fn mutate_in_place(&mut self, index: <B::Shape as Shape>::Index, value: bool) {
        self.set_weight_in_place(index, value)
    }
    fn loss_deltas<F: Fn(u32) -> i64>(
        &self,
        input: &B,
        loss_delta_fn: F,
    ) -> Vec<(<B::Shape as Shape>::Index, bool, i64)> {
        let cur_act = self.bma(input);
        let deltas = [
            loss_delta_fn(cur_act.saturating_add(1)),
            loss_delta_fn(cur_act.saturating_sub(1)),
        ];
        <B::Shape as Shape>::indices()
            .map(|i| {
                let weight = self.get_weight(i);
                let input = input.get_weight(i);
                (i, !weight, deltas[(weight ^ input) as usize])
            })
            .filter(|&(_, _, l)| l != 0)
            .collect()
    }
    fn acts(&self, input: &B, value: bool) -> B {
        let cur_act = self.bma(input);
        if cur_act == Self::THRESHOLD {
            // we need to go up by one to flip
            if value {
                self.map(input, |weight, input| {
                    !(weight.sign() ^ input) & !weight.sign()
                })
            } else {
                self.map(input, |weight, input| {
                    !(weight.sign() ^ input) & weight.sign()
                })
            }
        } else if cur_act == (Self::THRESHOLD + 1) {
            // we need to go down by one to flip
            if value {
                self.map(input, |weight, input| {
                    !(weight.sign() ^ input) | weight.sign()
                })
            } else {
                self.map(input, |weight, input| {
                    !(weight.sign() ^ input) | !weight.sign()
                })
            }
        } else {
            B::blit(cur_act > Self::THRESHOLD)
        }
    }
    fn input_acts(
        &self,
        input: &<bool as PackedElement<Self::Shape>>::Array,
    ) -> <bool as PackedElement<Self::Shape>>::Array {
        let cur_act = self.bma(input);
        if (cur_act == Self::THRESHOLD) | (cur_act == (Self::THRESHOLD + 1)) {
            self.map(input, |weight, input| weight.sign() ^ !input)
        } else {
            B::blit(cur_act > Self::THRESHOLD)
        }
    }
}

impl<T> WeightArray for (T, u32)
where
    bool: PackedElement<T::Shape>,
    Option<bool>: PackedElement<T::Shape, Array = T>,
    T: PackedArray<Weight = Option<bool>> + MaskedDistance + Copy + WBBZM,
    Standard: Distribution<T>,
    T::Word: TritWord<Word = <T as PackedArray>::UWord>,
    T::Shape: PackedIndexMap<bool>,
    <T::Shape as Shape>::IndexIter: Iterator<Item = <T::Shape as Shape>::Index>,
    <bool as PackedElement<T::Shape>>::Array: PackedArray<Weight = bool, Shape = T::Shape> + Copy,
{
    type Shape = T::Shape;
    type Weight = Option<bool>;
    const MAX: u32 = <T::Shape as Shape>::N as u32 * 2;
    const THRESHOLD: u32 = T::Shape::N as u32;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        let weights: T = rng.gen();
        (weights, weights.mask_zeros())
    }
    fn get_weight(&self, index: <T::Shape as Shape>::Index) -> Option<bool> {
        <T as PackedArray>::get_weight(&self.0, index)
    }
    fn bma(&self, input: &<bool as PackedElement<T::Shape>>::Array) -> u32 {
        self.0.masked_distance(input) * 2 + self.1
    }
    fn mutate_in_place(&mut self, index: <T::Shape as Shape>::Index, value: Option<bool>) {
        self.0.set_weight_in_place(index, value);
        self.1 = self.0.mask_zeros();
    }
    fn loss_deltas<F: Fn(u32) -> i64>(
        &self,
        input: &<bool as PackedElement<T::Shape>>::Array,
        loss_delta_fn: F,
    ) -> Vec<(<T::Shape as Shape>::Index, Option<bool>, i64)> {
        let cur_act = self.bma(input);
        let deltas = [
            [
                loss_delta_fn(cur_act.saturating_sub(1)),
                loss_delta_fn(cur_act.saturating_add(1)),
            ],
            [
                loss_delta_fn(cur_act.saturating_sub(2)),
                loss_delta_fn(cur_act.saturating_add(2)),
            ],
        ];

        <T::Shape as Shape>::indices()
            .map(|i| {
                let input = input.get_weight(i);
                if let Some(sign) = self.0.get_weight(i) {
                    if sign {
                        iter::once((i, Some(false), deltas[1][input as usize])).chain(iter::once((
                            i,
                            None,
                            deltas[0][input as usize],
                        )))
                    } else {
                        iter::once((i, None, deltas[0][!input as usize])).chain(iter::once((
                            i,
                            Some(true),
                            deltas[1][!input as usize],
                        )))
                    }
                } else {
                    iter::once((i, Some(false), deltas[0][input as usize])).chain(iter::once((
                        i,
                        Some(true),
                        deltas[0][!input as usize],
                    )))
                }
            })
            .flatten()
            .filter(|&(_, _, l)| l != 0)
            .collect()
    }
    fn acts(
        &self,
        input: &<bool as PackedElement<T::Shape>>::Array,
        value: Option<bool>,
    ) -> <bool as PackedElement<T::Shape>>::Array {
        let cur_act = self.bma(input);
        let value_sign = value.unwrap_or(false);
        let value_mask = value.is_some();

        if (cur_act + 2) <= Self::THRESHOLD {
            <<bool as PackedElement<T::Shape>>::Array as PackedArray>::blit(false)
        } else if cur_act > (Self::THRESHOLD + 2) {
            <<bool as PackedElement<T::Shape>>::Array as PackedArray>::blit(true)
        } else if (cur_act + 1) == Self::THRESHOLD {
            // go up 2 to activate
            if value_mask {
                if value_sign {
                    self.0.map(&input, |trit, input| {
                        ((!input ^ trit.sign()) & trit.mask()) & !trit.sign()
                    })
                } else {
                    self.0.map(&input, |trit, input| {
                        ((!input ^ trit.sign()) & trit.mask()) & trit.sign()
                    })
                }
            } else {
                <<bool as PackedElement<T::Shape>>::Array as PackedArray>::blit(false)
            }
        } else if cur_act == (Self::THRESHOLD + 2) {
            // go down 2 to deactivate
            if value_mask {
                if value_sign {
                    self.0.map(&input, |trit, input| {
                        !(((input ^ trit.sign()) & trit.mask()) & !trit.sign())
                    })
                } else {
                    self.0.map(&input, |trit, input| {
                        !(((input ^ trit.sign()) & trit.mask()) & trit.sign())
                    })
                }
            } else {
                <<bool as PackedElement<T::Shape>>::Array as PackedArray>::blit(true)
            }
        } else if cur_act == Self::THRESHOLD {
            // go up 1 to activate
            if value_mask {
                if value_sign {
                    self.0.map(&input, |trit, input| {
                        !((input ^ trit.sign()) & trit.mask()) & !input
                    })
                } else {
                    self.0.map(&input, |trit, input| {
                        !((input ^ trit.sign()) & trit.mask()) & input
                    })
                }
            } else {
                self.0
                    .map(&input, |trit, input| (input ^ !trit.sign()) & trit.mask())
            }
        } else if cur_act == (Self::THRESHOLD + 1) {
            // go down 1 to deactivate
            if value_mask {
                if value_sign {
                    self.0.map(&input, |trit, input| {
                        !(input & !(trit.sign() & trit.mask()))
                    })
                } else {
                    self.0.map(&input, |trit, input| {
                        (!((input ^ trit.sign()) & trit.mask()) & trit.mask()) | input
                    })
                }
            } else {
                self.0
                    .map(&input, |trit, input| !((input ^ trit.sign()) & trit.mask()))
            }
        } else {
            unreachable!();
        }
    }
    fn input_acts(
        &self,
        input: &<bool as PackedElement<Self::Shape>>::Array,
    ) -> <bool as PackedElement<Self::Shape>>::Array {
        let cur_act = self.bma(input);

        if (cur_act + 2) <= Self::THRESHOLD {
            <<bool as PackedElement<T::Shape>>::Array as PackedArray>::blit(false)
        } else if cur_act > (Self::THRESHOLD + 2) {
            <<bool as PackedElement<T::Shape>>::Array as PackedArray>::blit(true)
        } else if ((cur_act + 1) == Self::THRESHOLD) | (cur_act == Self::THRESHOLD) {
            self.0
                .map(&input, |trit, input| (!input ^ trit.sign()) & trit.mask())
        } else if (cur_act == (Self::THRESHOLD + 2)) | (cur_act == (Self::THRESHOLD + 1)) {
            self.0
                .map(&input, |trit, input| !((input ^ trit.sign()) & trit.mask()))
        } else {
            unreachable!();
        }
    }
}

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

pub trait PackedIndexMap<O>
where
    Self: Shape + Sized,
    O: PackedElement<Self>,
{
    fn index_map<F: Fn(Self::Index) -> O>(map_fn: F) -> <O as PackedElement<Self>>::Array;
}

impl<S: Shape + PackedIndexMapInner<W, ()>, W: Weight> PackedIndexMap<W> for S
where
    W: PackedElement<Self>,
    <S as Element<()>>::Array: Shape<Index = S::Index>,
{
    fn index_map<F: Fn(Self::Index) -> W>(map_fn: F) -> <W as PackedElement<Self>>::Array {
        <S as PackedIndexMapInner<W, ()>>::index_map_inner((), map_fn)
    }
}

pub trait PackedIndexMapInner<O: Weight, W: Shape>
where
    Self: Shape + Sized,
    O: PackedElement<Self>,
    Self: Element<W>,
    <Self as Element<W>>::Array: Shape,
{
    fn index_map_inner<F: Fn(<<Self as Element<W>>::Array as Shape>::Index) -> O>(
        outer_index: W::Index,
        map_fn: F,
    ) -> <O as PackedElement<Self>>::Array;
}

impl<
        O: PackedElement<S> + Weight,
        W: Shape,
        S: Shape + PackedIndexMapInner<O, <[(); L] as Element<W>>::Array>,
        const L: usize,
    > PackedIndexMapInner<O, W> for [S; L]
where
    [S; L]: Element<W, Array = <S as Element<<[(); L] as Element<W>>::Array>>::Array>,
    <[S; L] as Element<W>>::Array: Shape,
    [(); L]: Element<W>,
    <[(); L] as Element<W>>::Array: Shape,
    <S as Element<<[(); L] as Element<W>>::Array>>::Array: Shape,
    [<O as PackedElement<S>>::Array; L]: LongDefault,
    (u8, ()): Wrap<W::Index, Wrapped = <<[(); L] as Element<W>>::Array as Shape>::Index>,
    W::Index: Copy,
{
    fn index_map_inner<F: Fn(<<[S; L] as Element<W>>::Array as Shape>::Index) -> O>(
        outer_index: W::Index,
        map_fn: F,
    ) -> [<O as PackedElement<S>>::Array; L] {
        let mut target = <[<O as PackedElement<S>>::Array; L]>::long_default();
        for i in 0..L {
            target[i] =
                <S as PackedIndexMapInner<O, <[(); L] as Element<W>>::Array>>::index_map_inner(
                    (i as u8, ()).wrap(outer_index),
                    &map_fn,
                );
        }
        target
    }
}

pub trait PackedMap<I, O>
where
    Self: Shape + Sized,
    I: Element<Self> + Sized,
    O: PackedElement<Self>,
{
    fn map<F: Fn(&I) -> O>(
        input: &<I as Element<Self>>::Array,
        map_fn: F,
    ) -> <O as PackedElement<Self>>::Array;
}

impl<S: Shape + PackedMap<I, O>, I: Element<S>, O: Weight + PackedElement<S>, const L: usize>
    PackedMap<I, O> for [S; L]
where
    <O as PackedElement<Self>>::Array: LongDefault,
{
    fn map<F: Fn(&I) -> O>(
        input: &[<I as Element<S>>::Array; L],
        map_fn: F,
    ) -> <O as PackedElement<Self>>::Array {
        let mut target = <O as PackedElement<Self>>::Array::long_default();
        for i in 0..L {
            target[i] = <S as PackedMap<I, O>>::map(&input[i], &map_fn);
        }
        target
    }
}

pub trait BitMapPack<E: Element<Self::Shape>>
where
    Self: Activations,
{
    fn bit_map_pack<F: Fn(&E) -> bool>(
        input: &<E as Element<Self::Shape>>::Array,
        map_fn: F,
    ) -> Self;
}

pub trait WBBZM
where
    Self: PackedArray,
    bool: PackedElement<Self::Shape>,
{
    fn map<F: Fn(Self::Word, Self::UWord) -> Self::UWord>(
        &self,
        rhs: &<bool as PackedElement<Self::Shape>>::Array,
        map_fn: F,
    ) -> <bool as PackedElement<Self::Shape>>::Array;
}

impl<T: WBBZM + Copy, const L: usize> WBBZM for [T; L]
where
    T: PackedArray,
    bool: PackedElement<T::Shape>,
    [<bool as PackedElement<T::Shape>>::Array; L]: LongDefault,
{
    fn map<F: Fn(Self::Word, Self::UWord) -> Self::UWord>(
        &self,
        rhs: &[<bool as PackedElement<T::Shape>>::Array; L],
        map_fn: F,
    ) -> [<bool as PackedElement<T::Shape>>::Array; L] {
        let mut target = <[<bool as PackedElement<T::Shape>>::Array; L]>::long_default();
        for i in 0..L {
            target[i] = self[i].map(&rhs[i], &map_fn);
        }
        target
    }
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

pub trait BitWord {
    type Word;
    fn sign(self) -> Self::Word;
}

pub trait TritWord {
    type Word;
    fn sign(self) -> Self::Word;
    fn mask(self) -> Self::Word;
}

pub trait QuatWord {
    type Word;
    fn sign(self) -> Self::Word;
    fn magn(self) -> Self::Word;
}

pub trait PackedWord {
    type Weight;
    fn blit(weight: Self::Weight) -> Self;
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

/// Masked Hamming distance between two collections of bits of the same shape.
pub trait MaskedDistance
where
    Self: PackedArray<Weight = Option<bool>>,
    bool: PackedElement<Self::Shape>,
{
    /// Returns the number of bits that are different and mask bit is set.
    /// Note that this is not adjusted for mask count. You should probably add mask_zeros() / 2 to the result.
    fn masked_distance(&self, bits: &<bool as PackedElement<Self::Shape>>::Array) -> u32;
    fn mask_zeros(&self) -> u32;
}

impl<T: MaskedDistance + Copy + PackedArray<Weight = Option<bool>>, const L: usize> MaskedDistance
    for [T; L]
where
    bool: PackedElement<T::Shape>,
{
    fn masked_distance(&self, bits: &[<bool as PackedElement<T::Shape>>::Array; L]) -> u32 {
        let mut sum = 0_u32;
        for i in 0..L {
            sum += self[i].masked_distance(&bits[i]);
        }
        sum
    }
    fn mask_zeros(&self) -> u32 {
        self.iter().map(|x| x.mask_zeros()).sum()
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

macro_rules! impl_long_default_for_type {
    ($type:ty) => {
        impl LongDefault for $type {
            fn long_default() -> $type {
                <$type as Default>::default()
            }
        }
    };
}

impl_long_default_for_type!(bool);
impl_long_default_for_type!(Option<bool>);
impl_long_default_for_type!((bool, bool));
impl_long_default_for_type!(Option<(bool, bool)>);

macro_rules! for_uints {
    ($q_type:ident, $t_type:ident, $b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        impl_long_default_for_type!($u_type);

        /// A word of trits. The 0 element is the signs, the 1 element is the mask.
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

        impl PackedElement<[(); $len]> for Option<bool> {
            type Array = $t_type;
        }

        impl PackedWord for $t_type {
            type Weight = Option<bool>;
            fn blit(weight: Option<bool>) -> $t_type {
                $t_type(
                    (Wrapping(0) - Wrapping(weight.unwrap_or(false) as $u_type)).0,
                    (Wrapping(0) - Wrapping(weight.is_some() as $u_type)).0,
                )
            }
        }

        impl PackedArray for $t_type {
            type Weight = Option<bool>;
            type Word = $t_type;
            type UWord = $u_type;
            type Shape = [(); $len];
            fn blit(val: Option<bool>) -> $t_type {
                $t_type(
                    (Wrapping(0) - Wrapping(val.unwrap_or(false) as $u_type)).0,
                    (Wrapping(0) - Wrapping(val.is_some() as $u_type)).0,
                )
            }
            fn get_weight(&self, (b, _): (u8, ())) -> Option<bool> {
                self.get_trit(b as usize)
            }
            fn set_weight_in_place(&mut self, (b, _): (u8, ()), value: Option<bool>) {
                self.set_trit_in_place(b as usize, value);
            }
        }
        impl WBBZM for $t_type {
            fn map<F: Fn($t_type, $u_type) -> $u_type>(&self, rhs: &$b_type, map_fn: F) -> $b_type {
                $b_type(map_fn(*self, rhs.0))
            }
        }

        impl TritWord for $t_type {
            type Word = $u_type;
            #[inline(always)]
            fn sign(self) -> $u_type {
                self.0
            }
            #[inline(always)]
            fn mask(self) -> $u_type {
                self.1
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
                self.0 &= !(1 << index);
                self.0 |= ((value as $u_type) << index);
            }
        }
        impl PackedWord for $b_type {
            type Weight = bool;
            fn blit(weight: bool) -> $b_type {
                $b_type((Wrapping(0) - Wrapping(weight as $u_type)).0)
            }
        }

        impl BitWord for $b_type {
            type Word = $u_type;
            #[inline(always)]
            fn sign(self) -> $u_type {
                self.0
            }
        }

        impl WBBZM for $b_type {
            fn map<F: Fn($b_type, $u_type) -> $u_type>(&self, rhs: &$b_type, map_fn: F) -> $b_type {
                $b_type(map_fn(*self, rhs.0))
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
            type Word = $b_type;
            type UWord = $u_type;
            type Shape = [(); $len];
            fn blit(val: bool) -> $b_type {
                $b_type((Wrapping(0) - Wrapping(val as $u_type)).0)
            }
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

        impl QuatWord for $q_type {
            type Word = $u_type;
            #[inline(always)]
            fn sign(self) -> $u_type {
                self.0
            }
            #[inline(always)]
            fn magn(self) -> $u_type {
                self.1
            }
        }

        impl PackedArray for $q_type {
            type Weight = (bool, bool);
            type Word = $q_type;
            type UWord = $u_type;
            type Shape = [(); $len];
            fn blit(value: (bool, bool)) -> $q_type {
                $q_type(
                    (Wrapping(0) - Wrapping(value.0 as $u_type)).0,
                    (Wrapping(0) - Wrapping(value.1 as $u_type)).0,
                )
            }
            fn get_weight(&self, (index, _): (u8, ())) -> (bool, bool) {
                (((self.0 >> index) & 1) == 1, ((self.1 >> index) & 1) == 1)
            }
            fn set_weight_in_place(&mut self, (index, _): (u8, ()), value: (bool, bool)) {
                self.0 &= !(1 << index);
                self.0 |= ((value.0 as $u_type) << index);

                self.1 &= !(1 << index);
                self.1 |= ((value.1 as $u_type) << index);
            }
        }
        impl WBBZM for $q_type {
            fn map<F: Fn($q_type, $u_type) -> $u_type>(&self, rhs: &$b_type, map_fn: F) -> $b_type {
                $b_type(map_fn(*self, rhs.0))
            }
        }

        impl PackedWord for $q_type {
            type Weight = (bool, bool);
            fn blit(weight: (bool, bool)) -> $q_type {
                $q_type(
                    (Wrapping(0) - Wrapping(weight.0 as $u_type)).0,
                    (Wrapping(0) - Wrapping(weight.1 as $u_type)).0,
                )
            }
        }

        impl<I: Element<[(); $len], Array = [I; $len]>> PackedMap<I, bool> for [(); $len] {
            fn map<F: Fn(&I) -> bool>(input: &[I; $len], map_fn: F) -> $b_type {
                let mut target = 0;
                for b in 0..$len {
                    target |= (map_fn(&input[b]) as $u_type) << b;
                }
                $b_type(target)
            }
        }

        impl<I: Element<[(); $len], Array = [I; $len]>> PackedMap<I, Option<bool>> for [(); $len] {
            fn map<F: Fn(&I) -> Option<bool>>(input: &[I; $len], map_fn: F) -> $t_type {
                let mut sign = 0;
                let mut mask = 0;
                for b in 0..$len {
                    let trit = map_fn(&input[b]);
                    sign |= (trit.unwrap_or(false) as $u_type) << b;
                    mask |= (trit.is_some() as $u_type) << b;
                }
                $t_type(sign, mask)
            }
        }

        impl<I: Element<[(); $len], Array = [I; $len]>> PackedMap<I, (bool, bool)> for [(); $len] {
            fn map<F: Fn(&I) -> (bool, bool)>(input: &[I; $len], map_fn: F) -> $q_type {
                let mut sign = 0;
                let mut magn = 0;
                for b in 0..$len {
                    let (s, m) = map_fn(&input[b]);
                    sign |= (s as $u_type) << b;
                    magn |= (m as $u_type) << b;
                }
                $q_type(sign, magn)
            }
        }

        impl<W: Shape> PackedIndexMapInner<bool, W> for [(); $len]
        where
            [(); $len]: Element<W>,
            <[(); $len] as Element<W>>::Array: Shape,
            (u8, ()): Wrap<W::Index, Wrapped = <<[(); $len] as Element<W>>::Array as Shape>::Index>,
        {
            fn index_map_inner<
                F: Fn(<<[(); $len] as Element<W>>::Array as Shape>::Index) -> bool,
            >(
                outer_index: W::Index,
                map_fn: F,
            ) -> $b_type {
                let mut target = 0;
                for b in 0..$len {
                    target |= (map_fn((b as u8, ()).wrap(outer_index)) as $u_type) << b;
                }
                $b_type(target)
            }
        }

        impl MaskedDistance for $t_type {
            fn masked_distance(&self, bits: &$b_type) -> u32 {
                ((bits.0 ^ self.0) & self.1).count_ones()
            }
            fn mask_zeros(&self) -> u32 {
                self.1.count_zeros()
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
    use super::{
        b128, b16, b32, b8, t128, t16, t32, t8, MaskedDistance, PackedElement, Weight, WeightArray,
    };
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;

    macro_rules! test_bma {
        ($name:ident, $input:ty, $weights:ty, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: $input = rng.gen();
                    let weights = <$weights>::rand(&mut rng);

                    let sum = weights.bma(&inputs);
                    let true_sum = weights.bma_slow(&inputs);
                    assert_eq!(sum, true_sum);
                })
            }
        };
    }

    macro_rules! test_acts {
        ($name:ident, $input:ty, $weights:ty, $weight:ty, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: $input = rng.gen();
                    let weights = <$weights>::rand(&mut rng);

                    for val in <$weight>::states() {
                        let acts = weights.acts(&inputs, val);
                        let true_acts = weights.acts_slow(&inputs, val);
                        assert_eq!(acts, true_acts);
                    }
                })
            }
        };
    }

    macro_rules! test_input_acts {
        ($name:ident, $input:ty, $weights:ty, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: $input = rng.gen();
                    let weights = <$weights>::rand(&mut rng);

                    let acts = weights.input_acts(&inputs);
                    let true_acts = weights.input_acts_slow(&inputs);
                    assert_eq!(acts, true_acts);
                })
            }
        };
    }

    macro_rules! test_loss_deltas {
        ($name:ident, $input:ty, $weights:ty, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: $input = rng.gen();
                    let weights = <$weights>::rand(&mut rng);

                    let null_act = weights.bma(&inputs);
                    let mut losses = weights.loss_deltas(&inputs, |x| x as i64 - null_act as i64);
                    losses.sort();
                    let mut true_losses =
                        weights.loss_deltas_slow(&inputs, |x| x as i64 - null_act as i64);
                    true_losses.sort();
                    assert_eq!(losses, true_losses);
                })
            }
        };
    }

    macro_rules! test_group {
        ($name:ident, $input:ty, $weights:ty, $weight:ty, $n_iters:expr) => {
            #[cfg(test)]
            mod $name {
                use crate::bits::{
                    b128, b16, b32, b64, b8, t128, t16, t32, t64, t8, MaskedDistance,
                    PackedElement, Weight, WeightArray,
                };
                use rand::Rng;
                use rand::SeedableRng;
                use rand_hc::Hc128Rng;
                extern crate test;

                test_bma!(bma, $input, $weights, $n_iters);
                test_acts!(acts, $input, $weights, $weight, $n_iters);
                test_input_acts!(input_acts, $input, $weights, $n_iters);
                test_loss_deltas!(loss_deltas, $input, $weights, $n_iters);
            }
        };
    }

    test_group!(test_b8, b8, b8, bool, 10_000);
    test_group!(test_b16, b16, b16, bool, 10_000);
    test_group!(test_b32, b32, b32, bool, 10_000);
    test_group!(test_b64, b64, b64, bool, 1_000);
    test_group!(test_b128, b128, b128, bool, 1_000);

    test_group!(test_b8x1, [b8; 1], [b8; 1], bool, 10_000);
    test_group!(test_b8x2, [b8; 2], [b8; 2], bool, 10_000);
    test_group!(test_b8x3, [b8; 3], [b8; 3], bool, 10_000);
    test_group!(test_b8x4, [b8; 4], [b8; 4], bool, 10_000);

    test_group!(
        test_b8x1x2x3,
        [[[b8; 1]; 2]; 3],
        [[[b8; 1]; 2]; 3],
        bool,
        10_000
    );

    test_group!(test_t8, b8, (t8, u32), Option<bool>, 10_000);
    test_group!(test_t16, b16, (t16, u32), Option<bool>, 10_000);
    test_group!(test_t32, b32, (t32, u32), Option<bool>, 10_000);
    test_group!(test_t64, b64, (t64, u32), Option<bool>, 1_000);
    test_group!(test_t128, b128, (t128, u32), Option<bool>, 1_000);

    test_group!(test_t8x1, [b8; 1], ([t8; 1], u32), Option<bool>, 10_000);
    test_group!(test_t8x2, [b8; 2], ([t8; 2], u32), Option<bool>, 10_000);
    test_group!(test_t8x3, [b8; 3], ([t8; 3], u32), Option<bool>, 10_000);
    test_group!(test_t8x4, [b8; 4], ([t8; 4], u32), Option<bool>, 10_000);

    test_group!(
        test_t8x1x2x3,
        [[[b8; 1]; 2]; 3],
        ([[[t8; 1]; 2]; 3], u32),
        Option<bool>,
        10_000
    );
}
