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
    fn bma(self, input: bool) -> u32;
}

impl Weight for bool {
    const RANGE: u32 = 3;
    const N: usize = 2;
    fn states() -> iter::Cloned<slice::Iter<'static, bool>> {
        [true, false].iter().cloned()
    }
    fn bma(self, input: bool) -> u32 {
        (self ^ input) as u32
    }
}

impl Weight for Option<bool> {
    const RANGE: u32 = 3;
    const N: usize = 3;
    fn states() -> iter::Cloned<slice::Iter<'static, Option<bool>>> {
        [Some(true), None, Some(false)].iter().cloned()
    }
    fn bma(self, input: bool) -> u32 {
        if let Some(sign) = self {
            (sign ^ input) as u32 * 2
        } else {
            1
        }
    }
}

impl Weight for (bool, bool) {
    const RANGE: u32 = 4;
    const N: usize = 4;
    fn states() -> iter::Cloned<slice::Iter<'static, (bool, bool)>> {
        [(true, true), (true, false), (false, false), (false, true)]
            .iter()
            .cloned()
    }
    fn bma(self, input: bool) -> u32 {
        if self.1 {
            if input ^ self.0 {
                3
            } else {
                0
            }
        } else {
            if input ^ self.0 {
                2
            } else {
                1
            }
        }
    }
}

impl Weight for Option<(bool, bool)> {
    const RANGE: u32 = 5;
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
    fn bma(self, input: bool) -> u32 {
        if let Some(weight) = self {
            if weight.1 {
                if input ^ weight.0 {
                    4
                } else {
                    0
                }
            } else {
                if input ^ weight.0 {
                    3
                } else {
                    1
                }
            }
        } else {
            2
        }
    }
}

pub trait ConstructWeightArray<S> {
    type WeightArray;
}

impl<S: Shape> ConstructWeightArray<S> for bool
where
    bool: PackedElement<S>,
{
    type WeightArray = <bool as PackedElement<S>>::Array;
}

impl<S: Shape> ConstructWeightArray<S> for Option<bool>
where
    Option<bool>: PackedElement<S>,
{
    type WeightArray = (<Option<bool> as PackedElement<S>>::Array, u32);
}

impl<S: Shape> ConstructWeightArray<S> for (bool, bool)
where
    (bool, bool): PackedElement<S>,
{
    type WeightArray = <(bool, bool) as PackedElement<S>>::Array;
}

impl<S: Shape> ConstructWeightArray<S> for Option<(bool, bool)>
where
    Option<(bool, bool)>: PackedElement<S>,
{
    type WeightArray = (<Option<(bool, bool)> as PackedElement<S>>::Array, u32);
}

pub trait WeightArray<S: Shape, W: Weight>
where
    <W as ConstructWeightArray<S>>::WeightArray: Copy,
    bool: PackedElement<S>,
    S: Shape + PackedIndexMap<bool>,
    W: 'static + Weight + ConstructWeightArray<S>,
    <bool as PackedElement<S>>::Array: PackedArray<Weight = bool, Shape = S> + Copy,
{
    const MAX: u32;
    const THRESHOLD: u32;
    fn rand<R: Rng>(rng: &mut R) -> <W as ConstructWeightArray<S>>::WeightArray;
    fn get_weight(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        index: <S as Shape>::Index,
    ) -> W;
    /// multiply accumulate
    fn bma(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
    ) -> u32;
    fn bma_slow(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
    ) -> u32 {
        <S as Shape>::indices()
            .map(|i| Self::get_weight(weights, i).bma(input.get_weight(i)))
            .sum()
    }
    /// thresholded activation
    fn act(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
    ) -> bool {
        Self::bma(weights, input) > Self::THRESHOLD
    }
    fn act_is_alive(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
    ) -> bool {
        let act = Self::bma(weights, input);
        ((act + W::RANGE) > Self::THRESHOLD) | ((Self::THRESHOLD + W::RANGE) < act)
    }
    fn mutate_in_place(
        weights: &mut <W as ConstructWeightArray<S>>::WeightArray,
        index: <S as Shape>::Index,
        value: W,
    );
    fn mutate(
        mut weights: <W as ConstructWeightArray<S>>::WeightArray,
        index: <S as Shape>::Index,
        value: W,
    ) -> <W as ConstructWeightArray<S>>::WeightArray {
        Self::mutate_in_place(&mut weights, index, value);
        weights
    }
    /// loss delta if we were to set each of the elements to each different value. Is permited to prune null deltas.
    fn loss_deltas<F: Fn(u32) -> i64>(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<S as Shape>::Index, W, i64)>;
    /// Does the same thing as loss_deltas but is a lot slower.
    fn loss_deltas_slow<F: Fn(u32) -> i64>(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<S as Shape>::Index, W, i64)> {
        <W as Weight>::states()
            .map(|value| {
                <S as Shape>::indices()
                    .map(|index| {
                        (
                            index,
                            value,
                            loss_delta_fn(Self::bma(&Self::mutate(*weights, index, value), &input)),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .filter(|(_, _, l)| l.abs() as u64 > threshold)
            .collect()
    }
    /// Acts if we were to set each element to value
    fn acts(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
        value: W,
    ) -> <bool as PackedElement<S>>::Array;
    /// acts_slow does the same thing as grads, but it is a lot slower.
    fn acts_slow(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
        value: W,
    ) -> <bool as PackedElement<S>>::Array {
        <S as PackedIndexMap<bool>>::index_map(|index| {
            Self::act(&Self::mutate(*weights, index, value), input)
        })
    }
    /// Act if each input is flipped.
    fn input_acts(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
    ) -> <bool as PackedElement<S>>::Array;
    /// input_acts_slow does the same thing as input_acts, but it is a lot slower.
    fn input_acts_slow(
        weights: &<W as ConstructWeightArray<S>>::WeightArray,
        input: &<bool as PackedElement<S>>::Array,
    ) -> <bool as PackedElement<S>>::Array {
        <S as PackedIndexMap<bool>>::index_map(|index| {
            Self::act(weights, &input.set_weight(index, !input.get_weight(index)))
        })
    }
}

impl<S: Shape> WeightArray<S, bool> for ()
where
    <bool as PackedElement<S>>::Array:
        Copy + WBBZM + PackedArray<Weight = bool> + Distance + std::fmt::Debug,
    bool: PackedElement<S>,
    Standard: Distribution<<bool as PackedElement<S>>::Array>,
    S: PackedIndexMap<bool>,
    <<bool as PackedElement<S>>::Array as PackedArray>::Word:
        BitWord<Word = <<bool as PackedElement<S>>::Array as PackedArray>::UWord>,
    <S as Shape>::Index: std::fmt::Debug,
    <S as Shape>::IndexIter: Iterator<Item = <S as Shape>::Index>,
    <bool as PackedElement<S>>::Array: PackedArray<Weight = bool, Shape = S> + Copy,
    <S as Shape>::Index: Element<<<bool as PackedElement<S>>::Array as PackedArray>::Shape>,
    bool: ConstructWeightArray<S, WeightArray = <bool as PackedElement<S>>::Array>,
{
    const MAX: u32 = S::N as u32;
    const THRESHOLD: u32 = S::N as u32 / 2;
    fn rand<R: Rng>(rng: &mut R) -> <bool as PackedElement<S>>::Array {
        rng.gen()
    }
    fn get_weight(weights: &<bool as PackedElement<S>>::Array, index: <S as Shape>::Index) -> bool {
        <<bool as PackedElement<S>>::Array as PackedArray>::get_weight(weights, index)
    }
    fn bma(
        weights: &<bool as PackedElement<S>>::Array,
        input: &<bool as PackedElement<S>>::Array,
    ) -> u32 {
        weights.distance(input)
    }
    fn mutate_in_place(
        weights: &mut <bool as PackedElement<S>>::Array,
        index: <S as Shape>::Index,
        value: bool,
    ) {
        weights.set_weight_in_place(index, value)
    }
    fn loss_deltas<F: Fn(u32) -> i64>(
        weights: &<bool as PackedElement<S>>::Array,
        input: &<bool as PackedElement<S>>::Array,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<S as Shape>::Index, bool, i64)> {
        let cur_act = Self::bma(weights, input);
        let deltas = [
            loss_delta_fn(cur_act.saturating_add(1)),
            loss_delta_fn(cur_act.saturating_sub(1)),
        ];
        if deltas.iter().find(|d| d.abs() as u64 > threshold).is_some() {
            <S as Shape>::indices()
                .map(|i| {
                    let weight = Self::get_weight(weights, i);
                    let input = input.get_weight(i);
                    (i, !weight, deltas[(weight ^ input) as usize])
                })
                .filter(|(_, _, l)| l.abs() as u64 > threshold)
                .collect()
        } else {
            vec![]
        }
    }
    fn acts(
        weights: &<bool as PackedElement<S>>::Array,
        input: &<bool as PackedElement<S>>::Array,
        value: bool,
    ) -> <bool as PackedElement<S>>::Array {
        let cur_act = Self::bma(weights, input);
        if cur_act == Self::THRESHOLD {
            // we need to go up by one to flip
            if value {
                weights.map(input, |weight, input| {
                    !(weight.sign() ^ input) & !weight.sign()
                })
            } else {
                weights.map(input, |weight, input| {
                    !(weight.sign() ^ input) & weight.sign()
                })
            }
        } else if cur_act == (Self::THRESHOLD + 1) {
            // we need to go down by one to flip
            if value {
                weights.map(input, |weight, input| {
                    !(weight.sign() ^ input) | weight.sign()
                })
            } else {
                weights.map(input, |weight, input| {
                    !(weight.sign() ^ input) | !weight.sign()
                })
            }
        } else {
            <<bool as PackedElement<S>>::Array as PackedArray>::blit(cur_act > Self::THRESHOLD)
        }
    }
    fn input_acts(
        weights: &<bool as PackedElement<S>>::Array,
        input: &<bool as PackedElement<S>>::Array,
    ) -> <bool as PackedElement<S>>::Array {
        let cur_act = Self::bma(weights, input);
        if (cur_act == Self::THRESHOLD) | (cur_act == (Self::THRESHOLD + 1)) {
            weights.map(input, |weight, input| weight.sign() ^ !input)
        } else {
            <<bool as PackedElement<S>>::Array as PackedArray>::blit(cur_act > Self::THRESHOLD)
        }
    }
}

impl<S: Shape> WeightArray<S, Option<bool>> for ()
where
    bool: PackedElement<S>,
    Option<bool>: PackedElement<S>,
    <Option<bool> as PackedElement<S>>::Array:
        PackedArray<Weight = Option<bool>, Shape = S> + MaskedDistance + Copy + WBBZM,
    Standard: Distribution<<Option<bool> as PackedElement<S>>::Array>,
    <<Option<bool> as PackedElement<S>>::Array as PackedArray>::Word:
        TritWord<Word = <<Option<bool> as PackedElement<S>>::Array as PackedArray>::UWord>,
    S: PackedIndexMap<bool>,
    <S as Shape>::IndexIter: Iterator<Item = <S as Shape>::Index>,
    <bool as PackedElement<S>>::Array: PackedArray<Weight = bool, Shape = S> + Copy,
    Option<bool>:
        ConstructWeightArray<S, WeightArray = (<Option<bool> as PackedElement<S>>::Array, u32)>,
{
    const MAX: u32 = <S as Shape>::N as u32 * 2;
    const THRESHOLD: u32 = S::N as u32;
    fn rand<R: Rng>(rng: &mut R) -> (<Option<bool> as PackedElement<S>>::Array, u32) {
        let weights: <Option<bool> as PackedElement<S>>::Array = rng.gen();
        (weights, weights.mask_zeros())
    }
    fn get_weight(
        (trits, _): &(<Option<bool> as PackedElement<S>>::Array, u32),
        index: <S as Shape>::Index,
    ) -> Option<bool> {
        trits.get_weight(index)
    }
    fn bma(
        (trits, zeros): &(<Option<bool> as PackedElement<S>>::Array, u32),
        input: &<bool as PackedElement<S>>::Array,
    ) -> u32 {
        trits.masked_distance(input) * 2 + zeros
    }
    fn mutate_in_place(
        weights: &mut (<Option<bool> as PackedElement<S>>::Array, u32),
        index: <S as Shape>::Index,
        value: Option<bool>,
    ) {
        weights.0.set_weight_in_place(index, value);
        weights.1 = weights.0.mask_zeros();
    }
    fn loss_deltas<F: Fn(u32) -> i64>(
        weights: &(<Option<bool> as PackedElement<S>>::Array, u32),
        input: &<bool as PackedElement<S>>::Array,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<S as Shape>::Index, Option<bool>, i64)> {
        let cur_act = Self::bma(weights, input);
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
        if deltas
            .iter()
            .flatten()
            .find(|d| d.abs() as u64 > threshold)
            .is_some()
        {
            <S as Shape>::indices()
                .map(|i| {
                    let input = input.get_weight(i);
                    if let Some(sign) = weights.0.get_weight(i) {
                        if sign {
                            iter::once((i, Some(false), deltas[1][input as usize]))
                                .chain(iter::once((i, None, deltas[0][input as usize])))
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
                .filter(|(_, _, l)| l.abs() as u64 > threshold)
                .collect()
        } else {
            vec![]
        }
    }
    fn acts(
        weights: &(<Option<bool> as PackedElement<S>>::Array, u32),
        input: &<bool as PackedElement<S>>::Array,
        value: Option<bool>,
    ) -> <bool as PackedElement<S>>::Array {
        let cur_act = Self::bma(weights, input);
        let value_sign = value.unwrap_or(false);
        let value_mask = value.is_some();

        if (cur_act + 2) <= Self::THRESHOLD {
            <<bool as PackedElement<S>>::Array as PackedArray>::blit(false)
        } else if cur_act > (Self::THRESHOLD + 2) {
            <<bool as PackedElement<S>>::Array as PackedArray>::blit(true)
        } else if (cur_act + 1) == Self::THRESHOLD {
            // go up 2 to activate
            if value_mask {
                if value_sign {
                    weights.0.map(&input, |trit, input| {
                        ((!input ^ trit.sign()) & trit.mask()) & !trit.sign()
                    })
                } else {
                    weights.0.map(&input, |trit, input| {
                        ((!input ^ trit.sign()) & trit.mask()) & trit.sign()
                    })
                }
            } else {
                <<bool as PackedElement<S>>::Array as PackedArray>::blit(false)
            }
        } else if cur_act == (Self::THRESHOLD + 2) {
            // go down 2 to deactivate
            if value_mask {
                if value_sign {
                    weights.0.map(&input, |trit, input| {
                        !(((input ^ trit.sign()) & trit.mask()) & !trit.sign())
                    })
                } else {
                    weights.0.map(&input, |trit, input| {
                        !(((input ^ trit.sign()) & trit.mask()) & trit.sign())
                    })
                }
            } else {
                <<bool as PackedElement<S>>::Array as PackedArray>::blit(true)
            }
        } else if cur_act == Self::THRESHOLD {
            // go up 1 to activate
            if value_mask {
                if value_sign {
                    weights.0.map(&input, |trit, input| {
                        !((input ^ trit.sign()) & trit.mask()) & !input
                    })
                } else {
                    weights.0.map(&input, |trit, input| {
                        !((input ^ trit.sign()) & trit.mask()) & input
                    })
                }
            } else {
                weights
                    .0
                    .map(&input, |trit, input| (input ^ !trit.sign()) & trit.mask())
            }
        } else if cur_act == (Self::THRESHOLD + 1) {
            // go down 1 to deactivate
            if value_mask {
                if value_sign {
                    weights.0.map(&input, |trit, input| {
                        !(input & !(trit.sign() & trit.mask()))
                    })
                } else {
                    weights.0.map(&input, |trit, input| {
                        (!((input ^ trit.sign()) & trit.mask()) & trit.mask()) | input
                    })
                }
            } else {
                weights
                    .0
                    .map(&input, |trit, input| !((input ^ trit.sign()) & trit.mask()))
            }
        } else {
            unreachable!();
        }
    }
    fn input_acts(
        weights: &(<Option<bool> as PackedElement<S>>::Array, u32),
        input: &<bool as PackedElement<S>>::Array,
    ) -> <bool as PackedElement<S>>::Array {
        let cur_act = Self::bma(weights, input);

        if (cur_act + 2) <= Self::THRESHOLD {
            <<bool as PackedElement<S>>::Array as PackedArray>::blit(false)
        } else if cur_act > (Self::THRESHOLD + 2) {
            <<bool as PackedElement<S>>::Array as PackedArray>::blit(true)
        } else if ((cur_act + 1) == Self::THRESHOLD) | (cur_act == Self::THRESHOLD) {
            weights
                .0
                .map(&input, |trit, input| (!input ^ trit.sign()) & trit.mask())
        } else if (cur_act == (Self::THRESHOLD + 2)) | (cur_act == (Self::THRESHOLD + 1)) {
            weights
                .0
                .map(&input, |trit, input| !((input ^ trit.sign()) & trit.mask()))
        } else {
            unreachable!();
        }
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

/// Hamming distance between two collections of bits of the same shape.
pub trait Distance {
    /// Returns the number of bits that are different
    fn distance(&self, rhs: &Self) -> u32;
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

/// Hamming distance between quat an bit arrays
pub trait QuatDistance
where
    Self: PackedArray<Weight = (bool, bool)>,
    bool: PackedElement<Self::Shape>,
{
    fn quat_distance(&self, bits: &<bool as PackedElement<Self::Shape>>::Array) -> u32;
}

impl<T: QuatDistance + Copy + PackedArray<Weight = (bool, bool)>, const L: usize> QuatDistance
    for [T; L]
where
    bool: PackedElement<T::Shape>,
{
    fn quat_distance(&self, bits: &[<bool as PackedElement<T::Shape>>::Array; L]) -> u32 {
        let mut sum = 0_u32;
        for i in 0..L {
            sum += self[i].quat_distance(&bits[i]);
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

        impl QuatDistance for $q_type {
            fn quat_distance(&self, bits: &$b_type) -> u32 {
                ((bits.0 ^ self.0) & !self.1).count_ones()
                    + ((bits.0 ^ self.0) & self.1).count_ones() * 2
                    + ((bits.0 ^ !self.0) & self.1).count_ones() * 3
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

    #[cfg(test)]
    mod test_bma {
        use crate::bits::Weight;

        #[test]
        fn bit() {
            assert_eq!(false.bma(false), 0);
            assert_eq!(false.bma(true), 1);
            assert_eq!(true.bma(false), 1);
            assert_eq!(true.bma(true), 0);
        }

        #[test]
        fn trit() {
            assert_eq!(<Option<bool> as Weight>::bma(None, true), 1);
            assert_eq!(<Option<bool> as Weight>::bma(None, false), 1);
            assert_eq!(<Option<bool> as Weight>::bma(Some(true), true), 0);
            assert_eq!(<Option<bool> as Weight>::bma(Some(false), true), 2);
        }

        #[test]
        fn quat() {
            assert_eq!((true, true).bma(true), 0);
            assert_eq!((true, false).bma(true), 1);
            assert_eq!((false, false).bma(true), 2);
            assert_eq!((false, true).bma(true), 3);
        }
        #[test]
        fn pent() {
            assert_eq!(
                <Option<(bool, bool)> as Weight>::bma(Some((false, true)), false),
                0
            );
            assert_eq!(
                <Option<(bool, bool)> as Weight>::bma(Some((false, false)), false),
                1
            );
            assert_eq!(<Option<(bool, bool)> as Weight>::bma(None, false), 2);
            assert_eq!(
                <Option<(bool, bool)> as Weight>::bma(Some((true, false)), false),
                3
            );
            assert_eq!(
                <Option<(bool, bool)> as Weight>::bma(Some((true, true)), false),
                4
            );

            assert_eq!(
                <Option<(bool, bool)> as Weight>::bma(Some((false, true)), true),
                4
            );
            assert_eq!(
                <Option<(bool, bool)> as Weight>::bma(Some((false, false)), true),
                3
            );
            assert_eq!(<Option<(bool, bool)> as Weight>::bma(None, true), 2);
            assert_eq!(
                <Option<(bool, bool)> as Weight>::bma(Some((true, false)), true),
                1
            );
            assert_eq!(
                <Option<(bool, bool)> as Weight>::bma(Some((true, true)), true),
                0
            );
        }
    }

    macro_rules! test_bma {
        ($name:ident, $s:ty, $w:ty, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights = <() as WeightArray<$s, $w>>::rand(&mut rng);

                    let sum = <() as WeightArray<$s, $w>>::bma(&weights, &inputs);
                    let true_sum = <() as WeightArray<$s, $w>>::bma_slow(&weights, &inputs);
                    assert_eq!(sum, true_sum);
                })
            }
        };
    }

    macro_rules! test_acts {
        ($name:ident, $s:ty, $w:ty, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights = <() as WeightArray<$s, $w>>::rand(&mut rng);

                    for val in <$w>::states() {
                        let acts = <() as WeightArray<$s, $w>>::acts(&weights, &inputs, val);
                        let true_acts =
                            <() as WeightArray<$s, $w>>::acts_slow(&weights, &inputs, val);
                        assert_eq!(acts, true_acts);
                    }
                })
            }
        };
    }

    macro_rules! test_input_acts {
        ($name:ident, $s:ty, $w:ty, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights = <() as WeightArray<$s, $w>>::rand(&mut rng);

                    let acts = <() as WeightArray<$s, $w>>::input_acts(&weights, &inputs);
                    let true_acts = <() as WeightArray<$s, $w>>::input_acts_slow(&weights, &inputs);
                    assert_eq!(acts, true_acts);
                })
            }
        };
    }

    macro_rules! test_loss_deltas {
        ($name:ident, $s:ty, $w:ty, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights = <() as WeightArray<$s, $w>>::rand(&mut rng);

                    let null_act = <() as WeightArray<$s, $w>>::bma(&weights, &inputs);
                    for &threshold in &[0, 10, 100, 1000] {
                        dbg!(threshold);
                        let mut loss_deltas = <() as WeightArray<$s, $w>>::loss_deltas(
                            &weights,
                            &inputs,
                            threshold,
                            |x| x as i64 - null_act as i64,
                        );
                        loss_deltas.sort();
                        let mut true_loss_deltas = <() as WeightArray<$s, $w>>::loss_deltas_slow(
                            &weights,
                            &inputs,
                            threshold,
                            |x| x as i64 - null_act as i64,
                        );
                        true_loss_deltas.sort();
                        assert_eq!(loss_deltas, true_loss_deltas);
                    }
                })
            }
        };
    }

    macro_rules! shape_group {
        ($name:ident, $s:ty, $w:ty, $n_iters:expr) => {
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

                test_bma!(bma, $s, $w, $n_iters);
                test_acts!(acts, $s, $w, $n_iters);
                test_input_acts!(input_acts, $s, $w, $n_iters);
                test_loss_deltas!(loss_deltas, $s, $w, $n_iters);
            }
        };
    }

    macro_rules! weight_group {
        ($name:ident, $w:ty) => {
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

                shape_group!(test_b8, [(); 8], $w, 10_000);
                shape_group!(test_b16, [(); 16], $w, 10_000);
                shape_group!(test_b32, [(); 32], $w, 10_000);
                shape_group!(test_b64, [(); 64], $w, 1_000);
                shape_group!(test_b128, [(); 128], $w, 1_000);
                shape_group!(test_b8x1, [[(); 8]; 1], $w, 10_000);
                shape_group!(test_b8x2, [[(); 8]; 2], $w, 10_000);
                shape_group!(test_b8x3, [[(); 8]; 3], $w, 10_000);
                shape_group!(test_b8x4, [[(); 8]; 4], $w, 10_000);
                shape_group!(test_b8x1x2x3, [[[[(); 8]; 1]; 2]; 3], $w, 10_000);
            }
        };
    }

    weight_group!(test_bit, bool);
    weight_group!(test_trit, Option<bool>);
    //weight_group!(test_trit, (bool, bool));
}
