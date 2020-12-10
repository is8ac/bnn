// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use crate::shape::{LongDefault, Pack, Shape, Wrap};
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

pub trait BitPack<E> {
    type Word;
    type UWord;
    type T;
}

impl<S, E, const L: usize> BitPack<E> for [S; L]
where
    S: BitPack<E>,
{
    type Word = S::Word;
    type UWord = S::UWord;
    type T = [<S as BitPack<E>>::T; L];
}

pub trait Blit<E>
where
    Self: BitPack<E>,
{
    fn blit(val: E) -> <Self as BitPack<E>>::T;
}

impl<E, S: Blit<E>, const L: usize> Blit<E> for [S; L]
where
    <S as BitPack<E>>::T: Copy,
{
    fn blit(val: E) -> [<S as BitPack<E>>::T; L] {
        [S::blit(val); L]
    }
}

pub trait BitScaler
where
    Self: Sized + Copy,
{
    const RANGE: u32;
    const N: usize;
    fn states() -> iter::Cloned<slice::Iter<'static, Self>>;
    fn bma(self, input: bool) -> u32;
}

impl BitScaler for bool {
    const RANGE: u32 = 2;
    const N: usize = 2;
    fn states() -> iter::Cloned<slice::Iter<'static, bool>> {
        [true, false].iter().cloned()
    }
    fn bma(self, input: bool) -> u32 {
        (self ^ input) as u32
    }
}

impl BitScaler for Option<bool> {
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

impl BitScaler for (bool, bool) {
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

impl BitScaler for Option<(bool, bool)> {
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

pub trait WeightArray<S: Shape, W: BitScaler>
where
    S: Shape
        + PackedIndexMap<bool>
        + BitPack<bool>
        + BitPack<W>
        + BMA<W>
        + PackedIndexSGet<W>
        + PackedIndexSGet<bool>,
    W: 'static + BitScaler,
    <S as BitPack<W>>::T: Copy,
    <S as BitPack<bool>>::T: Copy,
{
    const MAX: u32;
    const THRESHOLD: u32;
    /// thresholded activation
    fn act(weights: &<S as BitPack<W>>::T, input: &<S as BitPack<bool>>::T) -> bool {
        <S as BMA<W>>::bma(weights, input) > <Self as WeightArray<S, W>>::THRESHOLD
    }
    fn act_is_alive(weights: &<S as BitPack<W>>::T, input: &<S as BitPack<bool>>::T) -> bool {
        let act = <S as BMA<W>>::bma(weights, input);
        ((act + W::RANGE) > <Self as WeightArray<S, W>>::THRESHOLD)
            | ((<Self as WeightArray<S, W>>::THRESHOLD + W::RANGE) < act)
    }
    /// loss delta if we were to set each of the elements to each different value. Is permited to prune null deltas.
    fn loss_deltas<F: Fn(u32) -> i64>(
        weights: &<S as BitPack<W>>::T,
        input: &<S as BitPack<bool>>::T,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<S as Shape>::Index, W, i64)>;
    /// Does the same thing as loss_deltas but is a lot slower.
    fn loss_deltas_slow<F: Fn(u32) -> i64>(
        weights: &<S as BitPack<W>>::T,
        input: &<S as BitPack<bool>>::T,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<S as Shape>::Index, W, i64)> {
        <W as BitScaler>::states()
            .map(|value| {
                <S as Shape>::indices()
                    .map(|index| {
                        (
                            index,
                            value,
                            loss_delta_fn(S::bma(&S::set(*weights, index, value), &input)),
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
        weights: &<S as BitPack<W>>::T,
        input: &<S as BitPack<bool>>::T,
        value: W,
    ) -> <S as BitPack<bool>>::T;
    /// acts_slow does the same thing as grads, but it is a lot slower.
    fn acts_slow(
        &weights: &<S as BitPack<W>>::T,
        input: &<S as BitPack<bool>>::T,
        value: W,
    ) -> <S as BitPack<bool>>::T {
        <S as PackedIndexMap<bool>>::index_map(|index| {
            <Self as WeightArray<S, W>>::act(&S::set(weights, index, value), input)
        })
    }
    /// Act if each input is flipped.
    fn input_acts(
        weights: &<S as BitPack<W>>::T,
        input: &<S as BitPack<bool>>::T,
    ) -> <S as BitPack<bool>>::T;
    /// input_acts_slow does the same thing as input_acts, but it is a lot slower.
    fn input_acts_slow(
        weights: &<S as BitPack<W>>::T,
        input: &<S as BitPack<bool>>::T,
    ) -> <S as BitPack<bool>>::T {
        <S as PackedIndexMap<bool>>::index_map(|index| {
            <Self as WeightArray<S, W>>::act(
                weights,
                &<S as PackedIndexSGet<bool>>::set(
                    *input,
                    index,
                    !<S as PackedIndexSGet<bool>>::get(input, index),
                ),
            )
        })
    }
}

impl<S> WeightArray<S, bool> for ()
where
    S: Shape + Blit<bool> + BMA<bool> + PackedIndexSGet<bool> + PackedIndexMap<bool> + WBBZM<bool>,
    <S as BitPack<bool>>::T: Copy,
    <S as BitPack<bool>>::Word: BitWord<Word = <S as BitPack<bool>>::UWord> + Copy,
    <S as BitPack<bool>>::UWord: Copy
        + ops::BitAnd<Output = <S as BitPack<bool>>::UWord>
        + ops::BitOr<Output = <S as BitPack<bool>>::UWord>
        + ops::BitXor<Output = <S as BitPack<bool>>::UWord>
        + ops::Not<Output = <S as BitPack<bool>>::UWord>,
{
    const MAX: u32 = S::N as u32;
    const THRESHOLD: u32 = S::N as u32 / 2;
    fn loss_deltas<F: Fn(u32) -> i64>(
        weights: &<S as BitPack<bool>>::T,
        input: &<S as BitPack<bool>>::T,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<S as Shape>::Index, bool, i64)> {
        let cur_act = S::bma(weights, input);
        let deltas = [
            loss_delta_fn(cur_act.saturating_add(1)),
            loss_delta_fn(cur_act.saturating_sub(1)),
        ];
        if deltas.iter().find(|d| d.abs() as u64 > threshold).is_some() {
            <S as Shape>::indices()
                .map(|i| {
                    let weight = S::get(weights, i);
                    let input = S::get(input, i);
                    (i, !weight, deltas[(weight ^ input) as usize])
                })
                .filter(|(_, _, l)| l.abs() as u64 > threshold)
                .collect()
        } else {
            vec![]
        }
    }
    fn acts(
        weights: &<S as BitPack<bool>>::T,
        input: &<S as BitPack<bool>>::T,
        value: bool,
    ) -> <S as BitPack<bool>>::T {
        let cur_act = S::bma(weights, input);
        if cur_act == <() as WeightArray<S, bool>>::THRESHOLD {
            // we need to go up by one to flip
            if value {
                S::map(weights, input, |weight, input| {
                    !(weight.sign() ^ input) & !weight.sign()
                })
            } else {
                S::map(weights, input, |weight, input| {
                    !(weight.sign() ^ input) & weight.sign()
                })
            }
        } else if cur_act == (<() as WeightArray<S, bool>>::THRESHOLD + 1) {
            // we need to go down by one to flip
            if value {
                S::map(weights, input, |weight, input| {
                    !(weight.sign() ^ input) | weight.sign()
                })
            } else {
                S::map(weights, input, |weight, input| {
                    !(weight.sign() ^ input) | !weight.sign()
                })
            }
        } else {
            S::blit(cur_act > <() as WeightArray<S, bool>>::THRESHOLD)
        }
    }
    fn input_acts(
        weights: &<S as BitPack<bool>>::T,
        input: &<S as BitPack<bool>>::T,
    ) -> <S as BitPack<bool>>::T {
        let cur_act = S::bma(weights, input);
        if (cur_act == <() as WeightArray<S, bool>>::THRESHOLD)
            | (cur_act == (<() as WeightArray<S, bool>>::THRESHOLD + 1))
        {
            S::map(weights, input, |weight, input| weight.sign() ^ !input)
        } else {
            S::blit(cur_act > <() as WeightArray<S, bool>>::THRESHOLD)
        }
    }
}

impl<S: Shape> WeightArray<S, Option<bool>> for ()
where
    S: BitPack<bool>
        + BitPack<Option<bool>>
        + BMA<Option<bool>>
        + PackedIndexSGet<Option<bool>>
        + PackedIndexSGet<bool>
        + PackedIndexMap<bool>
        + WBBZM<Option<bool>>
        + Blit<bool>,
    <S as BitPack<bool>>::T: Copy,
    <S as BitPack<Option<bool>>>::T: Copy,
    <S as BitPack<Option<bool>>>::Word: TritWord<Word = <S as BitPack<Option<bool>>>::UWord> + Copy,
    <S as BitPack<Option<bool>>>::UWord: Copy
        + ops::BitAnd<Output = <S as BitPack<Option<bool>>>::UWord>
        + ops::BitOr<Output = <S as BitPack<Option<bool>>>::UWord>
        + ops::BitXor<Output = <S as BitPack<Option<bool>>>::UWord>
        + ops::Not<Output = <S as BitPack<Option<bool>>>::UWord>,
{
    const MAX: u32 = <S as Shape>::N as u32 * 2;
    const THRESHOLD: u32 = S::N as u32;
    fn loss_deltas<F: Fn(u32) -> i64>(
        weights: &<S as BitPack<Option<bool>>>::T,
        input: &<S as BitPack<bool>>::T,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<S as Shape>::Index, Option<bool>, i64)> {
        let cur_act = S::bma(weights, input);
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
                    let input: bool = S::get(input, i);
                    if let Some(sign) = S::get(weights, i) {
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
        weights: &<S as BitPack<Option<bool>>>::T,
        input: &<S as BitPack<bool>>::T,
        value: Option<bool>,
    ) -> <S as BitPack<bool>>::T {
        let cur_act = S::bma(weights, input);
        let value_sign = value.unwrap_or(false);
        let value_mask = value.is_some();

        if (cur_act + 2) <= <() as WeightArray<S, Option<bool>>>::THRESHOLD {
            S::blit(false)
        } else if cur_act > (<() as WeightArray<S, Option<bool>>>::THRESHOLD + 2) {
            S::blit(true)
        } else if (cur_act + 1) == <() as WeightArray<S, Option<bool>>>::THRESHOLD {
            // go up 2 to activate
            if value_mask {
                if value_sign {
                    S::map(weights, &input, |trit, input| {
                        ((!input ^ trit.sign()) & trit.mask()) & !trit.sign()
                    })
                } else {
                    S::map(weights, &input, |trit, input| {
                        ((!input ^ trit.sign()) & trit.mask()) & trit.sign()
                    })
                }
            } else {
                S::blit(false)
            }
        } else if cur_act == (<() as WeightArray<S, Option<bool>>>::THRESHOLD + 2) {
            // go down 2 to deactivate
            if value_mask {
                if value_sign {
                    S::map(weights, &input, |trit, input| {
                        !(((input ^ trit.sign()) & trit.mask()) & !trit.sign())
                    })
                } else {
                    S::map(weights, &input, |trit, input| {
                        !(((input ^ trit.sign()) & trit.mask()) & trit.sign())
                    })
                }
            } else {
                S::blit(true)
            }
        } else if cur_act == <() as WeightArray<S, Option<bool>>>::THRESHOLD {
            // go up 1 to activate
            if value_mask {
                if value_sign {
                    S::map(weights, &input, |trit, input| {
                        !((input ^ trit.sign()) & trit.mask()) & !input
                    })
                } else {
                    S::map(weights, &input, |trit, input| {
                        !((input ^ trit.sign()) & trit.mask()) & input
                    })
                }
            } else {
                S::map(weights, &input, |trit, input| {
                    (input ^ !trit.sign()) & trit.mask()
                })
            }
        } else if cur_act == (<() as WeightArray<S, Option<bool>>>::THRESHOLD + 1) {
            // go down 1 to deactivate
            if value_mask {
                if value_sign {
                    S::map(weights, &input, |trit, input| {
                        !(input & !(trit.sign() & trit.mask()))
                    })
                } else {
                    S::map(weights, &input, |trit, input| {
                        (!((input ^ trit.sign()) & trit.mask()) & trit.mask()) | input
                    })
                }
            } else {
                S::map(weights, &input, |trit, input| {
                    !((input ^ trit.sign()) & trit.mask())
                })
            }
        } else {
            unreachable!();
        }
    }
    fn input_acts(
        weights: &<S as BitPack<Option<bool>>>::T,
        input: &<S as BitPack<bool>>::T,
    ) -> <S as BitPack<bool>>::T {
        let cur_act = S::bma(weights, input);

        if (cur_act + 2) <= <() as WeightArray<S, Option<bool>>>::THRESHOLD {
            S::blit(false)
        } else if cur_act > (<() as WeightArray<S, Option<bool>>>::THRESHOLD + 2) {
            S::blit(true)
        } else if ((cur_act + 1) == <() as WeightArray<S, Option<bool>>>::THRESHOLD)
            | (cur_act == <() as WeightArray<S, Option<bool>>>::THRESHOLD)
        {
            S::map(weights, &input, |trit, input| {
                (!input ^ trit.sign()) & trit.mask()
            })
        } else if (cur_act == (<() as WeightArray<S, Option<bool>>>::THRESHOLD + 2))
            | (cur_act == (<() as WeightArray<S, Option<bool>>>::THRESHOLD + 1))
        {
            S::map(weights, &input, |trit, input| {
                !((input ^ trit.sign()) & trit.mask())
            })
        } else {
            unreachable!();
        }
    }
}

pub trait PackedIndexMap<O>
where
    Self: Shape + BitPack<O>,
{
    fn index_map<F: Fn(Self::Index) -> O>(map_fn: F) -> <Self as BitPack<O>>::T;
}

impl<S, O> PackedIndexMap<O> for S
where
    S: Shape + PackedIndexMapInner<O, ()> + BitPack<O>,
{
    fn index_map<F: Fn(Self::Index) -> O>(map_fn: F) -> <S as BitPack<O>>::T {
        <S as PackedIndexMapInner<O, ()>>::index_map_inner((), map_fn)
    }
}

// W is the wrapper shape
pub trait PackedIndexMapInner<O, W>
where
    <W as Pack<Self>>::T: Shape,
    W: Pack<Self>,
    Self: Sized + BitPack<O>,
{
    fn index_map_inner<F: Fn(<<W as Pack<Self>>::T as Shape>::Index) -> O>(
        outer_index: W::Index,
        map_fn: F,
    ) -> <Self as BitPack<O>>::T;
}

impl<O, W, S, const L: usize> PackedIndexMapInner<O, W> for [S; L]
where
    W: Pack<[S; L]> + Pack<[(); L]>,
    S: BitPack<O> + PackedIndexMapInner<O, <W as Pack<[(); L]>>::T>,
    W::Index: Copy,
    <W as Pack<[S; L]>>::T:
        Shape<Index = <<<W as Pack<[(); L]>>::T as Pack<S>>::T as Shape>::Index>,
    <W as Pack<[(); L]>>::T: Shape + Pack<S>,
    <<W as Pack<[(); L]>>::T as Pack<S>>::T: Shape,
    [<S as BitPack<O>>::T; L]: LongDefault,
    (u8, ()): Wrap<W::Index, Wrapped = <<W as Pack<[(); L]>>::T as Shape>::Index>,
    <S as BitPack<O>>::T: Default,
{
    fn index_map_inner<F: Fn(<<W as Pack<[S; L]>>::T as Shape>::Index) -> O>(
        outer_index: W::Index,
        map_fn: F,
    ) -> [<S as BitPack<O>>::T; L] {
        let mut target = <[<S as BitPack<O>>::T; L]>::long_default();
        for i in 0..L {
            target[i] = <S as PackedIndexMapInner<O, <W as Pack<[(); L]>>::T>>::index_map_inner(
                (i as u8, ()).wrap(outer_index),
                &map_fn,
            );
        }
        target
    }
}

pub trait PackedMap<I, O>
where
    Self: Shape + Sized + Pack<I> + BitPack<O>,
{
    fn map<F: Fn(&I) -> O>(input: &<Self as Pack<I>>::T, map_fn: F) -> <Self as BitPack<O>>::T;
}

impl<S, I, O, const L: usize> PackedMap<I, O> for [S; L]
where
    S: Pack<I> + BitPack<O> + PackedMap<I, O>,
    [<S as BitPack<O>>::T; L]: LongDefault,
{
    fn map<F: Fn(&I) -> O>(input: &[<S as Pack<I>>::T; L], map_fn: F) -> [<S as BitPack<O>>::T; L] {
        let mut target = <[<S as BitPack<O>>::T; L]>::long_default();
        for i in 0..L {
            target[i] = <S as PackedMap<I, O>>::map(&input[i], &map_fn);
        }
        target
    }
}

/// Weight Bit Bit Zip Map
pub trait WBBZM<W>
where
    Self: Shape + BitPack<W> + BitPack<bool>,
{
    fn map<
        F: Fn(<Self as BitPack<W>>::Word, <Self as BitPack<W>>::UWord) -> <Self as BitPack<W>>::UWord,
    >(
        weights: &<Self as BitPack<W>>::T,
        rhs: &<Self as BitPack<bool>>::T,
        map_fn: F,
    ) -> <Self as BitPack<bool>>::T;
}

impl<S, W, const L: usize> WBBZM<W> for [S; L]
where
    S: WBBZM<W>,
    [<S as BitPack<bool>>::T; L]: LongDefault,
{
    fn map<
        F: Fn(<Self as BitPack<W>>::Word, <Self as BitPack<W>>::UWord) -> <Self as BitPack<W>>::UWord,
    >(
        weights: &[<S as BitPack<W>>::T; L],
        rhs: &[<S as BitPack<bool>>::T; L],
        map_fn: F,
    ) -> [<S as BitPack<bool>>::T; L] {
        let mut target = <[<S as BitPack<bool>>::T; L]>::long_default();
        for i in 0..L {
            target[i] = S::map(&weights[i], &rhs[i], &map_fn);
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

pub trait IncrementCounters
where
    Self: Shape + Sized + Pack<u32> + BitPack<bool>,
{
    fn counted_increment(
        bits: &<Self as BitPack<bool>>::T,
        counters: (usize, <Self as Pack<u32>>::T),
    ) -> (usize, <Self as Pack<u32>>::T) {
        (counters.0 + 1, Self::increment(bits, counters.1))
    }
    fn increment(
        bits: &<Self as BitPack<bool>>::T,
        mut counters: <Self as Pack<u32>>::T,
    ) -> <Self as Pack<u32>>::T {
        Self::increment_in_place(bits, &mut counters);
        counters
    }
    fn increment_in_place(bits: &<Self as BitPack<bool>>::T, counters: &mut <Self as Pack<u32>>::T);
}

impl<S: IncrementCounters, const L: usize> IncrementCounters for [S; L]
where
    S: Pack<u32> + BitPack<bool>,
{
    fn increment_in_place(
        bits: &[<S as BitPack<bool>>::T; L],
        counters: &mut [<S as Pack<u32>>::T; L],
    ) {
        for i in 0..L {
            S::increment_in_place(&bits[i], &mut counters[i]);
        }
    }
}

pub trait BMA<W>
where
    Self: BitPack<W> + BitPack<bool>,
{
    fn bma(weights: &<Self as BitPack<W>>::T, bits: &<Self as BitPack<bool>>::T) -> u32;
}

impl<S, W, const L: usize> BMA<W> for [S; L]
where
    S: BMA<W> + BitPack<bool> + BitPack<W>,
{
    fn bma(weights: &[<S as BitPack<W>>::T; L], bits: &[<S as BitPack<bool>>::T; L]) -> u32 {
        let mut sum = 0_u32;
        for i in 0..L {
            sum += S::bma(&weights[i], &bits[i]);
        }
        sum
    }
}

/// Packed Index Set/Get
pub trait PackedIndexSGet<W>
where
    Self: Shape + BitPack<W>,
{
    fn get(array: &<Self as BitPack<W>>::T, i: Self::Index) -> W;
    fn set_in_place(array: &mut <Self as BitPack<W>>::T, i: Self::Index, val: W);
    fn set(mut array: <Self as BitPack<W>>::T, i: Self::Index, val: W) -> <Self as BitPack<W>>::T {
        <Self as PackedIndexSGet<W>>::set_in_place(&mut array, i, val);
        array
    }
}

impl<W, S: PackedIndexSGet<W>, const L: usize> PackedIndexSGet<W> for [S; L] {
    fn get(array: &[<S as BitPack<W>>::T; L], (i, tail): (u8, S::Index)) -> W {
        S::get(&array[i as usize], tail)
    }
    fn set_in_place(array: &mut <Self as BitPack<W>>::T, (i, tail): (u8, S::Index), val: W) {
        S::set_in_place(&mut array[i as usize], tail, val)
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

        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $b_type(pub $u_type);

        /// A word of trits. The 0 element is the signs, the 1 element is the mask.
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $t_type(pub $u_type, pub $u_type);

        /// A word of quats. The 0 element is the sign, the 1 element is the magnitudes.
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $q_type(pub $u_type, pub $u_type);

        impl_long_default_for_type!($b_type);
        impl_long_default_for_type!($t_type);
        impl_long_default_for_type!($q_type);

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

        impl fmt::Debug for $b_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, $format_string, self.0)
            }
        }
        impl fmt::Debug for $t_type {
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

        impl fmt::Display for $b_type {
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
        impl fmt::Display for $q_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, $format_string, self.0)?;
                write!(f, $format_string, self.1)
            }
        }

        impl BitPack<bool> for [(); $len] {
            type Word = $b_type;
            type UWord = $u_type;
            type T = $b_type;
        }
        impl BitPack<Option<bool>> for [(); $len] {
            type Word = $t_type;
            type UWord = $u_type;
            type T = $t_type;
        }
        impl BitPack<(bool, bool)> for [(); $len] {
            type Word = $q_type;
            type UWord = $u_type;
            type T = $q_type;
        }

        impl BitWord for $b_type {
            type Word = $u_type;
            #[inline(always)]
            fn sign(self) -> $u_type {
                self.0
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

        impl BMA<bool> for [(); $len] {
            fn bma(&weights: &$b_type, &rhs: &$b_type) -> u32 {
                (weights.0 ^ rhs.0).count_ones()
            }
        }
        impl BMA<Option<bool>> for [(); $len] {
            fn bma(&weights: &$t_type, &rhs: &$b_type) -> u32 {
                weights.1.count_zeros() + ((weights.0 ^ rhs.0) & weights.1).count_ones() * 2
            }
        }

        impl WBBZM<bool> for [(); $len] {
            fn map<F: Fn($b_type, $u_type) -> $u_type>(
                weights: &$b_type,
                rhs: &$b_type,
                map_fn: F,
            ) -> $b_type {
                $b_type(map_fn(*weights, rhs.0))
            }
        }
        impl WBBZM<Option<bool>> for [(); $len] {
            fn map<F: Fn($t_type, $u_type) -> $u_type>(
                weights: &$t_type,
                rhs: &$b_type,
                map_fn: F,
            ) -> $b_type {
                $b_type(map_fn(*weights, rhs.0))
            }
        }
        impl WBBZM<(bool, bool)> for [(); $len] {
            fn map<F: Fn($q_type, $u_type) -> $u_type>(
                weights: &$q_type,
                rhs: &$b_type,
                map_fn: F,
            ) -> $b_type {
                $b_type(map_fn(*weights, rhs.0))
            }
        }

        impl PackedIndexSGet<bool> for [(); $len] {
            fn get(&array: &$b_type, (i, _): (u8, ())) -> bool {
                array.get_bit(i as usize)
            }
            fn set_in_place(array: &mut $b_type, (i, _): (u8, ()), val: bool) {
                array.set_bit_in_place(i as usize, val);
            }
        }
        impl PackedIndexSGet<Option<bool>> for [(); $len] {
            fn get(&array: &$t_type, (i, _): (u8, ())) -> Option<bool> {
                array.get_trit(i as usize)
            }
            fn set_in_place(array: &mut $t_type, (i, _): (u8, ()), val: Option<bool>) {
                array.set_trit_in_place(i as usize, val);
            }
        }

        impl IncrementCounters for [(); $len] {
            fn increment_in_place(&bits: &$b_type, counters: &mut [u32; $len]) {
                for i in 0..$len {
                    counters[i] += bits.get_bit(i) as u32;
                }
            }
        }

        impl<W: Shape> PackedIndexMapInner<bool, W> for [(); $len]
        where
            (u8, ()): Wrap<W::Index, Wrapped = <<W as Pack<[(); $len]>>::T as Shape>::Index>,
            <W as Pack<[(); $len]>>::T: Shape,
            W: Pack<[(); $len]>,
            <W as Pack<[(); $len]>>::T: Shape,
        {
            fn index_map_inner<F: Fn(<<W as Pack<[(); $len]>>::T as Shape>::Index) -> bool>(
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

        impl Blit<bool> for [(); $len] {
            fn blit(val: bool) -> $b_type {
                $b_type((Wrapping(0) - Wrapping(val as $u_type)).0)
            }
        }
        impl Blit<Option<bool>> for [(); $len] {
            fn blit(val: Option<bool>) -> $t_type {
                $t_type(
                    (Wrapping(0) - Wrapping(val.unwrap_or(false) as $u_type)).0,
                    (Wrapping(0) - Wrapping(val.is_some() as $u_type)).0,
                )
            }
        }
        impl Blit<(bool, bool)> for [(); $len] {
            fn blit(value: (bool, bool)) -> $q_type {
                $q_type(
                    (Wrapping(0) - Wrapping(value.0 as $u_type)).0,
                    (Wrapping(0) - Wrapping(value.1 as $u_type)).0,
                )
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

        impl PartialEq for $t_type {
            fn eq(&self, other: &Self) -> bool {
                ((self.0 & self.1) == (other.0 & self.1)) & (self.1 == other.1)
            }
        }
        impl Eq for $t_type {}

        impl PartialEq for $b_type {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl Eq for $b_type {}
    };
}

for_uints!(q8, t8, b8, u8, 8, "{:08b}");
for_uints!(q16, t16, b16, u16, 16, "{:016b}");
for_uints!(q32, t32, b32, u32, 32, "{:032b}");
for_uints!(q64, t64, b64, u64, 64, "{:064b}");
for_uints!(q128, t128, b128, u128, 128, "{:0128b}");

#[cfg(test)]
mod tests {
    use super::{b128, b16, b32, b8, t128, t16, t32, t8, BitScaler, WeightArray};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;

    #[cfg(test)]
    mod test_bma {
        use crate::bits::BitScaler;

        #[test]
        fn bit() {
            assert_eq!(false.bma(false), 0);
            assert_eq!(false.bma(true), 1);
            assert_eq!(true.bma(false), 1);
            assert_eq!(true.bma(true), 0);
        }

        #[test]
        fn trit() {
            assert_eq!(<Option<bool> as BitScaler>::bma(None, true), 1);
            assert_eq!(<Option<bool> as BitScaler>::bma(None, false), 1);
            assert_eq!(<Option<bool> as BitScaler>::bma(Some(true), true), 0);
            assert_eq!(<Option<bool> as BitScaler>::bma(Some(false), true), 2);
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
                <Option<(bool, bool)> as BitScaler>::bma(Some((false, true)), false),
                0
            );
            assert_eq!(
                <Option<(bool, bool)> as BitScaler>::bma(Some((false, false)), false),
                1
            );
            assert_eq!(<Option<(bool, bool)> as BitScaler>::bma(None, false), 2);
            assert_eq!(
                <Option<(bool, bool)> as BitScaler>::bma(Some((true, false)), false),
                3
            );
            assert_eq!(
                <Option<(bool, bool)> as BitScaler>::bma(Some((true, true)), false),
                4
            );
            assert_eq!(
                <Option<(bool, bool)> as BitScaler>::bma(Some((false, true)), true),
                4
            );
            assert_eq!(
                <Option<(bool, bool)> as BitScaler>::bma(Some((false, false)), true),
                3
            );
            assert_eq!(<Option<(bool, bool)> as BitScaler>::bma(None, true), 2);
            assert_eq!(
                <Option<(bool, bool)> as BitScaler>::bma(Some((true, false)), true),
                1
            );
            assert_eq!(
                <Option<(bool, bool)> as BitScaler>::bma(Some((true, true)), true),
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
                    let inputs: <$s as BitPack<bool>>::T = rng.gen();
                    let weights: <$s as BitPack<$w>>::T = rng.gen();

                    let sum = <$s as BMA<$w>>::bma(&weights, &inputs);
                    let true_sum: u32 = <$s as Shape>::indices()
                        .map(|i| {
                            <$s as PackedIndexSGet<$w>>::get(&weights, i).bma(<$s>::get(&inputs, i))
                        })
                        .sum();
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
                    let inputs: <$s as BitPack<bool>>::T = rng.gen();
                    let weights: <$s as BitPack<$w>>::T = rng.gen();

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
                    let inputs: <$s as BitPack<bool>>::T = rng.gen();
                    let weights: <$s as BitPack<$w>>::T = rng.gen();

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
                    let inputs: <$s as BitPack<bool>>::T = rng.gen();
                    let weights: <$s as BitPack<$w>>::T = rng.gen();

                    let null_act = <$s as BMA<$w>>::bma(&weights, &inputs);
                    for &threshold in &[0, 10, 100, 1000] {
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
                    b128, b16, b32, b64, b8, t128, t16, t32, t64, t8, BitPack, BitScaler,
                    PackedIndexSGet, Shape, WeightArray, BMA,
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
                    b128, b16, b32, b64, b8, t128, t16, t32, t64, t8, BitScaler, PackedIndexSGet,
                    Shape, WeightArray,
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
