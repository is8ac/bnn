/// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use crate::shape::{LongDefault, Pack, Shape, Wrap};
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    uint8x16_t, vaddq_u8, vandq_u8, vceqzq_u8, vld1q_dup_u32, vld1q_dup_u8, vld1q_s8, vld1q_u8,
    vmvnq_u8, vqshlq_u8, vqtbl1q_u8, vreinterpretq_u8_u32, vst1q_s8, vst1q_u8,
};
#[cfg(target_feature = "avx2")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi8, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpeq_epi8,
    _mm256_loadu_si256, _mm256_set1_epi32, _mm256_set1_epi8, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_storeu_si256,
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter;
use std::mem::{transmute, MaybeUninit};
use std::num::Wrapping;
use std::ops;
use std::ops::{Add, AddAssign};

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

/*
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct bit(pub bool);

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct trit(pub Option<bool>);

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct quat(pub (bool, bool));

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct pent(pub Option<(bool, bool)>);
*/

pub trait BitScaler
where
    Self: Sized + Copy,
    Self::ValuesShape: Pack<Self>,
{
    const RANGE: u32;
    const N: usize;
    type ValuesShape;
    const STATES: <Self::ValuesShape as Pack<Self>>::T;
    fn states() -> Vec<Self>;
    fn bma(self, input: bool) -> u32;
}

impl BitScaler for bool {
    const RANGE: u32 = 2;
    const N: usize = 2;
    type ValuesShape = [(); 2];
    const STATES: <Self::ValuesShape as Pack<Self>>::T = [true, false];
    fn states() -> Vec<bool> {
        vec![true, false]
    }
    #[inline(always)]
    fn bma(self, input: bool) -> u32 {
        (self ^ input) as u32
    }
}

impl BitScaler for Option<bool> {
    const RANGE: u32 = 3;
    const N: usize = 3;
    type ValuesShape = [(); 3];
    const STATES: <Self::ValuesShape as Pack<Self>>::T = [Some(true), None, Some(false)];
    fn states() -> Vec<Option<bool>> {
        vec![Some(true), None, Some(false)]
    }
    #[inline(always)]
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
    type ValuesShape = [(); 4];
    const STATES: <Self::ValuesShape as Pack<Self>>::T =
        [(true, true), (true, false), (false, false), (false, true)];
    fn states() -> Vec<(bool, bool)> {
        vec![(true, true), (true, false), (false, false), (false, true)]
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
    type ValuesShape = [(); 5];
    const STATES: <Self::ValuesShape as Pack<Self>>::T = [
        Some((true, true)),
        Some((true, false)),
        None,
        Some((false, false)),
        Some((false, true)),
    ];
    fn states() -> Vec<Option<(bool, bool)>> {
        vec![
            Some((true, true)),
            Some((true, false)),
            None,
            Some((false, false)),
            Some((false, true)),
        ]
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

pub trait WeightArray<W: BitScaler>
where
    Self: Shape
        + PackedIndexMap<bool>
        + BitPack<bool>
        + BitPack<W>
        + BMA<W>
        + PackedIndexSGet<W>
        + PackedIndexSGet<bool>,
    W: BitScaler,
    <Self as BitPack<W>>::T: Copy,
    <Self as BitPack<bool>>::T: Copy,
{
    const MAX: u32;
    const THRESHOLD: u32;
    /// thresholded activation
    fn act(weights: &<Self as BitPack<W>>::T, input: &<Self as BitPack<bool>>::T) -> bool {
        <Self as BMA<W>>::bma(weights, input) > <Self as WeightArray<W>>::THRESHOLD
    }
    fn mutant_act(cur_sum: u32) -> Option<bool> {
        let low = cur_sum <= (<Self as WeightArray<W>>::THRESHOLD - (W::RANGE - 1));
        let high = cur_sum >= (<Self as WeightArray<W>>::THRESHOLD + W::RANGE);
        Some(high).filter(|_| high | low)
    }
    //fn increment_dense_loss_deltas(weights: &<Self as BitPack<W>>::T, deltas: &mut <Self as Pack<i32>>::T);
    /// loss delta if we were to set each of the elements to each different value. Is permited to prune null deltas.
    fn loss_deltas<F: Fn(u32) -> i64>(
        weights: &<Self as BitPack<W>>::T,
        input: &<Self as BitPack<bool>>::T,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<Self as Shape>::Index, W, i64)>;
    /// Does the same thing as loss_deltas but is a lot slower.
    fn loss_deltas_slow<F: Fn(u32) -> i64>(
        weights: &<Self as BitPack<W>>::T,
        input: &<Self as BitPack<bool>>::T,
        threshold: u64,
        loss_delta_fn: F,
    ) -> Vec<(<Self as Shape>::Index, W, i64)> {
        <W as BitScaler>::states()
            .iter()
            .map(|&value| {
                <Self as Shape>::indices()
                    .map(|index| {
                        (
                            index,
                            value,
                            loss_delta_fn(Self::bma(&Self::set(*weights, index, value), &input)),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .filter(|(_, _, l)| l.abs() as u64 > threshold)
            .collect()
    }
    /// Acts if we were to set each element to value
    fn acts_simple(
        weights: &<Self as BitPack<W>>::T,
        input: &<Self as BitPack<bool>>::T,
        cur_sum: u32,
        value: W,
    ) -> <Self as BitPack<bool>>::T;
    fn acts(
        weights: &<Self as BitPack<W>>::T,
        input: &<Self as BitPack<bool>>::T,
        value: W,
    ) -> <Self as BitPack<bool>>::T {
        let cur_sum = Self::bma(weights, input);
        Self::acts_simple(weights, input, cur_sum, value)
    }
    /// acts_slow does the same thing as grads, but it is a lot slower.
    fn acts_slow(
        &weights: &<Self as BitPack<W>>::T,
        input: &<Self as BitPack<bool>>::T,
        value: W,
    ) -> <Self as BitPack<bool>>::T {
        <Self as PackedIndexMap<bool>>::index_map(|index| {
            <Self as WeightArray<W>>::act(&Self::set(weights, index, value), input)
        })
    }
    /// Act if each input is flipped.
    fn input_acts(
        weights: &<Self as BitPack<W>>::T,
        input: &<Self as BitPack<bool>>::T,
    ) -> <Self as BitPack<bool>>::T;
    /// input_acts_slow does the same thing as input_acts, but it is a lot slower.
    fn input_acts_slow(
        weights: &<Self as BitPack<W>>::T,
        input: &<Self as BitPack<bool>>::T,
    ) -> <Self as BitPack<bool>>::T {
        <Self as PackedIndexMap<bool>>::index_map(|index| {
            <Self as WeightArray<W>>::act(
                weights,
                &<Self as PackedIndexSGet<bool>>::set(
                    *input,
                    index,
                    !<Self as PackedIndexSGet<bool>>::get(input, index),
                ),
            )
        })
    }
}

impl<S> WeightArray<bool> for S
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
    fn acts_simple(
        weights: &<S as BitPack<bool>>::T,
        input: &<S as BitPack<bool>>::T,
        cur_act: u32,
        value: bool,
    ) -> <S as BitPack<bool>>::T {
        if cur_act == <S as WeightArray<bool>>::THRESHOLD {
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
        } else if cur_act == (<S as WeightArray<bool>>::THRESHOLD + 1) {
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
            S::blit(cur_act > <S as WeightArray<bool>>::THRESHOLD)
        }
    }
    fn input_acts(
        weights: &<S as BitPack<bool>>::T,
        input: &<S as BitPack<bool>>::T,
    ) -> <S as BitPack<bool>>::T {
        let cur_act = S::bma(weights, input);
        if (cur_act == <S as WeightArray<bool>>::THRESHOLD)
            | (cur_act == (<S as WeightArray<bool>>::THRESHOLD + 1))
        {
            S::map(weights, input, |weight, input| weight.sign() ^ !input)
        } else {
            S::blit(cur_act > <S as WeightArray<bool>>::THRESHOLD)
        }
    }
}

impl<S: Shape> WeightArray<Option<bool>> for S
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
    fn acts_simple(
        weights: &<S as BitPack<Option<bool>>>::T,
        input: &<S as BitPack<bool>>::T,
        cur_act: u32,
        value: Option<bool>,
    ) -> <S as BitPack<bool>>::T {
        let value_sign = value.unwrap_or(false);
        let value_mask = value.is_some();

        if (cur_act + 2) <= <S as WeightArray<Option<bool>>>::THRESHOLD {
            S::blit(false)
        } else if cur_act > (<S as WeightArray<Option<bool>>>::THRESHOLD + 2) {
            S::blit(true)
        } else if (cur_act + 1) == <S as WeightArray<Option<bool>>>::THRESHOLD {
            // go up 2 to activate
            if value_mask {
                if value_sign {
                    S::map(weights, &input, |w, input| {
                        ((!input ^ w.sign()) & w.mask()) & !w.sign()
                    })
                } else {
                    S::map(weights, &input, |w, input| {
                        ((!input ^ w.sign()) & w.mask()) & w.sign()
                    })
                }
            } else {
                S::blit(false)
            }
        } else if cur_act == (<S as WeightArray<Option<bool>>>::THRESHOLD + 2) {
            // go down 2 to deactivate
            if value_mask {
                if value_sign {
                    S::map(weights, &input, |w, input| {
                        !(((input ^ w.sign()) & w.mask()) & !w.sign())
                    })
                } else {
                    S::map(weights, &input, |w, input| {
                        !(((input ^ w.sign()) & w.mask()) & w.sign())
                    })
                }
            } else {
                S::blit(true)
            }
        } else if cur_act == <S as WeightArray<Option<bool>>>::THRESHOLD {
            // go up 1 to activate
            if value_mask {
                if value_sign {
                    S::map(weights, &input, |w, input| {
                        !((input ^ w.sign()) & w.mask()) & !input
                    })
                } else {
                    S::map(weights, &input, |w, input| {
                        !((input ^ w.sign()) & w.mask()) & input
                    })
                }
            } else {
                S::map(weights, &input, |w, input| (input ^ !w.sign()) & w.mask())
            }
        } else if cur_act == (<S as WeightArray<Option<bool>>>::THRESHOLD + 1) {
            // go down 1 to deactivate
            if value_mask {
                if value_sign {
                    S::map(weights, &input, |w, input| {
                        !(input & !(w.sign() & w.mask()))
                    })
                } else {
                    S::map(weights, &input, |w, input| {
                        (!((input ^ w.sign()) & w.mask()) & w.mask()) | input
                    })
                }
            } else {
                S::map(weights, &input, |w, input| !((input ^ w.sign()) & w.mask()))
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

        if (cur_act + 2) <= <S as WeightArray<Option<bool>>>::THRESHOLD {
            S::blit(false)
        } else if cur_act > (<S as WeightArray<Option<bool>>>::THRESHOLD + 2) {
            S::blit(true)
        } else if ((cur_act + 1) == <S as WeightArray<Option<bool>>>::THRESHOLD)
            | (cur_act == <S as WeightArray<Option<bool>>>::THRESHOLD)
        {
            S::map(weights, &input, |val, input| {
                (!input ^ val.sign()) & val.mask()
            })
        } else if (cur_act == (<S as WeightArray<Option<bool>>>::THRESHOLD + 2))
            | (cur_act == (<S as WeightArray<Option<bool>>>::THRESHOLD + 1))
        {
            S::map(weights, &input, |val, input| {
                !((input ^ val.sign()) & val.mask())
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

pub trait ZipBitFold<B, X, Y>
where
    Self: Shape + Pack<X> + BitPack<Y>,
{
    fn zip_fold<F: Fn(B, &X, Y) -> B>(
        acc: B,
        a: &<Self as Pack<X>>::T,
        bits: &<Self as BitPack<Y>>::T,
        fold_fn: F,
    ) -> B;
}

impl<B, X, Y, S, const L: usize> ZipBitFold<B, X, Y> for [S; L]
where
    S: ZipBitFold<B, X, Y> + Pack<X> + BitPack<Y>,
{
    fn zip_fold<F: Fn(B, &X, Y) -> B>(
        mut acc: B,
        a: &[<S as Pack<X>>::T; L],
        bits: &[<S as BitPack<Y>>::T; L],
        map_fn: F,
    ) -> B {
        for i in 0..L {
            acc = <S as ZipBitFold<B, X, Y>>::zip_fold(acc, &a[i], &bits[i], &map_fn);
        }
        acc
    }
}

pub trait BitMap<I, O>
where
    Self: Shape + BitPack<I> + Pack<O>,
{
    fn map_mut<F: Fn(I, &mut O)>(
        bits: &<Self as BitPack<I>>::T,
        target: &mut <Self as Pack<O>>::T,
        map_fn: F,
    );
}

impl<I, O, S, const L: usize> BitMap<I, O> for [S; L]
where
    S: Shape + BitPack<I> + Pack<O> + BitMap<I, O>,
{
    fn map_mut<F: Fn(I, &mut O)>(
        bits: &[<S as BitPack<I>>::T; L],
        target: &mut [<S as Pack<O>>::T; L],
        map_fn: F,
    ) {
        for i in 0..L {
            <S as BitMap<I, O>>::map_mut(&bits[i], &mut target[i], &map_fn)
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

pub trait IncrementCounters<T>
where
    Self: Shape + Sized + Pack<T> + BitPack<bool>,
    T: AddAssign + Add<Output = T> + FromBool,
{
    /*
    fn some_option_counted_increment(bits: &<Self as BitPack<bool>>::T, mut counters: (usize, <Self as Pack<T>>::T, u32)) -> (usize, <Self as Pack<T>>::T, u32) {
        counters.0 += 1;
        Self::increment_in_place(bits, &mut counters.1);
        counters
    }
    fn some_option_counted_increment_in_place(bits: &<Self as BitPack<bool>>::T, counters: &mut (usize, <Self as Pack<T>>::T, u32)) {
        counters.0 += 1;
        Self::increment_in_place(bits, &mut counters.1);
    }
    fn none_option_counted_increment(bit: bool, mut counters: (usize, <Self as Pack<T>>::T, u32)) -> (usize, <Self as Pack<T>>::T, u32) {
        counters.0 += 1;
        counters.2 += bit as u32;
        counters
    }
    fn none_option_counted_increment_in_place(bit: bool, counters: &mut (usize, <Self as Pack<T>>::T, u32)) {
        counters.0 += 1;
        counters.2 += bit as u32;
    }
    fn finalize_option_counted_increment(mut counters: (usize, <Self as Pack<u32>>::T, u32)) -> (usize, <Self as Pack<T>>::T) {
        Self::add_in_place(counters.2, &mut counters.1);
        (counters.0, counters.1)
    }
    */
    fn counted_increment_in_place(
        bits: &<Self as BitPack<bool>>::T,
        counters: &mut (T, <Self as Pack<T>>::T),
    ) {
        counters.0 += T::from_u8(1);
        Self::increment_in_place(bits, &mut counters.1);
    }
    fn counted_increment(
        bits: &<Self as BitPack<bool>>::T,
        counters: (T, <Self as Pack<T>>::T),
    ) -> (T, <Self as Pack<T>>::T) {
        (
            counters.0 + T::from_u8(1),
            Self::increment(bits, counters.1),
        )
    }
    fn increment(
        bits: &<Self as BitPack<bool>>::T,
        mut counters: <Self as Pack<T>>::T,
    ) -> <Self as Pack<T>>::T {
        Self::increment_in_place(bits, &mut counters);
        counters
    }
    fn add_in_place(val: T, counters: &mut <Self as Pack<T>>::T);
    fn increment_in_place(bits: &<Self as BitPack<bool>>::T, counters: &mut <Self as Pack<T>>::T);
}

impl<S: IncrementCounters<T>, T: FromBool + Copy + Add<Output = T> + AddAssign, const L: usize>
    IncrementCounters<T> for [S; L]
where
    S: Pack<T> + BitPack<bool>,
{
    fn add_in_place(val: T, counters: &mut <Self as Pack<T>>::T) {
        for i in 0..L {
            S::add_in_place(val, &mut counters[i]);
        }
    }
    #[inline(always)]
    fn increment_in_place(
        bits: &[<S as BitPack<bool>>::T; L],
        counters: &mut [<S as Pack<T>>::T; L],
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
    const THRESHOLD: u32;
}

impl<S, W, const L: usize> BMA<W> for [S; L]
where
    S: BMA<W> + BitPack<bool> + BitPack<W>,
{
    #[inline(always)]
    fn bma(weights: &[<S as BitPack<W>>::T; L], bits: &[<S as BitPack<bool>>::T; L]) -> u32 {
        let mut sum = 0_u32;
        for i in 0..L {
            sum += S::bma(&weights[i], &bits[i]);
        }
        sum
    }
    const THRESHOLD: u32 = <S as BMA<W>>::THRESHOLD * L as u32;
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

pub trait RandInit {
    fn rand_bits<R: Rng, const FREQ: usize>(rng: &mut R) -> Self;
}

impl<T: RandInit, const L: usize> RandInit for [T; L]
where
    [T; L]: Default,
{
    fn rand_bits<R: Rng, const FREQ: usize>(rng: &mut R) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = <T as RandInit>::rand_bits::<R, FREQ>(rng);
        }
        target
    }
}

#[cfg(target_feature = "avx2")]
const PSHUF_BYTE_MASK: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
];

#[cfg(target_arch = "aarch64")]
const TBL_BYTE_MASK_LOW: [u8; 16] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
#[cfg(target_arch = "aarch64")]
const TBL_BYTE_MASK_HIGH: [u8; 16] = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];

#[cfg(target_feature = "avx2")]
const BYTE_BIT_MASK: [u8; 32] = [
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
];

#[cfg(target_arch = "aarch64")]
const NEON_BYTE_BIT_MASK: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

#[cfg(target_arch = "aarch64")]
const NEON_BYTE_SHIFTS: [i8; 16] = [0, -1, -2, -3, -4, -5, -6, -7, 0, -1, -2, -3, -4, -5, -6, -7];

pub trait SIMDincrementCounters
where
    Self: Pack<u32> + BitPack<bool>,
    Self::SIMDbyts: Sized,
    Self::WordShape: Pack<SIMDword32, T = Self::SIMDbyts> + Pack<[(); 32], T = Self>,
{
    type SIMDbyts;
    type WordShape;
    /// This function must not be called more then 256 times!!
    fn simd_increment_in_place(bools: &Self::SIMDbyts, counters: &mut Self::SIMDbyts);
    fn add_to_u32s(byte_counters: &Self::SIMDbyts, counters: &mut <Self as Pack<u32>>::T);
    fn expand_bits(bits: &<Self as BitPack<bool>>::T) -> Self::SIMDbyts;
    fn init_counters() -> Self::SIMDbyts;
}

impl<T: SIMDincrementCounters, const L: usize> SIMDincrementCounters for [T; L]
where
    T::SIMDbyts: Sized + Copy,
{
    type SIMDbyts = [T::SIMDbyts; L];
    type WordShape = [T::WordShape; L];
    #[inline(always)]
    fn simd_increment_in_place(bits: &[T::SIMDbyts; L], counters: &mut [T::SIMDbyts; L]) {
        for i in 0..L {
            T::simd_increment_in_place(&bits[i], &mut counters[i]);
        }
    }
    fn add_to_u32s(byte_counters: &Self::SIMDbyts, counters: &mut <Self as Pack<u32>>::T) {
        for i in 0..L {
            T::add_to_u32s(&byte_counters[i], &mut counters[i]);
        }
    }
    fn expand_bits(bits: &<Self as BitPack<bool>>::T) -> Self::SIMDbyts {
        let mut target: [MaybeUninit<T::SIMDbyts>; L] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..L {
            target[i] = MaybeUninit::new(T::expand_bits(&bits[i]));
        }
        unsafe { target.as_ptr().cast::<[T::SIMDbyts; L]>().read() }
    }
    fn init_counters() -> [T::SIMDbyts; L] {
        let mut target: [MaybeUninit<T::SIMDbyts>; L] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..L {
            target[i] = MaybeUninit::new(T::init_counters());
        }
        unsafe { target.as_ptr().cast::<[T::SIMDbyts; L]>().read() }
    }
}

#[cfg(target_feature = "avx2")]
impl LongDefault for __m256i {
    #[cfg(target_feature = "avx2")]
    fn long_default() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }
}

#[cfg(target_arch = "aarch64")]
impl LongDefault for [uint8x16_t; 2] {
    #[cfg(target_arch = "aarch64")]
    fn long_default() -> [uint8x16_t; 2] {
        let dummy_data = [0u8; 32];
        unsafe { [vld1q_u8(&dummy_data[0]), vld1q_u8(&dummy_data[0])] }
    }
}

#[cfg(target_feature = "avx2")]
pub type SIMDword32 = __m256i;
#[cfg(target_arch = "aarch64")]
pub type SIMDword32 = [uint8x16_t; 2];
#[cfg(not(any(target_arch = "aarch64", target_feature = "avx2")))]
pub type SIMDword32 = [u8; 32];

impl SIMDincrementCounters for [(); 32] {
    #[cfg(target_feature = "avx2")]
    type SIMDbyts = __m256i;
    #[cfg(target_arch = "aarch64")]
    type SIMDbyts = [uint8x16_t; 2];
    #[cfg(not(any(target_arch = "aarch64", target_feature = "avx2")))]
    type SIMDbyts = [u8; 32];

    type WordShape = ();

    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn simd_increment_in_place(word: &__m256i, counters: &mut __m256i) {
        unsafe {
            *counters = _mm256_add_epi8(*counters, *word);
        }
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn simd_increment_in_place(word: &[uint8x16_t; 2], counters: &mut [uint8x16_t; 2]) {
        unsafe {
            counters[0] = vaddq_u8(counters[0], word[0]);
            counters[1] = vaddq_u8(counters[1], word[1]);
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "avx2")))]
    #[inline(always)]
    fn simd_increment_in_place(word: &[u8; 32], counters: &mut [u8; 32]) {
        for i in 0..32 {
            counters[i] += word[i];
        }
    }
    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn add_to_u32s(&byte_counters: &__m256i, counters: &mut [u32; 32]) {
        let mut target = [0u8; 32];
        unsafe {
            _mm256_storeu_si256(
                transmute::<&mut [u8; 32], &mut __m256i>(&mut target),
                byte_counters,
            );
        }

        for i in 0..32 {
            counters[i] += target[i] as u32;
        }
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn add_to_u32s(byte_counters: &[uint8x16_t; 2], counters: &mut [u32; 32]) {
        let mut target = [0u8; 32];
        unsafe {
            vst1q_u8(&mut target[0], byte_counters[0]);
            vst1q_u8(&mut target[16], byte_counters[1]);
        }
        for i in 0..32 {
            counters[i] += target[i] as u32;
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "avx2")))]
    #[inline(always)]
    fn add_to_u32s(byte_counters: &[u8; 32], counters: &mut [u32; 32]) {
        for i in 0..32 {
            counters[i] += byte_counters[i] as u32;
        }
    }
    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn expand_bits(word: &b32) -> __m256i {
        unsafe {
            let mask1 = _mm256_loadu_si256(transmute::<&[u8; 32], &__m256i>(&PSHUF_BYTE_MASK));
            let mask2 = _mm256_loadu_si256(transmute::<&[u8; 32], &__m256i>(&BYTE_BIT_MASK));

            let zeros = _mm256_setzero_si256();
            let ones = _mm256_set1_epi8(0b_1);

            let word: i32 = transmute::<u32, i32>(word.0);

            let expanded = _mm256_set1_epi32(word);
            let shuffled = _mm256_shuffle_epi8(expanded, mask1);
            let masked = _mm256_and_si256(shuffled, mask2);
            let is_zero = _mm256_cmpeq_epi8(masked, zeros);
            let bools = _mm256_andnot_si256(is_zero, ones);
            bools
        }
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn expand_bits(word: &b32) -> [uint8x16_t; 2] {
        unsafe {
            let mask1_low = vld1q_u8(&TBL_BYTE_MASK_LOW[0]);
            let mask1_high = vld1q_u8(&TBL_BYTE_MASK_HIGH[0]);
            let mask2 = vld1q_u8(&NEON_BYTE_BIT_MASK[0]);

            let expanded = vld1q_dup_u32(&word.0);
            let expanded = vreinterpretq_u8_u32(expanded);

            [
                neon_extract_bits(expanded, mask1_low, mask2),
                neon_extract_bits(expanded, mask1_high, mask2),
            ]
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "avx2")))]
    #[inline(always)]
    fn expand_bits(word: &b32) -> [u8; 32] {
        let mut target = [0u8; 32];
        for i in 0..32 {
            target[i] = word.get_bit_u8(i);
        }
        target
    }

    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn init_counters() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn init_counters() -> [uint8x16_t; 2] {
        let dummy_data = [0u8; 32];
        unsafe { [vld1q_u8(&dummy_data[0]), vld1q_u8(&dummy_data[0])] }
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "avx2")))]
    #[inline(always)]
    fn init_counters() -> [u8; 32] {
        [0u8; 32]
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_extract_bits(expanded: uint8x16_t, mask1: uint8x16_t, mask2: uint8x16_t) -> uint8x16_t {
    unsafe {
        let ones = vld1q_dup_u8(&1u8);
        let bit_shifts = vld1q_s8(&NEON_BYTE_SHIFTS[0]);

        let shuffled = vqtbl1q_u8(expanded, mask1);
        let shifted = vqshlq_u8(shuffled, bit_shifts);
        vandq_u8(shifted, ones)
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
//impl_long_default_for_type!((bool, bool));
impl_long_default_for_type!(Option<(bool, bool)>);

pub trait FromBool {
    fn from_bool(b: bool) -> Self;
    fn from_u8(b: u8) -> Self;
    fn to_u64(self) -> u64;
    const ONE: Self;
    const MAX: usize;
}

macro_rules! for_uints {
    ($q_type:ident, $t_type:ident, $b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        impl FromBool for $u_type {
            #[inline(always)]
            fn from_bool(b: bool) -> $u_type {
                b as $u_type
            }
            #[inline(always)]
            fn from_u8(b: u8) -> $u_type {
                b as $u_type
            }
            fn to_u64(self) -> u64 {
                self as u64
            }
            const ONE: $u_type = 1;
            const MAX: usize = <$u_type>::MAX as usize;
        }

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
            pub const ZEROS: $b_type = $b_type(0);
            #[inline(always)]
            pub fn count_ones(self) -> u32 {
                self.0.count_ones()
            }
            #[inline(always)]
            pub fn get_bit(self, index: usize) -> bool {
                ((self.0 >> index) & 1) == 1
            }
            #[inline(always)]
            pub fn get_bit_u8(self, index: usize) -> u8 {
                ((self.0 >> index) & 1) as u8
            }
            #[inline(always)]
            pub fn set_bit_in_place(&mut self, index: usize, value: bool) {
                self.0 &= !(1 << index);
                self.0 |= ((value as $u_type) << index);
            }
        }
        impl $t_type {
            #[inline(always)]
            fn get_trit(&self, index: usize) -> Option<bool> {
                let sign = (self.0 >> index) & 1 == 1;
                let magn = (self.1 >> index) & 1 == 1;
                Some(sign).filter(|_| magn)
            }
            #[inline(always)]
            fn set_trit_in_place(&mut self, index: usize, val: Option<bool>) {
                self.0 &= !(1 << index);
                self.0 |= ((val.unwrap_or(false) as $u_type) << index);

                self.1 &= !(1 << index);
                self.1 |= ((val.is_some() as $u_type) << index);
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
            #[inline(always)]
            fn bma(&weights: &$b_type, &rhs: &$b_type) -> u32 {
                (weights.0 ^ rhs.0).count_ones()
            }
            const THRESHOLD: u32 = $len / 2;
        }
        impl BMA<Option<bool>> for [(); $len] {
            #[inline(always)]
            fn bma(&weights: &$t_type, &rhs: &$b_type) -> u32 {
                weights.1.count_zeros() + ((weights.0 ^ rhs.0) & weights.1).count_ones() * 2
            }
            const THRESHOLD: u32 = $len;
        }

        impl WBBZM<bool> for [(); $len] {
            #[inline(always)]
            fn map<F: Fn($b_type, $u_type) -> $u_type>(
                weights: &$b_type,
                rhs: &$b_type,
                map_fn: F,
            ) -> $b_type {
                $b_type(map_fn(*weights, rhs.0))
            }
        }
        impl WBBZM<Option<bool>> for [(); $len] {
            #[inline(always)]
            fn map<F: Fn($t_type, $u_type) -> $u_type>(
                weights: &$t_type,
                rhs: &$b_type,
                map_fn: F,
            ) -> $b_type {
                $b_type(map_fn(*weights, rhs.0))
            }
        }
        impl WBBZM<(bool, bool)> for [(); $len] {
            #[inline(always)]
            fn map<F: Fn($q_type, $u_type) -> $u_type>(
                weights: &$q_type,
                rhs: &$b_type,
                map_fn: F,
            ) -> $b_type {
                $b_type(map_fn(*weights, rhs.0))
            }
        }

        impl<B, X> ZipBitFold<B, X, bool> for [(); $len] {
            fn zip_fold<F: Fn(B, &X, bool) -> B>(
                mut acc: B,
                a: &[X; $len],
                bits: &$b_type,
                fold_fn: F,
            ) -> B {
                for i in 0..$len {
                    acc = fold_fn(acc, &a[i], bits.get_bit(i));
                }
                acc
            }
        }
        impl<O> BitMap<bool, O> for [(); $len] {
            fn map_mut<F: Fn(bool, &mut O)>(bits: &$b_type, target: &mut [O; $len], map_fn: F) {
                for i in 0..$len {
                    map_fn(bits.get_bit(i), &mut target[i]);
                }
            }
        }
        impl RandInit for $b_type {
            fn rand_bits<R: Rng, const FREQ: usize>(rng: &mut R) -> $b_type {
                let mut target: $u_type = rng.gen();
                for _ in 0..FREQ {
                    target &= rng.gen::<$u_type>();
                }
                $b_type(target)
            }
        }
        impl PackedIndexSGet<bool> for [(); $len] {
            #[inline(always)]
            fn get(&array: &$b_type, (i, _): (u8, ())) -> bool {
                array.get_bit(i as usize)
            }
            #[inline(always)]
            fn set_in_place(array: &mut $b_type, (i, _): (u8, ()), val: bool) {
                array.set_bit_in_place(i as usize, val);
            }
        }
        impl PackedIndexSGet<Option<bool>> for [(); $len] {
            #[inline(always)]
            fn get(&array: &$t_type, (i, _): (u8, ())) -> Option<bool> {
                array.get_trit(i as usize)
            }
            #[inline(always)]
            fn set_in_place(array: &mut $t_type, (i, _): (u8, ()), val: Option<bool>) {
                array.set_trit_in_place(i as usize, val);
            }
        }

        impl<T: Add<Output = T> + AddAssign + Copy + AddAssign<T> + FromBool> IncrementCounters<T>
            for [(); $len]
        {
            fn add_in_place(val: T, counters: &mut [T; $len]) {
                for i in 0..$len {
                    counters[i] += val;
                }
            }
            #[inline(always)]
            fn increment_in_place(&bits: &$b_type, counters: &mut [T; $len]) {
                for i in 0..$len {
                    //counters[i] += T::from_bool(bits.get_bit(i));
                    counters[i] += T::from_u8(bits.get_bit_u8(i));
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
            #[inline(always)]
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

        impl<I> PackedMap<I, bool> for [(); $len] {
            fn map<F: Fn(&I) -> bool>(input: &[I; $len], map_fn: F) -> $b_type {
                let mut target = <$b_type>::default();
                for i in 0..$len {
                    target.set_bit_in_place(i, map_fn(&input[i]));
                }
                target
            }
        }
        impl<I> PackedMap<I, Option<bool>> for [(); $len] {
            fn map<F: Fn(&I) -> Option<bool>>(input: &[I; $len], map_fn: F) -> $t_type {
                let mut target = <$t_type>::default();
                for i in 0..$len {
                    target.set_trit_in_place(i, map_fn(&input[i]));
                }
                target
            }
        }

        impl Blit<bool> for [(); $len] {
            #[inline(always)]
            fn blit(val: bool) -> $b_type {
                $b_type((Wrapping(0) - Wrapping(val as $u_type)).0)
            }
        }
        impl Blit<Option<bool>> for [(); $len] {
            #[inline(always)]
            fn blit(val: Option<bool>) -> $t_type {
                $t_type(
                    (Wrapping(0) - Wrapping(val.unwrap_or(false) as $u_type)).0,
                    (Wrapping(0) - Wrapping(val.is_some() as $u_type)).0,
                )
            }
        }
        impl Blit<(bool, bool)> for [(); $len] {
            #[inline(always)]
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

pub trait GetBit {
    fn bit(self, i: usize) -> bool;
}

impl GetBit for usize {
    fn bit(self, i: usize) -> bool {
        ((self >> i) & 1) == 1
    }
}

#[cfg(test)]
mod tests {
    use super::{
        b128, b16, b32, b8, t128, t16, t32, t8, BitScaler, SIMDincrementCounters, WeightArray,
    };
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
                        let acts = <$s as WeightArray<$w>>::acts(&weights, &inputs, val);
                        let true_acts = <$s as WeightArray<$w>>::acts_slow(&weights, &inputs, val);
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

                    let acts = <$s as WeightArray<$w>>::input_acts(&weights, &inputs);
                    let true_acts = <$s as WeightArray<$w>>::input_acts_slow(&weights, &inputs);
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
                        let mut loss_deltas = <$s as WeightArray<$w>>::loss_deltas(
                            &weights,
                            &inputs,
                            threshold,
                            |x| x as i64 - null_act as i64,
                        );
                        loss_deltas.sort();
                        let mut true_loss_deltas = <$s as WeightArray<$w>>::loss_deltas_slow(
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

    #[test]
    fn simd_expand_bits() {
        let mut rng = Hc128Rng::seed_from_u64(0);

        //let mut simd_counter = <[(); 32] as SIMDincrementCounters>::init_counters();

        (0..10_000).for_each(|_| {
            let word: b32 = rng.gen();
            let mut test_counter = [0u32; 32];
            for i in 0..32 {
                test_counter[i] += word.get_bit_u8(i) as u32;
            }
            let mut simd_counter = [0u32; 32];

            let expanded = <[(); 32] as SIMDincrementCounters>::expand_bits(&word);
            <[(); 32] as SIMDincrementCounters>::add_to_u32s(&expanded, &mut simd_counter);
            assert_eq!(simd_counter, test_counter);
        });
    }
}
