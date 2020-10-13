use crate::bits::{BitArray, BitMapPack, MaskedDistance, TritArray, Weight, WeightArray};
use crate::shape::{Element, IndexGet, IndexMap, Map, Shape};
use rayon::prelude::*;
use std::iter;
use std::marker::PhantomData;

pub trait BaseCache
where
    Self::ChanCache: ChanCache,
{
    type ChanIndex;
    type ChanValue;
    type Input;
    type ChanCache;
    fn chan_cache(&self, chan_val: Self::ChanValue, chan_index: Self::ChanIndex) -> Self::ChanCache;
}

pub trait ChanCache {
    type Mutation;
    fn loss(&self, mutation: &Self::Mutation) -> u64;
}

#[derive(Copy, Clone)]
pub enum LayerIndex<H: Copy, T: Copy> {
    Head(H),
    Tail(T),
}

pub trait Layer<I>
where
    Self::BaseCacheType: BaseCache,
{
    type WeightIndex;
    type BaseCacheType;
    fn loss(&self, input: &I, class: usize) -> u64;
    //fn grads(&self, input: &I, class: usize) -> Vec<(Self::WeightIndex, Option<bool>, u64)>;
    fn grads_slow(&self, input: &I, class: usize) -> Vec<(Self::WeightIndex, Option<bool>, u64)>;
    //fn base_cache(&self, input: &I, class: usize) -> Self::BaseCacheType;
}

pub struct FcMSE<I: BitArray, W: Weight<I>, const C: usize> {
    pub fc: [<W as Weight<I>>::Weights; C],
}

impl<I: BitArray, W: Weight<I>, const C: usize> FcMSE<I, W, C> {
    fn mutate(mut self, value: W, class: usize, index: <I::BitShape as Shape>::Index) -> Self {
        self
    }
}

impl<I: BitArray, W: Weight<I>, const C: usize> Layer<I> for FcMSE<I, W, C>
where
    bool: Element<I::BitShape>,
    I::BitShape: IndexMap<bool, ()>,
    I::TritArrayType: MaskedDistance + Copy,
    [u32; C]: Default,
    W: Weight<I> + Copy,
    <W as Weight<I>>::Weights: WeightArray<W>,
{
    type WeightIndex = (u8, <I::BitShape as Shape>::Index);
    type BaseCacheType = FcTritMseBaseCache<I, C>;
    fn loss(&self, input: &I, class: usize) -> u64 {
        self.fc
            .iter()
            .enumerate()
            .map(|(i, trits)| {
                let act = trits.bma(input);
                let target = I::BitShape::N as u32 * (class == i) as u32;
                let dist = target.saturating_sub(act) | act.saturating_sub(target);
                (dist as u64).pow(2)
            })
            .sum()
    }
    /*
    fn grads(&self, input: &I, class: usize) -> Vec<(Self::WeightIndex, Option<bool>, u64)> {
        let sum_loss = self.loss(input, class);
        self.fc
            .iter()
            .enumerate()
            .map(|(c, trits)| {
                let target = (I::BitShape::N as u32 / 2) * (class == c) as u32;
                let act = trits.masked_distance(input);
                let dist = target.saturating_sub(act) | act.saturating_sub(target);
                let else_loss = sum_loss - (dist as u64).pow(2);

                let mut losses = [0u64; 4];
                for m in 0..2 {
                    let new_act = act.saturating_add(m as u32 + 1);
                    let dist = target.saturating_sub(new_act) | new_act.saturating_sub(target);
                    losses[m] = else_loss + (dist as u64).pow(2);
                }
                for m in 0..2 {
                    let new_act = act.saturating_sub(m as u32 + 1);
                    let dist = target.saturating_sub(new_act) | new_act.saturating_sub(target);
                    losses[2 + m] = else_loss + (dist as u64).pow(2);
                }
                dbg!(losses);

                // we have 5 bits:
                // - input sign
                // - cur sign
                // - cur magn
                // - targ sign
                // - targ magn
                let loss_lut = [0u64; 32];
                //for i in [false, true] {}

                <I::BitShape as Shape>::indices().map(move |i| {
                    let trit = trits.get_trit(i);
                    let dist = target.saturating_sub(act) | act.saturating_sub(target);
                    let full_loss = else_loss + (dist as u64).pow(2);
                    ((c as u8, i), trit, full_loss)
                })
            })
            .flatten()
            .collect()
    }
    */
    fn grads_slow(&self, input: &I, class: usize) -> Vec<(Self::WeightIndex, Option<bool>, u64)> {
        let null_loss = self.loss(input, class);
        I::BitShape::indices()
            .map(|i| {
                (0..C).map(|c| {
                    let index = (c as u8, i);
                    let mut weights = *self;
                    let loss = null_loss;
                    let value = None;
                    (index, value, loss)
                })
            })
            .flatten()
            .collect()
    }
    //fn base_cache(&self, input: &I, class: usize) -> Self::BaseCacheType {
    //    let mut acts = <[u32; C]>::default();
    //    for c in 0..C {
    //        acts[c] = self.fc[c].masked_distance(input);
    //    }

    //    FcTritMseBaseCache { input_type: PhantomData::default(), acts }
    //}
}

pub struct FcTritMseBaseCache<I: BitArray, const C: usize> {
    input_type: PhantomData<I>,
    acts: [u32; C],
}

impl<I: BitArray, const C: usize> BaseCache for FcTritMseBaseCache<I, C> {
    type ChanIndex = <<I as BitArray>::BitShape as Shape>::Index;
    type ChanValue = bool;
    type Input = I;
    type ChanCache = FcTritMseChanCache<I, C>;
    fn chan_cache(&self, chan_val: Self::ChanValue, chan_index: Self::ChanIndex) -> Self::ChanCache {
        FcTritMseChanCache { input_type: PhantomData::default() }
    }
}

pub struct FcTritMseChanCache<I, const C: usize> {
    input_type: PhantomData<I>,
}

impl<I: BitArray, const C: usize> ChanCache for FcTritMseChanCache<I, C> {
    type Mutation = bool;
    fn loss(&self, mutation: &bool) -> u64 {
        0u64
    }
}
