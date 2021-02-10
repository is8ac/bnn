use crate::bits::{
    BitPack, BitScaler, GetBit, IncrementCounters, PackedIndexMap, PackedIndexSGet, PackedMap,
    WeightArray, BMA,
};
use crate::image2d::{
    Conv, ImageShape, PixelFold, PixelIndexSGet, PixelMap, PixelPack, PixelZipMap,
    SegmentedConvFold, SegmentedPixelFold,
};
use crate::shape::{IndexGet, Map, Pack, Shape, ZipMap};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rayon::prelude::*;
use std::fmt;
use std::iter;
use std::marker::PhantomData;

pub struct NullIter<T> {
    item_type: PhantomData<T>,
}

impl<T> Iterator for NullIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        None
    }
}

#[derive(Debug)]
pub struct LayerIter<I: Shape, O: Shape, H> {
    cur_output_index: Option<O::Index>,
    input_index: I::IndexIter,
    output_index: O::IndexIter,
    head_iter: H,
}

impl<I: Shape, O: Shape, H> LayerIter<I, O, H> {
    fn new(head_iter: H) -> Self {
        let mut out_iter = O::indices();
        LayerIter {
            cur_output_index: out_iter.next(),
            input_index: I::indices(),
            output_index: out_iter,
            head_iter: head_iter,
        }
    }
}

impl<I: Shape, O: Shape, H: Iterator> Iterator for LayerIter<I, O, H>
where
    H::Item: Copy,
{
    type Item = LayerIndex<(O::Index, I::Index), H::Item>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(o) = self.cur_output_index {
            if let Some(i) = self.input_index.next() {
                Some(LayerIndex::Head((o, i)))
            } else {
                self.cur_output_index = self.output_index.next();
                self.input_index = I::indices();
                self.next()
            }
        } else {
            self.head_iter.next().map(LayerIndex::Tail)
        }
    }
}

pub trait Cache<I> {
    fn loss_delta(&self, input: &I) -> i64;
}

#[derive(Copy, Clone, Debug, Ord, Eq, PartialEq, PartialOrd, Hash)]
pub enum LayerIndex<H: Copy, T: Copy> {
    Head(H),
    Tail(T),
}

pub trait IndexDepth {
    fn depth(&self) -> usize;
}

impl<H: Copy, T: IndexDepth + Copy> IndexDepth for LayerIndex<H, T> {
    fn depth(&self) -> usize {
        match self {
            LayerIndex::Head(_) => 0,
            LayerIndex::Tail(t) => t.depth() + 1,
        }
    }
}

impl<I: IndexDepth, const X: usize, const Y: usize> IndexDepth for SegmentedAvgPoolIndex<I, X, Y> {
    fn depth(&self) -> usize {
        self.index.depth()
    }
}

impl<I> IndexDepth for (usize, I) {
    fn depth(&self) -> usize {
        0
    }
}

pub struct ChainedIndexIter<H, T> {
    head: H,
    tail: T,
}

impl<H: Iterator, T: Iterator> Iterator for ChainedIndexIter<H, T>
where
    H::Item: Copy,
    T::Item: Copy,
{
    type Item = LayerIndex<H::Item, T::Item>;
    fn next(&mut self) -> Option<LayerIndex<H::Item, T::Item>> {
        self.head
            .next()
            .map(|h| LayerIndex::Head(h))
            .or_else(|| self.tail.next().map(LayerIndex::Tail))
    }
}

pub trait Model<I, const C: usize>
where
    Self: Sized + Copy,
    Self::Weight: BitScaler,
    Self::IndexIter: Iterator<Item = Self::Index>,
    Self::Index: Copy + fmt::Debug,
{
    type Index;
    type Weight;
    type IndexIter;
    type Output;
    const N_PARAMS: usize;
    /// Rand init
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn apply(&self, input: &I) -> Self::Output;
    /// iter of all weight indices
    fn indices() -> Self::IndexIter;
    fn mutate_in_place(&mut self, index: Self::Index, weight: Self::Weight);
    fn mutate(mut self, index: Self::Index, weight: Self::Weight) -> Self {
        self.mutate_in_place(index, weight);
        self
    }
    fn top_act(&self, input: &I) -> usize;
    fn is_correct(&self, input: &I, class: usize) -> bool {
        self.top_act(input) == class
    }
    /// loss of the model
    fn loss(&self, input: &I, class: usize) -> u64;
    /// loss deltas for all mutations of all weights.
    fn loss_deltas(
        &self,
        input: &I,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)>;
    /// same as losses but a lot slower.
    fn loss_deltas_slow(
        &self,
        input: &I,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)> {
        let n_indicis = Self::indices().count();
        let null_loss = self.loss(input, class) as i64;
        <Self::Weight as BitScaler>::states()
            .iter()
            .map(|&w| {
                Self::indices()
                    .map(|i| {
                        let delta = self.mutate(i, w).loss(input, class) as i64 - null_loss;
                        (i, w, delta)
                    })
                    .filter(|(_, _, l)| l.abs() as u64 > threshold)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
}

pub trait ObjCache<I, const C: usize>
where
    Self: Model<I, C>,
    Self::ChanCache: Cache<Self::InputValue>,
{
    type Cache;
    type ChanCache;
    type InputIndex;
    type InputValue;
    // init the cache such that we can subtract input elements from it.
    fn cache(&self, input: &I, class: usize) -> Self::Cache;
    // subtract an element from the cache and prepare the cache for addition of a new input element.
    fn subtract_input(
        &self,
        cache: &Self::Cache,
        chan_index: Self::InputIndex,
        cur_value: &Self::InputValue,
    ) -> Self::ChanCache;
}

#[derive(Copy, Clone, Debug)]
pub struct FcMSE<S, W, const C: usize>
where
    [<S as BitPack<W>>::T; C]: Copy + std::fmt::Debug,
    S: Shape + BitPack<W>,
{
    pub fc: [<S as BitPack<W>>::T; C],
}

pub struct FcMSEindexIter<S: Shape, const C: usize> {
    class_index: usize,
    input_iter: S::IndexIter,
}

impl<S: Shape, const C: usize> Iterator for FcMSEindexIter<S, C> {
    type Item = (usize, S::Index);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.input_iter.next() {
            Some((self.class_index, i))
        } else {
            self.class_index += 1;
            if self.class_index >= C {
                None
            } else {
                self.input_iter = S::indices();
                self.input_iter.next().map(|i| (self.class_index, i))
            }
        }
    }
}

pub struct FcMSEcache<const C: usize> {
    cache: [(u32, u32); C],
    null_loss: i64,
}

pub struct FcMSEchanCache<W, const C: usize> {
    cache: [(W, u32, u32); C],
    null_loss: i64,
}

impl<W: BitScaler, const C: usize> Cache<bool> for FcMSEchanCache<W, C> {
    fn loss_delta(&self, &input: &bool) -> i64 {
        self.cache
            .iter()
            .map(|&(w, sum, target_act)| {
                let act = sum + w.bma(input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                (dist as u64).pow(2)
            })
            .sum::<u64>() as i64
            - self.null_loss
    }
}

impl<S, W, const C: usize> Model<<S as BitPack<bool>>::T, C> for FcMSE<S, W, C>
where
    S: Shape
        + PackedIndexMap<bool>
        + Copy
        + BitPack<W>
        + BitPack<bool>
        + BMA<W>
        + PackedIndexSGet<W>
        + PackedIndexSGet<bool>
        + BMA<W>
        + WeightArray<W>,
    W: BitScaler,
    <S as Shape>::Index: Copy,
    [(W, u32, u32); C]: Default,
    [(u32, u32); C]: Default,
    [u32; C]: Default,
    Standard: Distribution<[<S as BitPack<W>>::T; C]>,
    <S as BitPack<bool>>::T: Copy,
    <S as BitPack<W>>::T: Copy + std::fmt::Debug,
    S::Index: fmt::Debug,
{
    type Index = (usize, <S as Shape>::Index);
    type Weight = W;
    type IndexIter = FcMSEindexIter<S, C>;
    type Output = [u32; C];
    const N_PARAMS: usize = 10 * S::N;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        FcMSE { fc: rng.gen() }
    }
    fn apply(&self, input: &<S as BitPack<bool>>::T) -> [u32; C] {
        let mut target = <[u32; C]>::default();
        for c in 0..C {
            target[c] = S::bma(&self.fc[c], input);
        }
        target
    }
    fn indices() -> FcMSEindexIter<S, C> {
        FcMSEindexIter {
            class_index: 0,
            input_iter: S::indices(),
        }
    }
    fn mutate_in_place(&mut self, (head, tail): (usize, <S as Shape>::Index), weight: W) {
        S::set_in_place(&mut self.fc[head], tail, weight);
    }
    fn top_act(&self, input: &<S as BitPack<bool>>::T) -> usize {
        self.fc
            .iter()
            .map(|w| S::bma(w, input))
            .enumerate()
            .max_by_key(|(_, a)| *a)
            .unwrap()
            .0
    }
    fn loss(&self, input: &<S as BitPack<bool>>::T, class: usize) -> u64 {
        self.fc
            .iter()
            .enumerate()
            .map(|(c, w)| {
                let target_act = (c == class) as u32 * <S as WeightArray<W>>::MAX;
                let act = S::bma(w, input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                (dist as u64).pow(2)
            })
            .sum()
    }
    fn loss_deltas(
        &self,
        input: &<S as BitPack<bool>>::T,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, W, i64)> {
        self.fc
            .iter()
            .enumerate()
            .map(|(c, w)| {
                let target_act = (c == class) as u32 * <S as WeightArray<W>>::MAX;
                let act = S::bma(w, input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                let class_null_loss = (dist as u64).pow(2) as i64;
                <S as WeightArray<W>>::loss_deltas(&w, input, threshold, |act| {
                    ((act.saturating_sub(target_act) | target_act.saturating_sub(act)) as u64)
                        .pow(2) as i64
                        - class_null_loss
                })
                .iter()
                .map(|(i, w, l)| ((c, *i), *w, *l))
                .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
}

impl<S, W, const C: usize> ObjCache<<S as BitPack<bool>>::T, C> for FcMSE<S, W, C>
where
    Self: Model<<S as BitPack<bool>>::T, C>,
    S: Shape
        + PackedIndexMap<bool>
        + Copy
        + BitPack<W>
        + BitPack<bool>
        + BMA<W>
        + PackedIndexSGet<W>
        + PackedIndexSGet<bool>
        + BMA<W>
        + WeightArray<W>,
    W: BitScaler,
    <S as Shape>::Index: Copy,
    [(W, u32, u32); C]: Default,
    [(u32, u32); C]: Default,
    [u32; C]: Default,
    Standard: Distribution<[<S as BitPack<W>>::T; C]>,
    <S as BitPack<bool>>::T: Copy,
    <S as BitPack<W>>::T: Copy + std::fmt::Debug,
{
    type Cache = FcMSEcache<C>;
    type ChanCache = FcMSEchanCache<W, C>;
    type InputValue = bool;
    type InputIndex = <S as Shape>::Index;
    fn cache(&self, input: &<S as BitPack<bool>>::T, class: usize) -> FcMSEcache<C> {
        let mut target = <[(u32, u32); C]>::default();
        for c in 0..C {
            target[c] = (
                S::bma(&self.fc[c], input),
                (c == class) as u32 * <S as WeightArray<W>>::MAX,
            );
        }
        FcMSEcache {
            cache: target,
            null_loss: self.loss(input, class) as i64,
        }
    }
    fn subtract_input(
        &self,
        cache: &FcMSEcache<C>,
        chan_index: Self::InputIndex,
        &cur_value: &bool,
    ) -> FcMSEchanCache<W, C> {
        let mut target = <[(W, u32, u32); C]>::default();
        for c in 0..C {
            let weight = S::get(&self.fc[c], chan_index);
            target[c] = (
                weight,
                cache.cache[c].0 - weight.bma(cur_value),
                cache.cache[c].1,
            );
        }
        FcMSEchanCache {
            cache: target,
            null_loss: cache.null_loss,
        }
    }
}

#[derive(Copy, Clone)]
pub struct FC<I, W, O: Shape, H: Model<<O as BitPack<bool>>::T, C>, const C: usize>
where
    O: BitPack<bool> + Pack<<I as BitPack<W>>::T>,
    I: Shape + BitPack<W>,
    <O as Pack<<I as BitPack<W>>::T>>::T: Copy,
{
    fc: <O as Pack<<I as BitPack<W>>::T>>::T,
    tail: H,
}

pub struct FCcache<O: Shape, H: Model<<O as BitPack<bool>>::T, C>, const C: usize>
where
    O: BitPack<bool> + Pack<u32>,
{
    is_alive: bool,
    sums: <O as Pack<u32>>::T,
    true_class: usize,
    null_loss: i64,
    head: PhantomData<H>,
}

pub struct FCchanCache<I, W, O: Shape, H: Model<<O as BitPack<bool>>::T, C>, const C: usize>
where
    O: Pack<(W, u32)> + BitPack<bool>,
{
    input_shape: PhantomData<I>,
    is_alive: bool,
    sums: <O as Pack<(W, u32)>>::T,
    true_class: usize,
    null_loss: i64,
    head: H,
}

impl<
        I,
        W,
        O: Shape + PackedMap<(W, u32), bool>,
        H: Model<<O as BitPack<bool>>::T, C>,
        const C: usize,
    > Cache<bool> for FCchanCache<I, W, O, H, C>
where
    W: BitScaler,
    O: Pack<(W, u32)> + BitPack<bool>,
    I: WeightArray<W>,
    I: BMA<W> + PackedIndexMap<bool> + PackedIndexSGet<bool> + PackedIndexSGet<W>,
    <I as BitPack<W>>::T: Copy,
    <I as BitPack<bool>>::T: Copy,
{
    fn loss_delta(&self, &input: &bool) -> i64 {
        let hidden = <O as PackedMap<(W, u32), bool>>::map(&self.sums, |(w, sum)| {
            (sum + w.bma(input)) > <I as WeightArray<W>>::THRESHOLD
        });
        self.head.loss(&hidden, self.true_class) as i64 - self.null_loss
    }
}

impl<I, W, O, H, const C: usize> Model<<I as BitPack<bool>>::T, C> for FC<I, W, O, H, C>
where
    I: WeightArray<W>
        + Shape
        + PackedIndexMap<bool>
        + BitPack<bool>
        + PackedIndexSGet<W>
        + BMA<W>
        + PackedIndexSGet<bool>,
    W: BitScaler,
    H: Model<<O as BitPack<bool>>::T, C, Weight = W>,
    H: ObjCache<<O as BitPack<bool>>::T, C, InputIndex = O::Index, InputValue = bool>,
    O: Shape
        + Default
        + ZipMap<<I as BitPack<W>>::T, u32, (W, u32)>
        + Map<<I as BitPack<W>>::T, u32>
        + PackedMap<(W, u32), bool>
        + Pack<u32>
        + Pack<(W, u32)>
        + PackedMap<<I as BitPack<W>>::T, bool>
        + IndexGet<<I as BitPack<W>>::T>
        + PackedIndexSGet<bool>,
    Self: Copy,
    <I as BitPack<W>>::T: Copy,
    <I as BitPack<bool>>::T: Copy,
    <O as Pack<<I as BitPack<W>>::T>>::T: Copy,
    Standard: Distribution<<O as Pack<<I as BitPack<W>>::T>>::T>,
    LayerIndex<(O::Index, <I as Shape>::Index), H::Index>: fmt::Debug,
{
    type Index = LayerIndex<(O::Index, <I as Shape>::Index), H::Index>;
    type Weight = W;
    type IndexIter = LayerIter<I, O, H::IndexIter>;
    type Output = <O as BitPack<bool>>::T;
    const N_PARAMS: usize = I::N * O::N + H::N_PARAMS;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        FC {
            fc: rng.gen(),
            tail: H::rand(rng),
        }
    }
    fn apply(&self, input: &<I as BitPack<bool>>::T) -> <O as BitPack<bool>>::T {
        <O as PackedMap<<I as BitPack<W>>::T, bool>>::map(&self.fc, |w| {
            <I as WeightArray<W>>::act(&w, input)
        })
    }
    fn indices() -> Self::IndexIter {
        LayerIter::new(H::indices())
    }
    fn mutate_in_place(&mut self, index: Self::Index, weight: W) {
        match index {
            LayerIndex::Head((o, i)) => {
                I::set_in_place(O::index_get_mut(&mut self.fc, o), i, weight)
            }
            LayerIndex::Tail(i) => self.tail.mutate_in_place(i, weight),
        }
    }
    fn top_act(&self, input: &<I as BitPack<bool>>::T) -> usize {
        let hidden = self.apply(input);
        self.tail.top_act(&hidden)
    }
    fn loss(&self, input: &<I as BitPack<bool>>::T, class: usize) -> u64 {
        let hidden = self.apply(input);
        self.tail.loss(&hidden, class)
    }
    fn loss_deltas(
        &self,
        inputs: &<I as BitPack<bool>>::T,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, W, i64)> {
        let hidden = self.apply(inputs);
        // for a given output, if we flip it, what is the new loss?
        // for fc, we will only call it once, but for conv, we will call it many time for a given channel.
        let cache = self.tail.cache(&hidden, class);
        O::indices()
            .map(|o| {
                let input = O::get(&hidden, o);
                let weight_array = O::index_get(&self.fc, o);

                let chan_cache = self.tail.subtract_input(&cache, o, &input);
                let deltas = [chan_cache.loss_delta(&false), chan_cache.loss_delta(&true)];

                <I as WeightArray<W>>::loss_deltas(&weight_array, &inputs, threshold, |act| {
                    deltas[(act > <I as WeightArray<W>>::THRESHOLD) as usize]
                })
                .iter()
                .map(|&(i, w, l)| (LayerIndex::Head((o, i)), w, l))
                .collect::<Vec<_>>()
            })
            .flatten()
            .chain(
                self.tail
                    .loss_deltas(&hidden, threshold, class)
                    .iter()
                    .map(|&(i, w, l)| (LayerIndex::Tail(i), w, l)),
            )
            .collect()
    }
}

impl<I, W, O, H, const C: usize> ObjCache<<I as BitPack<bool>>::T, C> for FC<I, W, O, H, C>
where
    I: WeightArray<W>
        + Shape
        + PackedIndexMap<bool>
        + BitPack<bool>
        + PackedIndexSGet<W>
        + BMA<W>
        + PackedIndexSGet<bool>,
    <I as Shape>::Index: fmt::Debug,
    W: BitScaler,
    H: Model<<O as BitPack<bool>>::T, C, Weight = W>,
    H: ObjCache<<O as BitPack<bool>>::T, C, InputIndex = O::Index, InputValue = bool>,
    O: Shape
        + Default
        + ZipMap<<I as BitPack<W>>::T, u32, (W, u32)>
        + Map<<I as BitPack<W>>::T, u32>
        + PackedMap<(W, u32), bool>
        + Pack<u32>
        + Pack<(W, u32)>
        + PackedMap<<I as BitPack<W>>::T, bool>
        + IndexGet<<I as BitPack<W>>::T>
        + PackedIndexSGet<bool>,
    Self: Copy,
    Self::ChanCache: Cache<bool>,
    <I as BitPack<W>>::T: Copy,
    <O as Shape>::Index: fmt::Debug,
    <I as BitPack<bool>>::T: Copy,
    <O as Pack<<I as BitPack<W>>::T>>::T: Copy,
    Standard: Distribution<<O as Pack<<I as BitPack<W>>::T>>::T>,
{
    type Cache = FCcache<O, H, C>;
    type ChanCache = FCchanCache<I, W, O, H, C>;
    type InputIndex = <I as Shape>::Index;
    type InputValue = bool;
    fn cache(&self, input: &<I as BitPack<bool>>::T, class: usize) -> Self::Cache {
        FCcache {
            is_alive: true,
            sums: <O as Map<<I as BitPack<W>>::T, u32>>::map(&self.fc, |weights| {
                I::bma(weights, input)
            }),
            true_class: class,
            null_loss: self.loss(input, class) as i64,
            head: PhantomData::default(),
        }
    }
    fn subtract_input(
        &self,
        cache: &Self::Cache,
        chan_index: Self::InputIndex,
        &cur_value: &bool,
    ) -> Self::ChanCache {
        FCchanCache {
            input_shape: PhantomData::default(),
            is_alive: true,
            sums: <O as ZipMap<<I as BitPack<W>>::T, u32, (W, u32)>>::zip_map(
                &self.fc,
                &cache.sums,
                |weights, sum| {
                    let w = I::get(weights, chan_index);
                    (w, sum - w.bma(cur_value))
                },
            ),
            true_class: cache.true_class,
            null_loss: cache.null_loss,
            head: self.tail,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Conv2D<
    IS,
    IPS,
    OPS,
    W,
    H,
    const IMPL: usize,
    const PX: usize,
    const PY: usize,
    const C: usize,
> where
    IS: PixelPack<<IPS as BitPack<bool>>::T> + PixelPack<<OPS as BitPack<bool>>::T>,
    OPS: Shape + Pack<[[<IPS as BitPack<W>>::T; PY]; PX]> + BitPack<bool>,
    IPS: Shape + BitPack<W> + BitPack<bool>,
    H: Model<<IS as PixelPack<<OPS as BitPack<bool>>::T>>::I, C>,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy + fmt::Debug,
    IS: PixelPack<<IPS as BitPack<bool>>::T>,
{
    pub kernel: <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T,
    pub tail: H,
    image_shape: PhantomData<IS>,
}

impl<IS, IPS, OPS, W, H, const IMPL: usize, const PX: usize, const PY: usize, const C: usize>
    Model<<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I, C>
    for Conv2D<IS, IPS, OPS, W, H, IMPL, PX, PY, C>
where
    IPS: Shape + Copy + PackedIndexMap<bool> + BitPack<bool> + BitPack<W> + PackedIndexSGet<bool>,
    OPS: Shape
        + Copy
        + PackedMap<[[<IPS as BitPack<W>>::T; PY]; PX], bool>
        + BitPack<bool>
        + IndexGet<[[<IPS as BitPack<W>>::T; PY]; PX]>
        + PackedIndexSGet<bool>,
    IS: ImageShape
        + Copy
        + PixelIndexSGet<bool>
        + PixelIndexSGet<u32>
        + PixelIndexSGet<[[<IPS as BitPack<bool>>::T; PY]; PX]>
        + PixelMap<<OPS as BitPack<bool>>::T, bool, PX, PY>
        + PixelMap<bool, bool, PX, PY>
        + PixelMap<u32, bool, PX, PY>
        + PixelZipMap<u32, bool, bool, PX, PY>
        + PixelMap<[[<IPS as BitPack<bool>>::T; PY]; PX], bool, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, bool, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, <OPS as BitPack<bool>>::T, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, [[<IPS as BitPack<bool>>::T; PY]; PX], PX, PY>,
    H: Model<<IS as PixelPack<<OPS as BitPack<bool>>::T>>::I, C, Weight = W> + Copy,
    H: ObjCache<
        <IS as PixelPack<<OPS as BitPack<bool>>::T>>::I,
        C,
        InputIndex = OPS::Index,
        InputValue = <IS as PixelPack<bool>>::I,
    > + Copy,
    Standard: Distribution<<OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T>,
    W: BitScaler + Copy + Eq,
    <IS as PixelPack<bool>>::I: Copy,
    <IS as PixelPack<u32>>::I: std::fmt::Debug,
    <IS as PixelPack<<OPS as BitPack<bool>>::T>>::I: Default,
    <IPS as BitPack<bool>>::T: Copy,
    <IPS as BitPack<W>>::T: Copy,
    <[[IPS; PY]; PX] as Shape>::Index: fmt::Debug,
    <OPS as Shape>::Index: fmt::Debug,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy,
    [[IPS; PY]; PX]: WeightArray<W>
        + BitPack<W, T = [[<IPS as BitPack<W>>::T; PY]; PX]>
        + BitPack<bool, T = [[<IPS as BitPack<bool>>::T; PY]; PX]>
        + PackedIndexSGet<W>,
    <[[IPS; PY]; PX] as BitPack<W>>::T: Copy,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy + fmt::Debug,
    <[[IPS; PY]; PX] as BitPack<bool>>::T: Copy + Default,
    OPS::Index: Copy,
    [[IPS; PY]; PX]: Copy,
    LayerIndex<(OPS::Index, <[[IPS; PY]; PX] as Shape>::Index), H::Index>: Copy,
    <[[IPS; PY]; PX] as Shape>::Index: Copy,
    [[<IPS as BitPack<bool>>::T; PY]; PX]: Default,
    Self::Index: Copy,
{
    type Index = LayerIndex<(OPS::Index, <[[IPS; PY]; PX] as Shape>::Index), H::Index>;
    type Weight = W;
    type IndexIter = LayerIter<[[IPS; PY]; PX], OPS, H::IndexIter>;
    type Output = <IS as PixelPack<<OPS as BitPack<bool>>::T>>::I;
    const N_PARAMS: usize = IPS::N * OPS::N * PY * PX + H::N_PARAMS;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        Conv2D {
            kernel: rng.gen(),
            tail: H::rand(rng),
            image_shape: PhantomData::default(),
        }
    }
    fn apply(
        &self,
        input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I,
    ) -> <IS as PixelPack<<OPS as BitPack<bool>>::T>>::I {
        <IS as Conv<<IPS as BitPack<bool>>::T, <OPS as BitPack<bool>>::T, PX, PY>>::conv(
            input,
            |patch| {
                <OPS as PackedMap<<[[IPS; PY]; PX] as BitPack<W>>::T, bool>>::map(
                    &self.kernel,
                    |weights| <[[IPS; PY]; PX] as WeightArray<W>>::act(weights, &patch),
                )
            },
        )
    }
    fn indices() -> Self::IndexIter {
        LayerIter::new(H::indices())
    }
    fn mutate_in_place(&mut self, index: Self::Index, weight: W) {
        match index {
            LayerIndex::Head((o, i)) => {
                <[[IPS; PY]; PX]>::set_in_place(OPS::index_get_mut(&mut self.kernel, o), i, weight)
            }
            LayerIndex::Tail(i) => self.tail.mutate_in_place(i, weight),
        }
    }
    fn top_act(&self, input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I) -> usize {
        self.tail.top_act(&self.apply(input))
    }
    fn loss(&self, input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I, class: usize) -> u64 {
        self.tail.loss(&self.apply(input), class)
    }
    fn loss_deltas(
        &self,
        input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)> {
        match IMPL {
            0 => {
                let null_loss = self.loss(input, class) as i64;

                <Self::Weight as BitScaler>::states()
                    .iter()
                    .map(|&w| {
                        Self::indices()
                            .map(|i| {
                                (
                                    i,
                                    w,
                                    self.mutate(i, w).loss(input, class) as i64 - null_loss,
                                )
                            })
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .filter(|(_, _, l)| l.abs() as u64 > threshold)
                    .collect()
            }
            1 => {
                let null_acts = self.apply(input);
                // init the base cache
                let cache = self.tail.cache(&null_acts, class);
                // for each output channel
                OPS::indices()
                    .map(|o| {
                        // extract the channel of the null_acts
                        let null_chan_acts =
                            <IS as PixelMap<<OPS as BitPack<bool>>::T, bool, PX, PY>>::map(
                                &null_acts,
                                |pixel| OPS::get(&pixel, o),
                            );
                        // and subtract it from the cache
                        let chan_cache = self.tail.subtract_input(&cache, o, &null_chan_acts);
                        // also extract the output channel of the weights.
                        let weights_channel = OPS::index_get(&self.kernel, o);
                        // For each state of the weights,
                        <[[IPS; PY]; PX]>::indices()
                            .map(|i| {
                                let cur_weight: W = <[[IPS; PY]; PX] as PackedIndexSGet<W>>::get(
                                    weights_channel,
                                    i,
                                );
                                W::states()
                                    .iter()
                                    .filter(|&w| *w != cur_weight)
                                    .filter_map(|&w| {
                                        let new_weights_channel =
                                            <[[IPS; PY]; PX]>::set(*weights_channel, i, w);
                                        let new_acts = <IS as Conv<
                                            <IPS as BitPack<bool>>::T,
                                            bool,
                                            PX,
                                            PY,
                                        >>::conv(
                                            input,
                                            |patch| {
                                                <[[IPS; PY]; PX] as WeightArray<W>>::act(
                                                    &new_weights_channel,
                                                    &patch,
                                                )
                                            },
                                        );
                                        let loss_delta = chan_cache.loss_delta(&new_acts);
                                        if loss_delta.abs() as u64 > threshold {
                                            Some((LayerIndex::Head((o, i)), w, loss_delta))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .chain(
                        self.tail
                            .loss_deltas(&null_acts, threshold, class)
                            .iter()
                            .map(|&(i, w, l)| (LayerIndex::Tail(i), w, l)),
                    )
                    .collect()
            }
            2 => {
                // acts if we do nothing
                let null_acts = self.apply(input);
                // init the base cache
                let cache = self.tail.cache(&null_acts, class);
                // for each output channel
                OPS::indices()
                    .map(|o| {
                        // extract the channel of the null_acts
                        let null_chan_acts =
                            <IS as PixelMap<<OPS as BitPack<bool>>::T, bool, PX, PY>>::map(
                                &null_acts,
                                |pixel| OPS::get(&pixel, o),
                            );
                        // and subtract it from the cache
                        let chan_cache = self.tail.subtract_input(&cache, o, &null_chan_acts);
                        // also extract the output channel of the weights.
                        let weights_channel = OPS::index_get(&self.kernel, o);
                        // For each state of the weights,
                        W::states()
                            .iter()
                            .map(|&w| {
                                // for each output pixel input,
                                let chan_acts =
                                    <IS as Conv<
                                        <IPS as BitPack<bool>>::T,
                                        <[[IPS; PY]; PX] as BitPack<bool>>::T,
                                        PX,
                                        PY,
                                    >>::conv(input, |patch| {
                                        // what would the act be if we set the weight to w?
                                        <[[IPS; PY]; PX] as WeightArray<W>>::acts(
                                            weights_channel,
                                            &patch,
                                            w,
                                        )
                                    });
                                // and for each input channel,
                                <[[IPS; PY]; PX]>::indices()
                                    .filter_map(|i| {
                                        // extract the activation pixels
                                        let mut_weight_acts = <IS as PixelMap<
                                            <[[IPS; PY]; PX] as BitPack<bool>>::T,
                                            bool,
                                            PX,
                                            PY,
                                        >>::map(
                                            &chan_acts,
                                            |pixel| <[[IPS; PY]; PX]>::get(pixel, i),
                                        );
                                        // and compute the loss delta for that output
                                        let loss_delta = chan_cache.loss_delta(&mut_weight_acts);
                                        //let loss_delta = 1_i64;
                                        if loss_delta.abs() as u64 > threshold {
                                            Some((LayerIndex::Head((o, i)), w, loss_delta))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .chain(
                        self.tail
                            .loss_deltas(&null_acts, threshold, class)
                            .iter()
                            .map(|&(i, w, l)| (LayerIndex::Tail(i), w, l)),
                    )
                    .collect()
            }
            3 => {
                let chan_acts = <IS as Conv<
                    <IPS as BitPack<bool>>::T,
                    <[[IPS; PY]; PX] as BitPack<bool>>::T,
                    PX,
                    PY,
                >>::conv(input, |patch| patch);
                let patch_layers = <[[IPS; PY]; PX]>::indices()
                    .map(|i| {
                        <IS as PixelMap<<[[IPS; PY]; PX] as BitPack<bool>>::T, bool, PX, PY>>::map(
                            &chan_acts,
                            |pixel| <[[IPS; PY]; PX]>::get(pixel, i),
                        )
                    })
                    .collect::<Vec<<IS as PixelPack<bool>>::I>>();

                // acts if we do nothing
                let null_acts = self.apply(input);
                // init the base cache
                let cache = self.tail.cache(&null_acts, class);

                let states = W::states();
                // for each output channel
                OPS::indices()
                    .map(|o| {
                        let weights_channel = OPS::index_get(&self.kernel, o);
                        // extract the channel of the null_acts
                        let null_chan_full_sum = <IS as Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>>::conv(input, |patch| <[[IPS; PY]; PX] as BMA<W>>::bma(weights_channel, &patch));
                        let null_chan_acts = <IS as PixelMap<u32, bool, PX, PY>>::map(&null_chan_full_sum, |&sum| sum > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD);
                        // and subtract it from the cache
                        let chan_cache = self.tail.subtract_input(&cache, o, &null_chan_acts);
                        // also extract the output channel of the weights.
                        <[[IPS; PY]; PX]>::indices()
                            .zip(patch_layers.iter())
                            .map(|(p, layer)| {
                                states
                                    .iter()
                                    .filter_map(|&w| {
                                        let cur_weight: W = <[[IPS; PY]; PX] as PackedIndexSGet<W>>::get(weights_channel, p);
                                        if w != cur_weight {
                                            let new_weight_acts: <IS as PixelPack<bool>>::I = <IS as PixelZipMap<u32, bool, bool, PX, PY>>::zip_map(&null_chan_full_sum, layer, |sum, input| {
                                                ((sum + w.bma(input)) - cur_weight.bma(input)) > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD
                                            });
                                            let loss_delta = chan_cache.loss_delta(&new_weight_acts);
                                            Some((LayerIndex::<(OPS::Index, <[[IPS; PY]; PX] as Shape>::Index), H::Index>::Head((o, p)), w, loss_delta))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .filter(|(_, _, l)| l.abs() as u64 > threshold)
                    .chain(self.tail.loss_deltas(&null_acts, threshold, class).iter().map(|&(i, w, l)| (LayerIndex::Tail(i), w, l)))
                    .collect()
            }
            4 => {
                // acts if we do nothing
                let null_acts = self.apply(input);
                // init the base cache
                let cache = self.tail.cache(&null_acts, class);
                // for each output channel
                OPS::indices()
                    .map(|o| {
                        let weights_channel = OPS::index_get(&self.kernel, o);
                        // extract the channel of the null_acts
                        let patches = <IS as Conv<<IPS as BitPack<bool>>::T, [[<IPS as BitPack<bool>>::T; PY]; PX], PX, PY>>::conv(input, |patch| patch);
                        let null_chan_full_sum = <IS as Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>>::conv(input, |patch| <[[IPS; PY]; PX] as BMA<W>>::bma(weights_channel, &patch));
                        let null_chan_acts = <IS as PixelMap<u32, bool, PX, PY>>::map(&null_chan_full_sum, |&sum| sum > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD);
                        // and subtract it from the cache
                        let chan_cache = self.tail.subtract_input(&cache, o, &null_chan_acts);
                        W::states()
                            .iter()
                            .map(|&w| {
                                let acts: Vec<<IS as PixelPack<bool>>::I> = <IS as Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>>::indices().fold(
                                    <[[IPS; PY]; PX]>::indices().map(|_| null_chan_acts).collect::<Vec<<IS as PixelPack<bool>>::I>>(),
                                    |mut acc, spatial_index| {
                                        let sum: u32 = IS::get_pixel(&null_chan_full_sum, spatial_index);
                                        if <[[IPS; PY]; PX] as WeightArray<W>>::mutant_act(sum).is_none() {
                                            let patch = <IS as PixelIndexSGet<[[<IPS as BitPack<bool>>::T; PY]; PX]>>::get_pixel(&patches, spatial_index);
                                            let acts = <[[IPS; PY]; PX] as WeightArray<W>>::acts(weights_channel, &patch, w);

                                            <[[IPS; PY]; PX]>::indices().zip(acc.iter_mut()).for_each(|(i, target)| {
                                                let act: bool = <[[IPS; PY]; PX]>::get(&acts, i);
                                                <IS as PixelIndexSGet<bool>>::set_pixel_in_place(target, spatial_index, act);
                                            });
                                        }
                                        acc
                                    },
                                );
                                <[[IPS; PY]; PX]>::indices()
                                    .zip(acts.iter())
                                    .map(|(i, acts)| {
                                        let loss_delta = chan_cache.loss_delta(acts);
                                        (LayerIndex::<(OPS::Index, <[[IPS; PY]; PX] as Shape>::Index), H::Index>::Head((o, i)), w, loss_delta)
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .filter(|(_, _, l)| l.abs() as u64 > threshold)
                    .chain(self.tail.loss_deltas(&null_acts, threshold, class).iter().map(|&(i, w, l)| (LayerIndex::Tail(i), w, l)))
                    .collect()
            }
            _ => {
                panic!("no conv loss_deltas {}", IMPL)
            }
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct GlobalAvgPool<I, P, H, const PX: usize, const PY: usize> {
    pixel_shape: PhantomData<P>,
    image_shape: PhantomData<I>,
    tail: H,
}

impl<I, P, H, const PX: usize, const PY: usize> GlobalAvgPool<I, P, H, PX, PY>
where
    P: BitPack<bool> + Pack<u32> + IncrementCounters + PackedMap<u32, bool>,
    I: PixelPack<<P as BitPack<bool>>::T>
        + PixelFold<(usize, <P as Pack<u32>>::T), <P as BitPack<bool>>::T, PX, PY>,
    (usize, <P as Pack<u32>>::T): Default,
{
    fn pool(
        input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I,
    ) -> (u32, <P as BitPack<bool>>::T) {
        let (n, counts) = <I as PixelFold<
            (usize, <P as Pack<u32>>::T),
            <P as BitPack<bool>>::T,
            PX,
            PY,
        >>::pixel_fold(
            input,
            <(usize, <P as Pack<u32>>::T)>::default(),
            |acc, pixel| P::counted_increment(pixel, acc),
        );
        let threshold = n as u32 / 2;
        (
            threshold,
            <P as PackedMap<u32, bool>>::map(&counts, |&sum| sum > threshold),
        )
    }
}

impl<I, P, H, const PX: usize, const PY: usize, const C: usize>
    Model<<I as PixelPack<<P as BitPack<bool>>::T>>::I, C> for GlobalAvgPool<I, P, H, PX, PY>
where
    P: Shape
        + Copy
        + BitPack<bool>
        + Pack<u32>
        + IncrementCounters
        + PackedMap<u32, bool>
        + PackedIndexSGet<bool>,
    I: PixelPack<<P as BitPack<bool>>::T>
        + Copy
        + PixelPack<bool>
        + PixelFold<u32, bool, PX, PY>
        + PixelFold<(usize, <P as Pack<u32>>::T), <P as BitPack<bool>>::T, PX, PY>,
    H: Model<<P as BitPack<bool>>::T, C>
        + ObjCache<<P as BitPack<bool>>::T, C, InputIndex = P::Index, InputValue = bool>,
    (usize, <P as Pack<u32>>::T): Default,
{
    type Index = H::Index;
    type Weight = H::Weight;
    type IndexIter = H::IndexIter;
    type Output = <P as BitPack<bool>>::T;
    const N_PARAMS: usize = H::N_PARAMS;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        GlobalAvgPool {
            pixel_shape: PhantomData::default(),
            image_shape: PhantomData::default(),
            tail: H::rand(rng),
        }
    }
    fn apply(
        &self,
        input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I,
    ) -> <P as BitPack<bool>>::T {
        let (_, acts) = Self::pool(input);
        acts
    }
    fn indices() -> H::IndexIter {
        H::indices()
    }
    fn mutate_in_place(&mut self, index: H::Index, weight: H::Weight) {
        self.tail.mutate_in_place(index, weight)
    }
    fn top_act(&self, input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I) -> usize {
        self.tail.top_act(&self.apply(input))
    }
    fn loss(&self, input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I, class: usize) -> u64 {
        self.tail.loss(&self.apply(input), class)
    }
    fn loss_deltas(
        &self,
        input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)> {
        self.tail.loss_deltas(&self.apply(input), threshold, class)
    }
}

pub struct GlobalAvgPoolCache<TailCache, P: Shape, const C: usize>
where
    P: BitPack<bool>,
{
    tail_cache: TailCache,
    threshold: u32,
    acts: <P as BitPack<bool>>::T,
}

pub struct GlobalAvgPoolChanCache<
    W,
    TailChanCache,
    I,
    const PX: usize,
    const PY: usize,
    const C: usize,
> {
    threshold: u32,
    tail_chan_cache: TailChanCache,
    weight_type: PhantomData<W>,
    image_shape: PhantomData<I>,
}

impl<
        W: BitScaler,
        TailChanCache,
        I: ImageShape,
        const PX: usize,
        const PY: usize,
        const C: usize,
    > Cache<<I as PixelPack<bool>>::I> for GlobalAvgPoolChanCache<W, TailChanCache, I, PX, PY, C>
where
    TailChanCache: Cache<bool>,
    I: PixelFold<u32, bool, PX, PY>,
{
    fn loss_delta(&self, input: &<I as PixelPack<bool>>::I) -> i64 {
        let sum = <I as PixelFold<u32, bool, PX, PY>>::pixel_fold(input, 0u32, |sum, &bit| {
            sum + (bit as u32)
        });
        self.tail_chan_cache.loss_delta(&(sum > self.threshold))
    }
}

impl<I, P, H, const PX: usize, const PY: usize, const C: usize>
    ObjCache<<I as PixelPack<<P as BitPack<bool>>::T>>::I, C> for GlobalAvgPool<I, P, H, PX, PY>
where
    P: Shape
        + Copy
        + BitPack<bool>
        + Pack<u32>
        + IncrementCounters
        + PackedMap<u32, bool>
        + PackedIndexSGet<bool>,
    I: PixelPack<<P as BitPack<bool>>::T>
        + Copy
        + PixelPack<bool>
        + PixelFold<u32, bool, PX, PY>
        + PixelFold<(usize, <P as Pack<u32>>::T), <P as BitPack<bool>>::T, PX, PY>,
    H: ObjCache<<P as BitPack<bool>>::T, C, InputIndex = P::Index, InputValue = bool>,
    <H as ObjCache<<P as BitPack<bool>>::T, C>>::ChanCache: Cache<bool>,
    (usize, <P as Pack<u32>>::T): Default,
{
    type Cache = GlobalAvgPoolCache<H::Cache, P, C>;
    type ChanCache = GlobalAvgPoolChanCache<H::Weight, H::ChanCache, I, PX, PY, C>;
    type InputIndex = P::Index;
    type InputValue = <I as PixelPack<bool>>::I;
    fn cache(
        &self,
        input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I,
        class: usize,
    ) -> Self::Cache {
        let (threshold, acts) = Self::pool(input);
        GlobalAvgPoolCache {
            tail_cache: self.tail.cache(&acts, class),
            threshold: threshold,
            acts: acts,
        }
    }
    fn subtract_input(
        &self,
        cache: &Self::Cache,
        chan_index: Self::InputIndex,
        _: &Self::InputValue,
    ) -> Self::ChanCache {
        GlobalAvgPoolChanCache {
            threshold: cache.threshold,
            tail_chan_cache: self.tail.subtract_input(
                &cache.tail_cache,
                chan_index,
                &P::get(&cache.acts, chan_index),
            ),
            weight_type: PhantomData::default(),
            image_shape: PhantomData::default(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SegmentedAvgPool<
    I,
    P,
    H,
    const SX: usize,
    const SY: usize,
    const PX: usize,
    const PY: usize,
> {
    pixel_shape: PhantomData<P>,
    image_shape: PhantomData<I>,
    pub tail: H,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct SegmentedAvgPoolIndex<I, const X: usize, const Y: usize> {
    pub index: I,
}

pub struct SegmentedAvgPoolIndexIterator<I, const X: usize, const Y: usize> {
    iterator: I,
}

impl<I: Iterator, const X: usize, const Y: usize> Iterator
    for SegmentedAvgPoolIndexIterator<I, X, Y>
{
    type Item = SegmentedAvgPoolIndex<I::Item, X, Y>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iterator
            .next()
            .map(|x| SegmentedAvgPoolIndex { index: x })
    }
}

impl<I, P, H, const SX: usize, const SY: usize, const PX: usize, const PY: usize>
    SegmentedAvgPool<I, P, H, SY, SX, PX, PY>
where
    P: BitPack<bool> + Pack<u32> + IncrementCounters + PackedMap<u32, bool>,
    I: PixelPack<<P as BitPack<bool>>::T>
        + SegmentedPixelFold<(usize, <P as Pack<u32>>::T), <P as BitPack<bool>>::T, SX, SY, PX, PY>,
    (usize, <P as Pack<u32>>::T): Default,
    [[(u32, <P as BitPack<bool>>::T); SY]; SX]: Default,
    <P as BitPack<bool>>::T: fmt::Debug,
{
    pub fn seg_pool(
        input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I,
    ) -> [[(u32, <P as BitPack<bool>>::T); SY]; SX] {
        let mut target = <[[(u32, <P as BitPack<bool>>::T); SY]; SX]>::default();
        for sx in 0..SX {
            for sy in 0..SY {
                let (n, counts) = <I as SegmentedPixelFold<
                    (usize, <P as Pack<u32>>::T),
                    <P as BitPack<bool>>::T,
                    SX,
                    SY,
                    PX,
                    PY,
                >>::seg_fold(
                    input,
                    sx,
                    sy,
                    <(usize, <P as Pack<u32>>::T)>::default(),
                    |acc, pixel| P::counted_increment(pixel, acc),
                );
                let threshold = n as u32 / 2;
                target[sx][sy] = (
                    threshold,
                    <P as PackedMap<u32, bool>>::map(&counts, |&sum| sum > threshold),
                );
            }
        }
        target
    }
}

impl<
        I,
        P,
        H,
        const SX: usize,
        const SY: usize,
        const PX: usize,
        const PY: usize,
        const C: usize,
    > Model<<I as PixelPack<<P as BitPack<bool>>::T>>::I, C>
    for SegmentedAvgPool<I, P, H, SY, SX, PX, PY>
where
    [[(u32, <P as BitPack<bool>>::T); SY]; SX]: Default,
    P: Shape
        + Copy
        + BitPack<bool>
        + Pack<u32>
        + IncrementCounters
        + PackedMap<u32, bool>
        + PackedIndexSGet<bool>,
    I: PixelPack<<P as BitPack<bool>>::T>
        + Copy
        + PixelPack<bool>
        + SegmentedPixelFold<(usize, <P as Pack<u32>>::T), <P as BitPack<bool>>::T, SX, SY, PX, PY>,
    <P as BitPack<bool>>::T: fmt::Debug,
    H: Model<[[<P as BitPack<bool>>::T; SY]; SX], C>,
    (usize, <P as Pack<u32>>::T): Default,
    [[(); SY]; SX]: Map<(u32, <P as BitPack<bool>>::T), <P as BitPack<bool>>::T>
        + Pack<<P as BitPack<bool>>::T, T = [[<P as BitPack<bool>>::T; SY]; SX]>
        + Pack<(u32, <P as BitPack<bool>>::T), T = [[(u32, <P as BitPack<bool>>::T); SY]; SX]>,
    <P as BitPack<bool>>::T: Copy,
{
    type Index = SegmentedAvgPoolIndex<H::Index, SX, SY>;
    type Weight = H::Weight;
    type IndexIter = SegmentedAvgPoolIndexIterator<H::IndexIter, SX, SY>;
    type Output = [[<P as BitPack<bool>>::T; SY]; SX];
    const N_PARAMS: usize = H::N_PARAMS;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        SegmentedAvgPool {
            pixel_shape: PhantomData::default(),
            image_shape: PhantomData::default(),
            tail: H::rand(rng),
        }
    }
    fn apply(
        &self,
        input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I,
    ) -> [[<P as BitPack<bool>>::T; SY]; SX] {
        <[[(); SY]; SX] as Map<(u32, <P as BitPack<bool>>::T), <P as BitPack<bool>>::T>>::map(
            &Self::seg_pool(input),
            |&(_, avgs)| avgs,
        )
    }
    fn indices() -> SegmentedAvgPoolIndexIterator<H::IndexIter, SX, SY> {
        SegmentedAvgPoolIndexIterator {
            iterator: H::indices(),
        }
    }
    fn mutate_in_place(
        &mut self,
        index: SegmentedAvgPoolIndex<H::Index, SX, SY>,
        weight: H::Weight,
    ) {
        self.tail.mutate_in_place(index.index, weight)
    }
    fn top_act(&self, input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I) -> usize {
        self.tail.top_act(&self.apply(input))
    }
    fn loss(&self, input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I, class: usize) -> u64 {
        self.tail.loss(&self.apply(input), class)
    }
    fn loss_deltas(
        &self,
        input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)> {
        self.tail
            .loss_deltas(&self.apply(input), threshold, class)
            .iter()
            .map(|&(i, w, l)| (SegmentedAvgPoolIndex { index: i }, w, l))
            .collect()
    }
}

pub struct SegmentedAvgPoolCache<H, P: Shape, const SX: usize, const SY: usize, const C: usize>
where
    P: BitPack<bool>,
{
    tail: H,
    thresholds: [[u32; SY]; SX],
    acts: [[<P as BitPack<bool>>::T; SY]; SX],
    null_loss: i64,
    class: usize,
}

pub struct SegmentedAvgPoolChanCache<
    P,
    H,
    I,
    const SX: usize,
    const SY: usize,
    const PX: usize,
    const PY: usize,
    const C: usize,
> where
    P: BitPack<bool> + Shape,
{
    chan_index: P::Index,
    thresholds: [[u32; SY]; SX],
    tail: H,
    acts: [[<P as BitPack<bool>>::T; SY]; SX],
    null_loss: i64,
    image_shape: PhantomData<I>,
    class: usize,
}

impl<
        P,
        H,
        I: ImageShape,
        const SX: usize,
        const SY: usize,
        const PX: usize,
        const PY: usize,
        const C: usize,
    > Cache<<I as PixelPack<bool>>::I> for SegmentedAvgPoolChanCache<P, H, I, SX, SY, PX, PY, C>
where
    H: Model<[[<P as BitPack<bool>>::T; SY]; SX], C>,
    I: SegmentedPixelFold<u32, bool, SX, SY, PX, PY>,
    P: BitPack<bool> + Shape + PackedIndexSGet<bool>,
    [[<P as BitPack<bool>>::T; SY]; SX]: Copy,
{
    fn loss_delta(&self, input: &<I as PixelPack<bool>>::I) -> i64 {
        let mut target = self.acts;
        for sx in 0..SX {
            for sy in 0..SY {
                let count: u32 = <I as SegmentedPixelFold<u32, bool, SX, SY, PX, PY>>::seg_fold(
                    input,
                    sx,
                    sy,
                    0u32,
                    |sum, &pixel| sum + pixel as u32,
                );
                <P as PackedIndexSGet<bool>>::set_in_place(
                    &mut target[sx][sy],
                    self.chan_index,
                    count > self.thresholds[sx][sy],
                );
            }
        }

        self.tail.loss(&target, self.class) as i64 - self.null_loss
    }
}

impl<
        I,
        P,
        H,
        const SX: usize,
        const SY: usize,
        const PX: usize,
        const PY: usize,
        const C: usize,
    > ObjCache<<I as PixelPack<<P as BitPack<bool>>::T>>::I, C>
    for SegmentedAvgPool<I, P, H, SY, SX, PX, PY>
where
    P: Shape
        + Copy
        + BitPack<bool>
        + Pack<u32>
        + IncrementCounters
        + PackedMap<u32, bool>
        + PackedIndexSGet<bool>,
    I: Copy
        + PixelPack<bool>
        + PixelPack<<P as BitPack<bool>>::T>
        + SegmentedPixelFold<u32, bool, SX, SY, PX, PY>
        + SegmentedPixelFold<(usize, <P as Pack<u32>>::T), <P as BitPack<bool>>::T, SX, SY, PX, PY>,
    H: Model<[[<P as BitPack<bool>>::T; SY]; SX], C>,
    [[(u32, <P as BitPack<bool>>::T); SY]; SX]: Default,
    (usize, <P as Pack<u32>>::T): Default,
    Self: Model<<I as PixelPack<<P as BitPack<bool>>::T>>::I, C>,
    [[(); SY]; SX]: Shape
        + Map<(u32, <P as BitPack<bool>>::T), <P as BitPack<bool>>::T>
        + Map<(u32, <P as BitPack<bool>>::T), u32>
        + Pack<(u32, <P as BitPack<bool>>::T), T = [[(u32, <P as BitPack<bool>>::T); SY]; SX]>
        + Pack<<P as BitPack<bool>>::T, T = [[<P as BitPack<bool>>::T; SY]; SX]>
        + Pack<u32, T = [[u32; SY]; SX]>,
    [[<P as BitPack<bool>>::T; SY]; SX]: Copy,
    <P as BitPack<bool>>::T: fmt::Debug,
    <P as BitPack<bool>>::T: Copy,
{
    type Cache = SegmentedAvgPoolCache<H, P, SX, SY, C>;
    type ChanCache = SegmentedAvgPoolChanCache<P, H, I, SX, SY, PX, PY, C>;
    type InputIndex = P::Index;
    type InputValue = <I as PixelPack<bool>>::I;
    fn cache(
        &self,
        input: &<I as PixelPack<<P as BitPack<bool>>::T>>::I,
        class: usize,
    ) -> Self::Cache {
        let pooled = Self::seg_pool(input);

        let acts = <[[(); SY]; SX] as Map<
            (u32, <P as BitPack<bool>>::T),
            <P as BitPack<bool>>::T,
        >>::map(&pooled, |&(_, avgs)| avgs);
        let thresholds =
            <[[(); SY]; SX] as Map<(u32, <P as BitPack<bool>>::T), u32>>::map(&pooled, |&(t, _)| t);

        let null_loss = self.tail.loss(&acts, class) as i64;

        SegmentedAvgPoolCache {
            tail: self.tail,
            thresholds,
            acts,
            null_loss,
            class,
        }
    }
    fn subtract_input(
        &self,
        cache: &Self::Cache,
        chan_index: Self::InputIndex,
        _: &Self::InputValue,
    ) -> Self::ChanCache {
        SegmentedAvgPoolChanCache {
            chan_index: chan_index,
            thresholds: cache.thresholds,
            tail: self.tail,
            acts: cache.acts,
            null_loss: cache.null_loss,
            class: cache.class,
            image_shape: PhantomData::default(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FusedConvSegmentedAvgPoolFcMSE<
    IS,
    IPS,
    OPS,
    W,
    const IML: usize,
    const SX: usize,
    const SY: usize,
    const PX: usize,
    const PY: usize,
    const C: usize,
> where
    IPS: BitPack<W>,
    OPS: Pack<[[<IPS as BitPack<W>>::T; PY]; PX]> + BitPack<W>,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy + fmt::Debug,
    [[[<OPS as BitPack<W>>::T; SY]; SX]; C]: Copy + fmt::Debug,
{
    image_shape: PhantomData<IS>,
    pub conv: <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T,
    pub fc: [[[<OPS as BitPack<W>>::T; SY]; SX]; C],
}

impl<
        IS,
        IPS,
        OPS,
        W,
        const IMPL: usize,
        const SX: usize,
        const SY: usize,
        const PX: usize,
        const PY: usize,
        const C: usize,
    > FusedConvSegmentedAvgPoolFcMSE<IS, IPS, OPS, W, IMPL, SX, SY, PX, PY, C>
where
    W: BitScaler,
    IPS: BitPack<W> + BitPack<bool>,
    OPS: Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>
        + BitPack<W>
        + Pack<u32>
        + BitPack<bool>
        + ZipMap<u32, [[<IPS as BitPack<W>>::T; PY]; PX], u32>
        + PackedMap<u32, bool>,
    IS: PixelPack<<IPS as BitPack<bool>>::T>
        + SegmentedConvFold<(usize, <OPS as Pack<u32>>::T), <IPS as BitPack<bool>>::T, SX, SY, PX, PY>
        + SegmentedPixelFold<
            (usize, <OPS as Pack<u32>>::T),
            [[<IPS as BitPack<bool>>::T; PY]; PX],
            SX,
            SY,
            PX,
            PY,
        > + Conv<<IPS as BitPack<bool>>::T, [[<IPS as BitPack<bool>>::T; PY]; PX], PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, <OPS as BitPack<bool>>::T, PX, PY>,
    [[IPS; PY]; PX]: WeightArray<W>
        + BitPack<W, T = [[<IPS as BitPack<W>>::T; PY]; PX]>
        + BitPack<bool, T = [[<IPS as BitPack<bool>>::T; PY]; PX]>,
    OPS: WeightArray<W> + PackedMap<<[[IPS; PY]; PX] as BitPack<W>>::T, bool>,
    <OPS as BitPack<W>>::T: Copy,
    <OPS as BitPack<bool>>::T: Copy + fmt::Debug,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy + fmt::Debug,
    [[[<OPS as BitPack<W>>::T; SY]; SX]; C]: Copy + fmt::Debug,
    <OPS as Pack<u32>>::T: fmt::Debug,
    [u32; C]: Default,
    (usize, <OPS as Pack<u32>>::T): Default,
    <[[IPS; PY]; PX] as BitPack<W>>::T: Copy,
    [[<OPS as BitPack<bool>>::T; SY]; SX]: Default,
    [[<OPS as Pack<u32>>::T; SY]; SX]: Default,
    <[[IPS; PY]; PX] as BitPack<bool>>::T: Copy,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy,
    [(); C]: Pack<u32, T = [u32; C]>
        + ZipMap<u32, [[<OPS as BitPack<W>>::T; SY]; SX], u32>
        + Pack<[[<OPS as BitPack<W>>::T; SY]; SX], T = [[[<OPS as BitPack<W>>::T; SY]; SX]; C]>
        + Map<[[<OPS as BitPack<W>>::T; SY]; SX], u32>,
{
    fn fused_conv_pool_acts(
        &self,
        input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I,
    ) -> [u32; C] {
        (0..SX)
            .map(|sx| iter::repeat(sx).zip(0..SY))
            .flatten()
            .fold(<[u32; C]>::default(), |class_acts, (sx, sy)| {
                let (n, counts) = <IS as SegmentedConvFold<
                    (usize, <OPS as Pack<u32>>::T),
                    <IPS as BitPack<bool>>::T,
                    SX,
                    SY,
                    PX,
                    PY,
                >>::seg_conv_fold(
                    input,
                    sx,
                    sy,
                    <(usize, <OPS as Pack<u32>>::T)>::default(),
                    |(n, acc), patch| {
                        (
                            n + 1,
                            <OPS as ZipMap<u32, [[<IPS as BitPack<W>>::T; PY]; PX], u32>>::zip_map(
                                &acc,
                                &self.conv,
                                |sum, weights| {
                                    sum + (<[[IPS; PY]; PX] as WeightArray<W>>::act(
                                        weights, &patch,
                                    )) as u32
                                },
                            ),
                        )
                    },
                );
                let threshold = n as u32 / 2;
                let acts = <OPS as PackedMap<u32, bool>>::map(&counts, |&sum| sum > threshold);
                <[(); C] as ZipMap<u32, [[<OPS as BitPack<W>>::T; SY]; SX], u32>>::zip_map(
                    &class_acts,
                    &self.fc,
                    |sum, weights| sum + OPS::bma(&weights[sx][sy], &acts),
                )
            })
    }
    fn conv_pool_seg_acts(
        &self,
        input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I,
    ) -> [[<OPS as BitPack<bool>>::T; SY]; SX] {
        let patches = <IS as Conv<
            <IPS as BitPack<bool>>::T,
            [[<IPS as BitPack<bool>>::T; PY]; PX],
            PX,
            PY,
        >>::conv(input, |patch| patch);

        let mut seg_acts = <[[<OPS as BitPack<bool>>::T; SY]; SX]>::default();
        for sx in 0..SX {
            for sy in 0..SY {
                let (n, counts) = <IS as SegmentedPixelFold<
                    (usize, <OPS as Pack<u32>>::T),
                    [[<IPS as BitPack<bool>>::T; PY]; PX],
                    SX,
                    SY,
                    PX,
                    PY,
                >>::seg_fold(
                    &patches,
                    sx,
                    sy,
                    <(usize, <OPS as Pack<u32>>::T)>::default(),
                    |(n, acc), patch| {
                        (
                            n + 1,
                            <OPS as ZipMap<u32, [[<IPS as BitPack<W>>::T; PY]; PX], u32>>::zip_map(
                                &acc,
                                &self.conv,
                                |sum, weights| {
                                    sum + (<[[IPS; PY]; PX] as WeightArray<W>>::act(
                                        weights, &patch,
                                    )) as u32
                                },
                            ),
                        )
                    },
                );
                let threshold = n as u32 / 2;
                let acts = <OPS as PackedMap<u32, bool>>::map(&counts, |&sum| sum > threshold);
                seg_acts[sx][sy] = acts;
            }
        }
        seg_acts
    }
}

impl<
        IS,
        IPS,
        OPS,
        W,
        const IMPL: usize,
        const SX: usize,
        const SY: usize,
        const PX: usize,
        const PY: usize,
        const C: usize,
    > Model<<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I, C>
    for FusedConvSegmentedAvgPoolFcMSE<IS, IPS, OPS, W, IMPL, SX, SY, PX, PY, C>
where
    Self: Copy,
    OPS::Index: Send + Sync,
    OPS::IndexIter: Send + Sync,
    IPS::Index: Send,
    W: Send,
    <[[OPS; SY]; SX] as Shape>::Index: Send,
    <OPS as BitPack<W>>::T: Sync,
    IS: Sync,
    <OPS as BitPack<bool>>::T: Sync,
    ((usize, <[[OPS; SY]; SX] as Shape>::Index), W, i64): Send + Sync,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Sync,
    <IS as PixelPack<<IPS as BitPack<bool>>::T>>::I: Sync,
    W: BitScaler + fmt::Debug + std::cmp::PartialEq,
    OPS: Shape
        + WeightArray<W>
        + Pack<u32>
        + Pack<(bool, u32)>
        + Pack<[[IPS; PY]; PX]>
        + Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>
        + BitPack<W>
        + BitPack<bool>
        + IndexGet<u32>
        + IndexGet<[[<IPS as BitPack<W>>::T; PY]; PX]>
        + IndexGet<(bool, u32)>
        + Map<[[<IPS as BitPack<W>>::T; PY]; PX], (bool, u32)>
        + PackedMap<<[[IPS; PY]; PX] as BitPack<W>>::T, bool>
        + ZipMap<u32, [[<IPS as BitPack<W>>::T; PY]; PX], u32>
        + PackedMap<u32, bool>,
    IPS: Shape + BitPack<W> + BitPack<bool> + Pack<[u32; C]> + Pack<u32>,
    IS: PixelPack<<IPS as BitPack<bool>>::T>
        + PixelPack<<OPS as BitPack<bool>>::T>
        + PixelMap<[[<IPS as BitPack<bool>>::T; PY]; PX], bool, PX, PY>
        + PixelZipMap<u32, bool, bool, PX, PY>
        + PixelMap<u32, bool, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, <OPS as BitPack<bool>>::T, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, <OPS as Pack<(bool, u32)>>::T, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, [[<IPS as BitPack<bool>>::T; PY]; PX], PX, PY>
        + Conv<
            <IPS as BitPack<bool>>::T,
            (
                [[<IPS as BitPack<bool>>::T; PY]; PX],
                u32,
                Option<bool>,
                bool,
            ),
            PX,
            PY,
        > + PixelIndexSGet<<OPS as BitPack<bool>>::T>
        + PixelIndexSGet<<OPS as Pack<(bool, u32)>>::T>
        + SegmentedPixelFold<
            (usize, <OPS as Pack<u32>>::T),
            [[<IPS as BitPack<bool>>::T; PY]; PX],
            SX,
            SY,
            PX,
            PY,
        > + SegmentedPixelFold<
            (u32, u32),
            (
                [[<IPS as BitPack<bool>>::T; PY]; PX],
                u32,
                Option<bool>,
                bool,
            ),
            SX,
            SY,
            PX,
            PY,
        > + SegmentedPixelFold<
            (usize, u32),
            (
                [[<IPS as BitPack<bool>>::T; PY]; PX],
                u32,
                Option<bool>,
                bool,
            ),
            SX,
            SY,
            PX,
            PY,
        > + SegmentedPixelFold<
            (usize, u32, u32),
            (
                [[<IPS as BitPack<bool>>::T; PY]; PX],
                u32,
                Option<bool>,
                bool,
            ),
            SX,
            SY,
            PX,
            PY,
        > + SegmentedPixelFold<
            (usize, [[<IPS as Pack<u32>>::T; PY]; PX]),
            (
                [[<IPS as BitPack<bool>>::T; PY]; PX],
                u32,
                Option<bool>,
                bool,
            ),
            SX,
            SY,
            PX,
            PY,
        > + SegmentedPixelFold<
            (usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32),
            (
                [[<IPS as BitPack<bool>>::T; PY]; PX],
                u32,
                Option<bool>,
                bool,
            ),
            SX,
            SY,
            PX,
            PY,
        > + SegmentedConvFold<(usize, <OPS as Pack<u32>>::T), <IPS as BitPack<bool>>::T, SX, SY, PX, PY>,
    OPS::Index: fmt::Debug,
    Box<
        <IS as PixelPack<(
            [[<IPS as BitPack<bool>>::T; PY]; PX],
            u32,
            std::option::Option<bool>,
            bool,
        )>>::I,
    >: Default,
    [(); C]: Pack<u32, T = [u32; C]>
        + Pack<W, T = [W; C]>
        + Map<W, u32>
        + Map<u32, u32>
        + Map<[[<OPS as BitPack<W>>::T; SY]; SX], W>
        + Map<[[<OPS as BitPack<W>>::T; SY]; SX], [[W; SY]; SX]>
        + ZipMap<u32, W, u32>
        + ZipMap<u32, u32, u32>
        + ZipMap<[u32; C], u32, [u32; C]>
        + ZipMap<u32, [[<OPS as BitPack<W>>::T; SY]; SX], u32>
        + Pack<u32, T = [u32; C]>
        + Pack<[[W; SY]; SX], T = [[[W; SY]; SX]; C]>
        + Pack<[[<OPS as BitPack<W>>::T; SY]; SX], T = [[[<OPS as BitPack<W>>::T; SY]; SX]; C]>
        + ZipMap<u32, [[<OPS as BitPack<W>>::T; SY]; SX], u32>
        + Map<[[<OPS as BitPack<W>>::T; SY]; SX], u32>,
    [u32; C]: Default,
    [[Option<bool>; SY]; SX]: Default,
    [[<OPS as BitPack<bool>>::T; SY]; SX]: Default,
    [[IPS; PY]; PX]: Shape
        + IncrementCounters
        + IndexGet<[u32; C]>
        + ZipMap<[u32; C], u32, [u32; C]>
        + Map<[u32; C], [u32; C]>
        + Pack<u32, T = [[<IPS as Pack<u32>>::T; PY]; PX]>
        + Pack<[u32; C], T = [[<IPS as Pack<[u32; C]>>::T; PY]; PX]>
        + BitPack<W, T = [[<IPS as BitPack<W>>::T; PY]; PX]>
        + BitPack<bool, T = [[<IPS as BitPack<bool>>::T; PY]; PX]>
        + Shape<Index = (u8, (u8, <IPS as Shape>::Index))>
        + WeightArray<W>,
    [[OPS; PY]; PX]: WeightArray<W>,
    [[OPS; SY]; SX]: WeightArray<W>
        + BitPack<bool, T = [[<OPS as BitPack<bool>>::T; SY]; SX]>
        + BitPack<W, T = [[<OPS as BitPack<W>>::T; SY]; SX]>,
    [[<IPS as Pack<u32>>::T; PY]; PX]: Default,
    [[[<OPS as BitPack<W>>::T; SY]; SX]; C]: fmt::Debug + Copy,
    [[(); SY]; SX]: Map<<OPS as BitPack<W>>::T, W>
        + Pack<<OPS as BitPack<W>>::T, T = [[<OPS as BitPack<W>>::T; SY]; SX]>
        + Pack<W, T = [[W; SY]; SX]>,
    [[<IPS as Pack<[u32; C]>>::T; PY]; PX]: Default + fmt::Debug,
    [[<OPS as Pack<u32>>::T; SY]; SX]: Default,
    [[IPS; PY]; PX]: PackedMap<u32, bool>,
    <IPS as Pack<[u32; C]>>::T: Default,
    <IPS as Pack<u32>>::T: Default + fmt::Debug,
    <OPS as Pack<u32>>::T: Default + fmt::Debug,
    <IPS as BitPack<bool>>::T: fmt::Debug,
    <OPS as Pack<(bool, u32)>>::T: Copy,
    <OPS as BitPack<W>>::T: Copy,
    <OPS as BitPack<bool>>::T: Copy + fmt::Debug,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy + fmt::Debug,
    <OPS as Pack<[[IPS; PY]; PX]>>::T: Shape,
    <[[IPS; PY]; PX] as BitPack<W>>::T: Copy,
    <[[OPS; PY]; PX] as BitPack<bool>>::T: Copy,
    <[[IPS; PY]; PX] as Shape>::Index: fmt::Debug,
    <[[OPS; SY]; SX] as BitPack<bool>>::T: Copy,
    <[[OPS; PY]; PX] as BitPack<W>>::T: Copy,
    <OPS as BitPack<W>>::T: fmt::Debug,
    <IS as PixelPack<(
        [[<IPS as BitPack<bool>>::T; PY]; PX],
        u32,
        std::option::Option<bool>,
        bool,
    )>>::I: fmt::Debug,
    <[[OPS; SY]; SX] as Shape>::Index: fmt::Debug,
    <[[OPS; SY]; SX] as BitPack<W>>::T: Copy,
    <[[IPS; PY]; PX] as BitPack<bool>>::T: Copy,
    Standard: Distribution<<OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T>,
    Standard: Distribution<[[[<OPS as BitPack<W>>::T; SY]; SX]; C]>,
    SegmentedAvgPool<IS, OPS, FcMSE<[[OPS; SY]; SX], W, C>, SX, SY, PX, PY>: ObjCache<
        <IS as PixelPack<<OPS as BitPack<bool>>::T>>::I,
        C,
        InputValue = <IS as PixelPack<bool>>::I,
        InputIndex = OPS::Index,
    >,
    FcMSE<[[OPS; SY]; SX], W, C>: Model<
        [[<OPS as BitPack<bool>>::T; SY]; SX],
        C,
        Weight = W,
        Index = (usize, <[[OPS; SY]; SX] as Shape>::Index),
    >,
{
    type Index = LayerIndex<
        (<OPS as Shape>::Index, <[[IPS; PY]; PX] as Shape>::Index),
        SegmentedAvgPoolIndex<(usize, <[[OPS; SY]; SX] as Shape>::Index), SX, SY>,
    >;
    type Weight = W;
    type IndexIter = LayerIter<
        [[IPS; PY]; PX],
        OPS,
        SegmentedAvgPoolIndexIterator<FcMSEindexIter<[[OPS; SY]; SX], C>, SX, SY>,
    >;
    type Output = <IS as PixelPack<<OPS as BitPack<bool>>::T>>::I;
    const N_PARAMS: usize = <<OPS as Pack<[[IPS; PY]; PX]>>::T as Shape>::N;
    //const N_PARAMS: usize = <<OPS as Pack<[[IPS; PY]; PX]>>::T as Shape>::N + <[[[OPS; SY]; SX]; C] as Shape>::N;
    /// Rand init
    fn rand<R: Rng>(rng: &mut R) -> Self {
        FusedConvSegmentedAvgPoolFcMSE {
            image_shape: PhantomData::default(),
            conv: rng.gen(),
            fc: rng.gen(),
        }
    }
    fn apply(&self, input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I) -> Self::Output {
        <IS as Conv<<IPS as BitPack<bool>>::T, <OPS as BitPack<bool>>::T, PX, PY>>::conv(
            input,
            |patch| {
                <OPS as PackedMap<<[[IPS; PY]; PX] as BitPack<W>>::T, bool>>::map(
                    &self.conv,
                    |weights| <[[IPS; PY]; PX] as WeightArray<W>>::act(weights, &patch),
                )
            },
        )
    }
    /// iter of all weight indices
    fn indices() -> Self::IndexIter {
        LayerIter::new(SegmentedAvgPoolIndexIterator {
            iterator: FcMSEindexIter {
                class_index: 0,
                input_iter: <[[OPS; SY]; SX]>::indices(),
            },
        })
    }
    fn mutate_in_place(&mut self, index: Self::Index, weight: Self::Weight) {
        match index {
            LayerIndex::Head((o, i)) => {
                <[[IPS; PY]; PX]>::set_in_place(OPS::index_get_mut(&mut self.conv, o), i, weight)
            }
            LayerIndex::Tail(SegmentedAvgPoolIndex { index: (c, i) }) => {
                <[[OPS; SY]; SX]>::set_in_place(&mut self.fc[c], i, weight)
            }
        }
    }
    fn top_act(&self, input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I) -> usize {
        self.fused_conv_pool_acts(input)
            .iter()
            .enumerate()
            .max_by_key(|(_, &act)| act)
            .unwrap()
            .0
    }
    /// loss of the model
    fn loss(&self, input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I, class: usize) -> u64 {
        let class_acts = self.fused_conv_pool_acts(input);
        class_acts
            .iter()
            .enumerate()
            .map(|(c, &act)| {
                let target_act = (c == class) as u32 * <[[OPS; SY]; SX] as WeightArray<W>>::MAX;
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                (dist as u64).pow(2)
            })
            .sum()
    }
    /// loss deltas for all mutations of all weights.
    fn loss_deltas(
        &self,
        input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)> {
        match IMPL {
            5 => {
                let chan_acts = <IS as Conv<
                    <IPS as BitPack<bool>>::T,
                    <[[IPS; PY]; PX] as BitPack<bool>>::T,
                    PX,
                    PY,
                >>::conv(input, |patch| patch);
                let patch_layers = <[[IPS; PY]; PX]>::indices()
                    .map(|i| {
                        <IS as PixelMap<<[[IPS; PY]; PX] as BitPack<bool>>::T, bool, PX, PY>>::map(
                            &chan_acts,
                            |pixel| <[[IPS; PY]; PX]>::get(pixel, i),
                        )
                    })
                    .collect::<Vec<<IS as PixelPack<bool>>::I>>();

                let null_seg_acts = self.conv_pool_seg_acts(input);

                // acts if we do nothing
                let null_acts = self.apply(input);
                // init the base cache
                let fc_layer: FcMSE<[[OPS; SY]; SX], W, C> = FcMSE { fc: self.fc };
                let seg_pool_fc_layer: SegmentedAvgPool<
                    IS,
                    OPS,
                    FcMSE<[[OPS; SY]; SX], W, C>,
                    SX,
                    SY,
                    PX,
                    PY,
                > = SegmentedAvgPool {
                    pixel_shape: PhantomData::default(),
                    image_shape: PhantomData::default(),
                    tail: fc_layer,
                };

                let cache = seg_pool_fc_layer.cache(&null_acts, class);
                let states = W::states();
                // for each output channel
                OPS::indices()
                    .map(|o| {
                        let weights_channel = OPS::index_get(&self.conv, o);
                        // extract the channel of the null_acts
                        let null_chan_full_sum = <IS as Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>>::conv(input, |patch| <[[IPS; PY]; PX] as BMA<W>>::bma(weights_channel, &patch));
                        let null_chan_acts = <IS as PixelMap<u32, bool, PX, PY>>::map(&null_chan_full_sum, |&sum| sum > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD);
                        // and subtract it from the cache
                        let chan_cache = seg_pool_fc_layer.subtract_input(&cache, o, &null_chan_acts);
                        // also extract the output channel of the weights.
                        <[[IPS; PY]; PX]>::indices()
                            .zip(patch_layers.iter())
                            .map(|(p, layer)| {
                                states
                                    .iter()
                                    .filter_map(|&w| {
                                        let cur_weight: W = <[[IPS; PY]; PX] as PackedIndexSGet<W>>::get(weights_channel, p);
                                        if w != cur_weight {
                                            let new_weight_acts: <IS as PixelPack<bool>>::I = <IS as PixelZipMap<u32, bool, bool, PX, PY>>::zip_map(&null_chan_full_sum, layer, |sum, input| {
                                                ((sum + w.bma(input)) - cur_weight.bma(input)) > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD
                                            });
                                            let loss_delta = chan_cache.loss_delta(&new_weight_acts);
                                            Some((LayerIndex::Head((o, p)), w, loss_delta))
                                        } else {
                                            None
                                        }
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .chain(
                        fc_layer
                            .loss_deltas(&null_seg_acts, threshold, class)
                            .iter()
                            .map(|&(i, w, l)| (LayerIndex::Tail(SegmentedAvgPoolIndex { index: i }), w, l)),
                    )
                    .filter(|(_, _, l)| l.abs() as u64 > threshold)
                    .collect()
            }
            6 => {
                let null_class_acts = self.fused_conv_pool_acts(input);
                let null_seg_acts = self.conv_pool_seg_acts(input);
                let fc_layer: FcMSE<[[OPS; SY]; SX], W, C> = FcMSE { fc: self.fc };
                let target_acts: Vec<u32> = (0..C)
                    .map(|c| (c == class) as u32 * <[[OPS; SY]; SX] as WeightArray<W>>::MAX)
                    .collect();
                let null_loss: u64 = null_class_acts
                    .iter()
                    .zip(target_acts.iter())
                    .map(|(&act, &target_act)| {
                        let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                        (dist as u64).pow(2)
                    })
                    .sum();

                // for each output channel
                OPS::indices()
                    .map(|o| {
                        // extract the channel of the weights.
                        let weights_channel = OPS::index_get(&self.conv, o);
                        // extract the patches and bma of the input
                        let patches = <IS as Conv<<IPS as BitPack<bool>>::T, ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool), PX, PY>>::conv(input, |patch| {
                            let sum = <[[IPS; PY]; PX] as BMA<W>>::bma(weights_channel, &patch);
                            let sign = sum > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD;
                            let state = <[[IPS; PY]; PX] as WeightArray<W>>::mutant_act(sum);
                            (patch, sum, state, sign)
                        });
                        let else_class_acts: [u32; C] = (0..SX).map(|sx| iter::repeat(sx).zip(0..SY)).flatten().fold(null_class_acts, |class_acts, (sx, sy)| {
                            let (n, count) = <IS as SegmentedPixelFold<(usize, u32), ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool), SX, SY, PX, PY>>::seg_fold(
                                &patches,
                                sx,
                                sy,
                                <(usize, u32)>::default(),
                                |(n, c), (_, _, _, sign)| (n + 1, c + *sign as u32),
                            );

                            let act = count > (n as u32 / 2);
                            <[(); C] as ZipMap<u32, [[<OPS as BitPack<W>>::T; SY]; SX], u32>>::zip_map(&class_acts, &self.fc, |sum, weights| {
                                let w: W = OPS::get(&weights[sx][sy], o);
                                sum - w.bma(act)
                            })
                        });

                        let seg_states: [[Option<bool>; SY]; SX] = {
                            let mut ranges = <[[Option<bool>; SY]; SX]>::default();
                            for sx in 0..SX {
                                for sy in 0..SY {
                                    let (n, min, max) = <IS as SegmentedPixelFold<(usize, u32, u32), ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool), SX, SY, PX, PY>>::seg_fold(
                                        &patches,
                                        sx,
                                        sy,
                                        <(usize, u32, u32)>::default(),
                                        |(n, min, max), (_, _, state, _)| {
                                            if let Some(sign) = state {
                                                if *sign {
                                                    (n + 1, min + 1, max + 1)
                                                } else {
                                                    (n + 1, min, max)
                                                }
                                            } else {
                                                (n + 1, min, max + 1)
                                            }
                                        },
                                    );
                                    let threshold = n as u32 / 2;
                                    ranges[sx][sy] = Some(max > threshold).filter(|_| (min > threshold) | (max <= threshold));
                                }
                            }
                            ranges
                        };
                        let all_segs_dead = seg_states.iter().flatten().find(|x| x.is_none()).is_none();

                        if all_segs_dead {
                            vec![]
                        } else {
                            // for each weight
                            W::states()
                                .iter()
                                .map(|&w| {
                                    let patch_class_acts: [[<IPS as Pack<[u32; C]>>::T; PY]; PX] =
                                        (0..SX)
                                            .map(|sx| iter::repeat(sx).zip(0..SY))
                                            .flatten()
                                            .fold(<[[<IPS as Pack<[u32; C]>>::T; PY]; PX]>::default(), |class_acts, (sx, sy)| {
                                                let fc_weights: [W; C] =
                                                    <[(); C] as Map<[[<OPS as BitPack<W>>::T; SY]; SX], W>>::map(&self.fc, |class_weights| <OPS as PackedIndexSGet<W>>::get(&class_weights[sx][sy], o));

                                                if let Some(act) = seg_states[sx][sy] {
                                                    <[[IPS; PY]; PX] as Map<[u32; C], [u32; C]>>::map(&class_acts, |class_acts| {
                                                        <[(); C] as ZipMap<u32, W, u32>>::zip_map(&class_acts, &fc_weights, |sum, class_weight| sum + class_weight.bma(act))
                                                    })
                                                } else {
                                                    let counts = <IS as SegmentedPixelFold<
                                                        (usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32),
                                                        ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool),
                                                        SX,
                                                        SY,
                                                        PX,
                                                        PY,
                                                    >>::seg_fold(
                                                        &patches,
                                                        sx,
                                                        sy,
                                                        <(usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32)>::default(),
                                                        |mut acc, (patch, cur_sum, state, _)| {
                                                            if let Some(act) = state {
                                                                <[[IPS; PY]; PX]>::none_option_counted_increment_in_place(*act, &mut acc);
                                                            } else {
                                                                let acts = <[[IPS; PY]; PX]>::acts_simple(weights_channel, patch, *cur_sum, w);
                                                                <[[IPS; PY]; PX]>::some_option_counted_increment_in_place(&acts, &mut acc);
                                                            }
                                                            acc
                                                        },
                                                    );
                                                    let (n, seg_counts) = <[[IPS; PY]; PX]>::finalize_option_counted_increment(counts);
                                                    let threshold = n as u32 / 2;

                                                    <[[IPS; PY]; PX] as ZipMap<[u32; C], u32, [u32; C]>>::zip_map(&class_acts, &seg_counts, |class_acts, &count| {
                                                        let act = count > threshold;
                                                        <[(); C] as ZipMap<u32, W, u32>>::zip_map(&class_acts, &fc_weights, |sum, class_weight| sum + class_weight.bma(act))
                                                    })
                                                }
                                            });
                                    <[[IPS; PY]; PX] as Shape>::indices()
                                        .map(|i| {
                                            let mut_loss = <[[IPS; PY]; PX] as IndexGet<[u32; C]>>::index_get(&patch_class_acts, i)
                                                .iter()
                                                .zip(target_acts.iter())
                                                .zip(else_class_acts.iter())
                                                .map(|((&act, &target_act), else_class_act)| {
                                                    let act = act + else_class_act;
                                                    let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                                                    (dist as u64).pow(2)
                                                })
                                                .sum::<u64>();
                                            let index = LayerIndex::Head((o, i));
                                            let loss_delta = mut_loss as i64 - null_loss as i64;
                                            (index, w, loss_delta)
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .flatten()
                                .collect::<Vec<_>>()
                        }
                    })
                    .flatten()
                    .chain(
                        fc_layer
                            .loss_deltas(&null_seg_acts, threshold, class)
                            .iter()
                            .map(|&(i, w, l)| (LayerIndex::Tail(SegmentedAvgPoolIndex { index: i }), w, l)),
                    )
                    .filter(|(_, _, l)| l.abs() as u64 > threshold)
                    .collect()
            }
            7 => {
                let null_class_acts = self.fused_conv_pool_acts(input);
                let null_seg_acts = self.conv_pool_seg_acts(input);
                let fc_layer: FcMSE<[[OPS; SY]; SX], W, C> = FcMSE { fc: self.fc };
                let target_acts: Vec<u32> = (0..C)
                    .map(|c| (c == class) as u32 * <[[OPS; SY]; SX] as WeightArray<W>>::MAX)
                    .collect();
                let null_loss: u64 = null_class_acts
                    .iter()
                    .zip(target_acts.iter())
                    .map(|(&act, &target_act)| {
                        let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                        (dist as u64).pow(2)
                    })
                    .sum();

                // for each output channel
                OPS::indices()
                    .map(|o| {
                        // extract the channel of the weights.
                        let weights_channel = OPS::index_get(&self.conv, o);
                        // extract the patches and bma of the input
                        let patches = <IS as Conv<<IPS as BitPack<bool>>::T, ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool), PX, PY>>::conv(input, |patch| {
                            let sum = <[[IPS; PY]; PX] as BMA<W>>::bma(weights_channel, &patch);
                            let sign = sum > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD;
                            let state = <[[IPS; PY]; PX] as WeightArray<W>>::mutant_act(sum);
                            (patch, sum, state, sign)
                        });
                        let seg_states: Vec<((usize, usize), [W; C], Option<bool>)> = (0..SX)
                            .map(|sx| iter::repeat(sx).zip(0..SY))
                            .flatten()
                            .map(|(sx, sy)| {
                                let (n, min, max) = <IS as SegmentedPixelFold<(usize, u32, u32), ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool), SX, SY, PX, PY>>::seg_fold(
                                    &patches,
                                    sx,
                                    sy,
                                    <(usize, u32, u32)>::default(),
                                    |(n, min, max), (_, _, state, _)| {
                                        if let Some(sign) = state {
                                            if *sign {
                                                (n + 1, min + 1, max + 1)
                                            } else {
                                                (n + 1, min, max)
                                            }
                                        } else {
                                            (n + 1, min, max + 1)
                                        }
                                    },
                                );
                                let threshold = n as u32 / 2;
                                let state = Some(max > threshold).filter(|_| (min > threshold) | (max <= threshold));
                                let fc_class_weights: [W; C] =
                                    <[(); C] as Map<[[<OPS as BitPack<W>>::T; SY]; SX], W>>::map(&self.fc, |class_weights| <OPS as PackedIndexSGet<W>>::get(&class_weights[sx][sy], o));
                                ((sx, sy), fc_class_weights, state)
                            })
                            .collect();

                        let n_live = seg_states.iter().filter(|(_, _, state)| state.is_none()).count();
                        if n_live == 0 {
                            vec![]
                        } else {
                            let dead_class_acts: [u32; C] = seg_states.iter().fold(null_class_acts, |class_acts, &((sx, sy), _, state)| {
                                let act = <OPS as PackedIndexSGet<bool>>::get(&null_seg_acts[sx][sy], o);
                                if let Some(act) = state {
                                    class_acts
                                } else {
                                    <[(); C] as ZipMap<u32, [[<OPS as BitPack<W>>::T; SY]; SX], u32>>::zip_map(&class_acts, &self.fc, |sum, weights| {
                                        sum - <OPS as PackedIndexSGet<W>>::get(&weights[sx][sy], o).bma(act)
                                    })
                                }
                            });

                            if n_live == 1 {
                                let ((sx, sy), class_weights, _) = seg_states.iter().find(|(_, _, state)| state.is_none()).unwrap();
                                let null_seg_act = <OPS as PackedIndexSGet<bool>>::get(&null_seg_acts[*sx][*sy], o);
                                let loss_delta: i64 = <[(); C] as ZipMap<u32, W, u32>>::zip_map(&dead_class_acts, &class_weights, |sum, w| sum + w.bma(!null_seg_act))
                                    .iter()
                                    .zip(target_acts.iter())
                                    .map(|(&act, &target_act)| {
                                        let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                                        (dist as u64).pow(2)
                                    })
                                    .sum::<u64>() as i64
                                    - null_loss as i64;

                                // for each weight
                                W::states()
                                    .iter()
                                    .map(|&w: &W| {
                                        let counts = <IS as SegmentedPixelFold<
                                            (usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32),
                                            ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool),
                                            SX,
                                            SY,
                                            PX,
                                            PY,
                                        >>::seg_fold(
                                            &patches,
                                            *sx,
                                            *sy,
                                            <(usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32)>::default(),
                                            |mut acc, (patch, cur_sum, state, _)| {
                                                if let Some(act) = state {
                                                    <[[IPS; PY]; PX]>::none_option_counted_increment_in_place(*act, &mut acc);
                                                } else {
                                                    let acts = <[[IPS; PY]; PX]>::acts_simple(weights_channel, patch, *cur_sum, w);
                                                    <[[IPS; PY]; PX]>::some_option_counted_increment_in_place(&acts, &mut acc);
                                                }
                                                acc
                                            },
                                        );
                                        let (n, seg_counts) = <[[IPS; PY]; PX]>::finalize_option_counted_increment(counts);
                                        let threshold = n as u32 / 2;
                                        let seg_acts = <[[IPS; PY]; PX] as PackedMap<u32, bool>>::map(&seg_counts, |&count| count > threshold);

                                        <[[IPS; PY]; PX] as Shape>::indices()
                                            .filter_map(|i| {
                                                if <[[IPS; PY]; PX] as PackedIndexSGet<W>>::get(&weights_channel, i) == w {
                                                    None
                                                } else {
                                                    let act = <[[IPS; PY]; PX] as PackedIndexSGet<bool>>::get(&seg_acts, i);
                                                    if act == null_seg_act {
                                                        None
                                                    } else {
                                                        Some((LayerIndex::Head((o, i)), w, loss_delta))
                                                    }
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            // if we have > 1 live segment
                            } else {
                                let live_seg_class_acts: Vec<[[u32; C]; 2]> = seg_states
                                    .iter()
                                    .filter(|(_, _, state)| state.is_none())
                                    .map(|(_, class_weights, _)| {
                                        [
                                            <[(); C] as Map<W, u32>>::map(&class_weights, |w| w.bma(false)),
                                            <[(); C] as Map<W, u32>>::map(&class_weights, |w| w.bma(true)),
                                        ]
                                    })
                                    .collect();
                                let loss_deltas: Vec<i64> = (0..2usize.pow(live_seg_class_acts.len() as u32))
                                    .map(|i| {
                                        live_seg_class_acts
                                            .iter()
                                            .enumerate()
                                            .fold(dead_class_acts, |class_acts, (seg_index, live_class_acts)| {
                                                <[(); C] as ZipMap<u32, u32, u32>>::zip_map(&class_acts, &live_class_acts[i.bit(seg_index) as usize], |sum, live_act| sum + live_act)
                                            })
                                            .iter()
                                            .zip(target_acts.iter())
                                            .map(|(&act, &target_act)| {
                                                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                                                (dist as u64).pow(2)
                                            })
                                            .sum::<u64>() as i64
                                            - null_loss as i64
                                    })
                                    .collect();
                                // for each weight
                                W::states()
                                    .iter()
                                    .map(|&w: &W| {
                                        let seg_acts: Vec<<[[IPS; PY]; PX] as BitPack<bool>>::T> = seg_states
                                            .iter()
                                            .filter_map(|&((sx, sy), _, state)| {
                                                if let Some(_) = state {
                                                    None
                                                } else {
                                                    let counts = <IS as SegmentedPixelFold<
                                                        (usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32),
                                                        ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool),
                                                        SX,
                                                        SY,
                                                        PX,
                                                        PY,
                                                    >>::seg_fold(
                                                        &patches,
                                                        sx,
                                                        sy,
                                                        <(usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32)>::default(),
                                                        |mut acc, (patch, cur_sum, state, _)| {
                                                            if let Some(act) = state {
                                                                <[[IPS; PY]; PX]>::none_option_counted_increment_in_place(*act, &mut acc);
                                                            } else {
                                                                let acts = <[[IPS; PY]; PX]>::acts_simple(weights_channel, patch, *cur_sum, w);
                                                                <[[IPS; PY]; PX]>::some_option_counted_increment_in_place(&acts, &mut acc);
                                                            }
                                                            acc
                                                        },
                                                    );
                                                    let (n, seg_counts) = <[[IPS; PY]; PX]>::finalize_option_counted_increment(counts);
                                                    let threshold = n as u32 / 2;
                                                    let acts = <[[IPS; PY]; PX] as PackedMap<u32, bool>>::map(&seg_counts, |&count| count > threshold);
                                                    Some(acts)
                                                }
                                            })
                                            .collect();

                                        <[[IPS; PY]; PX] as Shape>::indices()
                                            .filter_map(|i| {
                                                if <[[IPS; PY]; PX] as PackedIndexSGet<W>>::get(&weights_channel, i) == w {
                                                    None
                                                } else {
                                                    let loss_index: usize = seg_acts.iter().enumerate().fold(0usize, |loss_index, (bit_index, seg)| {
                                                        loss_index | ((<[[IPS; PY]; PX] as PackedIndexSGet<bool>>::get(&seg, i) as usize) << bit_index)
                                                    });
                                                    Some((LayerIndex::Head((o, i)), w, loss_deltas[loss_index]))
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            }
                        }
                    })
                    .flatten()
                    .chain(
                        fc_layer
                            .loss_deltas(&null_seg_acts, threshold, class)
                            .iter()
                            .map(|&(i, w, l)| (LayerIndex::Tail(SegmentedAvgPoolIndex { index: i }), w, l)),
                    )
                    .filter(|(_, _, l): &(_, _, i64)| (l.abs() as u64) > threshold)
                    .collect()
            }
            8 => {
                let null_class_acts = self.fused_conv_pool_acts(input);
                let null_seg_acts = self.conv_pool_seg_acts(input);
                let fc_layer: FcMSE<[[OPS; SY]; SX], W, C> = FcMSE { fc: self.fc };
                let target_acts: Vec<u32> = (0..C)
                    .map(|c| (c == class) as u32 * <[[OPS; SY]; SX] as WeightArray<W>>::MAX)
                    .collect();
                let null_loss: u64 = null_class_acts
                    .iter()
                    .zip(target_acts.iter())
                    .map(|(&act, &target_act)| {
                        let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                        (dist as u64).pow(2)
                    })
                    .sum();

                // for each output channel
                OPS::indices()
                    .par_bridge()
                    .map(|o| {
                        // extract the channel of the weights.
                        let weights_channel = OPS::index_get(&self.conv, o);

                        // force alocate on heap to avoid stack overflow
                        let patches = {
                            let mut target = Box::<<IS as PixelPack<([[<IPS as BitPack<bool>>::T; PY]; PX], u32, std::option::Option<bool>, bool)>>::I>::default();
                            <IS as Conv<<IPS as BitPack<bool>>::T, ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool), PX, PY>>::mut_conv(input, &mut target, |patch| {
                                let sum = <[[IPS; PY]; PX] as BMA<W>>::bma(weights_channel, &patch);
                                let sign = sum > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD;
                                let state = <[[IPS; PY]; PX] as WeightArray<W>>::mutant_act(sum);
                                (patch, sum, state, sign)
                            });
                            target
                        };
                        let seg_states: Vec<((usize, usize), [W; C], Option<bool>)> = (0..SX)
                            .map(|sx| iter::repeat(sx).zip(0..SY))
                            .flatten()
                            .map(|(sx, sy)| {
                                let (n, min, max) = <IS as SegmentedPixelFold<(usize, u32, u32), ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool), SX, SY, PX, PY>>::seg_fold(
                                    &patches,
                                    sx,
                                    sy,
                                    <(usize, u32, u32)>::default(),
                                    |(n, min, max), (_, _, state, _)| {
                                        if let Some(sign) = state {
                                            if *sign {
                                                (n + 1, min + 1, max + 1)
                                            } else {
                                                (n + 1, min, max)
                                            }
                                        } else {
                                            (n + 1, min, max + 1)
                                        }
                                    },
                                );
                                let threshold = n as u32 / 2;
                                let state = Some(max > threshold).filter(|_| (min > threshold) | (max <= threshold));
                                let fc_class_weights: [W; C] =
                                    <[(); C] as Map<[[<OPS as BitPack<W>>::T; SY]; SX], W>>::map(&self.fc, |class_weights| <OPS as PackedIndexSGet<W>>::get(&class_weights[sx][sy], o));
                                ((sx, sy), fc_class_weights, state)
                            })
                            .collect();

                        let n_live = seg_states.iter().filter(|(_, _, state)| state.is_none()).count();
                        if n_live > 0 {
                            let dead_class_acts: [u32; C] = seg_states.iter().fold(null_class_acts, |class_acts, &((sx, sy), _, state)| {
                                let act = <OPS as PackedIndexSGet<bool>>::get(&null_seg_acts[sx][sy], o);
                                if let Some(_) = state {
                                    class_acts
                                } else {
                                    <[(); C] as ZipMap<u32, [[<OPS as BitPack<W>>::T; SY]; SX], u32>>::zip_map(&class_acts, &self.fc, |sum, weights| {
                                        let w: W = OPS::get(&weights[sx][sy], o);
                                        sum - w.bma(act)
                                    })
                                }
                            });

                            if n_live == 1 {
                                let ((sx, sy), class_weights, _) = seg_states.iter().find(|(_, _, state)| state.is_none()).unwrap();
                                let null_seg_act = <OPS as PackedIndexSGet<bool>>::get(&null_seg_acts[*sx][*sy], o);
                                let mut_loss = <[(); C] as ZipMap<u32, W, u32>>::zip_map(&dead_class_acts, &class_weights, |sum, w| sum + w.bma(!null_seg_act))
                                    .iter()
                                    .zip(target_acts.iter())
                                    .map(|(&act, &target_act)| {
                                        let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                                        (dist as u64).pow(2)
                                    })
                                    .sum::<u64>();

                                let loss_delta: i64 = mut_loss as i64 - null_loss as i64;

                                // for each weight
                                W::states()
                                    .iter()
                                    .map(|&w: &W| {
                                        let counts = <IS as SegmentedPixelFold<
                                            (usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32),
                                            ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool),
                                            SX,
                                            SY,
                                            PX,
                                            PY,
                                        >>::seg_fold(
                                            &patches,
                                            *sx,
                                            *sy,
                                            <(usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32)>::default(),
                                            |mut acc, (patch, cur_sum, state, _)| {
                                                if let Some(act) = state {
                                                    <[[IPS; PY]; PX]>::none_option_counted_increment_in_place(*act, &mut acc);
                                                } else {
                                                    let acts = <[[IPS; PY]; PX]>::acts_simple(weights_channel, patch, *cur_sum, w);
                                                    <[[IPS; PY]; PX]>::some_option_counted_increment_in_place(&acts, &mut acc);
                                                }
                                                acc
                                            },
                                        );
                                        let (n, seg_counts) = <[[IPS; PY]; PX]>::finalize_option_counted_increment(counts);
                                        let threshold = n as u32 / 2;
                                        let seg_acts = <[[IPS; PY]; PX] as PackedMap<u32, bool>>::map(&seg_counts, |&count| count > threshold);

                                        <[[IPS; PY]; PX] as Shape>::indices()
                                            .filter_map(|i| {
                                                if <[[IPS; PY]; PX] as PackedIndexSGet<W>>::get(&weights_channel, i) == w {
                                                    None
                                                } else {
                                                    let act = <[[IPS; PY]; PX] as PackedIndexSGet<bool>>::get(&seg_acts, i);
                                                    if act == null_seg_act {
                                                        None
                                                    } else {
                                                        Some((LayerIndex::Head((o, i)), w, loss_delta))
                                                    }
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            } else {
                                let live_seg_class_acts: Vec<[[u32; C]; 2]> = seg_states
                                    .iter()
                                    .filter(|(_, _, state)| state.is_none())
                                    .map(|(_, class_weights, _)| {
                                        [
                                            <[(); C] as Map<W, u32>>::map(&class_weights, |w| w.bma(false)),
                                            <[(); C] as Map<W, u32>>::map(&class_weights, |w| w.bma(true)),
                                        ]
                                    })
                                    .collect();
                                let loss_deltas: Vec<i64> = (0..2usize.pow(live_seg_class_acts.len() as u32))
                                    .map(|i| {
                                        live_seg_class_acts
                                            .iter()
                                            .enumerate()
                                            .fold(dead_class_acts, |class_acts, (seg_index, live_class_acts)| {
                                                <[(); C] as ZipMap<u32, u32, u32>>::zip_map(&class_acts, &live_class_acts[i.bit(seg_index) as usize], |sum, live_act| sum + live_act)
                                            })
                                            .iter()
                                            .zip(target_acts.iter())
                                            .map(|(&act, &target_act)| {
                                                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                                                (dist as u64).pow(2)
                                            })
                                            .sum::<u64>() as i64
                                            - null_loss as i64
                                    })
                                    .collect();
                                // for each weight
                                W::states()
                                    .iter()
                                    .map(|&w: &W| {
                                        let seg_acts: Vec<<[[IPS; PY]; PX] as BitPack<bool>>::T> = seg_states
                                            .iter()
                                            .filter_map(|&((sx, sy), _, state)| {
                                                if let Some(_) = state {
                                                    None
                                                } else {
                                                    let counts = <IS as SegmentedPixelFold<
                                                        (usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32),
                                                        ([[<IPS as BitPack<bool>>::T; PY]; PX], u32, Option<bool>, bool),
                                                        SX,
                                                        SY,
                                                        PX,
                                                        PY,
                                                    >>::seg_fold(
                                                        &patches,
                                                        sx,
                                                        sy,
                                                        <(usize, [[<IPS as Pack<u32>>::T; PY]; PX], u32)>::default(),
                                                        |mut acc, (patch, cur_sum, state, _)| {
                                                            if let Some(act) = state {
                                                                <[[IPS; PY]; PX]>::none_option_counted_increment_in_place(*act, &mut acc);
                                                            } else {
                                                                let acts = <[[IPS; PY]; PX]>::acts_simple(weights_channel, patch, *cur_sum, w);
                                                                <[[IPS; PY]; PX]>::some_option_counted_increment_in_place(&acts, &mut acc);
                                                            }
                                                            acc
                                                        },
                                                    );
                                                    let (n, seg_counts) = <[[IPS; PY]; PX]>::finalize_option_counted_increment(counts);
                                                    let threshold = n as u32 / 2;
                                                    let acts = <[[IPS; PY]; PX] as PackedMap<u32, bool>>::map(&seg_counts, |&count| count > threshold);
                                                    Some(acts)
                                                }
                                            })
                                            .collect();

                                        <[[IPS; PY]; PX] as Shape>::indices()
                                            .filter_map(|i| {
                                                if <[[IPS; PY]; PX] as PackedIndexSGet<W>>::get(&weights_channel, i) == w {
                                                    None
                                                } else {
                                                    let loss_index: usize = seg_acts.iter().enumerate().fold(0usize, |loss_index, (bit_index, seg)| {
                                                        loss_index | ((<[[IPS; PY]; PX] as PackedIndexSGet<bool>>::get(&seg, i) as usize) << bit_index)
                                                    });
                                                    Some((LayerIndex::Head((o, i)), w, loss_deltas[loss_index]))
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .flatten()
                                    .collect::<Vec<_>>()
                            }
                        } else {
                            vec![]
                        }
                    })
                    .flatten()
                    .chain(
                        fc_layer
                            .loss_deltas(&null_seg_acts, threshold, class)
                            .iter()
                            .par_bridge()
                            .map(|&(i, w, l)| (LayerIndex::Tail(SegmentedAvgPoolIndex { index: i }), w, l)),
                    )
                    .filter(|(_, _, l): &(_, _, i64)| (l.abs() as u64) > threshold)
                    .collect()
            }
            _ => {
                panic!("{} is not an implementation version", IMPL)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Cache, Conv2D, FcMSE, FusedConvSegmentedAvgPoolFcMSE, GlobalAvgPool, Model, ObjCache,
        SegmentedAvgPool, FC,
    };
    use crate::bits::{
        b128, b16, b32, b8, t128, t32, t8, BitPack, BitScaler, PackedIndexSGet, WeightArray,
    };
    use crate::shape::{Map, ZipMap};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;

    macro_rules! test_loss_deltas_conv {
        ($name:ident, $input_pixel:ty, $input_x:expr, $input_y:expr, $model:ty, $n_classes:expr, $n_iters:expr, $seed:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64($seed + 2);
                (0..$n_iters).for_each(|_| {
                    let inputs: [[$input_pixel; $input_y]; $input_x] = rng.gen();
                    let weights = <$model as Model<
                        [[$input_pixel; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut rng);
                    for class in 0..$n_classes {
                        let mut loss_deltas = <$model as Model<
                            [[$input_pixel; $input_y]; $input_x],
                            $n_classes,
                        >>::loss_deltas(
                            &weights, &inputs, 0, class
                        );
                        loss_deltas.sort();
                        let mut true_loss_deltas = <$model as Model<
                            [[$input_pixel; $input_y]; $input_x],
                            $n_classes,
                        >>::loss_deltas_slow(
                            &weights, &inputs, 0, class
                        );
                        true_loss_deltas.sort();
                        assert_eq!(true_loss_deltas, loss_deltas);
                    }
                })
            }
        };
    }

    macro_rules! compare_impl_loss_deltas_conv {
        ($name:ident, $input_pixel:ty, $input_x:expr, $input_y:expr, $model_b:ty, $model_a:ty, $n_classes:expr, $n_iters:expr, $seed:expr) => {
            #[test]
            fn $name() {
                (0..$n_iters).for_each(|i| {
                    let seed = i + ($seed * $n_iters);
                    let inputs: [[$input_pixel; $input_y]; $input_x] =
                        Hc128Rng::seed_from_u64(seed).gen();
                    let weights_a = <$model_a as Model<
                        [[$input_pixel; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut Hc128Rng::seed_from_u64(seed));
                    let weights_b = <$model_b as Model<
                        [[$input_pixel; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut Hc128Rng::seed_from_u64(seed));
                    for class in 0..$n_classes {
                        let mut loss_deltas_a = weights_a.loss_deltas(&inputs, 0, class);
                        loss_deltas_a.sort();
                        let mut loss_deltas_b = weights_b.loss_deltas(&inputs, 0, class);
                        loss_deltas_b.sort();
                        assert_eq!(loss_deltas_a, loss_deltas_b);
                    }
                })
            }
        };
    }

    macro_rules! test_input_loss_deltas_conv {
        ($name:ident, $input_pixel_shape:ty, $input_x:expr, $input_y:expr, $model:ty, $n_classes:expr, $n_chan_iters:expr, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let input: [[<$input_pixel_shape as BitPack<bool>>::T; $input_y]; $input_x] =
                        rng.gen();
                    let weights = <$model as Model<
                        [[<$input_pixel_shape as BitPack<bool>>::T; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut rng);

                    for class in 0..$n_classes {
                        let null_loss = weights.loss(&input, class);
                        let cache = weights.cache(&input, class);
                        for i in <$input_pixel_shape as Shape>::indices() {
                            let null_chan = <[[(); $input_y]; $input_x] as Map<
                                <$input_pixel_shape as BitPack<bool>>::T,
                                bool,
                            >>::map(&input, |pixel| {
                                <$input_pixel_shape>::get(pixel, i)
                            });
                            let chan_cache = weights.subtract_input(&cache, i, &null_chan);
                            for c in 0..$n_chan_iters {
                                let new_channel: [[bool; $input_y]; $input_x] = rng.gen();
                                let loss_delta = chan_cache.loss_delta(&new_channel);

                                let new_input = <[[(); $input_y]; $input_x] as ZipMap<
                                    <$input_pixel_shape as BitPack<bool>>::T,
                                    bool,
                                    <$input_pixel_shape as BitPack<bool>>::T,
                                >>::zip_map(
                                    &input,
                                    &new_channel,
                                    |&pixel, &bit| <$input_pixel_shape>::set(pixel, i, bit),
                                );
                                let true_loss = weights.loss(&new_input, class);
                                assert_eq!(loss_delta, true_loss as i64 - null_loss as i64);
                            }
                        }
                    }
                })
            }
        };
    }

    macro_rules! test_loss_deltas {
        ($name:ident, $input:ty, $weights:ty, $n_classes:expr, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: $input = rng.gen();
                    let weights = <$weights as Model<$input, $n_classes>>::rand(&mut rng);
                    for class in 0..$n_classes {
                        for &threshold in &[0, 10, 100, 1000] {
                            let mut true_loss_deltas =
                                weights.loss_deltas_slow(&inputs, threshold, class);
                            true_loss_deltas.sort();
                            let mut loss_deltas = weights.loss_deltas(&inputs, threshold, class);
                            loss_deltas.sort();
                            assert_eq!(true_loss_deltas, loss_deltas);
                        }
                    }
                })
            }
        };
    }

    macro_rules! test_input_loss_deltas {
        ($name:ident, $input_shape:ty, $weights:ty, $n_classes:expr, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: <$input_shape as BitPack<bool>>::T = rng.gen();
                    let weights = <$weights as Model<
                        <$input_shape as BitPack<bool>>::T,
                        $n_classes,
                    >>::rand(&mut rng);

                    for class in 0..$n_classes {
                        let null_loss = weights.loss(&inputs, class);
                        let cache = weights.cache(&inputs, class);
                        for i in <$input_shape as Shape>::indices() {
                            let chan_cache =
                                weights.subtract_input(&cache, i, &<$input_shape>::get(&inputs, i));
                            for sign in &[false, true] {
                                let loss_delta = chan_cache.loss_delta(sign);
                                let true_loss =
                                    weights.loss(&<$input_shape>::set(inputs, i, *sign), class);
                                assert_eq!(loss_delta, true_loss as i64 - null_loss as i64);
                            }
                        }
                    }
                })
            }
        };
    }

    test_loss_deltas!(rand_fcmse_bit_losses_small, b8, FcMSE<[(); 8], bool, 7>, 7, 10_000);
    test_loss_deltas!(rand_fcmse_bit_losses_large, [b128; 3], FcMSE<[[(); 128]; 3], bool, 10>, 10, 100);

    test_loss_deltas!(rand_fcmse_trit_losses_small, b8, FcMSE<[(); 8], Option<bool>, 7>, 7, 10_000);
    test_loss_deltas!(rand_fcmse_trit_losses_large, [b128; 3], FcMSE<[[(); 128]; 3], Option<bool>, 10>, 10, 100);

    test_loss_deltas!(rand_fc_fcmse_bit_losses_large, [b8; 3], FC<[[(); 8]; 3], bool, [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 10>, 10>, 10, 100);
    test_loss_deltas!(rand_fc_fcmse_trit_losses_large, [b8; 3], FC<[[(); 8]; 3], Option<bool>, [[(); 32]; 2], FcMSE<[[(); 32]; 2], Option<bool>, 10>, 10>, 10, 100);

    test_input_loss_deltas!(rand_fcmse_bit_input_losses, [(); 8], FcMSE<[(); 8], bool, 7>, 7, 10_000);
    test_input_loss_deltas!(rand_fcmse_trit_input_losses, [(); 8], FcMSE<[(); 8], Option<bool>, 7>, 7, 10_000);

    test_input_loss_deltas!(rand_fcmse_bit_input_losses_large, [[(); 32]; 3], FcMSE<[[(); 32]; 3], bool, 10>, 10, 1_000);
    test_input_loss_deltas!(rand_fcmse_trit_input_losses_large, [[(); 32]; 3], FcMSE<[[(); 32]; 3], Option<bool>, 10>, 10, 1_000);

    test_input_loss_deltas!(rand_fc_fcmse_bit_input_losses_large, [[(); 8]; 3], FC<[[(); 8]; 3], bool, [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 10>, 10>, 10, 100);
    test_input_loss_deltas!(rand_fc_fcmse_trit_input_losses_large, [[(); 8]; 3], FC<[[(); 8]; 3], Option<bool>, [[(); 32]; 2], FcMSE<[[(); 32]; 2], Option<bool>, 10>, 10>, 10, 100);

    test_loss_deltas_conv!(rand_seg_avg_pool_fcmse_bit_small, [b32; 1], 5, 5, SegmentedAvgPool<[[(); 5]; 5], [[(); 32]; 1], FcMSE<[[[[(); 32]; 1]; 2]; 2], bool, 7>, 2, 2, 3, 3>, 7, 100, 0);
    test_loss_deltas_conv!(rand_global_avg_pool_fcmse_bit_small, [b32; 1], 5, 5, GlobalAvgPool<[[(); 5]; 5], [[(); 32]; 1], FcMSE<[[(); 32]; 1], bool, 7>, 3, 3>, 7, 100, 0);
    test_loss_deltas_conv!(rand_global_avg_pool_fcmse_trit_small, [b32; 1], 5, 5, GlobalAvgPool<[[(); 5]; 5], [[(); 32]; 1], FcMSE<[[(); 32]; 1], Option<bool>, 7>, 3, 3>, 7, 100, 0);
    test_loss_deltas_conv!(rand_global_avg_pool_fcmse_bit_large, [b32; 2], 16, 16, GlobalAvgPool<[[(); 16]; 16], [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 7>, 3, 3>, 7, 100, 0);

    test_loss_deltas_conv!(
        rand_fused_conv_seg_avg_2x2_pool_fc_mse_trit_small,
        b8,
        16,
        16,
        FusedConvSegmentedAvgPoolFcMSE::<
            [[(); 16]; 16],
            [(); 8],
            [(); 8],
            Option<bool>,
            6,
            2,
            2,
            3,
            3,
            5,
        >,
        5,
        100,
        0
    );

    test_loss_deltas_conv!(
        rand_fused_conv_seg_avg_3x3_pool_fc_mse_bit_small,
        b8,
        16,
        16,
        FusedConvSegmentedAvgPoolFcMSE::<[[(); 16]; 16], [(); 8], [(); 8], bool, 6, 3, 3, 3, 3, 5>,
        5,
        500,
        0
    );

    test_loss_deltas_conv!(
        rand_fused_conv_seg_avg_2x2_pool_fc_mse_bit_small,
        b8,
        16,
        16,
        FusedConvSegmentedAvgPoolFcMSE::<[[(); 16]; 16], [(); 8], [(); 8], bool, 6, 2, 2, 3, 3, 5>,
        5,
        500,
        0
    );

    test_loss_deltas_conv!(
        rand_fused_conv_seg_avg_1x1_pool_fc_mse_bit_small,
        b8,
        16,
        16,
        FusedConvSegmentedAvgPoolFcMSE::<[[(); 16]; 16], [(); 8], [(); 8], bool, 6, 1, 1, 3, 3, 5>,
        5,
        500,
        0
    );

    test_loss_deltas_conv!(
        rand_fused_conv_seg_avg_2x2_pool_fc_mse_bit_large,
        [b32; 2],
        16,
        16,
        FusedConvSegmentedAvgPoolFcMSE::<
            [[(); 16]; 16],
            [[(); 32]; 2],
            [(); 32],
            bool,
            6,
            2,
            2,
            3,
            3,
            7,
        >,
        7,
        5,
        0
    );

    type FusedConvPoolModel5 = FusedConvSegmentedAvgPoolFcMSE<
        [[(); 16]; 16],
        [[(); 8]; 1],
        [[(); 8]; 1],
        bool,
        5,
        2,
        2,
        3,
        3,
        5,
    >;
    type FusedConvPoolModel6 = FusedConvSegmentedAvgPoolFcMSE<
        [[(); 16]; 16],
        [[(); 8]; 1],
        [[(); 8]; 1],
        bool,
        6,
        2,
        2,
        3,
        3,
        5,
    >;
    type FusedConvPoolModel7 = FusedConvSegmentedAvgPoolFcMSE<
        [[(); 16]; 16],
        [[(); 8]; 1],
        [[(); 8]; 1],
        bool,
        7,
        2,
        2,
        3,
        3,
        5,
    >;

    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_bit_impl_compare_5_6_a,
        [b8; 1],
        16,
        16,
        FusedConvPoolModel5,
        FusedConvPoolModel6,
        5,
        1000,
        0
    );
    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_bit_impl_compare_5_6_b,
        [b8; 1],
        16,
        16,
        FusedConvPoolModel5,
        FusedConvPoolModel6,
        5,
        1000,
        1
    );
    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_bit_impl_compare_5_6_c,
        [b8; 1],
        16,
        16,
        FusedConvPoolModel5,
        FusedConvPoolModel6,
        5,
        1000,
        2
    );
    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_bit_impl_compare_5_6_d,
        [b8; 1],
        16,
        16,
        FusedConvPoolModel5,
        FusedConvPoolModel6,
        5,
        1000,
        3
    );

    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_bit_impl_compare_5_7_a,
        [b8; 1],
        16,
        16,
        FusedConvPoolModel5,
        FusedConvPoolModel7,
        5,
        1000,
        0
    );
    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_bit_impl_compare_5_7_b,
        [b8; 1],
        16,
        16,
        FusedConvPoolModel5,
        FusedConvPoolModel7,
        5,
        1000,
        1
    );
    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_bit_impl_compare_5_7_c,
        [b8; 1],
        16,
        16,
        FusedConvPoolModel5,
        FusedConvPoolModel7,
        5,
        1000,
        2
    );
    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_bit_impl_compare_5_7_d,
        [b8; 1],
        16,
        16,
        FusedConvPoolModel5,
        FusedConvPoolModel7,
        5,
        1000,
        3
    );

    type FusedConvPoolModel5Large = FusedConvSegmentedAvgPoolFcMSE<
        [[(); 32]; 32],
        [[(); 32]; 1],
        [[(); 32]; 1],
        Option<bool>,
        5,
        3,
        3,
        3,
        3,
        10,
    >;
    type FusedConvPoolModel7Large = FusedConvSegmentedAvgPoolFcMSE<
        [[(); 32]; 32],
        [[(); 32]; 1],
        [[(); 32]; 1],
        Option<bool>,
        7,
        3,
        3,
        3,
        3,
        10,
    >;

    compare_impl_loss_deltas_conv!(
        fused_conv_seg_pool_trit_impl_compare_5_7_large,
        [b32; 1],
        32,
        32,
        FusedConvPoolModel5Large,
        FusedConvPoolModel7Large,
        10,
        100,
        0
    );

    type ConvPoolModel0 = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        bool,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], bool, 7>, 3, 3>,
        0,
        3,
        3,
        7,
    >;
    type ConvPoolModel1 = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        bool,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], bool, 7>, 3, 3>,
        1,
        3,
        3,
        7,
    >;
    type ConvPoolModel2 = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        bool,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], bool, 7>, 3, 3>,
        2,
        3,
        3,
        7,
    >;
    type ConvPoolModel3 = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        bool,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], bool, 7>, 3, 3>,
        3,
        3,
        3,
        7,
    >;
    type ConvPoolModel4 = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        bool,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], bool, 7>, 3, 3>,
        4,
        3,
        3,
        7,
    >;

    type TritConvPoolModel2 = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        Option<bool>,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], Option<bool>, 7>, 3, 3>,
        2,
        3,
        3,
        7,
    >;
    type TritConvPoolModel3 = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        Option<bool>,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], Option<bool>, 7>, 3, 3>,
        3,
        3,
        3,
        7,
    >;
    type TritConvPoolModel4 = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        Option<bool>,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], Option<bool>, 7>, 3, 3>,
        4,
        3,
        3,
        7,
    >;

    compare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_bit_impl_compare_0_3,
        [b32; 1],
        8,
        8,
        ConvPoolModel0,
        ConvPoolModel3,
        7,
        20,
        0
    );
    compare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_bit_impl_compare_1_3,
        [b32; 1],
        8,
        8,
        ConvPoolModel1,
        ConvPoolModel3,
        7,
        50,
        0
    );
    compare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_bit_impl_compare_2_3,
        [b32; 1],
        8,
        8,
        ConvPoolModel2,
        ConvPoolModel3,
        7,
        100,
        0
    );
    compare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_bit_impl_compare_4_3,
        [b32; 1],
        8,
        8,
        ConvPoolModel4,
        ConvPoolModel3,
        7,
        100,
        0
    );

    compare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_trit_impl_compare_2_3,
        [b32; 1],
        8,
        8,
        TritConvPoolModel2,
        TritConvPoolModel3,
        7,
        20,
        0
    );

    compare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_trit_impl_compare_4_3,
        [b32; 1],
        8,
        8,
        TritConvPoolModel4,
        TritConvPoolModel3,
        7,
        20,
        0
    );

    type ConvPoolModel = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 1],
        [[(); 32]; 1],
        bool,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 1], FcMSE<[[(); 32]; 1], bool, 7>, 3, 3>,
        1,
        3,
        3,
        7,
    >;

    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_small_a,
        [b32; 1],
        8,
        8,
        ConvPoolModel,
        7,
        40,
        0
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_small_b,
        [b32; 1],
        8,
        8,
        ConvPoolModel,
        7,
        40,
        1
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_small_c,
        [b32; 1],
        8,
        8,
        ConvPoolModel,
        7,
        40,
        2
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_small_d,
        [b32; 1],
        8,
        8,
        ConvPoolModel,
        7,
        40,
        3
    );

    type ConvPoolModelLarge = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 3],
        [[(); 32]; 2],
        bool,
        GlobalAvgPool<[[(); 8]; 8], [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 5>, 3, 3>,
        1,
        3,
        3,
        5,
    >;
    type Conv2x2SegPoolModelLarge = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 3],
        [[(); 32]; 2],
        bool,
        SegmentedAvgPool<
            [[(); 8]; 8],
            [[(); 32]; 2],
            FcMSE<[[[[(); 32]; 2]; 2]; 2], bool, 5>,
            2,
            2,
            3,
            3,
        >,
        1,
        3,
        3,
        5,
    >;
    type Conv3x3SegPoolModelLarge = Conv2D<
        [[(); 8]; 8],
        [[(); 32]; 3],
        [[(); 32]; 2],
        bool,
        SegmentedAvgPool<
            [[(); 8]; 8],
            [[(); 32]; 2],
            FcMSE<[[[[(); 32]; 2]; 3]; 3], bool, 5>,
            3,
            3,
            3,
            3,
        >,
        1,
        3,
        3,
        5,
    >;

    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_a,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        0
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_b,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        1
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_c,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        2
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_d,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        3
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_e,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        4
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_f,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        5
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_g,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        6
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_h,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        7
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_i,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        8
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_large_l,
        [b32; 3],
        8,
        8,
        ConvPoolModelLarge,
        5,
        3,
        9
    );

    test_loss_deltas_conv!(
        rand_conv_2x2_seg_avg_pool_fcmse_bit_large_a,
        [b32; 3],
        8,
        8,
        Conv2x2SegPoolModelLarge,
        5,
        3,
        0
    );
    test_loss_deltas_conv!(
        rand_conv_2x2_seg_avg_pool_fcmse_bit_large_b,
        [b32; 3],
        8,
        8,
        Conv2x2SegPoolModelLarge,
        5,
        3,
        1
    );
    test_loss_deltas_conv!(
        rand_conv_2x2_seg_avg_pool_fcmse_bit_large_c,
        [b32; 3],
        8,
        8,
        Conv2x2SegPoolModelLarge,
        5,
        3,
        2
    );
    test_loss_deltas_conv!(
        rand_conv_2x2_seg_avg_pool_fcmse_bit_large_d,
        [b32; 3],
        8,
        8,
        Conv2x2SegPoolModelLarge,
        5,
        3,
        3
    );

    test_loss_deltas_conv!(
        rand_conv_3x3_seg_avg_pool_fcmse_bit_large_a,
        [b32; 3],
        8,
        8,
        Conv3x3SegPoolModelLarge,
        5,
        3,
        0
    );
    test_loss_deltas_conv!(
        rand_conv_3x3_seg_avg_pool_fcmse_bit_large_b,
        [b32; 3],
        8,
        8,
        Conv3x3SegPoolModelLarge,
        5,
        3,
        1
    );
    test_loss_deltas_conv!(
        rand_conv_3x3_seg_avg_pool_fcmse_bit_large_c,
        [b32; 3],
        8,
        8,
        Conv3x3SegPoolModelLarge,
        5,
        3,
        2
    );
    test_loss_deltas_conv!(
        rand_conv_3x3_seg_avg_pool_fcmse_bit_large_d,
        [b32; 3],
        8,
        8,
        Conv3x3SegPoolModelLarge,
        5,
        3,
        3
    );

    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_bit_small, [(); 32], 6, 6, GlobalAvgPool<[[(); 6]; 6], [(); 32], FcMSE<[(); 32], bool, 7>, 3, 3>, 7, 200, 1000);
    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_trit_small, [(); 32], 6, 6, GlobalAvgPool<[[(); 6]; 6], [(); 32], FcMSE<[(); 32], Option<bool>, 7>, 3, 3>, 7, 200, 1000);
    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_bit_large, [[(); 32]; 2], 16, 16, GlobalAvgPool<[[(); 16]; 16], [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 7>, 3, 3>, 7, 20, 100);
}
