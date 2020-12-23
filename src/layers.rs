use crate::bits::{
    BitPack, BitScaler, IncrementCounters, PackedIndexMap, PackedIndexSGet, PackedMap, WeightArray,
    BMA,
};
use crate::image2d::{
    Conv, ImageShape, PixelFold, PixelIndexSGet, PixelMap, PixelPack, PixelZipMap,
};
use crate::shape::{IndexGet, Map, Pack, Shape, ZipMap};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::marker::PhantomData;

pub struct LayerIter<I: Shape, O: Shape, H, const C: usize> {
    cur_output_index: Option<O::Index>,
    input_index: I::IndexIter,
    output_index: O::IndexIter,
    head_iter: H,
}

impl<I: Shape, O: Shape, H, const C: usize> LayerIter<I, O, H, C> {
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

impl<I: Shape, O: Shape, H: Iterator, const C: usize> Iterator for LayerIter<I, O, H, C>
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
            self.head_iter.next().map(|i| LayerIndex::Tail(i))
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

impl<I> IndexDepth for (usize, I) {
    fn depth(&self) -> usize {
        0
    }
}

pub trait Model<I, const C: usize>
where
    Self: Sized + Copy,
    Self::Weight: BitScaler,
    Self::IndexIter: Iterator<Item = Self::Index>,
    Self::Index: Copy,
    Self::ChanCache: Cache<Self::InputValue>,
{
    type Index;
    type Weight;
    type IndexIter;
    type Cache;
    type ChanCache;
    type InputIndex;
    type InputValue;
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
    // init the cache such that we can subtract input elements from it.
    fn cache(&self, input: &I, class: usize) -> Self::Cache;
    // subtract an element from the cache and prepare the cache for addition of a new input element.
    fn subtract_input(
        &self,
        cache: &Self::Cache,
        chan_index: Self::InputIndex,
        cur_value: &Self::InputValue,
    ) -> Self::ChanCache;
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
                    .filter(|(_, _, l)| l.abs() as u64 > threshold)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
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
{
    type Index = (usize, <S as Shape>::Index);
    type Weight = W;
    type IndexIter = FcMSEindexIter<S, C>;
    type Cache = FcMSEcache<C>;
    type ChanCache = FcMSEchanCache<W, C>;
    type InputValue = bool;
    type InputIndex = <S as Shape>::Index;
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
    H: Model<<O as BitPack<bool>>::T, C, Weight = W, InputIndex = O::Index, InputValue = bool>,
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
    <I as BitPack<bool>>::T: Copy,
    <O as Pack<<I as BitPack<W>>::T>>::T: Copy,
    Standard: Distribution<<O as Pack<<I as BitPack<W>>::T>>::T>,
{
    type Index = LayerIndex<(O::Index, <I as Shape>::Index), H::Index>;
    type Weight = W;
    type IndexIter = LayerIter<I, O, H::IndexIter, C>;
    type Cache = FCcache<O, H, C>;
    type ChanCache = FCchanCache<I, W, O, H, C>;
    type InputIndex = <I as Shape>::Index;
    type InputValue = bool;
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

pub struct ConvCache {}

pub struct ConvChanCache<IS> {
    image_shape: PhantomData<IS>,
}

impl<IS> Cache<<IS as PixelPack<bool>>::I> for ConvChanCache<IS>
where
    IS: PixelPack<bool>,
{
    fn loss_delta(&self, input: &<IS as PixelPack<bool>>::I) -> i64 {
        0
    }
}

#[derive(Copy, Clone)]
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
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy,
    IS: PixelPack<<IPS as BitPack<bool>>::T>,
{
    kernel: <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T,
    tail: H,
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
        + PixelMap<<OPS as BitPack<bool>>::T, bool>
        + PixelMap<bool, bool>
        + PixelMap<u32, bool>
        + PixelZipMap<u32, bool, bool>
        + Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, bool, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, <OPS as BitPack<bool>>::T, PX, PY>
        + Conv<<IPS as BitPack<bool>>::T, [[<IPS as BitPack<bool>>::T; PY]; PX], PX, PY>
        + PixelMap<[[<IPS as BitPack<bool>>::T; PY]; PX], bool>,
    H: Model<
            <IS as PixelPack<<OPS as BitPack<bool>>::T>>::I,
            C,
            Weight = W,
            InputIndex = OPS::Index,
            InputValue = <IS as PixelPack<bool>>::I,
        > + Copy,
    Standard: Distribution<<OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T>,
    W: BitScaler + Copy + Eq,
    <IS as PixelPack<bool>>::I: Copy,
    <IS as PixelPack<<OPS as BitPack<bool>>::T>>::I: Default,
    <IPS as BitPack<bool>>::T: Copy,
    <IPS as BitPack<W>>::T: Copy,
    <OPS as Pack<[[<IPS as BitPack<W>>::T; PY]; PX]>>::T: Copy,
    [[IPS; PY]; PX]: WeightArray<W>
        + BitPack<W, T = [[<IPS as BitPack<W>>::T; PY]; PX]>
        + BitPack<bool, T = [[<IPS as BitPack<bool>>::T; PY]; PX]>
        + PackedIndexSGet<W>,
    <[[IPS; PY]; PX] as BitPack<W>>::T: Copy,
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
    type IndexIter = LayerIter<[[IPS; PY]; PX], OPS, H::IndexIter, C>;
    type Cache = ConvCache;
    type ChanCache = ConvChanCache<IS>;
    type InputValue = <IS as PixelPack<bool>>::I;
    type InputIndex = IPS::Index;
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
    fn cache(
        &self,
        input: &<IS as PixelPack<<IPS as BitPack<bool>>::T>>::I,
        class: usize,
    ) -> ConvCache {
        ConvCache {}
    }
    fn subtract_input(
        &self,
        cache: &ConvCache,
        chan_index: Self::InputIndex,
        cur_value: &Self::InputValue,
    ) -> ConvChanCache<IS> {
        ConvChanCache {
            image_shape: PhantomData::default(),
        }
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
                        let null_chan_acts = <IS as PixelMap<<OPS as BitPack<bool>>::T, bool>>::map(
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
                        let null_chan_acts = <IS as PixelMap<<OPS as BitPack<bool>>::T, bool>>::map(
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
                        <IS as PixelMap<<[[IPS; PY]; PX] as BitPack<bool>>::T, bool>>::map(
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
                        let null_chan_full_sum =
                            <IS as Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>>::conv(input, |patch| <[[IPS; PY]; PX] as BMA<W>>::bma(weights_channel, &patch));
                        let null_chan_acts = <IS as PixelMap<u32, bool>>::map(&null_chan_full_sum, |&sum| sum > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD);
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
                                            let new_weight_acts: <IS as PixelPack<bool>>::I =
                                                <IS as PixelZipMap<u32, bool, bool>>::zip_map(&null_chan_full_sum, layer, |sum, input| {
                                                    ((sum - cur_weight.bma(input)) + w.bma(input)) > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD
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
                        let null_chan_full_sum =
                            <IS as Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>>::conv(input, |patch| <[[IPS; PY]; PX] as BMA<W>>::bma(weights_channel, &patch));
                        let null_chan_acts = <IS as PixelMap<u32, bool>>::map(&null_chan_full_sum, |&sum| sum > <[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD);
                        // and subtract it from the cache
                        let chan_cache = self.tail.subtract_input(&cache, o, &null_chan_acts);
                        W::states()
                            .iter()
                            .map(|&w| {
                                let acts: Vec<<IS as PixelPack<bool>>::I> = <IS as Conv<<IPS as BitPack<bool>>::T, u32, PX, PY>>::indices().fold(
                                    <[[IPS; PY]; PX]>::indices().map(|_| null_chan_acts).collect::<Vec<<IS as PixelPack<bool>>::I>>(),
                                    |mut acc, spatial_index| {
                                        let sum: u32 = IS::get_pixel(&null_chan_full_sum, spatial_index);
                                        if (sum > (<[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD - (W::RANGE - 1)))
                                            & (sum < (<[[IPS; PY]; PX] as WeightArray<W>>::THRESHOLD + W::RANGE))
                                        {
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
    H: Model<<P as BitPack<bool>>::T, C, InputIndex = P::Index, InputValue = bool>,
    <H as Model<<P as BitPack<bool>>::T, C>>::ChanCache: Cache<bool>,
    (usize, <P as Pack<u32>>::T): Default,
{
    type Index = H::Index;
    type Weight = H::Weight;
    type IndexIter = H::IndexIter;
    type Cache = GlobalAvgPoolCache<H::Cache, P, C>;
    type ChanCache = GlobalAvgPoolChanCache<H::Weight, H::ChanCache, I, PX, PY, C>;
    type InputIndex = P::Index;
    type InputValue = <I as PixelPack<bool>>::I;
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

#[derive(Default, Copy, Clone)]
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
    tail: H,
}

#[cfg(test)]
mod tests {
    use super::{Cache, Conv2D, FcMSE, GlobalAvgPool, Model, FC};
    use crate::bits::{
        b128, b32, b8, t128, t32, t8, BitPack, BitScaler, PackedIndexSGet, WeightArray,
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
                let mut rng = Hc128Rng::seed_from_u64($seed);
                (0..$n_iters).for_each(|_| {
                    let inputs: [[$input_pixel; $input_y]; $input_x] = rng.gen();
                    let weights = <$model as Model<
                        [[$input_pixel; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut rng);
                    for class in 0..$n_classes {
                        let mut true_loss_deltas = weights.loss_deltas_slow(&inputs, 0, class);
                        true_loss_deltas.sort();
                        let mut loss_deltas = weights.loss_deltas(&inputs, 0, class);
                        loss_deltas.sort();
                        assert_eq!(true_loss_deltas, loss_deltas);
                    }
                })
            }
        };
    }

    macro_rules! comare_impl_loss_deltas_conv {
        ($name:ident, $input_pixel:ty, $input_x:expr, $input_y:expr, $model_b:ty, $model_a:ty, $n_classes:expr, $n_iters:expr, $seed:expr) => {
            #[test]
            fn $name() {
                (0..$n_iters).for_each(|i| {
                    let inputs: [[$input_pixel; $input_y]; $input_x] =
                        Hc128Rng::seed_from_u64($seed * i).gen();
                    let weights_a = <$model_a as Model<
                        [[$input_pixel; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut Hc128Rng::seed_from_u64($seed * i));
                    let weights_b = <$model_b as Model<
                        [[$input_pixel; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut Hc128Rng::seed_from_u64($seed * i));
                    for class in 0..$n_classes {
                        let mut loss_deltas_a = weights_a.loss_deltas(&inputs, 0, class);
                        loss_deltas_a.sort();
                        let mut loss_deltas_b = weights_b.loss_deltas(&inputs, 0, class);
                        loss_deltas_b.sort();
                        dbg!(loss_deltas_a.len());
                        dbg!(loss_deltas_b.len());
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

    test_loss_deltas_conv!(rand_global_avg_pool_fcmse_bit_small, [b32; 2], 5, 5, GlobalAvgPool<[[(); 5]; 5], [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 7>, 3, 3>, 7, 100, 0);
    test_loss_deltas_conv!(rand_global_avg_pool_fcmse_trit_small, [b32; 2], 5, 5, GlobalAvgPool<[[(); 5]; 5], [[(); 32]; 2], FcMSE<[[(); 32]; 2], Option<bool>, 7>, 3, 3>, 7, 100, 0);
    test_loss_deltas_conv!(rand_global_avg_pool_fcmse_bit_large, [b32; 2], 16, 16, GlobalAvgPool<[[(); 16]; 16], [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 7>, 3, 3>, 7, 100, 0);

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

    comare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_bit_impl_compare_0_3,
        [b32; 1],
        8,
        8,
        ConvPoolModel0,
        ConvPoolModel3,
        7,
        50,
        0
    );
    comare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_bit_impl_compare_1_3,
        [b32; 1],
        8,
        8,
        ConvPoolModel1,
        ConvPoolModel3,
        7,
        200,
        0
    );
    comare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_bit_impl_compare_2_3,
        [b32; 1],
        8,
        8,
        ConvPoolModel2,
        ConvPoolModel3,
        7,
        200,
        0
    );
    comare_impl_loss_deltas_conv!(
        conv_global_avg_pool_fcmse_bit_impl_compare_4_3,
        [b32; 1],
        8,
        8,
        ConvPoolModel4,
        ConvPoolModel3,
        7,
        200,
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
        80,
        0
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_small_b,
        [b32; 1],
        8,
        8,
        ConvPoolModel,
        7,
        80,
        1
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_small_c,
        [b32; 1],
        8,
        8,
        ConvPoolModel,
        7,
        80,
        2
    );
    test_loss_deltas_conv!(
        rand_conv_global_avg_pool_fcmse_bit_small_d,
        [b32; 1],
        8,
        8,
        ConvPoolModel,
        7,
        80,
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

    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_bit_small, [(); 32], 6, 6, GlobalAvgPool<[[(); 6]; 6], [(); 32], FcMSE<[(); 32], bool, 7>, 3, 3>, 7, 200, 1000);
    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_trit_small, [(); 32], 6, 6, GlobalAvgPool<[[(); 6]; 6], [(); 32], FcMSE<[(); 32], Option<bool>, 7>, 3, 3>, 7, 200, 1000);
    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_bit_large, [[(); 32]; 2], 16, 16, GlobalAvgPool<[[(); 16]; 16], [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 7>, 3, 3>, 7, 20, 100);
}
