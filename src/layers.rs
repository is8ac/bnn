use crate::bits::{
    BitScaler, IncrementCounters, PackedArray, PackedElement, PackedIndexMap, PackedMap,
    WeightArray, BMA,
};
use crate::image2d::{Conv, ImageShape, Pixel, PixelFold, PixelMap};
use crate::shape::{Element, IndexGet, LongDefault, Map, MutMap, Shape, ZipMap};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::marker::PhantomData;

pub struct LayerIter<I: Shape, O: Shape, H, const C: usize>
where
    bool: PackedElement<O>,
{
    cur_output_index: Option<O::Index>,
    input_index: I::IndexIter,
    output_index: O::IndexIter,
    head_iter: H,
}

impl<I: Shape, O: Shape, H, const C: usize> LayerIter<I, O, H, C>
where
    bool: PackedElement<O>,
{
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
    bool: PackedElement<O>,
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

pub trait Model<I, const C: usize>
where
    Self: Sized + Copy,
    Self::Weight: 'static + BitScaler,
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
        cur_value: Self::InputValue,
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
            .map(|w| {
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
pub struct FcMSE<S: Shape, W: PackedElement<S>, const C: usize>
where
    [<W as PackedElement<S>>::Array; C]: Copy + std::fmt::Debug,
{
    pub fc: [<W as PackedElement<S>>::Array; C],
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
            .enumerate()
            .map(|(i, &(w, sum, target_act))| {
                let act = sum + w.bma(input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                (dist as u64).pow(2)
            })
            .sum::<u64>() as i64
            - self.null_loss
    }
}

impl<S, W, const C: usize> Model<<bool as PackedElement<S>>::Array, C> for FcMSE<S, W, C>
where
    S: Shape + PackedIndexMap<bool> + Copy,
    W: 'static + BitScaler + PackedElement<S>,
    bool: PackedElement<S>,
    (): WeightArray<S, W>,
    <W as PackedElement<S>>::Array:
        Copy + std::fmt::Debug + BMA + PackedArray<Element = W, Shape = S>,
    [<W as PackedElement<S>>::Array; C]: LongDefault,
    <S as Shape>::Index: Copy,
    <bool as PackedElement<S>>::Array: Copy + PackedArray<Element = bool, Shape = S>,
    [(W, u32, u32); C]: Default,
    [(u32, u32); C]: Default,
    [u32; C]: Default,
    Standard: Distribution<[<W as PackedElement<S>>::Array; C]>,
{
    type Index = (usize, <S as Shape>::Index);
    type Weight = W;
    type IndexIter = FcMSEindexIter<S, C>;
    type Cache = FcMSEcache<C>;
    type ChanCache = FcMSEchanCache<W, C>;
    type InputValue = bool;
    type InputIndex = <S as Shape>::Index;
    type Output = [u32; C];
    fn rand<R: Rng>(rng: &mut R) -> Self {
        FcMSE { fc: rng.gen() }
    }
    fn apply(&self, input: &<bool as PackedElement<S>>::Array) -> [u32; C] {
        let mut target = <[u32; C]>::default();
        for c in 0..C {
            target[c] = self.fc[c].bma(input);
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
        self.fc[head].set_element_in_place(tail, weight);
    }
    fn top_act(&self, input: &<bool as PackedElement<S>>::Array) -> usize {
        self.fc
            .iter()
            .map(|w| w.bma(input))
            .enumerate()
            .max_by_key(|(_, a)| *a)
            .unwrap()
            .0
    }
    fn cache(&self, input: &<bool as PackedElement<S>>::Array, class: usize) -> FcMSEcache<C> {
        let mut target = <[(u32, u32); C]>::default();
        for c in 0..C {
            target[c] = (
                self.fc[c].bma(input),
                (c == class) as u32 * <() as WeightArray<S, W>>::MAX,
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
        cur_value: bool,
    ) -> FcMSEchanCache<W, C> {
        let mut target = <[(W, u32, u32); C]>::default();
        for c in 0..C {
            let weight = self.fc[c].get_element(chan_index);
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
    fn loss(&self, input: &<bool as PackedElement<S>>::Array, class: usize) -> u64 {
        self.fc
            .iter()
            .enumerate()
            .map(|(c, w)| {
                let target_act = (c == class) as u32 * <() as WeightArray<S, W>>::MAX;
                let act = w.bma(input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                (dist as u64).pow(2)
            })
            .sum()
    }
    fn loss_deltas(
        &self,
        input: &<bool as PackedElement<S>>::Array,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, W, i64)> {
        self.fc
            .iter()
            .enumerate()
            .map(|(c, w)| {
                let target_act = (c == class) as u32 * <() as WeightArray<S, W>>::MAX;
                let act = w.bma(input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                let class_null_loss = (dist as u64).pow(2) as i64;
                <()>::loss_deltas(&w, input, threshold, |act| {
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
pub struct FC<I, W, O: Shape, H: Model<<bool as PackedElement<O>>::Array, C>, const C: usize>
where
    bool: PackedElement<O>,
    I: Shape,
    W: PackedElement<I>,
    <W as PackedElement<I>>::Array: Element<O>,
    <<W as PackedElement<I>>::Array as Element<O>>::Array: Copy,
{
    fc: <<W as PackedElement<I>>::Array as Element<O>>::Array,
    tail: H,
}

pub struct FCcache<O: Shape, H: Model<<bool as PackedElement<O>>::Array, C>, const C: usize>
where
    bool: PackedElement<O>,
    u32: Element<O>,
{
    is_alive: bool,
    sums: <u32 as Element<O>>::Array,
    true_class: usize,
    null_loss: i64,
    head: PhantomData<H>,
}

pub struct FCchanCache<
    I,
    W,
    O: Shape,
    H: Model<<bool as PackedElement<O>>::Array, C>,
    const C: usize,
> where
    I: Shape,
    bool: PackedElement<O> + PackedElement<I>,
    <bool as PackedElement<I>>::Array: PackedArray<Element = bool, Shape = I> + Copy,
    (W, u32): Element<O>,
{
    input_shape: PhantomData<I>,
    is_alive: bool,
    sums: <(W, u32) as Element<O>>::Array,
    true_class: usize,
    null_loss: i64,
    head: H,
}

impl<
        I,
        W,
        O: Shape + PackedMap<(W, u32), bool>,
        H: Model<<bool as PackedElement<O>>::Array, C>,
        const C: usize,
    > Cache<bool> for FCchanCache<I, W, O, H, C>
where
    W: 'static + BitScaler + PackedElement<I>,
    I: Shape + PackedIndexMap<bool>,
    bool: PackedElement<O> + PackedElement<I>,
    <W as PackedElement<I>>::Array: Copy + BMA + PackedArray<Element = W, Shape = I>,
    <bool as PackedElement<I>>::Array: PackedArray<Element = bool, Shape = I> + Copy,
    (W, u32): Element<O>,
    (): WeightArray<I, W>,
{
    fn loss_delta(&self, &input: &bool) -> i64 {
        let hidden = <O as PackedMap<(W, u32), bool>>::map(&self.sums, |(w, sum)| {
            (sum + w.bma(input)) > <() as WeightArray<I, W>>::THRESHOLD
        });
        self.head.loss(&hidden, self.true_class) as i64 - self.null_loss
    }
}

impl<I, W, O, H, const C: usize> Model<<bool as PackedElement<I>>::Array, C> for FC<I, W, O, H, C>
where
    (): Element<O, Array = O> + WeightArray<I, W>,
    W: 'static + BitScaler + PackedElement<I>,
    H: Model<
        <bool as PackedElement<O>>::Array,
        C,
        Weight = W,
        InputIndex = O::Index,
        InputValue = bool,
    >,
    O: PackedMap<<W as PackedElement<I>>::Array, bool>
        + Shape
        + Default
        + MutMap<(), <W as PackedElement<I>>::Array>
        + ZipMap<<W as PackedElement<I>>::Array, u32, (W, u32)>
        + Map<<W as PackedElement<I>>::Array, u32>
        + PackedMap<(W, u32), bool>,
    Self: Copy,
    bool: PackedElement<I> + PackedElement<O>,
    I: Shape + PackedIndexMap<bool>,
    <W as PackedElement<I>>::Array: Element<O> + Copy + BMA + PackedArray<Element = W, Shape = I>,
    <bool as PackedElement<I>>::Array: PackedArray<Element = bool, Shape = I> + Copy,
    <<W as PackedElement<I>>::Array as Element<O>>::Array:
        IndexGet<O::Index, Element = <W as PackedElement<I>>::Array> + Default + Copy,
    Self::ChanCache: Cache<bool>,
    u32: Element<O>,
    (W, u32): Element<O>,
    <bool as PackedElement<O>>::Array: PackedArray<Shape = O, Element = bool>,
    Standard: Distribution<<<W as PackedElement<I>>::Array as Element<O>>::Array>,
{
    type Index = LayerIndex<(O::Index, <I as Shape>::Index), H::Index>;
    type Weight = W;
    type IndexIter = LayerIter<I, O, H::IndexIter, C>;
    type Cache = FCcache<O, H, C>;
    type ChanCache = FCchanCache<I, W, O, H, C>;
    type InputIndex = <I as Shape>::Index;
    type InputValue = bool;
    type Output = <bool as PackedElement<O>>::Array;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        FC {
            fc: rng.gen(),
            tail: H::rand(rng),
        }
    }
    fn apply(
        &self,
        input: &<bool as PackedElement<I>>::Array,
    ) -> <bool as PackedElement<O>>::Array {
        <O as PackedMap<<W as PackedElement<I>>::Array, bool>>::map(&self.fc, |w| {
            <() as WeightArray<I, W>>::act(&w, input)
        })
    }
    fn indices() -> Self::IndexIter {
        LayerIter::new(H::indices())
    }
    fn mutate_in_place(&mut self, index: Self::Index, weight: W) {
        match index {
            LayerIndex::Head((o, i)) => self.fc.index_get_mut(o).set_element_in_place(i, weight),
            LayerIndex::Tail(i) => self.tail.mutate_in_place(i, weight),
        }
    }
    fn top_act(&self, input: &<bool as PackedElement<I>>::Array) -> usize {
        let hidden = self.apply(input);
        self.tail.top_act(&hidden)
    }
    fn cache(&self, input: &<bool as PackedElement<I>>::Array, class: usize) -> Self::Cache {
        FCcache {
            is_alive: true,
            sums: <O as Map<<W as PackedElement<I>>::Array, u32>>::map(&self.fc, |weights| {
                weights.bma(input)
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
        cur_value: bool,
    ) -> Self::ChanCache {
        FCchanCache {
            input_shape: PhantomData::default(),
            is_alive: true,
            sums: <O as ZipMap<<W as PackedElement<I>>::Array, u32, (W, u32)>>::zip_map(
                &self.fc,
                &cache.sums,
                |weights, sum| {
                    let w = weights.get_element(chan_index);
                    (w, sum - w.bma(cur_value))
                },
            ),
            true_class: cache.true_class,
            null_loss: cache.null_loss,
            head: self.tail,
        }
    }
    fn loss(&self, input: &<bool as PackedElement<I>>::Array, class: usize) -> u64 {
        let hidden = self.apply(input);
        self.tail.loss(&hidden, class)
    }
    fn loss_deltas(
        &self,
        inputs: &<bool as PackedElement<I>>::Array,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, W, i64)> {
        let hidden = self.apply(inputs);
        // for a given output, if we flip it, what is the new loss?
        // for fc, we will only call it once, but for conv, we will call it many time for a given channel.
        let cache = self.tail.cache(&hidden, class);
        O::indices()
            .map(|o| {
                let input = hidden.get_element(o);
                let weight_array = self.fc.index_get(o);

                let chan_cache = self.tail.subtract_input(&cache, o, input);
                let deltas = [chan_cache.loss_delta(&false), chan_cache.loss_delta(&true)];

                <()>::loss_deltas(&weight_array, &inputs, threshold, |act| {
                    deltas[(act > <() as WeightArray<I, W>>::THRESHOLD) as usize]
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

impl<IS> Cache<<bool as Pixel<IS>>::Image> for ConvChanCache<IS>
where
    bool: Pixel<IS>,
{
    fn loss_delta(&self, input: &<bool as Pixel<IS>>::Image) -> i64 {
        0
    }
}

#[derive(Copy, Clone)]
pub struct Conv2D<IS, IPS, OPS, W, H, const PX: usize, const PY: usize, const C: usize>
where
    OPS: Shape,
    IPS: Shape,
    H: Model<<<bool as PackedElement<OPS>>::Array as Pixel<IS>>::Image, C>,
    bool: PackedElement<OPS>,
    <bool as PackedElement<OPS>>::Array: Pixel<IS>,
    W: PackedElement<[[IPS; PY]; PX]>,
    <W as PackedElement<[[IPS; PY]; PX]>>::Array: Element<OPS>,
    <<W as PackedElement<[[IPS; PY]; PX]>>::Array as Element<OPS>>::Array: Copy + Clone,
{
    kernel: <<W as PackedElement<[[IPS; PY]; PX]>>::Array as Element<OPS>>::Array,
    tail: H,
    image_shape: PhantomData<IS>,
}

impl<IS, IPS, OPS, W, H, const PX: usize, const PY: usize, const C: usize>
    Model<<<bool as PackedElement<IPS>>::Array as Pixel<IS>>::Image, C>
    for Conv2D<IS, IPS, OPS, W, H, PX, PY, C>
where
    IS: ImageShape
        + Copy
        + PixelMap<<bool as PackedElement<OPS>>::Array, bool>
        + Conv<<bool as PackedElement<IPS>>::Array, <bool as PackedElement<OPS>>::Array, PX, PY>
        + Conv<
            <bool as PackedElement<IPS>>::Array,
            <bool as PackedElement<[[IPS; PY]; PX]>>::Array,
            PX,
            PY,
        > + PixelMap<<bool as PackedElement<[[IPS; PY]; PX]>>::Array, bool>,
    IPS: Shape + Copy + PackedIndexMap<bool>,
    OPS: Shape + Copy + PackedMap<<W as PackedElement<[[IPS; PY]; PX]>>::Array, bool>,
    W: 'static + PackedElement<IPS> + Copy + BitScaler + PackedElement<[[IPS; PY]; PX]>,
    H: Model<
        <<bool as PackedElement<OPS>>::Array as Pixel<IS>>::Image,
        C,
        Weight = W,
        InputIndex = OPS::Index,
        InputValue = <bool as Pixel<IS>>::Image,
    >,
    bool: PackedElement<OPS> + PackedElement<IPS> + Pixel<IS>,
    <bool as PackedElement<OPS>>::Array: Pixel<IS> + Default + IndexGet<OPS::Index, Element = bool>,
    <bool as PackedElement<IPS>>::Array:
        Pixel<IS> + PackedArray<Shape = IPS, Element = bool> + Copy,
    <W as PackedElement<IPS>>::Array: Copy + PackedArray<Element = W, Shape = IPS> + BMA + Copy,
    <W as PackedElement<[[IPS; PY]; PX]>>::Array:
        Element<OPS> + PackedArray<Element = W, Shape = [[IPS; PY]; PX]> + BMA + Copy,
    <<W as PackedElement<[[IPS; PY]; PX]>>::Array as Element<OPS>>::Array:
        Copy + Clone + IndexGet<OPS::Index, Element = <W as PackedElement<[[IPS; PY]; PX]>>::Array>,
    <bool as PackedElement<[[IPS; PY]; PX]>>::Array: Pixel<IS> + Default,
    Standard: Distribution<<<W as PackedElement<[[IPS; PY]; PX]>>::Array as Element<OPS>>::Array>,
    (): WeightArray<IPS, W>,
{
    type Index = LayerIndex<(OPS::Index, <[[IPS; PY]; PX] as Shape>::Index), H::Index>;
    type Weight = W;
    type IndexIter = LayerIter<[[IPS; PY]; PX], OPS, H::IndexIter, C>;
    type Cache = ConvCache;
    type ChanCache = ConvChanCache<IS>;
    type InputValue = <bool as Pixel<IS>>::Image;
    type InputIndex = IPS::Index;
    type Output = <<bool as PackedElement<OPS>>::Array as Pixel<IS>>::Image;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        Conv2D {
            kernel: rng.gen(),
            tail: H::rand(rng),
            image_shape: PhantomData::default(),
        }
    }
    fn apply(
        &self,
        input: &<<bool as PackedElement<IPS>>::Array as Pixel<IS>>::Image,
    ) -> <<bool as PackedElement<OPS>>::Array as Pixel<IS>>::Image {
        //<() as WeightArray<[[IPS; PY]; PX], W>>::THRESHOLD;
        <IS as Conv<
            <bool as PackedElement<IPS>>::Array,
            <bool as PackedElement<OPS>>::Array,
            PX,
            PY,
        >>::conv(input, |patch| {
            /*
            <OPS as PackedMap<<W as PackedElement<[[IPS; PY]; PX]>>::Array, bool>>::map(&self.kernel, |weights| {
                <() as WeightArray<[[IPS; PY]; PX], W>>::act(weights, &patch)
                //weights.bma(&patch) > <() as WeightArray<[[IPS; PY]; PX], W>>::THRESHOLD
                //true
            })
            */
            Default::default()
        })
    }
    fn indices() -> Self::IndexIter {
        LayerIter::new(H::indices())
    }
    fn mutate_in_place(&mut self, index: Self::Index, weight: W) {
        match index {
            LayerIndex::Head((o, i)) => {
                self.kernel.index_get_mut(o).set_element_in_place(i, weight)
            }
            LayerIndex::Tail(i) => self.tail.mutate_in_place(i, weight),
        }
    }
    fn top_act(&self, input: &<<bool as PackedElement<IPS>>::Array as Pixel<IS>>::Image) -> usize {
        self.tail.top_act(&self.apply(input))
    }
    fn cache(
        &self,
        input: &<<bool as PackedElement<IPS>>::Array as Pixel<IS>>::Image,
        class: usize,
    ) -> ConvCache {
        ConvCache {}
    }
    fn subtract_input(
        &self,
        cache: &ConvCache,
        chan_index: Self::InputIndex,
        cur_value: Self::InputValue,
    ) -> ConvChanCache<IS> {
        ConvChanCache {
            image_shape: PhantomData::default(),
        }
    }
    fn loss(
        &self,
        input: &<<bool as PackedElement<IPS>>::Array as Pixel<IS>>::Image,
        class: usize,
    ) -> u64 {
        self.tail.loss(&self.apply(input), class)
    }
    fn loss_deltas(
        &self,
        input: &<<bool as PackedElement<IPS>>::Array as Pixel<IS>>::Image,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)> {
        let null_acts = self.apply(input);
        let cache = self.tail.cache(&null_acts, class);
        // for each output channel
        /*
        OPS::indices().map(|o| {
            let null_chan_acts = <IS as PixelMap<<bool as PackedElement<OPS>>::Array, bool>>::map(&null_acts, |pixel| pixel.get_element(o));
            let chan_cache = self.tail.subtract_input(&cache, o, null_chan_acts);
            let weights_channel = self.kernel.index_get(o);
            W::states().map(|w| {
                let chan_acts = <IS as Conv<<bool as PackedElement<IPS>>::Array, <bool as PackedElement<[[IPS; PY]; PX]>>::Array, PX, PY>>::conv(input, |patch| {
                    //<() as WeightArray<[[IPS; PY]; PX], W>>::acts(weights_channel, &patch, w)
                    <bool as PackedElement<[[IPS; PY]; PX]>>::Array::default()
                });
                <[[IPS; PY]; PX]>::indices().map(|i| {
                    let mut_weight_acts = <IS as PixelMap<<bool as PackedElement<[[IPS; PY]; PX]>>::Array, bool>>::map(&chan_acts, |pixel| pixel.get_element(i));
                    chan_cache.loss_delta(&mut_weight_acts)
                });
            });
        });
        */
        vec![]
    }
}

pub struct GlobalAvgPoolCache<TailCache, P: Shape, const C: usize>
where
    bool: PackedElement<P>,
{
    tail_cache: TailCache,
    threshold: u32,
    acts: <bool as PackedElement<P>>::Array,
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
    > Cache<<bool as Pixel<I>>::Image> for GlobalAvgPoolChanCache<W, TailChanCache, I, PX, PY, C>
where
    bool: Pixel<I>,
    TailChanCache: Cache<bool>,
    I: PixelFold<u32, bool, PX, PY>,
{
    fn loss_delta(&self, input: &<bool as Pixel<I>>::Image) -> i64 {
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
    P: Shape + PackedMap<u32, bool> + IncrementCounters,
    I: PixelFold<(usize, <u32 as Element<P>>::Array), <bool as PackedElement<P>>::Array, PX, PY>,
    bool: PackedElement<P>,
    <bool as PackedElement<P>>::Array: Pixel<I> + Default,
    u32: Element<P>,
    <u32 as Element<P>>::Array: Default,
{
    fn pool(
        input: &<<bool as PackedElement<P>>::Array as Pixel<I>>::Image,
    ) -> (u32, <bool as PackedElement<P>>::Array) {
        let (n, counts) = <I as PixelFold<
            (usize, <u32 as Element<P>>::Array),
            <bool as PackedElement<P>>::Array,
            PX,
            PY,
        >>::pixel_fold(
            input,
            <(usize, <u32 as Element<P>>::Array)>::default(),
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
    Model<<<bool as PackedElement<P>>::Array as Pixel<I>>::Image, C>
    for GlobalAvgPool<I, P, H, PX, PY>
where
    Self: Copy,
    P: Shape + PackedMap<u32, bool> + IncrementCounters,
    I: PixelFold<(usize, <u32 as Element<P>>::Array), <bool as PackedElement<P>>::Array, PX, PY>
        + PixelFold<u32, bool, PX, PY>,
    H: Model<<bool as PackedElement<P>>::Array, C, InputIndex = P::Index, InputValue = bool>,
    u32: Element<P>,
    <u32 as Element<P>>::Array: Default + std::fmt::Debug,
    bool: PackedElement<P> + Pixel<I>,
    <bool as PackedElement<P>>::Array: Pixel<I> + Default + PackedArray<Element = bool, Shape = P>,
    <H as Model<<bool as PackedElement<P>>::Array, C>>::ChanCache: Cache<bool>,
{
    type Index = H::Index;
    type Weight = H::Weight;
    type IndexIter = H::IndexIter;
    type Cache = GlobalAvgPoolCache<H::Cache, P, C>;
    type ChanCache = GlobalAvgPoolChanCache<H::Weight, H::ChanCache, I, PX, PY, C>;
    type InputIndex = P::Index;
    type InputValue = <bool as Pixel<I>>::Image;
    type Output = <bool as PackedElement<P>>::Array;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        GlobalAvgPool {
            pixel_shape: PhantomData::default(),
            image_shape: PhantomData::default(),
            tail: H::rand(rng),
        }
    }
    fn apply(
        &self,
        input: &<<bool as PackedElement<P>>::Array as Pixel<I>>::Image,
    ) -> <bool as PackedElement<P>>::Array {
        let (_, acts) = Self::pool(input);
        acts
    }
    fn indices() -> H::IndexIter {
        H::indices()
    }
    fn mutate_in_place(&mut self, index: H::Index, weight: H::Weight) {
        self.tail.mutate_in_place(index, weight)
    }
    fn top_act(&self, input: &<<bool as PackedElement<P>>::Array as Pixel<I>>::Image) -> usize {
        self.tail.top_act(&self.apply(input))
    }
    fn cache(
        &self,
        input: &<<bool as PackedElement<P>>::Array as Pixel<I>>::Image,
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
        _: Self::InputValue,
    ) -> Self::ChanCache {
        GlobalAvgPoolChanCache {
            threshold: cache.threshold,
            tail_chan_cache: self.tail.subtract_input(
                &cache.tail_cache,
                chan_index,
                cache.acts.get_element(chan_index),
            ),
            weight_type: PhantomData::default(),
            image_shape: PhantomData::default(),
        }
    }
    fn loss(
        &self,
        input: &<<bool as PackedElement<P>>::Array as Pixel<I>>::Image,
        class: usize,
    ) -> u64 {
        self.tail.loss(&self.apply(input), class)
    }
    fn loss_deltas(
        &self,
        input: &<<bool as PackedElement<P>>::Array as Pixel<I>>::Image,
        threshold: u64,
        class: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)> {
        self.tail.loss_deltas(&self.apply(input), threshold, class)
    }
}

#[cfg(test)]
mod tests {
    use super::{Cache, FcMSE, GlobalAvgPool, Model, FC};
    use crate::bits::{
        b128, b32, b8, t128, t32, t8, BitScaler, PackedArray, PackedElement, WeightArray,
    };
    use crate::shape::{Map, ZipMap};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;

    macro_rules! test_loss_deltas_conv {
        ($name:ident, $input_pixel:ty, $input_x:expr, $input_y:expr, $model:ty, $n_classes:expr, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: [[$input_pixel; $input_y]; $input_x] = rng.gen();
                    let weights = <$model as Model<
                        [[$input_pixel; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut rng);
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

    macro_rules! test_input_loss_deltas_conv {
        ($name:ident, $input_pixel_shape:ty, $input_x:expr, $input_y:expr, $model:ty, $n_classes:expr, $n_chan_iters:expr, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let input: [[<bool as PackedElement<$input_pixel_shape>>::Array; $input_y];
                        $input_x] = rng.gen();
                    let weights = <$model as Model<
                        [[<bool as PackedElement<$input_pixel_shape>>::Array; $input_y]; $input_x],
                        $n_classes,
                    >>::rand(&mut rng);

                    for class in 0..$n_classes {
                        let null_loss = weights.loss(&input, class);
                        let cache = weights.cache(&input, class);
                        for i in <$input_pixel_shape as Shape>::indices() {
                            let null_chan =
                                <[[(); $input_y]; $input_x] as Map<
                                    <bool as PackedElement<$input_pixel_shape>>::Array,
                                    bool,
                                >>::map(&input, |pixel| pixel.get_element(i));
                            let chan_cache = weights.subtract_input(&cache, i, null_chan);
                            for c in 0..$n_chan_iters {
                                let new_channel: [[bool; $input_y]; $input_x] = rng.gen();
                                let loss_delta = chan_cache.loss_delta(&new_channel);

                                let new_input = <[[(); $input_y]; $input_x] as ZipMap<
                                    <bool as PackedElement<$input_pixel_shape>>::Array,
                                    bool,
                                    <bool as PackedElement<$input_pixel_shape>>::Array,
                                >>::zip_map(
                                    &input,
                                    &new_channel,
                                    |pixel, &bit| pixel.set_element(i, bit),
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
                    let inputs: <bool as PackedElement<$input_shape>>::Array = rng.gen();
                    let weights = <$weights as Model<<bool as PackedElement<$input_shape>>::Array, $n_classes>>::rand(&mut rng);

                    for class in 0..$n_classes {
                        let null_loss = weights.loss(&inputs, class);
                        let cache = weights.cache(&inputs, class);
                        for i in <$input_shape as Shape>::indices() {
                            let chan_cache = weights.subtract_input(&cache, i, <<bool as PackedElement<$input_shape>>::Array as PackedArray>::get_element(&inputs, i));
                            for sign in &[false, true] {
                                let loss_delta = chan_cache.loss_delta(sign);
                                let true_loss = weights.loss(&inputs.set_element(i, *sign), class);
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

    test_loss_deltas_conv!(rand_global_avg_pool_fcmse_bit_small, [b32; 2], 16, 16, GlobalAvgPool<[[(); 16]; 16], [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 7>, 3, 3>, 7, 100);

    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_bit_small, [(); 32], 6, 6, GlobalAvgPool<[[(); 6]; 6], [(); 32], FcMSE<[(); 32], bool, 7>, 3, 3>, 7, 200, 1000);
    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_trit_small, [(); 32], 6, 6, GlobalAvgPool<[[(); 6]; 6], [(); 32], FcMSE<[(); 32], Option<bool>, 7>, 3, 3>, 7, 200, 1000);
    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_bit_large, [[(); 32]; 2], 16, 16, GlobalAvgPool<[[(); 16]; 16], [[(); 32]; 2], FcMSE<[[(); 32]; 2], bool, 7>, 3, 3>, 7, 20, 100);
    test_input_loss_deltas_conv!(rand_global_avg_pool_input_losses_bit_small_5x5, [(); 32], 8, 8, GlobalAvgPool<[[(); 8]; 8], [(); 32], FcMSE<[(); 32], bool, 7>, 5, 5>, 7, 20, 100);
}
