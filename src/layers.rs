use crate::bits::{
    ConstructWeightArray, PackedArray, PackedElement, PackedIndexMap, PackedMap, Weight,
    WeightArray,
};
use crate::shape::{Element, IndexGet, LongDefault, Map, MutMap, Shape, ZipMap};
use rand::Rng;
use std::marker::PhantomData;

pub trait Cache<I> {
    fn loss_delta(&self, input: I) -> i64;
}

#[derive(Copy, Clone, Debug, Ord, Eq, PartialEq, PartialOrd, Hash)]
pub enum LayerIndex<H: Copy, T: Copy> {
    Head(H),
    Tail(T),
}

pub trait Model<I, const C: usize>
where
    Self: Sized + Copy,
    Self::Weight: 'static + Weight,
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
    /// Rand init
    fn rand<R: Rng>(rng: &mut R) -> Self;
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
    fn loss_deltas(&self, input: &I, class: usize) -> Vec<(Self::Index, Self::Weight, i64)>;
    /// same as losses but a lot slower.
    fn loss_deltas_slow(&self, input: &I, class: usize) -> Vec<(Self::Index, Self::Weight, i64)> {
        let null_loss = self.loss(input, class) as i64;
        <Self::Weight as Weight>::states()
            .map(|w| {
                Self::indices()
                    .map(|i| {
                        (
                            i,
                            w,
                            self.mutate(i, w).loss(input, class) as i64 - null_loss,
                        )
                    })
                    .filter(|(_, _, l)| *l != 0)
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FcMSE<I, W: ConstructWeightArray<I>, const C: usize>
where
    [<W as ConstructWeightArray<I>>::WeightArray; C]: Copy + std::fmt::Debug,
{
    pub fc: [<W as ConstructWeightArray<I>>::WeightArray; C],
}

pub struct FcMSEindexIter<I: Shape, const C: usize> {
    class_index: usize,
    input_iter: I::IndexIter,
}

impl<I: Shape, const C: usize> Iterator for FcMSEindexIter<I, C> {
    type Item = (usize, I::Index);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.input_iter.next() {
            Some((self.class_index, i))
        } else {
            self.class_index += 1;
            if self.class_index >= C {
                None
            } else {
                self.input_iter = I::indices();
                self.input_iter.next().map(|i| (self.class_index, i))
            }
        }
    }
}

pub struct FcMSEcache<const C: usize> {
    cache: [(u32, u32); C],
    true_class: usize,
    null_loss: i64,
}

pub struct FcMSEchanCache<W, const C: usize> {
    cache: [(W, u32, u32); C],
    true_class: usize,
    null_loss: i64,
}

impl<W: Weight, const C: usize> Cache<bool> for FcMSEchanCache<W, C> {
    fn loss_delta(&self, input: bool) -> i64 {
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
    W: 'static + Weight + ConstructWeightArray<S>,
    bool: PackedElement<S>,
    (): WeightArray<S, W>,
    <W as ConstructWeightArray<S>>::WeightArray: Copy + std::fmt::Debug,
    [<W as ConstructWeightArray<S>>::WeightArray; C]: LongDefault,
    <S as Shape>::Index: Copy,
    <bool as PackedElement<S>>::Array: Copy + PackedArray<Weight = bool, Shape = S>,
    [(W, u32, u32); C]: Default,
    [(u32, u32); C]: Default,
{
    type Index = (usize, <S as Shape>::Index);
    type Weight = W;
    type IndexIter = FcMSEindexIter<S, C>;
    type Cache = FcMSEcache<C>;
    type ChanCache = FcMSEchanCache<W, C>;
    type InputValue = bool;
    type InputIndex = <S as Shape>::Index;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        let mut weights = <[<W as ConstructWeightArray<S>>::WeightArray; C]>::long_default();
        for c in 0..C {
            weights[c] = <() as WeightArray<S, W>>::rand(rng);
        }
        FcMSE { fc: weights }
    }
    fn indices() -> FcMSEindexIter<S, C> {
        FcMSEindexIter {
            class_index: 0,
            input_iter: S::indices(),
        }
    }
    fn mutate_in_place(&mut self, (head, tail): (usize, <S as Shape>::Index), weight: W) {
        <()>::mutate_in_place(&mut self.fc[head], tail, weight);
    }
    fn top_act(&self, input: &<bool as PackedElement<S>>::Array) -> usize {
        self.fc
            .iter()
            .map(|w| <()>::bma(w, input))
            .enumerate()
            .max_by_key(|(_, a)| *a)
            .unwrap()
            .0
    }
    fn cache(&self, input: &<bool as PackedElement<S>>::Array, class: usize) -> FcMSEcache<C> {
        let mut target = <[(u32, u32); C]>::default();
        for c in 0..C {
            target[c] = (
                <()>::bma(&self.fc[c], input),
                (c == class) as u32 * <() as WeightArray<S, W>>::MAX,
            );
        }
        FcMSEcache {
            cache: target,
            true_class: class,
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
            let weight = <()>::get_weight(&self.fc[c], chan_index);
            target[c] = (
                weight,
                cache.cache[c].0 - weight.bma(cur_value),
                cache.cache[c].1,
            );
        }
        FcMSEchanCache {
            cache: target,
            true_class: cache.true_class,
            null_loss: cache.null_loss,
        }
    }
    fn loss(&self, input: &<bool as PackedElement<S>>::Array, class: usize) -> u64 {
        self.fc
            .iter()
            .enumerate()
            .map(|(c, w)| {
                let target_act = (c == class) as u32 * <() as WeightArray<S, W>>::MAX;
                let act = <()>::bma(&w, input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                (dist as u64).pow(2)
            })
            .sum()
    }
    fn loss_deltas(
        &self,
        input: &<bool as PackedElement<S>>::Array,
        class: usize,
    ) -> Vec<(Self::Index, W, i64)> {
        self.fc
            .iter()
            .enumerate()
            .map(|(c, w)| {
                let target_act = (c == class) as u32 * <() as WeightArray<S, W>>::MAX;
                let act = <()>::bma(&w, input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                let class_null_loss = (dist as u64).pow(2) as i64;
                <()>::loss_deltas(&w, input, |act| {
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
pub struct FC<S, W, O: Shape, H: Model<<bool as PackedElement<O>>::Array, C>, const C: usize>
where
    bool: PackedElement<O>,
    S: Shape,
    W: ConstructWeightArray<S>,
    <<W as ConstructWeightArray<S>>::WeightArray as Element<O>>::Array: Copy,
    <W as ConstructWeightArray<S>>::WeightArray: Element<O>,
{
    fc: <<W as ConstructWeightArray<S>>::WeightArray as Element<O>>::Array,
    tail: H,
}

impl<W, S, O, H, const C: usize> FC<S, W, O, H, C>
where
    S: Shape + PackedIndexMap<bool>,
    (): WeightArray<S, W>,
    bool: PackedElement<O> + PackedElement<S>,
    W: 'static + ConstructWeightArray<S> + Weight,
    H: Model<<bool as PackedElement<O>>::Array, C>,
    <bool as PackedElement<S>>::Array: Copy + PackedArray<Weight = bool, Shape = S>,
    O: Shape + PackedMap<<W as ConstructWeightArray<S>>::WeightArray, bool>,
    <<W as ConstructWeightArray<S>>::WeightArray as Element<O>>::Array: Copy,
    <W as ConstructWeightArray<S>>::WeightArray: Element<O> + Copy,
{
    fn acts(&self, input: &<bool as PackedElement<S>>::Array) -> <bool as PackedElement<O>>::Array {
        <O as PackedMap<<W as ConstructWeightArray<S>>::WeightArray, bool>>::map(&self.fc, |w| {
            <() as WeightArray<S, W>>::act(&w, input)
        })
    }
}

pub struct FCiter<
    I: Shape,
    O: Shape,
    H: Model<<bool as PackedElement<O>>::Array, C>,
    const C: usize,
> where
    bool: PackedElement<O>,
{
    cur_output_index: Option<O::Index>,
    input_index: I::IndexIter,
    output_index: O::IndexIter,
    head_iter: H::IndexIter,
}

impl<I: Shape, O: Shape, H: Model<<bool as PackedElement<O>>::Array, C>, const C: usize>
    FCiter<I, O, H, C>
where
    bool: PackedElement<O>,
{
    fn new() -> Self {
        let mut out_iter = O::indices();
        FCiter {
            cur_output_index: out_iter.next(),
            input_index: I::indices(),
            output_index: out_iter,
            head_iter: H::indices(),
        }
    }
}

impl<I: Shape, O: Shape, H: Model<<bool as PackedElement<O>>::Array, C>, const C: usize> Iterator
    for FCiter<I, O, H, C>
where
    bool: PackedElement<O>,
{
    type Item = LayerIndex<(O::Index, I::Index), H::Index>;
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
    <bool as PackedElement<I>>::Array: PackedArray<Weight = bool, Shape = I> + Copy,
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
    W: 'static + Weight + ConstructWeightArray<I>,
    I: Shape + PackedIndexMap<bool>,
    bool: PackedElement<O> + PackedElement<I>,
    <W as ConstructWeightArray<I>>::WeightArray: Copy,
    <bool as PackedElement<I>>::Array: PackedArray<Weight = bool, Shape = I> + Copy,
    (W, u32): Element<O>,
    (): WeightArray<I, W>,
{
    fn loss_delta(&self, input: bool) -> i64 {
        let hidden = <O as PackedMap<(W, u32), bool>>::map(&self.sums, |(w, sum)| {
            (sum + w.bma(input)) > <() as WeightArray<I, W>>::THRESHOLD
        });
        self.head.loss(&hidden, self.true_class) as i64 - self.null_loss
    }
}

impl<I, W, O, H, const C: usize> Model<<bool as PackedElement<I>>::Array, C> for FC<I, W, O, H, C>
where
    (): Element<O, Array = O>,
    (): WeightArray<I, W>,
    W: 'static + Weight + ConstructWeightArray<I>,
    H: Model<
        <bool as PackedElement<O>>::Array,
        C,
        Weight = W,
        InputIndex = O::Index,
        InputValue = bool,
    >,
    O: PackedMap<<W as ConstructWeightArray<I>>::WeightArray, bool>
        + Shape
        + Default
        + MutMap<(), <W as ConstructWeightArray<I>>::WeightArray>
        + ZipMap<<W as ConstructWeightArray<I>>::WeightArray, u32, (W, u32)>
        + Map<<W as ConstructWeightArray<I>>::WeightArray, u32>
        + PackedMap<(W, u32), bool>,
    Self: Copy,
    bool: PackedElement<I> + PackedElement<O>,
    I: Shape + PackedIndexMap<bool>,
    <W as ConstructWeightArray<I>>::WeightArray: Element<O> + Copy,
    <bool as PackedElement<I>>::Array: PackedArray<Weight = bool, Shape = I> + Copy,
    <<W as ConstructWeightArray<I>>::WeightArray as Element<O>>::Array:
        IndexGet<O::Index, Element = <W as ConstructWeightArray<I>>::WeightArray> + Default + Copy,
    Self::ChanCache: Cache<bool>,
    u32: Element<O>,
    (W, u32): Element<O>,
    <bool as PackedElement<O>>::Array: PackedArray<Shape = O, Weight = bool>,
{
    type Index = LayerIndex<(O::Index, <I as Shape>::Index), H::Index>;
    type Weight = W;
    type IndexIter = FCiter<I, O, H, C>;
    type Cache = FCcache<O, H, C>;
    type ChanCache = FCchanCache<I, W, O, H, C>;
    type InputIndex = <I as Shape>::Index;
    type InputValue = bool;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        FC {
            fc: <O as MutMap<(), <W as ConstructWeightArray<I>>::WeightArray>>::map(
                &O::default(),
                &mut |_| <() as WeightArray<I, W>>::rand(rng),
            ),
            tail: H::rand(rng),
        }
    }
    fn indices() -> Self::IndexIter {
        FCiter::new()
    }
    fn mutate_in_place(&mut self, index: Self::Index, weight: W) {
        match index {
            LayerIndex::Head((o, i)) => {
                <()>::mutate_in_place(&mut self.fc.index_get_mut(o), i, weight)
            }
            LayerIndex::Tail(i) => self.tail.mutate_in_place(i, weight),
        }
    }
    fn top_act(&self, input: &<bool as PackedElement<I>>::Array) -> usize {
        let hidden = self.acts(input);
        self.tail.top_act(&hidden)
    }
    fn cache(&self, input: &<bool as PackedElement<I>>::Array, class: usize) -> Self::Cache {
        FCcache {
            is_alive: true,
            sums: <O as Map<<W as ConstructWeightArray<I>>::WeightArray, u32>>::map(
                &self.fc,
                |weights| <()>::bma(&weights, input),
            ),
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
            sums:
                <O as ZipMap<<W as ConstructWeightArray<I>>::WeightArray, u32, (W, u32)>>::zip_map(
                    &self.fc,
                    &cache.sums,
                    |weights, sum| {
                        let w = <()>::get_weight(&weights, chan_index);
                        (w, sum - w.bma(cur_value))
                    },
                ),
            true_class: cache.true_class,
            null_loss: cache.null_loss,
            head: self.tail,
        }
    }
    fn loss(&self, input: &<bool as PackedElement<I>>::Array, class: usize) -> u64 {
        let hidden = self.acts(input);
        self.tail.loss(&hidden, class)
    }
    fn loss_deltas(
        &self,
        inputs: &<bool as PackedElement<I>>::Array,
        class: usize,
    ) -> Vec<(Self::Index, W, i64)> {
        let hidden = self.acts(inputs);
        // for a given output, if we flip it, what is the new loss?
        // for fc, we will only call it once, but for conv, we will call it many time for a given channel.
        let cache = self.tail.cache(&hidden, class);
        O::indices()
            .map(|o| {
                let input = hidden.get_weight(o);
                let weight_array = self.fc.index_get(o);

                let chan_cache = self.tail.subtract_input(&cache, o, input);
                let deltas = [chan_cache.loss_delta(false), chan_cache.loss_delta(true)];

                <()>::loss_deltas(&weight_array, &inputs, |act| {
                    deltas[(act > <() as WeightArray<I, W>>::THRESHOLD) as usize]
                })
                .iter()
                .map(|&(i, w, l)| (LayerIndex::Head((o, i)), w, l))
                .collect::<Vec<_>>()
            })
            .flatten()
            .chain(
                self.tail
                    .loss_deltas(&hidden, class)
                    .iter()
                    .map(|&(i, w, l)| (LayerIndex::Tail(i), w, l)),
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Cache, FcMSE, Model, FC};
    use crate::bits::{
        b128, b32, b8, t128, t32, t8, PackedArray, PackedElement, Weight, WeightArray,
    };
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;

    macro_rules! test_loss_deltas {
        ($name:ident, $input:ty, $weights:ty, $n_classes:expr, $n_iters:expr) => {
            #[test]
            fn $name() {
                let mut rng = Hc128Rng::seed_from_u64(0);
                (0..$n_iters).for_each(|_| {
                    let inputs: $input = rng.gen();
                    let weights = <$weights as Model<$input, $n_classes>>::rand(&mut rng);
                    for class in 0..$n_classes {
                        let mut true_losses = weights.loss_deltas_slow(&inputs, class);
                        true_losses.sort();
                        let mut losses = weights.loss_deltas(&inputs, class);
                        losses.sort();
                        assert_eq!(true_losses, losses);
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
                            let chan_cache = weights.subtract_input(&cache, i, <<bool as PackedElement<$input_shape>>::Array as PackedArray>::get_weight(&inputs, i));
                            for &sign in &[false, true] {
                                let loss_delta = chan_cache.loss_delta(sign);
                                let true_loss = weights.loss(&inputs.set_weight(i, sign), class);
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
}
