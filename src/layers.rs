use crate::bits::{BitArray, BitMapPack, MaskedDistance, TritArray};
use crate::shape::{Element, IndexGet, Map, Shape};
use rayon::prelude::*;
use std::iter;

pub fn minibatch_sizes(min: usize, size: usize, scale: (usize, usize)) -> Vec<usize> {
    assert!(scale.0 < scale.1);
    let next_size = (size * scale.0) / scale.1;
    if next_size < min {
        vec![size]
    } else {
        let mut sizes = minibatch_sizes(min, next_size, scale);
        sizes.push(size);
        sizes
    }
}

pub trait FcObjective<T> {
    fn loss(&self, input: &T, class: usize) -> f32;
}

pub trait Trainable<T> {
    type Params;
    fn descend(self, inputs: &[T], classes: &[usize], params: &Self::Params) -> Self;
}

pub struct FcAuxTrainParams {
    pub min: usize,
    pub scale: (usize, usize),
    pub k: usize,
}

impl<I: BitArray, const C: usize> Trainable<I> for [I::TritArrayType; C]
where
    (): DescendFCaux<I, C>,
{
    type Params = FcAuxTrainParams;
    fn descend(self, inputs: &[I], classes: &[usize], params: &Self::Params) -> Self {
        <() as DescendFCaux<I, C>>::minibatch_train(
            self,
            inputs,
            classes,
            params.min,
            params.scale,
            params.k,
        )
    }
}

impl<T: BitArray, const C: usize> FcObjective<T> for [T::TritArrayType; C]
where
    [f32; C]: SoftMaxLoss,
    (): DescendFCaux<T, C>,
    [(); C]: Map<<T as BitArray>::TritArrayType, f32>,
{
    fn loss(&self, input: &T, class: usize) -> f32 {
        <[(); C] as Map<<T as BitArray>::TritArrayType, f32>>::map(self, |trits| {
            <() as DescendFCaux<T, C>>::chan_act(trits, input)
        })
        .softmax_loss(class)
    }
}

pub trait SoftMaxLoss {
    fn softmax_loss(&self, true_class: usize) -> f32;
}

impl<const C: usize> SoftMaxLoss for [f32; C]
where
    [f32; C]: Default,
{
    fn softmax_loss(&self, true_class: usize) -> f32 {
        let mut exp = <[f32; C]>::default();
        let mut sum_exp = 0f32;
        for c in 0..C {
            exp[c] = self[c].exp();
            sum_exp += exp[c];
        }
        let mut sum_loss = 0f32;
        for c in 0..C {
            let scaled = exp[c] / sum_exp;
            sum_loss += (scaled - (c == true_class) as u8 as f32).powi(2);
        }
        sum_loss
    }
}

pub trait DescendFCaux<I: BitArray, const C: usize> {
    fn chan_act(trits: &I::TritArrayType, input: &I) -> f32;
    fn observe_avg_loss_cached(
        chan_weights: &I::TritArrayType,
        inputs: &[I],
        acts_cache: &Vec<[f32; C]>,
        classes: &[usize],
        c: usize,
    ) -> f64;
    fn full_acts(weights: &[I::TritArrayType; C], inputs: &[I]) -> Vec<[f32; C]>;
    fn descend_chan(
        chan_weights: I::TritArrayType,
        inputs: &[I],
        acts_cache: &Vec<[f32; C]>,
        classes: &[usize],
        k: usize,
        sign: bool,
        c: usize,
    ) -> I::TritArrayType;
    fn descend_aux_weights(
        weights: [I::TritArrayType; C],
        inputs: &[I],
        classes: &[usize],
        chunks: &[(usize, usize)],
        k: usize,
        sign: bool,
    ) -> [I::TritArrayType; C];
    fn minibatch_train(
        weights: [I::TritArrayType; C],
        inputs: &[I],
        classes: &[usize],
        min: usize,
        scale: (usize, usize),
        k: usize,
    ) -> [I::TritArrayType; C];
}

impl<I, const C: usize> DescendFCaux<I, C> for ()
where
    I: Sync + BitArray + Send,
    [f32; C]: SoftMaxLoss,
    I::TritArrayType: MaskedDistance + TritArray + Sync + Copy,
    <<I as BitArray>::BitShape as Shape>::IndexIter: Iterator<Item = <I::BitShape as Shape>::Index>,
    <I::BitShape as Shape>::Index: Send + Sync + Copy,
    [(); C]: Map<I::TritArrayType, f32>,
{
    fn chan_act(trits: &I::TritArrayType, input: &I) -> f32 {
        (trits.masked_distance(input) as f32 - <I as BitArray>::BitShape::N as f32 / 4f32)
            / (<I as BitArray>::BitShape::N as f32 / 16f32)
    }
    fn observe_avg_loss_cached(
        trits: &I::TritArrayType,
        inputs: &[I],
        acts_cache: &Vec<[f32; C]>,
        classes: &[usize],
        c: usize,
    ) -> f64 {
        inputs
            .iter()
            .zip(acts_cache.iter().cloned().zip(classes.iter()))
            .map(|(input, (mut acts, &class))| {
                acts[c] = <() as DescendFCaux<I, C>>::chan_act(trits, input);
                acts.softmax_loss(class) as f64
                //0f64
            })
            .sum()
    }
    fn full_acts(weights: &[I::TritArrayType; C], inputs: &[I]) -> Vec<[f32; C]> {
        inputs
            .par_iter()
            .map(|input| {
                <[(); C] as Map<<I as BitArray>::TritArrayType, f32>>::map(&weights, |trits| {
                    <() as DescendFCaux<I, C>>::chan_act(trits, input)
                })
            })
            .collect()
    }
    fn descend_chan(
        trits: I::TritArrayType,
        inputs: &[I],
        acts_cache: &Vec<[f32; C]>,
        classes: &[usize],
        k: usize,
        sign: bool,
        c: usize,
    ) -> I::TritArrayType {
        let indices: Vec<_> = <I as BitArray>::BitShape::indices()
            .map(|i| Some(i))
            .chain(iter::once(None))
            .collect();
        let mut mutations: Vec<_> = indices
            .par_iter()
            .map(|index| {
                let mutant_trits = if let Some(i) = index {
                    trits.set_trit(
                        if trits.get_trit(&i).is_some() {
                            None
                        } else {
                            Some(sign)
                        },
                        &i,
                    )
                } else {
                    trits
                };
                let sum_loss =
                    <()>::observe_avg_loss_cached(&mutant_trits, inputs, acts_cache, classes, c);
                (sum_loss, index)
            })
            .collect();
        mutations.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
        mutations
            .iter()
            .take(k)
            .take_while(|(_, i)| i.is_some())
            .map(|(_, i)| i.unwrap())
            .fold(trits, |t, i| {
                t.set_trit(
                    if t.get_trit(&i).is_some() {
                        None
                    } else {
                        Some(sign)
                    },
                    &i,
                )
            })
    }
    fn descend_aux_weights(
        weights: [I::TritArrayType; C],
        inputs: &[I],
        classes: &[usize],
        chunks: &[(usize, usize)],
        k: usize,
        sign: bool,
    ) -> [I::TritArrayType; C] {
        assert_eq!(inputs.len(), classes.len());
        assert_eq!(C, chunks.len());
        (0..C)
            .zip(chunks.iter())
            .fold(weights, |mut weights, (c, &(start, end))| {
                let acts = <() as DescendFCaux<I, C>>::full_acts(&weights, &inputs[start..end]);
                weights[c] = <() as DescendFCaux<I, C>>::descend_chan(
                    weights[c],
                    &inputs[start..end],
                    &acts,
                    &classes[start..end],
                    k,
                    sign,
                    c,
                );
                weights
            })
    }
    fn minibatch_train(
        weights: [I::TritArrayType; C],
        inputs: &[I],
        classes: &[usize],
        min: usize,
        scale: (usize, usize),
        k: usize,
    ) -> [I::TritArrayType; C] {
        assert_eq!(inputs.len(), classes.len());
        let sizes = minibatch_sizes(min, inputs.len(), scale);
        sizes
            .iter()
            .map(|m| (0..inputs.len() / m).map(move |i| (i * m, (i + 1) * m)))
            .flatten()
            .collect::<Vec<(usize, usize)>>()
            .chunks_exact(C)
            .enumerate()
            .fold(weights, |weights, (i, chunks)| {
                <()>::descend_aux_weights(weights, &inputs, &classes, chunks, k, (i % 2) == 0)
            })
    }
}

pub trait FClayer<I: BitArray, O: BitArray, A: Trainable<O> + FcObjective<O>>
where
    I::TritArrayType: Element<O::BitShape>,
{
    fn chan_act(trits: &I::TritArrayType, input: &I) -> bool;
    fn full_hidden_acts(
        weights: &<I::TritArrayType as Element<O::BitShape>>::Array,
        input: &I,
    ) -> O;
    fn update_hidden_acts(
        trits: &I::TritArrayType,
        input: &I,
        target: O,
        index: &<O::BitShape as Shape>::Index,
    ) -> O;
    fn descend_chan(
        trits: I::TritArrayType,
        aux: &A,
        inputs: &[I],
        hidden_cache: &[O],
        classes: &[usize],
        index: &<O::BitShape as Shape>::Index,
        k: usize,
        sign: bool,
    ) -> I::TritArrayType;
    fn minibatch_train(
        weights: <I::TritArrayType as Element<O::BitShape>>::Array,
        aux: &A,
        inputs: &[I],
        classes: &[usize],
        chunks: &[(usize, usize)],
        k: usize,
        sign: bool,
    ) -> <I::TritArrayType as Element<O::BitShape>>::Array;
    fn descend_weights_minibatched(
        weights: <I::TritArrayType as Element<O::BitShape>>::Array,
        aux: A,
        aux_params: &A::Params,
        inputs: &[I],
        classes: &[usize],
        k: usize,
        min: usize,
        scale: (usize, usize),
    ) -> (<I::TritArrayType as Element<O::BitShape>>::Array, A);
}

impl<I, O, A> FClayer<I, O, A> for ()
where
    I: BitArray + Sync,
    O: BitArray + Sync + Copy + Send,
    A: FcObjective<O> + Trainable<O> + Sync + Send,
    I::TritArrayType: Copy + MaskedDistance + Element<O::BitShape> + TritArray + Sync,
    <O::BitShape as Shape>::Index: Sync,
    <I::BitShape as Shape>::Index: Send + Sync + Copy,
    <<I as BitArray>::BitShape as Shape>::IndexIter: Iterator<Item = <I::BitShape as Shape>::Index>,
    <<O as BitArray>::BitShape as Shape>::IndexIter: Iterator<Item = <O::BitShape as Shape>::Index>,
    O: BitMapPack<I::TritArrayType>,
    <I::TritArrayType as Element<O::BitShape>>::Array:
        IndexGet<<O::BitShape as Shape>::Index, Element = I::TritArrayType> + Sync,
{
    fn chan_act(trits: &I::TritArrayType, input: &I) -> bool {
        trits.masked_distance(input) > (I::BitShape::N as u32 / 4)
    }
    fn full_hidden_acts(
        weights: &<I::TritArrayType as Element<O::BitShape>>::Array,
        input: &I,
    ) -> O {
        O::bit_map_pack(&weights, |trits: &<I as BitArray>::TritArrayType| {
            <() as FClayer<I, O, A>>::chan_act(trits, input)
        })
    }
    fn update_hidden_acts(
        trits: &I::TritArrayType,
        input: &I,
        target: O,
        index: &<O::BitShape as Shape>::Index,
    ) -> O {
        target.set_bit(<() as FClayer<I, O, A>>::chan_act(trits, input), index)
    }
    fn descend_chan(
        trits: I::TritArrayType,
        aux: &A,
        inputs: &[I],
        hidden_cache: &[O],
        classes: &[usize],
        channel: &<O::BitShape as Shape>::Index,
        k: usize,
        sign: bool,
    ) -> I::TritArrayType {
        let indices: Vec<_> = <I as BitArray>::BitShape::indices()
            .map(|i| Some(i))
            .chain(iter::once(None))
            .collect();
        let mut mutations: Vec<(f64, Option<<I::BitShape as Shape>::Index>)> = indices
            .par_iter()
            .map(|index| {
                let mutant_trits = if let Some(i) = index {
                    trits.set_trit(
                        if trits.get_trit(&i).is_some() {
                            None
                        } else {
                            Some(sign)
                        },
                        &i,
                    )
                } else {
                    trits
                };
                let sum_loss = inputs
                    .iter()
                    .zip(hidden_cache.iter())
                    .zip(classes.iter().cloned())
                    .map(|((input, act), class)| {
                        let new_act = <() as FClayer<I, O, A>>::update_hidden_acts(
                            &mutant_trits,
                            input,
                            *act,
                            channel,
                        );
                        aux.loss(&new_act, class) as f64
                    })
                    .sum();
                (sum_loss, *index)
            })
            .collect();
        mutations.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
        mutations
            .iter()
            .take(k)
            .take_while(|(_, i)| i.is_some())
            .map(|(l, i)| (l, i.unwrap()))
            .fold(trits, |t, (_, i)| {
                t.set_trit(
                    if t.get_trit(&i).is_some() {
                        None
                    } else {
                        Some(sign)
                    },
                    &i,
                )
            })
    }
    fn minibatch_train(
        weights: <I::TritArrayType as Element<O::BitShape>>::Array,
        aux: &A,
        inputs: &[I],
        classes: &[usize],
        chunks: &[(usize, usize)],
        k: usize,
        sign: bool,
    ) -> <I::TritArrayType as Element<O::BitShape>>::Array {
        assert_eq!(chunks.len(), <O::BitShape as Shape>::N);
        chunks.iter().zip(O::BitShape::indices()).fold(
            weights,
            |weights, (&(chunk_start, chunk_end), channel)| {
                let hidden_cache: Vec<O> = inputs[chunk_start..chunk_end]
                    .par_iter()
                    .map(|input| <() as FClayer<I, O, A>>::full_hidden_acts(&weights, input))
                    .collect();
                let updated_chan_trits = <() as FClayer<I, O, A>>::descend_chan(
                    *weights.index_get(&channel),
                    &aux,
                    &inputs[chunk_start..chunk_end],
                    &hidden_cache,
                    &classes[chunk_start..chunk_end],
                    &channel,
                    k,
                    sign,
                );
                weights.index_set(&channel, updated_chan_trits)
            },
        )
    }
    fn descend_weights_minibatched(
        weights: <I::TritArrayType as Element<O::BitShape>>::Array,
        aux: A,
        aux_params: &A::Params,
        inputs: &[I],
        classes: &[usize],
        k: usize,
        min: usize,
        scale: (usize, usize),
    ) -> (<I::TritArrayType as Element<O::BitShape>>::Array, A) {
        assert_eq!(inputs.len(), classes.len());
        let sizes = minibatch_sizes(min, inputs.len(), scale);
        dbg!(sizes.len());
        sizes
            .iter()
            .map(|m| (0..inputs.len() / m).map(move |i| (i * m, (i + 1) * m)))
            .flatten()
            .collect::<Vec<(usize, usize)>>()
            .chunks_exact(O::BitShape::N)
            .enumerate()
            .fold((weights, aux), |(weights, aux), (index, chunks)| {
                dbg!(chunks[0].1 - chunks[0].0);
                let hidden_examples: Vec<O> = inputs
                    .par_iter()
                    .map(|input| <() as FClayer<I, O, A>>::full_hidden_acts(&weights, input))
                    .collect();
                let aux = aux.descend(&hidden_examples, classes, aux_params);
                (
                    <() as FClayer<I, O, A>>::minibatch_train(
                        weights,
                        &aux,
                        inputs,
                        classes,
                        chunks,
                        k,
                        (index % 2) == 0,
                    ),
                    aux,
                )
            })
    }
}
