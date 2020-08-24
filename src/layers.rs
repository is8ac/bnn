use crate::bits::{BitArray, MaskedDistance, TritArray};
use crate::shape::{Map, Shape};
use rayon::prelude::*;
use std::iter;

fn minibatchs(len: usize, n: usize, m: usize) -> Vec<(usize, usize)> {
    assert!(m <= len);
    (0..n)
        .map(|i| {
            let offset = (i * (len - m)) / (n - 1);
            (offset, offset + m)
        })
        .collect()
}

pub fn epoc_set(len: usize, batch_sizes: &[usize]) -> Vec<(usize, usize)> {
    batch_sizes
        .iter()
        .map(|m| (0..len / m).map(move |i| (i * m, (i + 1) * m)))
        .flatten()
        .collect()
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
                acts[c] = <()>::chan_act(trits, input);
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
                    <()>::chan_act(trits, input)
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
        let (n, updated_trits) = mutations
            .iter()
            .take(k)
            .take_while(|(_, i)| i.is_some())
            .map(|(l, i)| (l, i.unwrap()))
            .fold((0, trits), |(n, t), (loss, i)| {
                (
                    n + 1,
                    t.set_trit(
                        if t.get_trit(&i).is_some() {
                            None
                        } else {
                            Some(sign)
                        },
                        &i,
                    ),
                )
            });
        updated_trits
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
                let acts = <()>::full_acts(&weights, &inputs[start..end]);
                weights[c] = <()>::descend_chan(
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
        dbg!(sizes.len());
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
