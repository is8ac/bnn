use crate::bits::{PackedArray, PackedElement, Weight, WeightArray};
use crate::shape::{Element, IndexGet, IndexMap, LongDefault, Map, Shape};
use rand::Rng;
use rayon::prelude::*;
use std::iter;
use std::marker::PhantomData;

//pub trait BaseCache
//where
//    Self::ChanCache: ChanCache,
//{
//    type ChanIndex;
//    type ChanValue;
//    type Input;
//    type ChanCache;
//    fn chan_cache(&self, chan_val: Self::ChanValue, chan_index: Self::ChanIndex) -> Self::ChanCache;
//}
//
//pub trait ChanCache {
//    type Mutation;
//    fn loss(&self, mutation: &Self::Mutation) -> u64;
//}

#[derive(Copy, Clone)]
pub enum LayerIndex<H: Copy, T: Copy> {
    Head(H),
    Tail(T),
}

pub trait Model<I>
where
    Self: Sized + Copy,
    Self::Weight: 'static + Weight,
    Self::IndexIter: Iterator<Item = Self::Index>,
    Self::Index: Copy,
{
    type Index;
    type Weight;
    type IndexIter;
    fn rand<R: Rng>(rng: &mut R) -> Self;
    fn indices() -> Self::IndexIter;
    fn mutate_in_place(&mut self, index: Self::Index, weight: Self::Weight);
    fn mutate(mut self, index: Self::Index, weight: Self::Weight) -> Self {
        self.mutate_in_place(index, weight);
        self
    }
    fn loss(&self, input: &I, class: usize) -> u64;
    /// losses for all mutations of all weights.
    fn losses(&self, input: &I, class: usize) -> Vec<(Self::Index, Self::Weight, u64)>;
    /// same as losses but a lot slower.
    fn losses_slow(&self, input: &I, class: usize) -> Vec<(Self::Index, Self::Weight, u64)> {
        let null_loss = self.loss(input, class);
        <Self::Weight as Weight>::states()
            .map(|w| {
                Self::indices()
                    .map(|i| (i, w, self.mutate(i, w).loss(input, class)))
                    .collect::<Vec<_>>()
            })
            .flatten()
            .filter(|(_, _, l)| *l != null_loss)
            .collect()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FcMSE<I, const C: usize> {
    pub fc: [I; C],
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

impl<A, const C: usize> Model<<bool as PackedElement<A::Shape>>::Array> for FcMSE<A, C>
where
    [A; C]: LongDefault,
    A: WeightArray + Sized,
    bool: PackedElement<A::Shape>,
    [A; C]: LongDefault,
    <A::Shape as Shape>::Index: Copy,
    <bool as PackedElement<A::Shape>>::Array: Copy + PackedArray<Weight = bool, Shape = A::Shape>,
{
    type Index = (usize, <A::Shape as Shape>::Index);
    type Weight = A::Weight;
    type IndexIter = FcMSEindexIter<A::Shape, C>;
    fn rand<R: Rng>(rng: &mut R) -> Self {
        let mut weights = <[A; C]>::long_default();
        for c in 0..C {
            weights[c] = <A>::rand(rng);
        }
        FcMSE { fc: weights }
    }
    fn indices() -> FcMSEindexIter<A::Shape, C> {
        FcMSEindexIter {
            class_index: 0,
            input_iter: A::Shape::indices(),
        }
    }
    fn mutate_in_place(
        &mut self,
        (head, tail): (usize, <A::Shape as Shape>::Index),
        weight: A::Weight,
    ) {
        self.fc[head].mutate_in_place(tail, weight);
    }
    fn loss(&self, input: &<bool as PackedElement<A::Shape>>::Array, class: usize) -> u64 {
        self.fc
            .iter()
            .enumerate()
            .map(|(c, w)| {
                let target_act = (c == class) as u32 * A::MAX;
                let act = w.bma(input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                (dist as u64).pow(2)
            })
            .sum()
    }
    fn losses(
        &self,
        input: &<bool as PackedElement<A::Shape>>::Array,
        class: usize,
    ) -> Vec<(Self::Index, A::Weight, u64)> {
        let sum_loss = self.loss(input, class);

        self.fc
            .iter()
            .enumerate()
            .map(|(c, w)| {
                let target_act = (c == class) as u32 * A::MAX;
                let act = w.bma(input);
                let dist = act.saturating_sub(target_act) | target_act.saturating_sub(act);
                let part_loss = sum_loss - (dist as u64).pow(2);
                w.losses(input, |act| {
                    part_loss
                        + ((act.saturating_sub(target_act) | target_act.saturating_sub(act)) as u64)
                            .pow(2)
                })
                .iter()
                .map(|(i, w, l)| ((c, *i), *w, *l))
                .collect::<Vec<_>>()
            })
            .flatten()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{FcMSE, Model};
    use crate::bits::{PackedElement, Weight, WeightArray};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;

    type InputShape = [[(); 32]; 2];
    type InputType = <bool as PackedElement<InputShape>>::Array;
    type BitWeightArrayType = <bool as PackedElement<InputShape>>::Array;
    type TritWeightArrayType = (<Option<bool> as PackedElement<InputShape>>::Array, u32);

    #[test]
    fn rand_fcmse_bit_losses() {
        let mut rng = Hc128Rng::seed_from_u64(0);
        (0..100).for_each(|_| {
            let inputs: InputType = rng.gen();
            let weights = <FcMSE<BitWeightArrayType, 7> as Model<InputType>>::rand(&mut rng);
            for class in 0..7 {
                let mut true_losses = weights.losses_slow(&inputs, class);
                true_losses.sort();
                let mut losses = weights.losses(&inputs, class);
                losses.sort();
                assert_eq!(true_losses, losses);
            }
        })
    }
    #[test]
    fn rand_fcmse_trit_losses() {
        let mut rng = Hc128Rng::seed_from_u64(0);
        (0..100).for_each(|_| {
            let inputs: InputType = rng.gen();
            let weights = <FcMSE<TritWeightArrayType, 7> as Model<InputType>>::rand(&mut rng);
            for class in 0..7 {
                let mut true_losses = weights.losses_slow(&inputs, class);
                true_losses.sort();
                let mut losses = weights.losses(&inputs, class);
                losses.sort();
                assert_eq!(true_losses, losses);
            }
        })
    }
}
