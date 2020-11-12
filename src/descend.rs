use crate::layers::Model;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::time::Instant;

pub trait Descend<I, const C: usize>
where
    Self: Model<I, C>,
{
    fn avg_acc(&self, examples: &[(I, usize)]) -> f64;
    fn updates(
        &self,
        examples: &[(I, usize)],
        example_truncation: usize,
        n_updates: usize,
    ) -> Vec<(Self::Index, Self::Weight)>;
    fn train(
        self,
        examples: &Vec<(I, usize)>,
        example_truncation: usize,
        n_updates: usize,
        minibatch_size: usize,
    ) -> Self;
}

impl<T: Model<I, C> + Sync, I: Sync, const C: usize> Descend<I, C> for T
where
    Self::Index: Send + Eq + Hash + Ord + std::fmt::Debug,
    Self::Weight: Send + Eq + Hash + Ord + std::fmt::Debug,
{
    fn avg_acc(&self, examples: &[(I, usize)]) -> f64 {
        let n_correct: u64 = examples
            .par_iter()
            .map(|(image, class)| self.is_correct(image, *class) as u64)
            .sum();
        n_correct as f64 / examples.len() as f64
    }
    fn updates(
        &self,
        examples: &[(I, usize)],
        example_truncation: usize,
        n_updates: usize,
    ) -> Vec<(Self::Index, Self::Weight)> {
        //let start = Instant::now();
        let mut_map = examples
            .par_iter()
            .fold(
                || HashMap::<(Self::Index, Self::Weight), i64>::new(),
                |mut acc, (image, class)| {
                    let mut deltas = self.loss_deltas(image, *class);
                    deltas.sort_by(|a, b| b.2.abs().cmp(&a.2.abs()));
                    let pos: Vec<_> = deltas
                        .iter()
                        .filter(|&(_, _, l)| *l < 0)
                        .take(example_truncation)
                        .cloned()
                        .collect();
                    let neg: Vec<_> = deltas
                        .iter()
                        .filter(|&(_, _, l)| *l > 0)
                        .take(example_truncation)
                        .cloned()
                        .collect();
                    pos.iter()
                        .chain(neg.iter())
                        .fold(acc, |mut acc, &(i, m, l)| {
                            *acc.entry((i, m)).or_insert(0) += l;
                            acc
                        })
                },
            )
            .reduce_with(|mut a, b| {
                b.iter().for_each(|(k, v)| {
                    *a.entry(*k).or_insert(0) += v;
                });
                a
            })
            .unwrap();
        //println!("deltas time: {:?}", start.elapsed());
        let mut top_mutations = mut_map
            .iter()
            .filter(|(_, d)| **d < 0)
            .map(|(&(i, m), &d)| (d, i, m))
            .collect::<Vec<(i64, Self::Index, Self::Weight)>>();
        top_mutations.par_sort();
        top_mutations
            .iter()
            .map(|&(_, i, w)| (i, w))
            .take(n_updates)
            .collect()
    }
    fn train(
        self,
        examples: &Vec<(I, usize)>,
        example_truncation: usize,
        n_updates: usize,
        minibatch_size: usize,
    ) -> Self {
        examples
            .chunks_exact(minibatch_size)
            .fold(self, |weights, examples| {
                let mut updates = weights.updates(&examples, example_truncation, n_updates);
                //dbg!(&updates);
                let weights = updates
                    .drain(0..)
                    .fold(weights, |weights, (i, w)| weights.mutate(i, w));
                weights
            })
    }
}
