use crate::bits::GetBit;
use crate::layers::{IndexDepth, Model};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

pub trait Descend<I, const C: usize>
where
    Self: Model<I, C>,
{
    fn avg_acc(&self, examples: &[(I, usize)]) -> f64;
    fn updates(
        &self,
        examples: &[(I, usize)],
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
    ) -> Vec<(Self::Index, Self::Weight)>;
    fn evaluate_update_combinations(
        &self,
        examples: &[(I, usize)],
        candidate_mutations: &Vec<(Self::Index, Self::Weight)>,
    ) -> Vec<(Self::Index, Self::Weight)>;
    fn train(
        self,
        examples: &Vec<(I, usize)>,
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
        minibatch_size: usize,
    ) -> (Self, usize);
    fn recursively_train(
        self,
        examples: &Vec<(I, usize)>,
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
        minibatch_size: usize,
        min: usize,
        scale: (usize, usize),
    ) -> (Self, usize, usize);
    fn train_n_epochs(
        self,
        examples: &Vec<(I, usize)>,
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
        n_epochs: usize,
        starting_minibatch_size: usize,
        scale: (usize, usize),
    ) -> (Self, usize, usize);
}

impl<T: Model<I, C> + Sync, I: Sync, const C: usize> Descend<I, C> for T
where
    Self::Index: Send + Eq + Hash + Ord + std::fmt::Debug,
    Self::Weight: Send + Eq + Hash + Ord + std::fmt::Debug,
    <T as Model<I, C>>::Index: Sync + IndexDepth,
    <T as Model<I, C>>::Weight: Sync,
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
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
    ) -> Vec<(Self::Index, Self::Weight)> {
        let mut_map = examples
            .par_iter()
            .fold(
                || HashMap::<(Self::Index, Self::Weight), i64>::new(),
                |acc, (image, class)| {
                    let mut deltas = self.loss_deltas(image, threshold, *class);
                    //dbg!(deltas.len());
                    deltas.sort_by(|a, b| b.2.abs().cmp(&a.2.abs()));
                    deltas
                        .iter()
                        .filter(|&(_, _, l)| *l < 0)
                        .take(example_truncation)
                        .chain(
                            deltas
                                .iter()
                                .filter(|&(_, _, l)| *l > 0)
                                .take(example_truncation),
                        )
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
        let mut top_mutations = mut_map
            .iter()
            .filter(|(_, d)| **d < 0)
            .map(|(&(i, m), &d)| (d, i, m))
            .collect::<Vec<(i64, Self::Index, Self::Weight)>>();

        //(0..n_updates)
        //    .filter_map(|_| top_mutations.choose_weighted(rng, |&(l, _, _)| (l.abs() as u64).pow(3)).ok().map(|&(_, i, w)| (i, w)))
        //    .collect()

        top_mutations.par_sort();
        top_mutations
            .iter()
            .map(|&(_, i, w)| (i, w))
            .take(n_updates)
            .collect()
    }
    /// This runs in time exponential with candidate_mutations.len()
    fn evaluate_update_combinations(
        &self,
        examples: &[(I, usize)],
        candidate_mutations: &Vec<(Self::Index, Self::Weight)>,
    ) -> Vec<(Self::Index, Self::Weight)> {
        let set_size = 2usize.pow(candidate_mutations.len() as u32);
        // length will be 2^mutations.len()
        let losses: Vec<i64> = examples
            .par_iter()
            .fold(
                || (0..set_size).map(|_| 0i64).collect::<Vec<i64>>(),
                |acc, (image, class)| {
                    let null_loss = self.loss(image, *class) as i64;
                    acc.iter()
                        .enumerate()
                        .map(|(set_index, sum)| {
                            let new_model = candidate_mutations
                                .iter()
                                .enumerate()
                                .filter(|&(mask_index, _)| set_index.bit(mask_index))
                                .fold(*self, |model, (_, &(index, weight))| {
                                    model.mutate(index, weight)
                                });
                            sum + (new_model.loss(image, *class) as i64 - null_loss)
                        })
                        .collect()
                },
            )
            .reduce_with(|a, b| a.iter().zip(b.iter()).map(|(a, b)| a + b).collect())
            .unwrap();
        //dbg!(&losses);
        let best_set_index: usize = losses.iter().enumerate().min_by_key(|(_, l)| *l).unwrap().0;
        //dbg!(best_set_index);
        candidate_mutations
            .iter()
            .enumerate()
            .filter(|&(mask_index, _)| best_set_index.bit(mask_index))
            .map(|(_, &m)| m)
            .collect()
    }
    fn train(
        self,
        examples: &Vec<(I, usize)>,
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
        minibatch_size: usize,
    ) -> (T, usize) {
        //println!("{} {} {} {}", threshold, example_truncation, n_updates, minibatch_size);
        examples
            .chunks_exact(minibatch_size)
            .fold((self, 0usize), |(weights, n), examples| {
                let candidates =
                    weights.updates(&examples, threshold, example_truncation, n_updates);
                let best_set = weights.evaluate_update_combinations(&examples, &candidates);
                //dbg!(best_set.len());
                //best_set.iter().for_each(|(i, _)| {
                //    print!("{}, ", i.depth());
                //});
                //print!("\n",);
                let weights = best_set
                    .iter()
                    .fold(weights, |weights, &(i, w)| weights.mutate(i, w));
                (weights, n + (best_set.len() > 0) as usize)
            })
    }

    fn recursively_train(
        self,
        examples: &Vec<(I, usize)>,
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
        minibatch_size: usize,
        min: usize,
        scale: (usize, usize),
    ) -> (Self, usize, usize) {
        let (model, n_epochs, s) = if minibatch_size >= min {
            self.recursively_train(
                examples,
                threshold,
                example_truncation,
                n_updates,
                (minibatch_size * scale.0) / scale.1,
                min,
                scale,
            )
        } else {
            (self, 0, 0)
        };
        //dbg!(minibatch_size);
        let (model, n) = model.train(
            &examples,
            threshold,
            example_truncation,
            n_updates,
            minibatch_size,
        );
        (model, n_epochs + 1, s + n)
    }
    fn train_n_epochs(
        self,
        examples: &Vec<(I, usize)>,
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
        n_epochs: usize,
        starting_minibatch_size: usize,
        scale: (usize, usize),
    ) -> (Self, usize, usize) {
        (0..n_epochs).fold(
            (self, starting_minibatch_size, 0),
            |(model, minibatch_size, updates_sum), _| {
                let (model, updates) = model.train(
                    &examples,
                    threshold,
                    example_truncation,
                    n_updates,
                    minibatch_size,
                );
                (
                    model,
                    (minibatch_size * scale.0) / scale.1,
                    updates + updates_sum,
                )
            },
        )
    }
}
