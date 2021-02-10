use crate::bits::GetBit;
use crate::layers::{IndexDepth, LayerIndex, Model, SegmentedAvgPoolIndex};
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::time::Instant;

pub trait Collide {
    type InputIndex;
    fn collide(&self, rhs: &Self) -> bool;
    fn input_index(&self) -> Option<Self::InputIndex>;
}

impl<I: Eq + Copy> Collide for (usize, I) {
    type InputIndex = I;
    fn collide(&self, rhs: &Self) -> bool {
        (self.0 == rhs.0) | (self.1 == rhs.1)
    }
    fn input_index(&self) -> Option<Self::InputIndex> {
        Some(self.1)
    }
}

impl<I: Copy, const X: usize, const Y: usize> Collide
    for SegmentedAvgPoolIndex<(usize, (u8, (u8, I))), X, Y>
where
    (usize, (u8, (u8, I))): Collide,
{
    type InputIndex = I;
    fn collide(&self, rhs: &Self) -> bool {
        self.index.collide(&rhs.index)
    }
    fn input_index(&self) -> Option<Self::InputIndex> {
        Some(((self.index.1).1).1)
    }
}

impl<O: Copy + Eq, I: Copy + Eq, T: Copy + Collide<InputIndex = O>> Collide
    for LayerIndex<(O, I), T>
{
    type InputIndex = I;
    fn collide(&self, rhs: &Self) -> bool {
        match self {
            LayerIndex::Head(x) => match rhs {
                LayerIndex::Head(y) => (x.0 == y.0) | (x.1 == y.1),
                LayerIndex::Tail(y) => y.input_index().map(|i| i == x.0).unwrap_or(false),
            },
            LayerIndex::Tail(x) => match rhs {
                LayerIndex::Head(y) => x.input_index().map(|i| i == y.0).unwrap_or(false),
                LayerIndex::Tail(y) => x.collide(y),
            },
        }
    }
    fn input_index(&self) -> Option<Self::InputIndex> {
        if let LayerIndex::Head(h) = self {
            Some(h.1)
        } else {
            None
        }
    }
}

pub trait Descend<I, const C: usize>
where
    Self: Model<I, C>,
{
    fn avg_acc(&self, examples: &[(I, usize)]) -> f64;
    fn apply_top_nocollide(
        self,
        loss_deltas: Vec<(Self::Index, Self::Weight, i64)>,
        n: usize,
    ) -> (Self, Vec<(Self::Index, Self::Weight, i64)>, usize);
    fn updates(
        &self,
        examples: &[(I, usize)],
        threshold: u64,
        example_truncation: usize,
        n_updates: usize,
    ) -> Vec<(Self::Index, Self::Weight)>;
    fn updates_simple(
        &self,
        examples: &[(I, usize)],
        threshold: u64,
        example_truncation: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)>;
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
    Self::Index: Sync + Collide + IndexDepth + Send + Eq + Hash + Ord + std::fmt::Debug,
    Self::Weight: Sync + Send + Eq + Hash + Ord + std::fmt::Debug,
{
    fn avg_acc(&self, examples: &[(I, usize)]) -> f64 {
        let n_correct: u64 = examples
            .par_iter()
            .map(|(image, class)| self.is_correct(image, *class) as u64)
            .sum();
        n_correct as f64 / examples.len() as f64
    }
    fn apply_top_nocollide(
        self,
        loss_deltas: Vec<(Self::Index, Self::Weight, i64)>,
        n: usize,
    ) -> (Self, Vec<(Self::Index, Self::Weight, i64)>, usize) {
        loss_deltas
            .iter()
            .min_by_key(|&(_, _, l)| l)
            .map(|&(index, weight, l)| {
                self.mutate(index, weight).apply_top_nocollide(
                    loss_deltas
                        .iter()
                        .filter(|(i, w, l)| !index.collide(i))
                        .cloned()
                        .collect(),
                    n + 1,
                )
            })
            .unwrap_or_else(|| (self, loss_deltas, n))
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

        top_mutations.par_sort();
        top_mutations
            .iter()
            .map(|&(_, i, w)| (i, w))
            .take(n_updates)
            .collect()
    }
    fn updates_simple(
        &self,
        examples: &[(I, usize)],
        threshold: u64,
        example_truncation: usize,
    ) -> Vec<(Self::Index, Self::Weight, i64)> {
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
        mut_map
            .iter()
            .filter(|(_, d)| **d < 0)
            .map(|(&(i, m), &d)| (i, m, d))
            .collect::<Vec<(Self::Index, Self::Weight, i64)>>()

        /*
        examples
            .par_iter()
            .fold(
                || HashMap::<(Self::Index, Self::Weight), i64>::new(),
                |acc, (image, class)| self.loss_deltas(image, 0, *class).iter().map(|&(i, w, l)| ((i, w), l)).collect(),
            )
            .reduce_with(|mut a, b| {
                b.iter().for_each(|(k, v)| {
                    *a.entry(*k).or_insert(0) += v;
                });
                a
            })
            .unwrap()
            .iter()
            .filter(|(_, &l)| l < 0)
            .map(|(&(i, w), &l)| (i, w, l))
            .collect()
            */
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
        examples.chunks_exact(minibatch_size).enumerate().fold(
            (self, 0usize),
            |(weights, n), (i, examples)| {
                let grads_start = Instant::now();
                let updates = weights.updates_simple(&examples, threshold, example_truncation);
                let (weights, empty_vec, n_updates) = weights.apply_top_nocollide(updates, 0);
                //dbg!(n_updates);
                assert_eq!(empty_vec.len(), 0);
                (weights, n + n_updates)
            },
        )
    }
    /*
    fn train(self, examples: &Vec<(I, usize)>, threshold: u64, example_truncation: usize, n_updates: usize, minibatch_size: usize) -> (T, usize) {
        //println!("{} {} {} {}", threshold, example_truncation, n_updates, minibatch_size);
        examples.chunks_exact(minibatch_size).enumerate().fold((self, 0usize), |(weights, n), (i, examples)| {
            let grads_start = Instant::now();
            let candidates = weights.updates(&examples, threshold, example_truncation, n_updates);
            //dbg!(grads_start.elapsed());
            let eval_start = Instant::now();
            let best_set = weights.evaluate_update_combinations(&examples, &candidates);
            //dbg!(best_set.len());
            //best_set.iter().for_each(|(i, _)| {
            //    print!("{}, ", i.depth());
            //});
            //print!("\n",);
            let weights = best_set.iter().fold(weights, |weights, &(i, w)| weights.mutate(i, w));
            //if i % 100 == 0 {
            //    println!("{:.3}%", weights.avg_acc(&examples) * 100f64);
            //}
            //dbg!(eval_start.elapsed());
            (weights, n + (best_set.len() > 0) as usize)
        })
    }
    */
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
                //let updates = 0;
                println!(
                    "minibatch_size: {}, train acc: {:.3}% {}",
                    minibatch_size,
                    model.avg_acc(&examples) * 100f64,
                    updates
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

#[cfg(test)]
mod test {
    use super::Collide;
    use crate::layers::LayerIndex;

    type IndexType = LayerIndex<((u8, (u8, ())), (u8, (u8, (u8, ())))), (usize, (u8, (u8, ())))>;

    #[test]
    fn collide_test() {
        let t_a: IndexType = LayerIndex::Tail((3, (4, (5, ()))));
        let t_b: IndexType = LayerIndex::Tail((3, (4, (6, ()))));
        let t_c: IndexType = LayerIndex::Tail((2, (4, (5, ()))));
        let t_d: IndexType = LayerIndex::Tail((2, (4, (6, ()))));
        assert!(t_a.collide(&t_a));
        assert!(t_a.collide(&t_b));
        assert!(t_a.collide(&t_c));
        assert!(!t_a.collide(&t_d));

        let h_a: IndexType = LayerIndex::Head(((1, (2, ())), (3, (4, (5, ())))));
        let h_b: IndexType = LayerIndex::Head(((1, (2, ())), (3, (4, (6, ())))));
        let h_c: IndexType = LayerIndex::Head(((1, (3, ())), (3, (4, (5, ())))));
        let h_d: IndexType = LayerIndex::Head(((1, (3, ())), (3, (4, (6, ())))));
        assert!(h_a.collide(&h_a));
        assert!(h_a.collide(&h_b));
        assert!(h_a.collide(&h_c));
        assert!(!h_a.collide(&h_d));

        let t_a: IndexType = LayerIndex::Tail((3, (1, (2, ()))));
        let h_a: IndexType = LayerIndex::Head(((1, (2, ())), (3, (4, (5, ())))));
        let h_b: IndexType = LayerIndex::Head(((1, (3, ())), (3, (4, (5, ())))));
        assert!(t_a.collide(&h_a));
        assert!(!t_a.collide(&h_b));
    }
}
