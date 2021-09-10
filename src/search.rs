use crate::bits::{b64, t64};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;

pub trait Weights {
    type Mutation;
    type Index;
    fn mutations() -> Vec<Self::Mutation>;
    fn indices() -> Vec<Self::Index>;
    fn get(&self, i: Self::Index) -> Self::Mutation;
    fn set(&mut self, i: Self::Index, m: Self::Mutation);
    fn apply(self, i: Self::Index, m: Self::Mutation) -> Self;
    fn diff(&self, rhs: &Self) -> Vec<(Self::Index, Self::Mutation)>;
}

impl<const L: usize> Weights for [b64; L] {
    type Mutation = bool;
    type Index = usize;
    fn mutations() -> Vec<Self::Mutation> {
        vec![false, true]
    }
    fn indices() -> Vec<Self::Index> {
        (0..(64 * L)).collect()
    }
    fn get(&self, i: usize) -> bool {
        ((self[i / 64].0 >> (i % 64)) & 1) == 1
    }
    fn set(&mut self, i: usize, b: bool) {
        self[i / 64].0 &= !(1 << (i % 64));
        self[i / 64].0 |= (b as u64) << (i % 64);
    }
    fn apply(mut self, i: Self::Index, m: Self::Mutation) -> Self {
        self.set(i, m);
        self
    }
    fn diff(&self, rhs: &Self) -> Vec<(Self::Index, Self::Mutation)> {
        (0..(64 * L))
            .filter_map(|i| {
                let cur_state = self.get(i);
                let new_state = rhs.get(i);
                if cur_state == new_state {
                    None
                } else {
                    Some((i, new_state))
                }
            })
            .collect()
    }
}

impl<const L: usize> Weights for [t64; L] {
    type Mutation = Option<bool>;
    type Index = usize;
    fn mutations() -> Vec<Self::Mutation> {
        vec![Some(false), None, Some(true)]
    }
    fn indices() -> Vec<Self::Index> {
        (0..(64 * L)).collect()
    }
    fn get(&self, i: usize) -> Option<bool> {
        let sign = (self[i / 64].0 >> (i % 64)) & 1 == 1;
        let magn = (self[i / 64].1 >> (i % 64)) & 1 == 1;
        Some(sign).filter(|_| magn)
    }
    fn set(&mut self, i: usize, b: Option<bool>) {
        self[i / 64].0 &= !(1 << (i % 64));
        self[i / 64].0 |= ((b.unwrap_or(false) as u64) << (i % 64));
        self[i / 64].1 &= !(1 << (i % 64));
        self[i / 64].1 |= ((b.is_some() as u64) << (i % 64));
    }
    fn apply(mut self, i: Self::Index, m: Self::Mutation) -> Self {
        self.set(i, m);
        self
    }
    fn diff(&self, rhs: &Self) -> Vec<(Self::Index, Self::Mutation)> {
        (0..(64 * L))
            .filter_map(|i| {
                let cur_state = self.get(i);
                let new_state = rhs.get(i);
                if cur_state == new_state {
                    None
                } else {
                    Some((i, new_state))
                }
            })
            .collect()
    }
}

pub trait SearchManager<W> {
    // Must accept mutatations different form what was proposed.
    fn update(&mut self, mutations: &Vec<(W, u64)>) -> u64;
    // Must return no more then n mutations.
    fn mutation_candidates(&self, n: usize) -> Vec<W>;
    fn weights(&self) -> W;
}

pub struct UnitSearch<W: Weights>
where
    W::Index: Eq + Hash,
    W::Mutation: Eq + Hash,
{
    cur_state: W,
    cur_n_correct: u64,
    mutation_values: HashMap<(W::Index, W::Mutation), (usize, Option<u64>)>,
}

impl<W: Weights + Debug> fmt::Debug for UnitSearch<W>
where
    W::Index: Eq + Hash,
    W::Mutation: Eq + Hash,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnitSearch")
            .field("cur_state", &self.cur_state)
            .field("cur_n_correct", &self.cur_n_correct)
            .finish()
    }
}
impl<W: Weights> UnitSearch<W>
where
    W::Index: Eq + Hash + Copy,
    W::Mutation: Eq + Hash + Copy,
{
    pub fn init(w: W) -> Self {
        let indices = W::indices();
        UnitSearch {
            cur_state: w,
            cur_n_correct: 0u64,
            mutation_values: W::mutations()
                .iter()
                .map(|&m| {
                    indices
                        .iter()
                        .map(|&i| ((i, m), (0, None)))
                        .collect::<Vec<_>>()
                })
                .flatten()
                .collect(),
        }
    }
}

impl<W: Weights + Copy> SearchManager<W> for UnitSearch<W>
where
    W::Index: Eq + Hash + Copy + Ord + Debug,
    W::Mutation: Eq + Hash + Copy + Ord + Debug,
{
    fn update(&mut self, mutations: &Vec<(W, u64)>) -> u64 {
        //dbg!(mutations.len());
        mutations.iter().for_each(|(w, n_correct)| {
            let diff = self.cur_state.diff(w);
            //dbg!(diff.len());
            //let weight: f64 = (*n_correct as f64 / self.n_examples as f64) / diff.len() as f64;
            diff.iter().for_each(|k| {
                //dbg!(k);
                *self.mutation_values.get_mut(k).unwrap() = (0, Some(*n_correct));
            });
        });

        let (top_mutation, n_correct) = *mutations.iter().max_by_key(|(_, c)| *c).unwrap();

        //let applied_mutations = self.cur_state.diff(&top_mutation);

        if n_correct > self.cur_n_correct {
            self.cur_n_correct = n_correct;
            self.cur_state = top_mutation;
        }
        self.mutation_values
            .iter_mut()
            .for_each(|(k, (age, _))| *age += 1);
        //dbg!(&self.mutation_values);
        self.cur_n_correct
    }
    fn mutation_candidates(&self, n: usize) -> Vec<W> {
        let sorted_by_expected_value: HashMap<(W::Index, W::Mutation), u64> = {
            let mut sorted: Vec<((W::Index, W::Mutation), u64)> = self
                .mutation_values
                .iter()
                .filter(|(&(i, m), _)| self.cur_state.get(i) != m)
                .filter(|(&_, (_, n))| n.unwrap_or_default() > self.cur_n_correct)
                .filter_map(|(&k, (age, value))| value.map(|v| (k, v)))
                .collect();
            sorted.sort();
            sorted.sort_by_key(|(k, v)| Reverse(*v));
            sorted.drain(0..).take(n / 2).collect()
        };
        //dbg!(sorted_by_expected_value.len());
        let sorted_by_age = {
            let mut sorted: Vec<((W::Index, W::Mutation), usize)> = self
                .mutation_values
                .iter()
                .filter(|(&(i, m), _)| self.cur_state.get(i) != m)
                .filter(|(k, _)| !sorted_by_expected_value.contains_key(k))
                .map(|(&k, &(age, _))| (k, age))
                .collect();
            sorted.sort();
            sorted.sort_by_key(|(k, n)| Reverse(*n));
            sorted
        };
        //dbg!(&sorted_by_age[0..5]);
        //dbg!(sorted_by_age[sorted_by_age.len() - 1]);

        sorted_by_expected_value
            .iter()
            .map(|(k, _)| *k)
            .take(n / 2)
            .chain(sorted_by_age.iter().map(|(k, _)| *k))
            .inspect(|(i, m)| {
                //println!("{:?} {:?} {:?}", i, m, self.cur_state.get(*i));
            })
            .map(|(i, m)| self.cur_state.apply(i, m))
            .take(n)
            .collect()
    }
    fn weights(&self) -> W {
        self.cur_state
    }
}
