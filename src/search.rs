use crate::bits::b64;
use crate::bits::GetBit;
use crate::count_bits::{ShuffleTable, TargetAccumulator};
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::fmt::Debug;
use std::hash::Hash;

fn compute_exp_candidates<const I: usize, const N: usize, const E: u32>(
    weights: &[Option<bool>; I],
    threshold: u32,
    unit_counts: &[[[u64; N]; 2]; I],
) -> ([(usize, bool); E as usize], [u32; N]) {
    let mut sorted_candidates: Vec<_> = weights
        .iter()
        .zip(unit_counts.iter())
        .enumerate()
        .filter_map(|(i, (weight, counts))| {
            if let None = weight {
                Some(
                    counts
                        .iter()
                        .enumerate()
                        .map(|(s, counts)| counts.iter().max().map(|&a| (a, (i, s == 1))).unwrap())
                        .max()
                        .unwrap(),
                )
            } else {
                // skip
                None
            }
        })
        .collect();
    sorted_candidates.sort_by_key(|&(a, _)| Reverse(a));

    let candidates: [(usize, bool); E as usize] = sorted_candidates
        .iter()
        .map(|&(_, (i, s))| (i, s))
        .take(2usize.pow(E))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let thresholds: [u32; N] = (0..N)
        .map(|x| threshold + x as u32)
        .collect::<Vec<u32>>()
        .try_into()
        .unwrap();
    (candidates, thresholds)
}

fn update_weights<const I: usize, const N: usize, const E: u32>(
    weights: &mut ([Option<bool>; I], u32),
    exp_candidates: &[(usize, bool); E as usize],
    exp_counts: &[[u64; N]; 2usize.pow(E)],
    thresholds: &[u32; N],
) -> u64 {
    let (mask, (accuracy, threshold)): (usize, (u64, u32)) = exp_counts
        .iter()
        .enumerate()
        .map(|(mask, &counts)| {
            let (accuracy, threshold) = counts.iter().zip(thresholds.iter()).max().unwrap();
            (mask, (*accuracy, *threshold))
        })
        .max_by_key(|(mask, (a, _))| (*a, mask.count_zeros()))
        .unwrap();
    if mask == 0 {
        println!("stuck",);
    }
    weights.1 = threshold;

    exp_candidates.iter().enumerate().for_each(|(b, &(i, s))| {
        weights.0[i] = Some(s).filter(|_| mask.bit(b));
    });
    accuracy
}

fn weights_to_sparse<const I: usize>(weights: &[Option<bool>; I]) -> Vec<(usize, bool)> {
    weights
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.map(|s| (i, s)))
        .collect()
}

/*
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Weight {
    Set(bool),
    Unset([u64; 2]),
}

#[derive(Debug)]
pub struct TargetSearchManager<const I: usize, const N: usize, const E: u32>
where
    [(); E as usize]: ,
{
    pub n: Option<usize>,
    pub cur_acc: u64,
    pub cur_threshold: u32,
    pub weights: [Weight; I],
}

impl<const I: usize, const N: usize, const E: u32> TargetSearchManager<I, N, E>
where
    [(); E as usize]: ,
    [(); 2usize.pow(E)]: ,
{
    pub fn new() -> Self {
        TargetSearchManager {
            n: None,
            cur_acc: 0,
            cur_threshold: 0,
            weights: (0..I).map(|_| Weight::Unset([0, 0])).collect::<Vec<_>>().try_into().unwrap(),
        }
    }
    pub fn query(&self) -> ShuffleTable<N, E> {
        let mut exp_sort: Vec<(usize, i64, bool)> = self
            .weights
            .iter()
            .enumerate()
            .filter_map(|(i, &w)| {
                if let Weight::Unset((_, d)) = w {
                    Some((i, d[0].max(d[1]), d[0] < d[1]))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        exp_sort.sort_by_key(|&(i, d, s)| Reverse((d, i)));

        ShuffleTable::<N, E> {
            base: self
                .weights
                .iter()
                .enumerate()
                .filter_map(|(i, &w)| if let Weight::Set(sign) = w { Some((i, sign)) } else { None })
                .collect(),
            exp: exp_sort[0..E as usize].iter().map(|&(i, _, s)| (i, s)).collect::<Vec<_>>().try_into().unwrap(),
            thresholds: (0..N).map(|i| i as u32 + self.cur_threshold).collect::<Vec<u32>>().try_into().unwrap(),
        }
    }
    pub fn apply_unit(&mut self, unit_counts: [[[u64; N]; 2]; I]) {
        let set_weights: Vec<_> = self
            .weights
            .iter()
            .enumerate()
            .filter_map(|(i, w)| if let Weight::Set(s) = w { Some((i, *s)) } else { None })
            .collect();
        assert_eq!(set_weights, table.base);

        counts.unit_counts.iter().enumerate().for_each(|(i, c)| {
            if let Weight::Unset((age, delta)) = &mut self.weights[i] {
                *age = 0;
                for s in 0..2 {
                    delta[s] = *c[s].iter().max().unwrap() as i64 - self.cur_acc as i64;
                }
            } else {
                panic!("attempted to set already set weight!")
            }
        });

        self.weights.iter_mut().for_each(|w| {
            if let Weight::Unset((age, _)) = w {
                *age += 1;
            }
        });

        let (mutations, (accuracy, threshold)): (Vec<(usize, bool)>, (u64, u32)) = counts
            .exp_counts
            .iter()
            .enumerate()
            .map(|(mask, &counts)| {
                let signs: Vec<(usize, bool)> = table
                    .exp
                    .iter()
                    .enumerate()
                    .filter_map(|(b, &(i, sign))| if mask.bit(b) { Some((i, sign)) } else { None })
                    .collect();

                let (accuracy, threshold) = counts.iter().zip(table.thresholds.iter()).max().unwrap();
                (signs, (*accuracy, *threshold))
            })
            .max_by_key(|(_, (a, _))| *a)
            .unwrap();
        //dbg!(&mutations);
        if accuracy > self.cur_acc {
            //dbg!(accuracy);
            self.cur_acc = accuracy;
            self.cur_threshold = threshold;
            //dbg!(&mutations.len());
            mutations.iter().for_each(|&(i, s)| {
                if let Weight::Set(_) = self.weights[i] {
                    dbg!(i);
                    panic!("attempt to set already set weight.")
                }
                self.weights[i] = Weight::Set(s);
                //println!("updating",);
            });
        } else {
            //println!("stuck",);
        }

        self.cur_acc
    }
    pub fn apply_exp(&mut self, table: &ShuffleTable<N, E>, counts: &TargetAccumulator<I, N, E>) -> u64 {
        if let Some(n) = self.n {
            assert_eq!(counts.n, n);
            assert_eq!(counts.exp_counts[0b_0][0], self.cur_acc);
        }
        self.n = Some(counts.n);

        let set_weights: Vec<_> = self
            .weights
            .iter()
            .enumerate()
            .filter_map(|(i, w)| if let Weight::Set(s) = w { Some((i, *s)) } else { None })
            .collect();
        assert_eq!(set_weights, table.base);

        counts.unit_counts.iter().enumerate().for_each(|(i, c)| {
            if let Weight::Unset((age, delta)) = &mut self.weights[i] {
                *age = 0;
                for s in 0..2 {
                    delta[s] = *c[s].iter().max().unwrap() as i64 - self.cur_acc as i64;
                }
            } else {
                panic!("attempted to set already set weight!")
            }
        });

        self.weights.iter_mut().for_each(|w| {
            if let Weight::Unset((age, _)) = w {
                *age += 1;
            }
        });

        let (mutations, (accuracy, threshold)): (Vec<(usize, bool)>, (u64, u32)) = counts
            .exp_counts
            .iter()
            .enumerate()
            .map(|(mask, &counts)| {
                let signs: Vec<(usize, bool)> = table
                    .exp
                    .iter()
                    .enumerate()
                    .filter_map(|(b, &(i, sign))| if mask.bit(b) { Some((i, sign)) } else { None })
                    .collect();

                let (accuracy, threshold) = counts.iter().zip(table.thresholds.iter()).max().unwrap();
                (signs, (*accuracy, *threshold))
            })
            .max_by_key(|(_, (a, _))| *a)
            .unwrap();
        //dbg!(&mutations);
        if accuracy > self.cur_acc {
            //dbg!(accuracy);
            self.cur_acc = accuracy;
            self.cur_threshold = threshold;
            //dbg!(&mutations.len());
            mutations.iter().for_each(|&(i, s)| {
                if let Weight::Set(_) = self.weights[i] {
                    dbg!(i);
                    panic!("attempt to set already set weight.")
                }
                self.weights[i] = Weight::Set(s);
                //println!("updating",);
            });
        } else {
            //println!("stuck",);
        }

        self.cur_acc
    }
}
*/
