use crate::bits::{b64, t64};
use crate::count::ElementwiseAdd;
use crate::matrix::transpose;
use rayon::prelude::*;
use std::convert::TryFrom;
use std::convert::TryInto;

/// Requires that examples.len() be a multiple of
pub fn count_target_bits<const T: usize, const S: usize>(examples: &[[b64; T]]) -> Box<(u64, [[u64; 64]; T])>
where
    [u64; T * 64]: TryFrom<Vec<u64>>,
    [[b64; T]; S * 64]: TryFrom<Vec<[b64; T]>>,
{
    examples
        .par_chunks_exact(S * 64)
        .fold(
            || Box::new((0u64, [[0u64; 64]; T])),
            |mut acc, chunk| {
                let target: &[[b64; T]; S * 64] = <&[[b64; T]; S * 64]>::try_from(&chunk[0..]).unwrap();
                let t_target = transpose::<T, S>(target);
                acc.0 += (S * 64) as u64;
                for o in 0..T {
                    for ow in 0..64 {
                        acc.1[o][ow] += t_target[o][ow].iter().map(|x| x.count_ones()).sum::<u32>() as u64;
                    }
                }
                acc
            },
        )
        .reduce_with(|mut a, b| {
            a.elementwise_add(&b);
            a
        })
        .unwrap()
}

/// Requires that examples.len() be a multiple of
pub fn count_bits<const I: usize, const T: usize, const S: usize>(examples: &[([b64; I], [b64; T])]) -> Box<Counters<I, T>>
where
    [u64; I * 64]: TryFrom<Vec<u64>>,
    [u64; T * 64]: TryFrom<Vec<u64>>,
    [[b64; I]; S * 64]: TryFrom<Vec<[b64; I]>>,
    [[b64; T]; S * 64]: TryFrom<Vec<[b64; T]>>,
{
    examples
        .par_chunks_exact(S * 64)
        .fold(
            || Counters::<I, T>::new_box(),
            |mut acc, chunk| {
                let (input_chunk, target_chunk): (Vec<[b64; I]>, Vec<[b64; T]>) = chunk.iter().cloned().unzip();

                let input: &[[b64; I]; S * 64] = <&[[b64; I]; S * 64]>::try_from(&input_chunk[0..]).unwrap();
                let target: &[[b64; T]; S * 64] = <&[[b64; T]; S * 64]>::try_from(&target_chunk[0..]).unwrap();
                let t_input = transpose::<I, S>(input);
                let t_target = transpose::<T, S>(target);
                acc.increment::<S>(&t_input, &t_target);
                acc
            },
        )
        .reduce_with(|mut a, b| {
            Counters::merge(&mut a, b);
            a
        })
        .unwrap()
}

#[derive(Debug)]
pub struct Counters<const I: usize, const T: usize> {
    pub n: u64,
    pub counts: [[[[u64; 64]; I]; 64]; T],
    pub input: [[u64; 64]; I],
    pub target: [[u64; 64]; T],
}

impl<const I: usize, const T: usize> Counters<I, T>
where
    [u64; I * 64]: TryFrom<Vec<u64>>,
    [u64; T * 64]: TryFrom<Vec<u64>>,
{
    pub fn new_box() -> Box<Self> {
        Box::new(Counters {
            n: 0u64,
            counts: [[[[0u64; 64]; I]; 64]; T],
            input: [[0u64; 64]; I],
            target: [[0u64; 64]; T],
        })
    }
    pub fn merge(a: &mut Box<Counters<I, T>>, b: Box<Counters<I, T>>) {
        a.n += b.n;
        a.counts.elementwise_add(&b.counts);
        a.input.elementwise_add(&b.input);
        a.target.elementwise_add(&b.target);
    }
    pub fn increment<const L: usize>(&mut self, input: &[[[b64; L]; 64]; I], target: &[[[b64; L]; 64]; T]) {
        self.n += (L * 64) as u64;
        for i in 0..I {
            for o in 0..T {
                for iw in 0..64 {
                    for ow in 0..64 {
                        self.counts[o][ow][i][iw] += input[i][iw].iter().zip(target[o][ow].iter()).map(|(a, b)| (*a ^ *b).count_ones()).sum::<u32>() as u64;
                    }
                }
            }
        }
        for i in 0..I {
            for iw in 0..64 {
                self.input[i][iw] += input[i][iw].iter().map(|x| x.count_ones()).sum::<u32>() as u64;
            }
        }
        for o in 0..T {
            for ow in 0..64 {
                self.target[o][ow] += target[o][ow].iter().map(|x| x.count_ones()).sum::<u32>() as u64;
            }
        }
    }
    pub fn to_bits(&self) -> [[[b64; I]; 64]; T] {
        let mut target = [[[b64(0); I]; 64]; T];
        let threshold = self.n / 2;
        for tw in 0..T {
            for tb in 0..64 {
                for iw in 0..I {
                    for ib in 0..64 {
                        let sign = self.counts[tw][tb][iw][ib] > threshold;
                        target[tw][tb][iw].set_bit_in_place(ib, sign);
                    }
                }
            }
        }
        target
    }
    pub fn to_mask(&self, threshold: f64) -> [[[b64; I]; 64]; T] {
        let mut target = [[[b64(0); I]; 64]; T];
        let lower_threshold = (self.n as f64 * threshold) as u64;
        let upper_threshold = (self.n as f64 * (1f64 - threshold)) as u64;
        for tw in 0..T {
            for tb in 0..64 {
                for iw in 0..I {
                    for ib in 0..64 {
                        let mask = (self.counts[tw][tb][iw][ib] > upper_threshold) | (self.counts[tw][tb][iw][ib] < lower_threshold);
                        target[tw][tb][iw].set_bit_in_place(ib, mask);
                    }
                }
            }
        }
        target
    }
    pub fn to_trits(&self, threshold: f64) -> [[[t64; I]; 64]; T] {
        let mut target = [[[t64(0, 0); I]; 64]; T];
        let sign_threshold = self.n / 2;
        let lower_threshold = (self.n as f64 * threshold) as u64;
        let upper_threshold = (self.n as f64 * (1f64 - threshold)) as u64;
        for tw in 0..T {
            for tb in 0..64 {
                for iw in 0..I {
                    for ib in 0..64 {
                        let sign = self.counts[tw][tb][iw][ib] > sign_threshold;
                        let mask = (self.counts[tw][tb][iw][ib] > upper_threshold) | (self.counts[tw][tb][iw][ib] < lower_threshold);
                        target[tw][tb][iw].set_trit_in_place(ib, Some(sign).filter(|_| mask));
                    }
                }
            }
        }
        target
    }
}

pub trait HammingDist {
    type Bits;
    fn hamming_dist(&self, bits: &Self::Bits) -> u32;
}

impl<const L: usize> HammingDist for [b64; L] {
    type Bits = [b64; L];
    fn hamming_dist(&self, bits: &Self::Bits) -> u32 {
        let mut target = 0u32;
        for i in 0..L {
            target += (self[i].0 ^ bits[i].0).count_ones();
        }
        target
    }
}

impl<const L: usize> HammingDist for ([b64; L], [b64; L]) {
    type Bits = [b64; L];
    fn hamming_dist(&self, bits: &Self::Bits) -> u32 {
        let mut target = 0u32;
        for i in 0..L {
            target += ((self.0[i].0 ^ bits[i].0) & self.1[i].0).count_ones();
        }
        target
    }
}

impl<const L: usize> HammingDist for [t64; L] {
    type Bits = [b64; L];
    fn hamming_dist(&self, bits: &Self::Bits) -> u32 {
        let mut target = 0u32;
        for i in 0..L {
            target += ((self[i].0 ^ bits[i].0) & self[i].1).count_ones();
        }
        target
    }
}

pub fn count_act_dist_cache_local<W: Sync + Send + HammingDist<Bits = [b64; I]>, const I: usize, const T: usize, const S: usize>(
    examples: &[([b64; I], [b64; T])],
    weights: &[[Vec<W>; 64]; T],
) -> ActDists<I, T>
where
    [(); I * 64]: ,
{
    let weight_lens: [[usize; 64]; T] = weights
        .iter()
        .map(|x| x.iter().map(|x| x.len()).collect::<Vec<usize>>().try_into().unwrap())
        .collect::<Vec<[usize; 64]>>()
        .try_into()
        .unwrap();

    examples
        .par_chunks_exact(S)
        .fold(
            || ActDists::<I, T>::new_from_weight_lens(&weight_lens),
            |mut acc, chunk| {
                for w in 0..T {
                    for b in 0..64 {
                        chunk.iter().for_each(|(input, target)| {
                            let sign = target[w].get_bit(b);
                            weights[w][b].iter().zip(acc.counts[w][b].iter_mut()).for_each(|(trits, distr)| {
                                let act = trits.hamming_dist(input).min(I as u32 * 64 - 1);
                                distr[act as usize][sign as usize] += 1;
                            });
                        });
                    }
                }
                acc
            },
        )
        .reduce_with(|mut a, b| {
            a.merge(&b);
            a
        })
        .unwrap()
}

pub fn count_act_dist<W: Sync + Send + HammingDist<Bits = [b64; I]>, const I: usize, const T: usize, const S: usize>(
    examples: &[([b64; I], [b64; T])],
    weights: &[[Vec<W>; 64]; T],
) -> ActDists<I, T>
where
    [(); I * 64]: ,
{
    let weight_lens: [[usize; 64]; T] = weights
        .iter()
        .map(|x| x.iter().map(|x| x.len()).collect::<Vec<usize>>().try_into().unwrap())
        .collect::<Vec<[usize; 64]>>()
        .try_into()
        .unwrap();

    examples
        .par_chunks_exact(S)
        .fold(
            || ActDists::<I, T>::new_from_weight_lens(&weight_lens),
            |mut acc, chunk| {
                chunk.iter().fold(acc, |mut acc, (input, target)| {
                    for w in 0..T {
                        for b in 0..64 {
                            let sign = target[w].get_bit(b);
                            weights[w][b].iter().zip(acc.counts[w][b].iter_mut()).for_each(|(trits, distr)| {
                                let act = trits.hamming_dist(input).min(I as u32 * 64 - 1);
                                distr[act as usize][sign as usize] += 1;
                            });
                        }
                    }
                    acc
                })
            },
        )
        .reduce_with(|mut a, b| {
            a.merge(&b);
            a
        })
        .unwrap()
}

#[derive(Debug, Eq, PartialEq)]
pub struct SizedActDists<const I: usize, const T: usize, const N: usize>
where
    [(); I * 64]: ,
{
    counts: [[[[[u64; 2]; N]; 64]; T]; I * 64],
}

#[derive(Debug, Eq, PartialEq)]
pub struct ActDists<const I: usize, const T: usize>
where
    [(); I * 64]: ,
{
    counts: [[Vec<[[u64; 2]; I * 64]>; 64]; T],
}

impl<const I: usize, const T: usize> ActDists<I, T>
where
    [(); I * 64]: ,
{
    fn new_from_weight_lens(lens: &[[usize; 64]; T]) -> Self {
        let counts: [[Vec<[[u64; 2]; I * 64]>; 64]; T] = lens
            .iter()
            .map(|w| {
                w.iter()
                    .map(|&len| (0..len).map(|_| [[0u64; 2]; I * 64]).collect::<Vec<[[u64; 2]; I * 64]>>())
                    .collect::<Vec<Vec<[[u64; 2]; I * 64]>>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<[Vec<[[u64; 2]; I * 64]>; 64]>>()
            .try_into()
            .unwrap();

        ActDists { counts: counts }
    }
    fn merge(&mut self, rhs: &Self) {
        for t in 0..T {
            for b in 0..64 {
                rhs.counts[t][b].iter().zip(self.counts[t][b].iter_mut()).for_each(|(a_dist, b_dist)| {
                    for b in 0..I * 64 {
                        for s in 0..2 {
                            b_dist[b][s] += a_dist[b][s];
                        }
                    }
                });
            }
        }
    }
    pub fn find_thresholds(&self, target_counts: &[[u64; 64]; T]) -> [[Vec<(u32, u64)>; 64]; T] {
        self.counts
            .iter()
            .zip(target_counts.iter())
            .map(|(x, counts)| {
                x.iter()
                    .zip(counts.iter())
                    .map(|(x, targ_count)| x.iter().map(|dist| find_best_threshold(dist, *targ_count)).collect())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
    pub fn find_thresholds2(&self) -> [[Vec<(u32, u64)>; 64]; T] {
        self.counts
            .par_iter()
            .map(|x| {
                x.par_iter()
                    .map(|x| x.iter().map(|dist| find_best_threshold2(dist)).collect())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
}

fn find_best_threshold<const N: usize>(dist: &[[u64; 2]; N], targ: u64) -> (u32, u64) {
    let mut sum = 0u64;
    let mut threshold = 0u32;
    for i in 0..N {
        sum += dist[i][0];
        sum += dist[i][1];
        if sum > targ {
            break;
        }
        threshold = i as u32;
    }
    let acc = bit_acc(dist, threshold);
    (threshold, acc)
}

fn find_best_threshold2<const N: usize>(dist: &[[u64; 2]; N]) -> (u32, u64) {
    (0..N).map(|i| (i as u32, bit_acc(dist, i as u32))).max_by_key(|(_, acc)| *acc).unwrap()
}

fn bit_acc<const N: usize>(dist: &[[u64; 2]; N], threshold: u32) -> u64 {
    dist.iter().enumerate().map(|(a, pair)| pair[(a > threshold as usize) as usize]).sum()
}
