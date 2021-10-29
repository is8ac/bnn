use crate::bits::{b64, BitArray, GetBit};
use crate::bitslice::{bit_add_wrapping, bit_splat, comparator, extend, BitSlice, BlockTranspose};
use rayon::prelude::*;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::marker::PhantomData;

pub fn masked_hamming_dist<const L: usize>(bits: &[b64; L], trits: &([b64; L], [b64; L])) -> u32 {
    bits.iter()
        .zip(trits.0.iter())
        .zip(trits.1.iter())
        .map(|((b, s), m)| ((b.0 ^ s.0) & m.0).count_ones())
        .sum()
}

pub trait TritArray {
    fn set_trit(self, i: usize, t: bool) -> Self;
}

impl<const L: usize> TritArray for ([b64; L], [b64; L]) {
    fn set_trit(mut self, i: usize, s: bool) -> Self {
        self.0.set_bit_in_place(i, s);
        self.1.set_bit_in_place(i, true);
        self
    }
}

fn expand_thresholds<T: BitSlice + Copy, const N: usize, const L: usize>(
    thresholds: &[u32; N],
) -> [[T; L]; N] {
    let mut expanded_thresholds = [[T::zeros(); L]; N];
    for i in 0..N {
        expanded_thresholds[i] = bit_splat(thresholds[i]);
    }
    expanded_thresholds
}

fn exp_count<T: BitSlice + Copy, const N: usize, const L: usize, const E: u32>(
    partial_sum: &[T; L],
    bits: &[T; E as usize],
    target_bit: &T,
    thresholds: &[[T; L]; N],
    counters: &mut [[u64; N]; 2usize.pow(E)],
) where
    T: std::fmt::Debug,
    [T; E.log2() as usize + 1]: ,
{
    for mask in 0..2usize.pow(E) {
        let mut exp_sum = [T::zeros(); E.log2() as usize + 1];
        for b in 0..E {
            let expanded = extend(&[T::splat(mask.bit(b as usize)).and(bits[b as usize])]);
            exp_sum = bit_add_wrapping(&exp_sum, &expanded);
        }
        let exp_sum = extend(&exp_sum);
        let full_sum = bit_add_wrapping(&exp_sum, &partial_sum);

        for i in 0..N {
            let (_, _, gt) = comparator(&full_sum, &thresholds[i]);
            counters[mask][i] += gt.xor(*target_bit).not().count_bits() as u64;
        }
    }
}

fn unit_count<T: BitSlice + Copy, const I: usize, const N: usize, const P: usize>(
    partial_sum: &[T; P],
    inputs: &[T; I],
    target_bit: &T,
    thresholds: &[[T; P]; N],
    counters: &mut [[[u64; N]; 2]; I],
) {
    for i in 0..I {
        for s in 0..2 {
            let full_count =
                bit_add_wrapping(&partial_sum, &extend(&[inputs[i].xor(T::splat(s == 1))]));
            for t in 0..N {
                let (_, _, gt) = comparator(&full_count, &thresholds[t]);
                counters[i][s][t] += gt.xor(*target_bit).not().count_bits() as u64;
            }
        }
    }
}

fn compute_base_sum<T: BitSlice + Copy, const L: usize, const P: usize>(
    table: &Vec<(usize, bool)>,
    bits: &[T; L],
) -> [T; P] {
    table.iter().fold([T::zeros(); P], |sum, &(i, sign)| {
        bit_add_wrapping(&sum, &extend(&[bits[i].xor(T::splat(sign))]))
    })
}

fn extract_exp_bits<T: BitSlice + Copy, const L: usize, const E: u32>(
    table: &[(usize, bool); E as usize],
    bits: &[T; L],
) -> [T; E as usize] {
    let mut exp_target = [T::zeros(); E as usize];
    table
        .iter()
        .zip(exp_target.iter_mut())
        .for_each(|(&(i, s), t)| *t = bits[i].xor(T::splat(s)));
    exp_target
}

fn init_unit_acc<const I: usize, const O: usize, const N: usize>() -> Box<[[[[u64; N]; 2]; I]; O]> {
    Box::new(
        (0..O)
            .map(|_| [[[0u64; N]; 2]; I])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    )
}

fn merge_unit_acc<const I: usize, const O: usize, const N: usize>(
    mut a: Box<[[[[u64; N]; 2]; I]; O]>,
    b: Box<[[[[u64; N]; 2]; I]; O]>,
) -> Box<[[[[u64; N]; 2]; I]; O]> {
    for o in 0..O {
        for i in 0..I {
            for s in 0..2 {
                for n in 0..N {
                    a[o][i][s][n] += b[o][i][s][n];
                }
            }
        }
    }
    a
}

fn init_exp_acc<const O: usize, const N: usize, const E: u32>(
) -> Box<[[[u64; N]; 2usize.pow(E)]; O]> {
    Box::new(
        (0..O)
            .map(|_| [[0u64; N]; 2usize.pow(E)])
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    )
}

fn merge_exp_acc<const O: usize, const N: usize, const E: u32>(
    mut a: Box<[[[u64; N]; 2usize.pow(E)]; O]>,
    b: Box<[[[u64; N]; 2usize.pow(E)]; O]>,
) -> Box<[[[u64; N]; 2usize.pow(E)]; O]>
where
    [(); 2usize.pow(E)]: ,
{
    for o in 0..O {
        for m in 0..2usize.pow(E) {
            for n in 0..N {
                a[o][m][n] += b[o][m][n];
            }
        }
    }
    a
}

pub trait UnitCountBits<const I: usize, const O: usize, const N: usize> {
    fn unit_count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[(Vec<(usize, bool)>, [u32; N]); 64 * O],
        chunk_size: usize,
    ) -> Box<[[[[u64; N]; 2]; 64 * I]; 64 * O]>;
}

pub trait ExpCountBits<const I: usize, const O: usize, const N: usize, const E: u32>
where
    [(); E as usize]: ,
    [(); 2usize.pow(E)]: ,
    [(); E.log2() as usize + 1]: ,
{
    fn exp_count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[(Vec<(usize, bool)>, [(usize, bool); E as usize], [u32; N]); 64 * O],
        chunk_size: usize,
    ) -> Box<[[[u64; N]; 2usize.pow(E)]; 64 * O]>;
}

pub struct BitSliceBitCounter<T, const P: usize> {
    pub slice_type: PhantomData<T>,
}

impl<T, const I: usize, const O: usize, const P: usize, const N: usize> UnitCountBits<I, O, N>
    for BitSliceBitCounter<T, P>
where
    T: BitSlice + Copy + BlockTranspose<I> + BlockTranspose<O> + Sync + Send + std::fmt::Debug,
    [(); 64 * I]: ,
    [(); 64 * O]: ,
    [(); T::N]: ,
{
    fn unit_count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[(Vec<(usize, bool)>, [u32; N]); 64 * O],
        chunk_size: usize,
    ) -> Box<[[[[u64; N]; 2]; 64 * I]; 64 * O]> {
        assert_eq!(chunk_size % T::N, 0);

        let blocks: Vec<([T; 64 * I], [T; 64 * O])> = inputs
            .par_chunks_exact(T::N)
            .zip(targets.par_chunks_exact(T::N))
            .map(|(input, target)| {
                let input = T::block_transpose(<&[[b64; I]; T::N]>::try_from(input).unwrap());
                let target = T::block_transpose(<&[[b64; O]; T::N]>::try_from(target).unwrap());
                (input, target)
            })
            .collect();

        let expanded_thresholds: [[[T; P]; N]; 64 * O] = table
            .iter()
            .map(|(_, thresholds)| expand_thresholds::<T, N, P>(thresholds))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut counts = blocks
            .par_chunks(chunk_size / T::N)
            .fold(
                || init_unit_acc::<{ 64 * I }, { 64 * O }, N>(),
                |mut acc, chunk| {
                    for o in 0..(64 * O) {
                        chunk.iter().for_each(|(input, target)| {
                            let base_sum = compute_base_sum(&table[o].0, input);
                            unit_count::<T, { 64 * I }, N, P>(
                                &base_sum,
                                input,
                                &target[o],
                                &expanded_thresholds[o],
                                &mut acc[o],
                            );
                        });
                    }
                    acc
                },
            )
            .reduce_with(|a, b| merge_unit_acc::<{ 64 * I }, { 64 * O }, N>(a, b))
            .unwrap();

        for o in 0..(64 * O) {
            table[o].0.iter().for_each(|&(i, _)| {
                counts[o][i] = [[0u64; N]; 2];
            });
        }
        counts
    }
}

impl<T, const I: usize, const O: usize, const P: usize, const N: usize, const E: u32>
    ExpCountBits<I, O, N, E> for BitSliceBitCounter<T, P>
where
    T: BitSlice + Copy + BlockTranspose<I> + BlockTranspose<O> + Sync + Send + std::fmt::Debug,
    [(); 64 * I]: ,
    [(); 64 * O]: ,
    [(); T::N]: ,
    [(); E as usize]: ,
    [(); 2usize.pow(E)]: ,
    [(); E.log2() as usize + 1]: ,
{
    fn exp_count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[(Vec<(usize, bool)>, [(usize, bool); E as usize], [u32; N]); 64 * O],
        chunk_size: usize,
    ) -> Box<[[[u64; N]; 2usize.pow(E)]; 64 * O]> {
        assert_eq!(chunk_size % T::N, 0);

        let blocks: Vec<([T; 64 * I], [T; 64 * O])> = inputs
            .par_chunks_exact(T::N)
            .zip(targets.par_chunks_exact(T::N))
            .map(|(input, target)| {
                let input = T::block_transpose(<&[[b64; I]; T::N]>::try_from(input).unwrap());
                let target = T::block_transpose(<&[[b64; O]; T::N]>::try_from(target).unwrap());
                (input, target)
            })
            .collect();

        let expanded_thresholds: [[[T; P]; N]; 64 * O] = table
            .iter()
            .map(|(_, _, thresholds)| expand_thresholds::<T, N, P>(thresholds))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        blocks
            .par_chunks(chunk_size / T::N)
            .fold(
                || init_exp_acc::<{ 64 * O }, N, E>(),
                |mut acc, chunk| {
                    for o in 0..(64 * O) {
                        chunk.iter().for_each(|(input, target)| {
                            let base_sum = compute_base_sum(&table[o].0, input);
                            let exp_bits = extract_exp_bits::<T, { 64 * I }, E>(&table[o].1, input);
                            exp_count::<T, N, P, E>(
                                &base_sum,
                                &exp_bits,
                                &target[o],
                                &expanded_thresholds[o],
                                &mut acc[o],
                            );
                        });
                    }
                    acc
                },
            )
            .reduce_with(|a, b| merge_exp_acc::<{ 64 * O }, N, E>(a, b))
            .unwrap()
    }
}

pub struct PopCountBitCounter {}

impl<const I: usize, const O: usize, const N: usize> UnitCountBits<I, O, N> for PopCountBitCounter {
    fn unit_count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[(Vec<(usize, bool)>, [u32; N]); 64 * O],
        chunk_size: usize,
    ) -> Box<[[[[u64; N]; 2]; 64 * I]; 64 * O]> {
        let weights: Vec<[[([b64; I], [b64; I]); 2]; 64 * I]> = table
            .iter()
            .map(|table| {
                let base = table
                    .0
                    .iter()
                    .fold(([b64(0); I], [b64(0); I]), |w, &(i, s)| w.set_trit(i, s));
                let unit_weights: [[([b64; I], [b64; I]); 2]; 64 * I] = (0..64 * I)
                    .map(|i| [base.set_trit(i, false), base.set_trit(i, true)])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                unit_weights
            })
            .collect::<Vec<_>>();

        let mut counts = inputs
            .par_chunks(chunk_size)
            .zip(targets.par_chunks(chunk_size))
            .fold(
                || init_unit_acc::<{ 64 * I }, { 64 * O }, N>(),
                |mut acc, (input, target)| {
                    for o in 0..(64 * O) {
                        input.iter().zip(target.iter()).for_each(|(input, target)| {
                            for i in 0..(64 * I) {
                                for s in 0..2 {
                                    let count = masked_hamming_dist(&input, &weights[o][i][s]);
                                    for t in 0..N {
                                        acc[o][i][s][t] +=
                                            ((count > table[o].1[t]) == target.get_bit(o)) as u64;
                                    }
                                }
                            }
                        });
                    }
                    acc
                },
            )
            .reduce_with(|a, b| merge_unit_acc::<{ 64 * I }, { 64 * O }, N>(a, b))
            .unwrap();
        for o in 0..(64 * O) {
            table[o].0.iter().for_each(|&(i, _)| {
                counts[o][i] = [[0; N]; 2];
            });
        }
        counts
    }
}

impl<const I: usize, const O: usize, const N: usize, const E: u32> ExpCountBits<I, O, N, E>
    for PopCountBitCounter
where
    [(); E as usize]: ,
    [(); 2usize.pow(E)]: ,
    [(); E.log2() as usize + 1]: ,
{
    fn exp_count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[(Vec<(usize, bool)>, [(usize, bool); E as usize], [u32; N]); 64 * O],
        chunk_size: usize,
    ) -> Box<[[[u64; N]; 2usize.pow(E)]; 64 * O]>
    where
        [(); E as usize]: ,
        [(); 2usize.pow(E)]: ,
        [(); E.log2() as usize + 1]: ,
    {
        let weights: Vec<[([b64; I], [b64; I]); 2usize.pow(E)]> = table
            .iter()
            .map(|table| {
                let base = table
                    .0
                    .iter()
                    .fold(([b64(0); I], [b64(0); I]), |w, &(i, s)| w.set_trit(i, s));
                let exp_weights: [([b64; I], [b64; I]); 2usize.pow(E)] = (0..2usize.pow(E))
                    .map(|mask| {
                        table
                            .1
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| mask.bit(i))
                            .fold(base, |acc, (_, &(b, sign))| acc.set_trit(b, sign))
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                exp_weights
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        inputs
            .par_chunks(chunk_size)
            .zip(targets.par_chunks(chunk_size))
            .fold(
                || init_exp_acc::<{ 64 * O }, N, E>(),
                |mut acc, (input, target)| {
                    for o in 0..(64 * O) {
                        input.iter().zip(target.iter()).for_each(|(input, target)| {
                            for m in 0..2usize.pow(E) {
                                let count = masked_hamming_dist(&input, &weights[o][m]);
                                for t in 0..N {
                                    acc[o][m][t] +=
                                        ((count > table[o].2[t]) == target.get_bit(o)) as u64;
                                }
                            }
                        });
                    }
                    acc
                },
            )
            .reduce_with(|a, b| merge_exp_acc::<{ 64 * O }, N, E>(a, b))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::bits::b64;
    use crate::bitslice::BitArray64;
    use crate::count_bits::{BitSliceBitCounter, ExpCountBits, PopCountBitCounter, UnitCountBits};
    use crate::search::{compute_exp_candidates, weights_to_dense, weights_to_sparse};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::marker::PhantomData;

    #[test]
    fn exp_count() {
        rayon::ThreadPoolBuilder::new()
            .stack_size(2usize.pow(28))
            .build_global()
            .unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let input: Vec<[b64; 8]> = (0..2usize.pow(16)).map(|_| rng.gen()).collect();
        let target: Vec<[b64; 4]> = (0..2usize.pow(16)).map(|_| rng.gen()).collect();

        let popcount_counter = Box::new(PopCountBitCounter {});

        let bitslice_counter1_3 = BitSliceBitCounter::<BitArray64<1>, 3> {
            slice_type: PhantomData::default(),
        };
        let bitslice_counter2_3 = BitSliceBitCounter::<BitArray64<2>, 3> {
            slice_type: PhantomData::default(),
        };
        let bitslice_counter4_3 = BitSliceBitCounter::<BitArray64<4>, 3> {
            slice_type: PhantomData::default(),
        };

        let bitslice_counter1_4 = BitSliceBitCounter::<BitArray64<1>, 4> {
            slice_type: PhantomData::default(),
        };
        let bitslice_counter2_4 = BitSliceBitCounter::<BitArray64<2>, 4> {
            slice_type: PhantomData::default(),
        };
        let bitslice_counter4_4 = BitSliceBitCounter::<BitArray64<4>, 4> {
            slice_type: PhantomData::default(),
        };

        let bitslice_counter1_5 = BitSliceBitCounter::<BitArray64<1>, 5> {
            slice_type: PhantomData::default(),
        };
        let bitslice_counter2_5 = BitSliceBitCounter::<BitArray64<2>, 5> {
            slice_type: PhantomData::default(),
        };
        let bitslice_counter4_5 = BitSliceBitCounter::<BitArray64<4>, 5> {
            slice_type: PhantomData::default(),
        };

        let weights: [([Option<bool>; 512], u32); 256] = (0..256)
            .map(|i| {
                let mut w = ([None; 512], 1u32);
                w.0[i] = Some(i % 2 == 0);
                w.0[(i * 2) % 512] = Some(i % 2 == 0);
                w.0[(i * 3) % 512] = Some(i % 2 == 1);
                w.0[(i * 5) % 512] = Some(i % 2 == 0);
                w.0[(i * 7) % 512] = Some(i % 2 == 1);
                w.0[(i * 11) % 512] = Some(i % 2 == 0);
                w
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let sparse: [(Vec<(usize, bool)>, [u32; 2]); 256] = weights
            .iter()
            .map(|w| weights_to_sparse(w))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let popcount_unit_counts = popcount_counter.unit_count_bits(&input, &target, &sparse, 1024);

        assert_eq!(
            popcount_unit_counts,
            bitslice_counter1_3.unit_count_bits(&input, &target, &sparse, 1024)
        );
        assert_eq!(
            popcount_unit_counts,
            bitslice_counter2_3.unit_count_bits(&input, &target, &sparse, 1024)
        );
        assert_eq!(
            popcount_unit_counts,
            bitslice_counter4_3.unit_count_bits(&input, &target, &sparse, 1024)
        );

        assert_eq!(
            popcount_unit_counts,
            bitslice_counter1_4.unit_count_bits(&input, &target, &sparse, 1024)
        );
        assert_eq!(
            popcount_unit_counts,
            bitslice_counter2_4.unit_count_bits(&input, &target, &sparse, 1024)
        );
        assert_eq!(
            popcount_unit_counts,
            bitslice_counter4_4.unit_count_bits(&input, &target, &sparse, 1024)
        );

        assert_eq!(
            popcount_unit_counts,
            bitslice_counter1_5.unit_count_bits(&input, &target, &sparse, 1024)
        );
        assert_eq!(
            popcount_unit_counts,
            bitslice_counter2_5.unit_count_bits(&input, &target, &sparse, 1024)
        );
        assert_eq!(
            popcount_unit_counts,
            bitslice_counter4_5.unit_count_bits(&input, &target, &sparse, 1024)
        );

        let exp_candidates: [(Vec<(usize, bool)>, [(usize, bool); 7], [u32; 3]); 256] = weights
            .iter()
            .zip(popcount_unit_counts.iter())
            .map(|(w, counts)| compute_exp_candidates::<512, 2, 3, 7>(w, counts))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let popcount_exp_counts = <PopCountBitCounter as ExpCountBits<8, 4, 3, 7>>::exp_count_bits(
            &popcount_counter,
            &input,
            &target,
            &exp_candidates,
            1024,
        );
        assert_eq!(
            popcount_exp_counts,
            <_ as ExpCountBits<8, 4, 3, 7>>::exp_count_bits(
                &bitslice_counter1_4,
                &input,
                &target,
                &exp_candidates,
                1024
            )
        );
        assert_eq!(
            popcount_exp_counts,
            <_ as ExpCountBits<8, 4, 3, 7>>::exp_count_bits(
                &bitslice_counter2_4,
                &input,
                &target,
                &exp_candidates,
                1024
            )
        );
        assert_eq!(
            popcount_exp_counts,
            <_ as ExpCountBits<8, 4, 3, 7>>::exp_count_bits(
                &bitslice_counter4_4,
                &input,
                &target,
                &exp_candidates,
                1024
            )
        );

        assert_eq!(
            popcount_exp_counts,
            <_ as ExpCountBits<8, 4, 3, 7>>::exp_count_bits(
                &bitslice_counter1_5,
                &input,
                &target,
                &exp_candidates,
                1024
            )
        );
        assert_eq!(
            popcount_exp_counts,
            <_ as ExpCountBits<8, 4, 3, 7>>::exp_count_bits(
                &bitslice_counter2_5,
                &input,
                &target,
                &exp_candidates,
                1024
            )
        );
        assert_eq!(
            popcount_exp_counts,
            <_ as ExpCountBits<8, 4, 3, 7>>::exp_count_bits(
                &bitslice_counter4_5,
                &input,
                &target,
                &exp_candidates,
                1024
            )
        );
    }
}
