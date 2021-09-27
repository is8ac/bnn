#![feature(int_log)]
#![feature(generic_const_exprs)]
#![feature(avx512_target_feature)]
#![feature(stdsimd)]

use bnn::bits::{b64, BitArray, GetBit};
use bnn::bitslice::{
    bit_add, bit_add_wrapping, bit_splat, comparator, equality, extend, ragged_array_popcount,
    transpose_8, BitSlice, BlockTranspose,
};
//use bnn::matrix::{block_transpose_256, block_transpose_512};
use bnn::ecc::{decode_byte, encode_byte};
use bnn::shape::flatten_2d;
use rayon::prelude::*;
use std::arch::x86_64::{__m256i, __m512i, _mm256_setzero_si256};
use std::convert::TryFrom;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::mem;
use std::time::Instant;

fn masked_hamming_dist<const L: usize>(bits: &[b64; L], trits: &([b64; L], [b64; L])) -> u32 {
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

// Example:
// L==5
// N==3
// E==6
fn exp_count<T: BitSlice + Copy, const N: usize, const L: usize, const E: u32>(
    partial_sum: &[T; L],
    bits: &[T; E as usize],
    target_bit: &T,
    thresholds: &[[T; L]; N],
    counters: &mut [[u64; N]; 2usize.pow(E)],
) where
    T: std::fmt::Debug,
    [T; E.log2() as usize + 1]: ,
    [T; E.log2() as usize + 2]: ,
{
    for mask in 0..2usize.pow(E) {
        let mut exp_sum = [T::zeros(); E.log2() as usize + 1];
        //let mut exp_sum = [T::zeros(); L];
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

fn unit_counts<T: BitSlice + Copy, const N: usize, const L: usize>(
    partial_sum: &[T; L],
    bits: &[T],
    target_bit: &T,
    thresholds: &[[T; L]; N],
    counters: &mut [[[u64; N]; 2]],
) {
    assert_eq!(bits.len(), counters.len());

    bits.iter()
        .zip(counters.iter_mut())
        .for_each(|(input, counter)| {
            for i in 0..N {
                let full_count = bit_add_wrapping(&partial_sum, &extend(&[*input]));
                let (_, _, gt) = comparator(&full_count, &thresholds[i]);
                //let (lt, _, gt) = comparator(&partial_sum, &thresholds[i]);
                //counter[0][i] = gt.count_bits() as u64;
                counter[0][i] += gt.xor(*target_bit).not().count_bits() as u64;

                let full_count = bit_add_wrapping(&partial_sum, &extend(&[input.not()]));
                let (_, _, gt) = comparator(&full_count, &thresholds[i]);
                //let (lt, _, gt) = comparator(&partial_sum, &thresholds[i]);
                //counter[1][i] = gt.count_bits() as u64;
                counter[1][i] += gt.xor(*target_bit).not().count_bits() as u64;
            }
        });
}

fn partial_sum<T: BitSlice + Copy, const N: usize, const L: usize>(
    set_bits: &[T],
    unset_bits: &[T],
) -> [T; L] {
    let flipped_bits: Vec<T> = set_bits.iter().map(|x| x.not()).collect();
    let flipped_count = ragged_array_popcount::<T, L>(&flipped_bits);
    let normal_count = ragged_array_popcount::<T, L>(&unset_bits);
    let partial_count = bit_add_wrapping(&normal_count, &flipped_count);
    partial_count
}

#[derive(Debug)]
pub struct ShuffleTable<const B: usize, const N: usize, const E: u32>
where
    [(); E as usize]: ,
{
    base: Vec<(usize, bool)>,
    unit: [usize; B],
    exp: [(usize, bool); E as usize],
    thresholds: [u32; N],
}

impl<const B: usize, const N: usize, const E: u32> ShuffleTable<B, N, E>
where
    [(); E as usize]: ,
{
    fn extract_bits<T: BitSlice + Copy, const L: usize, const P: usize>(
        &self,
        bits: &[T; L],
    ) -> ([T; P], [T; B], [T; E as usize]) {
        let mut unit_target = [T::zeros(); B];
        let mut exp_target = [T::zeros(); E as usize];

        self.unit
            .iter()
            .zip(unit_target.iter_mut())
            .for_each(|(&i, t)| *t = bits[i]);
        self.exp
            .iter()
            .zip(exp_target.iter_mut())
            .for_each(|(&(i, s), t)| *t = bits[i].xor(T::splat(s)));

        let set_bits: Vec<T> = self
            .base
            .iter()
            .map(|&(i, s)| bits[i].xor(T::splat(s)))
            .collect();
        let sum = ragged_array_popcount(&set_bits);
        (sum, unit_target, exp_target)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct TargetAccumulator<const N: usize, const B: usize, const E: u32>
where
    [(); 2usize.pow(E)]: ,
{
    n: usize,
    unit_counts: [[[u64; N]; 2]; B],
    exp_counts: [[u64; N]; 2usize.pow(E)],
}

impl<const N: usize, const B: usize, const E: u32> TargetAccumulator<N, B, E>
where
    [(); 2usize.pow(E)]: ,
{
    fn new() -> Self {
        TargetAccumulator {
            n: 0,
            unit_counts: [[[0u64; N]; 2]; B],
            exp_counts: [[0u64; N]; 2usize.pow(E)],
        }
    }
    fn merge(&mut self, rhs: &Self) {
        self.n += rhs.n;
        for b in 0..B {
            for s in 0..2 {
                for t in 0..N {
                    self.unit_counts[b][s][t] += rhs.unit_counts[b][s][t];
                }
            }
        }
        for e in 0..2usize.pow(E) {
            for t in 0..N {
                self.exp_counts[e][t] += rhs.exp_counts[e][t];
            }
        }
    }
}

fn init_acc<const N: usize, const B: usize, const E: u32, const O: usize>(
) -> Box<[TargetAccumulator<N, B, E>; O]>
where
    [(); 2usize.pow(E)]: ,
{
    Box::new(
        (0..O)
            .map(|_| TargetAccumulator::<N, B, E>::new())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    )
}

fn merge_acc<const N: usize, const B: usize, const E: u32, const O: usize>(
    mut a: Box<[TargetAccumulator<N, B, E>; O]>,
    b: Box<[TargetAccumulator<N, B, E>; O]>,
) -> Box<[TargetAccumulator<N, B, E>; O]>
where
    [(); 2usize.pow(E)]: ,
{
    a.iter_mut().zip(b.iter()).for_each(|(a, b)| a.merge(b));
    a
}

pub trait CountBits<const I: usize, const O: usize, const B: usize, const N: usize, const E: u32>
where
    [(); E as usize]: ,
    [(); 2usize.pow(E)]: ,
{
    fn count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[ShuffleTable<B, N, E>; 64 * O],
    ) -> Box<[TargetAccumulator<N, B, E>; 64 * O]>;
}

pub struct BitSliceBitCounter<T, const P: usize> {
    n_threads: usize,
    slice_type: PhantomData<T>,
    chunk_size: usize,
}

impl<
        T,
        const I: usize,
        const O: usize,
        const P: usize,
        const B: usize,
        const N: usize,
        const E: u32,
    > CountBits<I, O, B, N, E> for BitSliceBitCounter<T, P>
where
    T: BitSlice + BlockTranspose<I> + BlockTranspose<O> + Copy + Sync + Send + std::fmt::Debug,
    [(); E as usize]: ,
    [(); 2usize.pow(E)]: ,
    [(); 64 * I]: ,
    [(); 64 * O]: ,
    [(); T::N]: ,
    [(); E.log2() as usize + 1]: ,
    [(); E.log2() as usize + 2]: ,
{
    fn count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[ShuffleTable<B, N, E>; 64 * O],
    ) -> Box<[TargetAccumulator<N, B, E>; 64 * O]> {
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
            .map(|table| expand_thresholds::<T, N, P>(&table.thresholds))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        blocks
            .par_chunks(self.chunk_size)
            //.par_iter()
            .fold(
                || init_acc::<N, B, E, { 64 * O }>(),
                |mut acc, chunk| {
                    for o in 0..(64 * O) {
                        chunk.iter().for_each(|(input, target)| {
                            acc[o].n += T::N;
                            let (base_sum, unit_bits, exp_bits) =
                                table[o].extract_bits::<T, { 64 * I }, P>(input);
                            unit_counts::<T, N, P>(
                                &base_sum,
                                &unit_bits,
                                &target[o],
                                &expanded_thresholds[o],
                                &mut acc[o].unit_counts,
                            );
                            exp_count::<T, N, P, E>(
                                &base_sum,
                                &exp_bits,
                                &target[o],
                                &expanded_thresholds[o],
                                &mut acc[o].exp_counts,
                            );
                        });
                    }
                    acc
                },
            )
            .reduce_with(|a, b| merge_acc::<N, B, E, { 64 * O }>(a, b))
            .unwrap()
    }
}

pub struct PopCountBitCounter {
    n_threads: usize,
    chunk_size: usize,
}

impl<const I: usize, const O: usize, const B: usize, const N: usize, const E: u32>
    CountBits<I, O, B, N, E> for PopCountBitCounter
where
    [(); E as usize]: ,
    [(); 2usize.pow(E)]: ,
    [(); 64 * I]: ,
    [(); 64 * O]: ,
    [(); E.log2() as usize + 1]: ,
{
    fn count_bits(
        &self,
        inputs: &[[b64; I]],
        targets: &[[b64; O]],
        table: &[ShuffleTable<B, N, E>; 64 * O],
    ) -> Box<[TargetAccumulator<N, B, E>; 64 * O]> {
        let weights: [(
            [[([b64; I], [b64; I]); 2]; B],
            [([b64; I], [b64; I]); 2usize.pow(E)],
        ); 64 * O] = table
            .iter()
            .map(|table| {
                let base = table
                    .base
                    .iter()
                    .fold(([b64(0); I], [b64(0); I]), |w, &(i, s)| w.set_trit(i, s));
                let unit_weights: [[([b64; I], [b64; I]); 2]; B] = table
                    .unit
                    .iter()
                    .map(|&i| [base.set_trit(i, false), base.set_trit(i, true)])
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let exp_weights: [([b64; I], [b64; I]); 2usize.pow(E)] = (0..2usize.pow(E))
                    .map(|mask| {
                        table
                            .exp
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| mask.bit(i))
                            .fold(base, |acc, (_, &(b, sign))| acc.set_trit(b, sign))
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                (unit_weights, exp_weights)
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        inputs
            .par_chunks(self.chunk_size)
            .zip(targets.par_chunks(self.chunk_size))
            .fold(
                || init_acc::<N, B, E, { 64 * O }>(),
                |mut acc, (input, target)| {
                    for o in 0..(64 * O) {
                        input.iter().zip(target.iter()).for_each(|(input, target)| {
                            acc[o].n += 1;
                            for b in 0..B {
                                for s in 0..2 {
                                    let count = masked_hamming_dist(&input, &weights[o].0[b][s]);
                                    for t in 0..N {
                                        acc[o].unit_counts[b][s][t] +=
                                            ((count > table[o].thresholds[t]) == target.get_bit(o))
                                                as u64;
                                    }
                                }
                            }
                            for m in 0..2usize.pow(E) {
                                let count = masked_hamming_dist(&input, &weights[o].1[m]);
                                for t in 0..N {
                                    //dbg!(o);
                                    acc[o].exp_counts[m][t] += ((count > table[o].thresholds[t])
                                        == target.get_bit(o))
                                        as u64;
                                }
                            }
                        });
                    }
                    acc
                },
            )
            .reduce_with(|a, b| merge_acc::<N, B, E, { 64 * O }>(a, b))
            .unwrap()
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .stack_size(2usize.pow(28))
        .build_global()
        .unwrap();

    let shuffle_tables: [_; 256] = (0..256)
        .map(|_| ShuffleTable::<32, 3, 6> {
            base: vec![(0, true), (1, false), (2, true), (3, false), (4, true)],
            unit: [
                5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8, 9, 10, 11, 12, 5, 6, 7, 8, 9, 10, 11, 12, 5,
                6, 7, 8, 9, 10, 11, 12,
            ],
            exp: [
                (13, false),
                (14, true),
                (15, false),
                (18, true),
                (19, true),
                (20, true),
            ],
            thresholds: [5, 6, 7],
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let bytes = std::fs::read("/big/temp/books_txt/Delphi Complete Works of Charles Dickens (Illustrated) - Charles Dickens.txt").unwrap();
    //let bytes: Vec<u8> = (0..(2usize.pow(25) + 5)).map(|i| (i % 50) as u8).collect();

    let expanded: Vec<[b64; 4]> = bytes.par_iter().map(|&b| encode_byte(b)).collect();
    //let expanded: Vec<[b64; 4]> = (0..(2usize.pow(24))).map(|_| [b64(0); 4]).collect();

    let window_start = Instant::now();
    let (input, target): (Vec<[b64; 8]>, Vec<[b64; 4]>) = expanded
        .par_windows(3)
        .map(|slice| {
            let input = flatten_2d::<b64, 2, 4>(<&[[b64; 4]; 2]>::try_from(&slice[0..2]).unwrap());
            (input, slice[2])
        })
        .take(2usize.pow(22))
        .unzip();
    dbg!(window_start.elapsed());
    dbg!(input.len());
    dbg!(target.len());

    let bit_counter = PopCountBitCounter {
        n_threads: 16,
        chunk_size: 2usize.pow(7),
    };

    //dbg!(&input[0..20]);
    let count_start = Instant::now();
    let counts1 = <PopCountBitCounter as CountBits<8, 4, 32, 3, 6>>::count_bits(
        &bit_counter,
        &input,
        &target,
        &shuffle_tables,
    );
    dbg!(&count_start.elapsed());

    let bit_counter = BitSliceBitCounter::<__m256i, 4> {
        n_threads: 16,
        slice_type: PhantomData::default(),
        chunk_size: 8,
    };

    let count_start = Instant::now();
    let counts2 = <BitSliceBitCounter<__m256i, 4> as CountBits<8, 4, 32, 3, 6>>::count_bits(
        &bit_counter,
        &input,
        &target,
        &shuffle_tables,
    );
    dbg!(&count_start.elapsed());

    for b in 0..256 {
        assert_eq!(counts1[b].n, input.len());
        assert_eq!(counts1[b].n, counts2[b].n);
    }

    assert_eq!(counts1[1].unit_counts, counts2[1].unit_counts);
    dbg!("pased unit");
    assert_eq!(counts1[1].exp_counts, counts2[1].exp_counts);
    assert_eq!(counts1, counts2);

    /*
    //dbg!(counts[100].unit_counts);
    //dbg!(counts[100].exp_counts);

    let duration = start.elapsed();
    dbg!(duration);
    dbg!(start.elapsed().as_nanos() as f64 / (input.len() / 256) as f64);
    */
}
