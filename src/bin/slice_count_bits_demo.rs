#![feature(int_log)]
#![feature(generic_const_exprs)]
#![feature(avx512_target_feature)]
#![feature(stdsimd)]

use bnn::bits::b64;
use bnn::bitslice::{
    bit_add, bit_add_wrapping, bit_splat, comparator, equality, extend, ragged_array_popcount,
    transpose_8, BitSlice, BlockTranspose,
};
//use bnn::matrix::{block_transpose_256, block_transpose_512};
use bnn::search::Weights;
use bnn::shape::flatten_2d;
use rayon::prelude::*;
use std::arch::x86_64::{__m256i, __m512i, _mm256_setzero_si256};
use std::convert::TryFrom;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::mem;
use std::time::Instant;

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
    [T; E.log2() as usize + 1]: ,
{
    for mask in 0..2usize.pow(E) {
        let mut exp_sum = [T::zeros(); E.log2() as usize + 1];
        for b in 0..E {
            let bit = ((mask >> b) & 1) == 1;
            let expanded = extend(&[T::splat(bit).and(bits[b as usize])]);
            exp_sum = bit_add_wrapping(&exp_sum, &expanded);
        }
        let exp_sum = extend(&exp_sum);
        let full_sum = bit_add_wrapping(&exp_sum, &partial_sum);

        for i in 0..N {
            let full_count = bit_add_wrapping(&partial_sum, &full_sum);
            let (lt, _, _) = comparator(&full_count, &thresholds[i]);
            counters[mask][i] = lt.xor(*target_bit).count_bits() as u64;
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
                let (lt, _, _) = comparator(&full_count, &thresholds[i]);
                counter[0][i] = lt.xor(*target_bit).count_bits() as u64;

                let full_count = bit_add_wrapping(&partial_sum, &extend(&[input.not()]));
                let (lt, _, _) = comparator(&full_count, &thresholds[i]);
                counter[1][i] = lt.xor(*target_bit).count_bits() as u64;
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
    set: Vec<(usize, bool)>,
    unit: [usize; B],
    exp: [(usize, bool); E as usize],
    thresholds: [u32; N],
}

impl<const B: usize, const N: usize, const E: u32> ShuffleTable<B, N, E>
where
    [(); E as usize]: ,
{
    fn gen_weights<W: Weights>(&self) -> (Vec<W>, [u32; N]) {
        (vec![], self.thresholds)
    }
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
            .set
            .iter()
            .map(|&(i, s)| bits[i].xor(T::splat(s)))
            .collect();
        let sum = ragged_array_popcount(&set_bits);
        (sum, unit_target, exp_target)
    }
}

#[derive(Debug)]
pub struct TargetAccumulator<const N: usize, const B: usize, const E: u32>
where
    [(); 2usize.pow(E)]: ,
{
    unit_counts: [[[u64; N]; 2]; B],
    exp_counts: [[u64; N]; 2usize.pow(E)],
}

impl<const N: usize, const B: usize, const E: u32> TargetAccumulator<N, B, E>
where
    [(); 2usize.pow(E)]: ,
{
    fn new() -> Self {
        TargetAccumulator {
            unit_counts: [[[0u64; N]; 2]; B],
            exp_counts: [[0u64; N]; 2usize.pow(E)],
        }
    }
    fn merge(&mut self, rhs: &Self) {
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

fn init_acc<T, const N: usize, const B: usize, const E: u32, const O: usize>(
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

fn merge_acc<T, const N: usize, const B: usize, const E: u32, const O: usize>(
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
            .fold(
                || init_acc::<T, N, B, E, { 64 * O }>(),
                |mut acc, chunk| {
                    for o in 0..(64 * O) {
                        chunk.iter().for_each(|(input, target)| {
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
            .reduce_with(|a, b| merge_acc::<T, N, B, E, { 64 * O }>(a, b))
            .unwrap()
    }
}

const NT: usize = 3;
const DEPTH: usize = 6;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .stack_size(2usize.pow(28))
        .build_global()
        .unwrap();

    let trits = ([b64(0b_10_u64); 8], [b64(0b_11_u64); 8]);

    let shuffle_tables: [_; 256] = (0..256)
        .map(|_| ShuffleTable::<8, 3, 3> {
            set: vec![(0, true), (1, false), (2, true), (3, false), (4, true)],
            unit: [5, 6, 7, 8, 9, 10, 11, 12],
            exp: [(13, false), (14, true), (15, true)],
            thresholds: [2, 3, 4],
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let expanded: Vec<[b64; 4]> = (0..(2usize.pow(24))).map(|_| [b64(0); 4]).collect();

    let window_start = Instant::now();
    let (input, target): (Vec<[b64; 8]>, Vec<[b64; 4]>) = expanded
        .par_windows(3)
        .map(|slice| {
            let input = flatten_2d::<b64, 2, 4>(<&[[b64; 4]; 2]>::try_from(&slice[0..2]).unwrap());
            (input, slice[2])
        })
        .unzip();
    dbg!(window_start.elapsed());

    let bit_counter = BitSliceBitCounter::<__m256i, 5> {
        n_threads: 16,
        slice_type: PhantomData::default(),
        chunk_size: 17,
    };

    let count_start = Instant::now();
    let counts = <BitSliceBitCounter<__m256i, 5> as CountBits<8, 4, 8, 3, 3>>::count_bits(
        &bit_counter,
        &input,
        &target,
        &shuffle_tables,
    );
    dbg!(&count_start.elapsed());

    dbg!(counts[100].unit_counts);
    dbg!(counts[100].exp_counts);

    /*
    let duration = start.elapsed();
    dbg!(duration);
    dbg!(start.elapsed().as_nanos() as f64 / (input.len() / 256) as f64);
    */
}
