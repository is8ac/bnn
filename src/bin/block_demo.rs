#![feature(const_generics)]
use bitnn::bits::{b16, b32, b64, BitArray, BitArrayOPs, BitWord, Distance, FloatBitIncrement};
use bitnn::shape::{Element, Flatten, Map, Shape};
extern crate rand;
use rand::SeedableRng;
use rand_hc::Hc128Rng;

use rand::Rng;

trait BlockCode<K>
where
    Self: BitArray + Element<K>,
    b64: Element<Self::BitShape>,
    bool: Element<K>,
    K: Shape,
{
    fn bool_block<const SCALE: f64>() -> Vec<<bool as Element<K>>::Array>;
    fn encoder<const SCALE: f64>() -> <Self as Element<K>>::Array;
    fn decoder<const SCALE: f64>() -> <b64 as Element<Self::BitShape>>::Array;
    fn apply_block(&self, block: &<Self as Element<K>>::Array) -> usize;
    fn reverse_block(block: &<Self as Element<K>>::Array, index: usize) -> Self;
    fn reverse_block_iter(block: &<Self as Element<K>>::Array, index: usize) -> Self;
}

impl<
        N: std::fmt::Debug
            + Default
            + FloatBitIncrement
            + BitArray
            + Copy
            + BitWord
            + BitArrayOPs
            + Distance<Rhs = N>,
        const K: usize,
    > BlockCode<[(); K]> for N
where
    //[Vec<bool>; K]: Default,
    N::BitShape: Flatten<bool> + Flatten<b64> + Map<u32, bool> + Map<f64, bool>,
    u32: Element<N::BitShape>,
    f64: Element<N::BitShape>,
    bool: Element<N::BitShape>,
    b64: Element<N::BitShape>,
    [b32; K]: Default,
    [f64; K]: Default + std::fmt::Debug,
    [(); K]: Flatten<N> + Shape + Sized,
    <u32 as Element<N::BitShape>>::Array: Default + std::fmt::Debug,
    <f64 as Element<N::BitShape>>::Array: Default + std::fmt::Debug,
    [bool; K]: Default,
{
    fn bool_block<const SCALE: f64>() -> Vec<[bool; K]> {
        (0..N::BIT_LEN)
            .map(|i| {
                let mut target = <[bool; K]>::default();
                for k in 0..K {
                    target[k] = block_bit::<SCALE>(i, k);
                }
                target
            })
            .collect()
    }
    // K must be <= 32.
    // N can be as big as you please, within reason.

    // SCALE is between 1.0 and 2.0
    // 2.0 will produce a K of log2(N)
    // smaller will produce more.
    fn encoder<const SCALE: f64>() -> [N; K] {
        let vec_block: Vec<N> = (0..K)
            .map(|k| {
                let block_row: Vec<bool> =
                    (0..N::BIT_LEN).map(|i| block_bit::<SCALE>(i, k)).collect();
                let bools = N::BitShape::from_vec(&block_row);
                N::bitpack(&bools)
            })
            .collect();
        <[(); K]>::from_vec(&vec_block)
    }

    // swap the dims of gen_encoder_block.
    // Only part of the u64 will be filled. Probably less then 32 bits.
    fn decoder<const SCALE: f64>() -> <b64 as Element<N::BitShape>>::Array {
        let vec_block: Vec<b64> = (0..N::BIT_LEN)
            .map(|i| {
                let mut target = 0u64;
                for k in 0..K {
                    target |= (block_bit::<SCALE>(i, k) as u64) << k;
                }
                b64(target)
            })
            .collect();
        N::BitShape::from_vec(&vec_block)
    }
    fn apply_block(&self, block: &[Self; K]) -> usize {
        let mut target = 0_u64;
        for i in 0..K {
            target |= ((self.distance(&block[i]) > (N::BIT_LEN as u32 / 2)) as u64) << i;
        }
        target as usize
    }
    fn reverse_block(block: &[N; K], index: usize) -> N {
        let mut counters = <u32 as Element<N::BitShape>>::Array::default();
        for i in 0..K {
            let sign = b64(index as u64).bit(i);
            block[i].flipped_increment_counters(sign, &mut counters);
        }
        //dbg!(&counters);
        let bools =
            <N::BitShape as Map<u32, bool>>::map(&counters, |&count| count > (K / 2) as u32);
        N::bitpack(&bools)
    }
    fn reverse_block_iter(block: &[N; K], index: usize) -> N {
        let threshold = N::BIT_LEN as f64 / 2.0;
        let index_bits = {
            let mut target = <[f64; K]>::default();
            for i in 0..K {
                target[i] = if b64(index as u64).bit(i) {
                    -1f64
                } else {
                    1f64
                } / N::BIT_LEN as f64;
            }
            target
        };
        //dbg!(b64(index as u64));
        let mut weights = {
            let mut weights = <[f64; K]>::default();
            for i in 0..K {
                weights[i] = 1f64;
            }
            weights
        };
        let mut n_bit_string = <N as BlockCode<[(); K]>>::reverse_block(block, index);
        for _ in 0..20 {
            n_bit_string = {
                let mut counters = <f64 as Element<N::BitShape>>::Array::default();
                for i in 0..K {
                    block[i].float_increment_counters(weights[i] * index_bits[i], &mut counters);
                }
                //dbg!(&counters);
                let bools = <N::BitShape as Map<f64, bool>>::map(&counters, |&count| count < 0.0);
                N::bitpack(&bools)
            };

            for i in 0..K {
                weights[i] = (if b64(index as u64).bit(i) {
                    N::BIT_LEN as u32 - n_bit_string.distance(&block[i])
                } else {
                    n_bit_string.distance(&block[i])
                } as f64);
                //dbg!(weights[i]);
            }
            //dbg!(n_bit_string);
            //dbg!(weights);
        }
        n_bit_string
    }
}

#[inline]
fn block_bit<const SCALE: f64>(n_i: usize, k_i: usize) -> bool {
    ((n_i as f64 / SCALE.powi(k_i as i32)) as usize % 2) == 0
}

const N_EXAMPLES: usize = 1000;
const WS: f64 = 1.4;
const KB: usize = 20;
type BlockType = [b32; 18];

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let encoder_block = <BlockType as BlockCode<[(); KB]>>::encoder::<WS>();
    let decoder_block = <BlockType as BlockCode<[(); KB]>>::decoder::<WS>();
    dbg!(encoder_block);
    //dbg!(decoder_block);
    let sum: u32 = (0..N_EXAMPLES)
        .map(|x| {
            let o_value: BlockType = rng.gen();
            //dbg!(o_value);
            let index = o_value.apply_block(&encoder_block);
            //dbg!(b64(index as u64));
            let d_value =
                <BlockType as BlockCode<[(); KB]>>::reverse_block_iter(&encoder_block, index);
            //dbg!(o_value.distance(&d_value));
            let n_index = o_value.apply_block(&encoder_block);
            //index.distance(&n_index)
            o_value.distance(&d_value)
        })
        .sum();
    dbg!(sum as f64 / N_EXAMPLES as f64);
}

// rustc --version rustc 1.41.0-nightly (412f43ac5 2019-11-24)
