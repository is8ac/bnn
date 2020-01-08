use crate::bits::{b64, BitArrayOPs, BitWord, Distance};
use crate::shape::{Element, Flatten, Map, Shape};

pub trait BlockCode<K>
where
    Self: Element<K>,
    K: Shape,
{
    fn encoder() -> <Self as Element<K>>::Array;
    fn apply_block(&self, block: &<Self as Element<K>>::Array) -> usize;
    fn reverse_block(block: &<Self as Element<K>>::Array, index: usize) -> Self;
}

impl<N: Copy + BitWord + BitArrayOPs + Distance, const K: usize> BlockCode<[(); K]> for N
where
    N::BitShape: Flatten<bool> + Map<u32, bool>,
    u32: Element<N::BitShape>,
    bool: Element<N::BitShape>,
    b64: Element<N::BitShape>,
    [(); K]: Flatten<N>,
    <u32 as Element<N::BitShape>>::Array: Default,
{
    // K must be <= 32.
    // N can be as big as you please, within reason.

    // SCALE is between 1.0 and 2.0
    // 2.0 will produce a K of log2(N)
    // smaller will produce more.
    fn encoder() -> [N; K] {
        let scale = (N::BIT_LEN as f64).powf(1f64 / K as f64);
        let vec_block: Vec<N> = (0..K)
            .map(|k| {
                let window = scale.powi(k as i32);
                let block_row: Vec<bool> = (0..N::BIT_LEN)
                    .map(|i| (((i as f64 / window) as usize % 2) == 0))
                    .collect();
                let bools = N::BitShape::from_vec(&block_row);
                N::bitpack(&bools)
            })
            .collect();
        <[(); K]>::from_vec(&vec_block)
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
        let bools =
            <N::BitShape as Map<u32, bool>>::map(&counters, |&count| count > (K / 2) as u32);
        N::bitpack(&bools)
    }
}
