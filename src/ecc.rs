use crate::bits::{b32, b64, GetBit, PackedIndexSGet, BMA};
use crate::random_tables::{RANDOM_TABLE_128, RANDOM_TABLE_256, RANDOM_TABLE_64};
use crate::shape::Shape;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::num::Wrapping;
use std::sync::atomic::{AtomicU32, Ordering};

pub trait ExpandByte: Sized + Default + Copy
where
    Standard: Distribution<Self>,
    Standard: Distribution<[Self; 256]>,
{
    const N: usize;
    const SEED: u64;
    const MIN_DIST: u32;
    fn count(&self) -> u32;
    fn gen_table() -> [Self; 256] {
        Hc128Rng::seed_from_u64(Self::SEED).gen()
        //[Self::default(); 256]
    }
    fn bruteforce_table() {
        let mut cur_min_dist = AtomicU32::new(0);
        (0u64..).par_bridge().for_each(|seed| {
            let table: [Self; 256] = Hc128Rng::seed_from_u64(seed).gen();

            let counts: Vec<u32> = table.iter().map(|x| x.count()).collect();
            let max: u32 = *counts.iter().max().unwrap();
            let min: u32 = *counts.iter().min().unwrap();

            let low = ((Self::N / 2) as f64 * 0.7) as u32;
            let high = ((Self::N / 2) as f64 * 1.3) as u32;

            if (max < high) & (min > low) {
                let mut counts = [0u32; 1024];

                for x in 0..256 {
                    for y in 0..256 {
                        if x != y {
                            let dist = table[x].distance(&table[y]);
                            counts[dist as usize] += 1;
                        }
                    }
                }
                let new_min_dist = counts.iter().enumerate().find(|(i, c)| **c > 0).unwrap().0;
                if new_min_dist as u32 > cur_min_dist.load(Ordering::Relaxed) {
                    println!("{} {}", seed, new_min_dist);
                    cur_min_dist.store(new_min_dist as u32, Ordering::Relaxed);
                }
            }
        })
    }
    fn distance(&self, rhs: &Self) -> u32;
    fn encode_byte(b: u8) -> Self;
    fn decode_byte(&self) -> u8;
}

impl ExpandByte for b64 {
    const N: usize = 64;
    const SEED: u64 = 17246404;
    const MIN_DIST: u32 = 19;
    fn count(&self) -> u32 {
        self.count_ones()
    }
    fn distance(&self, rhs: &b64) -> u32 {
        (self.0 ^ rhs.0).count_ones()
    }
    fn encode_byte(b: u8) -> Self {
        RANDOM_TABLE_64[b as usize]
    }
    fn decode_byte(&self) -> u8 {
        RANDOM_TABLE_64
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.distance(self))
            .unwrap()
            .0 as u8
    }
}

impl ExpandByte for [b64; 2] {
    const N: usize = 128;
    const SEED: u64 = 629745;
    const MIN_DIST: u32 = 46;
    fn count(&self) -> u32 {
        self.iter().map(|x| x.count_ones()).sum()
    }
    fn distance(&self, rhs: &Self) -> u32 {
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a.0 ^ b.0).count_ones())
            .sum()
    }
    fn encode_byte(b: u8) -> Self {
        RANDOM_TABLE_128[b as usize]
    }
    fn decode_byte(&self) -> u8 {
        RANDOM_TABLE_128
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.distance(self))
            .unwrap()
            .0 as u8
    }
}

impl ExpandByte for [b64; 4] {
    const N: usize = 256;
    const SEED: u64 = 1902564;
    const MIN_DIST: u32 = 102;
    fn count(&self) -> u32 {
        self.iter().map(|x| x.count_ones()).sum()
    }
    fn distance(&self, rhs: &Self) -> u32 {
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a.0 ^ b.0).count_ones())
            .sum()
    }
    fn encode_byte(b: u8) -> Self {
        RANDOM_TABLE_256[b as usize]
    }
    fn decode_byte(&self) -> u8 {
        RANDOM_TABLE_256
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.distance(self))
            .unwrap()
            .0 as u8
    }
}

lazy_static! {
    pub static ref RANDOM_TABLE_512: [[b64; 8]; 256] = <[b64; 8]>::gen_table();
}

impl ExpandByte for [b64; 8] {
    const N: usize = 256;
    const SEED: u64 = 1446115;
    const MIN_DIST: u32 = 219;
    fn count(&self) -> u32 {
        self.iter().map(|x| x.count_ones()).sum()
    }
    fn distance(&self, rhs: &Self) -> u32 {
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a.0 ^ b.0).count_ones())
            .sum()
    }
    fn encode_byte(b: u8) -> Self {
        RANDOM_TABLE_512[b as usize]
    }
    fn decode_byte(&self) -> u8 {
        RANDOM_TABLE_512
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.distance(self))
            .unwrap()
            .0 as u8
    }
}

#[cfg(test)]
mod test {
    use crate::bits::b64;
    use crate::ecc::ExpandByte;
    use rand::distributions::{Distribution, Standard};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    use std::convert::TryInto;

    fn test_expand_byte<T: ExpandByte>()
    where
        Standard: Distribution<T>,
    {
        let low = ((T::N / 2) as f64 * 0.7) as u32;
        let high = ((T::N / 2) as f64 * 1.3) as u32;
        dbg!(low);
        dbg!(high);
        T::gen_table().iter().for_each(|row| {
            let count = row.count();
            dbg!(count);
            assert!(count > low);
            assert!(count < high);
        });

        let mut rng = Hc128Rng::seed_from_u64(0);
        (0..10000).for_each(|_| {
            let input: u8 = rng.gen::<u8>();
            let encoded = T::encode_byte(input);
            let decoded = encoded.decode_byte();
            assert_eq!(input, decoded);
        });
    }

    #[test]
    fn expand_bytes_64() {
        test_expand_byte::<b64>();
    }
    #[test]
    fn expand_bytes_128() {
        test_expand_byte::<[b64; 2]>();
    }
    #[test]
    fn expand_bytes_256() {
        test_expand_byte::<[b64; 4]>();
    }
}
