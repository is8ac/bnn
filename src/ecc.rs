use crate::bits::{b32, b64, GetBit, PackedIndexSGet, BMA};
use crate::shape::Shape;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::convert::TryInto;
use std::num::Wrapping;
use std::sync::atomic::{AtomicU32, Ordering};

pub trait BitString {
    const WIDTH: usize;
    fn new_from_int(i: usize) -> Self;
    fn get_bit(&self, i: usize) -> bool;
    fn set_bit(&mut self, i: usize, b: bool);
    fn hamming_dist(&self, rhs: &Self) -> u32;
    fn masked_hamming_dist(&self, rhs: &Self, mask: &Self) -> u32;
    fn gen_mask(i: usize) -> Self;
    fn print_bits(&self);
}

impl<const L: usize> BitString for [b64; L] {
    const WIDTH: usize = 64 * L;
    fn new_from_int(i: usize) -> Self {
        let mut target = [b64(0); L];
        target[0].0 = i as u64;
        target
    }
    fn get_bit(&self, i: usize) -> bool {
        ((self[i / 64].0 >> (i % 64)) & 1) == 1
    }
    fn set_bit(&mut self, i: usize, b: bool) {
        self[i / 64].0 |= (b as u64) << (i % 64);
    }
    fn hamming_dist(&self, rhs: &Self) -> u32 {
        self.iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a.0 ^ b.0).count_ones())
            .sum()
    }
    fn masked_hamming_dist(&self, rhs: &Self, mask: &Self) -> u32 {
        self.iter()
            .zip(rhs.iter())
            .zip(mask.iter())
            .map(|((a, b), m)| ((a.0 ^ b.0) & m.0).count_ones())
            .sum()
    }
    fn gen_mask(i: usize) -> Self {
        let tw = i / 64;
        (0..L)
            .map(|w| {
                if w < tw {
                    b64(!0)
                } else if w > tw {
                    b64(0)
                } else {
                    b64(!((!0) << (i % 64)))
                }
            })
            .collect::<Vec<b64>>()
            .try_into()
            .unwrap()
    }
    fn print_bits(&self) {
        for i in 0..L {
            print!("{:064b}", self[i].0);
        }
        print!("\n",);
    }
}

trait EccTable {
    fn gen_table() -> Self;
    fn min_dist(&self) -> u32;
}

impl<S: BitString + std::fmt::Debug, const L: usize> EccTable for [S; L] {
    fn gen_table() -> [S; L] {
        let mut table: [S; L] = (0..L)
            .map(|i| S::new_from_int(i))
            .collect::<Vec<S>>()
            .try_into()
            .unwrap();
        for b in 0..S::WIDTH {
            let mask = S::gen_mask(b);
            for w in 0..L {
                let index = table
                    .iter()
                    .enumerate()
                    .filter(|&(w2, _)| w2 != w)
                    .min_by_key(|&(_, word)| word.masked_hamming_dist(&table[w], &mask))
                    .unwrap()
                    .0;
                table[w].set_bit(b, !table[index].get_bit(b));
            }
        }
        table
    }
    fn min_dist(&self) -> u32 {
        (0..L)
            .map(|x| {
                (0..L)
                    .filter(|&y| y != x)
                    .map(|y| self[x].hamming_dist(&self[y]))
                    .min()
                    .unwrap()
            })
            .min()
            .unwrap()
    }
}

lazy_static! {
    static ref RANDOM_TABLE_64: [[b64; 1]; 256] = <[[b64; 1]; 256]>::gen_table();
}

lazy_static! {
    static ref RANDOM_TABLE_128: [[b64; 2]; 256] = <[[b64; 2]; 256]>::gen_table();
}

lazy_static! {
    static ref RANDOM_TABLE_256: [[b64; 4]; 256] = <[[b64; 4]; 256]>::gen_table();
}

lazy_static! {
    static ref RANDOM_TABLE_512: [[b64; 8]; 256] = <[[b64; 8]; 256]>::gen_table();
}

pub trait ExpandByte: BitString {
    fn encode_byte(b: u8) -> Self;
    fn decode_byte(&self) -> u8;
}

impl ExpandByte for [b64; 1] {
    fn encode_byte(b: u8) -> Self {
        RANDOM_TABLE_64[b as usize]
    }
    fn decode_byte(&self) -> u8 {
        RANDOM_TABLE_64
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.hamming_dist(self))
            .unwrap()
            .0 as u8
    }
}

impl ExpandByte for [b64; 2] {
    fn encode_byte(b: u8) -> Self {
        RANDOM_TABLE_128[b as usize]
    }
    fn decode_byte(&self) -> u8 {
        RANDOM_TABLE_128
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.hamming_dist(self))
            .unwrap()
            .0 as u8
    }
}

impl ExpandByte for [b64; 4] {
    fn encode_byte(b: u8) -> Self {
        RANDOM_TABLE_256[b as usize]
    }
    fn decode_byte(&self) -> u8 {
        RANDOM_TABLE_256
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.hamming_dist(self))
            .unwrap()
            .0 as u8
    }
}

impl ExpandByte for [b64; 8] {
    fn encode_byte(b: u8) -> Self {
        RANDOM_TABLE_512[b as usize]
    }
    fn decode_byte(&self) -> u8 {
        RANDOM_TABLE_512
            .iter()
            .enumerate()
            .min_by_key(|(_, row)| row.hamming_dist(self))
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
