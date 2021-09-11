use crate::bits::{b32, b64, b8, GetBit, PackedIndexSGet, BMA};
use crate::shape::Shape;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::convert::TryInto;
use std::num::Wrapping;
use std::sync::atomic::{AtomicU32, Ordering};

pub fn generate_hadamard_matrix() -> [[b64; 4]; 256] {
    let mut generator_matrix = [[b64(0); 4]; 8];
    for i in 0..256 {
        for b in 0..8 {
            generator_matrix[b].set_bit(i, i.bit(b));
        }
    }

    let mut hadamard_matrix = [[b64(0); 4]; 256];
    for i in 0..256usize {
        for b in 0..8 {
            if i.bit(b) {
                hadamard_matrix[i].xor_in_place(&generator_matrix[b]);
            }
        }
    }
    hadamard_matrix
}

pub trait BitString {
    fn xor_in_place(&mut self, rhs: &Self);
    fn set_bit(&mut self, i: usize, b: bool);
    fn hamming_dist(&self, rhs: &Self) -> u32;
}

impl<const L: usize> BitString for [b64; L] {
    fn xor_in_place(&mut self, rhs: &Self) {
        let mut target = [b64(0); L];
        for i in 0..L {
            self[i].0 ^= rhs[i].0;
        }
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
}

lazy_static! {
    static ref HADAMARD_MATRIX_256: [[b64; 4]; 256] = generate_hadamard_matrix();
}

pub fn encode_byte(b: u8) -> [b64; 4] {
    HADAMARD_MATRIX_256[b as usize]
}

pub fn decode_byte(bits: &[b64; 4]) -> u8 {
    HADAMARD_MATRIX_256
        .iter()
        .enumerate()
        .min_by_key(|(_, row)| row.hamming_dist(bits))
        .unwrap()
        .0 as u8
}
