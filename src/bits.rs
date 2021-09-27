/// the bits mod contains traits to manipulate words of bits
/// and arrays of bits.
use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::Wrapping;

pub trait BitArray
where
    Self: Sized,
{
    fn set_bit_in_place(&mut self, i: usize, s: bool);
    fn set_bit(mut self, i: usize, s: bool) -> Self {
        self.set_bit_in_place(i, s);
        self
    }
    fn get_bit(&self, i: usize) -> bool;
}

pub trait FromBool {
    fn from_bool(b: bool) -> Self;
    fn from_u8(b: u8) -> Self;
    fn to_u64(self) -> u64;
    const ONE: Self;
    const MAX: usize;
}

macro_rules! for_uints {
    ($b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        impl FromBool for $u_type {
            #[inline(always)]
            fn from_bool(b: bool) -> $u_type {
                b as $u_type
            }
            #[inline(always)]
            fn from_u8(b: u8) -> $u_type {
                b as $u_type
            }
            fn to_u64(self) -> u64 {
                self as u64
            }
            const ONE: $u_type = 1;
            const MAX: usize = <$u_type>::MAX as usize;
        }

        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Serialize, Deserialize)]
        pub struct $b_type(pub $u_type);

        impl $b_type {
            pub const ZEROS: $b_type = $b_type(0);
            #[inline(always)]
            pub fn count_ones(self) -> u32 {
                self.0.count_ones()
            }
            #[inline(always)]
            pub fn get_bit(self, index: usize) -> bool {
                //dbg!(index);
                ((self.0 >> (($len - 1) - index)) & 1) == 1
            }
            #[inline(always)]
            pub fn get_bit_u8(self, index: usize) -> u8 {
                ((self.0 >> (($len - 1) - index)) & 1) as u8
            }
            #[inline(always)]
            pub fn set_bit_in_place(&mut self, index: usize, value: bool) {
                self.0 &= !(1 << (($len - 1) - index));
                self.0 |= ((value as $u_type) << (($len - 1) - index));
            }
            #[inline(always)]
            pub fn set_bit(mut self, index: usize, value: bool) -> Self {
                self.0 &= !(1 << (($len - 1) - index));
                self.0 |= ((value as $u_type) << (($len - 1) - index));
                self
            }
        }

        impl<const L: usize> BitArray for [$b_type; L] {
            fn set_bit_in_place(&mut self, i: usize, s: bool) {
                self[i / $len].set_bit_in_place(i % $len, s);
            }
            fn get_bit(&self, i: usize) -> bool {
                self[i / $len].get_bit(i % $len)
            }
        }

        impl Default for $b_type {
            fn default() -> Self {
                $b_type(0)
            }
        }

        impl fmt::Debug for $b_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, $format_string, self.0)
            }
        }

        impl fmt::Display for $b_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, $format_string, self.0)
            }
        }

        impl PartialEq for $b_type {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl Eq for $b_type {}
    };
}

for_uints!(b8, u8, 8, "{:08b}");
for_uints!(b16, u16, 16, "{:016b}");
for_uints!(b32, u32, 32, "{:032b}");
for_uints!(b64, u64, 64, "{:064b}");
for_uints!(b128, u128, 128, "{:0128b}");

pub trait GetBit {
    fn bit(self, i: usize) -> bool;
}

impl GetBit for usize {
    #[inline(always)]
    fn bit(self, i: usize) -> bool {
        ((self >> i) & 1) == 1
    }
}
impl GetBit for u8 {
    #[inline(always)]
    fn bit(self, i: usize) -> bool {
        ((self >> i) & 1) == 1
    }
}

#[cfg(test)]
mod tests {
    use super::{b128, b16, b32, b8, SIMDincrementCounters};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;

    #[test]
    fn simd_expand_bits() {
        let mut rng = Hc128Rng::seed_from_u64(0);

        //let mut simd_counter = <[(); 32] as SIMDincrementCounters>::init_counters();

        (0..10_000).for_each(|_| {
            let word: b32 = rng.gen();
            let mut test_counter = [0u32; 32];
            for i in 0..32 {
                test_counter[i] += word.get_bit_u8(i) as u32;
            }
            let mut simd_counter = [0u32; 32];

            let expanded = <[(); 32] as SIMDincrementCounters>::expand_bits(&word);
            <[(); 32] as SIMDincrementCounters>::add_to_u32s(&expanded, &mut simd_counter);
            assert_eq!(simd_counter, test_counter);
        });
    }
}
