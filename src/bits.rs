use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt;

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

macro_rules! for_uints {
    ($b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
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

        impl Distribution<$b_type> for Standard {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $b_type {
                $b_type(rng.gen())
            }
        }
    };
}

for_uints!(b32, u32, 32, "{:032b}");
for_uints!(b64, u64, 64, "{:064b}");

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
