use crate::bits::{b32, b64};
use std::num::Wrapping;

#[derive(Copy, Clone, Debug)]
pub struct BitArray64<const L: usize>([u64; L]);

pub trait BitSlice {
    const N: usize;
    fn zeros() -> Self;
    fn ones() -> Self;
    fn splat(sign: bool) -> Self;
    fn xor(self, rhs: Self) -> Self;
    fn or(self, rhs: Self) -> Self;
    fn and(self, rhs: Self) -> Self;
    fn not(self) -> Self;
    fn count_bits(self) -> u32;
}

impl<const L: usize> BitSlice for BitArray64<L> {
    const N: usize = 64 * L;
    fn zeros() -> Self {
        BitArray64([0; L])
    }
    fn ones() -> Self {
        BitArray64([!0; L])
    }
    fn splat(sign: bool) -> Self {
        BitArray64([(Wrapping(0u64) - Wrapping(sign as u64)).0; L])
    }
    fn xor(self, rhs: Self) -> Self {
        let mut target = BitArray64([0; L]);
        for w in 0..L {
            target.0[w] = self.0[w] ^ rhs.0[w];
        }
        target
    }
    fn or(self, rhs: Self) -> Self {
        let mut target = BitArray64([0; L]);
        for w in 0..L {
            target.0[w] = self.0[w] | rhs.0[w];
        }
        target
    }
    fn and(self, rhs: Self) -> Self {
        let mut target = BitArray64([0; L]);
        for w in 0..L {
            target.0[w] = self.0[w] & rhs.0[w];
        }
        target
    }
    fn not(self) -> Self {
        let mut target = BitArray64([0; L]);
        for w in 0..L {
            target.0[w] = !self.0[w];
        }
        target
    }
    fn count_bits(self) -> u32 {
        let mut target = 0u32;
        for w in 0..L {
            target += self.0[w].count_ones();
        }
        target
    }
}

macro_rules! impl_transpose {
    ($fn_name:ident, $b_type:ident, $u_type:ident, $len:expr) => {
        // Hacker's Delight 7-7
        pub fn $fn_name(a: &mut [$b_type; $len]) {
            let mut m: $u_type = !(0 as $u_type) >> $len / 2;
            let mut j: usize = $len / 2;
            while j != 0 {
                let mut k: usize = 0;
                let mut t: $u_type;
                while k < $len {
                    t = (a[k].0 ^ a[k | j].0 >> j) & m;
                    a[k].0 ^= t;
                    a[k | j].0 ^= t << j;
                    k = (k | j) + 1 & !j
                }
                j >>= 1;
                m ^= m << j
            }
        }
    };
}

impl_transpose!(transpose_32, b32, u32, 32);
impl_transpose!(transpose_64, b64, u64, 64);

pub trait BlockTranspose<const L: usize>
where
    Self: BitSlice + Sized,
{
    fn block_transpose(input: &[[b64; L]; Self::N]) -> [Self; 64 * L];
}

impl<const L: usize, const W: usize> BlockTranspose<L> for BitArray64<W> {
    fn block_transpose(input: &[[b64; L]; Self::N]) -> [BitArray64<W>; 64 * L] {
        let mut target = [BitArray64([0; W]); 64 * L];

        for l in 0..L {
            let mut block: [[b64; 64]; W] = [[b64(0); 64]; W];
            for w in 0..W {
                for b in 0..64 {
                    block[w][b] = input[w * 64 + b][l];
                }
                transpose_64(&mut block[w]);
            }
            for b in 0..64 {
                let mut row = [0; W];
                for w in 0..W {
                    row[w] = block[w][b].0;
                }
                target[l * 64 + b] = BitArray64(row);
            }
        }
        target
    }
}

fn half_comparator<T: BitSlice + Copy>(a: T, b: T) -> (T, T, T) {
    let lt = a.not().and(b);
    let gt = a.and(b.not());
    let eq = lt.or(gt).not();
    (lt, eq, gt)
}

fn full_comparator<T: BitSlice + Copy>(a: T, b: T, c: (T, T, T)) -> (T, T, T) {
    let x = half_comparator(a, b);
    let lt = x.0.or(x.2.not().and(c.0));
    let gt = x.2.or(x.0.not().and(c.2));
    (lt, lt.or(gt).not(), gt)
}

fn half_adder<T: BitSlice + Copy>(a: T, b: T) -> (T, T) {
    (a.xor(b), a.and(b))
}

fn full_adder<T: BitSlice + Copy>(a: T, b: T, c: T) -> (T, T) {
    let u = a.xor(b);
    (u.xor(c), a.and(b).or(u.and(c)))
}

pub fn equality<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> T {
    let mut acc = a[0].and(b[0]);
    for i in 1..L {
        acc = acc.and(a[i].xor(b[i]).not());
    }
    acc
}

pub fn comparator<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> (T, T, T) {
    let mut acc = half_comparator(a[0], b[0]);
    for i in 1..L {
        acc = full_comparator(a[i], b[i], acc);
    }
    acc
}

pub fn bit_add<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> [T; L + 1] {
    let mut acc = [T::zeros(); L + 1];
    let (zero, c) = half_adder(a[0], b[0]);
    acc[0] = zero;
    let mut carry = c;
    for i in 1..L {
        let (bit, c) = full_adder(a[i], b[i], carry);
        acc[i] = bit;
        carry = c;
    }
    acc[L] = carry;
    acc
}

pub fn bit_add_wrapping<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> [T; L] {
    let mut acc = [T::zeros(); L];
    let (zero, c) = half_adder(a[0], b[0]);
    acc[0] = zero;
    let mut carry = c;
    for i in 1..L {
        let (bit, c) = full_adder(a[i], b[i], carry);
        acc[i] = bit;
        carry = c;
    }
    acc
}

pub fn extend<T: BitSlice + Copy, const I: usize, const O: usize>(v: &[T; I]) -> [T; O] {
    let mut target = [T::zeros(); O];
    for i in 0..I {
        target[i] = v[i];
    }
    target
}

pub fn bit_splat<T: BitSlice + Copy, const L: usize>(value: u32) -> [T; L] {
    let mut target = [T::zeros(); L];
    for i in 0..L {
        let sign = (value >> i) & 1 == 1;
        target[i] = T::splat(sign);
    }
    target
}
