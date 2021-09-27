use crate::bits::{b128, b16, b32, b64, b8};
use std::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_extract_epi64, _mm256_or_si256, _mm256_set1_epi8,
    _mm256_setzero_si256, _mm256_xor_si256,
};
use std::arch::x86_64::{
    __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64,
    _mm512_set1_epi8, _mm512_setzero_si512, _mm512_xor_si512,
};
use std::convert::TryFrom;
use std::fmt;
use std::mem;

#[target_feature(enable = "avx2")]
unsafe fn avx2_zeros() -> __m256i {
    _mm256_setzero_si256()
}
#[target_feature(enable = "avx2")]
unsafe fn avx2_ones() -> __m256i {
    _mm256_set1_epi8(-1)
}
#[target_feature(enable = "avx2")]
unsafe fn avx2_splat(sign: bool) -> __m256i {
    _mm256_set1_epi8(0 - sign as i8)
}
#[target_feature(enable = "avx2")]
unsafe fn avx2_xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
}
#[target_feature(enable = "avx2")]
unsafe fn avx2_or(a: __m256i, b: __m256i) -> __m256i {
    _mm256_or_si256(a, b)
}
#[target_feature(enable = "avx2")]
unsafe fn avx2_and(a: __m256i, b: __m256i) -> __m256i {
    _mm256_and_si256(a, b)
}
#[target_feature(enable = "avx2")]
unsafe fn avx2_not(a: __m256i) -> __m256i {
    _mm256_xor_si256(a, _mm256_set1_epi8(-1))
}
#[target_feature(enable = "avx2")]
unsafe fn avx2_count_ones(a: __m256i) -> u32 {
    _mm256_extract_epi64(a, 0).count_ones()
        + _mm256_extract_epi64(a, 1).count_ones()
        + _mm256_extract_epi64(a, 2).count_ones()
        + _mm256_extract_epi64(a, 3).count_ones()
}

#[target_feature(enable = "avx512f")]
unsafe fn avx512_zeros() -> __m512i {
    _mm512_setzero_si512()
}
#[target_feature(enable = "avx512f")]
unsafe fn avx512_ones() -> __m512i {
    _mm512_set1_epi8(-1)
}
#[target_feature(enable = "avx512f")]
unsafe fn avx512_splat(sign: bool) -> __m512i {
    _mm512_set1_epi8(0 - sign as i8)
}
#[target_feature(enable = "avx512f")]
unsafe fn avx512_xor(a: __m512i, b: __m512i) -> __m512i {
    _mm512_xor_si512(a, b)
}
#[target_feature(enable = "avx512f")]
unsafe fn avx512_or(a: __m512i, b: __m512i) -> __m512i {
    _mm512_or_si512(a, b)
}
#[target_feature(enable = "avx512f")]
unsafe fn avx512_and(a: __m512i, b: __m512i) -> __m512i {
    _mm512_and_si512(a, b)
}
#[target_feature(enable = "avx512f")]
unsafe fn avx512_not(a: __m512i) -> __m512i {
    _mm512_xor_si512(a, _mm512_set1_epi8(-1))
}
#[target_feature(enable = "avx512f")]
unsafe fn avx512_count_ones(a: __m512i) -> u32 {
    let foo: [u64; 8] = mem::transmute(a);
    let mut target = 0u32;
    for i in 0..8 {
        target += foo[i].count_ones();
    }
    target
}

#[target_feature(enable = "avx512vpopcntdq")]
unsafe fn avx512popcnt_count_ones(a: __m512i) -> u32 {
    let counts = _mm512_popcnt_epi64(a);
    let sum: u64 = mem::transmute(_mm512_reduce_add_epi64(counts));
    sum as u32
}

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

impl BitSlice for __m256i {
    const N: usize = 256;
    fn zeros() -> Self {
        unsafe { avx2_zeros() }
    }
    fn ones() -> Self {
        unsafe { avx2_ones() }
    }
    fn splat(sign: bool) -> Self {
        unsafe { avx2_splat(sign) }
    }
    fn xor(self, rhs: Self) -> Self {
        unsafe { avx2_xor(self, rhs) }
    }
    fn or(self, rhs: Self) -> Self {
        unsafe { avx2_or(self, rhs) }
    }
    fn and(self, rhs: Self) -> Self {
        unsafe { avx2_and(self, rhs) }
    }
    fn not(self) -> Self {
        unsafe { avx2_not(self) }
    }
    fn count_bits(self) -> u32 {
        unsafe { avx2_count_ones(self) }
    }
}

impl BitSlice for __m512i {
    const N: usize = 512;
    fn zeros() -> Self {
        unsafe { avx512_zeros() }
    }
    fn ones() -> Self {
        unsafe { avx512_ones() }
    }
    fn splat(sign: bool) -> Self {
        unsafe { avx512_splat(sign) }
    }
    fn xor(self, rhs: Self) -> Self {
        unsafe { avx512_xor(self, rhs) }
    }
    fn or(self, rhs: Self) -> Self {
        unsafe { avx512_or(self, rhs) }
    }
    fn and(self, rhs: Self) -> Self {
        unsafe { avx512_and(self, rhs) }
    }
    fn not(self) -> Self {
        unsafe { avx512_not(self) }
    }
    #[cfg(not(target_feature = "avx512vpopcntdq"))]
    fn count_bits(self) -> u32 {
        unsafe { avx512_count_ones(self) }
    }
    #[cfg(target_feature = "avx512vpopcntdq")]
    fn count_bits(self) -> u32 {
        unsafe { avx512popcnt_count_ones(self) }
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
        impl BitSlice for $b_type {
            const N: usize = $len;
            fn zeros() -> Self {
                $b_type(0)
            }
            fn ones() -> Self {
                $b_type(!(0))
            }
            fn splat(sign: bool) -> Self {
                $b_type(0 - sign as $u_type)
            }
            fn xor(self, rhs: Self) -> Self {
                $b_type(self.0 ^ rhs.0)
            }
            fn or(self, rhs: Self) -> Self {
                $b_type(self.0 | rhs.0)
            }
            fn and(self, rhs: Self) -> Self {
                $b_type(self.0 & rhs.0)
            }
            fn not(self) -> Self {
                $b_type(!self.0)
            }
            fn count_bits(self) -> u32 {
                self.0.count_ones()
            }
        }
    };
}

impl_transpose!(transpose_8, b8, u8, 8);
impl_transpose!(transpose_16, b16, u16, 16);
impl_transpose!(transpose_32, b32, u32, 32);
impl_transpose!(transpose_64, b64, u64, 64);
impl_transpose!(transpose_128, b128, u128, 128);

pub trait BlockTranspose<const L: usize>
where
    Self: BitSlice + Sized,
{
    fn block_transpose(input: &[[b64; L]; Self::N]) -> [Self; 64 * L];
}

impl<const L: usize> BlockTranspose<L> for b64 {
    fn block_transpose(input: &[[b64; L]; 64]) -> [b64; 64 * L] {
        let mut target = [b64(0); 64 * L];

        for l in 0..L {
            let mut block: [b64; 64] = [b64(0); 64];
            for b in 0..64 {
                block[b] = input[b][l];
            }
            transpose_64(&mut block);
            for b in 0..64 {
                target[l * 64 + b] = block[b];
            }
        }
        target
    }
}

impl<const L: usize> BlockTranspose<L> for __m256i {
    fn block_transpose(input: &[[b64; L]; Self::N]) -> [__m256i; 64 * L] {
        let mut target = [unsafe { _mm256_setzero_si256() }; 64 * L];

        for l in 0..L {
            let mut block: [[b64; 64]; 4] = [[b64(0); 64]; 4];
            for w in 0..4 {
                for b in 0..64 {
                    block[w][b] = input[w * 64 + b][l];
                }
                transpose_64(&mut block[w]);
            }
            for b in 0..64 {
                let mut row = [b64(0); 4];
                for w in 0..4 {
                    row[w] = block[w][b];
                }
                target[l * 64 + b] = unsafe { mem::transmute(row) };
            }
        }
        target
    }
}

impl<const L: usize> BlockTranspose<L> for __m512i {
    fn block_transpose(input: &[[b64; L]; 512]) -> [__m512i; 64 * L] {
        let mut target = [unsafe { _mm512_setzero_si512() }; 64 * L];

        for l in 0..L {
            let mut block: [[b64; 64]; 8] = [[b64(0); 64]; 8];
            for w in 0..8 {
                for b in 0..64 {
                    block[w][b] = input[w * 64 + b][l];
                }
                transpose_64(&mut block[w]);
            }
            for b in 0..64 {
                let mut row = [b64(0); 8];
                for w in 0..8 {
                    row[w] = block[w][b];
                }
                target[l * 64 + b] = unsafe { mem::transmute(row) };
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

fn array_popcount_1<T: BitSlice + Copy>(v: &[T; 2]) -> [T; 2] {
    let (a, b) = half_adder(v[0], v[1]);
    [a, b]
}

macro_rules! adder_fns {
    ($len:expr, $array_add_fn:ident, $sub_array_add_fn:ident) => {
        fn $array_add_fn<T: BitSlice + Copy>(v: &[T; 2usize.pow($len)]) -> [T; $len + 1] {
            let a = $sub_array_add_fn(
                <&[T; 2usize.pow($len - 1)]>::try_from(&v[0..2usize.pow($len - 1)]).unwrap(),
            );
            let b = $sub_array_add_fn(
                <&[T; 2usize.pow($len - 1)]>::try_from(&v[2usize.pow($len - 1)..2usize.pow($len)])
                    .unwrap(),
            );

            let mut acc = [T::zeros(); $len + 1];
            let (zero, c) = half_adder(a[0], b[0]);
            acc[0] = zero;
            let mut carry = c;
            for i in 1..$len {
                let (bit, c) = full_adder(a[i], b[i], carry);
                acc[i] = bit;
                carry = c;
            }
            acc[$len] = carry;
            acc
        }
    };
}

adder_fns!(2, array_popcount_2, array_popcount_1);
adder_fns!(3, array_popcount_3, array_popcount_2);
adder_fns!(4, array_popcount_4, array_popcount_3);
adder_fns!(5, array_popcount_5, array_popcount_4);
adder_fns!(6, array_popcount_6, array_popcount_5);
adder_fns!(7, array_popcount_7, array_popcount_6);
adder_fns!(8, array_popcount_8, array_popcount_7);
adder_fns!(9, array_popcount_9, array_popcount_8);

pub fn extend<T: BitSlice + Copy, const I: usize, const O: usize>(v: &[T; I]) -> [T; O] {
    let mut target = [T::zeros(); O];
    for i in 0..I {
        target[i] = v[i];
    }
    target
}

pub fn ragged_array_popcount<T: BitSlice + Copy, const L: usize>(v: &[T]) -> [T; L] {
    assert!(v.len() < 2usize.pow(L as u32));
    let size: u32 = v.len().log2();
    let head = match size {
        0 => extend::<T, 1, L>(&[v[0]]),
        1 => extend(&array_popcount_1(<&[T; 2]>::try_from(&v[0..2]).unwrap())),
        2 => extend(&array_popcount_2(<&[T; 4]>::try_from(&v[0..4]).unwrap())),
        3 => extend(&array_popcount_3(<&[T; 8]>::try_from(&v[0..8]).unwrap())),
        4 => extend(&array_popcount_4(<&[T; 16]>::try_from(&v[0..16]).unwrap())),
        5 => extend(&array_popcount_5(<&[T; 32]>::try_from(&v[0..32]).unwrap())),
        6 => extend(&array_popcount_6(<&[T; 64]>::try_from(&v[0..64]).unwrap())),
        7 => extend(&array_popcount_7(
            <&[T; 128]>::try_from(&v[0..128]).unwrap(),
        )),
        8 => extend(&array_popcount_8(
            <&[T; 256]>::try_from(&v[0..256]).unwrap(),
        )),
        9 => extend(&array_popcount_9(
            <&[T; 512]>::try_from(&v[0..512]).unwrap(),
        )),
        _ => panic!("Size not implemented"),
    };
    if 2usize.pow(size) == v.len() {
        head
    } else {
        let tail = ragged_array_popcount::<T, L>(&v[(2usize.pow(size) as usize)..]);
        bit_add_wrapping(&head, &tail)
    }
}

pub fn bit_splat<T: BitSlice + Copy, const L: usize>(value: u32) -> [T; L] {
    let mut target = [T::zeros(); L];
    for i in 0..L {
        let sign = (value >> i) & 1 == 1;
        target[i] = T::splat(sign);
    }
    target
}
