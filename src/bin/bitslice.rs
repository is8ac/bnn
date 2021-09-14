#![feature(int_log)]
#![feature(avx512_target_feature)]
#![feature(stdsimd)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]
use std::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_extract_epi64, _mm256_or_si256, _mm256_set1_epi8,
    _mm256_setzero_si256, _mm256_xor_si256,
};
use std::arch::x86_64::{
    __m512i, _mm512_and_si512, _mm512_or_si512, _mm512_set1_epi8, _mm512_setzero_si512,
    _mm512_xor_si512,
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

trait BitSlice {
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
    fn count_bits(self) -> u32 {
        unsafe { avx512_count_ones(self) }
    }
}

macro_rules! impl_transpose {
    ($fn_name:ident, $type:ty, $len:expr) => {
        // Hacker's Delight 7-7
        fn $fn_name(a: &mut [$type; $len]) {
            let mut m: $type = !(0 as $type) >> $len / 2;
            let mut j: usize = $len / 2;
            while j != 0 {
                let mut k: usize = 0;
                let mut t: $type;
                while k < $len {
                    t = (a[k] ^ a[k | j] >> j) & m;
                    a[k] ^= t;
                    a[k | j] ^= t << j;
                    k = (k | j) + 1 & !j
                }
                j >>= 1;
                m ^= m << j
            }
        }
        impl BitSlice for $type {
            fn zeros() -> Self {
                0
            }
            fn ones() -> Self {
                !(0)
            }
            fn splat(sign: bool) -> Self {
                0 - sign as $type
            }
            fn xor(self, rhs: Self) -> Self {
                self ^ rhs
            }
            fn or(self, rhs: Self) -> Self {
                self | rhs
            }
            fn and(self, rhs: Self) -> Self {
                self & rhs
            }
            fn not(self) -> Self {
                !self
            }
            fn count_bits(self) -> u32 {
                self.count_ones()
            }
        }
    };
}

impl_transpose!(transpose_8, u8, 8);
impl_transpose!(transpose_16, u16, 16);
impl_transpose!(transpose_32, u32, 32);
impl_transpose!(transpose_64, u64, 64);
impl_transpose!(transpose_128, u128, 128);

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

fn equality<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> T {
    let mut acc = a[0].and(b[0]);
    for i in 1..L {
        acc = acc.and(a[i].xor(b[i]).not());
    }
    acc
}

fn comparator<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> (T, T, T) {
    let mut acc = half_comparator(a[0], b[0]);
    for i in 1..L {
        acc = full_comparator(a[i], b[i], acc);
    }
    acc
}

fn bit_add<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> [T; { L + 1 }] {
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

fn bit_add_wrapping<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> [T; L] {
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

trait ArrayPopcount<const L: u32>
where
    Self: Sized,
{
    fn array_popcount(v: &[Self; 2usize.pow(L)]) -> [Self; L as usize + 1];
}

impl<T: BitSlice + Copy> ArrayPopcount<1> for T {
    fn array_popcount(v: &[T; 2]) -> [T; 2] {
        let (a, b) = half_adder(v[0], v[1]);
        [a, b]
    }
}

macro_rules! impl_array_popcount {
    ($len:expr, $len_sub1:expr) => {
        impl<T: BitSlice + Copy> ArrayPopcount<$len> for T {
            fn array_popcount(v: &[T; 2usize.pow($len)]) -> [T; $len + 1] {
                let a = <T as ArrayPopcount<$len_sub1>>::array_popcount(
                    <&[T; 2usize.pow($len - 1)]>::try_from(&v[0..2usize.pow($len - 1)]).unwrap(),
                );
                let b = <T as ArrayPopcount<$len_sub1>>::array_popcount(
                    <&[T; 2usize.pow($len - 1)]>::try_from(
                        &v[2usize.pow($len - 1)..2usize.pow($len)],
                    )
                    .unwrap(),
                );
                bit_add::<T, $len>(&a, &b)
            }
        }
    };
}

impl_array_popcount!(2, 1);
impl_array_popcount!(3, 2);
impl_array_popcount!(4, 3);
impl_array_popcount!(5, 4);
impl_array_popcount!(6, 5);
//impl_array_popcount!(7, 6);
//impl_array_popcount!(8, 7);

fn extend<T: BitSlice + Copy, const I: usize, const O: usize>(v: &[T; I]) -> [T; O] {
    let mut target = [T::zeros(); O];
    for i in 0..I {
        target[i] = v[i];
    }
    target
}

fn ragged_array_popcount<T: BitSlice + Copy, const L: usize>(v: &[T]) -> [T; L] {
    assert!(v.len() < 2usize.pow(L as u32));
    dbg!(v.len());
    let size: u32 = v.len().log2();
    dbg!(size);
    let head = match size {
        0 => extend::<T, 1, L>(&[v[0]]),
        1 => extend(&<T as ArrayPopcount<1>>::array_popcount(
            <&[T; 2]>::try_from(&v[0..2]).unwrap(),
        )),
        2 => extend(&<T as ArrayPopcount<2>>::array_popcount(
            <&[T; 4]>::try_from(&v[0..4]).unwrap(),
        )),
        3 => extend(&<T as ArrayPopcount<3>>::array_popcount(
            <&[T; 8]>::try_from(&v[0..8]).unwrap(),
        )),
        4 => extend(&<T as ArrayPopcount<4>>::array_popcount(
            <&[T; 16]>::try_from(&v[0..16]).unwrap(),
        )),
        5 => extend(&<T as ArrayPopcount<5>>::array_popcount(
            <&[T; 32]>::try_from(&v[0..32]).unwrap(),
        )),
        6 => extend(&<T as ArrayPopcount<6>>::array_popcount(
            <&[T; 64]>::try_from(&v[0..64]).unwrap(),
        )),
        _ => panic!(),
    };
    if 2usize.pow(size) == v.len() {
        head
    } else {
        let tail = ragged_array_popcount::<T, L>(&v[(2usize.pow(size) as usize)..]);
        bit_add_wrapping(&head, &tail)
    }
}

fn bit_splat<T: BitSlice + Copy, const L: usize>(value: u32) -> [T; L] {
    let mut target = [T::zeros(); L];
    for i in 0..L {
        let sign = (value >> i) & 1 == 1;
        target[i] = T::splat(sign);
    }
    target
}

type S = __m512i;
pub fn foo(bits: &[S; 16], thresholds: &[u32; 8]) -> [u64; 8] {
    let mut expanded_thresholds = [[S::zeros(); 5]; 8];
    for i in 0..8 {
        expanded_thresholds[i] = bit_splat(thresholds[i]);
    }
    let count = <S as ArrayPopcount<4>>::array_popcount(bits);
    let mut target = [0u64; 8];
    for i in 0..8 {
        let (lt, _, _) = comparator(&count, &expanded_thresholds[i]);
        target[i] = lt.count_bits() as u64;
    }
    target
}

fn main() {
    let mut bits_a: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    transpose_8(&mut bits_a);
    bits_a.reverse();
    let mut bits_b: [u8; 8] = [3; 8];
    transpose_8(&mut bits_b);
    bits_b.reverse();
    let mut sum = bit_add_wrapping(&bits_a, &bits_b);
    sum.reverse();
    transpose_8(&mut sum);
    dbg!(sum);
    for i in 0..8 {
        println!("{:08b}", bits_a[i]);
    }
    dbg!();
    for i in 0..8 {
        println!("{:08b}", bits_b[i]);
    }
    dbg!();
    let (lt, eq, gt) = comparator(&bits_a, &bits_b);
    println!("{:08b}", lt);
    println!("{:08b}", eq);
    println!("{:08b}", gt);
    let eq = equality(&bits_a, &bits_b);
    println!("{:08b}", eq);

    let bits = vec![0b_00001111_u8];
    let count = ragged_array_popcount::<u8, 7>(&bits);
    for i in 0..7 {
        println!("{:08b}", count[i]);
    }
}
