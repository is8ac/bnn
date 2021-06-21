use bnn::bits::{b32, BitPack, IncrementCounters};
use bnn::shape::Pack;
use core::arch::x86_64::{
    __m256i, _mm256_add_epi8, _mm256_and_si256, _mm256_andnot_si256, _mm256_cmpeq_epi8,
    _mm256_loadu_si256, _mm256_set1_epi32, _mm256_set1_epi8, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_storeu_si256,
};
use rayon::prelude::*;
use std::mem::{transmute, MaybeUninit};
use std::time::Instant;

const mask1a: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
];

const mask2a: [u8; 32] = [
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 0,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
];

fn count_batch_safe(bits: &[b32]) -> [u8; 32] {
    bits.iter().fold([0u8; 32], |mut acc, word| {
        <[(); 32]>::increment_in_place(&b32(word.0), &mut acc);
        acc
    })
}

fn count_batch_simd(bits: &[b32]) -> [u8; 32] {
    assert!(bits.len() < 256);
    let mut target = [0u8; 32];
    unsafe {
        let mut acc = _mm256_setzero_si256();
        bits.iter().for_each(|&word| {
            let mask1 = _mm256_loadu_si256(transmute::<&[u8; 32], &__m256i>(&mask1a));
            let mask2 = _mm256_loadu_si256(transmute::<&[u8; 32], &__m256i>(&mask2a));

            let zeros = _mm256_setzero_si256();
            let ones = _mm256_set1_epi8(0b_1);

            let word: i32 = transmute::<u32, i32>(word.0);

            let expanded = _mm256_set1_epi32(word);
            let shuffled = _mm256_shuffle_epi8(expanded, mask1);
            let masked = _mm256_and_si256(shuffled, mask2);
            let is_zero = _mm256_cmpeq_epi8(masked, zeros);
            let bools = _mm256_andnot_si256(is_zero, ones);
            acc = _mm256_add_epi8(acc, bools);
        });
        _mm256_storeu_si256(transmute::<&mut [u8; 32], &mut __m256i>(&mut target), acc);
    }
    target
}

fn add_u8s(acc: &mut [u64; 32], u8s: [u8; 32]) {
    for i in 0..32 {
        acc[i] += u8s[i] as u64;
    }
}

trait SIMDincrementCounters
where
    Self: Pack<u8> + BitPack<bool>,
    Self::SIMDbyts: Sized,
{
    type SIMDbyts;
    /// This function must not be called more then 256 times!!
    fn simd_increment_in_place(bools: &Self::SIMDbyts, counters: &mut Self::SIMDbyts);
    fn expand_bits(bits: &<Self as BitPack<bool>>::T) -> Self::SIMDbyts;
    fn init_counters() -> Self::SIMDbyts;
}

impl<T: SIMDincrementCounters, const L: usize> SIMDincrementCounters for [T; L]
where
    T::SIMDbyts: Sized,
{
    type SIMDbyts = [T::SIMDbyts; L];
    #[inline(always)]
    fn simd_increment_in_place(bits: &[T::SIMDbyts; L], counters: &mut [T::SIMDbyts; L]) {
        for i in 0..L {
            T::simd_increment_in_place(&bits[i], &mut counters[i]);
        }
    }
    fn expand_bits(bits: &<Self as BitPack<bool>>::T) -> Self::SIMDbyts {
        let mut target: [MaybeUninit<T::SIMDbyts>; L] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..L {
            target[i] = MaybeUninit::new(T::expand_bits(&bits[i]));
        }
        unsafe { target.as_ptr().cast::<[T::SIMDbyts; L]>().read() }
    }
    fn init_counters() -> [T::SIMDbyts; L] {
        let mut target: [MaybeUninit<T::SIMDbyts>; L] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..L {
            target[i] = MaybeUninit::new(T::init_counters());
        }
        unsafe { target.as_ptr().cast::<[T::SIMDbyts; L]>().read() }
    }
}

impl SIMDincrementCounters for [(); 32] {
    #[cfg(target_feature = "avx2")]
    type SIMDbyts = __m256i;
    #[cfg(not(target_feature = "avx2"))]
    type SIMDbyts = [u8; 32];
    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn simd_increment_in_place(word: &__m256i, counters: &mut __m256i) {
        unsafe {
            *counters = _mm256_add_epi8(*counters, *word);
        }
    }
    #[cfg(not(target_feature = "avx2"))]
    #[inline(always)]
    fn simd_increment_in_place(word: &[u8; 32], counters: &mut [u8; 32]) {
        for i in 0..32 {
            counters[i] += word[i];
        }
    }
    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn expand_bits(word: &b32) -> __m256i {
        unsafe {
            let mask1 = _mm256_loadu_si256(transmute::<&[u8; 32], &__m256i>(&mask1a));
            let mask2 = _mm256_loadu_si256(transmute::<&[u8; 32], &__m256i>(&mask2a));

            let zeros = _mm256_setzero_si256();
            let ones = _mm256_set1_epi8(0b_1);

            let word: i32 = transmute::<u32, i32>(word.0);

            let expanded = _mm256_set1_epi32(word);
            let shuffled = _mm256_shuffle_epi8(expanded, mask1);
            let masked = _mm256_and_si256(shuffled, mask2);
            let is_zero = _mm256_cmpeq_epi8(masked, zeros);
            let bools = _mm256_andnot_si256(is_zero, ones);
            bools
        }
    }
    #[cfg(not(target_feature = "avx2"))]
    #[inline(always)]
    fn expand_bits(word: &b32) -> [u8; 32] {
        let mut target = [0u8; 32];
        for i in 0..32 {
            target[i] = word.get_bit_u8(i);
        }
        target
    }

    #[cfg(target_feature = "avx2")]
    #[inline(always)]
    fn init_counters() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }
    #[cfg(not(target_feature = "avx2"))]
    #[inline(always)]
    fn init_counters() -> [u8; 32] {
        [0u8; 32]
    }
}

fn main() {
    //dbg!(mask1a);
    //dbg!(mask2a);
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .stack_size(2usize.pow(31))
        .build_global()
        .unwrap();

    let bits: Vec<b32> = (0..2usize.pow(18)).map(|i| b32(i as u32)).collect();
    dbg!();

    let simd_start = Instant::now();
    let sub_counts = bits.chunks(255).fold([0u64; 32], |mut acc, chunk| {
        let counts = count_batch_simd(chunk);
        add_u8s(&mut acc, counts);
        acc
    });

    let simd_time = simd_start.elapsed();
    println!("simd: {:?}", simd_time);

    let safe_start = Instant::now();
    let sub_counts2 = bits.chunks(255).fold([0u64; 32], |mut acc, chunk| {
        let counts = count_batch_safe(chunk);
        add_u8s(&mut acc, counts);
        acc
    });
    let safe_time = safe_start.elapsed();
    println!("safe: {:?}", safe_time);

    dbg!(safe_time.as_nanos() as f64 / simd_time.as_nanos() as f64);

    assert_eq!(sub_counts, sub_counts2);
    dbg!(sub_counts2);
}
