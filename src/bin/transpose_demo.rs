#![feature(stdsimd)]
use std::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_add_epi8, _mm256_and_si256, _mm256_extract_epi64,
    _mm256_load_si256, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_slli_epi64, _mm256_srli_epi32, _mm256_store_si256,
    _mm256_xor_si256,
};
use std::convert::TryFrom;
use std::mem::transmute;
use std::mem::{self, MaybeUninit};

#[repr(align(32))]
#[derive(Copy, Clone)]
pub struct Word256([u8; 32]);

/*
#[cfg(target_feature = "avx2")]
#[inline(always)]
fn expand_bits(word: &u32) -> __m256i {
    unsafe {
        let mask1 = _mm256_loadu_si256(transmute::<&[u8; 32], &__m256i>(&PSHUF_BYTE_MASK));
        let mask2 = _mm256_loadu_si256(transmute::<&[u8; 32], &__m256i>(&BYTE_BIT_MASK));

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
*/

trait MulaPopcount {
    #[inline(always)]
    fn count_ones(&self) -> u32 {
        let counts: [u64; 4] = unsafe { transmute(self.mula_popcnt()) };
        (counts[0] + counts[1] + counts[2] + counts[3]) as u32
    }
    unsafe fn mula_popcnt(&self) -> __m256i;
}

impl MulaPopcount for Word256 {
    #[target_feature(enable = "avx2")]
    unsafe fn mula_popcnt(&self) -> __m256i {
        let v = transmute(self.0);
        let lookup: [u8; 32] = [
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        ];
        let lookup = transmute(lookup);
        let low_mask = _mm256_set1_epi8(0b_00001111);
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
        let popcnt1 = _mm256_shuffle_epi8(lookup, lo);
        let popcnt2 = _mm256_shuffle_epi8(lookup, hi);
        let total = _mm256_add_epi8(popcnt1, popcnt2);
        _mm256_sad_epu8(total, _mm256_setzero_si256())
    }
}

impl<T: MulaPopcount, const L: usize> MulaPopcount for [T; L] {
    #[inline(always)]
    unsafe fn mula_popcnt(&self) -> __m256i {
        let mut acc = _mm256_setzero_si256();
        for i in 0..L {
            let sum = self[i].mula_popcnt();
            acc = _mm256_add_epi64(acc, sum);
        }
        acc
    }
}

/*
#[target_feature(enable = "avx2")]
pub unsafe fn foo(a: [Word256; 64]) -> u32 {
    a.count_ones()
}
*/

#[target_feature(enable = "avx2")]
unsafe fn count(v: __m256i) -> __m256i {
    let lookup: [u8; 32] = [
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    ];
    let lookup = transmute(lookup);
    let low_mask = _mm256_set1_epi8(0b_00001111);
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    let popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    let popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    let total = _mm256_add_epi8(popcnt1, popcnt2);
    _mm256_sad_epu8(total, _mm256_setzero_si256())
}

#[target_feature(enable = "avx2")]
unsafe fn csa(h: &mut __m256i, l: &mut __m256i, a: __m256i, b: __m256i, c: __m256i) {
    let u = _mm256_xor_si256(a, b);
    *h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    *l = _mm256_xor_si256(u, c);
}

#[target_feature(enable = "avx2")]
unsafe fn half_adder(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    (_mm256_xor_si256(a, b), _mm256_and_si256(a, b))
}

#[target_feature(enable = "avx2")]
unsafe fn full_adder(a: __m256i, b: __m256i, c: __m256i) -> (__m256i, __m256i) {
    let u = _mm256_xor_si256(a, b);
    (
        _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c)),
        _mm256_xor_si256(u, c),
    )
}

#[target_feature(enable = "avx2")]
unsafe fn array_add1(v: &[__m256i; 2]) -> [__m256i; 2] {
    let (a, b) = half_adder(v[0], v[1]);
    [a, b]
}

macro_rules! adder_fns {
    ($len:expr, $adder_fn:ident, $array_add_fn:ident, $sub_array_add_fn:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $adder_fn(a: &[__m256i; $len], b: &[__m256i; $len]) -> [__m256i; $len + 1] {
            let mut acc = MaybeUninit::<[__m256i; $len + 1]>::uninit();
            let ptr = acc.as_mut_ptr();

            let (zero, c) = half_adder(a[0], b[0]);
            (*ptr)[0] = zero;
            let mut carry = c;
            for i in 1..$len {
                let (bit, c) = full_adder(a[i], b[i], carry);
                (*ptr)[i] = bit;
                carry = c;
            }
            (*ptr)[$len] = carry;
            mem::transmute::<_, [__m256i; $len + 1]>(acc)
        }
        #[target_feature(enable = "avx2")]
        unsafe fn $array_add_fn(v: &[__m256i; 2usize.pow($len)]) -> [__m256i; $len + 1] {
            let a = $sub_array_add_fn(
                <&[__m256i; 2usize.pow($len - 1)]>::try_from(&v[0..2usize.pow($len - 1)]).unwrap(),
            );
            let b = $sub_array_add_fn(
                <&[__m256i; 2usize.pow($len - 1)]>::try_from(
                    &v[2usize.pow($len - 1)..2usize.pow($len)],
                )
                .unwrap(),
            );
            $adder_fn(&a, &b)
        }
    };
}

adder_fns!(2, adder2, array_add2, array_add1);
adder_fns!(3, adder3, array_add3, array_add2);
adder_fns!(4, adder4, array_add4, array_add3);
adder_fns!(5, adder5, array_add5, array_add4);
adder_fns!(6, adder6, array_add6, array_add5);
adder_fns!(7, adder7, array_add7, array_add6);
adder_fns!(8, adder8, array_add8, array_add7);

adder_fns!(9, adder9, array_add9, array_add8);
adder_fns!(10, adder10, array_add10, array_add9);
adder_fns!(11, adder11, array_add11, array_add10);
adder_fns!(12, adder12, array_add12, array_add11);
adder_fns!(13, adder13, array_add13, array_add12);
adder_fns!(14, adder14, array_add14, array_add13);
adder_fns!(15, adder15, array_add15, array_add14);
adder_fns!(16, adder16, array_add16, array_add15);

pub fn foo(a: &[__m256i; 65536]) -> [__m256i; 17] {
    unsafe { array_add16(a) }
}

#[target_feature(enable = "avx2")]
unsafe fn count_bits_3(bits: &[__m256i; 3]) -> u32 {
    let mut total = count(bits[0]);
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(bits[1]), 1));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(bits[2]), 2));
    (_mm256_extract_epi64(total, 0)
        + _mm256_extract_epi64(total, 1)
        + _mm256_extract_epi64(total, 2)
        + _mm256_extract_epi64(total, 3)) as u32
}

#[target_feature(enable = "avx2")]
unsafe fn avx_hs(d: &[__m256i]) {
    let mut total = _mm256_setzero_si256();
    let mut ones = _mm256_setzero_si256();
    let mut twos = _mm256_setzero_si256();
    let mut fours = _mm256_setzero_si256();
    let mut eights = _mm256_setzero_si256();
    let mut sixteens = _mm256_setzero_si256();

    let twosA = _mm256_setzero_si256();
    let twosB = _mm256_setzero_si256();
    let foursA = _mm256_setzero_si256();
    let foursB = _mm256_setzero_si256();
    let eightsA = _mm256_setzero_si256();
    let eightsB = _mm256_setzero_si256();

    /*
    d.chunks(16).for_each(|chunk| {
        csa(&mut twosA, &mut ones, ones, chunk[0], chunk[1]);
        csa(&mut twosB, &mut ones, ones, chunk[2], chunk[3]);
        csa(&mut foursA, &mut twos, twos, twosA, twosB);
        csa(&mut twosA, &mut ones, ones, chunk[4], chunk[5]);
        csa(&mut twosB, &mut ones, ones, chunk[6], chunk[7]);
        csa(&mut foursB, &mut twos, twos, twosA, twosB);

        csa(&mut eightsA, &mut fours, fours, foursA, foursB);

        csa(&mut twosA, &mut ones, ones, chunk[8], chunk[9]);
        csa(&mut twosB, &mut ones, ones, chunk[10], chunk[11]);
        csa(&mut foursA, &mut twos, twos, twosA, twosB);
        csa(&mut twosA, &mut ones, ones, chunk[12], chunk[13]);
        csa(&mut twosB, &mut ones, ones, chunk[14], chunk[15]);
        csa(&mut foursB, &mut twos, twos, twosA, twosB);

        csa(&mut eightsB, &mut fours, fours, foursA, foursB);
        csa(&mut sixteens, &mut eights, eights, eightsA, eightsB);
        total = _mm256_add_epi64(total, count(sixteens));
    });
    */
    total = _mm256_slli_epi64(total, 4);
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(eights), 3));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(fours), 2));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(twos), 1));
    total = _mm256_add_epi64(total, count(ones));
    _mm256_extract_epi64(total, 0)
        + _mm256_extract_epi64(total, 1)
        + _mm256_extract_epi64(total, 2)
        + _mm256_extract_epi64(total, 3);
}

fn main() {
    let a = Word256([
        0, 1, 1, 7, 4, 213, 7, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    let b = Word256([
        7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5,
        6, 7,
    ]);
    //let target = unsafe { foo(a) };
    //dbg!(target.0);
    let bytes = [[0u8; 7]; 256];
    let int = [[0u8; 256]; 7];
    let target = [[Word256([0u8; 32]); 8]; 7];
    let count = a.count_ones();
    dbg!(count);
    let real_count: u32 = a.0.iter().map(|x| x.count_ones()).sum();
    dbg!(real_count);
}
