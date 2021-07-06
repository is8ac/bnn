use bnn::bits::{b32, b64, BitPack};
use bnn::shape::Pack;
use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;
use std::mem::{self, MaybeUninit};
use std::ops::{Add, AddAssign};
use std::time::Instant;

trait Counter: Add + AddAssign + Sized + std::iter::Sum + Copy {
    const ZERO: Self;
    const ONE: Self;
}

impl Counter for u8 {
    const ZERO: u8 = 0;
    const ONE: u8 = 1;
}

impl Counter for u16 {
    const ZERO: u16 = 0;
    const ONE: u16 = 1;
}

impl Counter for u32 {
    const ZERO: u32 = 0;
    const ONE: u32 = 1;
}

impl Counter for u64 {
    const ZERO: u64 = 0;
    const ONE: u64 = 1;
}

pub trait GetBit {
    fn bit(self, i: usize) -> bool;
}

impl GetBit for usize {
    #[inline(always)]
    fn bit(self, i: usize) -> bool {
        ((self >> i) & 1) == 1
    }
}

trait ExpIncrement<T, const V: bool>
where
    Self: BitPack<bool> + Pack<T>,
{
    type Acc;
    fn init_acc() -> Self::Acc;
    fn increment_acc(acc: &mut Self::Acc, bits: <Self as BitPack<bool>>::T);
    fn count(acc: Self::Acc) -> <Self as Pack<T>>::T;
}
/*

impl<C: Counter, T: ExpIncrement<C>, const L: usize> ExpIncrement<C> for [T; L] {
    fn init_acc() -> [Vec<T>; 4] {

    }
    fn increment_acc(acc: &mut [Vec<T>; 4], bits: u64) {
        for i in 0..L {

        }
    }
    fn count(acc: [Vec<T>; 4]) -> [T; 32] {

    }
}
*/

impl<T: Counter> ExpIncrement<T, true> for [(); 32] {
    type Acc = [Vec<T>; 4];
    fn init_acc() -> [Vec<T>; 4] {
        let mut data: [MaybeUninit<Vec<T>>; 4] = unsafe { MaybeUninit::uninit().assume_init() };
        for elem in &mut data[..] {
            *elem = MaybeUninit::new((0..256).map(|_| T::ZERO).collect());
        }
        unsafe { mem::transmute::<_, [Vec<T>; 4]>(data) }
    }
    fn increment_acc(acc: &mut [Vec<T>; 4], bits: b32) {
        let words = unsafe { mem::transmute::<b32, [u8; 4]>(bits) };
        for i in 0..4 {
            unsafe {
                *acc[i].get_unchecked_mut(words[i] as usize) += T::ONE;
            }
        }
    }
    fn count(acc: [Vec<T>; 4]) -> [T; 32] {
        let mut target = [T::ZERO; 32];
        for w in 0..4 {
            for b in 0..8 {
                target[w * 8 + b] = acc[w]
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i.bit(b))
                    .map(|(_, n)| *n)
                    .sum();
            }
        }
        target
    }
}

impl<T: Counter> ExpIncrement<T, false> for [(); 32] {
    type Acc = [[T; 256]; 4];
    fn init_acc() -> [[T; 256]; 4] {
        [[T::ZERO; 256]; 4]
    }
    fn increment_acc(acc: &mut [[T; 256]; 4], bits: b32) {
        let words = unsafe { mem::transmute::<b32, [u8; 4]>(bits) };
        for i in 0..4 {
            acc[i][words[i] as usize] += T::ONE;
        }
    }
    fn count(acc: [[T; 256]; 4]) -> [T; 32] {
        let mut target = [T::ZERO; 32];
        for w in 0..4 {
            for b in 0..8 {
                target[w * 8 + b] = acc[w]
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i.bit(b))
                    .map(|(_, n)| *n)
                    .sum();
            }
        }
        target
    }
}

impl<T: Counter> ExpIncrement<T, true> for [(); 64] {
    type Acc = [Vec<T>; 8];
    fn init_acc() -> [Vec<T>; 8] {
        let mut data: [MaybeUninit<Vec<T>>; 8] = unsafe { MaybeUninit::uninit().assume_init() };
        for elem in &mut data[..] {
            *elem = MaybeUninit::new((0..256).map(|_| T::ZERO).collect());
        }
        unsafe { mem::transmute::<_, [Vec<T>; 8]>(data) }
    }
    fn increment_acc(acc: &mut [Vec<T>; 8], bits: b64) {
        let words = unsafe { mem::transmute::<b64, [u8; 8]>(bits) };
        for i in 0..8 {
            unsafe {
                *acc[i].get_unchecked_mut(words[i] as usize) += T::ONE;
            }
        }
    }
    /*
    type Acc = [[T; 256]; 8];
    fn init_acc() -> [[T; 256]; 8] {
        [[T::ZERO; 256]; 8]
    }
    fn increment_acc(acc: &mut [[T; 256]; 8], bits: b64) {
        let words = unsafe { mem::transmute::<b64, [u8; 8]>(bits) };
        for i in 0..8 {
            acc[i][words[i] as usize] += T::ONE;
        }
    }
    */
    fn count(acc: [Vec<T>; 8]) -> [T; 64] {
        let mut target = [T::ZERO; 64];
        for w in 0..8 {
            for b in 0..8 {
                target[w * 8 + b] = acc[w]
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i.bit(b))
                    .map(|(_, n)| *n)
                    .sum();
            }
        }
        target
    }
}

pub fn foo(bit_strings: &Vec<b32>, acc: &mut [Vec<u32>; 4]) {
    bit_strings.iter().for_each(|bits| {
        <[(); 32] as ExpIncrement<u32, true>>::increment_acc(acc, *bits);
    });
}

pub fn bar(bit_strings: &Vec<b32>, acc: &mut [[u32; 256]; 4]) {
    bit_strings.iter().for_each(|bits| {
        <[(); 32] as ExpIncrement<u32, false>>::increment_acc(acc, *bits);
    });
}

const N_EXAMPLES: usize = 2usize.pow(28);
type CounterType = u32;
const N_BITS: usize = 32;
const V_TYPE: bool = false;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let bit_strings: Vec<b32> = (0..N_EXAMPLES).map(|_| rng.gen()).collect();

    let mut acc = Box::new(<[(); N_BITS] as ExpIncrement<CounterType, V_TYPE>>::init_acc());

    let start = Instant::now();

    bit_strings.iter().for_each(|bits| {
        <[(); N_BITS] as ExpIncrement<CounterType, V_TYPE>>::increment_acc(&mut acc, *bits);
    });
    dbg!(start.elapsed().as_nanos() as f64 / N_EXAMPLES as f64);
    let counts = <[(); N_BITS] as ExpIncrement<CounterType, V_TYPE>>::count(*acc);
    dbg!(counts);
}
