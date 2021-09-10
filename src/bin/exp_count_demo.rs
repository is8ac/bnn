use bnn::bits::{b32, b64, BitPack, ExpIncrement};
use bnn::shape::{LongDefault, Pack};
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

/*

pub fn foo(bit_strings: &Vec<b32>, acc: &mut [[u32; 256]; 4]) {
    bit_strings.iter().for_each(|bits| {
        <[(); 32] as ExpIncrement<u32>>::increment_acc(acc, *bits);
    });
}

*/
const N_EXAMPLES: usize = 2usize.pow(29);
type CounterType = u32;
type InputBitShape = [[(); 32]; 2];

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let bit_strings: Vec<<InputBitShape as BitPack<bool>>::T> =
        (0..N_EXAMPLES).map(|_| rng.gen()).collect();
    let mut acc = Box::new(<InputBitShape as ExpIncrement<CounterType>>::init_acc());

    let start = Instant::now();

    bit_strings.iter().for_each(|bits| {
        <InputBitShape as ExpIncrement<CounterType>>::increment_acc(&mut acc, bits);
    });
    dbg!(start.elapsed().as_nanos() as f64 / N_EXAMPLES as f64);
    let counts = <InputBitShape as ExpIncrement<CounterType>>::count(&*acc);
    //dbg!(counts);
}
