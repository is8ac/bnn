#![feature(const_generics)]
use bitnn::shape::{Array, Element, Map, MapMut, Shape, ZipFold};
use std::fmt;
use std::num::Wrapping;
use std::ops::{BitAnd, BitAndAssign, BitOrAssign, BitXor, BitXorAssign};

pub trait BitMul<I, O> {
    fn bit_mul(&self, input: &I) -> O;
}

impl<I, O: Default + Copy, T: BitMul<I, O>, const L: usize> BitMul<I, [O; L]> for [T; L]
where
    [O; L]: Default,
{
    fn bit_mul(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_mul(input);
        }
        target
    }
}

trait BitArray
where
    Self::BitShape: Shape,
    u32: Element<Self::BitShape>,
    bool: Element<Self::BitShape>,
{
    type BitShape;
    fn bitpack(bools: &<bool as Element<Self::BitShape>>::Array) -> Self;
    fn array_increment_counters(&self, counters: &mut <u32 as Element<Self::BitShape>>::Array);
    fn array_flipped_increment_counters(
        &self,
        sign: bool,
        counters: &mut <u32 as Element<Self::BitShape>>::Array,
    );
}

impl<T: BitArray, const L: usize> BitArray for [T; L]
where
    [T; L]: Default,
    T::BitShape: Shape,
    u32: Element<T::BitShape>,
    bool: Element<T::BitShape>,
{
    type BitShape = [T::BitShape; L];
    fn bitpack(bools: &<bool as Element<Self::BitShape>>::Array) -> Self {
        let mut target = Self::default();
        for i in 0..L {
            target[i] = T::bitpack(&bools[i]);
        }
        target
    }
    fn array_increment_counters(&self, counters: &mut <u32 as Element<Self::BitShape>>::Array) {
        for i in 0..L {
            self[i].array_increment_counters(&mut counters[i]);
        }
    }
    fn array_flipped_increment_counters(
        &self,
        sign: bool,
        counters: &mut <u32 as Element<Self::BitShape>>::Array,
    ) {
        for i in 0..L {
            self[i].array_flipped_increment_counters(sign, &mut counters[i]);
        }
    }
}

trait HammingDistance {
    fn hamming_distance(&self, rhs: &Self) -> u32;
}

impl<T: HammingDistance, const L: usize> HammingDistance for [T; L] {
    fn hamming_distance(&self, rhs: &Self) -> u32 {
        let mut sum = 0u32;
        for i in 0..L {
            sum += self[i].hamming_distance(&rhs[i]);
        }
        sum
    }
}

trait BitWord
where
    Self::BitShape: Shape,
    bool: Element<Self::BitShape>,
    u32: Element<Self::BitShape>,
{
    type BitShape;
    const BIT_LEN: usize;
    fn splat(sign: bool) -> Self;
    fn get_bit(&self, i: usize) -> bool;
    fn bitpack(bools: &<bool as Element<Self::BitShape>>::Array) -> Self;
    fn increment_counters(&self, counters: &mut <u32 as Element<Self::BitShape>>::Array);
    fn flipped_increment_counters(
        &self,
        sign: bool,
        counters: &mut <u32 as Element<Self::BitShape>>::Array,
    );
}

macro_rules! for_uints {
    ($b_type:ident, $u_type:ty, $len:expr, $format_string:expr) => {
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone)]
        struct $b_type(pub $u_type);

        impl BitWord for $b_type {
            type BitShape = [(); $len];
            const BIT_LEN: usize = $len;
            fn splat(sign: bool) -> Self {
                $b_type((Wrapping(0 as $u_type) - Wrapping(sign as $u_type)).0)
            }
            fn get_bit(&self, i: usize) -> bool {
                ((self.0 >> i) & 1) == 1
            }
            fn bitpack(bools: &<bool as Element<Self::BitShape>>::Array) -> Self {
                let mut bits = <$u_type>::default();
                for b in 0..$len {
                    bits |= (bools[b] as $u_type) << b;
                }
                $b_type(bits)
            }
            fn increment_counters(&self, counters: &mut [u32; $len]) {
                for b in 0..$len {
                    counters[b] += ((self.0 >> b) & 1) as u32
                }
            }
            fn flipped_increment_counters(
                &self,
                sign: bool,
                counters: &mut <u32 as Element<Self::BitShape>>::Array,
            ) {
                let word = *self ^ Self::splat(sign);
                word.increment_counters(counters);
            }
        }
        impl HammingDistance for $b_type {
            fn hamming_distance(&self, rhs: &$b_type) -> u32 {
                (self.0 ^ rhs.0).count_ones()
            }
        }
        impl<I: HammingDistance> BitMul<I, $b_type> for [(I, u32); $len] {
            fn bit_mul(&self, input: &I) -> $b_type {
                let mut target = $b_type(0);
                for b in 0..$len {
                    target |=
                        $b_type(((self[b].0.hamming_distance(input) < self[b].1) as $u_type) << b);
                }
                target
            }
        }
        impl BitXor for $b_type {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self::Output {
                $b_type(self.0 ^ rhs.0)
            }
        }
        impl BitXorAssign for $b_type {
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }
        impl BitOrAssign for $b_type {
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }
        impl BitAnd for $b_type {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self::Output {
                $b_type(self.0 & rhs.0)
            }
        }
        impl BitAndAssign for $b_type {
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }
        impl fmt::Display for $b_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, $format_string, self.0)
            }
        }

        impl fmt::Debug for $b_type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, $format_string, self.0)
            }
        }
    };
}

for_uints!(b8, u8, 8, "{:08b}");
for_uints!(b16, u16, 16, "{:016b}");
for_uints!(b32, u32, 32, "{:032b}");

type InputShape = [([[(); 3]; 4], [(); 4]); 5];
//type InputType = ([[u16; 3]; 4], [u16; 4]);
type InputType = <u8 as Element<InputShape>>::Array;
type InputElem = <InputType as Array<InputShape>>::Element;

fn main() {
    let foo = b8::splat(false);
    let bar = b8::splat(true);
    let baz = b32::splat(false);
    dbg!(bar.get_bit(5));
    println!("{}", foo);
    dbg!(foo ^ bar);
    dbg!(baz);
    dbg!(std::any::type_name::<InputShape>());
    dbg!(std::any::type_name::<<u8 as Element<InputShape>>::Array>());
    dbg!(std::any::type_name::<
        <InputType as Array<InputShape>>::Element,
    >());
    //let mut counters = <u32 as Element<InputShape>>::Array::default();
    //foo.increment_counters(&mut counters);
    //dbg!(counters);
    let weights = [(b16::splat(true), 5u32); 8];
    let output: b8 = weights.bit_mul(&b16::splat(false));
    dbg!(output);
}
