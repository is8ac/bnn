#![feature(int_log)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

pub trait GetBit {
    fn bit(self, i: usize) -> bool;
}

impl GetBit for usize {
    #[inline(always)]
    fn bit(self, i: usize) -> bool {
        ((self >> i) & 1) == 1
    }
}

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
        BitArray64([0u64 - sign as u64; L])
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

fn comparator<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> (T, T, T) {
    let mut acc = half_comparator(a[0], b[0]);
    for i in 1..L {
        acc = full_comparator(a[i], b[i], acc);
    }
    acc
}

fn bit_add<T: BitSlice + Copy, const L: usize>(a: &[T; L], b: &[T; L]) -> [T; L + 1] {
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

fn extend<T: BitSlice + Copy, const I: usize, const O: usize>(v: &[T; I]) -> [T; O] {
    let mut target = [T::zeros(); O];
    for i in 0..I {
        target[i] = v[i];
    }
    target
}

fn bit_splat<T: BitSlice + Copy, const L: usize>(value: u32) -> [T; L] {
    let mut target = [T::zeros(); L];
    for i in 0..L {
        let sign = (value >> i) & 1 == 1;
        target[i] = T::splat(sign);
    }
    target
}

fn exp_count<T: BitSlice + Copy, const N: usize, const L: usize, const E: u32>(
    partial_sum: &[T; L],
    bits: &[T; E as usize],
    target_bit: &T,
    thresholds: &[[T; L]; N],
    counters: &mut [[u64; N]; 2usize.pow(E)],
) where
    T: std::fmt::Debug,
    [T; E.log2() as usize + 1]: ,
{
    for mask in 0..2usize.pow(E) {
        let mut exp_sum = [T::zeros(); E.log2() as usize + 1];
        for b in 0..E {
            let expanded = extend(&[T::splat(mask.bit(b as usize)).and(bits[b as usize])]);
            exp_sum = bit_add_wrapping(&exp_sum, &expanded);
        }
        let exp_sum = extend(&exp_sum);
        let full_sum = bit_add_wrapping(&exp_sum, &partial_sum);

        for i in 0..N {
            let (_, _, gt) = comparator(&full_sum, &thresholds[i]);
            counters[mask][i] += gt.xor(*target_bit).not().count_bits() as u64;
        }
    }
}

fn unit_count<T: BitSlice + Copy, const I: usize, const N: usize, const P: usize>(
    partial_sum: &[T; P],
    inputs: &[T; I],
    target_bit: &T,
    thresholds: &[[T; P]; N],
    counters: &mut [[[u64; N]; 2]; I],
) {
    for i in 0..I {
        for t in 0..N {
            for s in 0..2 {
                let full_count =
                    bit_add_wrapping(&partial_sum, &extend(&[inputs[i].xor(T::splat(s == 1))]));
                let (_, _, gt) = comparator(&full_count, &thresholds[t]);
                counters[i][s][t] += gt.xor(*target_bit).not().count_bits() as u64;
            }
        }
    }
}

type BitWordType = BitArray64<1>;
const EXP_SIZE: u32 = 8;
const ACC_SIZE: usize = 5;

pub fn unit_count_demo(
    partial_sum: &[BitWordType; ACC_SIZE],
    inputs: &[BitWordType; 512],
    target_bit: &BitWordType,
    thresholds: &[[BitWordType; ACC_SIZE]; 2],
    counters: &mut [[[u64; 2]; 2]; 512],
) {
    unit_count::<BitWordType, 512, 2, ACC_SIZE>(
        partial_sum,
        inputs,
        target_bit,
        thresholds,
        counters,
    )
}

pub fn exp_count_demo(
    partial_sum: &[BitWordType; ACC_SIZE],
    bits: &[BitWordType; EXP_SIZE as usize],
    target_bit: &BitWordType,
    thresholds: &[[BitWordType; ACC_SIZE]; 3],
    counters: &mut [[u64; 3]; 2usize.pow(EXP_SIZE)],
) {
    exp_count::<BitWordType, 3, ACC_SIZE, EXP_SIZE>(
        partial_sum,
        bits,
        target_bit,
        thresholds,
        counters,
    )
}

fn main() {}
