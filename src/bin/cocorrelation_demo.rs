#![feature(const_generics)]

use bitnn::bits::{BitLen, GetBit};
//use bitnn::count::{IncrementCounters, IncrementFracCounters};

pub trait Shape {
    const N: usize;
    type Index;
    const SHAPE: Self::Index;
}

impl Shape for () {
    const N: usize = 1;
    type Index = ();
    const SHAPE: Self::Index = ();
}

impl<T: Shape, const L: usize> Shape for [T; L] {
    const N: usize = T::N * L;
    type Index = (usize, T::Index);
    const SHAPE: Self::Index = (L, T::SHAPE);
}

pub trait BitShape
where
    Self::Shape: Shape,
{
    type Shape;
}

impl BitShape for u8 {
    type Shape = [(); 8];
}

impl<T: BitShape, const L: usize> BitShape for [T; L] {
    type Shape = [T::Shape; L];
}

pub trait Bits
where
    Self: Sized + BitShape,
    bool: Element<Self::Shape>,
    u32: Element<Self::Shape>,
{
    fn bitpack(values: &<bool as Element<Self::Shape>>::Array) -> Self;
    fn increment_counters(&self, counters: &mut <u32 as Element<Self::Shape>>::Array);
}

impl Bits for u8
where
    u8: BitShape,
    bool: Element<Self::Shape>,
{
    fn bitpack(values: &[bool; 8]) -> u8 {
        let mut bits = 0u8;
        for b in 0..8 {
            bits |= (values[b] as u8) << b;
        }
        bits
    }
    fn increment_counters(&self, counters: &mut [u32; 8]) {
        for b in 0..8 {
            counters[b] += ((self >> b) & 1) as u32;
        }
    }
}

impl<T: Bits, const L: usize> Bits for [T; L]
where
    bool: Element<T::Shape>,
    [T; L]: Default,
    u32: Element<T::Shape>,
{
    fn bitpack(bools: &[<bool as Element<T::Shape>>::Array; L]) -> Self {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::bitpack(&bools[i]);
        }
        target
    }
    fn increment_counters(&self, counters: &mut [<u32 as Element<T::Shape>>::Array; L]) {
        for i in 0..L {
            self[i].increment_counters(&mut counters[i]);
        }
    }
}

pub trait IncrementFracCounters
where
    Self: Bits + BitShape,
    bool: Element<Self::Shape>,
    u32: Element<Self::Shape>,
{
    fn increment_frac_counters(&self, counters: &mut (usize, <u32 as Element<Self::Shape>>::Array));
}

impl<B: Bits + BitShape> IncrementFracCounters for B
where
    bool: Element<Self::Shape>,
    u32: Element<Self::Shape>,
{
    fn increment_frac_counters(&self, counters: &mut (usize, <u32 as Element<B::Shape>>::Array)) {
        counters.0 += 1;
        self.increment_counters(&mut counters.1);
    }
}

pub trait Element<S: Shape> {
    type Array;
}

impl<T: Sized> Element<()> for T {
    type Array = T;
}

impl<T: Element<S>, S: Shape, const L: usize> Element<[S; L]> for T {
    type Array = [T::Array; L];
}

trait Map<I: Element<Self>, O: Element<Self>>
where
    Self: Shape + Sized,
{
    fn map<F: Fn(&I) -> O>(
        input: &<I as Element<Self>>::Array,
        map_fn: F,
    ) -> <O as Element<Self>>::Array;
}

impl<I, O> Map<I, O> for () {
    fn map<F: Fn(&I) -> O>(input: &I, map_fn: F) -> O {
        map_fn(input)
    }
}

impl<S: Shape + Map<I, O>, I: Element<S>, O: Element<S>, const L: usize> Map<I, O> for [S; L]
where
    [<O as Element<S>>::Array; L]: Default,
{
    fn map<F: Fn(&I) -> O>(
        input: &[<I as Element<S>>::Array; L],
        map_fn: F,
    ) -> [<O as Element<S>>::Array; L] {
        let mut target = <[<O as Element<S>>::Array; L]>::default();
        for i in 0..L {
            target[i] = <S as Map<I, O>>::map(&input[i], &map_fn);
        }
        target
    }
}

trait Fold<B, I: Element<Self>>
where
    Self: Shape + Sized,
{
    fn fold<F: Fn(B, &I) -> B>(input: &<I as Element<Self>>::Array, acc: B, fold_fn: F) -> B;
}

impl<B, I: Element<(), Array = I> + Sized> Fold<B, I> for () {
    fn fold<F: Fn(B, &I) -> B>(input: &I, acc: B, fold_fn: F) -> B {
        fold_fn(acc, input)
    }
}

impl<S: Shape + Fold<B, I>, B, I: Element<S> + Sized, const L: usize> Fold<B, I> for [S; L] {
    fn fold<F: Fn(B, &I) -> B>(input: &[<I as Element<S>>::Array; L], mut acc: B, fold_fn: F) -> B {
        for i in 0..L {
            acc = <S as Fold<B, I>>::fold(&input[i], acc, &fold_fn);
        }
        acc
    }
}

trait ZipMap<A: Element<Self>, B: Element<Self>, O: Element<Self>>
where
    Self: Shape + Sized,
{
    fn zip_map<F: Fn(&A, &B) -> O>(
        a: &<A as Element<Self>>::Array,
        b: &<B as Element<Self>>::Array,
        map_fn: F,
    ) -> <O as Element<Self>>::Array;
}

impl<A: Element<(), Array = A> + Copy, B: Element<(), Array = B> + Copy, O> ZipMap<A, B, O> for () {
    fn zip_map<F: Fn(&A, &B) -> O>(a: &A, b: &B, map_fn: F) -> O {
        map_fn(a, b)
    }
}

impl<S: Shape + ZipMap<A, B, O>, A: Element<S>, B: Element<S>, O: Element<S>, const L: usize>
    ZipMap<A, B, O> for [S; L]
where
    <O as Element<[S; L]>>::Array: Default,
{
    fn zip_map<F: Fn(&A, &B) -> O>(
        a: &<A as Element<[S; L]>>::Array,
        b: &<B as Element<[S; L]>>::Array,
        map_fn: F,
    ) -> <O as Element<[S; L]>>::Array {
        let mut target = <[<O as Element<S>>::Array; L]>::default();
        for i in 0..L {
            target[i] = S::zip_map(&a[i], &b[i], &map_fn);
        }
        target
    }
}

// f64 values
trait Sum {
    fn sum(&self) -> f64;
}

impl Sum for f64 {
    fn sum(&self) -> f64 {
        *self
    }
}

impl<T: Sum, const L: usize> Sum for [T; L] {
    fn sum(&self) -> f64 {
        let mut sum = 0f64;
        for i in 0..L {
            sum += self[i].sum();
        }
        sum
    }
}

trait Min
where
    Self: Shape + Sized,
    f64: Element<Self>,
{
    fn min(values: &<f64 as Element<Self>>::Array) -> Option<(Self::Index, f64)>;
}

impl Min for () {
    fn min(&values: &f64) -> Option<((), f64)> {
        Some(((), values))
    }
}

impl<T: Min, const L: usize> Min for [T; L]
where
    f64: Element<T>,
    Self: Shape<Index = (usize, T::Index)>,
    T::Index: Copy,
{
    fn min(values: &[<f64 as Element<T>>::Array; L]) -> Option<((usize, T::Index), f64)> {
        let mut cur_min: Option<((usize, T::Index), f64)> = None;
        for i in 0..L {
            if let Some((sub_index, sub_min)) = T::min(&values[i]) {
                if let Some((_, min)) = cur_min {
                    if sub_min < min {
                        cur_min = Some(((i, sub_index), sub_min));
                    }
                } else {
                    cur_min = Some(((i, sub_index), sub_min));
                }
            }
        }
        cur_min
    }
}

trait FlipBool
where
    bool: Element<Self>,
    Self: Shape + Sized,
{
    fn flip_bool(bools: &mut <bool as Element<Self>>::Array, index: Self::Index);
}

impl FlipBool for () {
    fn flip_bool(bools: &mut bool, _: ()) {
        *bools = !*bools;
    }
}

impl<T: FlipBool + Shape, const L: usize> FlipBool for [T; L]
where
    bool: Element<T>,
{
    fn flip_bool(
        bools: &mut [<bool as Element<T>>::Array; L],
        (index, sub_index): (usize, T::Index),
    ) {
        T::flip_bool(&mut bools[index], sub_index);
    }
}

trait IncrementCountersMatrix<T: BitShape>
where
    Self: BitShape,
    u32: Element<Self::Shape>,
    [(usize, <u32 as Element<Self::Shape>>::Array); 2]: Element<T::Shape>,
{
    fn increment_counters_matrix(
        &self,
        counters_matrix: &mut <[(usize, <u32 as Element<Self::Shape>>::Array); 2] as Element<
            T::Shape,
        >>::Array,
        target: &T,
    );
}

impl<I: IncrementFracCounters + BitShape> IncrementCountersMatrix<u8> for I
where
    bool: Element<I::Shape>,
    u32: Element<I::Shape>,
{
    fn increment_counters_matrix(
        &self,
        counters_matrix: &mut [[(usize, <u32 as Element<Self::Shape>>::Array); 2]; 8],
        target: &u8,
    ) {
        for i in 0..<u8>::BIT_LEN {
            self.increment_frac_counters(&mut counters_matrix[i][target.bit(i) as usize]);
        }
    }
}

impl<I: IncrementCountersMatrix<T> + BitShape, T: BitShape, const L: usize>
    IncrementCountersMatrix<[T; L]> for I
where
    u32: Element<I::Shape>,
    [(usize, <u32 as Element<Self::Shape>>::Array); 2]: Element<T::Shape>,
{
    fn increment_counters_matrix(
        &self,
        counters_matrix: &mut [<[(usize, <u32 as Element<Self::Shape>>::Array); 2] as Element<T::Shape>>::Array;
                 L],
        target: &[T; L],
    ) {
        for w in 0..L {
            self.increment_counters_matrix(&mut counters_matrix[w], &target[w]);
        }
    }
}

fn bayes_magn<S: Shape + ZipMap<u32, u32, f64>>(
    counters: &[(usize, <u32 as Element<S>>::Array); 2],
) -> <f64 as Element<S>>::Array
where
    u32: Element<S>,
    f64: Element<S>,
{
    let n = (counters[0].0 + counters[1].0) as f64;
    let na = counters[0].0 as f64;
    let pa = na / n;
    <S as ZipMap<u32, u32, f64>>::zip_map(&counters[0].1, &counters[1].1, |&a, &b| {
        let pb = (a + b) as f64 / n;
        let pba = a as f64 / na;
        let pab = (pba * pa) / pb;
        ((pab * 2f64) - 1f64).abs()
    })
}

trait Mse
where
    Self: Shape + Sized,
    bool: Element<Self>,
    f64: Element<Self>,
    <f64 as Element<Self>>::Array: Element<Self>,
{
    fn single_mse(
        edges: &<<f64 as Element<Self>>::Array as Element<Self>>::Array,
        local_avgs: &<f64 as Element<Self>>::Array,
        mask: &<bool as Element<Self>>::Array,
    ) -> f64;
    fn bit_flip_mses(
        edges: &<<f64 as Element<Self>>::Array as Element<Self>>::Array,
        local_avgs: &<f64 as Element<Self>>::Array,
        mask: &<bool as Element<Self>>::Array,
    ) -> <f64 as Element<Self>>::Array;
}

impl<
        S: Shape
            + Map<f64, f64>
            + Map<<f64 as Element<S>>::Array, f64>
            + Map<<f64 as Element<S>>::Array, <f64 as Element<S>>::Array>
            + ZipMap<f64, f64, f64>
            + ZipMap<bool, f64, f64>
            + ZipMap<f64, bool, f64>
            + ZipMap<<f64 as Element<S>>::Array, f64, <f64 as Element<S>>::Array>
            + Fold<<f64 as Element<S>>::Array, <f64 as Element<S>>::Array>,
    > Mse for S
where
    bool: Element<S>,
    f64: Element<S>,
    <f64 as Element<S>>::Array: Element<S> + Sum + Default,
{
    fn single_mse(
        edges: &<<f64 as Element<S>>::Array as Element<S>>::Array,
        local_avgs: &<f64 as Element<S>>::Array,
        mask: &<bool as Element<S>>::Array,
    ) -> f64 {
        let n = <[[(); 8]; 3]>::N as f64;
        let avg = local_avgs.sum() / n;
        let local_counts = <S as Map<<f64 as Element<S>>::Array, f64>>::map(&edges, |edge_set| {
            masked_sum::<S>(edge_set, mask)
        });
        let scale = avg / (local_counts.sum() / n);
        <S as ZipMap<f64, f64, f64>>::zip_map(&local_avgs, &local_counts, |a, b| {
            (a - (b * scale)).powi(2)
        })
        .sum()
    }
    fn bit_flip_mses(
        edges: &<<f64 as Element<S>>::Array as Element<S>>::Array,
        local_avgs: &<f64 as Element<S>>::Array,
        mask: &<bool as Element<S>>::Array,
    ) -> <f64 as Element<S>>::Array {
        let n = <[[(); 8]; 3]>::N as f64;
        let avg = local_avgs.sum() / n;
        let bit_flip_local_counts = <S as Map<
            <f64 as Element<S>>::Array,
            <f64 as Element<S>>::Array,
        >>::map(&edges, |edge_set| {
            let sum = masked_sum::<S>(edge_set, mask);
            <S as ZipMap<bool, f64, f64>>::zip_map(&mask, &edge_set, |&mask_bit, &edge| {
                if mask_bit {
                    sum - edge
                } else {
                    sum + edge
                }
            })
        });
        let bit_flip_sums = S::fold(
            &bit_flip_local_counts,
            <f64 as Element<S>>::Array::default(),
            |a, b| <S as ZipMap<f64, f64, f64>>::zip_map(&a, b, |x, y| x + y),
        );
        let bit_flip_scales = <S as Map<f64, f64>>::map(&bit_flip_sums, |sum| avg / (sum / n));
        let bit_flip_local_mses = S::zip_map(
            &bit_flip_local_counts,
            local_avgs,
            |local_counts, local_avg| {
                <S as ZipMap<f64, f64, f64>>::zip_map(
                    local_counts,
                    &bit_flip_scales,
                    |count, scale| (local_avg - (count * scale)).powi(2),
                )
            },
        );
        S::fold(
            &bit_flip_local_mses,
            <f64 as Element<S>>::Array::default(),
            |a, b| <S as ZipMap<f64, f64, f64>>::zip_map(&a, b, |x, y| x + y),
        )
    }
}

fn masked_sum<S: Shape + ZipMap<f64, bool, f64>>(
    edge_set: &<f64 as Element<S>>::Array,
    mask: &<bool as Element<S>>::Array,
) -> f64
where
    bool: Element<S>,
    f64: Element<S>,
    <f64 as Element<S>>::Array: Sum,
{
    <S as ZipMap<f64, bool, f64>>::zip_map(
        &edge_set,
        &mask,
        |&edge, &mask_bit| {
            if mask_bit {
                edge
            } else {
                0f64
            }
        },
    )
    .sum()
}

trait GenMask
where
    Self: IncrementCountersMatrix<Self> + Bits + BitShape,
    bool: Element<Self::Shape>,
    u32: Element<Self::Shape>,
    [(usize, <u32 as Element<<Self as BitShape>::Shape>>::Array); 2]:
        Element<<Self as BitShape>::Shape>,
{
    fn gen_mask(
        matrix_counters: &<[(usize, <u32 as Element<Self::Shape>>::Array); 2] as Element<
            Self::Shape,
        >>::Array,
        value_counters: &[(usize, <u32 as Element<Self::Shape>>::Array); 2],
    ) -> Self;
}

impl<B: BitShape + Bits + IncrementFracCounters + IncrementCountersMatrix<B>> GenMask for B
where
    B::Shape: Mse
        + Min
        + FlipBool
        + ZipMap<u32, u32, f64>
        + ZipMap<f64, f64, f64>
        + ZipMap<<f64 as Element<<B as BitShape>::Shape>>::Array, f64, f64>
        + Map<f64, f64>
        + Map<<f64 as Element<B::Shape>>::Array, f64>
        + Map<[(usize, <u32 as Element<B::Shape>>::Array); 2], <f64 as Element<B::Shape>>::Array>,
    bool: Element<B::Shape>,
    u32: Element<B::Shape>,
    f64: Element<B::Shape>,
    <f64 as Element<B::Shape>>::Array: Element<<B as BitShape>::Shape> + Sum,
    [(usize, <u32 as Element<<B as BitShape>::Shape>>::Array); 2]: Element<<B as BitShape>::Shape>,
    (): Element<B::Shape, Array = B::Shape>,
    B::Shape: Default + Map<(), bool>,
{
    fn gen_mask(
        matrix_counters: &<[(usize, <u32 as Element<B::Shape>>::Array); 2] as Element<
            B::Shape,
        >>::Array,
        value_counters: &[(usize, <u32 as Element<<B as BitShape>::Shape>>::Array); 2],
    ) -> B {
        let values = bayes_magn::<B::Shape>(&value_counters);

        let edges = <B::Shape as Map<
            [(usize, <u32 as Element<B::Shape>>::Array); 2],
            <f64 as Element<B::Shape>>::Array,
        >>::map(&matrix_counters, |counters| {
            bayes_magn::<B::Shape>(&counters)
        });

        let ns =
            <B::Shape as Map<<f64 as Element<B::Shape>>::Array, f64>>::map(&edges, |edge_set| {
                edge_set.sum()
            });
        let mut mask = <B::Shape as Map<(), bool>>::map(&B::Shape::default(), |_| true);

        let local_avgs = <B::Shape as ZipMap<<f64 as Element<B::Shape>>::Array, f64, f64>>::zip_map(
            &edges,
            &ns,
            |edge_set, node_n| {
                <B::Shape as ZipMap<f64, f64, f64>>::zip_map(&edge_set, &values, |a, b| a * b).sum()
                    / node_n
            },
        );
        let mut cur_mse = B::Shape::single_mse(&edges, &local_avgs, &mask);
        let mut is_optima = false;
        while !is_optima {
            let bit_flip_mses = B::Shape::bit_flip_mses(&edges, &local_avgs, &mask);
            let (min_index, min_val) = <B::Shape as Min>::min(&bit_flip_mses).unwrap();
            if min_val < cur_mse {
                <B::Shape as FlipBool>::flip_bool(&mut mask, min_index);
                cur_mse = min_val;
            } else {
                is_optima = true;
            }
        }
        <B as Bits>::bitpack(&mask)
    }
}

type InputType = [u8; 3];
type InputShape = [[(); 8]; 3];
type FloatInputType = <f64 as Element<InputShape>>::Array;

fn main() {
    let examples = vec![
        ([0b_0010_0000, 0b_1111_1000, 0b_0000_0000], 0),
        ([0b_0010_0000, 0b_1111_0100, 0b_0000_0000], 0),
        ([0b_0010_0000, 0b_1101_0010, 0b_0000_0000], 0),
        ([0b_0010_0000, 0b_0000_0001, 0b_0010_0000], 0),
        ([0b_1111_0000, 0b_0000_1111, 0b_1111_1111], 0),
        ([0b_1111_0000, 0b_0000_1111, 0b_1111_1111], 0),
        ([0b_1111_1111, 0b_0000_1111, 0b_1111_1111], 0),
        ([0b_1111_1111, 0b_0010_0111, 0b_1101_1110], 1),
        ([0b_1111_1111, 0b_0010_1011, 0b_1101_1110], 1),
        ([0b_1111_1111, 0b_0000_1101, 0b_1101_1110], 1),
        ([0b_1101_1011, 0b_1111_1110, 0b_1101_1111], 1),
        ([0b_0000_1101, 0b_1111_0000, 0b_0000_0001], 1),
        ([0b_0000_1110, 0b_1111_0000, 0b_0000_0001], 1),
        ([0b_0000_0000, 0b_1111_0000, 0b_0000_0001], 1),
    ];

    let mut cocorrelation_counters = <[(
        usize,
        <u32 as Element<<InputType as BitShape>::Shape>>::Array,
    ); 2] as Element<<InputType as BitShape>::Shape>>::Array::default(
    );
    let mut counters = <[(
        usize,
        <u32 as Element<<InputType as BitShape>::Shape>>::Array,
    ); 2]>::default();
    for (example, class) in &examples {
        example.increment_counters_matrix(&mut cocorrelation_counters, example);
        example.increment_frac_counters(&mut counters[*class]);
    }
    let mask = <InputType as GenMask>::gen_mask(&cocorrelation_counters, &counters);
    dbg!(mask);
    for i in 0..3 {
        println!("{:08b}", mask[i]);
    }
}
