#![feature(const_generics)]

use bitnn::bits::{BitLen, FlipBit, GetBit};
use bitnn::count::{Counters, IncrementCounters, IncrementFracCounters};

trait Shape {
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

trait Element<S: Shape> {
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
    //<I as Element<S>>::Array: Map<S, I, O>,
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
    fn fold<F: Fn(B, &I) -> B>(input: &I, mut acc: B, fold_fn: F) -> B {
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
                if let Some((index, min)) = cur_min {
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
    fn flip_bool(bools: &mut bool, index: ()) {
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

trait IncrementCountersMatrix<T> {
    type MatrixCounterType;
    fn increment_counters_matrix(&self, counters_matrix: &mut Self::MatrixCounterType, target: &T);
}

impl<I: IncrementCounters + IncrementFracCounters> IncrementCountersMatrix<u8> for I {
    type MatrixCounterType =
        [[(usize, <Self as IncrementCounters>::BitCounterType); 2]; <u8>::BIT_LEN];
    fn increment_counters_matrix(
        &self,
        counters_matrix: &mut Self::MatrixCounterType,
        target: &u8,
    ) {
        for i in 0..<u8>::BIT_LEN {
            self.increment_frac_counters(&mut counters_matrix[i][target.bit(i) as usize]);
        }
    }
}

impl<I: IncrementCountersMatrix<T>, T, const L: usize> IncrementCountersMatrix<[T; L]> for I {
    type MatrixCounterType = [<Self as IncrementCountersMatrix<T>>::MatrixCounterType; L];
    fn increment_counters_matrix(
        &self,
        counters_matrix: &mut Self::MatrixCounterType,
        target: &[T; L],
    ) {
        for w in 0..L {
            self.increment_counters_matrix(&mut counters_matrix[w], &target[w]);
        }
    }
}

trait MatrixCounters {
    type FloatRatioType;
    fn elementwise_add(&mut self, other: &Self);
    fn bayes(&self) -> Self::FloatRatioType;
}

impl<InputCounters: Counters> MatrixCounters for [[(usize, InputCounters); 2]; 8]
where
    [InputCounters::FloatRatioType; 8]: Default,
{
    type FloatRatioType = [InputCounters::FloatRatioType; 8];
    fn elementwise_add(&mut self, other: &Self) {
        for b in 0..8 {
            for c in 0..2 {
                self[b][c].0 += other[b][c].0;
                self[b][c].1.elementwise_add(&other[b][c].1);
            }
        }
    }
    fn bayes(&self) -> Self::FloatRatioType {
        let mut target = <[InputCounters::FloatRatioType; 8]>::default();
        for b in 0..8 {
            let n = (self[b][0].0 + self[b][1].0) as f64;
            let na = self[b][1].0 as f64;
            let pa = na / n;
            target[b] = self[b][1].1.bayes(na, &self[b][0].1, n, pa);
        }
        target
    }
}

impl<T: MatrixCounters, const L: usize> MatrixCounters for [T; L]
where
    [T::FloatRatioType; L]: Default,
{
    type FloatRatioType = [T::FloatRatioType; L];
    fn elementwise_add(&mut self, other: &Self) {
        for i in 0..L {
            self[i].elementwise_add(&other[i]);
        }
    }
    fn bayes(&self) -> Self::FloatRatioType {
        let mut target = <[T::FloatRatioType; L]>::default();
        for i in 0..L {
            target[i] = self[i].bayes();
        }
        target
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
            + ZipMap<f64, bool, f64>
            + Map<<f64 as Element<S>>::Array, f64>
            + Map<f64, f64>
            + ZipMap<f64, f64, f64>
            + ZipMap<bool, f64, f64>
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
            <S as ZipMap<f64, bool, f64>>::zip_map(&edge_set, &mask, |&edge, &mask_bit| {
                if mask_bit {
                    edge
                } else {
                    0f64
                }
            })
            .sum()
        });
        dbg!(local_counts.sum());
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
        let bit_flip_local_counts = <S as ZipMap<
            <f64 as Element<S>>::Array,
            f64,
            <f64 as Element<S>>::Array,
        >>::zip_map(
            &edges,
            &local_avgs,
            |edge_set, local_avg| {
                let sum =
                    <S as ZipMap<f64, bool, f64>>::zip_map(&edge_set, &mask, |&edge, &mask_bit| {
                        if mask_bit {
                            edge
                        } else {
                            0f64
                        }
                    })
                    .sum();
                <S as ZipMap<bool, f64, f64>>::zip_map(&mask, &edge_set, |&mask_bit, &edge| {
                    if mask_bit {
                        sum - edge
                    } else {
                        sum + edge
                    }
                })
            },
        );
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

type InputType = [u8; 3];
type InputShape = [[(); 8]; 3];
type MatrixShape = <InputShape as Element<InputShape>>::Array;
type FloatInputType = <f64 as Element<InputShape>>::Array;
//type InputCounters = <InputType as BitWrap<u32>>::Wrap;
//type CocorrelationCounters = <InputType as BitWrap<InputCounters>>::Wrap;

fn main() {
    let examples = vec![
        ([0b_0000_0000, 0b_1111_0000, 0b_0000_0000], 0),
        ([0b_0000_0000, 0b_1111_0000, 0b_0000_0000], 0),
        ([0b_0000_0000, 0b_1111_0000, 0b_0000_0000], 0),
        ([0b_0000_0000, 0b_0000_0000, 0b_0000_0000], 0),
        ([0b_1111_0000, 0b_0000_1111, 0b_1111_1111], 0),
        ([0b_1111_0000, 0b_0000_1111, 0b_1111_1111], 0),
        ([0b_1111_1111, 0b_0000_1111, 0b_1111_1111], 0),
        ([0b_1111_1111, 0b_0000_1111, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_0000_1111, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_0000_1111, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_1111_1111, 0b_1111_1111], 1),
        ([0b_0000_1111, 0b_1111_0000, 0b_0000_0000], 1),
        ([0b_0000_1111, 0b_1111_0000, 0b_0000_0000], 1),
        ([0b_0000_0000, 0b_1111_0000, 0b_0000_0000], 1),
    ];

    let mut cocorrelation_counters =
        <<InputType as IncrementCountersMatrix<InputType>>::MatrixCounterType>::default();
    let mut counters = <[(usize, <InputType as IncrementCounters>::BitCounterType); 2]>::default();
    for (example, class) in &examples {
        example.increment_counters_matrix(&mut cocorrelation_counters, example);
        example.increment_frac_counters(&mut counters[*class]);
    }

    let values = bayes_magn::<InputShape>(&counters);
    dbg!(values);

    let edges = InputShape::map(&cocorrelation_counters, |counters| {
        bayes_magn::<InputShape>(&counters)
    });

    let ns = <InputShape as Map<FloatInputType, f64>>::map(&edges, |edge_set| edge_set.sum());
    //dbg!(ns);
    let mut mask = [[true; 8]; 3];

    let n = <[[(); 8]; 3]>::N;
    //let local_avgs = <[[f64; 8]; 3] as MatrixLoss<InputType>>::local_avg_val(&edges, &values);
    let local_avgs = InputShape::zip_map(&edges, &ns, |edge_set, node_n| {
        <InputShape as ZipMap<f64, f64, f64>>::zip_map(&edge_set, &values, |a, b| a * b).sum()
            / node_n
    });
    let avg = local_avgs.sum() / n as f64;
    let mut cur_mse = InputShape::single_mse(&edges, &local_avgs, &mask);
    dbg!(cur_mse);
    let bit_flip_mses = InputShape::bit_flip_mses(&edges, &local_avgs, &mask);
    dbg!(mask);
    dbg!(bit_flip_mses);
    let (min_index, min_val) = <InputShape as Min>::min(&bit_flip_mses).unwrap();
    dbg!(min_index);
    dbg!(min_val);
    if min_val < cur_mse {
        dbg!("flipping");
        <InputShape as FlipBool>::flip_bool(&mut mask, min_index);
        cur_mse = min_val;
    }
    dbg!(mask);
}
