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

    let values = {
        let n = (counters[0].0 + counters[1].0) as f64;
        let na = counters[0].0 as f64;
        let pa = na / n;
        <InputShape as ZipMap<u32, u32, f64>>::zip_map(&counters[0].1, &counters[1].1, |&a, &b| {
            let pb = (a + b) as f64 / n;
            let pba = a as f64 / na;
            let pab = (pba * pa) / pb;
            ((pab * 2f64) - 1f64).abs()
        })
    };
    dbg!(values);

    let edges = <InputShape as Map<
        [(usize, <u32 as Element<InputShape>>::Array); 2],
        <f64 as Element<InputShape>>::Array,
    >>::map(&cocorrelation_counters, |counters| {
        let n = (counters[0].0 + counters[1].0) as f64;
        let na = counters[0].0 as f64;
        let pa = na / n;
        <InputShape as ZipMap<u32, u32, f64>>::zip_map(&counters[0].1, &counters[1].1, |&a, &b| {
            let pb = (a + b) as f64 / n;
            let pba = a as f64 / na;
            let pab = (pba * pa) / pb;
            ((pab * 2f64) - 1f64).abs()
        })
    });

    for c in 0..3 {
        println!("[",);
        for i in 0..8 {
            print!("[");
            for tc in 0..3 {
                print!("[");
                for t in 0..8 {
                    print!("{:.2}, ", edges[c][i][tc][t]);
                }
                print!("], ");
            }
            println!("]");
        }
        println!("]",);
    }

    let ns = <InputShape as Map<FloatInputType, f64>>::map(&edges, |edge_set| edge_set.sum());
    //dbg!(ns);
    //let mut mask = [0b_1011_0110, 0b_0111_0101, 0b_0100_0101];
    //let mut mask = [[true, true, true, false, true, false, true, false]; 3];
    let mut mask = [[true; 8]; 3];

    let n = <[[(); 8]; 3]>::N;
    //let local_avgs = <[[f64; 8]; 3] as MatrixLoss<InputType>>::local_avg_val(&edges, &values);
    let local_avgs = InputShape::zip_map(&edges, &ns, |edge_set, node_n| {
        <InputShape as ZipMap<f64, f64, f64>>::zip_map(&edge_set, &values, |a, b| a * b).sum()
            / node_n
    });
    let avg = local_avgs.sum() / n as f64;
    let mut cur_mse = {
        let local_counts = InputShape::map(&edges, |edge_set| {
            <InputShape as ZipMap<f64, bool, f64>>::zip_map(&edge_set, &mask, |&edge, &mask_bit| {
                if mask_bit {
                    edge
                } else {
                    0f64
                }
            })
            .sum()
        });
        dbg!(local_counts.sum());
        let scale = avg / (local_counts.sum() / n as f64);
        InputShape::zip_map(&local_avgs, &local_counts, |a, b| (a - (b * scale)).powi(2)).sum()
    };
    dbg!(cur_mse);
    let bit_flip_local_counts =
        <InputShape as ZipMap<FloatInputType, f64, FloatInputType>>::zip_map(
            &edges,
            &local_avgs,
            |edge_set, local_avg| {
                let sum = <InputShape as ZipMap<f64, bool, f64>>::zip_map(
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
                .sum();
                <InputShape as ZipMap<bool, f64, f64>>::zip_map(
                    &mask,
                    &edge_set,
                    |&mask_bit, &edge| {
                        if mask_bit {
                            sum - edge
                        } else {
                            sum + edge
                        }
                    },
                )
            },
        );
    let bit_flip_sums = <InputShape as Fold<FloatInputType, FloatInputType>>::fold(
        &bit_flip_local_counts,
        FloatInputType::default(),
        |a, b| <InputShape as ZipMap<f64, f64, f64>>::zip_map(&a, b, |x, y| x + y),
    );
    let bit_flip_scales =
        <InputShape as Map<f64, f64>>::map(&bit_flip_sums, |sum| avg / (sum / n as f64));
    dbg!(bit_flip_scales);
    let bit_flip_local_mses = <InputShape as ZipMap<FloatInputType, f64, FloatInputType>>::zip_map(
        &bit_flip_local_counts,
        &local_avgs,
        |local_counts, local_avg| {
            <InputShape as ZipMap<f64, f64, f64>>::zip_map(
                local_counts,
                &bit_flip_scales,
                |count, scale| (local_avg - (count * scale)).powi(2),
            )
        },
    );
    let bit_flip_mses = <InputShape as Fold<FloatInputType, FloatInputType>>::fold(
        &bit_flip_local_mses,
        FloatInputType::default(),
        |a, b| <InputShape as ZipMap<f64, f64, f64>>::zip_map(&a, b, |x, y| x + y),
    );
    dbg!(bit_flip_mses);
    let mse_diffs = <InputShape as Map<f64, f64>>::map(&bit_flip_mses, |bitflip| cur_mse - bitflip);
    dbg!(mse_diffs);
}
