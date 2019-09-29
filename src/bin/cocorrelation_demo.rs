#![feature(const_generics)]

use bitnn::bits::{BitLen, GetBit};
use bitnn::count::{Counters, IncrementCounters, IncrementFracCounters};

// f64 values
trait Weights {
    const N: usize;
    fn mul_sum(&self, other: &Self) -> f64;
}

impl Weights for f64 {
    const N: usize = 1;
    fn mul_sum(&self, &other: &f64) -> f64 {
        *self * other
    }
}

impl<T: Weights, const L: usize> Weights for [T; L] {
    const N: usize = T::N * L;
    fn mul_sum(&self, other: &Self) -> f64 {
        let mut sum = 0f64;
        for i in 0..L {
            sum += self[i].mul_sum(&other[i]);
        }
        sum
    }
}

//trait Array<T> {
//    type T;
//    fn map<O, F: Fn(&T) -> O>(&self) -> Array<O>;
//}

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

//fn local_avg(node_val: u64, edges) -> f64 {
//
//}

type InputType = u8;
//type InputCounters = <InputType as BitWrap<u32>>::Wrap;
//type CocorrelationCounters = <InputType as BitWrap<InputCounters>>::Wrap;

fn main() {
    let examples = vec![
        (0b_1111_1111, 0),
        (0b_0010_1011, 0),
        (0b_0010_1001, 0),
        (0b_0011_1011, 0),
        (0b_0011_1011, 0),
        (0b_0011_1011, 0),
        (0b_0011_1011, 0),
        (0b_0010_1001, 0),
        (0b_0011_1011, 0),
        (0b_1101_1100, 1),
        (0b_1101_0110, 1),
        (0b_1101_0110, 1),
        (0b_1100_0100, 1),
        (0b_1101_0110, 1),
        (0b_1101_0110, 1),
        (0b_1101_0000, 1),
        (0b_1101_0010, 1),
        (0b_1101_0010, 1),
        (0b_1101_1010, 1),
        (0b_0001_1000, 1),
    ];

    let mut cocorrelation_counters =
        <<InputType as IncrementCountersMatrix<InputType>>::MatrixCounterType>::default();
    let mut counters = <[(usize, <InputType as IncrementCounters>::BitCounterType); 2]>::default();
    for (example, class) in &examples {
        example.increment_counters_matrix(&mut cocorrelation_counters, example);
        example.increment_frac_counters(&mut counters[*class]);
    }
    let n = (counters[0].0 + counters[1].0) as f64;
    let na = counters[0].0 as f64;
    let pa = na / n;
    dbg!(pa);

    let values = counters[0].1.bayes(na, &counters[1].1, n, pa);
    dbg!(values);

    let edges = cocorrelation_counters.bayes();
    for i in 0..8 {
        for t in 0..8 {
            print!("{:.3}, ", edges[i][t]);
        }
        print!("\n");
    }
}

//   | 0  | 1
// A | 50 | 50
// B | 50 | 50
