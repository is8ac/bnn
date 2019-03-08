extern crate rayon;
use rayon::prelude::*;
use std::marker::PhantomData;

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

impl BitLen for u32 {
    const BIT_LEN: usize = 32;
}

macro_rules! array_bit_len {
    ($len:expr) => {
        impl<T: BitLen> BitLen for [T; $len] {
            const BIT_LEN: usize = $len * T::BIT_LEN;
        }
    };
}

array_bit_len!(2);
array_bit_len!(3);

trait FlipBit {
    fn flip_bit(&mut self, b: usize);
}

impl FlipBit for u32 {
    fn flip_bit(&mut self, index: usize) {
        *self ^= 1 << index
    }
}

macro_rules! array_flip_bit {
    ($len:expr) => {
        impl<T: BitLen + FlipBit> FlipBit for [T; $len] {
            fn flip_bit(&mut self, index: usize) {
                self[index / T::BIT_LEN].flip_bit(index % T::BIT_LEN);
            }
        }
    };
}
array_flip_bit!(2);
array_flip_bit!(3);

pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
}

impl HammingDistance for u32 {
    #[inline(always)]
    fn hamming_distance(&self, other: &u32) -> u32 {
        (self ^ other).count_ones()
    }
}

macro_rules! array_hamming_distance {
    ($len:expr) => {
        impl<T: HammingDistance> HammingDistance for [T; $len] {
            fn hamming_distance(&self, other: &[T; $len]) -> u32 {
                let mut distance = 0u32;
                for i in 0..$len {
                    distance += self[i].hamming_distance(&other[i]);
                }
                distance
            }
        }
    };
}

array_hamming_distance!(2);
array_hamming_distance!(3);

pub trait Apply<I, O> {
    fn apply(&self, input: &I) -> O;
}

impl<I: HammingDistance + BitLen> Apply<I, u32> for [I; 32] {
    fn apply(&self, input: &I) -> u32 {
        let mut target = 0u32;
        for i in 0..32 {
            target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2)) as u32) << i;
        }
        target
    }
}
pub trait IsCorrect<I> {
    fn is_correct(&self, target: u8, input: I) -> bool;
}
impl<I: HammingDistance> IsCorrect<I> for [I; 10] {
    // the max activation is the target.
    #[inline(always)]
    fn is_correct(&self, target: u8, input: I) -> bool {
        let max = self[target as usize].hamming_distance(&input);
        for i in 0..10 {
            if i != target as usize {
                if self[i].hamming_distance(&input) >= max {
                    return false;
                }
            }
        }
        true
    }
}

pub trait ObjectiveEvalCreator<InputPatch, Weights, Embedding> {
    type ObjectiveEvalType;
    fn new_obj_eval(
        &self,
        weights: &Weights,
        head: &[Embedding; 10],
        examples: &[(u8, InputPatch)],
    ) -> Self::ObjectiveEvalType;
}

pub trait ObjectiveEval<InputPatch, Weights, Embedding> {
    fn flip_weights_bit(&mut self, o: usize, i: usize);
    fn flip_head_bit(&mut self, o: usize, i: usize);
    fn obj(&self) -> u64;
}

pub struct TestCPUObjectiveEvalCreator<InputPatch, Weights, Embedding> {
    input_pixel_type: PhantomData<InputPatch>,
    weights_type: PhantomData<Weights>,
    embedding_type: PhantomData<Embedding>,
}

impl TestCPUObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> {
    fn new() -> Self {
        TestCPUObjectiveEvalCreator {
            input_pixel_type: PhantomData,
            weights_type: PhantomData,
            embedding_type: PhantomData,
        }
    }
}

impl ObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    for TestCPUObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
{
    type ObjectiveEvalType =
        TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>;
    fn new_obj_eval(
        &self,
        weights: &[[[[u32; 2]; 3]; 3]; 32],
        head: &[u32; 10],
        examples: &[(u8, [[[u32; 2]; 3]; 3])],
    ) -> TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> {
        TestCPUObjectiveEval {
            weights: *weights,
            head: *head,
            examples: examples.iter().cloned().collect(),
        }
    }
}

pub struct TestCPUObjectiveEval<InputPatch, Weights, Embedding> {
    weights: Weights,
    head: [Embedding; 10],
    examples: Vec<(u8, InputPatch)>,
}

// This is a slow implementation of obj() and should not be used if performance is desired.
impl ObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    for TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
{
    fn flip_weights_bit(&mut self, o: usize, i: usize) {
        self.weights[o].flip_bit(i);
    }
    fn flip_head_bit(&mut self, o: usize, i: usize) {
        self.head[o].flip_bit(i);
    }
    fn obj(&self) -> u64 {
        self.examples
            .par_iter()
            .map(|(class, patch)| {
                let embedding = self.weights.apply(patch);
                self.head.is_correct(*class, embedding) as u64
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEval, TestCPUObjectiveEvalCreator,
    };
    use rand::prelude::*;
    use rand_hc::Hc128Rng;

    #[test]
    fn test_cpu_obj() {
        const N_EXAMPLES: usize = 1000000;
        let mut rng = Hc128Rng::seed_from_u64(42);
        let weights: [[[[u32; 2]; 3]; 3]; 32] = rng.gen();
        let head: [u32; 10] = rng.gen();
        let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..N_EXAMPLES)
            .map(|_| (rng.gen_range(0, 10), rng.gen()))
            .collect();

        let eval_creator: TestCPUObjectiveEvalCreator<
            [[[u32; 2]; 3]; 3],
            [[[[u32; 2]; 3]; 3]; 32],
            u32,
        > = TestCPUObjectiveEvalCreator::new();
        let mut obj_eval: TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> =
            eval_creator.new(&weights, &head, &examples);

        let obj1: u64 = obj_eval.obj();
        dbg!(obj1);
        let avg_obj = obj1 as f64 / N_EXAMPLES as f64;
        dbg!(avg_obj);
        assert!(avg_obj > 0.07);
        assert!(avg_obj < 0.09);

        obj_eval.flip_weights_bit(22, 5);
        let obj2: u64 = obj_eval.obj();
        dbg!(obj2);
        let avg_obj = obj2 as f64 / N_EXAMPLES as f64;
        dbg!(avg_obj);
        assert!(avg_obj > 0.07);
        assert!(avg_obj < 0.09);

        assert_ne!(obj1, obj2);

        obj_eval.flip_weights_bit(22, 5);
        let obj3: u64 = obj_eval.obj();
        assert_eq!(obj1, obj3);
    }
}
