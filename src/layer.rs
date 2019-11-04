use crate::bits::{BitArray, BitMul, Classify, Distance};
use crate::count::CountBits;
use crate::shape::Element;
use crate::weight::{GenParamClasses, GenParamSet, InputBits};
use rayon::prelude::*;
use std::boxed::Box;

pub trait FC<I, O, C>
where
    Self: Sized,
{
    fn apply(name: &str, examples: &Vec<(I, usize)>) -> (Self, Vec<(O, usize)>);
}

impl<
        T: BitMul<O, Input = I> + Sized + Send + Sync,
        I: BitArray
            + GenParamClasses<[(); C]>
            + GenParamSet<O, [(); C]>
            + Send
            + Sync
            + CountBits<{ C }>
            + InputBits<O, TrinaryWeights = T>,
        O: BitArray + Sync + Send,
        const C: usize,
    > FC<I, O, [(); C]> for T
where
    <(I::WordType, I::WordType) as Element<I::WordShape>>::Array: Send + Sync,
    u32: Element<I::BitShape> + Element<O::BitShape>,
    (I::WordType, I::WordType): Element<I::WordShape>,
    [(
        <(I::WordType, I::WordType) as Element<I::WordShape>>::Array,
        u32,
    ); C]: Classify<Input = I, ClassesShape = [(); C]>,
    bool: Element<I::BitShape> + Element<O::BitShape>,
    <u32 as Element<I::BitShape>>::Array: Element<I::BitShape>,
{
    fn apply(name: &str, examples: &Vec<(I, usize)>) -> (Self, Vec<(O, usize)>) {
        let (value_counters, matrix_counters, n_examples): (
            Box<[(usize, <u32 as Element<<I as BitArray>::BitShape>>::Array); C]>,
            Box<<<u32 as Element<I::BitShape>>::Array as Element<I::BitShape>>::Array>,
            usize,
        ) = I::count_bits(&examples);
        let layer_weights = <I as GenParamSet<O, [(); C]>>::gen_parm_set(
            n_examples,
            &value_counters,
            &matrix_counters,
        );
        let class_weights = <I as GenParamClasses<[(); C]>>::gen_parm_classes(
            n_examples,
            &value_counters,
            &matrix_counters,
        );
        let n_correct: u64 = examples
            .par_iter()
            .map(|(image, class)| (class_weights.max_class(image) == *class) as u64)
            .sum();
        println!("acc: {}%", (n_correct as f64 / n_examples as f64) * 100f64);
        let new_examples: Vec<_> = examples
            .par_iter()
            .map(|(image, class)| (layer_weights.bit_mul(image), *class))
            .collect();
        (layer_weights, new_examples)
    }
}

//trait Conv2D3x3<I, O> {
//    fn apply(examples: &Vec<()>)-> Vec<(_, usize)>;
//}
