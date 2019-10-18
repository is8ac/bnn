#![feature(const_generics)]
use bitnn::mask::{
    BitShape, Element, GenMask, IncrementFracCounters, IncrementHammingDistanceMatrix,
};
use std::boxed::Box;

type InputType = [u8; 3];

fn main() {
    let examples = vec![
        ([0b_0000_0000, 0b_1001_0000, 0b_0000_0000_u8], 0),
        ([0b_0000_0000, 0b_1100_0000, 0b_0000_0000], 0),
        ([0b_1111_1111, 0b_1110_1111, 0b_1111_1111], 0),
        ([0b_1111_1111, 0b_1111_1111, 0b_1111_1111], 0),
        ([0b_0000_0000, 0b_0111_0000, 0b_0000_0000], 0),
        ([0b_0000_0000, 0b_0011_0000, 0b_0000_0000], 0),
        ([0b_0000_0000, 0b_1001_0000, 0b_0000_0000], 0),
        ([0b_1111_1111, 0b_0110_1111, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_0011_1111, 0b_1111_1111], 1),
        ([0b_0000_0000, 0b_0001_0000, 0b_0000_0000], 1),
        ([0b_1111_1111, 0b_0000_1111, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_1000_1111, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_1100_1111, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_1110_1111, 0b_1111_1111], 1),
    ]; // [0b_0000_0000, 0b_1111_1110, 0b_0000_0000]

    let mut dist_matrix_counters = Box::<
        <<u32 as Element<<InputType as BitShape>::Shape>>::Array as Element<
            <InputType as BitShape>::Shape,
        >>::Array,
    >::default();
    let mut value_counters = Box::<
        [(
            usize,
            <u32 as Element<<InputType as BitShape>::Shape>>::Array,
        ); 2],
    >::default();
    for (example, class) in &examples {
        example.increment_hamming_distance_matrix(&mut *dist_matrix_counters, example);
        example.increment_frac_counters(&mut value_counters[*class]);
    }
    //dbg!(&dist_matrix_counters);
    let mask =
        <InputType as GenMask>::gen_mask(&dist_matrix_counters, examples.len(), &value_counters);
    dbg!(mask);
    for i in 0..3 {
        println!("{:08b}", mask[i]);
    }
}
