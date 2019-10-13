#![feature(const_generics)]
use bitnn::mask::{BitShape, Element, GenMask, IncrementCountersMatrix, IncrementFracCounters};

type InputType = [u8; 3];

fn main() {
    let examples = vec![
        ([0b_1111_1111, 0b_1000_0000, 0b_1110_1111_u8], 0),
        ([0b_1111_1111, 0b_1100_0000, 0b_1110_1111], 0),
        ([0b_1111_1111, 0b_1110_0000, 0b_1110_1111], 0),
        ([0b_1111_1111, 0b_1111_0000, 0b_1110_1111], 0),
        ([0b_0000_0000, 0b_0111_0000, 0b_0000_0000], 0),
        ([0b_0000_0000, 0b_0011_0000, 0b_0000_0000], 0),
        ([0b_0000_0000, 0b_0001_0000, 0b_0000_0000], 0),
        ([0b_1111_1111, 0b_0111_0000, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_0011_0000, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_0001_0000, 0b_1111_1111], 1),
        ([0b_1111_1111, 0b_0000_0000, 0b_1110_1111], 1),
        ([0b_0000_0000, 0b_1000_0000, 0b_0001_0000], 1),
        ([0b_0000_0000, 0b_1100_0000, 0b_0001_0000], 1),
        ([0b_0000_0000, 0b_1110_0000, 0b_0001_0000], 1),
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
