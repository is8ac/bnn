use bitnn::bits::{b4, b8, BitArray, BitWord, Distance, IncrementCooccurrenceMatrix};
use bitnn::shape::Element;
use bitnn::weight::UnsupervisedCluster;

type InputType = [b8; 2];
type TargetType = b4;

fn main() {
    let examples: Vec<[b8; 2]> = vec![
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1100_1100), b8(0b_1100_1100)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1111_1111), b8(0b_0000_0000)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
        [b8(0b_1010_1010), b8(0b_1010_1010)],
    ];

    let mut counts = Box::<
        <[(
            usize,
            <u32 as Element<<InputType as BitArray>::BitShape>>::Array,
        ); 2] as Element<<InputType as BitArray>::BitShape>>::Array,
    >::default();
    for example in &examples {
        example.increment_cooccurrence_matrix(&mut *counts, example);
    }
    let weights = <[(
        <(
            <InputType as BitArray>::WordType,
            <InputType as BitArray>::WordType,
        ) as Element<<InputType as BitArray>::WordShape>>::Array,
        u32,
    ); TargetType::BIT_LEN] as UnsupervisedCluster<InputType, TargetType>>::unsupervised_cluster(
        &counts,
        examples.len(),
    );
    for cluster in &weights {
        println!("{:?}", cluster);
    }
    for example in &examples {
        let acts: Vec<_> = weights.iter().map(|x| x.0.distance(example)).collect();
        println!("{:?}", acts);
    }
}
