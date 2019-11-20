use bitnn::bits::{b8, IncrementHammingDistanceMatrix};
use bitnn::layer::CountBits;
use bitnn::shape::{Element, Fold, Map, Shape, ZipFold};

trait ClusteringMse<T: Shape>
where
    Self: Shape + Sized,
    bool: Element<Self>,
    f32: Element<Self>,
    <bool as Element<Self>>::Array: Element<T>,
    <f32 as Element<Self>>::Array: Element<Self>,
{
    fn clustering_mse(
        edges: &Box<<<f32 as Element<Self>>::Array as Element<Self>>::Array>,
        signs: &<<bool as Element<Self>>::Array as Element<T>>::Array,
    ) -> f32;
}

impl<
        S: Shape
            + Sized
            + ZipFold<u32, bool, bool>
            + ZipFold<f32, bool, f32>
            + ZipFold<f32, bool, <f32 as Element<S>>::Array>,
        T: Shape + Fold<f32, <bool as Element<S>>::Array>,
    > ClusteringMse<T> for S
where
    bool: Element<Self>,
    f32: Element<Self>,
    <bool as Element<Self>>::Array: Element<T>,
    <f32 as Element<Self>>::Array: Element<Self>,
{
    fn clustering_mse(
        edges: &Box<<<f32 as Element<S>>::Array as Element<S>>::Array>,
        signs: &<<bool as Element<S>>::Array as Element<T>>::Array,
    ) -> f32 {
        let sum_loss = <T as Fold<f32, <bool as Element<S>>::Array>>::fold(
            &signs,
            0f32,
            |acc, feature_signs| {
                let intra_distence = <S as ZipFold<f32, bool, <f32 as Element<S>>::Array>>::zip_fold(
                    feature_signs,
                    &edges,
                    0f32,
                    |acc, source_sign, edges| {
                        acc + <S as ZipFold<f32, bool, f32>>::zip_fold(
                            feature_signs,
                            edges,
                            0f32,
                            |acc, target_sign, &edge| {
                                if target_sign ^ source_sign {
                                    -edge
                                } else {
                                    edge
                                }
                            },
                        )
                    },
                ) / InputShape::N as f32;
                let inter_closeness = <T as Fold<f32, <bool as Element<S>>::Array>>::fold(
                    &signs,
                    0f32,
                    |acc, other_signs| {
                        acc + <S as ZipFold<u32, bool, bool>>::zip_fold(
                            feature_signs,
                            other_signs,
                            0u32,
                            |acc, a, b| acc + (a == b) as u32,
                        ) as f32
                            / InputShape::N as f32
                    },
                ) / TargetShape::N as f32;
                dbg!((intra_distence, inter_closeness));
                acc + inter_closeness.powi(2) + intra_distence.powi(2)
            },
        );
        sum_loss as f32 / TargetShape::N as f32
    }
}
type InputShape = [[(); 8]; 3];
type TargetShape = [(); 3];

fn main() {
    let examples: Vec<[b8; 3]> = vec![
        [b8(0b_1011_0110), b8(0b_1011_0110), b8(0b_1011_0110)],
        [b8(0b_1011_0110), b8(0b_1011_0110), b8(0b_1011_0110)],
        [b8(0b_1011_0110), b8(0b_1011_0110), b8(0b_1011_0110)],
        [b8(0b_1011_0110), b8(0b_1011_0110), b8(0b_1011_0110)],
        [b8(0b_1011_0110), b8(0b_1011_0110), b8(0b_1011_0110)],
        [b8(0b_1011_0110), b8(0b_1011_0110), b8(0b_1011_0110)],
        [b8(0b_1011_0110), b8(0b_1011_0110), b8(0b_1011_0110)],
        [b8(0b_1110_0011), b8(0b_1110_0011), b8(0b_1110_0011)],
        [b8(0b_1110_0011), b8(0b_1110_0011), b8(0b_1110_0011)],
        [b8(0b_1110_0011), b8(0b_1110_0011), b8(0b_1110_0011)],
        [b8(0b_1110_0011), b8(0b_1110_0011), b8(0b_1110_0011)],
        [b8(0b_1110_0011), b8(0b_1110_0011), b8(0b_1110_0011)],
        [b8(0b_1110_0011), b8(0b_1110_0011), b8(0b_1110_0011)],
        [b8(0b_1110_0011), b8(0b_1110_0011), b8(0b_1110_0011)],
        [b8(0b_0100_1011), b8(0b_0100_1011), b8(0b_0100_1011)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
        [b8(0b_1111_0000), b8(0b_1111_0000), b8(0b_1111_0000)],
    ];
    let mut counts =
        Box::<<<u32 as Element<InputShape>>::Array as Element<InputShape>>::Array>::default();
    for example in &examples {
        example.increment_hamming_distance_matrix(&mut *counts, example);
    }
    let n_examples = examples.len();
    //dbg!(counts);
    dbg!(n_examples);
    let edges = <Box<InputShape> as Map<
        <u32 as Element<InputShape>>::Array,
        <f32 as Element<InputShape>>::Array,
    >>::map(&counts, |row| {
        <InputShape as Map<u32, f32>>::map(row, |&c| c as f32 / n_examples as f32)
    });
    let mut signs = <TargetShape as Map<(), <bool as Element<InputShape>>::Array>>::map(
        &TargetShape::default(),
        |_| <InputShape as Map<(), bool>>::map(&InputShape::default(), |_| true),
    );
    //let signs = [
    //    [true, true, false, true, false, false, false, true],
    //    [false, false, true, true, true, false, false, false],
    //    [false, true, true, false, true, true, true, true],
    //    [true, true, false, false, false, true, false, true],
    //    [false, false, true, false, false, true, true, true],
    //];
    let avg_loss = <InputShape as ClusteringMse<TargetShape>>::clustering_mse(&edges, &signs);
    dbg!(avg_loss);
}
