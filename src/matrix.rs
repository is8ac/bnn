use crate::bits::{
    b32, t32, BitMap, BitPack, FromBool, IncrementCounters, PackedMap, SIMDincrementCounters,
    SIMDword32, BMA,
};
use crate::count::ElementwiseAdd;
use crate::ecc;
use crate::shape::{IndexGet, IndexMap, LongDefault, Map, Pack, Shape, ZipMap};
use rayon::prelude::*;
use std::ops::{Add, AddAssign};
use std::time::Instant;

pub trait CacheLocalMatrixBatch<I>
where
    Self: BitPack<bool>
        + Pack<u8>
        + Pack<u32>
        + Pack<[(u32, <I as Pack<u32>>::T); 2]>
        + Pack<
            [(
                u32,
                <<I as SIMDincrementCounters>::WordShape as Pack<[u32; 32]>>::T,
            ); 2],
        > + Pack<[[u32; 32]; 2]>
        + Pack<[(u32, <I as Pack<[u32; 32]>>::T); 2]>,
    I: SIMDincrementCounters + Pack<[u32; 32]>,
    <I as SIMDincrementCounters>::WordShape:
        Pack<<Self as Pack<[[u32; 32]; 2]>>::T> + Pack<[u32; 32]>,
{
    fn safe_increment_counters_batch(
        example: &[(<I as BitPack<bool>>::T, <Self as BitPack<bool>>::T)],
        acc: &mut <Self as Pack<[(u32, <I as Pack<u32>>::T); 2]>>::T,
    );
    fn allocate_acc() -> Box<(
        <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
        <Self as Pack<u32>>::T,
        u32,
    )>;
    fn cache_local_count_matrix_batch(
        examples: &[(<I as BitPack<bool>>::T, <Self as BitPack<bool>>::T)],
        acc: &mut (
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        ),
    );
    fn merge(
        a: &mut (
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        ),
        b: &(
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        ),
    );
    fn word_transpose_weights_matrix(
        acc: &(
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        ),
    ) -> <Self as Pack<[(u32, <I as Pack<u32>>::T); 2]>>::T;
}

impl<I, T> CacheLocalMatrixBatch<I> for T
where
    T: IndexGet<u32>
        + BitPack<bool>
        + Pack<u8>
        + Map<u32, u32>
        + Pack<u32>
        + Map<u8, u32>
        + Pack<[(u32, <I as Pack<[u32; 32]>>::T); 2]>
        + Map<[SIMDword32; 2], [[u32; 32]; 2]>
        + Map<[[u32; 32]; 2], [[u32; 32]; 2]>
        + BitMap<bool, [(u32, <I as Pack<u32>>::T); 2]>
        + IndexGet<[[u32; 32]; 2]>
        + IndexMap<
            [(
                u32,
                <<I as SIMDincrementCounters>::WordShape as Pack<[u32; 32]>>::T,
            ); 2],
            (),
        > + BitMap<bool, [SIMDword32; 2]>
        + IncrementCounters<u8>,
    I: SIMDincrementCounters + Pack<[u32; 32]> + IncrementCounters<u32>,
    <T as Pack<u8>>::T: LongDefault,
    <T as Pack<[SIMDword32; 2]>>::T: LongDefault,
    Box<(
        <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
        <Self as Pack<u32>>::T,
        u32,
    )>: LongDefault,
    <I as SIMDincrementCounters>::WordShape: IndexGet<b32>
        + Pack<b32, T = <I as BitPack<bool>>::T>
        + Pack<[u32; 32], T = <I as Pack<u32>>::T>
        + IndexMap<[u32; 32], ()>
        + IndexGet<<Self as Pack<[[u32; 32]; 2]>>::T>
        + Map<<Self as Pack<[[u32; 32]; 2]>>::T, <Self as Pack<[[u32; 32]; 2]>>::T>,
{
    fn safe_increment_counters_batch(
        examples: &[(<I as BitPack<bool>>::T, <Self as BitPack<bool>>::T)],
        acc: &mut <Self as Pack<[(u32, <I as Pack<u32>>::T); 2]>>::T,
    ) {
        examples.iter().for_each(|(input, target)| {
            <Self as BitMap<bool, [(u32, <I as Pack<u32>>::T); 2]>>::map_mut(
                target,
                acc,
                |sign, counters| {
                    <I as IncrementCounters<u32>>::counted_increment_in_place(
                        input,
                        &mut counters[sign as usize],
                    );
                },
            );
        });
    }
    fn allocate_acc() -> Box<(
        <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
        <Self as Pack<u32>>::T,
        u32,
    )> {
        Box::<(
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        )>::long_default()
    }
    fn word_transpose_weights_matrix(
        acc: &(
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        ),
    ) -> <Self as Pack<[(u32, <I as Pack<u32>>::T); 2]>>::T {
        <Self as IndexMap<
            [(
                u32,
                <<I as SIMDincrementCounters>::WordShape as Pack<[u32; 32]>>::T,
            ); 2],
            (),
        >>::index_map((), |o| {
            let count = <Self as IndexGet<u32>>::index_get(&acc.1, o);
            [
                (
                    acc.2 - *count,
                    <<I as SIMDincrementCounters>::WordShape as IndexMap<[u32; 32], ()>>::index_map(
                        (),
                        |iw| {
                            let slice = <<I as SIMDincrementCounters>::WordShape as IndexGet<
                                <Self as Pack<[[u32; 32]; 2]>>::T,
                            >>::index_get(&acc.0, iw);
                            <Self as IndexGet<[[u32; 32]; 2]>>::index_get(&slice, o)[0]
                        },
                    ),
                ),
                (
                    *count,
                    <<I as SIMDincrementCounters>::WordShape as IndexMap<[u32; 32], ()>>::index_map(
                        (),
                        |iw| {
                            let slice = <<I as SIMDincrementCounters>::WordShape as IndexGet<
                                <Self as Pack<[[u32; 32]; 2]>>::T,
                            >>::index_get(&acc.0, iw);
                            <Self as IndexGet<[[u32; 32]; 2]>>::index_get(&slice, o)[1]
                        },
                    ),
                ),
            ]
        })
    }

    fn cache_local_count_matrix_batch(
        examples: &[(<I as BitPack<bool>>::T, <Self as BitPack<bool>>::T)],
        acc: &mut (
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        ),
    ) {
        assert!(examples.len() < 256);
        //dbg!(std::mem::size_of::<<O as Pack<[(u64, <I as Pack<T>>::T); 2]>>::T>());
        //let expanded_input = I::expand_bits(&examples[0].0);

        <I as SIMDincrementCounters>::WordShape::indices().for_each(|iw| {
            let mut simd_counters_row = <Self as Pack<[SIMDword32; 2]>>::T::long_default();
            examples.iter().for_each(|(input, target)| {
                let input_word: &b32 =
                    <<I as SIMDincrementCounters>::WordShape as IndexGet<b32>>::index_get(
                        input, iw,
                    );
                let simd_input_word = <[(); 32] as SIMDincrementCounters>::expand_bits(input_word);
                <Self as BitMap<bool, [SIMDword32; 2]>>::map_mut(
                    target,
                    &mut simd_counters_row,
                    |sign, counters| {
                        <[(); 32] as SIMDincrementCounters>::simd_increment_in_place(
                            &simd_input_word,
                            &mut counters[sign as usize],
                        );
                    },
                )
            });

            let acc_row = <<I as SIMDincrementCounters>::WordShape as IndexGet<
                <Self as Pack<[[u32; 32]; 2]>>::T,
            >>::index_get_mut(&mut acc.0, iw);
            <Self as Map<[SIMDword32; 2], [[u32; 32]; 2]>>::map_mut(
                &simd_counters_row,
                acc_row,
                |simd_word, u32_word| {
                    <[(); 32] as SIMDincrementCounters>::add_to_u32s(
                        &simd_word[0],
                        &mut u32_word[0],
                    );
                    <[(); 32] as SIMDincrementCounters>::add_to_u32s(
                        &simd_word[1],
                        &mut u32_word[1],
                    );
                },
            );
        });

        let target_counters: <Self as Pack<u8>>::T = examples.iter().fold(
            <Self as Pack<u8>>::T::long_default(),
            |mut acc, (_, target)| {
                <Self as IncrementCounters<u8>>::increment_in_place(target, &mut acc);
                acc
            },
        );
        <Self as Map<u8, u32>>::map_mut(&target_counters, &mut acc.1, |&u8_count, u32_count| {
            *u32_count += u8_count as u32;
        });
        acc.2 += examples.len() as u32;
    }
    fn merge(
        a: &mut (
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        ),
        b: &(
            <<I as SIMDincrementCounters>::WordShape as Pack<<Self as Pack<[[u32; 32]; 2]>>::T>>::T,
            <Self as Pack<u32>>::T,
            u32,
        ),
    ) {
        a.2 += b.2;
        <Self as Map<u32, u32>>::map_mut(&b.1, &mut a.1, |b, a| {
            *a += b;
        });
        <<I as SIMDincrementCounters>::WordShape as Map<
            <Self as Pack<[[u32; 32]; 2]>>::T,
            <Self as Pack<[[u32; 32]; 2]>>::T,
        >>::map_mut(&b.0, &mut a.0, |b, a| {
            <Self as Map<[[u32; 32]; 2], [[u32; 32]; 2]>>::map_mut(b, a, |b, a| {
                <[[(); 32]; 2] as Map<u32, u32>>::map_mut(b, a, |b, a| {
                    *a += b;
                });
            });
        });
    }
}

#[cfg(test)]
mod test {
    use crate::bits::{
        b128, b16, b32, b8, t128, t16, t32, t8, BitMap, BitPack, BitScaler, IncrementCounters,
        SIMDincrementCounters, SIMDword32, WeightArray,
    };
    use crate::matrix::CacheLocalMatrixBatch;
    use crate::shape::{LongDefault, Pack};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;

    type InputShape = [[[(); 32]; 8]; 2];
    type TargetShape = [[(); 32]; 4];

    #[test]
    fn cache_local_count() {
        let mut rng = Hc128Rng::seed_from_u64(0);

        let acc = Box::<(
            <<InputShape as SIMDincrementCounters>::WordShape as Pack<
                <TargetShape as Pack<[[u32; 32]; 2]>>::T,
            >>::T,
            <TargetShape as Pack<u32>>::T,
            u32,
        )>::long_default();

        let examples: Vec<_> = (0..1000)
            .map(|_| {
                let input: <InputShape as BitPack<bool>>::T = rng.gen();
                let target: <TargetShape as BitPack<bool>>::T = rng.gen();
                (input, target)
            })
            .collect();

        let acc = examples.chunks(255).fold(acc, |mut acc, chunk| {
            <TargetShape as CacheLocalMatrixBatch<InputShape>>::cache_local_count_matrix_batch(
                chunk, &mut acc,
            );
            acc
        });
        let transposed_acc =
            <TargetShape as CacheLocalMatrixBatch<InputShape>>::word_transpose_weights_matrix(&acc);

        let test_acc = examples.iter().fold(
            Box::<<TargetShape as Pack<[(u32, <InputShape as Pack<u32>>::T); 2]>>::T>::long_default(
            ),
            |mut acc, example| {
                <TargetShape as CacheLocalMatrixBatch<InputShape>>::safe_increment_counters(
                    example, &mut acc,
                );
                acc
            },
        );
        assert_eq!(*test_acc, transposed_acc);
    }
}
