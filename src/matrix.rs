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

fn word_transpose_weights_matrix<IWS, OS>(
    input: &<IWS as Pack<<OS as Pack<[[u32; 32]; 2]>>::T>>::T,
    counts: &<OS as Pack<u32>>::T,
    &n: &u32,
) -> <OS as Pack<[(u32, <IWS as Pack<[u32; 32]>>::T); 2]>>::T
where
    OS: IndexGet<u32>
        + Pack<[u32; 2]>
        + Pack<[u32; 2]>
        + IndexGet<[u32; 2]>
        + IndexGet<[[u32; 32]; 2]>
        + Pack<[[u32; 32]; 2]>
        + Pack<[(u32, <IWS as Pack<[u32; 32]>>::T); 2]>
        + IndexMap<[(u32, <IWS as Pack<[u32; 32]>>::T); 2], ()>,
    IWS: Pack<<OS as Pack<[[u32; 32]; 2]>>::T>
        + Pack<[u32; 32]>
        + IndexMap<[u32; 32], ()>
        + IndexGet<<OS as Pack<[[u32; 32]; 2]>>::T>,
{
    <OS as IndexMap<[(u32, <IWS as Pack<[u32; 32]>>::T); 2], ()>>::index_map((), |o| {
        let count = <OS as IndexGet<u32>>::index_get(counts, o);
        [
            (
                n - *count,
                <IWS as IndexMap<[u32; 32], ()>>::index_map((), |iw| {
                    let slice =
                        <IWS as IndexGet<<OS as Pack<[[u32; 32]; 2]>>::T>>::index_get(input, iw);
                    <OS as IndexGet<[[u32; 32]; 2]>>::index_get(&slice, o)[0]
                }),
            ),
            (
                *count,
                <IWS as IndexMap<[u32; 32], ()>>::index_map((), |iw| {
                    let slice =
                        <IWS as IndexGet<<OS as Pack<[[u32; 32]; 2]>>::T>>::index_get(input, iw);
                    <OS as IndexGet<[[u32; 32]; 2]>>::index_get(&slice, o)[1]
                }),
            ),
        ]
    })
}

pub fn cache_local_count_matrix_batch<I, O>(
    examples: &[(<I as BitPack<bool>>::T, <O as BitPack<bool>>::T)],
    u32_acc: &mut <<I as SIMDincrementCounters>::WordShape as Pack<
        <O as Pack<[[u32; 32]; 2]>>::T,
    >>::T,
    counts: &mut <O as Pack<u32>>::T,
    n: &mut u32,
) where
    I: BitPack<bool>
        + Pack<u32>
        + IncrementCounters<u32>
        + ZipMap<u32, u32, u32>
        + SIMDincrementCounters,
    O: BitPack<bool>
        + Pack<[[u32; 32]; 2]>
        + Pack<[u32; 2]>
        + Pack<u32>
        + Pack<[(u32, <I as Pack<u32>>::T); 2]>
        + Pack<[SIMDword32; 2]>
        + Map<[SIMDword32; 2], [[u32; 32]; 2]>
        + Map<(), [(u32, I::SIMDbyts); 2]>
        + Map<u8, u32>
        + IncrementCounters<u8>
        + Pack<u8>
        + Map<[(u32, I::SIMDbyts); 2], [(u32, <I as Pack<u32>>::T); 2]>
        + BitMap<bool, [SIMDword32; 2]>
        + ZipMap<
            [(u32, <I as Pack<u32>>::T); 2],
            [(u32, <I as Pack<u32>>::T); 2],
            [(u32, <I as Pack<u32>>::T); 2],
        > + BitMap<bool, [(u64, I::SIMDbyts); 2]>,
    <I as BitPack<bool>>::T: Sync,
    <O as Pack<[SIMDword32; 2]>>::T: LongDefault,
    <O as Pack<()>>::T: Default,
    <O as Pack<u8>>::T: LongDefault + std::fmt::Debug,
    <O as BitPack<bool>>::T: Sync,
    <O as Pack<[(u32, I::SIMDbyts); 2]>>::T: LongDefault,
    [(); 32]: SIMDincrementCounters,
    <I as SIMDincrementCounters>::WordShape: Pack<<O as Pack<[[u32; 32]; 2]>>::T>
        + IndexGet<<O as Pack<[[u32; 32]; 2]>>::T>
        + IndexGet<b32>
        + Pack<b32, T = <I as BitPack<bool>>::T>
        + Map<[SIMDword32; 2], [[u32; 32]; 2]>,
    [(); 2]: ZipMap<(u32, <I as Pack<u32>>::T), (u32, <I as Pack<u32>>::T), (u32, <I as Pack<u32>>::T)>
        + Pack<(u32, <I as Pack<u32>>::T), T = [(u32, <I as Pack<u32>>::T); 2]>,
{
    assert!(examples.len() < 256);
    //dbg!(std::mem::size_of::<<O as Pack<[(u64, <I as Pack<T>>::T); 2]>>::T>());
    //let expanded_input = I::expand_bits(&examples[0].0);

    <I as SIMDincrementCounters>::WordShape::indices().for_each(|iw| {
        let mut simd_counters_row = <O as Pack<[SIMDword32; 2]>>::T::long_default();
        examples.iter().for_each(|(input, target)| {
            let input_word: &b32 =
                <<I as SIMDincrementCounters>::WordShape as IndexGet<b32>>::index_get(input, iw);
            let simd_input_word = <[(); 32] as SIMDincrementCounters>::expand_bits(input_word);
            <O as BitMap<bool, [SIMDword32; 2]>>::map_mut(
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
            <O as Pack<[[u32; 32]; 2]>>::T,
        >>::index_get_mut(u32_acc, iw);
        <O as Map<[SIMDword32; 2], [[u32; 32]; 2]>>::map_mut(
            &simd_counters_row,
            acc_row,
            |simd_word, u32_word| {
                <[(); 32] as SIMDincrementCounters>::add_to_u32s(&simd_word[0], &mut u32_word[0]);
                <[(); 32] as SIMDincrementCounters>::add_to_u32s(&simd_word[1], &mut u32_word[1]);
            },
        );
    });

    let target_counters: <O as Pack<u8>>::T = examples.iter().fold(
        <O as Pack<u8>>::T::long_default(),
        |mut acc, (_, target)| {
            <O as IncrementCounters<u8>>::increment_in_place(target, &mut acc);
            acc
        },
    );
    <O as Map<u8, u32>>::map_mut(&target_counters, counts, |&u8_count, u32_count| {
        *u32_count += u8_count as u32;
    });
    *n += examples.len() as u32;
}

#[cfg(test)]
mod test {
    use crate::bits::{
        b128, b16, b32, b8, t128, t16, t32, t8, BitMap, BitPack, BitScaler, IncrementCounters,
        SIMDincrementCounters, SIMDword32, WeightArray,
    };
    use crate::matrix::{cache_local_count_matrix_batch, word_transpose_weights_matrix};
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
            cache_local_count_matrix_batch::<InputShape, TargetShape>(
                chunk, &mut acc.0, &mut acc.1, &mut acc.2,
            );
            acc
        });
        let transposed_acc = word_transpose_weights_matrix::<
            <InputShape as SIMDincrementCounters>::WordShape,
            TargetShape,
        >(&acc.0, &acc.1, &acc.2);

        let test_acc = examples.iter().fold(
            Box::<<TargetShape as Pack<[(u32, <InputShape as Pack<u32>>::T); 2]>>::T>::long_default(
            ),
            |mut acc, (input, target)| {
                <TargetShape as BitMap<bool, [(u32, <InputShape as Pack<u32>>::T); 2]>>::map_mut(
                    target,
                    &mut acc,
                    |sign, counters| {
                        <InputShape as IncrementCounters<u32>>::counted_increment_in_place(
                            input,
                            &mut counters[sign as usize],
                        );
                    },
                );
                acc
            },
        );
        assert_eq!(*test_acc, transposed_acc);
    }
}
