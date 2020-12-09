#[macro_use]
extern crate criterion;
use bnn::bits::{
    b128, b16, b32, b64, b8, t128, t16, t32, t64, t8, IncrementCounters, MaskedDistance,
    PackedElement, Weight, WeightArray, BMA,
};
use bnn::shape::{Element, LongDefault, Shape};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::time::Duration;

macro_rules! bench_bma {
    ($group:expr, $name:expr, $i:expr, $s:ty, $w:ty) => {
        let mut rng = Hc128Rng::seed_from_u64(0);
        $group.bench_with_input(BenchmarkId::new($name, $i), &$i, |b, _| {
            b.iter_batched(
                || {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights = <() as WeightArray<$s, $w>>::rand(&mut rng);
                    (inputs, weights)
                },
                |(inputs, weights)| <() as WeightArray<$s, $w>>::bma(&weights, &inputs),
                BatchSize::SmallInput,
            )
        });
    };
}

macro_rules! bench_acts {
    ($group:expr, $name:expr, $i:expr, $s:ty, $w:ty) => {
        let mut rng = Hc128Rng::seed_from_u64(0);
        $group.bench_with_input(BenchmarkId::new($name, $i), &$i, |b, _| {
            b.iter_batched(
                || {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights = <() as WeightArray<$s, $w>>::rand(&mut rng);
                    let sign: $w = rng.gen();
                    (inputs, weights, sign)
                },
                |(inputs, weights, sign)| {
                    <() as WeightArray<$s, $w>>::acts(&weights, &inputs, sign)
                },
                BatchSize::SmallInput,
            )
        });
    };
}

macro_rules! bench_input_acts {
    ($group:expr, $name:expr, $i:expr, $s:ty, $w:ty) => {
        let mut rng = Hc128Rng::seed_from_u64(0);
        $group.bench_with_input(BenchmarkId::new($name, $i), &$i, |b, _| {
            b.iter_batched(
                || {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights = <() as WeightArray<$s, $w>>::rand(&mut rng);
                    (inputs, weights)
                },
                |(inputs, weights)| <() as WeightArray<$s, $w>>::input_acts(&weights, &inputs),
                BatchSize::SmallInput,
            )
        });
    };
}

fn bit_acts(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_acts");

    bench_acts!(group, "u8", 1, [[(); 8]; 1], bool);
    bench_acts!(group, "u8", 2, [[(); 8]; 2], bool);
    bench_acts!(group, "u8", 3, [[(); 8]; 3], bool);
    bench_acts!(group, "u8", 4, [[(); 8]; 4], bool);
    bench_acts!(group, "u8", 32, [[(); 8]; 32], bool);

    bench_acts!(group, "u16", 32, [[(); 16]; 32], bool);
    bench_acts!(group, "u32", 32, [[(); 32]; 32], bool);
    bench_acts!(group, "u64", 32, [[(); 64]; 32], bool);
    bench_acts!(group, "u128", 32, [[(); 128]; 32], bool);

    bench_acts!(group, "3x3_u8", 9, [[[(); 8]; 3]; 3], bool);

    group.finish()
}

fn trit_acts(c: &mut Criterion) {
    let mut group = c.benchmark_group("trit_acts");

    bench_acts!(group, "u8", 1, [[(); 8]; 1], Option<bool>);
    bench_acts!(group, "u8", 2, [[(); 8]; 2], Option<bool>);
    bench_acts!(group, "u8", 3, [[(); 8]; 3], Option<bool>);
    bench_acts!(group, "u8", 4, [[(); 8]; 4], Option<bool>);
    bench_acts!(group, "u8", 32, [[(); 8]; 32], Option<bool>);

    bench_acts!(group, "u16", 32, [[(); 16]; 32], Option<bool>);
    bench_acts!(group, "u32", 32, [[(); 32]; 32], Option<bool>);
    bench_acts!(group, "u64", 32, [[(); 64]; 32], Option<bool>);
    bench_acts!(group, "u128", 32, [[(); 128]; 32], Option<bool>);

    bench_acts!(group, "3x3_u8", 9, [[[(); 8]; 3]; 3], Option<bool>);

    group.finish()
}

fn bit_input_acts(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_acts");

    bench_input_acts!(group, "u8", 1, [[(); 8]; 1], bool);
    bench_input_acts!(group, "u8", 2, [[(); 8]; 2], bool);
    bench_input_acts!(group, "u8", 3, [[(); 8]; 3], bool);
    bench_input_acts!(group, "u8", 4, [[(); 8]; 4], bool);
    bench_input_acts!(group, "3x3_u8", 9, [[[(); 8]; 3]; 3], bool);

    group.finish()
}

fn trit_input_acts(c: &mut Criterion) {
    let mut group = c.benchmark_group("trit_acts");

    bench_input_acts!(group, "u8", 1, [[(); 8]; 1], Option<bool>);
    bench_input_acts!(group, "u8", 2, [[(); 8]; 2], Option<bool>);
    bench_input_acts!(group, "u8", 3, [[(); 8]; 3], Option<bool>);
    bench_input_acts!(group, "u8", 4, [[(); 8]; 4], Option<bool>);
    bench_input_acts!(group, "3x3_u8", 9, [[[(); 8]; 3]; 3], Option<bool>);

    group.finish()
}

fn bit_bma(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_bma");

    bench_bma!(group, "u8", 1, [[(); 8]; 1], bool);
    bench_bma!(group, "u8", 2, [[(); 8]; 2], bool);
    bench_bma!(group, "u8", 3, [[(); 8]; 3], bool);
    bench_bma!(group, "u8", 4, [[(); 8]; 4], bool);
    bench_bma!(group, "u8", 8, [[(); 8]; 8], bool);
    bench_bma!(group, "u8", 16, [[(); 8]; 16], bool);
    bench_bma!(group, "u8", 32, [[(); 8]; 32], bool);

    bench_bma!(group, "3x3_u8", 9, [[[(); 8]; 3]; 3], bool);

    bench_bma!(group, "u16", 1, [[(); 16]; 1], bool);
    bench_bma!(group, "u16", 2, [[(); 16]; 2], bool);
    bench_bma!(group, "u16", 3, [[(); 16]; 3], bool);
    bench_bma!(group, "u16", 4, [[(); 16]; 4], bool);
    bench_bma!(group, "u16", 8, [[(); 16]; 8], bool);
    bench_bma!(group, "u16", 16, [[(); 16]; 16], bool);
    bench_bma!(group, "u16", 32, [[(); 16]; 32], bool);

    bench_bma!(group, "3x3_u16", 9, [[[(); 16]; 3]; 3], bool);

    bench_bma!(group, "u32", 1, [[(); 32]; 1], bool);
    bench_bma!(group, "u32", 2, [[(); 32]; 2], bool);
    bench_bma!(group, "u32", 3, [[(); 32]; 3], bool);
    bench_bma!(group, "u32", 4, [[(); 32]; 4], bool);
    bench_bma!(group, "u32", 8, [[(); 32]; 8], bool);
    bench_bma!(group, "u32", 16, [[(); 32]; 16], bool);
    bench_bma!(group, "u32", 32, [[(); 32]; 32], bool);

    bench_bma!(group, "3x3_u32", 9, [[[(); 32]; 3]; 3], bool);

    bench_bma!(group, "u64", 1, [[(); 64]; 1], bool);
    bench_bma!(group, "u64", 2, [[(); 64]; 2], bool);
    bench_bma!(group, "u64", 3, [[(); 64]; 3], bool);
    bench_bma!(group, "u64", 4, [[(); 64]; 4], bool);
    bench_bma!(group, "u64", 8, [[(); 64]; 8], bool);
    bench_bma!(group, "u64", 16, [[(); 64]; 16], bool);
    bench_bma!(group, "u64", 32, [[(); 64]; 32], bool);

    bench_bma!(group, "3x3_u64", 9, [[[(); 64]; 3]; 3], bool);
    bench_bma!(group, "32x32_u64", 1024, [[[(); 64]; 32]; 32], bool);

    bench_bma!(group, "u128", 1, [[(); 128]; 1], bool);
    bench_bma!(group, "u128", 2, [[(); 128]; 2], bool);
    bench_bma!(group, "u128", 3, [[(); 128]; 3], bool);
    bench_bma!(group, "u128", 4, [[(); 128]; 4], bool);
    bench_bma!(group, "u128", 8, [[(); 128]; 8], bool);
    bench_bma!(group, "u128", 16, [[(); 128]; 16], bool);
    bench_bma!(group, "u128", 32, [[(); 128]; 32], bool);

    bench_bma!(group, "3x3_u128", 9, [[[(); 128]; 3]; 3], bool);

    group.finish()
}

fn trit_bma(c: &mut Criterion) {
    let mut group = c.benchmark_group("trit_bma");

    bench_bma!(group, "u8", 1, [[(); 8]; 1], Option<bool>);
    bench_bma!(group, "u8", 2, [[(); 8]; 2], Option<bool>);
    bench_bma!(group, "u8", 3, [[(); 8]; 3], Option<bool>);
    bench_bma!(group, "u8", 4, [[(); 8]; 4], Option<bool>);
    bench_bma!(group, "u8", 8, [[(); 8]; 8], Option<bool>);
    bench_bma!(group, "u8", 16, [[(); 8]; 16], Option<bool>);
    bench_bma!(group, "u8", 32, [[(); 8]; 32], Option<bool>);

    bench_bma!(group, "3x3_u8", 9, [[[(); 8]; 3]; 3], Option<bool>);

    bench_bma!(group, "u16", 1, [[(); 16]; 1], Option<bool>);
    bench_bma!(group, "u16", 2, [[(); 16]; 2], Option<bool>);
    bench_bma!(group, "u16", 3, [[(); 16]; 3], Option<bool>);
    bench_bma!(group, "u16", 4, [[(); 16]; 4], Option<bool>);
    bench_bma!(group, "u16", 8, [[(); 16]; 8], Option<bool>);
    bench_bma!(group, "u16", 16, [[(); 16]; 16], Option<bool>);
    bench_bma!(group, "u16", 32, [[(); 16]; 32], Option<bool>);

    bench_bma!(group, "3x3_u16", 9, [[[(); 16]; 3]; 3], Option<bool>);

    bench_bma!(group, "u32", 1, [[(); 32]; 1], Option<bool>);
    bench_bma!(group, "u32", 2, [[(); 32]; 2], Option<bool>);
    bench_bma!(group, "u32", 3, [[(); 32]; 3], Option<bool>);
    bench_bma!(group, "u32", 4, [[(); 32]; 4], Option<bool>);
    bench_bma!(group, "u32", 8, [[(); 32]; 8], Option<bool>);
    bench_bma!(group, "u32", 16, [[(); 32]; 16], Option<bool>);
    bench_bma!(group, "u32", 32, [[(); 32]; 32], Option<bool>);

    bench_bma!(group, "3x3_u32", 9, [[[(); 32]; 3]; 3], Option<bool>);

    bench_bma!(group, "u64", 1, [[(); 64]; 1], Option<bool>);
    bench_bma!(group, "u64", 2, [[(); 64]; 2], Option<bool>);
    bench_bma!(group, "u64", 3, [[(); 64]; 3], Option<bool>);
    bench_bma!(group, "u64", 4, [[(); 64]; 4], Option<bool>);
    bench_bma!(group, "u64", 8, [[(); 64]; 8], Option<bool>);
    bench_bma!(group, "u64", 16, [[(); 64]; 16], Option<bool>);
    bench_bma!(group, "u64", 32, [[(); 64]; 32], Option<bool>);

    bench_bma!(group, "32x32_u64", 1024, [[[(); 64]; 32]; 32], Option<bool>);

    bench_bma!(group, "u128", 1, [[(); 128]; 1], Option<bool>);
    bench_bma!(group, "u128", 2, [[(); 128]; 2], Option<bool>);
    bench_bma!(group, "u128", 3, [[(); 128]; 3], Option<bool>);
    bench_bma!(group, "u128", 4, [[(); 128]; 4], Option<bool>);
    bench_bma!(group, "u128", 8, [[(); 128]; 8], Option<bool>);
    bench_bma!(group, "u128", 16, [[(); 128]; 16], Option<bool>);
    bench_bma!(group, "u128", 32, [[(); 128]; 32], Option<bool>);

    bench_bma!(group, "3x3_u128", 9, [[[(); 128]; 3]; 3], Option<bool>);

    group.finish()
}

macro_rules! bench_increment_counters {
    ($group:expr, $name:expr, $i:expr, $s:ty) => {
        let mut rng = Hc128Rng::seed_from_u64(0);
        $group.bench_with_input(BenchmarkId::new($name, $i), &$i, |b, _| {
            b.iter_batched(
                || {
                    let inputs: Vec<<bool as PackedElement<$s>>::Array> =
                        (0..1000).map(|_| rng.gen()).collect();
                    let counters = <u32 as Element<$s>>::Array::long_default();
                    (inputs, counters)
                },
                |(inputs, counters)| {
                    inputs.iter().fold(counters, |counters, bits| {
                        <$s as IncrementCounters>::increment(bits, counters)
                    })
                },
                BatchSize::SmallInput,
            )
        });
    };
}

fn increment_counters_word(c: &mut Criterion) {
    let mut group = c.benchmark_group("increment_counters_word");

    bench_increment_counters!(group, "w", 8, [(); 8]);
    bench_increment_counters!(group, "w", 16, [(); 16]);
    bench_increment_counters!(group, "w", 32, [(); 32]);
    bench_increment_counters!(group, "w", 64, [(); 64]);
    bench_increment_counters!(group, "w", 128, [(); 128]);

    bench_increment_counters!(group, "w_3x3", 8, [[[(); 8]; 3]; 3]);
    bench_increment_counters!(group, "w_3x3", 16, [[[(); 16]; 3]; 3]);
    bench_increment_counters!(group, "w_3x3", 32, [[[(); 32]; 3]; 3]);
    bench_increment_counters!(group, "w_3x3", 64, [[[(); 64]; 3]; 3]);
    bench_increment_counters!(group, "w_3x3", 128, [[[(); 128]; 3]; 3]);

    group.finish()
}

fn increment_counters_len(c: &mut Criterion) {
    let mut group = c.benchmark_group("increment_counters_len");

    bench_increment_counters!(group, "u8", 1, [[(); 8]; 1]);
    bench_increment_counters!(group, "u8", 2, [[(); 8]; 2]);
    bench_increment_counters!(group, "u8", 3, [[(); 8]; 3]);
    bench_increment_counters!(group, "u8", 4, [[(); 8]; 4]);
    bench_increment_counters!(group, "u8", 8, [[(); 8]; 8]);
    bench_increment_counters!(group, "u8", 16, [[(); 8]; 16]);
    bench_increment_counters!(group, "u8", 32, [[(); 8]; 32]);

    group.finish()
}

macro_rules! bench_trit_bma_1 {
    ($group:expr, $name:expr, $i:expr, $s:ty) => {
        let mut rng = Hc128Rng::seed_from_u64(0);
        $group.bench_with_input(BenchmarkId::new($name, $i), &$i, |b, _| {
            b.iter_batched(
                || {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights = <() as WeightArray<$s, Option<bool>>>::rand(&mut rng);
                    (inputs, weights)
                },
                |(inputs, weights)| <() as WeightArray<$s, Option<bool>>>::bma(&weights, &inputs),
                BatchSize::SmallInput,
            )
        });
    };
}

macro_rules! bench_trit_bma_2 {
    ($group:expr, $name:expr, $i:expr, $s:ty) => {
        let mut rng = Hc128Rng::seed_from_u64(0);
        $group.bench_with_input(BenchmarkId::new($name, $i), &$i, |b, _| {
            b.iter_batched(
                || {
                    let inputs: <bool as PackedElement<$s>>::Array = rng.gen();
                    let weights: <Option<bool> as PackedElement<$s>>::Array = rng.gen();
                    (inputs, weights)
                },
                |(inputs, weights)| weights.bma(&inputs),
                BatchSize::SmallInput,
            )
        });
    };
}

fn trit_bma_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("bit_bma");

    bench_trit_bma_1!(group, "u8", 1, [[(); 8]; 1]);
    bench_trit_bma_2!(group, "u8", 2, [[(); 8]; 1]);

    bench_trit_bma_1!(group, "u32", 1, [[(); 32]; 1]);
    bench_trit_bma_2!(group, "u32", 2, [[(); 32]; 1]);

    bench_trit_bma_1!(group, "u32_3x3x4", 1, [[[[(); 32]; 3]; 3]; 4]);
    bench_trit_bma_2!(group, "u32_3x3x4", 2, [[[[(); 32]; 3]; 3]; 4]);

    bench_trit_bma_1!(group, "u32_8", 1, [[(); 32]; 8]);
    bench_trit_bma_2!(group, "u32_8", 2, [[(); 32]; 8]);

    bench_trit_bma_1!(group, "u32_32", 1, [[(); 32]; 32]);
    bench_trit_bma_2!(group, "u32_32", 2, [[(); 32]; 32]);

    bench_trit_bma_1!(group, "u64_3x3", 1, [[[(); 64]; 3]; 3]);
    bench_trit_bma_2!(group, "u64_3x3", 2, [[[(); 64]; 3]; 3]);

    bench_trit_bma_1!(group, "u64_8", 1, [[(); 64]; 8]);
    bench_trit_bma_2!(group, "u64_8", 2, [[(); 64]; 8]);

    bench_trit_bma_1!(group, "u64_32", 1, [[(); 64]; 32]);
    bench_trit_bma_2!(group, "u64_32", 2, [[(); 64]; 32]);

    bench_trit_bma_1!(group, "u64_16", 1, [[(); 64]; 16]);
    bench_trit_bma_2!(group, "u64_16", 2, [[(); 64]; 16]);

    group.finish()
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    //config = Criterion::default().measurement_time(Duration::from_secs(5));
    //targets = bit_bma, trit_bma, bit_acts, trit_acts, bit_input_acts, trit_input_acts
    //targets = increment_counters_len, increment_counters_word
    targets = trit_bma_compare
}

criterion_main!(benches);
