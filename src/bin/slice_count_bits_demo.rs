#![feature(int_log)]
#![feature(generic_const_exprs)]
#![feature(avx512_target_feature)]
#![feature(stdsimd)]

use bnn::bench::{PerfResults, PerfTest};
use bnn::bits::{b128, b64};
use bnn::bitslice::{BitArray64, BitSlice};
use bnn::count_bits::{BitSliceBitCounter, ExpCountBits, PopCountBitCounter, UnitCountBits};
use bnn::ecc::{decode_byte, encode_byte};
use bnn::layer;
use bnn::search::{compute_exp_candidates, update_weights, weights_to_dense, weights_to_sparse};
use bnn::shape::flatten_2d;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::uint8x16_t;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(target_feature = "avx2")]
use std::arch::x86_64::{__m256i, __m512i, _mm256_setzero_si256};
use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::marker::PhantomData;
use std::mem;
use std::path::Path;
use std::time::Instant;

fn run_test(
    alg: &Box<dyn CountBits>,
    input: &[[b64; 8]],
    target: &[[b64; 4]],
    n_threads: usize,
    chunk_size: usize,
) -> PerfTest {
    let total_start = Instant::now();

    assert_eq!(input.len(), target.len());
    let weights: [([Option<bool>; 512], u32); 256] = (0..256)
        .map(|i| {
            let mut w = ([None; 512], 1u32);
            w.0[i] = Some(i % 2 == 0);
            w.0[(i * 2) % 512] = Some(i % 2 == 0);
            w.0[(i * 3) % 512] = Some(i % 2 == 1);
            w.0[(i * 5) % 512] = Some(i % 2 == 0);
            w.0[(i * 7) % 512] = Some(i % 2 == 1);
            w.0[(i * 11) % 512] = Some(i % 2 == 0);
            w
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let sparse: [(Vec<(usize, bool)>, [u32; UNIT_THRESHOLDS]); 256] = weights
        .iter()
        .map(|w| weights_to_sparse(w))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let unit_start = Instant::now();
    let unit_counts = alg.unit_count_bits(&input, &target, &sparse, n_threads, chunk_size);
    let unit_duration = unit_start.elapsed();

    let mut hasher = DefaultHasher::new();
    unit_counts.hash(&mut hasher);
    let unit_hash = hasher.finish();

    let exp_candidates: [(Vec<(usize, bool)>, [(usize, bool); 8 as usize], [u32; 3]); 256] =
        weights
            .iter()
            .zip(unit_counts.iter())
            .map(|(w, counts)| compute_exp_candidates::<512, 2, 3, 8>(w, counts))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

    let exp_start = Instant::now();
    let exp_counts = alg.exp_count_bits(&input, &target, &exp_candidates, n_threads, chunk_size);
    let exp_duration = exp_start.elapsed();

    let weights = weights
        .chunks(64)
        .map(|chunk| {
            chunk
                .iter()
                .map(|w| weights_to_dense(w))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let apply_start = Instant::now();
    let l: Vec<[b64; 4]> = target
        .par_windows(2)
        .map(|slice| {
            let flat = flatten_2d::<b64, 2, 4>(<&[[b64; 4]; 2]>::try_from(&slice[0..2]).unwrap());
            layer::apply(&weights, flat)
        })
        .collect();
    let apply_duration = apply_start.elapsed();

    let mut hasher = DefaultHasher::new();
    exp_counts.hash(&mut hasher);
    let exp_hash = hasher.finish();

    let total_duration = total_start.elapsed();
    PerfTest {
        algorithm: alg.name(),
        n_threads: n_threads,
        chunk_size: chunk_size,
        n_examples: input.len(),
        unit_nanos: unit_duration.as_nanos(),
        exp_nanos: exp_duration.as_nanos(),
        apply_nanos: apply_duration.as_nanos(),
        total_nanos: total_duration.as_nanos(),
        unit_hash: unit_hash,
        exp_hash: exp_hash,
    }
}

const N_THRESHOLDS: usize = 3;
const N_EXP: u32 = 8;

const UNIT_THRESHOLDS: usize = 2;

trait CountBits: ExpCountBits<8, 4, 3, 8> + UnitCountBits<8, 4, 2> {
    fn name(&self) -> String;
    fn width(&self) -> usize;
    fn cost(&self) -> f64;
}

impl CountBits for PopCountBitCounter {
    fn name(&self) -> String {
        format!("popcnt")
    }
    fn width(&self) -> usize {
        64 // not actualy, but is makes things faster
    }
    fn cost(&self) -> f64 {
        220f64
    }
}

impl<const B: usize, const L: usize> CountBits for BitSliceBitCounter<BitArray64<L>, B>
where
    [(); BitArray64::<L>::N]: ,
{
    fn name(&self) -> String {
        format!("bitslice-BitArray64x{}", L)
    }
    fn width(&self) -> usize {
        64 * L
    }
    fn cost(&self) -> f64 {
        8f64 / L as f64
    }
}

#[cfg(target_feature = "avx512f")]
impl<const B: usize> CountBits for BitSliceBitCounter<__m512i, B> {
    fn name(&self) -> String {
        format!("bitslice-avx512")
    }
    fn width(&self) -> usize {
        512
    }
    fn cost(&self) -> f64 {
        1f64
    }
}

#[cfg(target_feature = "avx2")]
impl<const B: usize> CountBits for BitSliceBitCounter<__m256i, B> {
    fn name(&self) -> String {
        format!("bitslice-avx2")
    }
    fn width(&self) -> usize {
        256
    }
    fn cost(&self) -> f64 {
        2f64
    }
}

#[cfg(target_feature = "neon")]
impl<const B: usize> CountBits for BitSliceBitCounter<uint8x16_t, B> {
    fn name(&self) -> String {
        format!("bitslice-neon")
    }
    fn width(&self) -> usize {
        128
    }
    fn cost(&self) -> f64 {
        4f64
    }
}

impl<const B: usize> CountBits for BitSliceBitCounter<b128, B> {
    fn name(&self) -> String {
        format!("bitslice-u128")
    }
    fn width(&self) -> usize {
        128
    }
    fn cost(&self) -> f64 {
        4f64
    }
}

impl<const B: usize> CountBits for BitSliceBitCounter<b64, B> {
    fn name(&self) -> String {
        format!("bitslice-u64")
    }
    fn width(&self) -> usize {
        64
    }
    fn cost(&self) -> f64 {
        8f64
    }
}

fn main() {
    let n_cores = num_cpus::get();
    rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .stack_size(2usize.pow(30))
        .build_global()
        .unwrap();

    let bytes = std::fs::read(
        "Delphi Complete Works of Charles Dickens (Illustrated) - Charles Dickens.txt",
    )
    .unwrap();

    let expanded: Vec<[b64; 4]> = bytes.par_iter().map(|&b| encode_byte(b)).collect();

    let (input, target): (Vec<[b64; 8]>, Vec<[b64; 4]>) = expanded
        .par_windows(3)
        .map(|slice| {
            let input = flatten_2d::<b64, 2, 4>(<&[[b64; 4]; 2]>::try_from(&slice[0..2]).unwrap());
            (input, slice[2])
        })
        .unzip();

    let mut counters: Vec<Box<dyn CountBits>> = vec![
        //Box::new(PopCountBitCounter {}),
        Box::new(BitSliceBitCounter::<b64, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<b128, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<1>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<2>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<4>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<8>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<16>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<32>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<64>, 4> {
            slice_type: PhantomData::default(),
        }),
    ];

    #[cfg(target_feature = "avx2")]
    {
        println!("avx2 is enabled",);
        counters.push(Box::new(BitSliceBitCounter::<__m256i, 4> {
            slice_type: PhantomData::default(),
        }));
    }
    #[cfg(target_feature = "avx512f")]
    {
        println!("avx512 is enabled",);
        counters.push(Box::new(BitSliceBitCounter::<__m512i, 4> {
            slice_type: PhantomData::default(),
        }));
    }
    #[cfg(target_feature = "neon")]
    {
        println!("neon is enabled",);
        counters.push(Box::new(BitSliceBitCounter::<uint8x16_t, 4> {
            slice_type: PhantomData::default(),
        }));
    }
    counters.iter().for_each(|x| {
        dbg!(x.name());
    });

    let mut results: Vec<_> = [32, 48, 64, 96]
        .iter()
        .cloned()
        .map(|n_threads| {
            (1..=4096)
                .map(|chunk_size| (n_threads, chunk_size))
                .collect::<Vec<_>>()
        })
        .flatten()
        .map(|(n_threads, chunk_size)| {
            counters
                .iter()
                .filter(|alg| alg.width() <= chunk_size)
                .filter(|alg| (chunk_size % alg.width()) == 0)
                .map(|alg| {
                    let cost = alg.cost() / n_threads as f64;
                    let dataset_size: usize =
                        ((100_000f64 / cost) as usize / chunk_size) * chunk_size;
                    let dataset_size = dataset_size.min(2usize.pow(25));
                    println!(
                        "{} {} {} {}",
                        alg.name(),
                        n_threads,
                        dataset_size,
                        chunk_size
                    );
                    run_test(
                        alg,
                        &input[0..dataset_size],
                        &target[0..dataset_size],
                        n_threads,
                        chunk_size,
                    )
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();

    dbg!(results.len());

    //results.sort_by_key(|r| Reverse(r.total_nanos));
    //dbg!(&results[0..30]);
    let perf_results = PerfResults {
        machine_type: "".to_string(),
        cpu_arch: "".to_string(),
        n_cores: n_cores,
        n_physical: num_cpus::get_physical(),
        price: 0.0,
        spot_price: 0.0,
        tests: results,
    };

    let mut w = File::create(&Path::new("perf_results.json")).unwrap();
    serde_json::to_writer(w, &perf_results).unwrap();
}
