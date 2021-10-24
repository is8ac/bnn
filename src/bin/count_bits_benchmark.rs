#![feature(generic_const_exprs)]

use bnn::bench::{PerfResults, PerfTest};
use bnn::bits::b64;
use bnn::bitslice::{BitArray64, BitSlice};
use bnn::count_bits::{BitSliceBitCounter, ExpCountBits, PopCountBitCounter, UnitCountBits};
use bnn::ecc::encode_byte;
use bnn::layer;
use bnn::search::{compute_exp_candidates, weights_to_dense, weights_to_sparse};
use bnn::shape::flatten_2d;
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::convert::TryFrom;
use std::convert::TryInto;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::path::Path;
use std::time::Instant;

fn run_test(alg: &Box<dyn CountBits>, input: &[[b64; 8]], target: &[[b64; 4]], chunk_size: usize) -> PerfTest {
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

    let sparse: [(Vec<(usize, bool)>, [u32; UNIT_THRESHOLDS]); 256] = weights.iter().map(|w| weights_to_sparse(w)).collect::<Vec<_>>().try_into().unwrap();

    let unit_start = Instant::now();
    let unit_counts = alg.unit_count_bits(&input, &target, &sparse, chunk_size);
    let unit_duration = unit_start.elapsed();

    let mut hasher = DefaultHasher::new();
    unit_counts.hash(&mut hasher);
    let unit_hash = hasher.finish();

    let exp_candidates: [(Vec<(usize, bool)>, [(usize, bool); 8 as usize], [u32; 3]); 256] = weights
        .iter()
        .zip(unit_counts.iter())
        .map(|(w, counts)| compute_exp_candidates::<512, 2, 3, 8>(w, counts))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let exp_start = Instant::now();
    let exp_counts = alg.exp_count_bits(&input, &target, &exp_candidates, chunk_size);
    let exp_duration = exp_start.elapsed();

    let weights = weights
        .chunks(64)
        .map(|chunk| chunk.iter().map(|w| weights_to_dense(w)).collect::<Vec<_>>().try_into().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let apply_start = Instant::now();
    let _l: Vec<[b64; 4]> = target
        .par_windows(2)
        .map(|slice| {
            let flat = flatten_2d::<b64, 2, 4>(<&[[b64; 4]; 2]>::try_from(&slice[0..2]).unwrap());
            layer::apply(&weights, &flat)
        })
        .collect();
    let apply_duration = apply_start.elapsed();

    let mut hasher = DefaultHasher::new();
    exp_counts.hash(&mut hasher);
    let exp_hash = hasher.finish();

    let total_duration = total_start.elapsed();
    PerfTest {
        algorithm: alg.name(),
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

const UNIT_THRESHOLDS: usize = 2;

trait CountBits: ExpCountBits<8, 4, 3, 8> + UnitCountBits<8, 4, 2> {
    fn name(&self) -> String;
    fn width(&self) -> usize;
}

impl CountBits for PopCountBitCounter {
    fn name(&self) -> String {
        format!("popcnt")
    }
    fn width(&self) -> usize {
        1
    }
}

impl<const B: usize, const L: usize> CountBits for BitSliceBitCounter<BitArray64<L>, B>
where
    [(); BitArray64::<L>::N]: ,
{
    fn name(&self) -> String {
        format!("bitslice-BitArray64x{}_u{}", L, B)
    }
    fn width(&self) -> usize {
        64 * L
    }
}

fn main() {
    let n_cores = num_cpus::get();
    dbg!(n_cores);
    //rayon::ThreadPoolBuilder::new().num_threads(16).stack_size(2usize.pow(30)).build_global().unwrap();
    rayon::ThreadPoolBuilder::new().stack_size(2usize.pow(30)).build_global().unwrap();

    let bytes = std::fs::read("Delphi Complete Works of Charles Dickens (Illustrated) - Charles Dickens.txt").unwrap();

    let expanded: Vec<[b64; 4]> = bytes.par_iter().map(|&b| encode_byte(b)).collect();

    let (input, target): (Vec<[b64; 8]>, Vec<[b64; 4]>) = expanded
        .par_windows(3)
        .map(|slice| {
            let input = flatten_2d::<b64, 2, 4>(<&[[b64; 4]; 2]>::try_from(&slice[0..2]).unwrap());
            (input, slice[2])
        })
        .unzip();

    let counters: Vec<Box<dyn CountBits>> = vec![
        //Box::new(PopCountBitCounter {}),
        Box::new(BitSliceBitCounter::<BitArray64<1>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<2>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<3>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<4>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<5>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<6>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<7>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<8>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<9>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<10>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<12>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<16>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<24>, 4> {
            slice_type: PhantomData::default(),
        }),
        Box::new(BitSliceBitCounter::<BitArray64<32>, 4> {
            slice_type: PhantomData::default(),
        }),
    ];

    let results: Vec<_> = counters
        .iter()
        .map(|alg| {
            let chunk_size = (4096 / alg.width()) * alg.width();
            let dataset_size = (2usize.pow(24) / chunk_size) * chunk_size;
            println!("{} {} {}", alg.name(), dataset_size, chunk_size);
            run_test(alg, &input[0..dataset_size], &target[0..dataset_size], chunk_size)
        })
        .collect::<Vec<_>>();

    dbg!(results.len());

    let perf_results = PerfResults {
        machine_type: "".to_string(),
        cpu_arch: "".to_string(),
        n_cores: n_cores,
        n_physical: num_cpus::get_physical(),
        price: 0.0,
        spot_price: 0.0,
        tests: results,
    };

    let w = File::create(&Path::new("perf_results.json")).unwrap();
    serde_json::to_writer(w, &perf_results).unwrap();
}
