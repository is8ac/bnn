use rand::{Rng, SeedableRng};
use rand_hc::Hc128Rng;
use std::convert::TryFrom;
use std::time::Instant;

#[inline(always)]
fn transpose_64_slow(input: &mut [u64; 64]) {
    let mut target = [0u64; 64];

    for y in 0..64 {
        for x in 0..64 {
            let bit = (input[x] >> y) & 0b_1u64;
            target[y] |= bit << x;
        }
    }
    *input = target;
}

// Hacker's Delight 7-7
#[inline(always)]
fn transpose_64(a: &mut [u64; 64]) {
    let mut m: u64 = !0u64 >> 32;
    let mut j: usize = 32;
    while j != 0 {
        let mut k: usize = 0;
        let mut t: u64;
        while k < 64 {
            t = (a[k] ^ a[k | j] >> j) & m;
            a[k] ^= t;
            a[k | j] ^= t << j;
            k = (k | j) + 1 & !j
        }
        j >>= 1;
        m ^= m << j
    }
}

fn transpose<const L: usize>(input: &[[u64; L]]) -> [[Vec<u64>; 64]; L] {
    let len = input.len() / 64;
    let target: Vec<[Vec<u64>; 64]> = (0..L)
        .map(|_| {
            let target: Vec<Vec<u64>> = (0..64).map(|_| (0..len).map(|_| 0u64).collect()).collect();
            <[Vec<u64>; 64]>::try_from(target).unwrap()
        })
        .collect();
    let mut target = <[[Vec<u64>; 64]; L]>::try_from(target).unwrap();

    input.chunks(64).enumerate().for_each(|(i, chunk)| {
        let chunk: &[[u64; L]; 64] = <&[[u64; L]; 64]>::try_from(chunk).unwrap();
        for l in 0..L {
            let mut block: [u64; 64] = [0u64; 64];
            for i in 0..64 {
                block[i] = chunk[i][l];
            }
            transpose_64(&mut block);
            for w in 0..64 {
                target[l][w][i] = block[w];
            }
        }
    });

    target
}

const N_EXAMPLES_EXP: u32 = 18;
const N_EXAMPLES: usize = 2usize.pow(N_EXAMPLES_EXP);
const INPUT_LEN: usize = 16;
const TARGET_LEN: usize = 4;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let input: Vec<[u64; INPUT_LEN]> = (0..N_EXAMPLES).map(|_| rng.gen()).collect();
    let target: Vec<[u64; TARGET_LEN]> = (0..N_EXAMPLES).map(|_| rng.gen()).collect();

    let transpose_start = Instant::now();
    let t_input = transpose(&input);
    let t_target = transpose(&target);
    println!(
        "transpose: {:.3} ns/example",
        transpose_start.elapsed().as_nanos() as f64 / N_EXAMPLES as f64
    );
    assert_eq!(t_input[0][0].len(), N_EXAMPLES / 64);

    let popcnt_start = Instant::now();
    let mut counts = Box::new([[[[0u32; 64]; INPUT_LEN]; 64]; TARGET_LEN]);
    for i in 0..INPUT_LEN {
        for o in 0..TARGET_LEN {
            for iw in 0..64 {
                for ow in 0..64 {
                    counts[o][ow][i][iw] = t_input[i][iw]
                        .iter()
                        .zip(t_target[o][ow].iter())
                        .map(|(a, b)| (a & b).count_ones())
                        .sum();
                }
            }
        }
    }
    let mut cv_counts = Box::new([[[[0u32; 64]; INPUT_LEN]; 64]; INPUT_LEN]);
    for i in 0..INPUT_LEN {
        for o in 0..INPUT_LEN {
            for iw in 0..64 {
                for ow in 0..64 {
                    cv_counts[o][ow][i][iw] = t_input[i][iw]
                        .iter()
                        .zip(t_input[o][ow].iter())
                        .map(|(a, b)| (a & b).count_ones())
                        .sum();
                }
            }
        }
    }
    let mut i_counts = Box::new([[0u32; 64]; INPUT_LEN]);
    for i in 0..INPUT_LEN {
        for iw in 0..64 {
            i_counts[i][iw] = t_input[i][iw].iter().map(|x| x.count_ones()).sum();
        }
    }
    let mut o_counts = Box::new([[0u32; 64]; TARGET_LEN]);
    for o in 0..TARGET_LEN {
        for ow in 0..64 {
            o_counts[o][ow] = t_target[o][ow].iter().map(|x| x.count_ones()).sum();
        }
    }
    println!(
        "popcnt: {:.3} ns/example",
        popcnt_start.elapsed().as_nanos() as f64 / N_EXAMPLES as f64
    );
    dbg!(counts[0][0][0][0]);
    dbg!(o_counts[0][0]);
    dbg!(i_counts[0][0]);
    dbg!(N_EXAMPLES / 4);
    dbg!(cv_counts);
}
