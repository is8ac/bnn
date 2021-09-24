use crate::bits::{b64, t64};
use crate::bitslice::BitSlice;
use crate::count::ElementwiseAdd;
use std::arch::x86_64::{__m256i, _mm256_setzero_si256};
use std::arch::x86_64::{__m512i, _mm512_setzero_si512};
use std::convert::TryFrom;
use std::convert::TryInto;
use std::mem;

#[inline(always)]
fn transpose_64_slow(input: &mut [u64; 64]) {
    let mut target = [0u64; 64];

    for y in 0..64 {
        for x in 0..64 {
            let bit = (input[x] >> y) & 0b_1u64;
            target[-(y as isize - 63) as usize] |= bit << (-(x as isize - 63) as usize);
        }
    }
    *input = target;
}

// Hacker's Delight 7-3 fig 7-7
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
#[inline(never)]
pub fn transpose<const L: usize, const N: usize>(
    input: &[[b64; L]; N * 64],
) -> [[[b64; N]; 64]; L] {
    let mut target: [[[b64; N]; 64]; L] = (0..L)
        .map(|_| {
            (0..64)
                .map(|_| {
                    (0..N)
                        .map(|_| b64(0u64))
                        .collect::<Vec<b64>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<[b64; N]>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<[[b64; N]; 64]>>()
        .try_into()
        .unwrap();
    input.chunks(64).enumerate().for_each(|(i, chunk)| {
        let chunk: &[[b64; L]; 64] = <&[[b64; L]; 64]>::try_from(chunk).unwrap();
        for l in 0..L {
            let mut block: [u64; 64] = [0u64; 64];
            for i in 0..64 {
                block[i] = chunk[i][l].0;
            }
            transpose_64(&mut block);
            for w in 0..64 {
                target[l][w][i] = b64(block[w]);
            }
        }
    });

    target
}

#[inline(never)]
pub fn transpose_256<const L: usize, const N: usize>(
    input: &[[b64; L]; N * 256],
) -> [[__m256i; N]; 64 * L] {
    let mut target: [[__m256i; N]; 64 * L] = (0..(L * 64))
        .map(|_| {
            (0..N)
                .map(|_| unsafe { _mm256_setzero_si256() })
                .collect::<Vec<__m256i>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<[__m256i; N]>>()
        .try_into()
        .unwrap();

    input.chunks(256).enumerate().for_each(|(i, chunk)| {
        let chunk: &[[b64; L]; 256] = <&[[b64; L]; 256]>::try_from(chunk).unwrap();
        for l in 0..L {
            let mut block: [[u64; 64]; 4] = [[0u64; 64]; 4];
            for w in 0..4 {
                for b in 0..64 {
                    block[w][b] = chunk[w * 64 + b][l].0;
                }
                transpose_64(&mut block[w]);
            }
            for b in 0..64 {
                let mut row = [0u64; 4];
                for w in 0..4 {
                    row[w] = block[w][b];
                }
                target[l * 64 + b][i] = unsafe { mem::transmute(row) };
            }
        }
    });

    target
}

pub fn block_transpose_256<const L: usize>(input: &[[b64; L]; 256]) -> [__m256i; 64 * L] {
    let mut target = [unsafe { _mm256_setzero_si256() }; 64 * L];

    for l in 0..L {
        let mut block: [[u64; 64]; 4] = [[0u64; 64]; 4];
        for w in 0..4 {
            for b in 0..64 {
                block[w][b] = input[w * 64 + b][l].0;
            }
            transpose_64(&mut block[w]);
        }
        for b in 0..64 {
            let mut row = [0u64; 4];
            for w in 0..4 {
                row[w] = block[w][b];
            }
            target[l * 64 + b] = unsafe { mem::transmute(row) };
        }
    }
    target
}

pub fn block_transpose_512<const L: usize>(input: &[[b64; L]; 512]) -> [__m512i; 64 * L] {
    let mut target = [unsafe { _mm512_setzero_si512() }; 64 * L];

    for l in 0..L {
        let mut block: [[u64; 64]; 8] = [[0u64; 64]; 8];
        for w in 0..8 {
            for b in 0..64 {
                block[w][b] = input[w * 64 + b][l].0;
            }
            transpose_64(&mut block[w]);
        }
        for b in 0..64 {
            let mut row = [0u64; 8];
            for w in 0..8 {
                row[w] = block[w][b];
            }
            target[l * 64 + b] = unsafe { mem::transmute(row) };
        }
    }
    target
}

#[derive(Debug)]
pub struct Counters<const I: usize, const T: usize> {
    pub n: u64,
    pub cv: [[[[u64; 64]; I]; 64]; I],
    pub counts: [[[[u64; 64]; I]; 64]; T],
    pub input: [[u64; 64]; I],
    pub target: [[u64; 64]; T],
}

impl<const I: usize, const T: usize> Counters<I, T>
where
    [u64; I * 64]: TryFrom<Vec<u64>>,
    [u64; T * 64]: TryFrom<Vec<u64>>,
{
    pub fn new_box() -> Box<Self> {
        Box::new(Counters {
            n: 0u64,
            cv: [[[[0u64; 64]; I]; 64]; I],
            counts: [[[[0u64; 64]; I]; 64]; T],
            input: [[0u64; 64]; I],
            target: [[0u64; 64]; T],
        })
    }
    pub fn merge(a: &mut Box<Counters<I, T>>, b: Box<Counters<I, T>>) {
        a.n += b.n;
        a.cv.elementwise_add(&b.cv);
        a.counts.elementwise_add(&b.counts);
        a.input.elementwise_add(&b.input);
        a.target.elementwise_add(&b.target);
    }
    pub fn increment<const L: usize>(
        &mut self,
        input: &[[[b64; L]; 64]; I],
        target: &[[[b64; L]; 64]; T],
    ) {
        self.n += (L * 64) as u64;
        for i in 0..I {
            for o in 0..T {
                for iw in 0..64 {
                    for ow in 0..64 {
                        self.counts[o][ow][i][iw] += input[i][iw]
                            .iter()
                            .zip(target[o][ow].iter())
                            .map(|(a, b)| (*a ^ *b).count_ones())
                            .sum::<u32>()
                            as u64;
                    }
                }
            }
        }
        for i in 0..I {
            for o in 0..I {
                for iw in 0..64 {
                    for ow in 0..64 {
                        self.cv[o][ow][i][iw] += input[i][iw]
                            .iter()
                            .zip(input[o][ow].iter())
                            .map(|(a, b)| (*a ^ *b).count_ones())
                            .sum::<u32>() as u64;
                    }
                }
            }
        }
        for i in 0..I {
            for iw in 0..64 {
                self.input[i][iw] +=
                    input[i][iw].iter().map(|x| x.count_ones()).sum::<u32>() as u64;
            }
        }
        for o in 0..T {
            for ow in 0..64 {
                self.target[o][ow] +=
                    target[o][ow].iter().map(|x| x.count_ones()).sum::<u32>() as u64;
            }
        }
    }
    pub fn to_trits(&self) -> [[([t64; I], u32); 64]; T] {
        let input = self.input.iter().flatten().cloned().collect::<Vec<u64>>();
        let input = <&[u64; I * 64]>::try_from(&input[0..]).unwrap();

        let cv = {
            let mut targ = Box::new([[0u64; I * 64]; I * 64]);
            for x in 0..I * 64 {
                for y in 0..I * 64 {
                    targ[x][y] = self.cv[x / 64][x % 64][y / 64][y % 64];
                }
            }
            targ
        };

        let mut target = [[([t64(0, 0); I], 0u32); 64]; T];
        for t in 0..T {
            for b in 0..64 {
                let mut i_xor_t = [0u64; I * 64];
                for i in 0..I * 64 {
                    i_xor_t[i] = self.counts[t][b][i / 64][i % 64];
                }
                let trits =
                    quantize::<{ I * 64 }>(self.n, self.target[t][b], &input, &i_xor_t, &cv);
                for i in 0..I * 64 {
                    target[t][b].0[i / 64].set_trit_in_place(i % 64, trits[i]);
                }
                let n_set: u32 = trits.iter().map(|trit| trit.is_some() as u32).sum();
                target[t][b].1 = n_set;
            }
        }
        target
    }
}

fn quantize<const I: usize>(
    n: u64,
    t: u64,
    i: &[u64; I],
    i_xor_t: &[u64; I],
    i_cv: &[[u64; I]; I],
) -> [Option<bool>; I] {
    let cv_probs: [[f64; I]; I] = i_cv
        .iter()
        .map(|x| {
            x.iter()
                .map(|&c| -((c as f64 / n as f64) * 2f64 - 1f64).abs())
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mut targ_magns: Vec<f64> = i_xor_t
        .iter()
        .map(|&x| ((x as f64 / n as f64) * 2f64 - 1f64).abs())
        .collect();
    let mut indices: Vec<usize> = Vec::new();
    let mut cost = 0f64;
    while cost < 3.34f64 {
        let mut values: Vec<(usize, f64)> = cv_probs
            .iter()
            .zip(targ_magns.iter())
            .map(|(cv, p)| indices.iter().map(|&index| cv[index]).product::<f64>() * p)
            .enumerate()
            .collect();
        //dbg!(&values);
        values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (i, weight) = values.pop().unwrap();
        //dbg!(i);
        indices.push(i);
        cost += targ_magns[i];
        //dbg!(cost);
    }
    //dbg!(&indices);
    let mut target = [None; I];
    indices.iter().for_each(|&i| {
        let sign = (i_xor_t[i] as f64 / n as f64) > 0.5;
        target[i] = Some(sign);
    });
    target
}

#[cfg(test)]
mod test {
    use crate::matrix::{transpose_64, transpose_64_slow};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    use std::convert::TryInto;

    #[test]
    fn transpose_64_test() {
        let mut rng = Hc128Rng::seed_from_u64(0);
        (0..1000).for_each(|_| {
            let input: [u64; 64] = (0..64)
                .map(|_| rng.gen())
                .collect::<Vec<u64>>()
                .try_into()
                .unwrap();
            let mut fast = input;
            transpose_64(&mut fast);
            let mut slow = input;
            transpose_64_slow(&mut slow);
            assert_eq!(fast, slow);
        });
    }
}
