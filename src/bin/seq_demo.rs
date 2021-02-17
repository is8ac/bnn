use bnn::bits::{
    b128, b32, b64, BitMap, BitPack, BitScaler, PackedIndexSGet, PackedMap, WeightArray,
    ZipBitFold, BMA,
};
use bnn::shape::{IndexGet, LongDefault, Map, Pack, Shape};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::str;
use std::time::Instant;

fn onehot_encode_byte(byte: u8) -> [b32; 8] {
    let mut target = [b32::ZEROS; 8];
    target[(byte / 32) as usize].set_bit_in_place((byte % 32) as usize, true);
    target
}

fn top1_byte(acts: &[[u32; 32]; 8]) -> u8 {
    <[[(); 32]; 8] as Shape>::indices()
        .enumerate()
        .max_by_key(|&(_, i)| <[[(); 32]; 8] as IndexGet<u32>>::index_get(acts, i))
        .unwrap()
        .0 as u8
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
enum FcIndex<I: Shape, O: Shape, T: Shape> {
    Encoder((O::Index, I::Index)),
    Decoder((T::Index, O::Index)),
}

impl<I: Shape, O: Shape, T: Shape> FcIndex<I, O, T>
where
    I::Index: Eq,
    O::Index: Eq,
    T::Index: Eq,
{
    fn collide(&self, rhs: &Self) -> bool {
        match self {
            FcIndex::Encoder((ao, ai)) => match rhs {
                FcIndex::Encoder((bo, bi)) => (ao == bo) | (ai == bi),
                FcIndex::Decoder((bt, bo)) => ao == bo,
            },
            FcIndex::Decoder((at, ao)) => match rhs {
                FcIndex::Encoder((bo, bi)) => ao == bo,
                FcIndex::Decoder((bt, bo)) => (at == bt) | (ao == bo),
            },
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct FC<W, I, O, T>
where
    I: Shape + BitPack<W>,
    O: Shape + Pack<<I as BitPack<W>>::T> + BitPack<W>,
    T: Shape + Pack<<O as BitPack<W>>::T>,
    <O as Pack<<I as BitPack<W>>::T>>::T: Copy + Eq + std::fmt::Debug,
    <T as Pack<<O as BitPack<W>>::T>>::T: Copy + Eq + std::fmt::Debug,
{
    encoder: <O as Pack<<I as BitPack<W>>::T>>::T,
    decoder: <T as Pack<<O as BitPack<W>>::T>>::T,
}

impl<W, I, O, T> FC<W, I, O, T>
where
    FC<W, I, O, T>: Copy,
    W: BitScaler + std::fmt::Debug,
    O: Shape
        + Map<i32, i32>
        + Map<<I as BitPack<W>>::T, <I as Pack<i32>>::T>
        + Map<<I as BitPack<W>>::T, <<W as BitScaler>::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>
        + Map<
            <W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T,
            <W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T,
        > + Pack<u32>
        + Pack<i32>
        + Pack<<I as Pack<i32>>::T>
        + Pack<<I as BitPack<W>>::T>
        + Pack<<<W as BitScaler>::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>
        + BitMap<bool, i32>
        + BitPack<W>
        + BitPack<bool>
        + IndexGet<i32>
        + IndexGet<<I as BitPack<W>>::T>
        + IndexGet<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>
        + PackedMap<<I as BitPack<W>>::T, bool>
        + WeightArray<W>,
    I: Shape
        + BitPack<W>
        + BitPack<bool>
        + WeightArray<W>
        + Pack<i32>
        + BitMap<bool, i32>
        + IndexGet<i32>
        + Map<i32, i32>,
    T: Shape
        + Map<<O as BitPack<W>>::T, u32>
        + Map<
            <W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T,
            <W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T,
        > + Pack<<O as Pack<i32>>::T>
        + Pack<<O as BitPack<W>>::T>
        + Pack<<<W as BitScaler>::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>
        + Pack<<<W as BitScaler>::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>
        + BitPack<bool>
        + PackedIndexSGet<bool>
        + IndexGet<u32>
        + IndexGet<<<W as BitScaler>::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>
        + IndexGet<<O as BitPack<W>>::T>
        + ZipBitFold<i32, u32, bool>,
    W::ValuesShape: Pack<<I as Pack<i32>>::T>
        + Pack<<O as Pack<i32>>::T>
        + Map<W, <I as Pack<i32>>::T>
        + IndexGet<<I as Pack<i32>>::T>
        + IndexGet<W>
        + IndexGet<<O as Pack<i32>>::T>
        + Map<<O as Pack<i32>>::T, <O as Pack<i32>>::T>
        + Map<<I as Pack<i32>>::T, <I as Pack<i32>>::T>,
    I::Index: Eq,
    O::Index: Eq,
    T::Index: Eq,
    FcIndex<I, O, T>: Copy + std::fmt::Debug,
    <T as Pack<u32>>::T: Copy,
    <I as BitPack<bool>>::T: Copy,
    <I as BitPack<W>>::T: Copy,
    <O as Pack<u32>>::T: LongDefault,
    <O as BitPack<bool>>::T: Copy,
    <O as BitPack<W>>::T: Copy,
    <O as Pack<<I as BitPack<W>>::T>>::T: Copy + Eq + std::fmt::Debug,
    <T as Pack<<O as BitPack<W>>::T>>::T: Copy + Eq + std::fmt::Debug,
    (
        <O as Pack<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::T,
        <T as Pack<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::T,
    ): LongDefault + Sync + Send,
    Standard: Distribution<<O as Pack<<I as BitPack<W>>::T>>::T>,
    Standard: Distribution<<T as Pack<<O as BitPack<W>>::T>>::T>,
{
    fn rand<R: Rng>(rng: &mut R) -> Self {
        FC {
            encoder: rng.gen(),
            decoder: rng.gen(),
        }
    }
    fn mutate(mut self, index: FcIndex<I, O, T>, value: W) -> Self {
        match index {
            FcIndex::Encoder((o, i)) => {
                let weights =
                    <O as IndexGet<<I as BitPack<W>>::T>>::index_get_mut(&mut self.encoder, o);
                <I as PackedIndexSGet<W>>::set_in_place(weights, i, value);
            }
            FcIndex::Decoder((t, o)) => {
                let weights =
                    <T as IndexGet<<O as BitPack<W>>::T>>::index_get_mut(&mut self.decoder, t);
                <O as PackedIndexSGet<W>>::set_in_place(weights, o, value);
            }
        }
        self
    }
    fn hidden(&self, input: &<I as BitPack<bool>>::T) -> <O as BitPack<bool>>::T {
        <O as PackedMap<<I as BitPack<W>>::T, bool>>::map(&self.encoder, |w| {
            <I as WeightArray<W>>::act(&w, input)
        })
    }
    fn acts(&self, hidden: &<O as BitPack<bool>>::T) -> <T as Pack<u32>>::T {
        <T as Map<<O as BitPack<W>>::T, u32>>::map(&self.decoder, |w| O::bma(w, &hidden))
    }
    fn loss(&self, acts: &<T as Pack<u32>>::T, target: &<T as BitPack<bool>>::T) -> i32 {
        <T as ZipBitFold<i32, u32, bool>>::zip_fold(0, &acts, target, |loss, &act, sign| {
            let targ = O::N as u32 * sign as u32;
            let dist = act.saturating_sub(targ) | targ.saturating_sub(act);
            loss + dist.pow(2) as i32
        })
    }
    fn increment_dense_loss_deltas(
        &self,
        input: &<I as BitPack<bool>>::T,
        target: &<T as BitPack<bool>>::T,
        deltas: &mut (
            <O as Pack<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::T,
            <T as Pack<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::T,
        ),
    ) {
        let states = W::STATES;
        let hidden = self.hidden(input);
        let acts = self.acts(&hidden);
        let null_loss = self.loss(&acts, target);

        <O as Shape>::indices().for_each(|o| {
            let weights = <O as IndexGet<<I as BitPack<W>>::T>>::index_get(&self.encoder, o);
            let cur_sum = <I as BMA<W>>::bma(weights, input);
            if <I as WeightArray<W>>::mutant_act(cur_sum).is_none() {
                let deltas = <O as IndexGet<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::index_get_mut(&mut deltas.0, o);
                let cur_act = cur_sum > <I as WeightArray<W>>::THRESHOLD;
                let new_hidden = <O as PackedIndexSGet<bool>>::set(hidden, o, !cur_act);
                let new_loss = self.loss(&self.acts(&new_hidden), target);
                let mut_loss_delta = new_loss - null_loss;
                <W::ValuesShape as Map<W, <I as Pack<i32>>::T>>::map_mut(&W::STATES, deltas, |&w, deltas| {
                    let acts = <I as WeightArray<W>>::acts(weights, input, w);
                    <I as BitMap<bool, i32>>::map_mut(&acts, deltas, |new_act, delta| {
                        if new_act != cur_act {
                            *delta += mut_loss_delta;
                        }
                    })
                })
            }
        });

        <T as Shape>::indices().for_each(|t| {
            let deltas =
                <T as IndexGet<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::index_get_mut(
                    &mut deltas.1,
                    t,
                );
            let weights = <T as IndexGet<<O as BitPack<W>>::T>>::index_get(&self.decoder, t);
            let cur_sum = <O as BMA<W>>::bma(weights, &hidden);
            let else_loss = {
                let targ = O::N as u32 * <T as PackedIndexSGet<bool>>::get(target, t) as u32;
                let dist = cur_sum.saturating_sub(targ) | targ.saturating_sub(cur_sum);
                null_loss - dist.pow(2) as i32
            };
            <W::ValuesShape as Shape>::indices().for_each(|wi| {
                let new_weight = <W::ValuesShape as IndexGet<W>>::index_get(&states, wi);
                let deltas =
                    <W::ValuesShape as IndexGet<<O as Pack<i32>>::T>>::index_get_mut(deltas, wi);
                <O as Shape>::indices().for_each(|o| {
                    let hidden_sign: bool = <O as PackedIndexSGet<bool>>::get(&hidden, o);
                    let null_weight: W = <O as PackedIndexSGet<W>>::get(&weights, o);

                    let null_bit_act = null_weight.bma(hidden_sign);
                    let new_bit_act = new_weight.bma(hidden_sign);

                    let new_loss = {
                        let new_sum = (cur_sum - null_bit_act) + new_bit_act;
                        let targ =
                            O::N as u32 * <T as PackedIndexSGet<bool>>::get(target, t) as u32;
                        let dist = new_sum.saturating_sub(targ) | targ.saturating_sub(new_sum);
                        else_loss + dist.pow(2) as i32
                    };

                    let delta = <O as IndexGet<i32>>::index_get_mut(deltas, o);
                    *delta += new_loss - null_loss;
                });
            });
        })
    }
    fn increment_dense_loss_deltas_slow(
        &self,
        input: &<I as BitPack<bool>>::T,
        target: &<T as BitPack<bool>>::T,
        deltas: &mut (
            <O as Pack<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::T,
            <T as Pack<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::T,
        ),
    ) {
        let null_loss = self.loss(&self.acts(&self.hidden(input)), target);
        let states = W::STATES;

        <O as Shape>::indices().for_each(|o| {
            let deltas =
                <O as IndexGet<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::index_get_mut(
                    &mut deltas.0,
                    o,
                );
            <W::ValuesShape as Shape>::indices().for_each(|wi| {
                let w = <W::ValuesShape as IndexGet<W>>::index_get(&states, wi);
                let deltas =
                    <W::ValuesShape as IndexGet<<I as Pack<i32>>::T>>::index_get_mut(deltas, wi);
                <I as Shape>::indices().for_each(|i| {
                    let mut_weights = self.mutate(FcIndex::Encoder((o, i)), *w);
                    let mut_loss =
                        mut_weights.loss(&mut_weights.acts(&mut_weights.hidden(input)), target);

                    let delta = <I as IndexGet<i32>>::index_get_mut(deltas, i);
                    *delta += mut_loss - null_loss;
                })
            })
        });

        <T as Shape>::indices().for_each(|t| {
            let deltas =
                <T as IndexGet<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::index_get_mut(
                    &mut deltas.1,
                    t,
                );
            <W::ValuesShape as Shape>::indices().for_each(|wi| {
                let states = W::STATES;
                let w = <W::ValuesShape as IndexGet<W>>::index_get(&states, wi);
                let deltas =
                    <W::ValuesShape as IndexGet<<O as Pack<i32>>::T>>::index_get_mut(deltas, wi);
                <O as Shape>::indices().for_each(|o| {
                    let mut_weights = self.mutate(FcIndex::Decoder((t, o)), *w);
                    let mut_loss =
                        mut_weights.loss(&mut_weights.acts(&mut_weights.hidden(input)), target);

                    let delta = <O as IndexGet<i32>>::index_get_mut(deltas, o);
                    *delta += mut_loss - null_loss;
                })
            })
        });
    }
    fn merge_deltas(
        a: (
            <O as Pack<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::T,
            <T as Pack<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::T,
        ),
        mut b: (
            <O as Pack<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::T,
            <T as Pack<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::T,
        ),
    ) -> (
        <O as Pack<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::T,
        <T as Pack<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::T,
    ) {
        <O as Map<
            <W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T,
            <W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T,
        >>::map_mut(&a.0, &mut b.0, |a, b| {
            <W::ValuesShape as Map<<I as Pack<i32>>::T, <I as Pack<i32>>::T>>::map_mut(
                a,
                b,
                |a, b| <I as Map<i32, i32>>::map_mut(a, b, |a, b| *b += a),
            )
        });
        <T as Map<
            <W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T,
            <W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T,
        >>::map_mut(&a.1, &mut b.1, |a, b| {
            <W::ValuesShape as Map<<O as Pack<i32>>::T, <O as Pack<i32>>::T>>::map_mut(
                a,
                b,
                |a, b| <O as Map<i32, i32>>::map_mut(a, b, |a, b| *b += a),
            )
        });
        b
    }
    // filters to neg
    fn to_sparse_vec(
        deltas: (
            <O as Pack<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::T,
            <T as Pack<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::T,
        ),
        threshold: i32,
    ) -> Vec<(FcIndex<I, O, T>, W, i32)> {
        let states = W::STATES;
        <O as Shape>::indices()
            .map(|o| {
                let deltas =
                    <O as IndexGet<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::index_get(
                        &deltas.0, o,
                    );
                <W::ValuesShape as Shape>::indices()
                    .map(|wi| {
                        let deltas = <W::ValuesShape as IndexGet<<I as Pack<i32>>::T>>::index_get(
                            deltas, wi,
                        );
                        let w = <W::ValuesShape as IndexGet<W>>::index_get(&states, wi);
                        <I as Shape>::indices()
                            .map(|i| {
                                let delta = <I as IndexGet<i32>>::index_get(deltas, i);
                                (FcIndex::Encoder((o, i)), *w, *delta)
                            })
                            .filter(|&(_, _, d)| d < threshold)
                            .collect::<Vec<_>>()
                    })
                    .flatten()
                    .collect::<Vec<_>>()
            })
            .flatten()
            .chain(
                <T as Shape>::indices()
                    .map(|t| {
                        let deltas = <T as IndexGet<
                            <W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T,
                        >>::index_get(&deltas.1, t);
                        <W::ValuesShape as Shape>::indices()
                            .map(|wi| {
                                let deltas =
                                    <W::ValuesShape as IndexGet<<O as Pack<i32>>::T>>::index_get(
                                        deltas, wi,
                                    );
                                let w = <W::ValuesShape as IndexGet<W>>::index_get(&states, wi);
                                <O as Shape>::indices()
                                    .map(|o| {
                                        let delta = <O as IndexGet<i32>>::index_get(deltas, o);
                                        (FcIndex::Decoder((t, o)), *w, *delta)
                                    })
                                    .filter(|&(_, _, d)| d < threshold)
                                    .collect::<Vec<_>>()
                            })
                            .flatten()
                            .collect::<Vec<_>>()
                    })
                    .flatten(),
            )
            .collect()
    }
    fn apply_top_n_nocollide(
        mut self,
        deltas: Vec<(FcIndex<I, O, T>, W, i32)>,
        n: usize,
        max_n: usize,
    ) -> (Self, Vec<(FcIndex<I, O, T>, W, i32)>, usize) {
        deltas
            .iter()
            .min_by_key(|&(_, _, l)| l)
            .filter(|_| n < max_n)
            .map(|&(index, weight, _)| {
                match index {
                    FcIndex::Encoder(_) => {
                        println!("encoder",);
                    }
                    _ => (),
                }
                self.mutate(index, weight).apply_top_n_nocollide(
                    deltas
                        .iter()
                        .filter(|(i, w, _)| !index.collide(i))
                        .cloned()
                        .collect(),
                    n + 1,
                    max_n,
                )
            })
            .unwrap_or_else(|| (self, deltas, n))
    }
    fn update(
        self,
        deltas: (
            <O as Pack<<W::ValuesShape as Pack<<I as Pack<i32>>::T>>::T>>::T,
            <T as Pack<<W::ValuesShape as Pack<<O as Pack<i32>>::T>>::T>>::T,
        ),
        max_n: usize,
    ) -> Self {
        let deltas = Self::to_sparse_vec(deltas, 0);
        self.apply_top_n_nocollide(deltas, 0, max_n).0
    }
}

fn load_data<P: AsRef<Path>, const N: usize>(path: P) -> Vec<([[b32; 8]; N], [b32; 8])>
where
    [[b32; 8]; N]: Default,
{
    let file = File::open(path).unwrap();
    let mut buf_reader = BufReader::new(file);
    let bytes: Vec<[b32; 8]> = buf_reader
        .bytes()
        .map(|x| onehot_encode_byte(x.unwrap()))
        .collect();
    (0..bytes.len() - (N + 1))
        .map(|i| {
            let mut input = <[[b32; 8]; N]>::default();
            for x in 0..N {
                input[x] = bytes[i + x];
            }
            let target = bytes[i + N];
            (input, target)
        })
        .collect()
}

const N_STEPS: usize = 4;
type WeightType = bool;
type CharShape = [[(); 32]; 8];
type InputShape = [CharShape; N_STEPS];
type OutputShape = [[(); 32]; 4];
type DeltasAcc = (
    <OutputShape as Pack<
        <<WeightType as BitScaler>::ValuesShape as Pack<<InputShape as Pack<i32>>::T>>::T,
    >>::T,
    <CharShape as Pack<
        <<WeightType as BitScaler>::ValuesShape as Pack<<OutputShape as Pack<i32>>::T>>::T,
    >>::T,
);

const N_WORKERS: usize = 8;
const BATCH_SIZE: usize = 1000;

fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(N_WORKERS)
        .stack_size(2usize.pow(30))
        .build_global()
        .unwrap();
    dbg!(std::mem::size_of::<DeltasAcc>());
    let mut rng = Hc128Rng::seed_from_u64(0);

    let ngrams = load_data::<_, N_STEPS>("tiny-shakespeare.txt");

    let model = FC::<bool, InputShape, OutputShape, CharShape>::rand(&mut rng);

    let chunk_size = BATCH_SIZE / N_WORKERS;
    let chunks: Vec<(usize, usize)> = (0..N_WORKERS)
        .map(|i| (i * chunk_size, (i + 1) * chunk_size))
        .collect();
    let model = ngrams.chunks_exact(BATCH_SIZE).fold(model, |model, slice| {
        let start = Instant::now();
        let deltas = chunks
            .par_iter()
            .map(|range| {
                slice[range.0..range.1].iter().fold(
                    <DeltasAcc>::long_default(),
                    |mut deltas, (input, target)| {
                        model.increment_dense_loss_deltas(&input, &target, &mut deltas);
                        deltas
                    },
                )
            })
            .reduce_with(|a, b| FC::<bool, InputShape, OutputShape, CharShape>::merge_deltas(a, b))
            .unwrap();
        //dbg!(start.elapsed());

        let model = model.update(deltas, OutputShape::N / 4);
        let bytes: Vec<u8> = slice
            .iter()
            .take(20)
            .map(|(input, target)| {
                let acts = model.acts(&model.hidden(input));
                top1_byte(&acts)
            })
            .collect();
        dbg!(str::from_utf8(&bytes));
        let n_correct: u64 = slice
            .iter()
            .map(|(input, target)| {
                let acts = model.acts(&model.hidden(input));
                let top_byte = top1_byte(&acts);
                let expanded_byte = onehot_encode_byte(top_byte);

                (expanded_byte == *target) as u64
            })
            .sum();
        dbg!(n_correct as f64 / slice.len() as f64);
        model
    });
}
