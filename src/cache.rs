use crate::shape::Shape;
use std::marker::PhantomData;

pub trait BaseCache
where
    Self::ChanCache: ChanCache,
{
    type ChanIndex;
    type ChanValue;
    type Input;
    type ChanCache;
    //fn init(input: &Self::Input, true_class: usize) -> Self;
    fn chan_cache(&self, chan_val: Self::ChanValue, chan_index: Self::ChanIndex) -> Self::ChanCache;
    //fn loss(&self) -> u64;
}

#[derive(Debug)]
pub struct TritMseBaseCache<InputShape: Shape, const C: usize> {
    input_type: PhantomData<InputShape>,
    sum_loss: u64,
    true_class: usize,
}

#[derive(Debug)]
pub struct TritMseChanCache<InputShape: Shape, const C: usize> {
    input_type: PhantomData<InputShape>,
    else_sum_loss: u64,
    chan_target: u32,
}

/// assums trit weights input
impl<InputShape: Shape, const C: usize> BaseCache for TritMseBaseCache<InputShape, C>
where
    [f64; C]: Default,
{
    type ChanIndex = usize;
    type ChanValue = u32;
    type Input = [u32; C];
    type ChanCache = TritMseChanCache<InputShape, C>;
    //fn init(inputs: &Self::Input, true_class: usize) -> Self {
    //    let mut sum_loss = 0u64;
    //    for c in 0..C {
    //        let target = (InputShape::N as u32 / 2) * (true_class == c) as u32;
    //        let dist = target.saturating_sub(inputs[c]) | inputs[c].saturating_sub(target);
    //        sum_loss += (dist as u64).pow(2);
    //    }

    //    TritMseBaseCache {
    //        input_type: PhantomData::default(),
    //        sum_loss,
    //        true_class: true_class,
    //    }
    //}
    fn chan_cache(&self, chan_val: Self::ChanValue, chan_index: usize) -> Self::ChanCache {
        let target = (InputShape::N as u32 / 2) * (self.true_class == chan_index) as u32;
        let dist = target.saturating_sub(chan_val) | chan_val.saturating_sub(target);
        TritMseChanCache {
            input_type: PhantomData::default(),
            else_sum_loss: self.sum_loss - (dist as u64).pow(2),
            chan_target: target,
        }
    }
    //fn loss(&self) -> u64 {
    //    self.sum_loss
    //}
}

pub trait ChanCache {
    type Mutation;
    fn loss(&self, mutation: &Self::Mutation) -> u64;
    //fn loss_delta_full(acts: &Self::Input, true_class: usize, max_act: u32) -> f64;
}

impl<InputShape: Shape, const C: usize> ChanCache for TritMseChanCache<InputShape, C> {
    type Mutation = u32;
    fn loss(&self, &mutation: &Self::Mutation) -> u64 {
        let dist = self.chan_target.saturating_sub(mutation) | mutation.saturating_sub(self.chan_target);
        self.else_sum_loss + (dist as u64).pow(2)
    }
}

#[cfg(test)]
mod tests {
    use super::{BaseCache, ChanCache, TritMseBaseCache, TritMseChanCache};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_hc::Hc128Rng;
    extern crate test;
    use crate::shape::Shape;
    use test::Bencher;

    const N_CLASSES: usize = 10;
    type InputShape = [(); 32];

    #[test]
    fn rand_mse_cache() {
        let mut rng = Hc128Rng::seed_from_u64(0);
        (0..10000).for_each(|_| {
            let input: [u32; N_CLASSES] = {
                let mut input = [0; N_CLASSES];
                for i in 0..N_CLASSES {
                    input[i] = rng.gen_range(0, InputShape::N as u32);
                }
                input
            };
            let class = rng.gen_range(0, N_CLASSES);
            let chan = rng.gen_range(0, N_CLASSES);
            let true_loss: u64 = input
                .iter()
                .enumerate()
                .map(|(i, act)| {
                    let target = (InputShape::N as u32 / 2) * (class == i) as u32;
                    let dist = target.saturating_sub(*act) | act.saturating_sub(target);
                    (dist as u64).pow(2)
                })
                .sum();

            let base_cache = TritMseBaseCache::<InputShape, N_CLASSES>::init(&input, class);
            let null_loss = base_cache.loss();
            assert_eq!(null_loss, true_loss);
            let chan_cache = base_cache.chan_cache(input[chan], chan);
            let mut_loss = chan_cache.loss(&input[chan]);
            assert_eq!(null_loss, mut_loss);

            let mut_val: u32 = rng.gen_range(0, InputShape::N as u32);
            let mut mut_input = input;
            mut_input[chan] = mut_val;
            let true_mut_loss: u64 = mut_input
                .iter()
                .enumerate()
                .map(|(i, act)| {
                    let target = (InputShape::N as u32 / 2) * (class == i) as u32;
                    let dist = target.saturating_sub(*act) | act.saturating_sub(target);
                    (dist as u64).pow(2)
                })
                .sum();
            let new_mut_loss = chan_cache.loss(&mut_val);
            assert_eq!(new_mut_loss, true_mut_loss);
        });
    }

    #[bench]
    fn bench_cached_loss(b: &mut Bencher) {
        let mut rng = Hc128Rng::seed_from_u64(0);

        let input: [u32; N_CLASSES] = {
            let mut input = [0; N_CLASSES];
            for i in 0..N_CLASSES {
                input[i] = rng.gen_range(0, InputShape::N as u32);
            }
            input
        };
        let update = test::black_box(0);
        let class = test::black_box(5);

        b.iter(|| {
            let base_cache = TritMseBaseCache::<InputShape, N_CLASSES>::init(&input, class);
            let mut sum_loss = 0u64;
            for o in 0..N_CLASSES {
                let chan_cache = base_cache.chan_cache(input[o], o);
                for i in 0..InputShape::N as u32 {
                    let mut_loss = chan_cache.loss(&(update + i));
                    sum_loss += mut_loss;
                }
            }
            sum_loss
        });
    }
    #[bench]
    fn bench_noncached_loss(b: &mut Bencher) {
        let mut rng = Hc128Rng::seed_from_u64(0);

        let input: [u32; N_CLASSES] = {
            let mut input = [0; N_CLASSES];
            for i in 0..N_CLASSES {
                input[i] = rng.gen_range(0, InputShape::N as u32);
            }
            input
        };
        let update = test::black_box(0);
        let class = test::black_box(5);

        b.iter(|| {
            let mut sum_loss = 0u64;
            for o in 0..N_CLASSES {
                for i in 0..InputShape::N as u32 {
                    let mut mut_input = input;
                    mut_input[o] = (update + i);
                    let true_mut_loss: u64 = mut_input
                        .iter()
                        .enumerate()
                        .map(|(i, act)| {
                            let target = (InputShape::N as u32 / 2) * (class == i) as u32;
                            let dist = target.saturating_sub(*act) | act.saturating_sub(target);
                            (dist as u64).pow(2)
                        })
                        .sum();
                    sum_loss += true_mut_loss;
                }
            }
            sum_loss
        });
    }
}
