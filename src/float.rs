use crate::bits::{BitArray, BFMA};
use crate::shape::{Element, Shape};
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};

// bit input, float weights and float output.
pub trait BFFVM<S: Shape>
where
    Self: BitArray,
    f32: Element<Self::BitShape> + Element<S>,
    <f32 as Element<Self::BitShape>>::Array: Element<S>,
{
    fn float_mul(
        &self,
        weights: &<<f32 as Element<Self::BitShape>>::Array as Element<S>>::Array,
    ) -> <f32 as Element<S>>::Array;
}

impl<T: BitArray + BFMA> BFFVM<()> for T
where
    f32: Element<T::BitShape>,
{
    fn float_mul(&self, weights: &<f32 as Element<T::BitShape>>::Array) -> f32 {
        self.bfma(weights)
    }
}

impl<S: Shape, T: BitArray + BFFVM<S>, const L: usize> BFFVM<[S; L]> for T
where
    f32: Element<T::BitShape> + Element<S>,
    <f32 as Element<T::BitShape>>::Array: Element<S>,
    [<f32 as Element<S>>::Array; L]: Default,
{
    fn float_mul(
        &self,
        weights: &[<<f32 as Element<Self::BitShape>>::Array as Element<S>>::Array; L],
    ) -> [<f32 as Element<S>>::Array; L] {
        let mut target = <[<f32 as Element<S>>::Array; L]>::default();
        for i in 0..L {
            target[i] = self.float_mul(&weights[i]);
        }
        target
    }
}

/// bit input, float weights and float softmax loss.
pub trait FloatLoss<I: BitArray, C: Shape>
where
    u32: Element<C>,
{
    fn loss(&self, input: &I, true_class: usize) -> f32;
    ///// Given a list of the number of times that this input is of each class, compute the loss.
    fn counts_loss(&self, input: &I, classes_counts: &<u32 as Element<C>>::Array) -> f32;
}

impl<I: BitArray + BFMA, const C: usize> FloatLoss<I, [(); C]>
    for [(<f32 as Element<I::BitShape>>::Array, f32); C]
where
    f32: Element<I::BitShape>,
    [f32; C]: Default,
    [[f32; 2]; C]: Default,
    rand::distributions::Standard:
        rand::distributions::Distribution<<f32 as Element<<I as BitArray>::BitShape>>::Array>,
    I::BitShape: Default,
    [<f32 as Element<I::BitShape>>::Array; C]: Noise,
{
    fn loss(&self, input: &I, true_class: usize) -> f32 {
        let mut exp = <[f32; C]>::default();
        let mut sum_exp = 0f32;
        for c in 0..C {
            exp[c] = (input.bfma(&self[c].0) + self[c].1).exp();
            sum_exp += exp[c];
        }
        let mut sum_loss = 0f32;
        for c in 0..C {
            let scaled = exp[c] / sum_exp;
            sum_loss += (scaled - (c == true_class) as u8 as f32).powi(2);
        }
        sum_loss
    }
    /// `weights.counts_loss(&input, &counts)`` is equivalent to
    ///
    /// ```
    ///counts
    ///    .iter()
    ///    .enumerate()
    ///    .map(|(class, count)| weights.loss(&input, class) * *count as f32)
    ///    .sum()
    ///```
    /// except that floating point is imprecise so it often won't actualy be the same.
    fn counts_loss(&self, input: &I, counts: &[u32; C]) -> f32 {
        let (total_sum_loss, losses_diffs) = {
            let (exp, sum_exp) = {
                let mut exp = <[f32; C]>::default();
                let mut sum_exp = 0f32;
                for c in 0..C {
                    exp[c] = (input.bfma(&self[c].0) + self[c].1).exp();
                    sum_exp += exp[c];
                }
                (exp, sum_exp)
            };
            let mut losses_diffs = <[f32; C]>::default();
            let mut total_sum_loss = 0f32;
            for c in 0..C {
                let scaled = exp[c] / sum_exp;
                let loss_0 = scaled.powi(2);
                let loss_1 = (scaled - 1f32).powi(2);
                losses_diffs[c] = loss_1 - loss_0;
                total_sum_loss += loss_0;
            }
            (total_sum_loss, losses_diffs)
        };
        let mut sum_loss = 0f32;
        for c in 0..C {
            sum_loss += (total_sum_loss + losses_diffs[c]) * counts[c] as f32;
        }
        sum_loss
    }
}

/// Normal distribution centered around 0
pub trait Noise {
    fn noise<R: RngCore>(rng: &mut R, sdev: f32) -> Self;
}

impl Noise for f32 {
    fn noise<R: RngCore>(rng: &mut R, sdev: f32) -> f32 {
        let normal = Normal::new(0f32, sdev).unwrap();
        normal.sample(rng)
    }
}

impl<T: Noise, const L: usize> Noise for [T; L]
where
    [T; L]: Default,
{
    fn noise<RNG: RngCore>(rng: &mut RNG, sdev: f32) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::noise(rng, sdev);
        }
        target
    }
}

/// Returns a version of self mutated by a slice of noise
pub trait Mutate {
    const NOISE_LEN: usize;
    fn mutate(&self, noise: &[f32]) -> Self;
}

impl Mutate for f32 {
    const NOISE_LEN: usize = 1;
    fn mutate(&self, noise: &[f32]) -> Self {
        self + noise[0]
    }
}

impl<A: Mutate, B: Mutate> Mutate for (A, B) {
    const NOISE_LEN: usize = A::NOISE_LEN + B::NOISE_LEN;
    fn mutate(&self, noise: &[f32]) -> Self {
        (
            self.0.mutate(&noise[0..]),
            self.1.mutate(&noise[A::NOISE_LEN..]),
        )
    }
}

impl<T: Mutate, const L: usize> Mutate for [T; L]
where
    [T; L]: Default,
{
    const NOISE_LEN: usize = T::NOISE_LEN * L;
    fn mutate(&self, noise: &[f32]) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = self[i].mutate(&noise[i * T::NOISE_LEN..(i + 1) * T::NOISE_LEN]);
        }
        target
    }
}
