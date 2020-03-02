#![feature(const_generics)]
use bitnn::count::ElementwiseAdd;
use bitnn::float::Noise;
use bitnn::shape::Map;
use rand::distributions;
use rand::{Rng, SeedableRng};
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};
use rand_hc::Hc128Rng;

trait Divide {
    fn divide(&self, n: f64) -> Self;
}

impl Divide for f64 {
    fn divide(&self, n: f64) -> Self {
        *self / n
    }
}

impl<T: Divide, const L: usize> Divide for [T; L]
where
    [T; L]: Default,
{
    fn divide(&self, n: f64) -> Self {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = self[i].divide(n);
        }
        target
    }
}

impl<A: Divide, B: Divide> Divide for (A, B) {
    fn divide(&self, n: f64) -> Self {
        (self.0.divide(n), self.1.divide(n))
    }
}

trait Add {
    fn add(&self, other: &Self) -> Self;
}

impl Add for f64 {
    fn add(&self, other: &Self) -> Self {
        self + other
    }
}

impl<T: Add, const L: usize> Add for [T; L]
where
    [T; L]: Default,
{
    fn add(&self, other: &[T; L]) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = self[i].add(&other[i]);
        }
        target
    }
}

trait Tanh {
    fn tanh_forward(&self) -> Self;
    fn tanh_grad(&self, grad: &Self) -> Self;
}

impl Tanh for f64 {
    fn tanh_forward(&self) -> f64 {
        self.tanh()
    }
    fn tanh_grad(&self, grad: &Self) -> f64 {
        (1f64 - self.powi(2)) * grad
    }
}

impl<T: Tanh, const L: usize> Tanh for [T; L]
where
    [T; L]: Default,
{
    fn tanh_forward(&self) -> [T; L] {
        let mut output = <[T; L]>::default();
        for i in 0..L {
            output[i] = self[i].tanh_forward();
        }
        output
    }
    fn tanh_grad(&self, grad: &[T; L]) -> [T; L] {
        let mut output = <[T; L]>::default();
        for i in 0..L {
            output[i] = self[i].tanh_grad(&grad[i]);
        }
        output
    }
}

trait VecMul<const I: usize, const O: usize> {
    fn vec_mul(&self, input: &[f64; I]) -> [f64; O];
    fn weight_grads(&self, input: &[f64; I], grads: &[f64; O], weight_grads: &mut Self);
    fn input_grads(&self, grads: &[f64; O]) -> [f64; I];
}

impl<const I: usize, const O: usize> VecMul<I, O> for [[f64; I]; O]
where
    [f64; O]: Default,
    [[f64; I]; O]: Default,
    [f64; I]: Default,
{
    fn vec_mul(&self, input: &[f64; I]) -> [f64; O] {
        let mut target = <[f64; O]>::default();
        for o in 0..O {
            let mut act = 0f64;
            for i in 0..I {
                act += input[i] * self[o][i];
            }
            target[o] = act;
        }
        target
    }
    fn weight_grads(&self, input: &[f64; I], target_grads: &[f64; O], grads: &mut Self) {
        for o in 0..O {
            for i in 0..I {
                grads[o][i] += target_grads[o] / input[i];
            }
        }
    }
    fn input_grads(&self, target_grads: &[f64; O]) -> [f64; I] {
        let mut grads = <[f64; I]>::default();
        for i in 0..I {
            let mut grad = 0f64;
            for o in 0..O {
                grad += target_grads[o] / self[o][i];
            }
            grads[i] = grad / O as f64;
        }
        grads
    }
}

trait SquaredErrorLoss {
    fn mse(&self, class: usize) -> f64;
    fn mse_grads(&self, class: usize, loss: f64) -> Self;
}

impl<const L: usize> SquaredErrorLoss for [f64; L]
where
    [f64; L]: Default,
{
    fn mse(&self, class: usize) -> f64 {
        let mut sum_loss = 0f64;
        for i in 0..L {
            let hot = (i == class) as u8 as f64;
            sum_loss += (self[i] - hot).powi(2);
        }
        sum_loss / L as f64
    }
    fn mse_grads(&self, class: usize, loss: f64) -> [f64; L] {
        let mut grads = <[f64; L]>::default();
        let sum_loss = loss * L as f64;
        for i in 0..L {
            let hot = (i == class) as u8 as f64;
            grads[i] = (sum_loss / ((self[i] - hot) * 2f64));
        }
        grads
    }
}

trait SMMSEloss {
    fn sm_mse_loss(&self, class: usize) -> f64;
    fn sm_mse_grads(&self, delta: f64, class: usize, avg_loss: f64) -> Self;
}

impl<const L: usize> SMMSEloss for [f64; L]
where
    [f64; L]: Default,
{
    fn sm_mse_loss(&self, class: usize) -> f64 {
        let mut exp = <[f64; L]>::default();
        let mut sum_exp = 0f64;
        for c in 0..L {
            exp[c] = self[c].exp();
            sum_exp += exp[c];
        }
        let mut sum_loss = 0f64;
        for c in 0..L {
            let x = exp[c] / sum_exp;
            let hot = (c == class) as u8 as f64;
            sum_loss += (x - hot).powi(2);
        }
        sum_loss
    }
    fn sm_mse_grads(&self, delta: f64, class: usize, avg_loss: f64) -> [f64; L] {
        let base_loss = self.sm_mse_loss(class);
        let mut grads = <[f64; L]>::default();
        for c in 0..L {
            let mut new_input = *self;
            new_input[c] += delta;
            let new_loss = new_input.sm_mse_loss(class);
            grads[c] = ((base_loss - new_loss) / delta) * avg_loss;
        }
        grads
    }
}

struct PatchCache<const I: usize, const O: usize> {
    input: [f64; I],
    h: [f64; O],
    hb: [f64; O],
    tanh: [f64; O],
}

impl<const I: usize, const O: usize> PatchCache<I, O>
where
    [[f64; I]; O]: VecMul<I, O>,
    [f64; I]: Default,
    [f64; O]: Default + Tanh,
{
    fn forward(input: [f64; I], model: &([[f64; I]; O], [f64; O])) -> Self {
        let h = model.0.vec_mul(&input);
        let hb = model.1.add(&h);
        PatchCache {
            input: input,
            h: h,
            hb: hb,
            tanh: hb.tanh_forward(),
        }
    }
    fn output(&self) -> [f64; O] {
        self.tanh
    }
    fn backward(
        self,
        model: &([[f64; I]; O], [f64; O]),
        grads: &mut ([[f64; I]; O], [f64; O]),
        tanh_grads: &[f64; O],
    ) {
        let tanh_grads = self.tanh.tanh_grad(&tanh_grads);
        grads.1.elementwise_add(&tanh_grads);
        model.0.weight_grads(&self.input, &tanh_grads, &mut grads.0);
    }
}

struct ObjCache<const O: usize, const C: usize> {
    input: [f64; O],
    h: [f64; C],
    hb: [f64; C],
    class: usize,
}

impl<const O: usize, const C: usize> ObjCache<O, C>
where
    [[f64; O]; C]: VecMul<O, C>,
    [f64; O]: Default,
    [f64; C]: Default + SquaredErrorLoss + std::fmt::Debug,
{
    fn forward(input: [f64; O], model: &([[f64; O]; C], [f64; C]), class: usize) -> Self {
        let h = model.0.vec_mul(&input);
        let hb = model.1.add(&h);
        ObjCache {
            input: input,
            h: h,
            hb: hb,
            class: class,
        }
    }
    fn loss(&self) -> f64 {
        self.hb.sm_mse_loss(self.class)
    }
    fn backward(
        self,
        model: &([[f64; O]; C], [f64; C]),
        grads: &mut ([[f64; O]; C], [f64; C]),
        loss: f64,
    ) -> [f64; O] {
        assert!(!loss.is_nan());
        let sm_grads = self.hb.sm_mse_grads(0.00001, self.class, loss);
        grads.1.elementwise_add(&sm_grads);
        model.0.weight_grads(&self.input, &sm_grads, &mut grads.0);
        model.0.input_grads(&sm_grads)
    }
}

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);

    let examples = vec![
        ([-1.3f64, 3.8, 3.8, 3.8, 1.7, 0.2, -0.7], 0),
        ([-1.3f64, 3.8, -3.8, 3.8, 1.7, 0.2, 0.7], 1),
        ([-1.3f64, 3.8, 3.8, 3.8, -1.7, 0.2, 0.7], 2),
        ([1.3f64, -3.8, -3.0, -3.8, 1.7, -0.2, -0.3], 2),
    ];

    let mut model = <(([[f64; 7]; 5], [f64; 5]), ([[f64; 5]; 3], [f64; 3]))>::noise(&mut rng, 0.1);
    for i in 0..10000 {
        let mut caches: Vec<_> = examples
            .iter()
            .map(|&(input, class)| {
                let patch_cache = PatchCache::forward(input, &model.0);
                let obj_cache = ObjCache::forward(patch_cache.output(), &model.1, class);
                (patch_cache, obj_cache)
            })
            .collect();
        let sum_loss: f64 = caches.iter().map(|(_, o)| o.loss()).sum();
        let loss = sum_loss / caches.len() as f64;
        dbg!(loss);
        let (n, sum_grads) = caches.drain(0..).fold(
            <(
                usize,
                (([[f64; 7]; 5], [f64; 5]), ([[f64; 5]; 3], [f64; 3])),
            )>::default(),
            |mut grads, (pc, oc)| {
                grads.0 += 1;
                let tanh_grads = oc.backward(&model.1, &mut (grads.1).1, loss * 0.01);
                pc.backward(&model.0, &mut (grads.1).0, &tanh_grads);
                grads
            },
        );
        let avg_grads = sum_grads.divide(n as f64);
        //dbg!(avg_grads);
        model.elementwise_add(&avg_grads);
    }
}
