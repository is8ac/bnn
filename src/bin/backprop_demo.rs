#![feature(const_generics)]
use rand::distributions;
use rand::{Rng, SeedableRng};
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};
use rand_hc::Hc128Rng;

/// Normal distribution centered around 0
pub trait Noise {
    fn noise<R: RngCore>(rng: &mut R, sdev: f64) -> Self;
}

impl Noise for f64 {
    fn noise<R: RngCore>(rng: &mut R, sdev: f64) -> f64 {
        let normal = Normal::new(0f64, sdev).unwrap();
        normal.sample(rng)
    }
}

impl<T: Noise, const L: usize> Noise for [T; L]
where
    [T; L]: Default,
{
    fn noise<RNG: RngCore>(rng: &mut RNG, sdev: f64) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = T::noise(rng, sdev);
        }
        target
    }
}

trait Tanh<const L: usize> {
    fn tanh(&self) -> [f64; L];
    fn tanh_grad(&self, grad: &[f64; L]) -> [f64; L];
}

impl<const L: usize> Tanh<L> for [f64; L]
where
    [f64; L]: Default,
{
    fn tanh(&self) -> [f64; L] {
        let mut output = <[f64; L]>::default();
        for i in 0..L {
            output[i] = self[i].tanh();
        }
        output
    }
    fn tanh_grad(&self, grad: &[f64; L]) -> [f64; L] {
        let mut output = <[f64; L]>::default();
        for i in 0..L {
            output[i] = (1f64 - self[i].powi(2)) * grad[i];
        }
        output
    }
}

fn tanh_grad(input: f64, grad: f64) -> f64 {
    (1f64 - input.powi(2)) * grad
}

trait VecMul<const I: usize, const O: usize> {
    fn vec_mul(&self, input: &[f64; I]) -> [f64; O];
    fn weight_grads(&self, input: &[f64; I], grads: &[f64; O]) -> Self;
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
    fn weight_grads(&self, input: &[f64; I], target_grads: &[f64; O]) -> Self {
        let mut grads = <[[f64; I]; O]>::default();
        for o in 0..O {
            for i in 0..I {
                grads[o][i] = target_grads[o] / input[i];
            }
        }
        grads
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

fn softmax(input: &[f64; 3]) -> [f64; 3] {
    let mut exp = <[f64; 3]>::default();
    let mut sum_exp = 0f64;
    for c in 0..3 {
        exp[c] = input[c].exp();
        sum_exp += exp[c];
    }
    let mut output = [0f64; 3];
    for c in 0..3 {
        output[c] = exp[c] / sum_exp;
    }
    output
}

fn softmax_grads(target_grads: &[f64; 3], output: &[f64; 3]) -> [f64; 3] {
    let mut sum = 0f64;
    for i in 0..3 {
        sum += target_grads[i] * output[i];
    }
    let mut grads = [0f64; 3];
    for i in 0..3 {
        grads[i] = (target_grads[i] - sum) * output[i];
    }
    grads
}

fn squared_diff_loss(input: &[f64; 3], class: usize) -> f64 {
    let mut sum_loss = 0f64;
    for i in 0..3 {
        let hot = (i == class) as u8 as f64;
        sum_loss += (input[i] - hot).powi(2);
    }
    sum_loss
}

fn squared_diff_loss_grads(input: &[f64; 3], class: usize, loss: f64) -> [f64; 3] {
    let mut grads = [0f64; 3];
    for i in 0..3 {
        let hot = (i == class) as u8 as f64;
        grads[i] = loss / ((input[i] - hot) * 2f64);
    }
    grads
}

struct ExampleCache {
    input: [f64; 5],
    h1: [f64; 3],
    h1b: [f64; 3],
    sm: [f64; 3],
    class: usize,
}

impl ExampleCache {
    fn foreword(&mut self, m1: &[[f64; 5]; 3], b1: &[f64; 3]) -> f64 {
        self.h1 = m1.vec_mul(&self.input);
        //self.h1b = b1.add(&self.h1);
        self.sm = softmax(&self.h1);
        squared_diff_loss(&self.sm, self.class)
    }
    fn backword(&self, loss: f32) {}
}

#[derive(Default)]
struct Model {
    l1: [[f64; 5]; 3],
    l2: [[f64; 3]; 2],
}

impl Model {
    fn acts(&self, input: &[f64; 5]) -> [f64; 2] {
        let mut hidden = [0f64; 3];
        for o in 0..3 {
            let mut act = 0f64;
            for i in 0..5 {
                act += input[i] * self.l1[o][i];
            }
            hidden[o] = act.tanh();
        }
        let mut acts = [0f64; 2];
        for a in 0..2 {
            let mut act = 0f64;
            for h in 0..3 {
                act += hidden[h] * self.l2[a][h];
            }
            acts[a] = act;
        }
        acts
    }
    fn rand_init() -> Self {
        let mut rng = Hc128Rng::seed_from_u64(0);
        Model {
            l1: <[[f64; 5]; 3]>::noise(&mut rng, 0.1),
            l2: <[[f64; 3]; 2]>::noise(&mut rng, 0.1),
        }
    }
}

fn main() {
    let input = [0.98432f64, -0.4321, -0.284, 0.43287, 0.0];
    //dbg!(input);
    //let act: Vec<_> = input.iter().map(|x| x.tanh()).collect();
    //dbg!(act);
    //let grads: Vec<_> = input.iter().map(|&x| tanh_grad(x)).collect();
    //dbg!(grads);

    let mut rng = Hc128Rng::seed_from_u64(0);
    let layer = <[[f64; 5]; 3]>::noise(&mut rng, 0.1f64);
    let weight_grad = {
        let delta = 0.1234567;
        let mut layer_p = layer;
        let o1 = layer_p.vec_mul(&input);
        layer_p[1][1] += delta;
        let o2 = layer_p.vec_mul(&input);
        let mut grad = [0f64; 3];
        for i in 0..3 {
            grad[i] = o2[i] - o1[i];
        }
        grad
    };
    //dbg!(weight_grad);
    let weight_grads = layer.weight_grads(&input, &weight_grad);
    dbg!(weight_grads[1][1]);

    let grad = {
        let delta = 0.1234567;
        let mut input = input;
        let o1 = layer.vec_mul(&input);
        input[1] += delta;
        let o2 = layer.vec_mul(&input);
        let mut grad = [0f64; 3];
        for i in 0..3 {
            grad[i] = o2[i] - o1[i];
        }
        grad
    };
    //dbg!(grad);
    let input_grads = layer.input_grads(&grad);
    dbg!(input_grads[1]);

    let input = 0.0526127f64;
    let o1 = input.tanh();
    dbg!(o1);
    let o2 = (input + 0.0000001234).tanh();
    let o_grad = o2 - o1;
    dbg!(o_grad);
    let c_grad = tanh_grad(input, o_grad);
    dbg!(c_grad);
    let sm_output = softmax(&[-0.4, 2.1, 0.13432]);
    dbg!(sm_output);
    let sm_grads = softmax_grads(&[0.0, 0.1, 0.0], &sm_output);
    dbg!(sm_grads);

    let input1 = softmax(&[-0.04, 0.1, 0.13432]);
    let input2 = {
        let mut input = input1;
        input[1] += 0.000000000123456789;
        input
    };
    let l1 = squared_diff_loss(&input1, 1);
    let l2 = squared_diff_loss(&input2, 1);
    dbg!(l1);
    dbg!(l2);
    let grad = l2 - l1;
    dbg!(grad);
    let input_grads = squared_diff_loss_grads(&input1, 1, grad);
    dbg!(input_grads[1]);
}
