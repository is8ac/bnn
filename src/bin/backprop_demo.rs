#![feature(const_generics)]
use bitnn::count::ElementwiseAdd;
use bitnn::float::Noise;
use bitnn::shape::{Element, Fold, IndexGet, IndexMap, Map, MapMut, Shape, ZipFold, ZipMap};
use rand::SeedableRng;
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

trait VecMul<I: Shape, O: Shape>
where
    f64: Element<I> + Element<O>,
{
    fn vec_mul(&self, input: &<f64 as Element<I>>::Array) -> <f64 as Element<O>>::Array;
    fn weight_grads(
        input: &<f64 as Element<I>>::Array,
        grads: &<f64 as Element<O>>::Array,
        weight_grads: &mut Self,
    );
    fn input_grads(&self, grads: &<f64 as Element<O>>::Array) -> <f64 as Element<I>>::Array;
    fn input_grads_delta<F: Fn(&<f64 as Element<O>>::Array) -> f64>(
        &self,
        output: &<f64 as Element<O>>::Array,
        input: &<f64 as Element<I>>::Array,
        input_delta: f64,
        null_loss: f64,
        loss_fn: F,
        target_update: f64,
    ) -> <f64 as Element<I>>::Array;
}

impl<
        I: Shape
            + Fold<f64, f64>
            + MapMut<f64, f64>
            + Map<f64, f64>
            + ZipFold<f64, f64, f64>
            + IndexMap<f64, ()>
            + Element<(), Array = I>,
        O: Shape
            + MapMut<f64, <f64 as Element<I>>::Array>
            + ZipMap<<f64 as Element<I>>::Array, f64, f64>
            + Map<<f64 as Element<I>>::Array, f64>
            + ZipFold<<f64 as Element<I>>::Array, <f64 as Element<I>>::Array, f64>,
    > VecMul<I, O> for <<f64 as Element<I>>::Array as Element<O>>::Array
where
    <f64 as Element<O>>::Array: Default,
    <<f64 as Element<I>>::Array as Element<O>>::Array: Default,
    <f64 as Element<I>>::Array: Default + Element<O> + IndexGet<<I as Shape>::Index, Element = f64>,
    f64: Element<I> + Element<O>,
{
    fn vec_mul(&self, input: &<f64 as Element<I>>::Array) -> <f64 as Element<O>>::Array {
        <O as Map<<f64 as Element<I>>::Array, f64>>::map(self, |row| {
            <I as ZipFold<f64, f64, f64>>::zip_fold(input, &row, 0f64, |sum, i, w| sum + i * w)
        })
    }
    fn weight_grads(
        input: &<f64 as Element<I>>::Array,
        target_grads: &<f64 as Element<O>>::Array,
        grads: &mut Self,
    ) {
        <O as MapMut<f64, <f64 as Element<I>>::Array>>::map_mut(
            grads,
            target_grads,
            |mut grads, target| {
                <I as MapMut<f64, f64>>::map_mut(&mut grads, input, |grad, x| {
                    *grad += target / x;
                });
            },
        );
    }
    fn input_grads(&self, target_grads: &<f64 as Element<O>>::Array) -> <f64 as Element<I>>::Array {
        let grads_sums =
            <O as ZipFold<<f64 as Element<I>>::Array, <f64 as Element<I>>::Array, f64>>::zip_fold(
                self,
                target_grads,
                <<f64 as Element<I>>::Array>::default(),
                |mut sums, row, target_grad| {
                    <I as MapMut<f64, f64>>::map_mut(&mut sums, &row, |sum, weight| {
                        *sum += target_grad / weight
                    });
                    sums
                },
            );
        <I as Map<f64, f64>>::map(&grads_sums, |x| x / O::N as f64)
    }
    fn input_grads_delta<F: Fn(&<f64 as Element<O>>::Array) -> f64>(
        &self,
        output: &<f64 as Element<O>>::Array,
        input: &<f64 as Element<I>>::Array,
        input_delta: f64,
        null_loss: f64,
        loss_fn: F,
        target_update: f64,
    ) -> <f64 as Element<I>>::Array {
        <I as IndexMap<f64, ()>>::index_map((), |i| {
            let input_val = input.index_get(i);
            let new_output = <O as ZipMap<<f64 as Element<I>>::Array, f64, f64>>::zip_map(
                self,
                output,
                |w, o| o - w.index_get(i) * input_val + w.index_get(i) * (input_val + input_delta),
            );
            let diff = null_loss - loss_fn(&new_output);
            dbg!(diff);
            (input_delta / (null_loss - loss_fn(&new_output))) * target_update
        })
    }
}

trait SMMSEloss {
    fn sm_mse_loss(&self, class: usize) -> f64;
    fn sm_mse_grads(&self, delta: f64, class: usize, target_update: f64) -> Self;
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
    fn sm_mse_grads(&self, delta: f64, class: usize, target_update: f64) -> [f64; L] {
        let base_loss = self.sm_mse_loss(class);
        let mut grads = <[f64; L]>::default();
        for c in 0..L {
            let mut new_input = *self;
            new_input[c] += delta;
            let new_loss = new_input.sm_mse_loss(class);
            grads[c] = (delta / (base_loss - new_loss)) * target_update;
        }
        grads
    }
}

struct PatchCache<I: Shape, O: Shape>
where
    f64: Element<I> + Element<O>,
{
    input: <f64 as Element<I>>::Array,
    tanh: <f64 as Element<O>>::Array,
}

impl<I: Shape, O: Shape> PatchCache<I, O>
where
    <<f64 as Element<I>>::Array as Element<O>>::Array: VecMul<I, O>,
    <f64 as Element<I>>::Array: Default + Element<O>,
    <f64 as Element<O>>::Array: Default + Tanh + Add + ElementwiseAdd + Copy,
    f64: Element<I> + Element<O>,
{
    fn forward(
        input: <f64 as Element<I>>::Array,
        model: &(
            <<f64 as Element<I>>::Array as Element<O>>::Array,
            <f64 as Element<O>>::Array,
        ),
    ) -> Self {
        let h = model.0.vec_mul(&input);
        PatchCache {
            input: input,
            tanh: model.1.add(&h).tanh_forward(),
        }
    }
    fn output(&self) -> <f64 as Element<O>>::Array {
        self.tanh
    }
    fn backward(
        self,
        grads: &mut (
            <<f64 as Element<I>>::Array as Element<O>>::Array,
            <f64 as Element<O>>::Array,
        ),
        tanh_grads: &<f64 as Element<O>>::Array,
    ) {
        let tanh_grads = self.tanh.tanh_grad(&tanh_grads);
        grads.1.elementwise_add(&tanh_grads);
        <<<f64 as Element<I>>::Array as Element<O>>::Array>::weight_grads(
            &self.input,
            &tanh_grads,
            &mut grads.0,
        );
    }
}

struct ObjCache<const O: usize, const C: usize> {
    input: [f64; O],
    h: [f64; C],
    hb: [f64; C],
    loss: f64,
    class: usize,
}

impl<const O: usize, const C: usize> ObjCache<O, C>
where
    [[f64; O]; C]: VecMul<[(); O], [(); C]>,
    [f64; O]: Default,
    [f64; C]: Default + std::fmt::Debug,
{
    fn forward(input: [f64; O], model: &([[f64; O]; C], [f64; C]), class: usize) -> Self {
        let h = model.0.vec_mul(&input);
        let hb = model.1.add(&h);
        ObjCache {
            input: input,
            h: h,
            hb: hb,
            loss: hb.sm_mse_loss(class),
            class: class,
        }
    }
    fn loss(&self) -> f64 {
        self.loss
    }
    fn backward(
        self,
        model: &([[f64; O]; C], [f64; C]),
        grads: &mut ([[f64; O]; C], [f64; C]),
        loss: f64,
    ) -> [f64; O] {
        assert!(!loss.is_nan());
        let sm_grads = self.hb.sm_mse_grads(0.0000001, self.class, loss);
        grads.1.elementwise_add(&sm_grads);
        <[[f64; O]; C]>::weight_grads(&self.input, &sm_grads, &mut grads.0);
        //model.0.input_grads(&sm_grads)
        model.0.input_grads_delta(
            &self.h,
            &self.input,
            0.0000001,
            self.loss,
            |new_output| new_output.add(&model.1).sm_mse_loss(self.class),
            loss,
        )
    }
}

type HiddenShape = [(); 5];

type PatchWeightsType = (
    <[f64; 7] as Element<HiddenShape>>::Array,
    <f64 as Element<HiddenShape>>::Array,
);
type ObjectiveWeightsType = ([<f64 as Element<HiddenShape>>::Array; 3], [f64; 3]);

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    let model = ObjectiveWeightsType::noise(&mut rng, 0.1);

    let old_input = <[f64; 5]>::noise(&mut rng, 1.0);
    let class = 3;

    let delta = 0.000000000000123456789;
    for i in 0..5 {
        let mut input = old_input;
        let mut tmp_model = model;
        let mut w_grads = ObjectiveWeightsType::default();
        let obj_cache = ObjCache::forward(input, &model, class);
        let loss_a = obj_cache.loss();
        dbg!(loss_a);
        let tanh_grads = obj_cache.backward(&model, &mut w_grads, delta);
        dbg!(tanh_grads);

        input[i] += tanh_grads[i];
        let obj_cache = ObjCache::forward(input, &model, class);
        let loss_b = obj_cache.loss();
        dbg!(loss_b);
        dbg!((loss_a - loss_b) / delta);
    }
    /*

    let examples = vec![
        ([1.1f64, 3.8, 3.1, 3.8, 0.1, 0.2, -0.7], 0),
        //([1.3f64, 3.2, 2.8, 3.8, -0.7, -0.2, 0.7], 0),
        ([-1.3f64, 3.8, -3.8, 3.8, 1.7, 0.2, -0.7], 1),
        ([1.3f64, 3.8, 3.8, -3.8, -1.7, -0.2, -0.7], 1),
        ([-1.3f64, 3.8, 3.8, 3.8, -1.7, 0.2, 0.7], 2),
        ([1.3f64, -3.8, -3.0, -3.8, 1.7, 0.2, -0.3], 2),
    ];

    let mut model = <(PatchWeightsType, ObjectiveWeightsType)>::noise(&mut rng, 0.1);
    for i in 0..10000 {
        let mut caches: Vec<_> = examples
            .iter()
            .map(|&(input, class)| {
                let patch_cache = PatchCache::<[(); 7], HiddenShape>::forward(input, &model.0);
                let obj_cache = ObjCache::forward(patch_cache.output(), &model.1, class);
                (patch_cache, obj_cache)
            })
            .collect();
        let sum_loss: f64 = caches.iter().map(|(_, o)| o.loss()).sum();
        let loss = sum_loss / caches.len() as f64;
        dbg!(loss);
        let (n, sum_grads) = caches.drain(0..).fold(
            <(usize, (PatchWeightsType, ObjectiveWeightsType))>::default(),
            |mut grads, (pc, oc)| {
                grads.0 += 1;
                let tanh_grads = oc.backward(&model.1, &mut (grads.1).1, loss * 0.001);
                pc.backward(&mut (grads.1).0, &tanh_grads);
                grads
            },
        );
        let avg_grads = sum_grads.divide(n as f64);
        //dbg!(avg_grads);
        model.elementwise_add(&avg_grads);
    }
    */
}
