#![feature(const_generics)]
use bitnn::count::ElementwiseAdd;
use bitnn::float::Noise;
use bitnn::shape::{Element, Fold, IndexGet, IndexMap, Map, MapMut, Shape, ZipFold, ZipMap};
use rand::SeedableRng;
use rand_hc::Hc128Rng;

trait Clip {
    fn clip(&self, x: f64) -> Self;
}

impl Clip for f64 {
    fn clip(&self, x: f64) -> f64 {
        self.max(-x).min(x)
    }
}

impl<T: Clip, const L: usize> Clip for [T; L]
where
    [T; L]: Default,
{
    fn clip(&self, x: f64) -> [T; L] {
        let mut target = <[T; L]>::default();
        for i in 0..L {
            target[i] = self[i].clip(x);
        }
        target
    }
}

impl<A: Clip, B: Clip> Clip for (A, B) {
    fn clip(&self, x: f64) -> (A, B) {
        (self.0.clip(x), self.1.clip(x))
    }
}

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

trait FakeTanh {
    fn fake_tanh_forward(&self) -> Self;
    fn fake_tanh_grad(&self, grad: &Self) -> Self;
}

impl FakeTanh for f64 {
    fn fake_tanh_forward(&self) -> f64 {
        self.max(-3.0).min(3.0)
    }
    fn fake_tanh_grad(&self, grad: &Self) -> f64 {
        grad * (self.abs() < 3.0) as u8 as f64
    }
}

impl<T: FakeTanh, const L: usize> FakeTanh for [T; L]
where
    [T; L]: Default,
{
    fn fake_tanh_forward(&self) -> [T; L] {
        let mut output = <[T; L]>::default();
        for i in 0..L {
            output[i] = self[i].fake_tanh_forward();
        }
        output
    }
    fn fake_tanh_grad(&self, grad: &[T; L]) -> [T; L] {
        let mut output = <[T; L]>::default();
        for i in 0..L {
            output[i] = self[i].fake_tanh_grad(&grad[i]);
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
            + Map<f64, f64>
            + Fold<f64, f64>
            + MapMut<f64, f64>
            + IndexMap<f64, ()>
            + Element<(), Array = I>
            + ZipFold<f64, f64, f64>,
        O: Shape
            + Map<<f64 as Element<I>>::Array, f64>
            + MapMut<f64, <f64 as Element<I>>::Array>
            + ZipMap<<f64 as Element<I>>::Array, f64, f64>
            + ZipFold<<f64 as Element<I>>::Array, <f64 as Element<I>>::Array, f64>,
    > VecMul<I, O> for <<f64 as Element<I>>::Array as Element<O>>::Array
where
    f64: Element<I> + Element<O>,
    <f64 as Element<O>>::Array: Default,
    <f64 as Element<I>>::Array: Default + Element<O> + IndexGet<<I as Shape>::Index, Element = f64>,
    <<f64 as Element<I>>::Array as Element<O>>::Array: Default,
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
                <I as MapMut<f64, f64>>::map_mut(&mut grads, input, |grad, input_val| {
                    *grad += (target / input_val).min(0.01).max(-0.01);
                });
            },
        );
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
            (input_delta / (diff + 0.00000000001)) * target_update
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
            let diff = base_loss - new_loss;
            grads[c] = (delta / (diff + 0.0000000001)) * target_update;
            //if grads[c].is_infinite() {
            //    dbg!(base_loss);
            //    dbg!(new_loss);
            //    panic!();
            //}
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
    <f64 as Element<O>>::Array: Default + FakeTanh + Add + ElementwiseAdd + Copy + std::fmt::Debug,
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
            tanh: model.1.add(&h).fake_tanh_forward(),
        }
    }
    fn output(&self) -> <f64 as Element<O>>::Array {
        self.tanh
    }
    fn backward(
        self,
        w_grads: &mut (
            <<f64 as Element<I>>::Array as Element<O>>::Array,
            <f64 as Element<O>>::Array,
        ),
        grads: &<f64 as Element<O>>::Array,
    ) {
        let tanh_grads = self.tanh.fake_tanh_grad(&grads);

        w_grads.1.elementwise_add(&tanh_grads);
        <<<f64 as Element<I>>::Array as Element<O>>::Array>::weight_grads(
            &self.input,
            &tanh_grads,
            &mut w_grads.0,
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
    [f64; O]: Default + std::fmt::Debug,
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
        update: f64,
    ) -> [f64; O] {
        const delta: f64 = 0.00000000001;
        assert!(!update.is_nan());
        //dbg!(&self.hb);
        let sm_grads = self.hb.sm_mse_grads(delta, self.class, update * self.loss);
        //dbg!(&sm_grads);
        //if sm_grads.iter().find(|x| x.is_nan()).is_some() {
        //    panic!();
        //}
        grads.1.elementwise_add(&sm_grads);
        <[[f64; O]; C]>::weight_grads(&self.input, &sm_grads, &mut grads.0);

        let input_grads = model.0.input_grads_delta(
            &self.h,
            &self.input,
            delta,
            self.loss,
            |new_output| new_output.add(&model.1).sm_mse_loss(self.class),
            update * self.loss,
        );
        //dbg!(&input_grads);
        if input_grads.iter().find(|x| x.is_infinite()).is_some() {
            panic!();
        }
        input_grads
    }
}

type HiddenShape = [(); 10];

type PatchWeightsType = (
    <[f64; 7] as Element<HiddenShape>>::Array,
    <f64 as Element<HiddenShape>>::Array,
);
type ObjectiveWeightsType = ([<f64 as Element<HiddenShape>>::Array; 3], [f64; 3]);

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(0);
    /*
    let base_patch_model = PatchWeightsType::noise(&mut rng, 0.2);
    let obj_model = ObjectiveWeightsType::noise(&mut rng, 0.2);

    let patch_input = <[f64; 7]>::noise(&mut rng, 0.5);
    let class = 1;

    let update = 0.00000001f64;

    for &base_input in &[-5.0, -3.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 3.0, 5.0] {
        let mut input: f64 = base_input;
        let t1 = input.fake_tanh_forward();
        let grad = input.fake_tanh_grad(&update);
        input += grad;
        let t2 = input.fake_tanh_forward();
        dbg!((t2 - t1) / update);
    }

    for i in 0..7 {
        for o in 0..5 {
            let mut patch_model = base_patch_model;
            let mut p_grads = PatchWeightsType::default();
            let mut o_grads = ObjectiveWeightsType::default();
            let patch_cache =
                PatchCache::<[(); 7], HiddenShape>::forward(patch_input, &patch_model);
            let obj_cache = ObjCache::<5, 3>::forward(patch_cache.output(), &obj_model, class);
            let loss_a = obj_cache.loss();
            let tanh_grads = obj_cache.backward(&obj_model, &mut o_grads, update);
            patch_cache.backward(&mut p_grads, &tanh_grads);
            patch_model.0[o][i] += p_grads.0[o][i];

            let patch_cache =
                PatchCache::<[(); 7], HiddenShape>::forward(patch_input, &patch_model);
            let obj_cache = ObjCache::<5, 3>::forward(patch_cache.output(), &obj_model, class);
            let loss_b = obj_cache.loss();
            dbg!((loss_a - loss_b) / update);
        }
    }

    */
    let examples = vec![
        ([1.1f64, 3.8, 3.1, 3.8, 0.1, 0.2, -0.7], 0),
        ([1.3f64, 3.2, 2.8, 3.8, -0.7, -0.2, 0.7], 0),
        ([-1.3f64, 3.8, -3.8, 3.8, 1.7, 0.2, -0.7], 1),
        ([1.3f64, 3.8, 3.8, -3.8, -1.7, -0.2, -0.7], 1),
        ([-1.3f64, 3.8, 3.8, 3.8, -1.7, 0.2, 0.7], 2),
        ([1.3f64, -3.8, -3.0, -3.8, 1.7, 0.2, -0.3], 2),
    ];

    let mut model = <(PatchWeightsType, ObjectiveWeightsType)>::noise(&mut rng, 0.01);
    for i in 0..100000 {
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

        //let losses: Vec<f64> = caches.iter().map(|(_, o)| o.loss()).collect();
        //dbg!(losses);

        if loss.is_nan() {
            dbg!(&model);
            panic!();
        }

        if loss > 1.0 {
            let losses: Vec<f64> = caches.iter().map(|(_, o)| o.loss()).collect();
            dbg!(losses);
            panic!();
        }

        let (n, sum_grads) = caches.drain(0..).fold(
            <(usize, (PatchWeightsType, ObjectiveWeightsType))>::default(),
            |mut grads, (pc, oc)| {
                grads.0 += 1;
                let tanh_grads = oc.backward(&model.1, &mut (grads.1).1, 0.00001);
                pc.backward(&mut (grads.1).0, &tanh_grads);
                grads
            },
        );
        let avg_grads = sum_grads.divide(n as f64);
        //dbg!(&(avg_grads.0).1);
        if (avg_grads.0).1.iter().find(|x| x.is_nan()).is_some() {
            panic!();
        }
        //dbg!((avg_grads.0).1);
        model.elementwise_add(&avg_grads);
        //dbg!((model.0).1);
    }
    dbg!(model);
}
