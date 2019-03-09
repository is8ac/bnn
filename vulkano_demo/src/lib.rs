extern crate rand;
extern crate rayon;
extern crate time;
extern crate vulkano;
extern crate vulkano_shaders;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;
use time::PreciseTime;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::{CpuAccessibleBuffer, DeviceLocalBuffer, ImmutableBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::descriptor::descriptor_set::{
    PersistentDescriptorSet, PersistentDescriptorSetBuf, StdDescriptorPoolAlloc,
};
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::{Device, Queue};
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::{Instance, QueueFamily};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

impl BitLen for u32 {
    const BIT_LEN: usize = 32;
}

macro_rules! array_bit_len {
    ($len:expr) => {
        impl<T: BitLen> BitLen for [T; $len] {
            const BIT_LEN: usize = $len * T::BIT_LEN;
        }
    };
}

array_bit_len!(2);
array_bit_len!(3);

pub trait FlipBit {
    fn flip_bit(&mut self, b: usize);
}

impl FlipBit for u32 {
    fn flip_bit(&mut self, index: usize) {
        *self ^= 1 << index
    }
}

macro_rules! array_flip_bit {
    ($len:expr) => {
        impl<T: BitLen + FlipBit> FlipBit for [T; $len] {
            fn flip_bit(&mut self, index: usize) {
                self[index / T::BIT_LEN].flip_bit(index % T::BIT_LEN);
            }
        }
    };
}
array_flip_bit!(2);
array_flip_bit!(3);

pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
}

impl HammingDistance for u32 {
    #[inline(always)]
    fn hamming_distance(&self, other: &u32) -> u32 {
        (self ^ other).count_ones()
    }
}

macro_rules! array_hamming_distance {
    ($len:expr) => {
        impl<T: HammingDistance> HammingDistance for [T; $len] {
            fn hamming_distance(&self, other: &[T; $len]) -> u32 {
                let mut distance = 0u32;
                for i in 0..$len {
                    distance += self[i].hamming_distance(&other[i]);
                }
                distance
            }
        }
    };
}

array_hamming_distance!(2);
array_hamming_distance!(3);

pub trait Apply<I, O> {
    fn apply(&self, input: &I) -> O;
}

impl<I: HammingDistance + BitLen> Apply<I, u32> for [I; 32] {
    fn apply(&self, input: &I) -> u32 {
        let mut target = 0u32;
        for i in 0..32 {
            target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2)) as u32) << i;
        }
        target
    }
}
pub trait IsCorrect<I> {
    fn is_correct(&self, target: u8, input: I) -> bool;
}
impl<I: HammingDistance> IsCorrect<I> for [I; 10] {
    // the max activation is the target.
    #[inline(always)]
    fn is_correct(&self, target: u8, input: I) -> bool {
        let max = self[target as usize].hamming_distance(&input);
        for i in 0..10 {
            if i != target as usize {
                if self[i].hamming_distance(&input) >= max {
                    return false;
                }
            }
        }
        true
    }
}

pub trait ObjectiveEvalCreator<InputPatch, Weights, Embedding> {
    type ObjectiveEvalType;
    fn new_obj_eval(
        &self,
        weights: &Weights,
        head: &[Embedding; 10],
        examples: &[(u8, InputPatch)],
    ) -> Self::ObjectiveEvalType;
}

pub trait ObjectiveEval<InputPatch, Weights, Embedding> {
    fn flip_weights_bit(&mut self, o: usize, i: usize);
    fn flip_head_bit(&mut self, o: usize, i: usize);
    fn obj(&mut self) -> u64;
}

pub struct TestCPUObjectiveEvalCreator<InputPatch, Weights, Embedding> {
    input_pixel_type: PhantomData<InputPatch>,
    weights_type: PhantomData<Weights>,
    embedding_type: PhantomData<Embedding>,
}

impl TestCPUObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> {
    pub fn new() -> Self {
        TestCPUObjectiveEvalCreator {
            input_pixel_type: PhantomData,
            weights_type: PhantomData,
            embedding_type: PhantomData,
        }
    }
}

impl ObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    for TestCPUObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
{
    type ObjectiveEvalType =
        TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>;
    fn new_obj_eval(
        &self,
        weights: &[[[[u32; 2]; 3]; 3]; 32],
        head: &[u32; 10],
        examples: &[(u8, [[[u32; 2]; 3]; 3])],
    ) -> TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> {
        TestCPUObjectiveEval {
            weights: *weights,
            head: *head,
            examples: examples.iter().cloned().collect(),
        }
    }
}

pub struct TestCPUObjectiveEval<InputPatch, Weights, Embedding> {
    weights: Weights,
    head: [Embedding; 10],
    examples: Vec<(u8, InputPatch)>,
}

// This is a slow implementation of obj() and should not be used if performance is desired.
impl ObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    for TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
{
    fn flip_weights_bit(&mut self, o: usize, i: usize) {
        self.weights[o].flip_bit(i);
    }
    fn flip_head_bit(&mut self, o: usize, i: usize) {
        self.head[o].flip_bit(i);
    }
    fn obj(&mut self) -> u64 {
        self.examples
            .par_iter()
            .map(|(class, patch)| {
                let embedding = self.weights.apply(patch);
                self.head.is_correct(*class, embedding) as u64
            })
            .sum()
    }
}

pub struct GlslLikeCPUObjectiveEvalCreator<InputPatch, Weights, Embedding> {
    input_pixel_type: PhantomData<InputPatch>,
    weights_type: PhantomData<Weights>,
    embedding_type: PhantomData<Embedding>,
}

impl GlslLikeCPUObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> {
    pub fn new() -> Self {
        GlslLikeCPUObjectiveEvalCreator {
            input_pixel_type: PhantomData,
            weights_type: PhantomData,
            embedding_type: PhantomData,
        }
    }
}

impl ObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    for GlslLikeCPUObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
{
    type ObjectiveEvalType =
        GlslLikeCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>;
    fn new_obj_eval(
        &self,
        weights: &[[[[u32; 2]; 3]; 3]; 32],
        head: &[u32; 10],
        examples: &[(u8, [[[u32; 2]; 3]; 3])],
    ) -> GlslLikeCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> {
        GlslLikeCPUObjectiveEval {
            weights: *weights,
            head: *head,
            examples: examples.iter().cloned().collect(),
        }
    }
}

pub struct GlslLikeCPUObjectiveEval<InputPatch, Weights, Embedding> {
    weights: Weights,
    head: [Embedding; 10],
    examples: Vec<(u8, InputPatch)>,
}

// This is an implementation of obj() that is almost identical to the glsl implementation.
impl ObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    for GlslLikeCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
{
    fn flip_weights_bit(&mut self, o: usize, i: usize) {
        self.weights[o].flip_bit(i);
    }
    fn flip_head_bit(&mut self, o: usize, i: usize) {
        self.head[o].flip_bit(i);
    }
    fn obj(&mut self) -> u64 {
        let threshold: u32 = ((32 * 2) * 9) / 2;
        self.examples
            .par_iter()
            .map(|(class, patch)| {
                let mut target = 0b0u32;
                for b in 0..32 {
                    let mut sum = 0;
                    for x in 0..3 {
                        for y in 0..3 {
                            for i in 0..2 {
                                sum += (patch[x][y][i] ^ self.weights[b][x][y][i]).count_ones();
                            }
                        }
                    }
                    target |= ((sum > threshold) as u32) << b;
                }
                let mut is_good = true;
                let max_obj = (self.head[*class as usize] ^ target).count_ones();
                for o in 0..10 {
                    is_good = (((self.head[o] ^ target).count_ones() < max_obj)
                        | (o == *class as usize))
                        & is_good;
                }
                is_good as u64
            })
            .sum()
    }
}

mod reduce_sum {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/reduce_sum.glsl",
    }
}

mod patch_apply {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/patch_apply_2-1.glsl",
    }
}

pub struct VulkanObjectiveEvalCreator<InputPatch, Weights, Embedding> {
    input_type: PhantomData<InputPatch>,
    weights_type: PhantomData<Weights>,
    embedding_type: PhantomData<Embedding>,
    instance: Arc<Instance>,
}

impl VulkanObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> {
    pub fn new() -> Self {
        let instance = Instance::new(None, &InstanceExtensions::none(), None)
            .expect("failed to create instance");

        Self {
            input_type: PhantomData,
            weights_type: PhantomData,
            embedding_type: PhantomData,
            instance: instance,
        }
    }
}

impl ObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    for VulkanObjectiveEvalCreator<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
{
    type ObjectiveEvalType = VulkanObjectiveEval<
        patch_apply::Layout,
        reduce_sum::Layout,
        [[[u32; 2]; 3]; 3],
        [[[[u32; 2]; 3]; 3]; 32],
        u32,
    >;
    fn new_obj_eval(
        &self,
        weights: &[[[[u32; 2]; 3]; 3]; 32],
        head: &[u32; 10],
        examples: &[(u8, [[[u32; 2]; 3]; 3])],
    ) -> Self::ObjectiveEvalType {
        let physical = PhysicalDevice::enumerate(&self.instance)
            .next()
            .expect("no device available");

        let queue_family: QueueFamily = physical
            .queue_families()
            .find(|&q| q.supports_compute())
            .expect("couldn't find a compute queue family");

        let (device, mut queues) = Device::new(
            physical,
            &Features::none(),
            &DeviceExtensions::none(),
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device");

        let queue = queues.next().unwrap();

        let prm_buffer =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), *weights)
                .expect("failed to create buffer");

        let head_buffer =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), *head)
                .expect("failed to create buffer");

        let (input_buffer, _) = ImmutableBuffer::from_iter(
            examples.iter().map(|(_, input)| input).cloned(),
            BufferUsage::all(),
            queue.clone(),
        )
        .expect("failed to create buffer");

        let (labels_buffer, _) = ImmutableBuffer::from_iter(
            examples.iter().map(|(label, _)| *label as u32),
            BufferUsage::all(),
            queue.clone(),
        )
        .expect("failed to create buffer");

        let embeddings_buffer: Arc<DeviceLocalBuffer<[u32]>> = DeviceLocalBuffer::array(
            device.clone(),
            examples.len(),
            BufferUsage::all(),
            [queue_family].iter().cloned(),
        )
        .expect("failed to create DeviceLocalBuffer");

        let objs_buffer: Arc<DeviceLocalBuffer<[u32]>> = DeviceLocalBuffer::array(
            device.clone(),
            examples.len(),
            BufferUsage::all(),
            [queue_family].iter().cloned(),
        )
        .expect("failed to create DeviceLocalBuffer");

        let obj_sums_iter =
            (0..(examples.len() as f64 / 125 as f64).ceil() as usize).map(|_| 0u32);
        let obj_sums_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), obj_sums_iter)
                .expect("failed to create buffer");

        let pa_shader =
            patch_apply::Shader::load(device.clone()).expect("failed to create shader module");
        let pa_compute_pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &pa_shader.main_entry_point(), &())
                .expect("failed to create compute pipeline"),
        );

        let pa_set = Arc::new(
            PersistentDescriptorSet::start(pa_compute_pipeline.clone(), 0)
                .add_buffer(prm_buffer.clone())
                .unwrap()
                .add_buffer(head_buffer.clone())
                .unwrap()
                .add_buffer(input_buffer.clone())
                .unwrap()
                .add_buffer(labels_buffer.clone())
                .unwrap()
                .add_buffer(embeddings_buffer.clone())
                .unwrap()
                .add_buffer(objs_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let rs_shader =
            reduce_sum::Shader::load(device.clone()).expect("failed to create shader module");
        let rs_compute_pipeline = Arc::new(
            ComputePipeline::new(device.clone(), &rs_shader.main_entry_point(), &())
                .expect("failed to create compute pipeline"),
        );

        let rs_set = Arc::new(
            PersistentDescriptorSet::start(rs_compute_pipeline.clone(), 0)
                .add_buffer(objs_buffer.clone())
                .unwrap()
                .add_buffer(obj_sums_buffer.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        Self::ObjectiveEvalType {
            N_EXAMPLES: examples.len(),
            input_patch_type: PhantomData,
            device: device.clone(),
            queue: queue,
            weights_buffer: prm_buffer,
            head_buffer: head_buffer,
            obj_sums_buffer: obj_sums_buffer,
            apply_compute_pipeline: pa_compute_pipeline,
            apply_descriptor_set: pa_set,
            reduce_sum_compute_pipeline: rs_compute_pipeline,
            reduce_sum_descriptor_set: rs_set,
            is_initialized: false,
            unclean_output_index: None,
            embedding_is_clean: false,
            obj_sum_is_clean: false,
        }
    }
}

pub struct VulkanObjectiveEval<ApplyLayout, ReduceSumLayout, InputPatch, Weights, Embedding> {
    N_EXAMPLES: usize,
    input_patch_type: PhantomData<InputPatch>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    weights_buffer: Arc<CpuAccessibleBuffer<Weights>>,
    head_buffer: Arc<CpuAccessibleBuffer<[Embedding; 10]>>,
    obj_sums_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    apply_compute_pipeline: Arc<ComputePipeline<PipelineLayout<ApplyLayout>>>,
    apply_descriptor_set: Arc<
        PersistentDescriptorSet<
            Arc<ComputePipeline<PipelineLayout<patch_apply::Layout>>>,
            (
                (
                    (
                        (
                            (
                                (
                                    (),
                                    PersistentDescriptorSetBuf<
                                        Arc<CpuAccessibleBuffer<[[[[u32; 2]; 3]; 3]; 32]>>,
                                    >,
                                ),
                                PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<[u32; 10]>>>,
                            ),
                            PersistentDescriptorSetBuf<Arc<ImmutableBuffer<[[[[u32; 2]; 3]; 3]]>>>,
                        ),
                        PersistentDescriptorSetBuf<Arc<ImmutableBuffer<[u32]>>>,
                    ),
                    PersistentDescriptorSetBuf<Arc<DeviceLocalBuffer<[u32]>>>,
                ),
                PersistentDescriptorSetBuf<Arc<DeviceLocalBuffer<[u32]>>>,
            ),
            StdDescriptorPoolAlloc,
        >,
    >,
    reduce_sum_compute_pipeline: Arc<ComputePipeline<PipelineLayout<ReduceSumLayout>>>,
    reduce_sum_descriptor_set: Arc<
        PersistentDescriptorSet<
            Arc<ComputePipeline<PipelineLayout<reduce_sum::Layout>>>,
            (
                (
                    (),
                    PersistentDescriptorSetBuf<Arc<DeviceLocalBuffer<[u32]>>>,
                ),
                PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<[u32]>>>,
            ),
            StdDescriptorPoolAlloc,
        >,
    >,
    is_initialized: bool,
    unclean_output_index: Option<usize>,
    embedding_is_clean: bool,
    obj_sum_is_clean: bool,
}

impl
    VulkanObjectiveEval<
        patch_apply::Layout,
        reduce_sum::Layout,
        [[[u32; 2]; 3]; 3],
        [[[[u32; 2]; 3]; 3]; 32],
        u32,
    >
{
    fn patch_apply() {}
    fn reduce_sum_obj(&mut self, batch_size: usize) -> u64 {
        let rs_push_constants = reduce_sum::ty::PushConstantData {
            batch_size: batch_size as u32,
        };

        let rs_command_buffer =
            AutoCommandBufferBuilder::new(self.device.clone(), self.queue.family())
                .unwrap()
                .dispatch(
                    [
                        ((self.N_EXAMPLES as f64 / 64f64) / batch_size as f64).ceil() as u32,
                        1,
                        1,
                    ],
                    self.reduce_sum_compute_pipeline.clone(),
                    self.reduce_sum_descriptor_set.clone(),
                    rs_push_constants,
                )
                .unwrap()
                .build()
                .unwrap();

        let finished = rs_command_buffer.execute(self.queue.clone()).unwrap();

        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        self.obj_sum_is_clean = true;
        let content = self.obj_sums_buffer.read().unwrap();
        content.par_iter().map(|x| *x as u64).sum()
    }
}

impl ObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32>
    for VulkanObjectiveEval<
        patch_apply::Layout,
        reduce_sum::Layout,
        [[[u32; 2]; 3]; 3],
        [[[[u32; 2]; 3]; 3]; 32],
        u32,
    >
{
    fn flip_weights_bit(&mut self, o: usize, i: usize) {
        let mut weights = self.weights_buffer.write().unwrap();
        weights[o].flip_bit(i);
        self.unclean_output_index = Some(o);
        self.embedding_is_clean = false;
        self.obj_sum_is_clean = false;
    }
    fn flip_head_bit(&mut self, o: usize, i: usize) {
        let mut head = self.head_buffer.write().unwrap();
        head[o].flip_bit(i);
        self.obj_sum_is_clean = false;
    }
    fn obj(&mut self) -> u64 {
        let pa_push_constants = patch_apply::ty::PushConstantData {
            output_index: self.unclean_output_index.unwrap_or(0) as u32,
            full_apply: if !self.is_initialized {
                1
            } else if !self.embedding_is_clean {
                0
            } else {
                2
            },
        };
        let pa_command_buffer =
            AutoCommandBufferBuilder::new(self.device.clone(), self.queue.family())
                .unwrap()
                .dispatch(
                    [(self.N_EXAMPLES as f64 / 64f64).ceil() as u32, 1, 1],
                    self.apply_compute_pipeline.clone(),
                    self.apply_descriptor_set.clone(),
                    pa_push_constants,
                )
                .unwrap()
                .build()
                .unwrap();

        let finished = pa_command_buffer.execute(self.queue.clone()).unwrap();
        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        self.is_initialized = true;
        self.unclean_output_index = None;
        self.embedding_is_clean = true;
        self.reduce_sum_obj(125)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEval, TestCPUObjectiveEvalCreator,
        VulkanObjectiveEval, VulkanObjectiveEvalCreator,
    };
    use rand::prelude::*;
    use rand_hc::Hc128Rng;

    #[test]
    fn test_cpu_obj() {
        // use a prime to make the shader code more likely to fail.
        const N_EXAMPLES: usize = 104729;
        let mut rng = Hc128Rng::seed_from_u64(42);
        let weights: [[[[u32; 2]; 3]; 3]; 32] = rng.gen();
        let head: [u32; 10] = rng.gen();
        let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..N_EXAMPLES)
            .map(|_| (rng.gen_range(0, 10), rng.gen()))
            .collect();

        let eval_creator: TestCPUObjectiveEvalCreator<
            [[[u32; 2]; 3]; 3],
            [[[[u32; 2]; 3]; 3]; 32],
            u32,
        > = TestCPUObjectiveEvalCreator::new();
        let mut obj_eval: TestCPUObjectiveEval<[[[u32; 2]; 3]; 3], [[[[u32; 2]; 3]; 3]; 32], u32> =
            eval_creator.new_obj_eval(&weights, &head, &examples);

        let obj1: u64 = obj_eval.obj();
        dbg!(obj1);
        let avg_obj = obj1 as f64 / N_EXAMPLES as f64;
        dbg!(avg_obj);
        assert!(avg_obj > 0.07);
        assert!(avg_obj < 0.09);

        obj_eval.flip_weights_bit(22, 5);
        let obj2: u64 = obj_eval.obj();
        dbg!(obj2);
        let avg_obj = obj2 as f64 / N_EXAMPLES as f64;
        dbg!(avg_obj);
        assert!(avg_obj > 0.07);
        assert!(avg_obj < 0.09);

        assert_ne!(obj1, obj2);

        obj_eval.flip_weights_bit(22, 5);
        let obj3: u64 = obj_eval.obj();
        assert_eq!(obj1, obj3);
    }
    #[test]
    fn glsl_like_cpu() {
        const N_EXAMPLES: usize = 104729;
        let mut rng = Hc128Rng::seed_from_u64(42);
        let weights: [[[[u32; 2]; 3]; 3]; 32] = rng.gen();
        let head: [u32; 10] = rng.gen();
        let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..N_EXAMPLES)
            .map(|_| (rng.gen_range(0, 10), rng.gen()))
            .collect();

        let test_eval_creator = TestCPUObjectiveEvalCreator::new();
        let mut test_obj_eval = test_eval_creator.new_obj_eval(&weights, &head, &examples);
        let test_obj: u64 = test_obj_eval.obj();

        let glsl_like_eval_creator = TestCPUObjectiveEvalCreator::new();
        let mut glsl_like_obj_eval =
            glsl_like_eval_creator.new_obj_eval(&weights, &head, &examples);
        let glsl_like_obj = glsl_like_obj_eval.obj();

        assert_eq!(test_obj, glsl_like_obj);
    }
    #[test]
    fn vulkan_obj() {
        let mut rng = Hc128Rng::seed_from_u64(1);
        let weights: [[[[u32; 2]; 3]; 3]; 32] = rng.gen();
        let head: [u32; 10] = rng.gen();
        let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..104729)
            .map(|_| (rng.gen_range(0, 10), rng.gen()))
            .collect();

        let vk_eval_creator = VulkanObjectiveEvalCreator::new();
        let mut vk_obj_eval = vk_eval_creator.new_obj_eval(&weights, &head, &examples);
        let test_eval_creator = TestCPUObjectiveEvalCreator::new();
        let mut test_obj_eval = test_eval_creator.new_obj_eval(&weights, &head, &examples);

        let vk_obj: u64 = vk_obj_eval.obj();
        let test_obj: u64 = test_obj_eval.obj();
        assert_eq!(vk_obj, test_obj);

        let mut rng = Hc128Rng::seed_from_u64(2);
        let weights: [[[[u32; 2]; 3]; 3]; 32] = rng.gen();
        let head: [u32; 10] = rng.gen();
        let examples: Vec<(u8, [[[u32; 2]; 3]; 3])> = (0..100000)
            .map(|_| (rng.gen_range(0, 10), rng.gen()))
            .collect();
        let mut vk_obj_eval = vk_eval_creator.new_obj_eval(&weights, &head, &examples);
        let mut test_obj_eval = test_eval_creator.new_obj_eval(&weights, &head, &examples);

        for &(o, i) in &[(0, 0), (0, 5), (1, 5), (0, 0), (31, 9 * 32 * 2 - 1)] {
            vk_obj_eval.flip_weights_bit(o, i);
            test_obj_eval.flip_weights_bit(o, i);

            let vk_obj: u64 = vk_obj_eval.obj();
            let test_obj: u64 = test_obj_eval.obj();
            assert_eq!(vk_obj, test_obj);
        }
    }
}
