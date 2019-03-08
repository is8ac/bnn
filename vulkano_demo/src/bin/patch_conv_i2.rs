extern crate rand;
extern crate rayon;
extern crate time;
extern crate vulkano;
extern crate vulkano_demo;
extern crate vulkano_shaders;
use rand::prelude::*;
use rand_hc::Hc128Rng;
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
use vulkano_demo::ObjectiveEvalCreator;

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
    fn new() -> Self {
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
            N_EXAMPLES,
            BufferUsage::all(),
            [queue_family].iter().cloned(),
        )
        .expect("failed to create DeviceLocalBuffer");

        let objs_buffer: Arc<DeviceLocalBuffer<[u32]>> = DeviceLocalBuffer::array(
            device.clone(),
            N_EXAMPLES,
            BufferUsage::all(),
            [queue_family].iter().cloned(),
        )
        .expect("failed to create DeviceLocalBuffer");

        let obj_sums_iter = (0..N_EXAMPLES / SUM_BATCH_SIZE).map(|_| 0u32);
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
            weights_buffer: CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::all(),
                *weights,
            )
            .expect("failed to create weights buffer"),
            head_buffer: CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), *head)
                .expect("failed to create head buffer"),
            input_data_buffer: input_buffer,
            labels_buffer: labels_buffer,
            embeddings_buffer: DeviceLocalBuffer::array(
                device.clone(),
                examples.len(),
                BufferUsage::all(),
                [queue_family].iter().cloned(),
            )
            .expect("failed to create embeddings buffer"),
            objs_buffer: DeviceLocalBuffer::array(
                device.clone(),
                N_EXAMPLES,
                BufferUsage::all(),
                [queue_family].iter().cloned(),
            )
            .expect("failed to create DeviceLocalBuffer"),
            obj_sums_buffer: CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::all(),
                (0..N_EXAMPLES / SUM_BATCH_SIZE).map(|_| 0u32),
            )
            .expect("failed to create buffer"),
            apply_compute_pipeline: pa_compute_pipeline,
            apply_descriptor_set: pa_set,
            reduce_sum_compute_pipeline: rs_compute_pipeline,
            reduce_sum_descriptor_set: rs_set,
        }
    }
}

pub struct VulkanObjectiveEval<
    ApplyLayout,
    ReduceSumLayout,
    InputPatch,
    Weights,
    Embedding,
> {
    weights_buffer: Arc<CpuAccessibleBuffer<Weights>>,
    head_buffer: Arc<CpuAccessibleBuffer<[Embedding; 10]>>,
    input_data_buffer: Arc<ImmutableBuffer<[InputPatch]>>,
    labels_buffer: Arc<ImmutableBuffer<[u32]>>,
    embeddings_buffer: Arc<DeviceLocalBuffer<[Embedding]>>,
    objs_buffer: Arc<DeviceLocalBuffer<[u32]>>,
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
}

const N_EXAMPLES: usize = (50000 * 900) / 3;
//const N_EXAMPLES: usize = (175781 * 256);
const SUM_BATCH_SIZE: usize = 125;
const output_index: u32 = 7;
const full_apply: u32 = 1;
const INPUT_LEN: usize = 2;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(42);

    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance)
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

    let mut params: [[[[u32; INPUT_LEN]; 3]; 3]; 32] = rng.gen();
    let prm_buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), params)
        .expect("failed to create buffer");
    let objective_head: [u32; 10] = rng.gen();
    let head_buffer =
        CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), objective_head)
            .expect("failed to create buffer");

    let input_data: Vec<[[[u32; INPUT_LEN]; 3]; 3]> = (0..N_EXAMPLES).map(|_| rng.gen()).collect();
    let (input_buffer, _) = ImmutableBuffer::from_iter(
        input_data.iter().cloned(),
        BufferUsage::all(),
        queue.clone(),
    )
    .expect("failed to create buffer");

    let labels_data: Vec<u32> = (0..N_EXAMPLES).map(|_| rng.gen_range(0, 10)).collect();
    let (labels_buffer, _) = ImmutableBuffer::from_iter(
        labels_data.iter().cloned(),
        BufferUsage::all(),
        queue.clone(),
    )
    .expect("failed to create buffer");

    let embeddings_buffer: Arc<DeviceLocalBuffer<[u32]>> = DeviceLocalBuffer::array(
        device.clone(),
        N_EXAMPLES,
        BufferUsage::all(),
        [queue_family].iter().cloned(),
    )
    .expect("failed to create DeviceLocalBuffer");

    let objs_buffer: Arc<DeviceLocalBuffer<[u32]>> = DeviceLocalBuffer::array(
        device.clone(),
        N_EXAMPLES,
        BufferUsage::all(),
        [queue_family].iter().cloned(),
    )
    .expect("failed to create DeviceLocalBuffer");

    let obj_sums_iter = (0..N_EXAMPLES / SUM_BATCH_SIZE).map(|_| 0u32);
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
    let rs_push_constants = reduce_sum::ty::PushConstantData {
        batch_size: SUM_BATCH_SIZE as u32,
    };

    let pa_push_constants = patch_apply::ty::PushConstantData {
        output_index: output_index,
        full_apply: 1,
    };
    let pa_command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .dispatch(
            [N_EXAMPLES as u32 / 64, 1, 1],
            pa_compute_pipeline.clone(),
            pa_set.clone(),
            pa_push_constants,
        )
        .unwrap()
        .build()
        .unwrap();

    let finished = pa_command_buffer.execute(queue.clone()).unwrap();

    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    {
        let mut content = prm_buffer.write().unwrap();
        content[output_index as usize][2][2][1] ^= 1u32 << 27;
        params[output_index as usize][2][2][1] ^= 1u32 << 27;
    }

    let start = PreciseTime::now();
    let pa_push_constants = patch_apply::ty::PushConstantData {
        output_index: output_index,
        full_apply: 0,
    };
    let pa_command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .dispatch(
            [N_EXAMPLES as u32 / 64, 1, 1],
            pa_compute_pipeline.clone(),
            pa_set.clone(),
            pa_push_constants,
        )
        .unwrap()
        .build()
        .unwrap();

    let finished = pa_command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    println!(
        "GPU mills: {:?}",
        start.to(PreciseTime::now()).num_milliseconds()
    );

    let rs_push_constants = reduce_sum::ty::PushConstantData {
        batch_size: SUM_BATCH_SIZE as u32,
    };

    let rs_command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .dispatch(
            [(N_EXAMPLES as u32 / 64) / SUM_BATCH_SIZE as u32, 1, 1],
            rs_compute_pipeline.clone(),
            rs_set.clone(),
            rs_push_constants,
        )
        .unwrap()
        .build()
        .unwrap();

    let finished = rs_command_buffer.execute(queue.clone()).unwrap();

    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let content = obj_sums_buffer.read().unwrap();
    let gpu_obj: u64 = content.par_iter().map(|x| *x as u64).sum();
    dbg!(gpu_obj);
    println!(
        "sum GPU mills: {:?}",
        start.to(PreciseTime::now()).num_milliseconds()
    );
    assert_eq!(content.len(), N_EXAMPLES / SUM_BATCH_SIZE);
    dbg!(content.len());

    let threshold: u32 = (32 * 9) / 2;
    let start = PreciseTime::now();
    let cpu_objs: Vec<u32> = input_data
        .par_iter()
        .zip(labels_data.par_iter())
        .map(|(patch, class)| {
            let mut target = 0b0u32;
            for b in 0..32 {
                let mut sum = 0;
                for x in 0..3 {
                    for y in 0..3 {
                        for i in 0..INPUT_LEN {
                            sum += (patch[x][y][i] ^ params[b][x][y][i]).count_ones();
                        }
                    }
                }
                target |= ((sum > threshold) as u32) << b;
            }

            let max = (objective_head[*class as usize] ^ target).count_ones();
            for o in 0..10 {
                if o != *class as usize {
                    if (objective_head[o] ^ target).count_ones() >= max {
                        return 0;
                    }
                }
            }
            1
        })
        .collect();
    let cpu_obj: u64 = cpu_objs.par_iter().map(|x| *x as u64).sum();
    println!(
        "CPU mills: {:?}",
        start.to(PreciseTime::now()).num_milliseconds()
    );
    assert_eq!(cpu_obj, gpu_obj);
}
