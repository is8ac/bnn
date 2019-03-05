// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! This example contains the source code of the second part of the guide at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the guide itself.

extern crate rand;
extern crate rayon;
extern crate time;
extern crate vulkano;
extern crate vulkano_shaders;

use rand::prelude::*;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::sync::Arc;
use time::PreciseTime;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::{CpuAccessibleBuffer, DeviceLocalBuffer, ImmutableBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

mod reduce_sum {
    vulkano_shaders::shader! {
    ty: "compute",
        src: "
        #version 450

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        layout(set = 0, binding = 0) buffer InputData {
            uint data[];
        } input_data;

        layout(set = 0, binding = 1) buffer OutputSum {
            uint data[];
        } output_sum;

        layout(push_constant) uniform PushConstantData {
            uint batch_size;
        } pc;

        void main() {
            uint i;

            uint sum = 0;
            uint base = gl_GlobalInvocationID.x * pc.batch_size;

            for (i=0; i<pc.batch_size; i+=1) {
                sum += input_data.data[base + i];
            }
            output_sum.data[gl_GlobalInvocationID.x] = sum;
        }"
    }
}

mod patch_apply {
    vulkano_shaders::shader! {
    ty: "compute",
        src: "
        #version 450

        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

        layout(set = 0, binding = 0) buffer Prms {
            uint[32][3][3] data;
        } prms;

        layout(set = 0, binding = 1) buffer ObjHead {
            uint[10] data;
        } head;

        layout(set = 0, binding = 2) buffer InputPatch {
            uint[3][3] data[];
        } input_patch;

        layout(set = 0, binding = 3) buffer Label {
            uint data[];
        } label;

        layout(set = 0, binding = 4) buffer Embedding {
            uint data[];
        } embedding;

        layout(set = 0, binding = 5) buffer Objective {
            uint data[];
        } objective;

        layout(push_constant) uniform PushConstantData {
            uint output_index;
            uint full_apply;
        } pc;

        void main() {
            const uint onebits = 4294967295;
            const uint threshold = (32 * 9) / 2;
            uint x;
            uint y;
            uint b;
            uint o;
            uint target;
            uint sum_bits;
            uint max;

            if (pc.full_apply == 1) {
                target = 0;
                for (b=0; b<32; b+=1){
                    sum_bits = 0;
                    for (x=0; x<3; x+=1){
                        for (y=0; y<3; y+=1){
                            sum_bits += bitCount(prms.data[b][x][y] ^ input_patch.data[gl_GlobalInvocationID.x][x][y]);
                        }
                    }
                    target |= uint(sum_bits > threshold) << b;
                }
                embedding.data[gl_GlobalInvocationID.x] = target;
            } else {
                sum_bits = 0;
                for (x=0; x<3; x+=1){
                    for (y=0; y<3; y+=1){
                        sum_bits += bitCount(prms.data[pc.output_index][x][y] ^ input_patch.data[gl_GlobalInvocationID.x][x][y]);
                    }
                }
                embedding.data[gl_GlobalInvocationID.x] &= (onebits ^ (uint(1) << pc.output_index));
                embedding.data[gl_GlobalInvocationID.x] |= uint(sum_bits > threshold) << pc.output_index;
            }

            // now we can start on the objective head
            max = bitCount(head.data[label.data[gl_GlobalInvocationID.x]] ^ embedding.data[gl_GlobalInvocationID.x]);
            for (o=0; o<10; o+=1) {
                if (o != label.data[gl_GlobalInvocationID.x]) {
                    if (bitCount(head.data[o] ^ embedding.data[gl_GlobalInvocationID.x]) >= max) {
                        objective.data[gl_GlobalInvocationID.x] = 0;
                        return;
                    }
                }
            }
            objective.data[gl_GlobalInvocationID.x] = 1;
        }"
    }
}

const N_EXAMPLES: usize = (50000 * 900);
const SUM_BATCH_SIZE: usize = 25;
const output_index: u32 = 7;
const full_apply: u32 = 0;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(42);

    let instance = Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    let queue_family = physical
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

    let mut params: [[[u32; 3]; 3]; 32] = rng.gen();
    let prm_buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), params).expect("failed to create buffer");
    let objective_head: [u32; 10] = rng.gen();
    let head_buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), objective_head).expect("failed to create buffer");

    let input_data: Vec<[[u32; 3]; 3]> = (0..N_EXAMPLES).map(|_| rng.gen()).collect();
    let (input_buffer, _) =
        ImmutableBuffer::from_iter(input_data.iter().cloned(), BufferUsage::all(), queue.clone()).expect("failed to create buffer");

    let labels_data: Vec<u32> = (0..N_EXAMPLES).map(|_| rng.gen_range(0, 10)).collect();
    let (labels_buffer, _) =
        ImmutableBuffer::from_iter(labels_data.iter().cloned(), BufferUsage::all(), queue.clone()).expect("failed to create buffer");

    let embeddings_buffer: Arc<DeviceLocalBuffer<[u32]>> =
        DeviceLocalBuffer::array(device.clone(), N_EXAMPLES, BufferUsage::all(), [queue_family].iter().cloned())
            .expect("failed to create DeviceLocalBuffer");

    let objs_buffer: Arc<DeviceLocalBuffer<[u32]>> =
        DeviceLocalBuffer::array(device.clone(), N_EXAMPLES, BufferUsage::all(), [queue_family].iter().cloned())
            .expect("failed to create DeviceLocalBuffer");

    let obj_sums_iter = (0..N_EXAMPLES / SUM_BATCH_SIZE).map(|_| 0u32);
    let obj_sums_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), obj_sums_iter).expect("failed to create buffer");

    let pa_shader = patch_apply::Shader::load(device.clone()).expect("failed to create shader module");
    let pa_compute_pipeline =
        Arc::new(ComputePipeline::new(device.clone(), &pa_shader.main_entry_point(), &()).expect("failed to create compute pipeline"));

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

    let rs_shader = reduce_sum::Shader::load(device.clone()).expect("failed to create shader module");
    let rs_compute_pipeline =
        Arc::new(ComputePipeline::new(device.clone(), &rs_shader.main_entry_point(), &()).expect("failed to create compute pipeline"));

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

    finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

    {
        let mut content = prm_buffer.write().unwrap();
        content[output_index as usize][2][2] ^= 1u32 << 27;
        params[output_index as usize][2][2] ^= 1u32 << 27;
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
    finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();
    println!("GPU mills: {:?}", start.to(PreciseTime::now()).num_milliseconds());

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

    finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

    let content = obj_sums_buffer.read().unwrap();
    let gpu_obj: u64 = content.par_iter().map(|x| *x as u64).sum();
    dbg!(gpu_obj);
    println!("sum GPU mills: {:?}", start.to(PreciseTime::now()).num_milliseconds());
    assert_eq!(content.len(), N_EXAMPLES / SUM_BATCH_SIZE);
    dbg!(content.len());
    println!("{:?}", &content[0..100]);

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
                        sum += (patch[x][y] ^ params[b][x][y]).count_ones();
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
    println!("CPU mills: {:?}", start.to(PreciseTime::now()).num_milliseconds());
    assert_eq!(cpu_obj, gpu_obj);
}
