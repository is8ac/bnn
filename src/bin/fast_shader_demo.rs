extern crate bitnn;
extern crate rand;
extern crate time;
use bitnn::datasets::cifar;

use bitnn::layers::{Apply, MirrorHammingDistance};
use bitnn::objective_eval::IsCorrect;
use bitnn::objective_eval::{ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEvalCreator};
use bitnn::{
    BitLen, ExtractPatches, FlipBit, FlipBitIndexed, GetMirroredWords, GetPatch, GetWord,
    HammingDistance, WordLen,
};
use rand::prelude::*;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
extern crate vulkano;
extern crate vulkano_shaders;
use rayon::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::iter;
use std::marker::PhantomData;
use std::path::Path;
use std::thread;
use std::thread::JoinHandle;

use time::PreciseTime;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuf};
use vulkano::descriptor::pipeline_layout::PipelineLayout;
use vulkano::device::Queue;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync;
use vulkano::sync::GpuFuture;

use std::sync::Arc;

mod fast_obj_update_e1 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_update_e1.glsl",
    }
}

mod fast_obj_update_e2 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_update_e2.glsl",
    }
}

mod transition_input_word_e1 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_input_word_trans_e1.glsl",
    }
}

mod transition_input_word_e2 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_input_word_trans_e2.glsl",
    }
}

mod clean_embedding_bit_e1 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_embedding_bit_clean_e1.glsl",
    }
}
mod clean_embedding_bit_e2 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_embedding_bit_clean_e2.glsl",
    }
}

mod head_obj_update_e1 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_head_update_e1.glsl",
    }
}

mod head_obj_update_e2 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_head_update_e2.glsl",
    }
}

mod replace_cache_parts_e1 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_cache_replace_input_e1.glsl",
    }
}
mod replace_cache_parts_e2 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_cache_replace_input_e2.glsl",
    }
}

// Note that when embedding word len get to be ~8, it may be worth switching to cached head partial sum so as to make head obj linear with embedding size.
// However while it is small, the memory overhead of the 10 nonconstant partial sums is probably not worth the compute cost savings.
// This is for 10 classes.
// For 100 classes, the transition point will be far larger and it's probably not worth it.
#[derive(Debug, Copy, Clone)]
pub struct FastExampleMirroredCache<Embedding> {
    input_word: [u32; 2], // a pair of 32 bits. This is mirrored, so we need a normal and a flipped input word.
    input_partial_sums: [u32; 2], // a pair of integers, one for normal one for flipped.
    embedding: Embedding,
    true_class: u32,
}

#[derive(Debug, Copy, Clone)]
pub struct FastExampleMirroredParts {
    input_word: [u32; 2],
    input_partial_sums: [u32; 2],
}

pub trait NewMirror3x3FastCache<Weights, Embedding> {
    fn new_full(
        &self,
        weights: &Weights,
        class: usize,
        embedding_index: usize,
        patch_index: usize,
    ) -> FastExampleMirroredCache<Embedding>;
    fn new_parts(&self, weights_patch: &Self, patch_index: usize) -> FastExampleMirroredParts;
}

macro_rules! impl_NewMirror3x3FastCache_for_FastExampleMirrored {
    ($input_len:expr, $output_len:expr) => {
        impl
            NewMirror3x3FastCache<
                [[[[[u32; $input_len]; 3]; 3]; 16]; $output_len],
                [u32; $output_len],
            > for [[[u32; $input_len]; 3]; 3]
        {
            fn new_full(
                &self,
                weights: &[[[[[u32; $input_len]; 3]; 3]; 16]; $output_len],
                class: usize,
                embedding_index: usize,
                patch_index: usize,
            ) -> FastExampleMirroredCache<[u32; $output_len]> {
                let embedding = weights.apply(self);
                let weights_patch = weights.get_patch(embedding_index);
                let input_words = self.get_mirrored_words(patch_index);
                FastExampleMirroredCache {
                    input_word: input_words,
                    input_partial_sums: [
                        weights_patch.normal_hamming_distance(self)
                            - weights_patch
                                .get_word(patch_index)
                                .hamming_distance(&input_words[0]),
                        weights_patch.fliped_hamming_distance(self)
                            - weights_patch
                                .get_word(patch_index)
                                .hamming_distance(&input_words[1]),
                    ],
                    embedding: embedding,
                    true_class: class as u32,
                }
            }
            fn new_parts(
                &self,
                weights_patch: &[[[u32; $input_len]; 3]; 3],
                patch_index: usize,
            ) -> FastExampleMirroredParts {
                let input_words = self.get_mirrored_words(patch_index);
                FastExampleMirroredParts {
                    input_word: input_words,
                    input_partial_sums: [
                        weights_patch.normal_hamming_distance(&self)
                            - weights_patch
                                .get_word(patch_index)
                                .hamming_distance(&input_words[0]),
                        weights_patch.fliped_hamming_distance(&self)
                            - weights_patch
                                .get_word(patch_index)
                                .hamming_distance(&input_words[1]),
                    ],
                }
            }
        }
    };
}
//impl_NewMirror3x3FastCache_for_FastExampleMirrored!(4, 2);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(3, 2);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(2, 2);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(1, 2);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(1, 1);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(2, 1);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(3, 1);
//impl_NewMirror3x3FastCache_for_FastExampleMirrored!(4, 1);

pub struct FastCacheVKObjEval<ShaderLayout, InputPatch, Weights, Embedding> {
    n_examples: usize,
    examples: Arc<Vec<InputPatch>>,
    sum_batch_size: usize,
    weights: Weights,
    head: [Embedding; 10],
    input_patch_type: PhantomData<InputPatch>,
    embedding_type: PhantomData<Embedding>,
    weights_type: PhantomData<Weights>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    obj_sums_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    cache_buffer: Arc<ImmutableBuffer<[FastExampleMirroredCache<Embedding>]>>,
    update_pipeline: Arc<ComputePipeline<PipelineLayout<ShaderLayout>>>,
    update_descriptor_set: Arc<
        PersistentDescriptorSet<
            Arc<ComputePipeline<PipelineLayout<ShaderLayout>>>,
            (
                (
                    (),
                    PersistentDescriptorSetBuf<
                        Arc<ImmutableBuffer<[FastExampleMirroredCache<Embedding>]>>,
                    >,
                ),
                PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<[u32]>>>,
            ),
        >,
    >,
    embedding_index: usize,
    patch_index: usize,
    embedding_is_clean: bool,
    next_input_words_buffer_join_handle: Option<JoinHandle<Arc<ImmutableBuffer<[[u32; 2]]>>>>,
    next_input_words_buffer_patch_index: usize,
    next_input_words_buffer_embedding_index: usize,
}

macro_rules! impl_fastexamplemirrored_over_embedding {
    ($shader_mod_name:ident, $input_trans_shader_mod_name:ident, $clean_embedding_bit_mod_name:ident, $head_obj_update_mod_name:ident, $replace_cache_parts_shader_mod_name:ident, $embedding_len:expr) => {
        impl<InputPatch: 'static + Sync + Send + GetWord + GetMirroredWords + BitLen + Copy + NewMirror3x3FastCache<[[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>>
            FastCacheVKObjEval<$shader_mod_name::Layout, InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>
        {
            fn new_vk_fast_cache(
                examples: &[(u8, InputPatch)],
                weights: &[[InputPatch; 16]; $embedding_len],
                head: &[[u32; $embedding_len]; 10],
                embedding_index: usize,
                patch_index: usize,
                batch_size: usize,
            ) -> Self {
                let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
                let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
                let queue_family = physical.queue_families().find(|&q| q.supports_compute()).unwrap();

                // Now initializing the device.
                let (device, mut queues) = Device::new(
                    physical,
                    physical.supported_features(),
                    &DeviceExtensions::none(),
                    [(queue_family, 0.5)].iter().cloned(),
                )
                .unwrap();

                let queue = queues.next().unwrap();

                let pipeline = Arc::new({
                    let shader = $shader_mod_name::Shader::load(device.clone()).unwrap();
                    ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
                });

                let sums_buffer = {
                    let data_iter = (0..(examples.len() as f64 / batch_size as f64).ceil() as u32).map(|_| 0u32);
                    CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data_iter).unwrap()
                };

                let cache_buffer = {
                    let cache_vec: Vec<_> = examples
                        .par_iter()
                        .map(|(class, patch)| patch.new_full(weights, *class as usize, embedding_index, patch_index))
                        .collect();
                    let (cache_buffer, _) = ImmutableBuffer::from_iter(cache_vec.iter().cloned(), BufferUsage::all(), queue.clone()).unwrap();
                    cache_buffer
                };

                let set = Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(cache_buffer.clone())
                        .unwrap()
                        .add_buffer(sums_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                FastCacheVKObjEval {
                    n_examples: examples.len(),
                    examples: Arc::new(examples.iter().map(|(_, patch)| patch).cloned().collect()),
                    sum_batch_size: batch_size,
                    weights: *weights,
                    head: *head,
                    embedding_type: PhantomData,
                    input_patch_type: PhantomData,
                    weights_type: PhantomData,
                    device: device,
                    queue: queue,
                    obj_sums_buffer: sums_buffer,
                    cache_buffer: cache_buffer,
                    update_pipeline: pipeline,
                    update_descriptor_set: set,
                    embedding_index: embedding_index,
                    patch_index: patch_index,
                    embedding_is_clean: true,
                    next_input_words_buffer_join_handle: None,
                    next_input_words_buffer_patch_index: 0,
                    next_input_words_buffer_embedding_index: 0,
                }
            }

            fn sum_obj(&self) -> u64 {
                let pa_push_constants = $shader_mod_name::ty::PushConstantData {
                    head: self.head,
                    embedding_bit_index: self.embedding_index as u32 % 16,
                    embedding_word_index: self.embedding_index as u32 / 16,
                    threshold: (<InputPatch>::BIT_LEN / 2) as u32,
                    weights_word: self.weights.get_patch(self.embedding_index).get_word(self.patch_index),
                    batch_size: self.sum_batch_size as u32,
                };
                let n_workgroups = ((self.n_examples as f64 / 256f64) / self.sum_batch_size as f64).ceil() as u32;
                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                    .unwrap()
                    .dispatch(
                        [n_workgroups, 1, 1],
                        self.update_pipeline.clone(),
                        self.update_descriptor_set.clone(),
                        pa_push_constants,
                    )
                    .unwrap()
                    .build()
                    .unwrap();

                let future = sync::now(self.device.clone())
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap();
                future.wait(None).unwrap();

                //dbg!(self.patch_index);
                //let start = PreciseTime::now();
                let data_buffer_content = self.obj_sums_buffer.read().unwrap();
                let gpu_sum: u64 = data_buffer_content.par_iter().map(|&x| x as u64).sum();
                //println!("sum ms: {}", start.to(PreciseTime::now()).num_milliseconds());
                gpu_sum
            }
            fn transition_input_word(&mut self, target_weight_index: usize) {
                let start = PreciseTime::now();
                if (self.next_input_words_buffer_patch_index != target_weight_index) | (self.next_input_words_buffer_embedding_index != self.embedding_index) {
                    println!("an input word buffer creator thread was started, but it was for the wrong input word", );
                    if let Some(handle) = self.next_input_words_buffer_join_handle.take() {
                        handle.join().expect("can't join thread");
                    }
                    println!("it is now dead", );
                    self.next_input_words_buffer_join_handle = None;
                }
                if self.next_input_words_buffer_join_handle.is_none() {
                    println!("no correct input words buffer had been created so creating now...", );
                    self.start_prepare_input_words_buffer(target_weight_index);
                }
                let new_words_buffer = self.next_input_words_buffer_join_handle.take().unwrap().join().expect("can't join thread");
                //println!("get pre prepared input words ms: {}", start.to(PreciseTime::now()).num_milliseconds());

                let pipeline = Arc::new({
                    let shader = $input_trans_shader_mod_name::Shader::load(self.device.clone()).unwrap();
                    ComputePipeline::new(self.device.clone(), &shader.main_entry_point(), &()).unwrap()
                });
                let set = Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(self.cache_buffer.clone())
                        .unwrap()
                        .add_buffer(new_words_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                let push_constants = $input_trans_shader_mod_name::ty::PushConstantData {
                    new_weights_word: self.weights.get_patch(self.embedding_index).get_word(target_weight_index),
                    old_weights_word: self.weights.get_patch(self.embedding_index).get_word(self.patch_index),
                };
                let n_workgroups = (self.n_examples as f64 / 1024f64).ceil() as u32;
                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                    .unwrap()
                    .dispatch([n_workgroups, 1, 1], pipeline.clone(), set.clone(), push_constants)
                    .unwrap()
                    .build()
                    .unwrap();

                let future = sync::now(self.device.clone())
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap();
                future.wait(None).unwrap();
                self.patch_index = target_weight_index;
                //println!("full trans time: {}", start.to(PreciseTime::now()).num_milliseconds());
            }

            fn start_prepare_input_words_buffer(&mut self, target_weight_index: usize) {
                if let Some(handle) = self.next_input_words_buffer_join_handle.take() {
                    println!("unneeded thread was created, joining before creating a new one", );
                    handle.join().expect("can't join thread");
                }
                let examples = self.examples.clone();
                let queue = self.queue.clone();
                let child = thread::spawn(move || {
                    let start = PreciseTime::now();
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
                    let data_vec = pool.install(||{
                        let data_vec: Vec<[u32; 2]> = examples
                            .par_iter()
                            .map(|patch| patch.get_mirrored_words(target_weight_index))
                            .collect();
                        data_vec
                    });

                    let (new_words_buffer, _) = ImmutableBuffer::from_iter(data_vec.iter().cloned(), BufferUsage::all(), queue).unwrap();
                    //println!("new words prep time: {}", start.to(PreciseTime::now()).num_milliseconds());
                    new_words_buffer
                });
                self.next_input_words_buffer_join_handle = Some(child);
                self.next_input_words_buffer_patch_index = target_weight_index;
                self.next_input_words_buffer_embedding_index = self.embedding_index;
            }

            fn clean_embedding_bit(&mut self) {
                let start = PreciseTime::now();
                let pipeline = Arc::new({
                    let shader = $clean_embedding_bit_mod_name::Shader::load(self.device.clone()).unwrap();
                    ComputePipeline::new(self.device.clone(), &shader.main_entry_point(), &()).unwrap()
                });
                let set = Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(self.cache_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                let push_constants = $clean_embedding_bit_mod_name::ty::PushConstantData {
                    embedding_bit_index: self.embedding_index as u32 % 16,
                    embedding_word_index: self.embedding_index as u32 / 16,
                    threshold: (<InputPatch>::BIT_LEN / 2) as u32,
                    weights_word: self.weights.get_patch(self.embedding_index).get_word(self.patch_index),
                };
                let n_workgroups = (self.n_examples as f64 / 1024f64).ceil() as u32;
                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                    .unwrap()
                    .dispatch([n_workgroups, 1, 1], pipeline.clone(), set.clone(), push_constants)
                    .unwrap()
                    .build()
                    .unwrap();

                let future = sync::now(self.device.clone())
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap();
                future.wait(None).unwrap();
                self.embedding_is_clean = true;
                //println!("embedding bit clean ms: {}", start.to(PreciseTime::now()).num_milliseconds());
            }
            fn replace_cache_parts(&mut self, target_embedding_index: usize) {
                if !self.embedding_is_clean {
                    self.clean_embedding_bit();
                }
                let new_weights_patch = self.weights.get_patch(target_embedding_index);
                let new_cache_parts_buffer = {
                    let cache_vec: Vec<_> = self.examples
                        .par_iter()
                        .map(|patch| patch.new_parts(&new_weights_patch, 0))
                        .collect();
                    let (cache_buffer, _) = ImmutableBuffer::from_iter(cache_vec.iter().cloned(), BufferUsage::all(), self.queue.clone()).unwrap();
                    cache_buffer
                };


                let start = PreciseTime::now();
                let pipeline = Arc::new({
                    let shader = $replace_cache_parts_shader_mod_name::Shader::load(self.device.clone()).unwrap();
                    ComputePipeline::new(self.device.clone(), &shader.main_entry_point(), &()).unwrap()
                });
                let set = Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(self.cache_buffer.clone())
                        .unwrap()
                        .add_buffer(new_cache_parts_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                let n_workgroups = (self.n_examples as f64 / 1024f64).ceil() as u32;
                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                    .unwrap()
                    .dispatch([n_workgroups, 1, 1], pipeline.clone(), set.clone(), ())
                    .unwrap()
                    .build()
                    .unwrap();

                let future = sync::now(self.device.clone())
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap();
                future.wait(None).unwrap();
                self.embedding_index = target_embedding_index;
                self.patch_index = 0;
                dbg!(self.embedding_index);
                dbg!(self.patch_index);
            }
            fn sum_head_obj(&mut self) -> u64 {
                if !self.embedding_is_clean {
                    self.clean_embedding_bit();
                }
                let pipeline = Arc::new({
                    let shader = $head_obj_update_mod_name::Shader::load(self.device.clone()).unwrap();
                    ComputePipeline::new(self.device.clone(), &shader.main_entry_point(), &()).unwrap()
                });
                let set = Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(self.cache_buffer.clone())
                        .unwrap()
                        .add_buffer(self.obj_sums_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                let push_constants = $head_obj_update_mod_name::ty::PushConstantData {
                    head: self.head,
                    batch_size: self.sum_batch_size as u32,
                };
                let n_workgroups = ((self.n_examples as f64 / 256f64) / self.sum_batch_size as f64).ceil() as u32;
                let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family())
                    .unwrap()
                    .dispatch([n_workgroups, 1, 1], pipeline.clone(), set.clone(), push_constants)
                    .unwrap()
                    .build()
                    .unwrap();

                let future = sync::now(self.device.clone())
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap();
                future.wait(None).unwrap();

                let start = PreciseTime::now();
                let data_buffer_content = self.obj_sums_buffer.read().unwrap();
                let gpu_sum: u64 = data_buffer_content.par_iter().map(|&x| x as u64).sum();
                gpu_sum
            }
        }
        impl<InputPatch: 'static + BitLen + GetMirroredWords + GetWord + Copy + Sync + Send + NewMirror3x3FastCache<[[InputPatch; 16]; $embedding_len], [u32; $embedding_len]> + WordLen>
            ObjectiveEval<InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>
            for FastCacheVKObjEval<$shader_mod_name::Layout, InputPatch, [[InputPatch; 16]; $embedding_len], [u32; $embedding_len]>
        where
            [[InputPatch; 16]; $embedding_len]: FlipBitIndexed,
        {
            fn flip_weights_bit(&mut self, o: usize, i: usize) {
                if o != self.embedding_index {
                    println!("transitioning to embedding bit {:?}", o);
                    self.replace_cache_parts(o);
                }
                let new_patch_index = i / 32;
                if new_patch_index != self.patch_index {
                    self.transition_input_word(new_patch_index);
                    if (new_patch_index + 1) < InputPatch::WORD_LEN {
                        self.start_prepare_input_words_buffer(new_patch_index + 1);
                    }
                }
                self.weights.flip_bit_indexed(o, i);
                self.embedding_is_clean = false;
            }
            fn flip_head_bit(&mut self, o: usize, i: usize) {
                self.head[o].flip_bit(i);
                if !self.embedding_is_clean {
                    self.clean_embedding_bit();
                }
            }
            fn obj(&mut self) -> u64 {
                if self.embedding_is_clean {
                    self.sum_head_obj()
                } else {
                    self.sum_obj()
                }
            }
        }
    };
}

impl_fastexamplemirrored_over_embedding!(
    fast_obj_update_e1,
    transition_input_word_e1,
    clean_embedding_bit_e1,
    head_obj_update_e1,
    replace_cache_parts_e1,
    1
);
//impl_fastexamplemirrored_over_embedding!(
//    fast_obj_update_e2,
//    transition_input_word_e2,
//    clean_embedding_bit_e2,
//    head_obj_update_e2,
//    replace_cache_parts_e2,
//    2
//);

//const N_EXAMPLES: usize = 50000 * 900;
//const N_EXAMPLES: usize = 65536;
//const N_EXAMPLES: usize = 104729;
//const BATCH_SIZE: usize = 1;
const BATCH_SIZE: usize = 5 * 5 * 5 * 3 * 3;

const INPUT_LEN: usize = 1;
const EMBEDDING_LEN: usize = 1;

// 3
//[src/bin/fast_shader_demo.rs:618] cur_obj as f64 / N_EXAMPLES as f64 = 0.13022181057777693
//one pass gpu mills: PT236.899094380S

// 1
//[src/bin/fast_shader_demo.rs:618] cur_obj as f64 / N_EXAMPLES as f64 = 0.1479914827793639
//one pass gpu mills: PT67.013053554S

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(1);

    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    let cifar_examples = cifar::load_images_from_base(cifar_base_path, 50_000);

    let examples = {
        let mut patches: Vec<(u8, [[[u32; INPUT_LEN]; 3]; 3])> = cifar_examples
            .iter()
            .map(|(class, image)| iter::repeat(*class as u8).zip(image.patches()))
            .flatten()
            .collect();
        // and shuffle them
        patches.shuffle(&mut rng);
        patches
    };

    let mut full_weights: [[[[[u32; INPUT_LEN]; 3]; 3]; 16]; EMBEDDING_LEN] = rng.gen();
    let full_head: [[u32; EMBEDDING_LEN]; 10] = rng.gen();
    //let examples: Vec<_> = (0..N_EXAMPLES as u32).map(|_| (rng.gen_range(0, 10), rng.gen())).collect();

    let mut fast_cache = FastCacheVKObjEval::<_, _, _, [u32; EMBEDDING_LEN]>::new_vk_fast_cache(
        &examples,
        &full_weights,
        &full_head,
        0,
        0,
        BATCH_SIZE,
    );

    let start = PreciseTime::now();
    let mut cur_obj = fast_cache.obj();
    for he in 0..(16 * EMBEDDING_LEN) {
        for c in 0..10 {
            fast_cache.flip_head_bit(c, he);
            let new_obj = fast_cache.obj();
            if new_obj > cur_obj {
                cur_obj = new_obj;
                dbg!(cur_obj as f64 / examples.len() as f64);
            } else {
                fast_cache.flip_head_bit(c, he);
            }
        }
    }
    println!(
        "head pass gpu mills: {}",
        start.to(PreciseTime::now()).num_milliseconds()
    );

    for e in 0..(16 * EMBEDDING_LEN) {
        for he in 0..(16 * EMBEDDING_LEN) {
            for c in 0..10 {
                fast_cache.flip_head_bit(c, he);
                let new_obj = fast_cache.obj();
                if new_obj > cur_obj {
                    cur_obj = new_obj;
                    dbg!(cur_obj as f64 / examples.len() as f64);
                } else {
                    fast_cache.flip_head_bit(c, he);
                }
            }
        }
        for i in 0..(9 * 32 * INPUT_LEN) {
            fast_cache.flip_weights_bit(e, i);
            let new_obj = fast_cache.obj();
            if new_obj > cur_obj {
                cur_obj = new_obj;
                dbg!(cur_obj as f64 / examples.len() as f64);
            } else {
                //println!("reverting", );
                fast_cache.flip_weights_bit(e, i);
            }
        }
    }

    println!("one pass gpu mills: {}", start.to(PreciseTime::now()));
    println!(
        "one pass gpu mills: {}",
        start.to(PreciseTime::now()).num_milliseconds()
    );
}
