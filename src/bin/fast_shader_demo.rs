extern crate bitnn;
extern crate rand;
extern crate time;
use bitnn::layers::{Apply, MirrorHammingDistance};
use bitnn::objective_eval::IsCorrect;
use bitnn::objective_eval::{ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEvalCreator};
use bitnn::{BitLen, HammingDistance};
use bitnn::{FlipBit, GetMirroredWords, GetPatch, GetWord, WordLen};
use rand::prelude::*;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
extern crate vulkano;
extern crate vulkano_shaders;
use rayon::prelude::*;
use std::marker::PhantomData;

use time::PreciseTime;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
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

mod transition_input_word {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_input_word_trans.glsl",
    }
}

// Note that when embedding word len get to be ~8, it may be worth switching to cached head partial sum so as to make head obj linear with embedding size.
// However while it is small, the memory overhead of the 10 nonconstant partial sums is probably not worth the compute cost savings.
// This is for 10 classes.
// For 100 classes, the transition point will be far larger and it's probably not worth it.
#[derive(Debug, Copy, Clone)]
pub struct FastExampleMirrored<Embedding> {
    input_word: [u32; 2],         // a pair of 32 bits. This is mirrored, so we need a normal and a flipped input word.
    input_partial_sums: [u32; 2], // a pair of integers, one for normal one for flipped.
    embedding: Embedding,
    true_class: u32,
}

pub trait NewMirror3x3FastCache<InputPatch, Weights, Embedding> {
    fn new(input: &InputPatch, weights: &Weights, class: usize, embedding_word: usize, embedding_bit: usize, patch_index: usize) -> Self;
}

macro_rules! impl_NewMirror3x3FastCache_for_FastExampleMirrored {
    ($input_len:expr, $output_len:expr) => {
        impl NewMirror3x3FastCache<[[[u32; $input_len]; 3]; 3], [[[[[u32; $input_len]; 3]; 3]; 16]; $output_len], [u32; $output_len]>
            for FastExampleMirrored<[u32; $output_len]>
        {
            fn new(
                input: &[[[u32; $input_len]; 3]; 3],
                weights: &[[[[[u32; $input_len]; 3]; 3]; 16]; $output_len],
                class: usize,
                embedding_word: usize,
                embedding_bit: usize,
                patch_index: usize,
            ) -> FastExampleMirrored<[u32; $output_len]> {
                let embedding = weights.apply(input);
                let input_words = input.get_mirrored_words(patch_index);
                FastExampleMirrored {
                    input_word: input_words,
                    input_partial_sums: [
                        weights[embedding_word][embedding_bit].normal_hamming_distance(&input)
                            - weights[embedding_word][embedding_bit]
                                .get_word(patch_index)
                                .hamming_distance(&input_words[0]),
                        weights[embedding_word][embedding_bit].fliped_hamming_distance(&input)
                            - weights[embedding_word][embedding_bit]
                                .get_word(patch_index)
                                .hamming_distance(&input_words[1]),
                    ],
                    embedding: embedding,
                    true_class: class as u32,
                }
            }
            //fn transition_input_word(
            //    &mut self,
            //    input: [[[u32; 4]; 3]; 3],
            //    new_input_pixel_word: usize,
            //    cur_weight_word: u32,
            //    new_weight_word: u32,
            //    input_x: usize,
            //    input_y: usize,
            //    new_input_word_index: usize,
            //) {
            //    self.input_partial_sums[0] += cur_weight_word.hamming_distance(&self.input_word[0]);
            //    self.input_partial_sums[1] += cur_weight_word.hamming_distance(&self.input_word[1]);

            //    self.input_word = if input_x == 0 {
            //        [input[0][input_y][new_input_pixel_word], input[2][input_y][new_input_pixel_word]]
            //    } else if input_x == 2 {
            //        [input[2][input_y][new_input_pixel_word], input[0][input_y][new_input_pixel_word]]
            //    } else {
            //        [input[1][input_y][new_input_pixel_word], input[1][input_y][new_input_pixel_word]]
            //    };

            //    self.input_partial_sums[0] -= new_weight_word.hamming_distance(&self.input_word[0]);
            //    self.input_partial_sums[1] -= new_weight_word.hamming_distance(&self.input_word[1]);
            //}
        }
    };
}
//impl_NewMirror3x3FastCache_for_FastExampleMirrored!(4, 2);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(3, 2);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(2, 2);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(1, 1);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(2, 1);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(3, 1);
//impl_NewMirror3x3FastCache_for_FastExampleMirrored!(4, 1);

pub struct FastCacheVKObjEval<ShaderLayout, InputPatch, Weights, Embedding> {
    n_examples: usize,
    examples: Vec<InputPatch>,
    sum_batch_size: usize,
    weights_patch: InputPatch,
    head: [Embedding; 10],
    input_patch_type: PhantomData<InputPatch>,
    embedding_type: PhantomData<Embedding>,
    weights_type: PhantomData<Weights>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    obj_sums_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    cache_buffer: Arc<ImmutableBuffer<[FastExampleMirrored<Embedding>]>>,
    update_pipeline: Arc<ComputePipeline<PipelineLayout<ShaderLayout>>>,
    update_descriptor_set: Arc<
        vulkano::descriptor::descriptor_set::PersistentDescriptorSet<
            std::sync::Arc<vulkano::pipeline::ComputePipeline<vulkano::descriptor::pipeline_layout::PipelineLayout<ShaderLayout>>>,
            (
                (
                    (),
                    vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf<
                        std::sync::Arc<vulkano::buffer::ImmutableBuffer<[FastExampleMirrored<Embedding>]>>,
                    >,
                ),
                vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf<std::sync::Arc<vulkano::buffer::CpuAccessibleBuffer<[u32]>>>,
            ),
        >,
    >,
    embedding_word_index: usize,
    embedding_bit_index: usize,
    patch_index: usize,
    //weights_patch_word: u32,
}

macro_rules! impl_fastexamplemirrored_over_embedding {
    ($shader_mod_name:ident, $embedding_len:expr) => {
        impl<InputPatch: Sync + GetWord + GetMirroredWords + BitLen + Copy, Weights: Sync + GetPatch<InputPatch>>
            FastCacheVKObjEval<$shader_mod_name::Layout, InputPatch, Weights, [u32; $embedding_len]>
        where
            FastExampleMirrored<[u32; $embedding_len]>: NewMirror3x3FastCache<InputPatch, Weights, [u32; $embedding_len]>,
        {
            fn new_vk_fast_cache(
                examples: &Vec<(u8, InputPatch)>,
                weights: &Weights,
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
                    let data_iter = (0..(N_EXAMPLES as f64 / batch_size as f64).ceil() as u32).map(|_| 0u32);
                    CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data_iter).unwrap()
                };

                let data_vec: Vec<_> = examples
                    .par_iter()
                    .map(|(class, patch)| {
                        FastExampleMirrored::new(
                            &patch,
                            weights,
                            *class as usize,
                            embedding_index / 16,
                            embedding_index % 16,
                            patch_index,
                        )
                    })
                    .collect();
                let (cache_buffer, _) = ImmutableBuffer::from_iter(data_vec.iter().cloned(), BufferUsage::all(), queue.clone()).unwrap();

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
                    examples: examples.iter().map(|(_, patch)| patch).cloned().collect(),
                    sum_batch_size: batch_size,
                    weights_patch: weights.get_patch(embedding_index),
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
                    embedding_word_index: embedding_index / 16,
                    embedding_bit_index: embedding_index % 16,
                    patch_index: patch_index,
                    //weights_patch_word: weights.get_patch(embedding_index).get_word(0),
                }
            }

            fn sum_obj(&self) -> u64 {
                let pa_push_constants = $shader_mod_name::ty::PushConstantData {
                    head: self.head,
                    embedding_bit_index: self.embedding_bit_index as u32,
                    embedding_word: self.embedding_word_index as u32,
                    threshold: (<InputPatch>::BIT_LEN / 2) as u32,
                    weights_word: self.weights_patch.get_word(self.patch_index),
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

                let start = PreciseTime::now();
                let data_buffer_content = self.obj_sums_buffer.read().unwrap();
                let gpu_sum: u64 = data_buffer_content.par_iter().map(|&x| x as u64).sum();
                gpu_sum
            }
            fn transition_input_word(&mut self, target_weight_index: usize) {
                let start = PreciseTime::now();
                let data_vec: Vec<[u32; 2]> = self
                    .examples
                    .par_iter()
                    .map(|patch| patch.get_mirrored_words(target_weight_index))
                    .collect();
                println!("trans CPU ms: {}", start.to(PreciseTime::now()).num_milliseconds());

                let (new_words_buffer, _) = ImmutableBuffer::from_iter(data_vec.iter().cloned(), BufferUsage::all(), self.queue.clone()).unwrap();
                println!("trans init ms: {}", start.to(PreciseTime::now()).num_milliseconds());

                let start = PreciseTime::now();
                let pipeline = Arc::new({
                    let shader = transition_input_word::Shader::load(self.device.clone()).unwrap();
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
                let push_constants = transition_input_word::ty::PushConstantData {
                    new_weights_word: self.weights_patch.get_word(target_weight_index),
                    old_weights_word: self.weights_patch.get_word(self.patch_index),
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
                println!("gpu trans ms: {}", start.to(PreciseTime::now()).num_milliseconds());
                self.patch_index = target_weight_index;
                //self.weights_patch_word = self.weights_patch.get_word(target_weight_index);
            }
        }
    };
}

impl_fastexamplemirrored_over_embedding!(fast_obj_update_e1, 1);
impl_fastexamplemirrored_over_embedding!(fast_obj_update_e2, 2);

const N_EXAMPLES: usize = 50000 * 900;
//const N_EXAMPLES: usize = 65536 * 16;
//const N_EXAMPLES: usize = 104729;
const BATCH_SIZE: usize = 5 * 5 * 5 * 3 * 3;
const EMBEDDING_BIT_INDEX: usize = 1;

const INPUT_LEN: usize = 2;
const EMBEDDING_LEN: usize = 2;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(1);
    let mut full_weights: [[[[[u32; INPUT_LEN]; 3]; 3]; 16]; EMBEDDING_LEN] = rng.gen();
    let full_head: [[u32; EMBEDDING_LEN]; 10] = rng.gen();
    let examples: Vec<_> = (0..N_EXAMPLES as u32).map(|_| (rng.gen_range(0, 10), rng.gen())).collect();

    let test_eval_creator = TestCPUObjectiveEvalCreator::new();
    let mut test_obj_eval = test_eval_creator.new_obj_eval(&full_weights, &full_head, &examples);

    let mut fast_cache = FastCacheVKObjEval::<_, _, _, [u32; EMBEDDING_LEN]>::new_vk_fast_cache(
        &examples,
        &full_weights,
        &full_head,
        EMBEDDING_BIT_INDEX,
        0,
        BATCH_SIZE,
    );

    let mut cur_obj = fast_cache.sum_obj();
    //let start = PreciseTime::now();
    //for w in 0..(9 * 2) {
    //    fast_cache.transition_input_word(w);
    //    for b in 0..32 {
    //        let i = (w * 32) + b;
    //        fast_cache.weights_patch.flip_bit(i);
    //        let new_obj = fast_cache.sum_obj();
    //        if new_obj > cur_obj {
    //            cur_obj = new_obj;
    //            dbg!(new_obj as f64 / N_EXAMPLES as f64);
    //            test_obj_eval.flip_weights_bit(EMBEDDING_BIT_INDEX, i);
    //        } else {
    //            fast_cache.weights_patch.flip_bit(i);
    //        }
    //    }
    //}

    //println!("embedding bit ms: {}", start.to(PreciseTime::now()).num_milliseconds());

    let fc_gpu_sum = fast_cache.sum_obj();
    let test_obj: u64 = test_obj_eval.obj();
    assert_eq!(fc_gpu_sum, test_obj);
    dbg!(fc_gpu_sum);

    let start = PreciseTime::now();
    fast_cache.transition_input_word(1);
    println!("trans ms: {}", start.to(PreciseTime::now()).num_milliseconds());

    //test_obj_eval.flip_weights_bit(EMBEDDING_BIT_INDEX, 33);
    //fast_cache.weights_patch.flip_bit(33);

    let fc_gpu_sum = fast_cache.sum_obj();
    let test_obj: u64 = test_obj_eval.obj();
    assert_eq!(fc_gpu_sum, test_obj);

    let start = PreciseTime::now();
    println!("starting gpu");
    for i in 0..32 {
        let fc_gpu_sum = fast_cache.sum_obj();
    }
    println!("32 fast gpu mills: {}", start.to(PreciseTime::now()).num_milliseconds());
}
