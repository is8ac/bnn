extern crate bitnn;
extern crate rand;
extern crate time;
use bitnn::layers::{Apply, MirrorHammingDistance};
use bitnn::objective_eval::IsCorrect;
use bitnn::objective_eval::{ObjectiveEval, ObjectiveEvalCreator, TestCPUObjectiveEvalCreator};
use bitnn::{BitLen, HammingDistance};
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

mod fast_obj_update {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/fast_mirror_update.glsl",
    }
}

//mod transition_input_word {
//    vulkano_shaders::shader! {
//        ty: "compute",
//        path: "shaders/fast_mirror_input_word_trans.glsl",
//    }
//}


#[derive(Debug, Copy, Clone)]
pub struct FastExampleMirrored {
    input_word: [u32; 2],         // a pair of 32 bits. This is mirrored, so we need a normal and a flipped input word.
    input_partial_sums: [u32; 2], // a pair of integers, one for normal one for flipped.
    embedding_word: u32,          // 32 bits. This is the bits of the embedding, two bits will be updated acording to the input and weights.
    true_class: u32,
}

impl FastExampleMirrored {
    fn is_good(&self, head_words: &[u32; 10]) -> bool {
        let max_obj = (self.embedding_word ^ head_words[self.true_class as usize]).count_ones();
        let mut is_good = true;
        for c in 0..10 {
            is_good = ((c == self.true_class as usize) | ((self.embedding_word ^ head_words[c]).count_ones() < max_obj)) && is_good;
        }
        is_good
    }
}

pub trait NewMirror3x3FastCache<InputPatch, Weights, Embedding> {
    fn new(
        input: &InputPatch,
        weights: &Weights,
        head: &[Embedding; 10],
        class: usize,
        embedding_word: usize,
        embedding_bit: usize,
        input_x: usize,
        input_y: usize,
        input_pixel_word: usize,
    ) -> FastExampleMirrored;
}

macro_rules! impl_NewMirror3x3FastCache_for_FastExampleMirrored {
    ($input_len:expr, $output_len:expr) => {
        impl NewMirror3x3FastCache<[[[u32; $input_len]; 3]; 3], [[[[[u32; $input_len]; 3]; 3]; 16]; $output_len], [u32; $output_len]>
            for FastExampleMirrored
        {
            fn new(
                input: &[[[u32; $input_len]; 3]; 3],
                weights: &[[[[[u32; $input_len]; 3]; 3]; 16]; $output_len],
                head: &[[u32; $output_len]; 10],
                class: usize,
                embedding_word: usize,
                embedding_bit: usize,
                input_x: usize,
                input_y: usize,
                input_pixel_word: usize,
            ) -> FastExampleMirrored {
                let embedding = weights.apply(input);
                let input_words = if input_x == 0 {
                    [input[0][input_y][input_pixel_word], input[2][input_y][input_pixel_word]]
                } else if input_x == 2 {
                    [input[2][input_y][input_pixel_word], input[0][input_y][input_pixel_word]]
                } else {
                    [input[1][input_y][input_pixel_word], input[1][input_y][input_pixel_word]]
                };
                FastExampleMirrored {
                    input_word: input_words,
                    input_partial_sums: [
                        weights[embedding_word][embedding_bit].normal_hamming_distance(&input)
                            - weights[embedding_word][embedding_bit][input_x][input_y][input_pixel_word].hamming_distance(&input_words[0]),
                        weights[embedding_word][embedding_bit].fliped_hamming_distance(&input)
                            - weights[embedding_word][embedding_bit][input_x][input_y][input_pixel_word].hamming_distance(&input_words[1]),
                    ],
                    embedding_word: embedding[embedding_word],
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
//impl_NewMirror3x3FastCache_for_FastExampleMirrored!(3, 2);
//impl_NewMirror3x3FastCache_for_FastExampleMirrored!(2, 2);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(1, 1);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(2, 1);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(3, 1);
impl_NewMirror3x3FastCache_for_FastExampleMirrored!(4, 1);

pub struct FastCacheVKObjEval<InputPatch, Weights, Embedding> {
    n_examples: usize,
    sum_batch_size: usize,
    weights_patch: InputPatch,
    head_words: [u32; 10],
    input_patch_type: PhantomData<InputPatch>,
    embedding_type: PhantomData<Embedding>,
    weights_type: PhantomData<Weights>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    obj_sums_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    update_pipeline: Arc<ComputePipeline<PipelineLayout<fast_obj_update::Layout>>>,
    update_descriptor_set: Arc<
        vulkano::descriptor::descriptor_set::PersistentDescriptorSet<
            std::sync::Arc<vulkano::pipeline::ComputePipeline<vulkano::descriptor::pipeline_layout::PipelineLayout<fast_obj_update::Layout>>>,
            (
                (
                    (),
                    vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf<
                        std::sync::Arc<vulkano::buffer::ImmutableBuffer<[FastExampleMirrored]>>,
                    >,
                ),
                vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuf<std::sync::Arc<vulkano::buffer::CpuAccessibleBuffer<[u32]>>>,
            ),
        >,
    >,
    embedding_word_index: usize,
    embedding_bit_index: usize,
    weights_patch_word: u32,
}

pub trait GetPatch<T> {
    fn get_patch(&self, index: usize) -> T;
}
macro_rules! impl_getpatch_for_weights {
    ($words:expr) => {
        impl<T: Copy> GetPatch<T> for [[T; 16]; $words] {
            fn get_patch(&self, index: usize) -> T {
                self[index / 16][index % 16]
            }
        }
    };
}
impl_getpatch_for_weights!(1);
impl_getpatch_for_weights!(2);
impl_getpatch_for_weights!(3);
impl_getpatch_for_weights!(4);

pub trait GetWord {
    fn get_word(&self, i: usize) -> u32;
    const WORD_LEN: usize;
}

impl GetWord for u32 {
    fn get_word(&self, i: usize) -> u32 {
        if i != 0 {
            panic!("there is is only 1 word in a u32!")
        }
        *self
    }
    const WORD_LEN: usize = 1;
}

macro_rules! impl_getword_for_array {
    ($len:expr) => {
        impl<T: GetWord> GetWord for [T; $len] {
            fn get_word(&self, i: usize) -> u32 {
                self[i / T::WORD_LEN].get_word(i % T::WORD_LEN)
            }
            const WORD_LEN: usize = T::WORD_LEN * $len;
        }
    };
}
impl_getword_for_array!(1);
impl_getword_for_array!(2);
impl_getword_for_array!(3);
impl_getword_for_array!(4);

pub trait GetHeadWords {
    fn get_head_words(&self, embedding_word: usize) -> [u32; 10];
}

macro_rules! impl_getheadwords_for_head {
    ($len:expr) => {
        impl GetHeadWords for [[u32; $len]; 10] {
            fn get_head_words(&self, embedding_word_index: usize) -> [u32; 10] {
                let mut head_words = [0u32; 10];
                for c in 0..10 {
                    head_words[c] = self[c][embedding_word_index];
                }
                head_words
            }
        }
    };
}

impl_getheadwords_for_head!(1);
//impl_getheadwords_for_head!(2);
//impl_getheadwords_for_head!(3);
//impl_getheadwords_for_head!(4);

impl<InputPatch: Sync + GetWord + BitLen, Weights: Sync + GetPatch<InputPatch>, Embedding: Sync> FastCacheVKObjEval<InputPatch, Weights, Embedding>
where
    FastExampleMirrored: NewMirror3x3FastCache<InputPatch, Weights, Embedding>,
    [Embedding; 10]: GetHeadWords,
{
    fn new_vk_fast_cache(
        examples: &Vec<(u8, InputPatch)>,
        weights: &Weights,
        head: &[Embedding; 10],
        embedding_index: usize,
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
            let shader = fast_obj_update::Shader::load(device.clone()).unwrap();
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
                    &head,
                    *class as usize,
                    embedding_index / 16,
                    embedding_index % 16,
                    0,
                    0,
                    0,
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
            sum_batch_size: batch_size,
            weights_patch: weights.get_patch(embedding_index),
            head_words: head.get_head_words(embedding_index / 16),
            embedding_type: PhantomData,
            input_patch_type: PhantomData,
            weights_type: PhantomData,
            device: device,
            queue: queue,
            obj_sums_buffer: sums_buffer,
            update_pipeline: pipeline,
            update_descriptor_set: set,
            embedding_word_index: embedding_index / 16,
            embedding_bit_index: embedding_index % 16,
            weights_patch_word: weights.get_patch(embedding_index % 16).get_word(embedding_index / 16),
        }
    }

    fn sum_obj(&self) -> u64 {
        let pa_push_constants = fast_obj_update::ty::PushConstantData {
            head_words: self.head_words,
            embedding_bit_index: self.embedding_bit_index as u32,
            threshold: (<InputPatch>::BIT_LEN / 2) as u32,
            weights_word: self.weights_patch_word,
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
    fn transition_input_word(&mut self, target_input_index: usize) {

    }
}

const N_EXAMPLES: usize = 50000 * 900;
//const N_EXAMPLES: usize = 65536 * 16;
//const N_EXAMPLES: usize = 104729;
const BATCH_SIZE: usize = 5 * 5 * 5 * 3 * 3;
const EMBEDDING_BIT_INDEX: usize = 5;

const INPUT_LEN: usize = 1;


fn main() {
    let mut rng = Hc128Rng::seed_from_u64(1);
    let mut full_weights: [[[[[u32; INPUT_LEN]; 3]; 3]; 16]; 1] = rng.gen();
    let full_head: [[u32; 1]; 10] = rng.gen();
    let examples: Vec<_> = (0..N_EXAMPLES as u32).map(|_| (rng.gen_range(0, 10), rng.gen())).collect();

    let mut fast_cache = FastCacheVKObjEval::new_vk_fast_cache(&examples, &full_weights, &full_head, EMBEDDING_BIT_INDEX, BATCH_SIZE);
    let _ = fast_cache.sum_obj();

    let test_eval_creator = TestCPUObjectiveEvalCreator::new();
    let mut test_obj_eval = test_eval_creator.new_obj_eval(&full_weights, &full_head, &examples);

    test_obj_eval.flip_weights_bit(EMBEDDING_BIT_INDEX, 7);
    fast_cache.weights_patch_word ^= 1 << 7;

    let start = PreciseTime::now();
    println!("starting gpu");
    for i in 0..1000 {
        let fc_gpu_sum = fast_cache.sum_obj();
    }
    println!("fast gpu mills: {}", start.to(PreciseTime::now()).num_milliseconds());

    let fc_gpu_sum = fast_cache.sum_obj();
    let test_obj: u64 = test_obj_eval.obj();
    assert_eq!(fc_gpu_sum, test_obj);
    dbg!(fc_gpu_sum);
}
