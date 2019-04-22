extern crate bitnn;
extern crate rand;
extern crate rand_hc;
extern crate rayon;
extern crate time;

use bitnn::datasets::mnist;
use bitnn::layers::Apply;
use bitnn::{BitLen, FlipBit, FlipBitIndexed, GetBit, GetPatch, HammingDistance, SetBit};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::path::Path;
use time::PreciseTime;

struct CacheItem<Weights, Embedding> {
    weights_type: PhantomData<Weights>,
    input: [u64; 13],
    class: u8,
    embedding: Embedding,
    output: [u32; 10],
    bit_losses: [f64; 2],
    cur_embedding_act: u32,
}

fn loss_from_scaled_output(input: &[f64; 10], true_class: usize) -> f64 {
    let mut exp = [0f64; 10];
    let mut sum_exp = 0f64;
    for c in 0..10 {
        exp[c] = (input[c]).exp();
        sum_exp += exp[c];
    }
    let sum_loss: f64 = exp
        .iter()
        .enumerate()
        .map(|(c, x)| ((x / sum_exp) - (c == true_class) as u8 as f64).powi(2))
        .sum();
    //sum_loss / 10f64
    sum_loss
}

fn loss_from_embedding<Embedding: HammingDistance + BitLen>(
    embedding: &Embedding,
    head: &[Embedding; 10],
    true_class: usize,
) -> f64 {
    let mut scaled = [0f64; 10];
    for c in 0..10 {
        let sum = head[c].hamming_distance(&embedding);
        scaled[c] = sum as f64 / <Embedding>::BIT_LEN as f64;
    }

    loss_from_scaled_output(&scaled, true_class)
}

impl<Embedding: HammingDistance + BitLen + GetBit + SetBit + Default + Copy, Weights: Apply<[u64; 13], Embedding>>
    CacheItem<Weights, Embedding>
{
    fn compute_embedding(&mut self, weights: &Weights) {
        self.embedding = weights.apply(&self.input);
    }
    fn update_embedding_bit(&mut self, weights_patch: &[u64; 13], embedding_bit_index: usize) {
        self.embedding.set_bit(
            embedding_bit_index,
            weights_patch.hamming_distance(&self.input) > (<[u64; 13]>::BIT_LEN / 2) as u32,
        );
    }
    fn compute_embedding_act(&mut self, weights_patch: &[u64; 13]) {
        self.cur_embedding_act = weights_patch.hamming_distance(&self.input);
    }
    fn compute_output(&mut self, head: &[Embedding; 10]) {
        for c in 0..10 {
            let sum = head[c].hamming_distance(&self.embedding);
            self.output[c] = sum;
        }
    }
    fn compute_bit_losses(&mut self, head: &[Embedding; 10], embedding_bit_index: usize) {
        let mut embedding = self.embedding;
        embedding.set_bit(embedding_bit_index, false);
        self.bit_losses[0] = loss_from_embedding(&embedding, head, self.class as usize);
        embedding.set_bit(embedding_bit_index, true);
        self.bit_losses[1] = loss_from_embedding(&embedding, head, self.class as usize);
    }
    fn update_head_output_from_bit(
        &mut self,
        class_index: usize,
        embedding_bit_index: usize,
        head_bit: bool,
    ) {
        let embedding_bit = self.embedding.bit(embedding_bit_index);
        if head_bit ^ embedding_bit {
            // if !=, changing will decrease activation
            self.output[class_index] -= 1;
        } else {
            self.output[class_index] += 1;
        };
    }
    fn update_cur_embedding_act_from_bit(&mut self, input_bit_index: usize, weights_bit: bool) {
        let input_bit = self.input.bit(input_bit_index);
        if weights_bit ^ input_bit {
            // if !=, changing will decrease activation
            self.cur_embedding_act -= 1;
        } else {
            self.cur_embedding_act += 1;
        };
    }
    fn loss_from_head_bit(
        &self,
        embedding_bit_index: usize,
        class_index: usize,
        head_bit: bool,
    ) -> f64 {
        let input_bit = self.embedding.bit(embedding_bit_index);
        let mut output = self.output;
        if head_bit ^ input_bit {
            // if !=, changing will decrease activation
            output[class_index] = self.output[class_index] - 1;
        } else {
            output[class_index] = self.output[class_index] + 1;
        };
        let mut scaled = [0f64; 10];
        for c in 0..10 {
            scaled[c] = output[c] as f64 / <Embedding>::BIT_LEN as f64;
        }
        loss_from_scaled_output(&scaled, self.class as usize)
    }
    fn loss_from_bit(&self, input_bit_index: usize, weights_bit: bool) -> f64 {
        let input_bit = self.input.bit(input_bit_index);
        let new_act = if weights_bit ^ input_bit {
            // if !=, changing will decrease activation
            self.cur_embedding_act - 1
        } else {
            self.cur_embedding_act + 1
        };
        self.bit_losses[(new_act > (<[u64; 13]>::BIT_LEN / 2) as u32) as usize]
    }
    fn true_loss(&self, weights: &Weights, head: &[Embedding; 10]) -> f64 {
        let embedding = weights.apply(&self.input);
        loss_from_embedding(&embedding, head, self.class as usize)
    }
    fn is_correct(&self) -> bool {
        self.output
            .iter()
            .enumerate()
            .max_by_key(|(_, v)| *v)
            .unwrap()
            .0
            == self.class as usize
    }
    fn new(input: &[u64; 13], class: u8) -> Self {
        CacheItem {
            weights_type: PhantomData,
            input: *input,
            embedding: Embedding::default(),
            output: [0u32; 10],
            bit_losses: [0f64; 2],
            class: class,
            cur_embedding_act: 0u32,
        }
    }
}

pub struct CacheBatch<Weights, Embedding> {
    items: Vec<CacheItem<Weights, Embedding>>,
    weights: Weights,
    head: [Embedding; 10],
    embedding_bit_index: usize,
    class_index: usize,
    embedding_is_clean: bool,
    output_is_clean: bool,
}

impl<
        Weights: GetPatch<[u64; 13]> + Send + FlipBitIndexed + Sync + Copy + Apply<[u64; 13], Embedding>,
        Embedding: Send + Sync + Copy + FlipBit + HammingDistance + GetBit + BitLen + SetBit + Default,
    > CacheBatch<Weights, Embedding>
{
    fn new(
        weights: &Weights,
        head: &[Embedding; 10],
        examples: &Vec<([u64; 13], u8)>,
        embedding_bit_index: usize,
    ) -> Self {
        let weights_patch = weights.get_patch(embedding_bit_index);
        CacheBatch {
            items: examples
                .par_iter()
                .map(|(input, class)| {
                    let mut cache = CacheItem::new(input, *class as u8);
                    cache.compute_embedding(weights);
                    cache.compute_output(head);
                    cache.compute_bit_losses(head, embedding_bit_index);
                    cache.compute_embedding_act(&weights_patch);
                    cache
                })
                .collect(),
            weights: *weights,
            head: *head,
            embedding_bit_index: embedding_bit_index,
            class_index: 0,
            embedding_is_clean: true,
            output_is_clean: true,
        }
    }
    fn head_bit_loss(&mut self, embedding_bit_index: usize, class_index: usize) -> f64 {
        if !self.embedding_is_clean {
            let embedding_bit_index = self.embedding_bit_index;
            let weights_patch = self.weights.get_patch(self.embedding_bit_index);
            let _: Vec<_> = self
                .items
                .par_iter_mut()
                .map(|cache| cache.update_embedding_bit(&weights_patch, embedding_bit_index))
                .collect();
            self.embedding_is_clean = true;
        }
        let head = self.head;
        if !self.output_is_clean {
            let _: Vec<_> = self
                .items
                .par_iter_mut()
                .map(|cache| cache.compute_output(&head))
                .collect();
            self.output_is_clean = true;
        }
        let head_bit = self.head[class_index].bit(embedding_bit_index);
        let sum_loss: f64 = self
            .items
            .par_iter()
            .map(|cache| cache.loss_from_head_bit(embedding_bit_index, class_index, head_bit))
            .sum();
        sum_loss as f64 / self.items.len() as f64
    }
    fn bit_loss(&mut self, input_bit_index: usize) -> f64 {
        let cur_mut_bit_val = self
            .weights
            .get_patch(self.embedding_bit_index)
            .bit(input_bit_index);
        let sum_loss: f64 = self
            .items
            .par_iter()
            .map(|cache| cache.loss_from_bit(input_bit_index, cur_mut_bit_val))
            .sum();
        sum_loss / self.items.len() as f64
    }
    fn true_loss(&mut self) -> f64 {
        let sum_loss: f64 = self
            .items
            .par_iter()
            .map(|cache| cache.true_loss(&self.weights, &self.head))
            .sum();
        sum_loss / self.items.len() as f64
    }
    fn acc(&mut self) -> f64 {
        let head = self.head;
        let sum_iscorrect: u64 = self
            .items
            .par_iter_mut()
            .map(|cache| {
                cache.compute_output(&head);
                cache.is_correct() as u64
            })
            .sum();
        sum_iscorrect as f64 / self.items.len() as f64
    }
    fn flip_weights_bit(&mut self, input_bit_index: usize) {
        self.embedding_is_clean = false;
        self.output_is_clean = false;
        let cur_mut_bit_val = self
            .weights
            .get_patch(self.embedding_bit_index)
            .bit(input_bit_index);
        let _: Vec<_> = self
            .items
            .par_iter_mut()
            .map(|cache| cache.update_cur_embedding_act_from_bit(input_bit_index, cur_mut_bit_val))
            .collect();
        self.weights
            .flip_bit_indexed(self.embedding_bit_index, input_bit_index);
    }
    fn flip_head_bit(&mut self, class: usize, embedding_bit_index: usize) {
        let head_bit = self.head[class].bit(embedding_bit_index);
        let _: Vec<_> = self
            .items
            .par_iter_mut()
            .map(|cache| cache.update_head_output_from_bit(class, embedding_bit_index, head_bit))
            .collect();
        self.output_is_clean = true;
        self.head[class].flip_bit(embedding_bit_index);
    }
    fn transition_embedding_bit(&mut self, new_embedding_bit_index: usize) {
        let old_embedding_bit_index = self.embedding_bit_index;
        let old_weights_patch = self.weights.get_patch(old_embedding_bit_index);
        let new_weights_patch = self.weights.get_patch(new_embedding_bit_index);
        let head = self.head;
        let _: Vec<_> = self
            .items
            .par_iter_mut()
            .map(|cache| {
                cache.update_embedding_bit(&old_weights_patch, old_embedding_bit_index);
                cache.compute_embedding_act(&new_weights_patch);
                cache.compute_bit_losses(&head, new_embedding_bit_index);
            })
            .collect();
        self.embedding_bit_index = new_embedding_bit_index;
    }
}

const N_EXAMPLES: usize = 50_000;

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(8);
    let images = mnist::load_images_bitpacked(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"),
        N_EXAMPLES,
    );
    let classes = mnist::load_labels(
        Path::new("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"),
        N_EXAMPLES,
    );
    let examples: Vec<([u64; 13], u8)> = images
        .iter()
        .cloned()
        .zip(classes.iter().map(|x| *x as u8))
        .collect();

    let weights: [[[u64; 13]; 32]; 4] = rng.gen();
    let head: [[u32; 4]; 10] = rng.gen();

    let mut cache_batch = CacheBatch::new(&weights, &head, &examples, 0);

    let start = PreciseTime::now();
    let mut cur_loss = cache_batch.bit_loss(0);
    for p in 0..3 {
        for e in 0..<[u32; 4]>::BIT_LEN {
            //for c in 0..10 {
            //    for he in 0..<[u32; 4]>::BIT_LEN {
            //        let new_loss = cache_batch.head_bit_loss(he, c);
            //        if new_loss < cur_loss {
            //            cache_batch.flip_head_bit(c, he);
            //            cur_loss = new_loss;
            //            println!("head {} {} {}", c, he, new_loss);
            //        }
            //    }
            //}
            cache_batch.transition_embedding_bit(e);
            for b in 0..<[u64; 13]>::BIT_LEN {
                let new_loss = cache_batch.bit_loss(b);
                //dbg!(new_loss);
                if new_loss < cur_loss {
                    cur_loss = new_loss;
                    cache_batch.flip_weights_bit(b);
                    println!("{} {}: {:?}", e, b, new_loss);
                }
            }
        }
        let avg_acc = cache_batch.acc();
        println!("acc at {}: {}%", p, avg_acc * 100f64);
    }
    let avg_acc = cache_batch.acc();
    println!("acc:  {}%", avg_acc * 100f64);
    println!("time: {}", start.to(PreciseTime::now()));
    //let avg_loss = cache_batch.bit_loss(1);
    //println!("loss: {}", avg_loss);
    // 83.194%
    // full head: 83.294% PT496S
    //   no head: 81.27% PT66S
}
