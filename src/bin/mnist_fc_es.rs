extern crate bitnn;
extern crate rayon;
extern crate time;
use rayon::prelude::*;

use bitnn::datasets::mnist;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, pack_3x3, pixelmap, unary, Patch};
use time::PreciseTime;

const TRAIN_SIZE: usize = 60_000;

struct ReadoutHead10<T: Patch> {
    weights: [T; 10],
    biases: [i32; 10],
}

impl<T: Patch + Default + Copy + Sync + Send> ReadoutHead10<T> {
    fn acc(&self, examples: &Vec<(usize, T)>) -> f64 {
        let test_correct: u64 = examples
            .par_iter()
            .map(|input| {
                (input.0 == self
                    .weights
                    .iter()
                    .zip(self.biases.iter())
                    .map(|(base_point, bias)| base_point.hamming_distance(&input.1) as i32 - bias)
                    .enumerate()
                    .max_by_key(|(_, dist)| *dist)
                    .unwrap()
                    .0) as u64
            }).sum();
        test_correct as f64 / examples.len() as f64
    }
    fn new_from_split(examples: &Vec<(usize, T)>) -> Self {
        let by_class = featuregen::split_by_label(&examples, 10);
        let mut readout = ReadoutHead10 {
            weights: [T::default(); 10],
            biases: [0i32; 10],
        };
        for class in 0..10 {
            let grads = featuregen::grads_one_shard(&by_class, class);
            let sign_bits: Vec<bool> = grads.iter().map(|x| *x > 0f64).collect();
            readout.weights[class] = T::bitpack(&sign_bits);
            let sum_activation: u64 = examples.iter().map(|(_, input)| readout.weights[class].hamming_distance(&input) as u64).sum();
            readout.biases[class] = (sum_activation as f64 / examples.len() as f64) as i32;
        }
        readout
    }
    // filter cases:
    // - mut_class == targ_class: true
    // -
    fn bitwise_ascend_acc(&mut self, examples: &Vec<(usize, T)>) {
        for mut_class in 0..10 {
            let mut activation_diffs: Vec<(T, i32, bool)> = examples
                .iter()
                .map(|(targ_class, input)| {
                    let mut activations: Vec<i32> = self
                        .weights
                        .iter()
                        .zip(self.biases.iter())
                        .map(|(base_point, bias)| base_point.hamming_distance(&input) as i32 - bias)
                        .collect();

                    let targ_act = activations[*targ_class]; // the activation for target class of this example.
                    let mut_act = activations[mut_class]; // the activation which we are mutating.
                    activations[*targ_class] = -10000;
                    activations[mut_class] = -10000;
                    let max_other_activations = activations.iter().max().unwrap(); // the max activation of all the classes not in the target class or mut class.
                    let diff = {
                        if *targ_class == mut_class {
                            mut_act - max_other_activations
                        } else {
                            mut_act - targ_act
                        }
                    };
                    (input, diff, *targ_class == mut_class, (targ_act > *max_other_activations) | (*targ_class == mut_class)) // diff betwene the activation of the
                }).filter(|(_, _, _, keep)| *keep)
                .map(|(input, diff, sign, _)| (*input, diff, sign))
                .collect();

            println!("len activations: {:?}", activation_diffs.len());
            // note that this sum correct is not the true acc, it is working on the subset that can be made correct or incorrect by this activation.
            let mut sum_correct: i64 = activation_diffs
                .par_iter()
                .map(|(_, diff, sign)| {
                    if *sign {
                        // if we want the mut_act to be bigger,
                        *diff > 0 // count those which are bigger,
                    } else {
                        // otherwise,
                        *diff < 0 // count those which are smaller.
                    }
                } as i64).sum();
            println!("acc: {:?}%", (sum_correct as f64 / examples.len() as f64) * 100f64);
            for b in 0..T::bit_len() {
                // the new weights bit
                let new_weights_bit = !self.weights[mut_class].get_bit(b);
                // if we were to flip the bit of the weights,
                let new_sum_correct: i64 = activation_diffs
                    .par_iter()
                    .map(|(input, diff, sign)| {
                        // new diff is the diff after flipping the weights bit
                        let new_diff = {
                            if (input.get_bit(b) ^ new_weights_bit) {
                                // flipping the bit would make mut_act larger
                                diff + 2
                            } else {
                                diff - 2
                            }
                        };
                        // do we want mut_act to be smaller or larger?
                        (if *sign { new_diff > 0 } else { new_diff < 0 }) as i64
                    }).sum();
                if new_sum_correct > sum_correct {
                    println!("new: {:?} {:?}", new_sum_correct, sum_correct);
                    sum_correct = new_sum_correct;
                    // actually flip the bit
                    self.weights[mut_class].flip_bit(b);
                    // now update each
                    for i in activation_diffs.iter_mut() {
                        if i.0.get_bit(b) ^ new_weights_bit {
                            i.1 += 2;
                        } else {
                            i.1 -= 2;
                        }
                    }
                }
            }
            println!("true acc: {:?}%", self.acc(&examples) * 100f64);
        }
    }
}

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, _)> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    let mut readout = ReadoutHead10::new_from_split(&examples);

    for i in 0..10 {
        println!("round {:}", i);
        let start = PreciseTime::now();
        readout.bitwise_ascend_acc(&examples);
        println!("{}", start.to(PreciseTime::now()));
    }

    let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
    let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
    let test_examples: Vec<(usize, _)> = test_labels.iter().zip(test_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

    let accuracy = readout.acc(&test_examples);
    println!("acc: {:?}%", accuracy * 100f64);
}
// 86.6%
// 5
// PT16.23
