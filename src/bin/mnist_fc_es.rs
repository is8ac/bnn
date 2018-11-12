extern crate bitnn;
extern crate rayon;
extern crate time;
use rayon::prelude::*;

use bitnn::datasets::mnist;
use bitnn::featuregen;
use bitnn::layers::{bitvecmul, Patch};
use time::PreciseTime;

trait ClassificationModel<T: Send + Sync>: Send + Sync {
    fn classify(&self, &T) -> usize;
    fn acc(&self, examples: &Vec<(usize, T)>) -> f64 {
        let sum_correct: u64 = examples.par_iter().map(|(class, input)| (self.classify(input) == *class) as u64).sum();
        sum_correct as f64 / examples.len() as f64
    }
    fn ascend_acc(&mut self, examples: &Vec<(usize, T)>);
    fn from_examples(&Vec<(usize, T)>) -> Self;
}

pub struct ReadoutHead10<T: Patch> {
    weights: [T; 10],
    biases: [i32; 10],
}

impl<T: Patch + Default + Copy> ReadoutHead10<T> {
    fn foo() {
        println!("bar",);
    }
}

impl<T: Sync + Send + Patch + Copy + Default> ClassificationModel<T> for ReadoutHead10<T> {
    fn classify(&self, input: &T) -> usize {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(base_point, bias)| base_point.hamming_distance(&input) as i32 - bias)
            .enumerate()
            .max_by_key(|(_, dist)| *dist)
            .unwrap()
            .0
    }
    fn ascend_acc(&mut self, examples: &Vec<(usize, T)>) {
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
            for b in 0..T::bit_len() {
                // the new weights bit
                let new_weights_bit = !self.weights[mut_class].get_bit(b);
                // if we were to flip the bit of the weights,
                let new_sum_correct: i64 = activation_diffs
                    .par_iter()
                    .map(|(input, diff, sign)| {
                        // new diff is the diff after flipping the weights bit
                        let new_diff = {
                            if input.get_bit(b) ^ new_weights_bit {
                                // flipping the bit would make mut_act larger
                                diff + 2
                            } else {
                                diff - 2
                            }
                        };
                        // do we want mut_act to be smaller or larger?
                        // same as this statement:
                        //(if *sign { new_diff > 0 } else { new_diff < 0 }) as i64
                        ((*sign ^ (new_diff < 0)) & (new_diff != 0)) as i64
                    }).sum();
                if new_sum_correct > sum_correct {
                    sum_correct = new_sum_correct;
                    // actually flip the bit
                    self.weights[mut_class].flip_bit(b);
                    // now update each
                    activation_diffs
                        .par_iter_mut()
                        .map(|i| {
                            if i.0.get_bit(b) ^ new_weights_bit {
                                i.1 += 2;
                            } else {
                                i.1 -= 2;
                            }
                        }).collect::<Vec<_>>();
                }
            }
        }
    }
    fn from_examples(examples: &Vec<(usize, T)>) -> Self {
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
}

struct HiddenLayer<I: Patch, O: Patch, R: ClassificationModel<O>> {
    weights: Vec<I>,
    biases: Vec<Vec<u32>>,
    dummy_hidden: O,
    next_layer: R,
}

impl<I: Patch + Copy + Default, O: Patch + Copy + Default, R: ClassificationModel<O>> ClassificationModel<I> for HiddenLayer<I, O, R> {
    fn classify(&self, input: &I) -> usize {
        self.next_layer.classify(&self.apply(&input))
    }
    fn ascend_acc(&mut self, examples: &Vec<(usize, I)>) {
        let mut next_layer_examples_bits: Vec<(usize, Vec<bool>)> = examples
            .par_iter()
            .map(|(class, input)| {
                let distances = bitvecmul::vbvm(&self.weights, &input);
                let mut bools = vec![false; O::bit_len()];
                for c in 0..distances.len() {
                    for i in 0..self.biases[0].len() {
                        bools[(c * self.biases[0].len()) + i] = distances[c] > self.biases[c][i];
                    }
                }
                (*class, bools)
            }).collect();
        let next_layer_examples: Vec<(usize, O)> = next_layer_examples_bits.par_iter().map(|(class, bools)| (*class, O::bitpack(&bools.as_slice()))).collect();
        self.next_layer.ascend_acc(&next_layer_examples);
        let mut accuracy = self.next_layer.acc(&next_layer_examples);
        println!("starting layer");
        for o in 0..self.weights.len() {
            for b in 0..I::bit_len() {
                self.weights[o].flip_bit(b);
                let next_layer_examples: Vec<(usize, O)> = next_layer_examples_bits
                    .par_iter_mut()
                    .zip(examples.par_iter())
                    .map(|((class, bools), (_, input))| {
                        let distance = input.hamming_distance(&self.weights[o]);
                        for i in 0..self.biases[0].len() {
                            bools[(o * self.biases[0].len()) + i] = distance > self.biases[o][i];
                        }
                        (*class, O::bitpack(&bools.as_slice()))
                    }).collect();

                let new_accuracy = self.next_layer.acc(&next_layer_examples);
                //println!("new acc: {:?}", new_accuracy);
                if new_accuracy > accuracy {
                    accuracy = new_accuracy;
                    println!("new acc: {:?}", accuracy);
                } else {
                    self.weights[o].flip_bit(b);
                }
            }
        }
        println!("finished layer",);
        let next_layer_examples: Vec<(usize, O)> = next_layer_examples_bits.par_iter().map(|(class, bools)| (*class, O::bitpack(&bools.as_slice()))).collect();
        self.next_layer.ascend_acc(&next_layer_examples);
    }

    fn from_examples(examples: &Vec<(usize, I)>) -> Self {
        let train_inputs = featuregen::split_by_label(&examples, 10);
        let (features_vec, biases) = featuregen::gen_hidden_features(&train_inputs, 4, 6, 2);
        let next_layer_train_examples: Vec<(usize, O)> = examples
            .iter()
            .map(|(class, input)| (*class, featuregen::apply_unary(input, &features_vec, &biases)))
            .collect();

        HiddenLayer {
            weights: features_vec,
            biases: biases,
            dummy_hidden: O::default(),
            next_layer: R::from_examples(&next_layer_train_examples),
        }
    }
}

impl<I: Patch + Default + Copy, O: Default + Copy + Patch, R: ClassificationModel<O>> HiddenLayer<I, O, R> {
    fn apply(&self, input: &I) -> O {
        featuregen::apply_unary(input, &self.weights, &self.biases)
    }
    fn apply_all_examples(&self, examples: &Vec<(usize, I)>) -> Vec<(usize, O)> {
        examples.par_iter().map(|(class, input)| (*class, self.apply(input))).collect()
    }
}

const TRAIN_SIZE: usize = 60_000;

fn main() {
    let images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/train-images-idx3-ubyte"), TRAIN_SIZE);
    let labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/train-labels-idx1-ubyte"), TRAIN_SIZE);
    let examples: Vec<(usize, _)> = labels.iter().zip(images.iter()).map(|(&label, &image)| (label as usize, image)).collect();
    //let mut model: ReadoutHead10<_> = ReadoutHead10::from_examples(&examples);
    let mut model: HiddenLayer<_, [u128; 2], ReadoutHead10<_>> = ClassificationModel::from_examples(&examples);

    let total_start = PreciseTime::now();
    model.ascend_acc(&examples);
    model.ascend_acc(&examples);
    println!("total: {}", total_start.to(PreciseTime::now()));

    let test_images = mnist::load_images_bitpacked(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-images-idx3-ubyte"), 10000);
    let test_labels = mnist::load_labels(&String::from("/home/isaac/big/cache/datasets/mnist/t10k-labels-idx1-ubyte"), 10000);
    let test_examples: Vec<(usize, _)> = test_labels.iter().zip(test_images.iter()).map(|(&label, &image)| (label as usize, image)).collect();

    let accuracy = model.acc(&test_examples);
    println!("acc: {:?}%", accuracy * 100f64);
}
// 87.6%
// 10 iters: PT17.23
// 84.4
