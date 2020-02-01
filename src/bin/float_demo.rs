use bitnn::bits::{b32, b8, BitArray, BFBVM};
use bitnn::float::{FloatLoss, Mutate, Noise};
use bitnn::shape::Element;
use rand::{Rng, SeedableRng};
use rand_core::RngCore;
use rand_distr::{Distribution, Normal};
use rand_hc::Hc128Rng;

type InputType = [b8; 2];
type HiddenType = [b32; 5];

type WeightsType = <(
    <f32 as Element<<InputType as BitArray>::BitShape>>::Array,
    f32,
) as Element<<HiddenType as BitArray>::BitShape>>::Array;
type ObjectiveType = [(
    <f32 as Element<<HiddenType as BitArray>::BitShape>>::Array,
    f32,
); N_CLASSES];

const N_CLASSES: usize = 5;
const ITERS: usize = 5000;

fn main() {
    let inputs: Vec<(InputType, [u32; N_CLASSES])> = vec![
        ([b8(0b0100_1111), b8(0b0100_1110)], [90, 1, 0, 0, 0]),
        ([b8(0b1011_1100), b8(0b1011_0100)], [1, 52, 0, 1, 1]),
        ([b8(0b1110_1100), b8(0b1110_0100)], [0, 1, 78, 0, 1]),
        ([b8(0b0010_1010), b8(0b1110_0011)], [0, 0, 2, 0, 63]),
        ([b8(0b0101_1100), b8(0b0101_1010)], [40, 0, 1, 0, 0]),
        ([b8(0b0000_1111), b8(0b0111_0101)], [1, 0, 0, 0, 50]),
        ([b8(0b0101_1100), b8(0b1100_0000)], [0, 0, 0, 90, 0]),
        ([b8(0b1110_0010), b8(0b0101_0101)], [0, 1, 0, 0, 70]),
    ];
    let mutations = {
        let mut rng = Hc128Rng::seed_from_u64(0);
        let mut params = <(WeightsType, ObjectiveType)>::default();

        let mut cur_sum_loss: f64 = inputs
            .iter()
            .map(|(input, counts)| params.1.counts_loss(&params.0.bfbvm(input), counts) as f64)
            .sum();
        dbg!(cur_sum_loss / inputs.len() as f64);

        let normal = Normal::new(0f32, 0.03).unwrap();
        let mut noise: Vec<f32> = (0..WeightsType::NOISE_LEN + ObjectiveType::NOISE_LEN + ITERS)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let mut n_updates = 0;
        let mut mutations = Vec::<usize>::new();
        for i in 0..ITERS {
            let perturbed_params = params.mutate(&noise[i..]);
            let new_sum_loss: f64 = inputs
                .iter()
                .map(|(input, counts)| {
                    let hidden = perturbed_params.0.bfbvm(input);
                    perturbed_params.1.counts_loss(&hidden, counts) as f64
                })
                .sum();
            if new_sum_loss < cur_sum_loss {
                println!("{} {}", i, new_sum_loss / inputs.len() as f64);
                cur_sum_loss = new_sum_loss;
                params = perturbed_params;
                mutations.push(i);
                n_updates += 1;
            }
        }
        dbg!(n_updates);
        let acc = cur_sum_loss / inputs.len() as f64;
        dbg!(acc);
        mutations
    };

    let normal = Normal::new(0f32, 0.03).unwrap();
    let mut rng = Hc128Rng::seed_from_u64(0);
    let mut noise: Vec<f32> = (0..WeightsType::NOISE_LEN + ObjectiveType::NOISE_LEN + ITERS)
        .map(|_| normal.sample(&mut rng))
        .collect();
    let reconstituted_params = mutations.iter().fold(
        <(WeightsType, ObjectiveType)>::default(),
        |prms, mutation| prms.mutate(&noise[*mutation..]),
    );
    let mut cur_sum_loss: f64 = inputs
        .iter()
        .map(|(input, counts)| {
            reconstituted_params
                .1
                .counts_loss(&reconstituted_params.0.bfbvm(input), counts) as f64
        })
        .sum();
    dbg!(cur_sum_loss / inputs.len() as f64);
}
