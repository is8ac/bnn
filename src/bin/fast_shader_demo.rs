extern crate bitnn;
extern crate time;
use bitnn::datasets::cifar;

use bitnn::objective_eval::{
    ObjectiveEval, ObjectiveEvalCreator, VulkanFastCacheObjectiveEvalCreator,
};
use bitnn::{BitLen, ExtractPatches};
use rand::prelude::*;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::iter;
use std::path::Path;

use time::PreciseTime;
//50_000/1: 900
//50_000/2: 1000
//50_000/4: 800
//50_000/8: 500
//50_000/16: 200
//50_000/32: 100
//50_000/64: 60
//50_000/128: 30
//50_000/256: 20

//const RS_BATCH_SIZE: usize = 20;

const INPUT_LEN: usize = 3;
const EMBEDDING_LEN: usize = 2;

// 3
//[src/bin/fast_shader_demo.rs:618] cur_obj as f64 / N_EXAMPLES as f64 = 0.13022181057777693
//one pass gpu mills: PT236.899094380S

// 1
//[src/bin/fast_shader_demo.rs:618] cur_obj as f64 / N_EXAMPLES as f64 = 0.1479914827793639
//one pass gpu mills: PT67.013053554S

fn main() {
    let mut rng = Hc128Rng::seed_from_u64(1);

    let cifar_base_path = Path::new("/home/isaac/big/cache/datasets/cifar-10-batches-bin");
    let cifar_examples = cifar::load_images_from_base(cifar_base_path, 50_000 / 256);

    let patches = {
        let mut patches: Vec<(u8, [[[u32; INPUT_LEN]; 3]; 3])> = cifar_examples
            .iter()
            .map(|(class, image)| iter::repeat(*class as u8).zip(image.patches()))
            .flatten()
            .collect();
        // and shuffle them
        patches.shuffle(&mut rng);
        patches
    };

    let full_weights: [[[[[u32; INPUT_LEN]; 3]; 3]; 16]; EMBEDDING_LEN] = rng.gen();
    let full_head: [[u32; EMBEDDING_LEN]; 10] = rng.gen();
    //let examples: Vec<_> = (0..N_EXAMPLES as u32).map(|_| (rng.gen_range(0, 10), rng.gen())).collect();
    //let rs_batch_size = patches.len() / (56 * 64 * 2);
    let eval_creator = VulkanFastCacheObjectiveEvalCreator::new();
    let mut fast_cache = eval_creator.new_obj_eval(&full_weights, &full_head, &patches);

    let start = PreciseTime::now();
    for b in 0..<[[[u32; INPUT_LEN]; 3]; 3]>::BIT_LEN {
        fast_cache.flip_weights_bit(0, b);
        let _ = fast_cache.obj();
    }
    let n_ms = start.to(PreciseTime::now()).num_milliseconds();
    println!("one patch word mills: {}", n_ms);
    dbg!(patches.len());
    dbg!(n_ms as f64 / <[[[u32; INPUT_LEN]; 3]; 3]>::BIT_LEN as f64);
}
