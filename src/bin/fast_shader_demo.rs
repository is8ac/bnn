extern crate bitnn;
extern crate time;
use bitnn::datasets::cifar;

use bitnn::objective_eval::{
    ObjectiveEval, ObjectiveEvalCreator, VulkanFastCacheObjectiveEvalCreator,
};
use bitnn::ExtractPatches;
use rand::prelude::*;
use rand::SeedableRng;
use rand_hc::Hc128Rng;
use std::iter;
use std::path::Path;

use time::PreciseTime;

const BATCH_SIZE: usize = 5 * 5 * 5 * 3 * 3;

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

    let full_weights: [[[[[u32; INPUT_LEN]; 3]; 3]; 16]; EMBEDDING_LEN] = rng.gen();
    let full_head: [[u32; EMBEDDING_LEN]; 10] = rng.gen();
    //let examples: Vec<_> = (0..N_EXAMPLES as u32).map(|_| (rng.gen_range(0, 10), rng.gen())).collect();

    let eval_creator = VulkanFastCacheObjectiveEvalCreator::new(BATCH_SIZE);
    let mut fast_cache = eval_creator.new_obj_eval(&full_weights, &full_head, &examples);
    //let mut fast_cache = FastCacheVKObjEval::<_, _, _, [u32; EMBEDDING_LEN]>::new_vk_fast_cache(
    //    &examples,
    //    &full_weights,
    //    &full_head,
    //    0,
    //    0,
    //    BATCH_SIZE,
    //);

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
