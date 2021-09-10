use bnn::bits::b64;
use bnn::ecc::ExpandByte;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rand::SeedableRng;
use rand_hc::Hc128Rng;

fn main() {
    <[b64; 4]>::bruteforce_table();
    //dbg!(Hc128Rng::seed_from_u64(17246404).gen::<[u64; 256]>());
    //dbg!(Hc128Rng::seed_from_u64(629745).gen::<[[u64; 2]; 256]>());
    //dbg!(Hc128Rng::seed_from_u64(1902564).gen::<[[u64; 4]; 256]>());
    //dbg!(Hc128Rng::seed_from_u64(1446115).gen::<[[b64; 8]; 256]>());
}

/*
| T | seed | dist |
| - | ---- | ---- |
| b64 | 17246404 | 19 |
| [b64; 2] | 629745 | 46 |
| [b64; 4] | 1902564 | 102 |
| [b64; 8] | 34430 | 218 |
*/
