use crate::bits::b64;
use crate::count_bits::masked_hamming_dist;

pub fn apply(weights: &[[(([b64; 8], [b64; 8]), u32); 64]; 4], input: &[b64; 8]) -> [b64; 4] {
    let mut target = [b64(0); 4];
    for w in 0..4 {
        for b in 0..64 {
            let sign = masked_hamming_dist(input, &weights[w][b].0) > weights[w][b].1;
            target[w].set_bit_in_place(b, sign);
        }
    }
    target
}
