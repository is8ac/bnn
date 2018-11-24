extern crate bitnn;

use bitnn::layers::{Layer2D, Patch, Pool2x2, WeightsMatrix, ArrayPack};


fn main() {
    let packed = <[u64; 16]>::from_vec_padded(&(0..15).map(|i|i + 5).collect());
    let packed = <[u64; 16]>::from_vec_cropped(&(0..17).map(|i|i + 5).collect());
    println!("{:?}", packed);
    let input: [[u8; 32]; 32] = [[0b1001_1010u8; 32]; 32];

    let weights3x3 = [([[0b1001_1001u8; 3]; 3], 3u32); 16];
    let weights1x1 = [(0b1001_1001u16, 0u32); 32];
    let fc_weights = [([[0b1001_1010u32; 8]; 8], 0u32); 128];

    let flat_fc = input.conv_3x3(&weights3x3).or_pool_2x2().conv_1x1(&weights1x1).or_pool_2x2().mul(&fc_weights);
    println!("{:0128b}", flat_fc);
}
