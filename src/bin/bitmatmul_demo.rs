extern crate bitnn;

use bitnn::layers::{Layer2D, Patch, Pool2x2, WeightsMatrix};

trait ArrayPack<I> {
    fn from_vec_padded(&Vec<I>) -> Self;
    fn from_vec_cropped(&Vec<I>) -> Self;
}

macro_rules! array_pack_trait {
    ($len:expr) => {
        impl<I: Default + Copy> ArrayPack<I> for [I; $len] {
            fn from_vec_padded(input: &Vec<I>) -> [I; $len] {
                if input.len() > $len {
                    panic!("can't fit a vec of len {:} into an array of len {:}. Consider making the vec shorter or use a longer array.", input.len(), $len);
                }
                let mut output = [I::default(); $len];
                for (i, elem) in input.iter().enumerate() {
                    output[i] = *elem;
                }
                output
            }
            fn from_vec_cropped(input: &Vec<I>) -> [I; $len] {
                if input.len() < $len {
                    panic!("can't get an array of len {:} from an array of len {:}. Consider making the vec longer or use a smaller array.", $len, input.len());
                }
                let mut output = [I::default(); $len];
                for i in 0..$len {
                    output[i] = input[i];
                }
                output
            }
        }
    };
}
array_pack_trait!(8);
array_pack_trait!(16);
array_pack_trait!(32);
array_pack_trait!(64);
array_pack_trait!(128);


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
