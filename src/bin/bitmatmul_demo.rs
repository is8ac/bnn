extern crate bitnn;

use bitnn::layers::{Patch, WeightsMatrix};

trait Layer2D<I, IP: Patch + Default + Copy, O, OP: Patch> {
    fn conv_3x3<W: WeightsMatrix<[[IP; 3]; 3], OP>>(&self, &W) -> O;
    fn conv_1x1<W: WeightsMatrix<IP, OP>>(&self, &W) -> O;
}

macro_rules! layer2d_trait {
    ($x_len:expr, $y_len:expr) => {
        impl<IP: Patch + Copy + Default, OP: Patch + Default + Copy> Layer2D<[[IP; $y_len]; $x_len], IP, [[OP; $y_len]; $x_len], OP> for [[IP; $y_len]; $x_len] {
            fn conv_3x3<W: WeightsMatrix<[[IP; 3]; 3], OP>>(&self, weights: &W) -> [[OP; $y_len]; $x_len] {
                let mut output = [[OP::default(); $y_len]; $x_len];
                for x in 0..($x_len - 2) {
                    for y in 0..($y_len - 2) {
                        let patch = [
                            [self[x + 0][y + 0], self[x + 1][y + 1], self[x + 2][y + 2]],
                            [self[x + 0][y + 0], self[x + 1][y + 1], self[x + 2][y + 2]],
                            [self[x + 0][y + 0], self[x + 1][y + 1], self[x + 2][y + 2]],
                        ];
                        output[x + 1][y + 1] = weights.mul(&patch);
                    }
                }
                output
            }
            fn conv_1x1<W: WeightsMatrix<IP, OP>>(&self, weights: &W) -> [[OP; $y_len]; $x_len] {
                let mut output = [[OP::default(); $y_len]; $x_len];
                for x in 0..$x_len {
                    for y in 0..$y_len {
                        output[x][y] = weights.mul(&self[x][y]);
                    }
                }
                output
            }
        }
    };
}

layer2d_trait!(4, 4);
layer2d_trait!(8, 8);
layer2d_trait!(16, 16);
layer2d_trait!(32, 32);

trait Pool2x2<I, IP: Patch, O> {
    fn or_pool_2x2(&self) -> O;
}

macro_rules! or_pool2x2_trait {
    ($x_len:expr, $y_len:expr) => {
        impl<P: Patch + Copy + Default> Pool2x2<[[P; $y_len]; $x_len], P, [[P; ($y_len / 2)]; ($x_len / 2)]> for [[P; $y_len]; $x_len] {
            fn or_pool_2x2(&self) -> [[P; ($y_len / 2)]; ($x_len / 2)] {
                let mut pooled = [[P::default(); $y_len / 2]; $x_len / 2];
                for x in 0..($x_len / 2) {
                    let x_base = x * 2;
                    for y in 0..($y_len / 2) {
                        let y_base = y * 2;
                        pooled[x][y] = self[x_base + 0][y_base + 0]
                            .bit_or(&self[x_base + 0][y_base + 1])
                            .bit_or(&self[x_base + 1][y_base + 0])
                            .bit_or(&self[x_base + 1][y_base + 1]);
                    }
                }
                pooled
            }
        }
    };
}

or_pool2x2_trait!(4, 4);
or_pool2x2_trait!(8, 8);
or_pool2x2_trait!(16, 16);
or_pool2x2_trait!(32, 32);
or_pool2x2_trait!(64, 64);

fn main() {
    let input: [[u8; 32]; 32] = [[0b1001_1010u8; 32]; 32];

    let weights3x3 = [([[0b1001_1001u8; 3]; 3], 3u32); 16];
    let weights1x1 = [(0b1001_1001u16, 0u32); 32];
    let fc_weights = [([[0b1001_1010u32; 8]; 8], 0u32); 128];

    let flat_fc = input.conv_3x3(&weights3x3).or_pool_2x2().conv_1x1(&weights1x1).or_pool_2x2().mul(&fc_weights);
    println!("{:0128b}", flat_fc);
}
