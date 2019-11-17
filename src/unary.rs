use crate::bits::{b32, b8};

macro_rules! to_unary {
    ($name:ident, $type:ty, $len:expr) => {
        fn $name(input: u8) -> $type {
            !((!0) << (input / (256 / $len) as u8))
        }
    };
}

to_unary!(to_2, u8, 2);
to_unary!(to_3, u8, 3);
to_unary!(to_10, u32, 10);
to_unary!(to_32, u32, 32);

pub trait ToUnary<B> {
    fn to_unary(&self) -> B;
}

impl ToUnary<b8> for [u8; 3] {
    fn to_unary(&self) -> b8 {
        b8(to_3(self[0]) as u8 | ((to_3(self[1]) as u8) << 3) | ((to_2(self[2]) as u8) << 6))
    }
}

impl ToUnary<b32> for [u8; 3] {
    fn to_unary(&self) -> b32 {
        b32(to_10(self[0]) as u32
            | ((to_10(self[1]) as u32) << 10)
            | ((to_10(self[2]) as u32) << 20))
    }
}

impl ToUnary<[b32; 3]> for [u8; 3] {
    fn to_unary(&self) -> [b32; 3] {
        [
            b32(to_32(self[0])),
            b32(to_32(self[1])),
            b32(to_32(self[2])),
        ]
    }
}

pub trait NormalizeAndBitpack<O> {
    fn normalize_and_bitpack(&self) -> O;
}

impl<T: Copy> NormalizeAndBitpack<[[T; 3]; 3]> for [[T; 3]; 3] {
    #[inline(always)]
    fn normalize_and_bitpack(&self) -> [[T; 3]; 3] {
        *self
    }
}

// slide the min to 0
impl<T> NormalizeAndBitpack<[[T; 3]; 3]> for [[[u8; 3]; 3]; 3]
where
    [u8; 3]: ToUnary<T>,
    [[T; 3]; 3]: Default,
{
    fn normalize_and_bitpack(&self) -> [[T; 3]; 3] {
        let mut mins = [255_u8; 3];
        for x in 0..3 {
            for y in 0..3 {
                for c in 0..3 {
                    mins[c] = self[x][y][c].min(mins[c]);
                }
            }
        }
        let mut target = <[[T; 3]; 3]>::default();
        for x in 0..3 {
            for y in 0..3 {
                let mut pixel = [0_u8; 3];
                for c in 0..3 {
                    pixel[c] = self[x][y][c] - mins[c];
                }
                target[x][y] = pixel.to_unary();
            }
        }
        target
    }
}
