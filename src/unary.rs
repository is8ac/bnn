use crate::bits::{b32, b8};

pub fn u8x3_to_b32x3(input: [u8; 3]) -> [b32; 3] {
    [
        b32(to_32(input[0])),
        b32(to_32(input[1])),
        b32(to_32(input[2])),
    ]
}

pub fn u8x3_to_b32(input: [u8; 3]) -> b32 {
    b32(to_10(input[0]) | (to_10(input[1]) << 10) | (to_10(input[2]) << 20))
}

pub fn u8x3_to_b8(input: [u8; 3]) -> b8 {
    b8(to_3(input[0]) | (to_3(input[1]) << 3) | (to_2(input[2]) << 6))
}

macro_rules! to_unary {
    ($name:ident, $u_type:ty, $len:expr) => {
        pub fn $name(input: u8) -> $u_type {
            !((!<$u_type>::default()) << (input / (256 / $len) as u8) as usize)
        }
    };
}

// f32 ranged from -1 to +1.
pub fn f32_to_b32(f: f32) -> b32 {
    let shift = (((f + 1.0) / 2.0) * 32.0) as usize;
    b32(!0u32 << shift)
}

to_unary!(to_10, u32, 10);
to_unary!(to_32, u32, 32);

to_unary!(to_2, u8, 2);
to_unary!(to_3, u8, 3);
to_unary!(to_8, u8, 8);
