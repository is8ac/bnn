use crate::bits::{b32, GetBit, PackedIndexSGet, BMA};
use crate::shape::Shape;

fn encoder_table() -> [[b32; 8]; 256] {
    let mut table: [[b32; 8]; 256] = [[b32::ZEROS; 8]; 256];
    for x in 0..256 {
        <[[(); 32]; 8] as Shape>::indices()
            .enumerate()
            .for_each(|(y, i)| {
                <[[(); 32]; 8] as PackedIndexSGet<bool>>::set_in_place(
                    &mut table[x],
                    i,
                    (y ^ x).count_ones() > 4,
                )
            });
    }
    table
}

lazy_static! {
    pub static ref ENCODER_TABLE: [[b32; 8]; 256] = encoder_table();
}

fn decoder_table() -> [[b32; 8]; 8] {
    let mut table: [[b32; 8]; 8] = [[b32::ZEROS; 8]; 8];
    for x in 0..8 {
        <[[(); 32]; 8] as Shape>::indices()
            .enumerate()
            .for_each(|(y, i)| {
                <[[(); 32]; 8] as PackedIndexSGet<bool>>::set_in_place(&mut table[x], i, y.bit(x))
            });
    }
    table
}

lazy_static! {
    pub static ref DECODER_TABLE: [[b32; 8]; 8] = decoder_table();
}

pub fn encode_byte(byte: u8) -> [b32; 8] {
    ENCODER_TABLE[byte as usize]
}

pub fn decode_byte(expanded: &[b32; 8]) -> u8 {
    let mut byte = 0u8;
    for b in 0..8 {
        byte |= ((<[[(); 32]; 8] as BMA<bool>>::bma(expanded, &DECODER_TABLE[b]) > 128) as u8) << b;
    }
    byte
}
