use crate::bits::b64;

pub fn generate_hadamard_matrix() -> [[b64; 4]; 256] {
    let mut hadamard_matrix = [[b64(0); 4]; 256];
    for x in 0..256_usize {
        for y in 0..256_usize {
            let sign = ((x & y).count_ones() & 1) == 1;
            hadamard_matrix[x].set_bit(y, sign);
        }
    }
    hadamard_matrix
}

pub fn generate_thin_hadamard_matrix() -> [[b64; 2]; 256] {
    let mut hadamard_matrix = [[b64(0); 2]; 256];
    for x in 0..256_usize {
        for y in 0..128_usize {
            let sign = ((x & (y + 128)).count_ones() & 1) == 1;
            hadamard_matrix[x].set_bit(y, sign);
        }
    }
    hadamard_matrix
}

pub fn generate_wide_hadamard_matrix() -> [[b64; 8]; 256] {
    let mut hadamard_matrix = [[b64(0); 8]; 256];
    for x in 0..256_usize {
        for y in 0..512_usize {
            let sign = ((x & y).count_ones() & 1) == 1;
            hadamard_matrix[x].set_bit(y, sign);
        }
    }
    hadamard_matrix
}

pub trait BitString {
    fn set_bit(&mut self, i: usize, b: bool);
    fn hamming_dist(&self, rhs: &Self) -> u32;
}

impl<const L: usize> BitString for [b64; L] {
    fn set_bit(&mut self, i: usize, b: bool) {
        self[i / 64].0 |= (b as u64) << (i % 64);
    }
    fn hamming_dist(&self, rhs: &Self) -> u32 {
        self.iter().zip(rhs.iter()).map(|(a, b)| (a.0 ^ b.0).count_ones()).sum()
    }
}

lazy_static! {
    static ref HADAMARD_MATRIX_256: [[b64; 4]; 256] = generate_hadamard_matrix();
}

pub fn encode_byte(b: u8) -> [b64; 4] {
    HADAMARD_MATRIX_256[b as usize]
}

pub fn decode_byte(bits: &[b64; 4]) -> u8 {
    HADAMARD_MATRIX_256.iter().enumerate().min_by_key(|(_, row)| row.hamming_dist(bits)).unwrap().0 as u8
}
