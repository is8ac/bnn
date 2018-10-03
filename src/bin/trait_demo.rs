trait Distance {
    fn hamming_distance(&self, &Self) -> u32;
}

macro_rules! primitive_hamming_distance {
    ($type:ty) => {
        impl Distance for $type {
            fn hamming_distance(&self, other: &$type) -> u32 {
                (self ^ other).count_ones()
            }
        }
    };
}

primitive_hamming_distance!(u8);
primitive_hamming_distance!(u16);
primitive_hamming_distance!(u32);
primitive_hamming_distance!(u64);
primitive_hamming_distance!(u128);

macro_rules! array_hamming_distance {
    ($len:expr) => {
        impl<T: Distance> Distance for [T; $len] {
            fn hamming_distance(&self, other: &[T; $len]) -> u32 {
                let mut distance = 0;
                for i in 0..$len {
                    distance += self[i].hamming_distance(&other[i]);
                }
                distance
            }
        }
    };
}

array_hamming_distance!(1);
array_hamming_distance!(2);
array_hamming_distance!(3);

macro_rules! primitive_bit_vecmul {
    ($name:ident, $type:ty, $len:expr) => {
        fn $name<T: Distance>(weights: &[(T, u32); $len], input: &T) -> $type {
            let mut output = 0 as $type;
            for b in 0..$len {
                output = output | ((((weights[b].0).hamming_distance(input) >= weights[b].1) as $type) << b);
            }
            output
        }
    };
}

primitive_bit_vecmul!(bit_vecmul_u3, u8, 3);
primitive_bit_vecmul!(bit_vecmul_u7, u8, 7);
primitive_bit_vecmul!(bit_vecmul_u8, u8, 8);
primitive_bit_vecmul!(bit_vecmul_u14, u16, 14);
primitive_bit_vecmul!(bit_vecmul_u16, u16, 16);
primitive_bit_vecmul!(bit_vecmul_u32, u32, 32);
primitive_bit_vecmul!(bit_vecmul_u64, u64, 64);
primitive_bit_vecmul!(bit_vecmul_u128, u128, 128);

macro_rules! to_unary {
    ($name:ident, $type:ty, $len:expr) => {
        fn $name(input: u8) -> $type {
            println!("{:?} {:?}", (input / (255 / $len)), input);
            !((!0) << (input / (255 / $len)))
        }
    };
}

to_unary!(to_unary_3, u8, 3);
to_unary!(to_unary_7, u8, 7);
to_unary!(to_unary_14, u16, 14);

macro_rules! pack_3x3 {
    ($name:ident, $in_type:ty, $out_type:ty, $out_len:expr) => {
        fn $name(pixels: [$in_type; 9]) -> $out_type {
            let mut word = 0 as $out_type;
            for i in 0..9 {
                word = word | ((pixels[i] as $out_type) << (i * ($out_len / 9)))
            }
            word
        }
    };
}

pack_3x3!(pack_3x3_3, u8, u32, 32);
pack_3x3!(pack_3x3_7, u8, u64, 64);
pack_3x3!(pack_3x3_14, u16, u128, 128);


fn main() {
    let input = [0u64; 3];
    let weights = [([0u64; 3], 0u32); 7];
    println!("{:0128b}", bit_vecmul_u7(&weights, &input));
    println!("{:08b}", to_unary_3(0));
    println!("{:08b}", to_unary_3(128));
    println!("{:08b}", to_unary_3(255));
    println!("{:08b}", to_unary_7(0));
    println!("{:08b}", to_unary_7(128));
    println!("{:08b}", to_unary_7(255));
    println!("{:016b}", to_unary_14(0));
    println!("{:016b}", to_unary_14(128));
    println!("{:016b}", to_unary_14(255));

    println!("{:032b}", pack_3x3_3([to_unary_3(128); 9]));
}
