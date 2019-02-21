#[macro_use]
extern crate serde_derive;
extern crate bincode;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate time;

pub mod datasets {
    pub mod cifar {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;

        pub fn encode_unary_rgb(bytes: [u8; 3]) -> u64 {
            let mut ones = 0u64;
            for color in 0..3 {
                for i in 0..bytes[color] / 12 {
                    ones = ones | 0b1u64 << (color * 21 + i as usize);
                }
            }
            ones
        }
        pub fn parse_rgb_u64(bits: u64) -> [u8; 3] {
            let mut bytes = [0u8; 3];
            for color in 0..3 {
                bytes[color] = (((bits >> (color * 21)) & 0b111111111111111111111u64).count_ones()
                    * 12) as u8; // 21 bit mask
            }
            bytes
        }
        pub fn load_images_10(path: &String, size: usize) -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open data");

            let mut image_bytes: [u8; 1024 * 3] = [0; 1024 * 3];
            let mut label: [u8; 1] = [0; 1];
            let mut images: Vec<(usize, [[[u8; 3]; 32]; 32])> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut label).expect("can't read label");
                file.read_exact(&mut image_bytes)
                    .expect("can't read images");
                let mut image = [[[0u8; 3]; 32]; 32];
                for x in 0..32 {
                    for y in 0..32 {
                        image[x][y] = [
                            image_bytes[(0 * 1024) + (y * 32) + x],
                            image_bytes[(1 * 1024) + (y * 32) + x],
                            image_bytes[(2 * 1024) + (y * 32) + x],
                        ];
                    }
                }
                images.push((label[0] as usize, image));
            }
            return images;
        }
    }
    pub mod mnist {
        use std::fs::File;
        use std::io::prelude::*;
        use std::path::Path;
        pub fn load_labels(path: &String, size: usize) -> Vec<usize> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 8] = [0; 8];
            file.read_exact(&mut header).expect("can't read header");

            let mut byte: [u8; 1] = [0; 1];
            let mut labels: Vec<usize> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut byte).expect("can't read label");
                labels.push(byte[0] as usize);
            }
            return labels;
        }
        pub fn load_images_u8_1chan(path: &String, size: usize) -> Vec<[[[u8; 1]; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[[[u8; 1]; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image = [[[0u8; 1]; 28]; 28];
                for p in 0..784 {
                    image[p / 28][p % 28][0] = images_bytes[p];
                }
                images.push(image);
            }
            return images;
        }
        pub fn load_images(path: &String, size: usize) -> Vec<[[u8; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[[u8; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image = [[0u8; 28]; 28];
                for p in 0..784 {
                    image[p / 28][p % 28] = images_bytes[p];
                }
                images.push(image);
            }
            return images;
        }
        pub fn load_images_bitpacked(path: &String, size: usize) -> Vec<[u64; 13]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in reverse order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[u64; 13]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image_words: [u64; 13] = [0; 13];
                for p in 0..784 {
                    let word_index = p / 64;
                    image_words[word_index] =
                        image_words[word_index] | (((images_bytes[p] > 128) as u64) << p % 64);
                }
                images.push(image_words);
            }
            return images;
        }
        pub fn load_images_64chan(path: &String, size: usize) -> Vec<[[[u64; 1]; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            let mut images: Vec<[[[u64; 1]; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes)
                    .expect("can't read images");
                let mut image = [[[0u64; 1]; 28]; 28];
                for p in 0..784 {
                    let mut ones = 0u64;
                    for i in 0..images_bytes[p] / 4 {
                        ones = ones | 1 << i;
                    }
                    image[p / 28][p % 28][0] = ones;
                }
                images.push(image);
            }
            return images;
        }
    }
}

macro_rules! for_uints {
    ($tokens:tt) => {
        $tokens!(u8);
        $tokens!(u16);
        $tokens!(u32);
        $tokens!(u64);
        $tokens!(u128);
    };
}

macro_rules! for_ref_uints {
    ($tokens:tt) => {
        $tokens!(&u8);
        $tokens!(&u16);
        $tokens!(&u32);
        $tokens!(&u64);
        $tokens!(&u128);
    };
}

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

macro_rules! primitive_bit_len {
    ($type:ty, $len:expr) => {
        impl BitLen for $type {
            const BIT_LEN: usize = $len;
        }
    };
}

primitive_bit_len!(u8, 8);
primitive_bit_len!(u16, 16);
primitive_bit_len!(u32, 32);
primitive_bit_len!(u64, 64);
primitive_bit_len!(u128, 128);

macro_rules! array_bit_len {
    ($len:expr) => {
        impl<T: BitLen> BitLen for [T; $len] {
            const BIT_LEN: usize = $len * T::BIT_LEN;
        }
    };
}

array_bit_len!(2);
array_bit_len!(3);
array_bit_len!(4);
array_bit_len!(5);
array_bit_len!(6);
array_bit_len!(7);
array_bit_len!(8);
array_bit_len!(9);
array_bit_len!(13);
array_bit_len!(16);
array_bit_len!(28);
array_bit_len!(32);
array_bit_len!(49);
array_bit_len!(64);
array_bit_len!(128);

// types that are FullLen can have an array as long as there are bits in the type.
pub trait FullLen<K> {
    type FullArray;
}

macro_rules! impl_full_len {
    ($target:ty) => {
        impl<K> FullLen<K> for $target {
            type FullArray = [K; <$target>::BIT_LEN];
        }
    };
}
for_uints!(impl_full_len);

// types that are HalfLen can have an array half as long as there are bits in the type.
pub trait HalfLen<K> {
    type HalfArray;
}

macro_rules! impl_half_len {
    ($target:ty) => {
        impl<K: Default + Copy + Sized> HalfLen<K> for $target {
            type HalfArray = [K; <$target>::BIT_LEN / 2];
        }
    };
}
impl_half_len!(u32);
//for_uints!(impl_half_len);

pub trait HammingDist {
    fn hd(self, Self) -> u32;
}

macro_rules! impl_hammingdist {
    ($type:ty) => {
        impl HammingDist for $type {
            #[inline(always)]
            fn hd(self, other: Self) -> u32 {
                (self ^ other).count_ones()
            }
        }
    };
}
for_uints!(impl_hammingdist);
for_ref_uints!(impl_hammingdist);

pub trait Patch: Send + Sync + Sized + BitLen {
    fn hamming_distance(&self, &Self) -> u32;
    fn distance_and_threshold(&self, other: &Self) -> bool {
        self.hamming_distance(other) > (Self::BIT_LEN / 2) as u32
    }
    fn bit_increment(&self, &mut [u32]);
    fn bitpack(&[bool]) -> Self;
    fn bit_or(&self, &Self) -> Self;
    fn flip_bit(&mut self, usize);
    fn get_bit(&self, usize) -> bool;
}

macro_rules! primitive_patch {
    ($type:ty) => {
        impl Patch for $type {
            #[inline(always)]
            fn hamming_distance(&self, other: &Self) -> u32 {
                (self ^ other).count_ones()
            }
            fn bit_increment(&self, counters: &mut [u32]) {
                if counters.len() != <$type>::BIT_LEN {
                    panic!(
                        "primitive increment: counters is {:?}, should be {:?}",
                        counters.len(),
                        <$type>::BIT_LEN
                    );
                }
                for i in 0..<$type>::BIT_LEN {
                    counters[i] += ((self >> i) & 0b1 as $type) as u32;
                }
            }
            fn bitpack(bools: &[bool]) -> $type {
                if bools.len() != <$type>::BIT_LEN {
                    panic!(
                        "primitive bitpack: counters is {:?}, should be {:?}",
                        bools.len(),
                        <$type>::BIT_LEN
                    );
                }
                let mut val = 0 as $type;
                for i in 0..<$type>::BIT_LEN {
                    val = val | ((bools[i] as $type) << i);
                }
                val
            }
            fn bit_or(&self, other: &$type) -> $type {
                self | other
            }
            fn flip_bit(&mut self, index: usize) {
                *self ^= 1 << index
            }
            fn get_bit(&self, index: usize) -> bool {
                ((self >> index) & 0b1) == 1
            }
        }
    };
}

for_uints!(primitive_patch);

macro_rules! array_patch {
    ($len:expr) => {
        impl<T: Patch + Copy + Default> Patch for [T; $len] {
            fn hamming_distance(&self, other: &[T; $len]) -> u32 {
                let mut distance = 0;
                for i in 0..$len {
                    distance += self[i].hamming_distance(&other[i]);
                }
                distance
            }
            fn bit_increment(&self, counters: &mut [u32]) {
                if counters.len() != ($len * T::BIT_LEN) {
                    panic!(
                        "array increment: counters is {:?}, should be {:?}",
                        counters.len(),
                        $len * T::BIT_LEN
                    );
                }
                for i in 0..$len {
                    self[i].bit_increment(&mut counters[i * T::BIT_LEN..(i + 1) * T::BIT_LEN]);
                }
            }
            fn bitpack(bools: &[bool]) -> [T; $len] {
                if bools.len() != ($len * T::BIT_LEN) {
                    panic!(
                        "array bitpack: bools is {:?}, should be {:?}",
                        bools.len(),
                        $len * T::BIT_LEN
                    );
                }
                let mut val = [T::default(); $len];
                for i in 0..$len {
                    val[i] = T::bitpack(&bools[i * T::BIT_LEN..(i + 1) * T::BIT_LEN]);
                }
                val
            }
            fn bit_or(&self, other: &Self) -> Self {
                let mut output = [T::default(); $len];
                for i in 0..$len {
                    output[i] = self[i].bit_or(&other[i]);
                }
                output
            }
            fn flip_bit(&mut self, index: usize) {
                self[index / T::BIT_LEN].flip_bit(index % T::BIT_LEN);
            }
            fn get_bit(&self, index: usize) -> bool {
                self[index / T::BIT_LEN].get_bit(index % T::BIT_LEN)
            }
        }
    };
}

array_patch!(2);
array_patch!(3);
array_patch!(4);
array_patch!(5);
array_patch!(6);
array_patch!(7);
array_patch!(8);
array_patch!(9);
array_patch!(13);
array_patch!(16);
array_patch!(28);
array_patch!(32);
array_patch!(49);
array_patch!(64);

#[macro_use]
pub mod layers {
    use super::{BitLen, HammingDist, Patch};
    use bincode::{deserialize_from, serialize_into};
    use rayon::prelude::*;
    use std::fs::File;
    use std::io::BufWriter;
    use std::marker::PhantomData;
    use std::mem::transmute;
    use std::path::Path;
    use time::PreciseTime;

    macro_rules! patch_3x3 {
        ($input:expr, $x:expr, $y:expr) => {
            [
                [
                    $input[$x + 0][$y + 0],
                    $input[$x + 0][$y + 1],
                    $input[$x + 0][$y + 2],
                ],
                [
                    $input[$x + 1][$y + 0],
                    $input[$x + 1][$y + 1],
                    $input[$x + 1][$y + 2],
                ],
                [
                    $input[$x + 2][$y + 0],
                    $input[$x + 2][$y + 1],
                    $input[$x + 2][$y + 2],
                ],
            ]
        };
    }

    pub trait IsCorrect<I> {
        fn is_correct(&self, target: u8, input: I) -> bool;
    }
    // Note the neither of these implementations have a mean of 0.1
    // for an input of 128 bits, is_correct is ~0.0844437
    // and not_incorrect is ~0.118793.
    // Both should approach 0.1 as bitlen of the input reaches infinity.
    // is_correct is slightly faster then not_incorrect.
    impl<I: HammingDist + Copy> IsCorrect<I> for [I; 10] {
        // the max activation is the target.
        #[inline(always)]
        fn is_correct(&self, target: u8, input: I) -> bool {
            let max = self[target as usize].hd(input);
            for i in 0..10 {
                if i != target as usize {
                    if self[i].hd(input) >= max {
                        return false;
                    }
                }
            }
            true
        }
    }

    pub trait Apply<I, O> {
        fn apply(&self, &I) -> O;
        fn update(&self, input: &I, target: &mut O, _: usize) {
            *target = self.apply(input);
        }
    }

    macro_rules! primitive_dense_apply {
        ($type:ty) => {
            impl<I: Patch> Apply<I, $type> for [I; <$type>::BIT_LEN] {
                fn apply(&self, input: &I) -> $type {
                    let mut target = <$type>::default();
                    for i in 0..<$type>::BIT_LEN {
                        target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2))
                            as $type)
                            << i;
                    }
                    target
                }
                fn update(&self, input: &I, target: &mut $type, i: usize) {
                    *target &= !(1 << i); // unset the bit
                    *target |= ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2))
                        as $type)
                        << i;
                }
            }
        };
    }

    primitive_dense_apply!(u8);
    primitive_dense_apply!(u16);
    primitive_dense_apply!(u32);
    primitive_dense_apply!(u64);
    primitive_dense_apply!(u128);

    pub trait SaveLoad
    where
        Self: Sized,
    {
        fn write_to_fs(&self, path: &Path);
        fn new_from_fs(path: &Path) -> Option<Self>;
    }

    pub trait NewFromRng {
        fn new_from_rng<RNG: rand::Rng>(rng: &mut RNG) -> Self;
    }

    macro_rules! primitive_activations_new_from_seed {
        ($len:expr) => {
            impl<I: Patch> NewFromRng for [I; $len]
            where
                rand::distributions::Standard: rand::distributions::Distribution<I>,
            {
                fn new_from_rng<RNG: rand::Rng>(rng: &mut RNG) -> Self {
                    rng.gen::<[I; $len]>()
                }
            }
        };
    }
    primitive_activations_new_from_seed!(8);
    primitive_activations_new_from_seed!(10);
    primitive_activations_new_from_seed!(16);
    primitive_activations_new_from_seed!(32);

    pub trait Mutate {
        fn mutate(&mut self, output_index: usize, input_index: usize);
        const OUTPUT_LEN: usize;
        const INPUT_LEN: usize;
    }

    macro_rules! primitive_mutate_matrix_trait {
        ($len:expr) => {
            impl<I: Patch> Mutate for [I; $len] {
                fn mutate(&mut self, output_index: usize, input_index: usize) {
                    self[output_index].flip_bit(input_index);
                }
                const OUTPUT_LEN: usize = $len;
                const INPUT_LEN: usize = I::BIT_LEN;
            }
        };
    }

    primitive_mutate_matrix_trait!(8);
    primitive_mutate_matrix_trait!(16);
    primitive_mutate_matrix_trait!(32);
    primitive_mutate_matrix_trait!(64);
    primitive_mutate_matrix_trait!(128);

    pub trait OptimizeHead<I> {
        fn optimize_head(&mut self, examples: &[(u8, I)]) -> u64;
    }

    impl<I: Patch + Copy + Default + Sync + HammingDist> OptimizeHead<I> for [I; 10] {
        fn optimize_head(&mut self, examples: &[(u8, I)]) -> u64 {
            let backup_self = *self;
            let all_start = PreciseTime::now();
            let before_obj: u64 = examples
                .par_iter()
                .map(|(class, input)| self.is_correct(*class, *input) as u64)
                .sum();
            for mut_class in 0..10usize {
                // first we need to calculate the activations for each example.
                let mut activation_diffs: Vec<(I, u32, bool)> = examples
                    .par_iter()
                    .map(|(targ_class, input)| {
                        // calculate the activations for all 10 outputs.
                        let mut activations = [0u32; 10];
                        for i in 0..10 {
                            activations[i] = self[i].hamming_distance(input);
                        }
                        // the activation for target class of this example.
                        let targ_act = activations[*targ_class as usize];
                        // ignore the target activation,
                        activations[*targ_class as usize] = 0;
                        // and the mut activation.
                        activations[mut_class] = 0;
                        // the max activation of all the classes not in the target class or mut class.
                        let max_other_activations = activations.iter().max().unwrap();
                        let max: u32 = {
                            if *targ_class as usize == mut_class {
                                *max_other_activations
                            } else {
                                targ_act
                            }
                        };
                        (
                            input,
                            max,
                            *targ_class as usize == mut_class,
                            (targ_act > *max_other_activations)
                                | (*targ_class as usize == mut_class),
                        )
                    })
                    .filter(|(_, _, _, keep)| *keep) // now we filter out the examples which have no chance of ever being correct.
                    .map(|(input, diff, sign, _)| (*input, diff, sign)) // and remove keep.
                    .collect();

                // note that this sum correct is not the true acc,
                // it is working on the subset that can be made correct or incorrect by this activation.
                let mut sum_correct: u64 = activation_diffs
                    .par_iter()
                    .map(|(input, max, sign)| {
                        // calculate the activatio for the mut class.
                        let mut_act = self[mut_class].hamming_distance(input);
                        // is it on the correct side of max?
                        ((*sign ^ (mut_act < *max)) & (mut_act != *max)) as u64
                    })
                    .sum();
                for b in 0..I::BIT_LEN {
                    self[mut_class].flip_bit(b);
                    let new_sum_correct: u64 = activation_diffs
                        .par_iter()
                        .map(|(input, max, sign)| {
                            let mut_act = self[mut_class].hamming_distance(input);
                            ((*sign ^ (mut_act < *max)) & (mut_act != *max)) as u64
                        })
                        .sum();
                    if new_sum_correct > sum_correct {
                        sum_correct = new_sum_correct;
                    } else {
                        // revert the bit
                        self[mut_class].flip_bit(b);
                    }
                }
            }
            let mut after_obj: u64 = examples
                .par_iter()
                .map(|(class, input)| self.is_correct(*class, *input) as u64)
                .sum();
            if before_obj > after_obj {
                println!("reverting obj regression: {} > {}", before_obj, after_obj);
                *self = backup_self;
                after_obj = before_obj;
            }
            println!("head update time: {:?}", all_start.to(PreciseTime::now()));
            after_obj
        }
    }

    pub trait SimplifyBits<T> {
        fn simplify(&self) -> T;
    }

    macro_rules! simplify_bits_trait {
        ($in_type:ty, $out_type:ty) => {
            impl SimplifyBits<$out_type> for $in_type {
                fn simplify(&self) -> $out_type {
                    unsafe { transmute::<$in_type, $out_type>(*self) }
                }
            }
        };
    }
    simplify_bits_trait!([u64; 4], [u128; 2]);
    simplify_bits_trait!([u16; 2], u32);
    simplify_bits_trait!([u32; 2], u64);

    simplify_bits_trait!([u128; 8], [u128; 8]);
    simplify_bits_trait!([u8; 8], u64);
    simplify_bits_trait!([u16; 4], u64);
    simplify_bits_trait!([[u8; 28]; 28], [[u128; 7]; 7]);

    macro_rules! simplify_to_1u128_bits_trait {
        ($in_type:ty) => {
            simplify_bits_trait!($in_type, u128);
        };
    }

    simplify_to_1u128_bits_trait!([u8; 16]);
    simplify_to_1u128_bits_trait!([u16; 8]);
    simplify_to_1u128_bits_trait!([u32; 4]);
    simplify_to_1u128_bits_trait!([u64; 2]);

    macro_rules! simplify_bits_trait_array {
        ($in_type:ty, $len:expr) => {
            simplify_bits_trait!($in_type, [u128; $len]);
        };
    }

    simplify_bits_trait_array!([u8; 32], 2);
    simplify_bits_trait_array!([u8; 784], 49);
    simplify_bits_trait_array!([[u8; 28]; 28], 49);
    simplify_bits_trait_array!([u32; 8], 2);
    simplify_bits_trait_array!([u64; 8], 4);

    pub mod unary {
        macro_rules! to_unary {
            ($name:ident, $type:ty, $len:expr) => {
                pub fn $name(input: u8) -> $type {
                    !((!0) << (input / (255 / $len)))
                }
            };
        }

        to_unary!(to_3, u8, 3);
        to_unary!(to_4, u8, 4);
        to_unary!(to_5, u8, 5);
        to_unary!(to_6, u8, 6);
        to_unary!(to_7, u8, 7);
        to_unary!(to_8, u8, 8);
        to_unary!(to_10, u32, 10);
        to_unary!(to_11, u32, 11);
        to_unary!(to_14, u16, 14);
        to_unary!(to_21, u32, 21);
        to_unary!(to_22, u32, 22);
        to_unary!(to_42, u64, 42);
        to_unary!(to_43, u64, 43);

        pub fn rgb_to_u14(pixels: [u8; 3]) -> u16 {
            to_4(pixels[0]) as u16
                | ((to_5(pixels[1]) as u16) << 4)
                | ((to_5(pixels[2]) as u16) << 9)
        }
        pub fn rgb_to_u16(pixels: [u8; 3]) -> u16 {
            to_5(pixels[0]) as u16
                | ((to_5(pixels[1]) as u16) << 5)
                | ((to_6(pixels[2]) as u16) << 10)
        }
        pub fn rgb_to_u32(pixels: [u8; 3]) -> u32 {
            to_11(pixels[0]) as u32
                | ((to_11(pixels[1]) as u32) << 11)
                | ((to_10(pixels[2]) as u32) << 22)
        }
        pub fn rgb_to_u64(pixels: [u8; 3]) -> u64 {
            to_21(pixels[0]) as u64
                | ((to_21(pixels[1]) as u64) << 21)
                | ((to_22(pixels[2]) as u64) << 42)
        }
        pub fn rgb_to_u128(pixels: [u8; 3]) -> u128 {
            to_43(pixels[0]) as u128
                | ((to_43(pixels[1]) as u128) << 43)
                | ((to_42(pixels[2]) as u128) << 86)
        }
    }

    pub trait ConcatImages<I> {
        fn concat_images(inputs: I) -> Self;
    }

    macro_rules! impl_concat_image {
        ($depth:expr, $x_size:expr, $y_size:expr) => {
            impl<IP: Default + Copy, OP: Default + Copy>
                ConcatImages<[&[[IP; $y_size]; $x_size]; $depth]> for [[OP; $y_size]; $x_size]
            where
                [IP; $depth]: SimplifyBits<OP>,
            {
                fn concat_images(
                    input: [&[[IP; $y_size]; $x_size]; $depth],
                ) -> [[OP; $y_size]; $x_size] {
                    let mut target = <[[OP; $y_size]; $x_size]>::default();
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            let mut target_pixel = [IP::default(); $depth];
                            for i in 0..$depth {
                                target_pixel[i] = input[i][x][y];
                            }
                            target[x][y] = target_pixel.simplify();
                        }
                    }
                    target
                }
            }
        };
    }
    impl_concat_image!(2, 32, 32);
    impl_concat_image!(4, 32, 32);

    pub fn vec_concat_2_examples<'a, I: 'a + Sync, C: ConcatImages<[&'a I; 2]> + Sync + Send>(
        a: &'a Vec<(usize, I)>,
        b: &'a Vec<(usize, I)>,
    ) -> Vec<(usize, C)> {
        assert_eq!(a.len(), b.len());
        a.par_iter()
            .zip(b.par_iter())
            .map(|((a_class, a_image), (b_class, b_image))| {
                assert_eq!(a_class, b_class);
                (*a_class, C::concat_images([a_image, b_image]))
            })
            .collect()
    }

    pub fn vec_concat_4_examples<'a, I: 'a + Sync, C: ConcatImages<[&'a I; 4]> + Sync + Send>(
        a: &'a Vec<(usize, I)>,
        b: &'a Vec<(usize, I)>,
        c: &'a Vec<(usize, I)>,
        d: &'a Vec<(usize, I)>,
    ) -> Vec<(usize, C)> {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        assert_eq!(a.len(), d.len());
        a.par_iter()
            .zip(b.par_iter())
            .zip(c.par_iter())
            .zip(d.par_iter())
            .map(
                |(
                    (((a_class, a_image), (b_class, b_image)), (c_class, c_image)),
                    (d_class, d_image),
                )| {
                    assert_eq!(a_class, b_class);
                    assert_eq!(a_class, c_class);
                    assert_eq!(a_class, d_class);
                    (
                        *a_class,
                        C::concat_images([a_image, b_image, c_image, d_image]),
                    )
                },
            )
            .collect()
    }

    #[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
    pub struct Conv3x3<I, FN, O> {
        input_pixel_type: PhantomData<I>,
        pub map_fn: FN,
        output_type: PhantomData<O>,
    }

    macro_rules! conv3x3_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, FN: Apply<[[I; 3]; 3], O>, O: Default + Copy>
                Apply<[[I; $y_size]; $x_size], [[O; $y_size]; $x_size]> for Conv3x3<I, FN, O>
            {
                fn apply(&self, input: &[[I; $y_size]; $x_size]) -> [[O; $y_size]; $x_size] {
                    let mut target = [[O::default(); $y_size]; $x_size];
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            target[x + 1][y + 1] = self.map_fn.apply(&patch_3x3!(input, x, y));
                        }
                    }
                    target
                }
                fn update(
                    &self,
                    input: &[[I; $y_size]; $x_size],
                    target: &mut [[O; $y_size]; $x_size],
                    index: usize,
                ) {
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            self.map_fn.update(
                                &patch_3x3!(input, x, y),
                                &mut target[x + 1][y + 1],
                                index,
                            );
                        }
                    }
                }
            }
        };
    }

    conv3x3_apply_trait!(32, 32);
    conv3x3_apply_trait!(16, 16);
    conv3x3_apply_trait!(8, 8);
    conv3x3_apply_trait!(4, 4);

    impl<I: Sync, FN: Apply<[[I; 3]; 3], O> + Mutate, O: Sync> Mutate for Conv3x3<I, FN, O> {
        fn mutate(&mut self, output_index: usize, input_index: usize) {
            self.map_fn.mutate(output_index, input_index);
        }
        const OUTPUT_LEN: usize = FN::OUTPUT_LEN;
        const INPUT_LEN: usize = FN::INPUT_LEN;
    }

    macro_rules! impl_saveload_conv3x3 {
        ($len:expr) => {
            impl<I: Default + Copy, O> SaveLoad for Conv3x3<I, [[[I; 3]; 3]; $len], O>
            where
                I: serde::Serialize,
                for<'de> I: serde::Deserialize<'de>,
            {
                fn write_to_fs(&self, path: &Path) {
                    let vec_params: Vec<[[I; 3]; 3]> = self.map_fn.iter().map(|x| *x).collect();
                    let mut f = BufWriter::new(File::create(path).unwrap());
                    serialize_into(&mut f, &vec_params).unwrap();
                }
                // This will return:
                // - Some if the file exists and is good
                // - None of the file does not exist
                // and will panic if the file is exists but is bad.
                fn new_from_fs(path: &Path) -> Option<Self> {
                    File::open(&path)
                        .map(|f| deserialize_from(f).unwrap())
                        .map(|vec_params: Vec<[[I; 3]; 3]>| {
                            if vec_params.len() != $len {
                                panic!("input is of len {} not {}", vec_params.len(), $len);
                            }
                            let mut params = [<[[I; 3]; 3]>::default(); $len];
                            for i in 0..$len {
                                params[i] = vec_params[i];
                            }
                            Conv3x3 {
                                input_pixel_type: PhantomData,
                                map_fn: params,
                                output_type: PhantomData,
                            }
                        })
                        .ok()
                }
            }
        };
    }
    impl_saveload_conv3x3!(8);
    impl_saveload_conv3x3!(16);
    impl_saveload_conv3x3!(32);
    impl_saveload_conv3x3!(64);
    impl_saveload_conv3x3!(128);

    #[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
    pub struct Conv2x2Stride2<I, P, FN, O> {
        input_pixel_type: PhantomData<I>,
        patch_type: PhantomData<P>,
        pub map_fn: FN,
        output_type: PhantomData<O>,
    }

    macro_rules! impl_saveload_conv2x2 {
        ($len:expr) => {
            impl<I, P: Copy + Default, O> SaveLoad for Conv2x2Stride2<I, P, [P; $len], O>
            where
                [I; 4]: SimplifyBits<P>,
                P: serde::Serialize,
                for<'de> P: serde::Deserialize<'de>,
            {
                fn write_to_fs(&self, path: &Path) {
                    let vec_params: Vec<P> = self.map_fn.iter().map(|x| *x).collect();
                    let mut f = BufWriter::new(File::create(path).unwrap());
                    serialize_into(&mut f, &vec_params).unwrap();
                }
                // This will return:
                // - Some if the file exists and is good
                // - None of the file does not exist
                // and will panic if the file is exists but is bad.
                fn new_from_fs(path: &Path) -> Option<Self> {
                    File::open(&path)
                        .map(|f| deserialize_from(f).unwrap())
                        .map(|vec_params: Vec<P>| {
                            if vec_params.len() != $len {
                                panic!("input is of len {} not {}", vec_params.len(), $len);
                            }
                            let mut params = [P::default(); $len];
                            for i in 0..$len {
                                params[i] = vec_params[i];
                            }
                            Conv2x2Stride2 {
                                input_pixel_type: PhantomData,
                                patch_type: PhantomData,
                                output_type: PhantomData,
                                map_fn: params,
                            }
                        })
                        .ok()
                }
            }
        };
    }
    impl_saveload_conv2x2!(8);
    impl_saveload_conv2x2!(16);
    impl_saveload_conv2x2!(32);
    impl_saveload_conv2x2!(64);
    impl_saveload_conv2x2!(128);

    impl<I: Sync, P: Sync, FN: Apply<P, O> + Mutate, O: Sync> Mutate for Conv2x2Stride2<I, P, FN, O>
    where
        [I; 4]: SimplifyBits<P>,
    {
        fn mutate(&mut self, output_index: usize, input_index: usize) {
            self.map_fn.mutate(output_index, input_index);
        }
        const OUTPUT_LEN: usize = FN::OUTPUT_LEN;
        const INPUT_LEN: usize = FN::INPUT_LEN;
    }

    macro_rules! patch_conv_2x2_apply_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, P: Sync, FN: Apply<P, O>, O: Default + Copy>
                Apply<[[I; $y_size]; $x_size], [[O; $y_size / 2]; $x_size / 2]>
                for Conv2x2Stride2<I, P, FN, O>
            where
                [I; 4]: SimplifyBits<P>,
            {
                fn apply(
                    &self,
                    input: &[[I; $y_size]; $x_size],
                ) -> [[O; $y_size / 2]; $x_size / 2] {
                    let mut target = [[O::default(); $y_size / 2]; $x_size / 2];
                    for x in 0..($x_size / 2) {
                        let x_base = x * 2;
                        for y in 0..($y_size / 2) {
                            let y_base = y * 2;
                            target[x][y] = self.map_fn.apply(
                                &([
                                    input[x_base + 0][y_base + 0],
                                    input[x_base + 0][y_base + 1],
                                    input[x_base + 1][y_base + 0],
                                    input[x_base + 1][y_base + 1],
                                ])
                                .simplify(),
                            );
                        }
                    }
                    target
                }
                fn update(
                    &self,
                    input: &[[I; $y_size]; $x_size],
                    target: &mut [[O; $y_size / 2]; $x_size / 2],
                    index: usize,
                ) {
                    for x in 0..($x_size / 2) {
                        let x_base = x * 2;
                        for y in 0..($y_size / 2) {
                            let y_base = y * 2;
                            self.map_fn.update(
                                &([
                                    input[x_base + 0][y_base + 0],
                                    input[x_base + 0][y_base + 1],
                                    input[x_base + 1][y_base + 0],
                                    input[x_base + 1][y_base + 1],
                                ])
                                .simplify(),
                                &mut target[x][y],
                                index,
                            );
                        }
                    }
                }
            }
        };
    }

    patch_conv_2x2_apply_trait!(32, 32);
    patch_conv_2x2_apply_trait!(16, 16);
    patch_conv_2x2_apply_trait!(8, 8);
    patch_conv_2x2_apply_trait!(4, 4);

    pub trait ExtractPixels<P> {
        fn pixels(&self) -> Vec<P>;
    }

    macro_rules! extract_pixels_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<P: Copy> ExtractPixels<P> for [[P; $y_size]; $x_size] {
                fn pixels(&self) -> Vec<P> {
                    let mut pixels = Vec::with_capacity(($y_size - 2) * ($x_size - 2));
                    for x in 1..$x_size - 1 {
                        for y in 1..$y_size - 1 {
                            pixels.push(self[x][y]);
                        }
                    }
                    pixels
                }
            }
        };
    }

    extract_pixels_trait!(32, 32);
    extract_pixels_trait!(16, 16);
    extract_pixels_trait!(8, 8);
    extract_pixels_trait!(4, 4);

    pub trait Extract3x3Patches<P> {
        fn patches(&self) -> Vec<[[P; 3]; 3]>;
    }

    macro_rules! extract_patch_3x3_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<P: Copy> Extract3x3Patches<P> for [[P; $y_size]; $x_size] {
                fn patches(&self) -> Vec<[[P; 3]; 3]> {
                    let mut patches = Vec::with_capacity(($y_size - 2) * ($x_size - 2));
                    for x in 0..$x_size - 2 {
                        for y in 0..$y_size - 2 {
                            patches.push(patch_3x3!(self, x, y));
                        }
                    }
                    patches
                }
            }
        };
    }

    extract_patch_3x3_trait!(32, 32);
    extract_patch_3x3_trait!(16, 16);
    extract_patch_3x3_trait!(8, 8);
    extract_patch_3x3_trait!(4, 4);

    pub trait Extract2x2PatchesStrided<OP> {
        fn patches2x2(&self) -> Vec<OP>;
    }

    macro_rules! extract_patch_4_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I: Copy, O> Extract2x2PatchesStrided<O> for [[I; $y_size]; $x_size]
            where
                [I; 4]: SimplifyBits<O>,
            {
                fn patches2x2(&self) -> Vec<O> {
                    let mut patches = Vec::with_capacity(($y_size - 2) * ($x_size - 2));
                    for x in 0..($x_size / 2) {
                        let x_base = x * 2;
                        for y in 0..($y_size / 2) {
                            let y_base = y * 2;
                            patches.push(
                                ([
                                    self[x_base + 0][y_base + 0],
                                    self[x_base + 0][y_base + 1],
                                    self[x_base + 1][y_base + 0],
                                    self[x_base + 1][y_base + 1],
                                ])
                                .simplify(),
                            );
                        }
                    }
                    patches
                }
            }
        };
    }

    extract_patch_4_trait!(32, 32);
    extract_patch_4_trait!(16, 16);
    extract_patch_4_trait!(8, 8);
    extract_patch_4_trait!(4, 4);

    pub trait PixelMap<I, O, OI> {
        fn pixel_map(&self, &Fn(&I) -> O) -> OI;
    }

    macro_rules! pixel_map_trait {
        ($x_size:expr, $y_size:expr) => {
            impl<I, O: Copy + Default> PixelMap<I, O, [[O; $y_size]; $x_size]>
                for [[I; $y_size]; $x_size]
            {
                fn pixel_map(&self, map_fn: &Fn(&I) -> O) -> [[O; $y_size]; $x_size] {
                    let mut output = [[O::default(); $y_size]; $x_size];
                    for x in 0..$x_size {
                        for y in 0..$y_size {
                            output[x][y] = map_fn(&self[x][y]);
                        }
                    }
                    output
                }
            }
        };
    }

    pixel_map_trait!(32, 32);
    pixel_map_trait!(28, 28);
    pixel_map_trait!(16, 16);

}

#[cfg(test)]
mod tests {
    use super::layers::unary;
    use super::Patch;
    #[test]
    fn patch_count() {
        let mut counters = vec![0u32; 32];
        [0b1011_1111u8, 0b0100_0000u8, 0b0100_0000u8, 0b1000_1010u8].bit_increment(&mut counters);
        let bools: Vec<_> = counters.iter().map(|&x| x != 0).collect();
        let avg = <[u16; 2]>::bitpack(&bools);

        let mut counters2 = vec![0u32; 32];
        avg.bit_increment(&mut counters2);
        println!("{:?}", counters);
        assert_eq!(counters, counters2)
    }
    #[test]
    fn patch_dist() {
        assert_eq!(123u8.hamming_distance(&123u8), 0);
        assert_eq!(0b1010_1000u8.hamming_distance(&0b1010_0111u8), 4);
        assert_eq!(
            [0b1111_0000u8, 0b1111_0000u8].hamming_distance(&[0b0000_1100u8, 0b1111_1111u8]),
            6 + 4
        );
    }
    #[test]
    fn unary() {
        assert_eq!(unary::to_3(128), 0b0000_0001u8);
        assert_eq!(unary::to_3(255), 0b0000_0111u8);
        assert_eq!(unary::to_7(255), 0b0111_1111u8);
        assert_eq!(unary::to_7(0), 0b0000_0000u8);
    }
    #[test]
    fn rgb_pack() {
        assert_eq!(unary::rgb_to_u14([255, 255, 255]), 0b0011_1111_1111_1111u16);
        assert_eq!(unary::rgb_to_u14([128, 128, 128]), 0b0000_0110_0011_0011u16);
        assert_eq!(unary::rgb_to_u14([0, 0, 0]), 0b0u16);
        assert_eq!(unary::rgb_to_u32([0, 0, 0]), 0b0u32);
        assert_eq!(unary::rgb_to_u32([255, 255, 255]), !0b0u32);
    }
    #[test]
    fn layer_simple() {
        let examples = vec![
            (0usize, 0b1111_1111u8),
            (1, 0b0),
            (2, 0b0),
            (0, 0b1111_1111u8),
            (0, 0b1111_1111u8),
        ];
        let model = Layer::<u8, [u8; 16], u16, [u16; 10]>::new_from_split(&examples);
        let obj = model.objective(&(0, 0b1111_1111u8));
        assert_eq!(obj, 1f64);
        let acc = model.accuracy(&(0, 0b0111_1011));
        assert_eq!(acc, 1f64);

        let mut model =
            Layer::<u8, [u8; 16], u16, Layer<u16, [u16; 32], u32, [u32; 10]>>::new_from_split(
                &examples,
            );
        let obj = model.objective(&(0, 0b1111_1111u8));
        assert_eq!(obj, 1f64);
        let acc = model.accuracy(&(0, 0b0111_1011));
        assert_eq!(acc, 1f64);
        let avg_obj = model.optimize_head(&examples, 20);
    }
    #[test]
    fn layer_conv() {
        let examples = vec![
            (0usize, [[0u8; 8]; 8]),
            (0usize, [[0u8; 8]; 8]),
            (0usize, [[0u8; 8]; 8]),
            (1usize, [[!0u8; 8]; 8]),
            (1usize, [[!0u8; 8]; 8]),
            (1usize, [[!0u8; 8]; 8]),
        ];
        let mut model = Patch3x3NotchedConv::<u8, u64, [u64; 10], f64>::new_from_split(&examples);
        let obj = model.objective(&examples[0]);
        assert_eq!(obj, 1f64);
        let obj = model.optimize_head(&examples, 20);

        let mut model = Layer::<
            [[u8; 8]; 8],
            Patch3x3NotchedConv<u8, u64, [u64; 16], u16>,
            [[u16; 8]; 8],
            Patch3x3NotchedConv<u16, u128, [u128; 10], f64>,
        >::new_from_split(&examples);

        let obj = model.objective(&examples[0]);
        assert_eq!(obj, 1f64);
        let obj = model.objective(&examples[4]);
        assert!(obj > 0.5f64);
        let obj = model.optimize_head(&examples, 20);
    }
}
