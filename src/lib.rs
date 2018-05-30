#![feature(test)]
extern crate test;

extern crate rand;
#[macro_use]
pub mod datasets {
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
        pub fn load_labels_onehot(path: &String, size: usize, onval: u16) -> Vec<[u16; 10]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 8] = [0; 8];
            file.read_exact(&mut header).expect("can't read header");

            let mut byte: [u8; 1] = [0; 1];
            let mut labels: Vec<[u16; 10]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut byte).expect("can't read label");
                let mut label = [0u16; 10];
                label[byte[0] as usize] = onval;
                labels.push(label);
            }
            return labels;
        }
        pub fn load_images_bitpacked(path: &String, size: usize) -> Vec<[u64; 13]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[u64; 13]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image_words: [u64; 13] = [0; 13];
                for p in 0..784 {
                    let word_index = p / 64;
                    image_words[word_index] = image_words[word_index] | (((images_bytes[p] > 128) as u64) << p % 64);
                }
                images.push(image_words);
            }
            return images;
        }
        pub fn load_images_i32(path: &String, size: usize) -> Vec<[i32; 800]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[i32; 800]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image_words = [0i32; 800];
                for p in 0..784 {
                    image_words[p] = images_bytes[p] as i32;
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

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[[[u64; 1]; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
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
        pub fn load_images_u8(path: &String, size: usize) -> Vec<[[u8; 28]; 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[[u8; 28]; 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image = [[0u8; 28]; 28];
                for p in 0..784 {
                    image[p / 28][p % 28] = images_bytes[p];
                }
                images.push(image);
            }
            return images;
        }
        pub fn load_images_64chan_flat(path: &String, size: usize) -> Vec<[u64; 28 * 28]> {
            let path = Path::new(path);
            let mut file = File::open(&path).expect("can't open images");
            let mut header: [u8; 16] = [0; 16];
            file.read_exact(&mut header).expect("can't read header");

            let mut images_bytes: [u8; 784] = [0; 784];

            // bitpack the image into 13 64 bit words.
            // There will be unused space in the last word, this is acceptable.
            // the bits of each words will be in revere order,
            // rev() the slice before use if you want them in the correct order.
            let mut images: Vec<[u64; 28 * 28]> = Vec::new();
            for _ in 0..size {
                file.read_exact(&mut images_bytes).expect("can't read images");
                let mut image = [0u64; 28 * 28];
                for p in 0..784 {
                    let mut ones = 0u64;
                    for i in 0..images_bytes[p] / 4 {
                        ones = ones | 1 << i;
                    }
                    image[p] = ones;
                }
                images.push(image);
            }
            return images;
        }
    }
}

#[macro_use]
pub mod params {
    #[macro_export]
    macro_rules! i16_1d {
        ($name:ident, $size:expr) => {
            #[derive(Clone, Copy)]
            struct $name {
                thresholds: [i16; $size],
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "1d i16 [{:?}]", $size)
                }
            }
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    for i in 0..$size {
                        if self.thresholds[i] != other.thresholds[i] {
                            return false;
                        }
                    }
                    true
                }
            }
            impl $name {
                fn eq(&self, other: &Self) -> bool {
                    for i in 0..$size {
                        if self.thresholds[i] != other.thresholds[i] {
                            return false;
                        }
                    }
                    true
                }
                fn new_nil() -> $name {
                    $name { thresholds: [0i16; $size] }
                }
                fn new_const(const_val: i16) -> $name {
                    $name {
                        thresholds: [const_val; $size],
                    }
                }
                fn child(&self, random_ints: &Fn() -> i16) -> $name {
                    let mut child = $name::new_nil();
                    for i in 0..$size {
                        child.thresholds[i] = self.thresholds[i] + random_ints();
                    }
                    child
                }
                fn write(&self, wtr: &mut Vec<u8>) {
                    for i in 0..$size {
                        wtr.write_i16::<BigEndian>(self.thresholds[i]).unwrap();
                    }
                }
                fn read(rdr: &mut std::io::Read) -> $name {
                    let mut object = $name::new_nil();
                    for i in 0..$size {
                        object.thresholds[i] = rdr.read_i16::<BigEndian>().unwrap();
                    }
                    object
                }
            }
        };
    }
    #[macro_export]
    macro_rules! u64_2d {
        ($name:ident, $dim0:expr, $dim1:expr) => {
            #[derive(Clone, Copy)]
            struct $name {
                weights: [[u64; $dim1]; $dim0],
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "2d u64 [{:?}, {:?}]", $dim0, $dim1)
                }
            }
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            if self.weights[d0][d1] != other.weights[d0][d1] {
                                return false;
                            }
                        }
                    }
                    true
                }
            }
            impl $name {
                fn new_nil() -> $name {
                    $name {
                        weights: [[0u64; $dim1]; $dim0],
                    }
                }
                fn new_random() -> $name {
                    let mut new_params = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            new_params.weights[d0][d1] = rand::random::<u64>();
                        }
                    }
                    new_params
                }
                fn child(&self, random_bits: &Fn() -> u64) -> $name {
                    let mut child = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            child.weights[d0][d1] = self.weights[d0][d1] ^ random_bits();
                        }
                    }
                    child
                }
                fn write(&self, wtr: &mut Vec<u8>) {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            wtr.write_u64::<BigEndian>(self.weights[d0][d1]).unwrap();
                        }
                    }
                }
                fn read(rdr: &mut std::io::Read) -> $name {
                    let mut object = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            object.weights[d0][d1] = rdr.read_u64::<BigEndian>().unwrap();
                        }
                    }
                    object
                }
            }
        };
    }
    #[macro_export]
    macro_rules! u32_2d {
        ($name:ident, $dim0:expr, $dim1:expr) => {
            #[derive(Clone, Copy)]
            struct $name {
                weights: [[u32; $dim1]; $dim0],
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "2d u32 [{:?}, {:?}]", $dim0, $dim1)
                }
            }
            impl $name {
                fn new_nil() -> $name {
                    $name {
                        weights: [[0u32; $dim1]; $dim0],
                    }
                }
                fn new_random() -> $name {
                    let mut new_params = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            new_params.weights[d0][d1] = rand::random::<u32>();
                        }
                    }
                    new_params
                }
                fn child(&self, random_bits: &Fn() -> u32) -> $name {
                    let mut child = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            child.weights[d0][d1] = self.weights[d0][d1] ^ random_bits();
                        }
                    }
                    child
                }
                fn write(&self, wtr: &mut Vec<u8>) {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            wtr.write_u32::<BigEndian>(self.weights[d0][d1]).unwrap();
                        }
                    }
                }
                fn read(rdr: &mut std::io::Read) -> $name {
                    let mut object = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            object.weights[d0][d1] = rdr.read_u32::<BigEndian>().unwrap();
                        }
                    }
                    object
                }
            }
        };
    }
    #[macro_export]
    macro_rules! u64_3d {
        ($name:ident, $dim0:expr, $dim1:expr, $dim2:expr) => {
            #[derive(Clone, Copy)]
            struct $name {
                weights: [[[u64; $dim2]; $dim1]; $dim0],
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "3d u64 [{:?}, {:?}, {:?}]", $dim0, $dim1, $dim2)
                }
            }
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                if self.weights[d0][d1][d2] != other.weights[d0][d1][d2] {
                                    return false;
                                }
                            }
                        }
                    }
                    true
                }
            }

            impl $name {
                fn new_nil() -> $name {
                    $name {
                        weights: [[[0u64; $dim2]; $dim1]; $dim0],
                    }
                }
                fn new_random() -> $name {
                    let mut new_params = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                new_params.weights[d0][d1][d2] = rand::random::<u64>();
                            }
                        }
                    }
                    new_params
                }
                fn child(&self, random_bits: &Fn() -> u64) -> $name {
                    let mut child = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                child.weights[d0][d1][d2] = self.weights[d0][d1][d2] ^ random_bits();
                            }
                        }
                    }
                    child
                }
                fn write(&self, wtr: &mut Vec<u8>) {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                wtr.write_u64::<BigEndian>(self.weights[d0][d1][d2]).unwrap();
                            }
                        }
                    }
                }
                fn read(rdr: &mut std::io::Read) -> $name {
                    let mut object = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                object.weights[d0][d1][d2] = rdr.read_u64::<BigEndian>().unwrap();
                            }
                        }
                    }
                    object
                }
            }
        };
    }
    #[macro_export]
    macro_rules! make_0val {
        (u8) => {
            0u8
        };
        (u16) => {
            0u16
        };
        (u32) => {
            0u32
        };
        (u64) => {
            0u64
        };
        (i8) => {
            0i8
        };
        (i16) => {
            0i16
        };
        (i32) => {
            0i32
        };
        (i64) => {
            0i64
        };
    }
    #[macro_export]
    macro_rules! read_type {
        ($rdr:expr,u8) => {
            $rdr.read_u8::<BigEndian>()
        };
        ($rdr:expr,u16) => {
            $rdr.read_u16::<BigEndian>()
        };
        ($rdr:expr,u32) => {
            $rdr.read_u32::<BigEndian>()
        };
        ($rdr:expr,u64) => {
            $rdr.read_u64::<BigEndian>()
        };
    }
    #[macro_export]
    macro_rules! write_type {
        ($wtr:expr, $val:expr,u8) => {
            $wtr.write_u8::<BigEndian>($val)
        };
        ($wtr:expr, $val:expr,u16) => {
            $wtr.write_u16::<BigEndian>($val)
        };
        ($wtr:expr, $val:expr,u32) => {
            $wtr.write_u32::<BigEndian>($val)
        };
        ($wtr:expr, $val:expr,u64) => {
            $wtr.write_u64::<BigEndian>($val)
        };
    }
    #[macro_export]
    macro_rules! prms_2d {
        ($name:ident, $type:ident, $dim0:expr, $dim1:expr) => {
            #[derive(Clone, Copy)]
            struct $name {
                weights: [[$type; $dim1]; $dim0],
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "2d params [{:?}, {:?}]", $dim0, $dim1)
                }
            }
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            if self.weights[d0][d1] != other.weights[d0][d1] {
                                return false;
                            }
                        }
                    }
                    true
                }
            }
            impl $name {
                fn new_nil() -> $name {
                    $name {
                        weights: [[make_0val!($type); $dim1]; $dim0],
                    }
                }
                fn new_random() -> $name {
                    let mut new_params = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            new_params.weights[d0][d1] = rand::random::<$type>();
                        }
                    }
                    new_params
                }
                fn child(&self, random_bits: &Fn() -> $type) -> $name {
                    let mut child = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            child.weights[d0][d1] = self.weights[d0][d1] ^ random_bits();
                        }
                    }
                    child
                }
                fn write(&self, wtr: &mut Vec<u8>) {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            write_type!(wtr, self.weights[d0][d1], $type).unwrap();
                        }
                    }
                }
                fn read(rdr: &mut std::io::Read) -> $name {
                    let mut object = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            object.weights[d0][d1] = read_type!(rdr, $type).unwrap();
                        }
                    }
                    object
                }
            }
        };
    }

    #[macro_export]
    macro_rules! prms_3d {
        ($name:ident, $type:ident, $dim0:expr, $dim1:expr, $dim2:expr) => {
            #[derive(Clone, Copy)]
            struct $name {
                weights: [[[$type; $dim2]; $dim1]; $dim0],
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "3d params [{:?}, {:?}, {:?}]", $dim0, $dim1, $dim2)
                }
            }
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                if self.weights[d0][d1][d2] != other.weights[d0][d1][d2] {
                                    return false;
                                }
                            }
                        }
                    }
                    true
                }
            }
            impl $name {
                fn new_nil() -> $name {
                    $name {
                        weights: [[[make_0val!($type); $dim2]; $dim1]; $dim0],
                    }
                }
                fn new_random() -> $name {
                    let mut new_params = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                new_params.weights[d0][d1][d2] = rand::random::<$type>();
                            }
                        }
                    }
                    new_params
                }
                fn child(&self, random_bits: &Fn() -> $type) -> $name {
                    let mut child = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                child.weights[d0][d1][d2] = self.weights[d0][d1][d2] ^ random_bits();
                            }
                        }
                    }
                    child
                }
                fn write(&self, wtr: &mut Vec<u8>) {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                write_type!(wtr, self.weights[d0][d1][d2], $type).unwrap();
                            }
                        }
                    }
                }
                fn read(rdr: &mut std::io::Read) -> $name {
                    let mut object = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                object.weights[d0][d1][d2] = read_type!(rdr, $type).unwrap();
                            }
                        }
                    }
                    object
                }
            }
        };
    }
    #[macro_export]
    macro_rules! prms_4d {
        ($name:ident, $type:ident, $dim0:expr, $dim1:expr, $dim2:expr, $dim3:expr) => {
            #[derive(Clone, Copy)]
            struct $name {
                weights: [[[[$type; $dim3]; $dim2]; $dim1]; $dim0],
            }
            impl fmt::Debug for $name {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    write!(f, "4d params [{:?}, {:?}, {:?}, {:?}]", $dim0, $dim1, $dim2, $dim3)
                }
            }
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                for d3 in 0..$dim3 {
                                    if self.weights[d0][d1][d2][d3] != other.weights[d0][d1][d2][d3] {
                                        return false;
                                    }
                                }
                            }
                        }
                    }
                    true
                }
            }
            impl $name {
                fn new_nil() -> $name {
                    $name {
                        weights: [[[[make_0val!($type); $dim3]; $dim2]; $dim1]; $dim0],
                    }
                }
                fn new_random() -> $name {
                    let mut new_params = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                for d3 in 0..$dim3 {
                                    new_params.weights[d0][d1][d2][d3] = rand::random::<$type>();
                                }
                            }
                        }
                    }
                    new_params
                }
                fn child(&self, random_bits: &Fn() -> $type) -> $name {
                    let mut child = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                for d3 in 0..$dim3 {
                                    child.weights[d0][d1][d2][d3] = self.weights[d0][d1][d2][d3] ^ random_bits();
                                }
                            }
                        }
                    }
                    child
                }
                fn write(&self, wtr: &mut Vec<u8>) {
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                for d3 in 0..$dim3 {
                                    write_type!(wtr, self.weights[d0][d1][d2][d3], $type).unwrap();
                                }
                            }
                        }
                    }
                }
                fn read(rdr: &mut std::io::Read) -> $name {
                    let mut object = $name::new_nil();
                    for d0 in 0..$dim0 {
                        for d1 in 0..$dim1 {
                            for d2 in 0..$dim2 {
                                for d3 in 0..$dim3 {
                                    object.weights[d0][d1][d2][d3] = read_type!(rdr, $type).unwrap();
                                }
                            }
                        }
                    }
                    object
                }
            }
        };
    }
}

#[macro_use]
pub mod layers {
    extern crate rand;
    use rand::Rng;
    #[macro_export]
    macro_rules! random_bits {
        ($pow:expr, $type:ty) => {
            || -> $type {
                let mut val = rand::random::<$type>();
                for _ in 0..$pow {
                    val = val & rand::random::<$type>();
                }
                val
            }
        };
    }
    pub fn random_int_plusminus_one() -> i16 {
        rand::thread_rng().gen_range(-1, 2)
    }
    #[macro_export]
    macro_rules! clamp_3d {
        ($type:ident, $min:expr, $max:expr, $dim0:expr, $dim1:expr, $dim2:expr) => {
            |input: &[[[$type; $dim2]; $dim1]; $dim0]| -> [[[$type; $dim2]; $dim1]; $dim0] {
                let mut output = [[[make_0val!($type); $dim2]; $dim1]; $dim0];
                for d0 in 0..$dim0 {
                    for d1 in 0..$dim1 {
                        for d2 in 0..$dim2 {
                            output[d0][d1][d2] = input[d0][d1][d1].max($min).min($max);
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! downshift_1d {
        ($type:ident, $pow:expr, $size:expr) => {
            |input: &[$type; $size]| -> [$type; $size] {
                let mut output = [make_0val!($type); $size];
                for i in 0..$size {
                    output[i] = input[i] >> $pow;
                }
                output
            }
        };
    }

    #[macro_export]
    macro_rules! downshift_3d {
        ($type:ident, $pow:expr, $dim0:expr, $dim1:expr, $dim2:expr) => {
            |input: &[[[$type; $dim2]; $dim1]; $dim0]| -> [[[$type; $dim2]; $dim1]; $dim0] {
                let mut output = [[[make_0val!($type); $dim2]; $dim1]; $dim0];
                for d0 in 0..$dim0 {
                    for d1 in 0..$dim1 {
                        for d2 in 0..$dim2 {
                            output[d0][d1][d2] = input[d0][d1][d1] >> $pow;
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! log_dist_1d {
        ($type:ty, $prefix:expr, $size:expr) => {
            |input: &[$type; $size]| {
                let mut sum = 0;
                let mut max = 0;
                let mut min = 0;
                for i in 0..$size {
                    if input[i] > max {
                        max = input[i];
                    }
                    if input[i] < min {
                        min = input[i];
                    }
                    sum += input[i];
                }
                let avg = sum as f64 / $size as f64;
                println!("{:?} {:?} {:?} {:?}", $prefix, min, avg, max);
            }
        };
    }

    #[macro_export]
    macro_rules! log_dist {
        ($type:ty, $prefix:expr, $dim0:expr, $dim1:expr, $dim2:expr) => {
            |input: &[[[$type; $dim2]; $dim1]; $dim0]| {
                let mut sum = 0;
                let mut max = 0;
                let mut min = 0;
                for d0 in 0..$dim0 {
                    for d1 in 0..$dim1 {
                        for d2 in 0..$dim2 {
                            if input[d0][d1][d2] > max {
                                max = input[d0][d1][d2];
                            }
                            if input[d0][d1][d2] < min {
                                min = input[d0][d1][d2];
                            }
                            sum += input[d0][d1][d2];
                        }
                    }
                }
                let avg = sum as f64 / ($dim0 * $dim1 * $dim2) as f64;
                println!("{:?} {:?} {:?} {:?}", $prefix, min, avg, max);
            }
        };
    }
    #[macro_export]
    macro_rules! conv_3x3_u32 {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr, $max:expr, $shift:expr) => {
            fn $name(
                input: &[[[i32; $in_chans * 32]; $y_size]; $x_size],
                filter: &[u8; $in_chans * $out_chans * 3 * 3],
            ) -> [[[u32; $out_chans]; $y_size]; $x_size] {
                let mut output = [[[0u32; $out_chans]; $y_size]; $x_size];
                for x in 1..$x_size - 1 {
                    // for all the pixels in the output, inset by one
                    for y in 1..$y_size - 1 {
                        for o in 0..$out_chans {
                            let mut sum = 0i32;
                            let o_offset = o * $in_chans * 3 * 3;
                            for iw in 0..$in_chans {
                                let i_offset = o_offset + iw * 3 * 3;
                                for ib in 0..8 {
                                    let mask = 0b1u8 << ib;
                                    let i = iw * 8 + ib;
                                    for ix in 0..3 {
                                        let ix_offset = i_offset + ix * 3;
                                        for iy in 0..3 {
                                            if (filter[ix_offset + iy] & mask) == mask {
                                                sum += input[x + ix - 1][y + iy - 1][i];
                                            } else {
                                                sum -= input[x + ix - 1][y + iy - 1][i];
                                            }
                                        }
                                    }
                                }
                            }
                            output[x][y][o] = (sum >> $shift).max(0).min($max) as u32;
                        }
                    }
                }
                output
            }
        };
    }

    #[macro_export]
    macro_rules! conv3x3_i32 {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr) => {
            fn $name(
                input: &[[[i32; $in_chans * 32]; $y_size]; $x_size],
                filter: &[[[[u32; 3]; 3]; $in_chans]; $out_chans],
            ) -> [[[i32; $out_chans]; $y_size]; $x_size] {
                let mut output = [[[0i32; $out_chans]; $y_size]; $x_size];
                for x in 1..$x_size - 1 {
                    // for all the pixels in the output, inset by one
                    for y in 1..$y_size - 1 {
                        for o in 0..$out_chans {
                            for ib in 0..32 {
                                let mask = 0b1u32 << ib;
                                for iw in 0..$in_chans {
                                    let i = iw * 32 + ib;
                                    for ix in 0..3 {
                                        for iy in 0..3 {
                                            if (filter[o][iw][ix][iy] & mask) == mask {
                                                output[x][y][o] += input[x + ix - 1][y + iy - 1][i];
                                            } else {
                                                output[x][y][o] -= input[x + ix - 1][y + iy - 1][i];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                output
            }
        };
    }
    // single channel input variant.
    // output will by multiplied by 32.
    #[macro_export]
    macro_rules! conv3x3_u8_1chan_i32 {
        ($name:ident, $x_size:expr, $y_size:expr, $out_chans:expr) => {
            fn $name(input: &[[u8; $y_size]; $x_size], filter: &[[[u32; 3]; 3]; $out_chans]) -> [[[i32; $out_chans * 32]; $y_size]; $x_size] {
                let mut output = [[[0i32; $out_chans * 32]; $y_size]; $x_size];
                for x in 1..$x_size - 1 {
                    // for all the pixels in the output, inset by one
                    for y in 1..$y_size - 1 {
                        for ob in 0..32 {
                            let mask = 0b1u32 << ob;
                            for ow in 0..$out_chans {
                                let o = ow * 32 + ob;
                                for ix in 0..3 {
                                    for iy in 0..3 {
                                        if (filter[ow][ix][iy] & mask) == mask {
                                            output[x][y][o] += input[x + ix - 1][y + iy - 1] as i32;
                                        } else {
                                            output[x][y][o] -= input[x + ix - 1][y + iy - 1] as i32;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                output
            }
        };
    }

    #[macro_export]
    macro_rules! conv3x3 {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr, $bias_size:expr) => {
            fn $name(
                input: &[[[u64; $in_chans]; $y_size]; $x_size],
                weights: &[[[u64; $in_chans]; 9]; $out_chans * 64],
            ) -> [[[u64; $out_chans]; $y_size]; $x_size] {
                let mut output = [[[0u64; $out_chans]; $y_size]; $x_size];
                let threshold = (9 * 32 * $in_chans) as u32;
                for x in 1..$x_size - 1 { // for all the pixels in the output, inset by one
                    for y in 1..$y_size - 1 {
                        for ow in 0..$out_chans { // for each word of the output channels,
                            for ob in 0..64 - $bias_size { // for each of the 64 bits of that output word ignoring the bias bits
                                let wi = ow * 64 + ob;
                                let mut sum = 0; // sum holds all the 3 * 3 * input_chans * 64 bits.
                                for iw in 0..$in_chans { // for each word of the input,
                                    // we take each of the 9 pixels of the input patch and xor the weight for that [output_chan, pixel, input_chan], with the input
                                   sum += (weights[wi][0][iw] ^ input[x - 1][y - 1][iw]).count_ones()
                                        + (weights[wi][1][iw] ^ input[x + 0][y - 1][iw]).count_ones()
                                        + (weights[wi][2][iw] ^ input[x + 1][y - 1][iw]).count_ones()
                                        + (weights[wi][3][iw] ^ input[x - 1][y + 0][iw]).count_ones()
                                        + (weights[wi][4][iw] ^ input[x + 0][y + 0][iw]).count_ones()
                                        + (weights[wi][5][iw] ^ input[x + 1][y + 0][iw]).count_ones()
                                        + (weights[wi][6][iw] ^ input[x - 1][y + 1][iw]).count_ones()
                                        + (weights[wi][7][iw] ^ input[x + 0][y + 1][iw]).count_ones()
                                        + (weights[wi][8][iw] ^ input[x + 1][y + 1][iw]).count_ones();
                                }
                                output[x][y][ow] = output[x][y][ow] | (((sum > threshold) as u64) << ob);
                            }
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! conv3x3_5sparse {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr, $bias_size:expr) => {
            fn $name(
                input: &[[[u64; $in_chans]; $y_size]; $x_size],
                weights: &[[[u64; $in_chans]; 5]; $out_chans * 64],
            ) -> [[[u64; $out_chans]; $y_size]; $x_size] {
                let mut output = [[[0u64; $out_chans]; $y_size]; $x_size];
                let threshold = (5 * 32 * $in_chans) as u32;
                for x in 1..$x_size - 1 { // for all the pixels in the output, inset by one
                    for y in 1..$y_size - 1 {
                        for ow in 0..$out_chans { // for each word of the output channels,
                            for ob in 0..64 - $bias_size { // for each of the 64 bits of that output word,
                                let wi = ow * 64 + ob;
                                let mut sum = 0; // sum holds all the 3 * 3 * input_chans * 64 bits.
                                for iw in 0..$in_chans { // for each word of the input,
                                    // we take each of the 9 pixels of the input patch and xor the weight for that [output_chan, pixel, input_chan], with the input
                                   sum += (weights[wi][0][iw] ^ input[x + 0][y + 0][iw]).count_ones()
                                        + (weights[wi][1][iw] ^ input[x + 0][y - 1][iw]).count_ones()
                                        + (weights[wi][2][iw] ^ input[x + 0][y + 1][iw]).count_ones()
                                        + (weights[wi][3][iw] ^ input[x - 1][y + 0][iw]).count_ones()
                                        + (weights[wi][4][iw] ^ input[x + 1][y + 0][iw]).count_ones();
                                }
                                output[x][y][ow] = output[x][y][ow] | (((sum > threshold) as u64) << ob);
                            }
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! conv1x1 {
        ($name:ident, $x_size:expr, $y_size:expr, $in_chans:expr, $out_chans:expr, $bias_size:expr) => {
            fn $name(
                input: &[[[u64; $in_chans]; $x_size]; $y_size],
                weights: &[[u64; $in_chans]; $out_chans * 64],
            ) -> [[[u64; $out_chans]; $x_size]; $y_size] {
                let mut output = [[[0u64; $out_chans]; $x_size]; $y_size];
                let threshold = (32 * $in_chans) as u32;
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        for ow in 0..$out_chans {
                            for ob in 0..64 - $bias_size {
                                let mut sum = 0;
                                for iw in 0..$in_chans {
                                    sum += (weights[ow * 64 + ob][iw] ^ input[x][y][iw]).count_ones()
                                }
                                output[x][y][ow] = output[x][y][ow] | (((sum > threshold) as u64) << ob);
                            }
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! concat3d {
        ($name:ident, $x_size:expr, $y_size:expr, $a_chan:expr, $b_chans:expr) => {
            fn $name(
                input_a: &[[[u64; $a_chans]; $y_size]; $x_size],
                input_b: &[[[u64; $b_chans]; $y_size]; $x_size],
            ) -> [[[u64; $a_chan + $b_chans]; $y_size]; $x_size] {
                let mut output = [[[0u64; $a_chan + $b_chans]; $x_size]; $y_size];
                for x in 0..$x_size {
                    for y in 0..$y_size {
                        for chan in 0..$a_chan {
                            output[x][y][chan] = input_a[x][y][chan];
                        }
                        for chan in 0..$b_chan {
                            output[x][y][$a_chan + chan] = input_a[x][y][chan];
                        }
                    }
                }
            }
        };
    }
    /// 2 by 2 or pooling. Takes 2x2 patches, `or`s the 4 bits together. Reduces image size by a factor of 2 in each dimention.
    #[macro_export]
    macro_rules! max_pool {
        ($name:ident, $type:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[$type; $num_chans]; $x_size]; $y_size]) -> [[[$type; $num_chans]; $y_size / 2]; $x_size / 2] {
                let mut output = [[[make_0val!($type); $num_chans]; $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let ix = x * 2;
                    for y in 0..$y_size / 2 {
                        let iy = y * 2;
                        for chan in 0..$num_chans {
                            output[x][y][chan] = input[ix + 0][iy + 0][chan]
                                .max(input[ix + 1][iy + 0][chan])
                                .max(input[ix + 0][iy + 1][chan])
                                .max(input[ix + 1][iy + 1][chan]);
                        }
                    }
                }
                output
            }
        };
    }
    /// 2 by 2 or pooling. Takes 2x2 patches, `or`s the 4 bits together. Reduces image size by a factor of 2 in each dimention.
    #[macro_export]
    macro_rules! pool_or2x2 {
        ($name:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[u64; $num_chans]; $x_size]; $y_size]) -> [[[u64; $num_chans]; $y_size / 2]; $x_size / 2] {
                let mut output = [[[0u64; $num_chans]; $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let ix = x * 2;
                    for y in 0..$y_size / 2 {
                        let iy = y * 2;
                        for chan in 0..$num_chans {
                            output[x][y][chan] =
                                input[ix + 0][iy + 0][chan] | input[ix + 1][iy + 0][chan] | input[ix + 0][iy + 1][chan] | input[ix + 1][iy + 1][chan];
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! pool_and2x2 {
        ($name:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[u64; $num_chans]; $x_size]; $y_size]) -> [[[u64; $num_chans]; $y_size / 2]; $x_size / 2] {
                let mut output = [[[0u64; $num_chans]; $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    for y in 0..$y_size / 2 {
                        for chan in 0..$num_chans {
                            output[x][y][chan] = (input[x * 2 + 0][y * 2 + 0][chan]
                                & input[x * 2 + 1][y * 2 + 0][chan]
                                & input[x * 2 + 0][y * 2 + 1][chan]
                                & input[x * 2 + 1][y * 2 + 1][chan]);
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! pool_andor2x2 {
        ($name:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[u64; $num_chans]; $x_size]; $y_size]) -> [[[u64; $num_chans]; $y_size / 2]; $x_size / 2] {
                let mut output = [[[0u64; $num_chans]; $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    for y in 0..$y_size / 2 {
                        for chan in 0..$num_chans {
                            output[x][y][chan] = (input[x * 2 + 0][y * 2 + 0][chan] & input[x * 2 + 1][y * 2 + 0][chan])
                                | (input[x * 2 + 0][y * 2 + 1][chan] & input[x * 2 + 1][y * 2 + 1][chan]);
                        }
                    }
                }
                output
            }
        };
    }
    /// takes [x, y, chan] (or any other 3 dimentional vector)
    // assumes that the outer one pixels are empty.
    #[macro_export]
    macro_rules! flatten3d {
        ($name:ident, $type:ident, $x_size:expr, $y_size:expr, $num_chans:expr) => {
            fn $name(input: &[[[$type; $num_chans]; $x_size]; $y_size]) -> [$type; ($x_size - 2) * ($y_size - 2) * $num_chans] {
                let mut output = [make_0val!($type); ($x_size - 2) * ($y_size - 2) * $num_chans];
                let mut index = 0;
                for x in 1..$x_size - 1 {
                    for y in 1..$y_size - 1 {
                        for c in 0..$num_chans {
                            output[index] = input[x][y][c];
                            index += 1;
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! dense_bits2bits {
        ($name:ident, $input_size:expr, $output_size:expr, $bias_size:expr) => {
            fn $name(
                input: &[u64; $input_size],
                weights: &[[u64; $input_size]; $output_size * 64],
                thresholds: &[i16; $output_size * 64],
            ) -> [u64; $output_size] {
                let mut output = [0u64; $output_size];
                for o in 0..$output_size {
                    for b in 0..64 - $bias_size {
                        let mut sum = 0u32;
                        for i in 0..$input_size {
                            sum += (self.weights[o * 64 + b][i] ^ input[i]).count_ones();
                        }
                        output[o] = output[o] | (((sum as i16 > self.thresholds[o * 64 + b]) as u64) << b);
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! bitwise_onehot {
        ($name:ident, $size:expr) => {
            fn $name(input: u8) -> [u64; $size] {
                let mut output = [0u64; $size];
                output[(input / 64) as usize] = 0b1u64 << (input % 64);
                output
            }
        };
    }
    #[macro_export]
    macro_rules! bitwise_onehot_seq {
        ($name:ident, $seq_len:expr, $size:expr) => {
            fn $name(input: &[u8; $seq_len]) -> [[u64; $size]; $seq_len] {
                let mut output = [[0u64; $size]; $seq_len];
                for e in 0..$seq_len {
                    output[e][input[e] / 64] = 0b1u64 << (input[e] % 64);
                }
            }
        };
    }
    #[macro_export]
    macro_rules! lstm_cell {
        ($name:ident, $state_size:expr, $input_size:expr) => {
            fn $name(
                x: &[u64; $input_size],
                h: &[u64; $state_size],
                c: &mut [i32; $state_size * 64],
                weights: &[[[u64; $state_size + $input_size]; $state_size * 64]; 4],
            ) -> [u64; $state_size] {
                let input = {
                    let mut input = [0u64; ($input_size + $state_size)];
                    for i in 0..$input_size {
                        input[i] = x[i];
                    }
                    for i in 0..$state_size {
                        input[i + $input_size] = h[i];
                    }
                    input
                };
                //println!("input: {:?}", input);
                let threshold = (($input_size + $state_size) * 64 / 2) as u32;
                let tmp_values = {
                    let mut values = [[false; $state_size * 64]; 4];
                    for g in 0..4 {
                        for s in 0..$state_size {
                            for b in 0..64 {
                                let si = s * 64 + b;
                                let mut sum = 0;
                                for i in 0..$input_size + $state_size {
                                    sum += (weights[g][si][i] ^ input[i]).count_ones();
                                }
                                //if (g == 2) & (s == 1) & (b == 42) {
                                //    //println!("sum: {:?} {:?}", sum, threshold);
                                //}
                                values[g][si] = sum >= threshold;
                            }
                        }
                    }
                    values
                };
                //println!("{:?} {:?} {:?}", tmp_values[0][10], tmp_values[0][60], tmp_values[0][70]);
                // forget gate
                for s in 0..$state_size * 64 {
                    c[s] = c[s] * tmp_values[0][s] as i32;
                }
                //println!("c: {:?}", (c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]));
                // update
                for s in 0..$state_size * 64 {
                    c[s] = c[s] + (tmp_values[2][s] as i8 * (tmp_values[1][s] as i8 * 2 - 1)) as i32;
                }
                // we are now done updating the state.
                let mut output = [0u64; $state_size];
                for i in 0..$state_size {
                    for b in 0..64 {
                        output[i] = output[i] | (((tmp_values[3][i * 64 + b]) & (c[i * 64 + b] > 3)) as u64) << b;
                    }
                }
                //println!("h: {:064b}", h[0]);
                output
            }
        };
    }
    #[macro_export]
    macro_rules! rnn_cell {
        ($name:ident, $state_size:expr, $input_size:expr) => {
            fn $name(
                x: &[u64; $input_size],
                h: &[u64; $state_size],
                weights: &[[u64; $state_size + $input_size]; $state_size * 64],
            ) -> [u64; $state_size] {
                let mut output = [0u64; $state_size];
                let threshold = (($input_size + $state_size) * 32) as u32;
                for s in 0..$state_size {
                    for b in 0..64 {
                        let mut sum = 0;
                        let wi = s * 64 + b;
                        for i in 0..$input_size {
                            sum += (weights[wi][i] ^ x[i]).count_ones();
                        }
                        for i in 0..$state_size {
                            sum += (weights[wi][$input_size + i] ^ h[i]).count_ones();
                        }
                        output[s] = output[s] | (((sum > threshold) as u64) << b);
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! rnn_unrolled {
        ($name:ident, $seq_len:expr, $state_size:expr, $input_size:expr, $output_size:expr, $bias_size:expr) => {
            fn $name(
                input: &[[u64; $input_size]; $seq_len],
                rnn_weights: &[[u64; $state_size + $input_size]; $state_size * 64],
                output_weights: &[[u64; $state_size]; $output_size],
                state: &mut [u64; $state_size],
            ) -> [[u32; $output_size]; $seq_len] {
                let threshold = (($state_size + $input_size) * 64 / 2) as u32;
                let mut output = [[0u32; $output_size]; $seq_len];
                for e in 0..$seq_len {
                    let mut state_sums = [0u32; $state_size * 64];
                    // first we must calculate the total popcounts for each state.
                    // note that it it is not yet safe to mutate the state, we need it intact for all of this loop.
                    for s in 0..$state_size {
                        for b in 0..64 {
                            let weight_index = s * 64 + b;
                            for i in 0..$input_size {
                                state_sums[s] += (rnn_weights[weight_index][i] ^ input[e][i]).count_ones();
                            }
                            for i in $input_size..$input_size + $state_size {
                                state_sums[s] += (rnn_weights[weight_index][i] ^ state[i]).count_ones();
                            }
                        }
                    }
                    // now it is safe to reset each word of the state to 0 and replace it with the bits of the new state.
                    for s in 0..$state_size {
                        state[s] = 0;
                        for b in 0..64 - $bias_size {
                            state[s] = state[s] | (((state_sums[s * 64 + b] > threshold) as u64) << b);
                        }
                    }
                    // and finaly we can calculate the outputs.
                    for o in 0..$output_size {
                        for s in 0..$state_size {
                            output[e][o] += (output_weights[o][s] ^ state[s]).count_ones();
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! dense_bits2ints {
        ($name:ident, $input_size:expr, $output_size:expr) => {
            fn $name(input: &[u64; $input_size], weights: &[[u64; $input_size]; $output_size]) -> [u32; $output_size] {
                let mut output = [0u32; $output_size];
                for i in 0..$input_size {
                    for o in 0..$output_size {
                        output[o] += (input[i] ^ weights[o][i]).count_ones();
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! dense_ints2ints {
        ($name:ident, $input_size:expr, $output_size:expr) => {
            fn $name(input: &[i32; $input_size * 32], weights: &[[u32; $input_size]; $output_size]) -> [i32; $output_size] {
                let mut output = [0i32; $output_size];
                for i in 0..$input_size {
                    for b in 0..32 {
                        let index = i * 32 + b;
                        let mask = 0b1u32 << b;
                        for o in 0..$output_size {
                            if (weights[o][i] & mask) == mask {
                                output[o] += input[index];
                            } else {
                                output[o] -= input[index];
                            }
                        }
                    }
                }
                output
            }
        };
    }
    #[macro_export]
    macro_rules! int_loss {
        ($name:ident, $size:expr) => {
            fn $name(actual: &[i16; $size], target: &[i16; $size]) -> i64 {
                actual
                    .iter()
                    .zip(target.iter())
                    .map(|(&a, &t)| {
                        let diff = (a - t) as i64;
                        //println!("diff: {:?}", diff * diff);
                        diff * diff
                    })
                    .sum()
            }
        };
    }
    #[macro_export]
    macro_rules! max_index {
        ($name:ident, $size:expr, $type:ty) => {
            fn $name(input: [u32; $size]) -> $type {
                let mut max = 0u32;
                let mut index = 0;
                for i in 0..$size {
                    if input[i] > max {
                        max = input[i];
                        index = i;
                    }
                }
                index as $type
            }
        };
    }
}
#[cfg(test)]
mod tests {
    extern crate byteorder;
    extern crate rand;

    use test::Bencher;
    const C0: usize = 1;
    const C1: usize = 2;
    conv3x3!(conv30, 30, 30, C0, C1, 3);
    pool_or2x2!(pool30, 30, 30, C0);
    flatten3d!(flatten, 30, 30, C1);
    rnn_unrolled!(rnn_layer, 7, 2, 5, 3);

    //input: &[[u64; $input_size]; $seq_len],
    //rnn_weights: &[[u64; $state_size + $input_size]; $state_size * 64],
    //output_weights: &[[u64; $state_size]; $output_size],
    //state: &mut [u64; $state_size],

    #[test]
    fn rnn_test() {
        let input = [[12345u64; 5]; 7];
        let rnn_weights = [[1234567u64; 2 + 5]; 2 * 64];
        let output_weights = [[1234u64; 2]; 3];
        let mut state = [0u64; 2 * 64];
        let output = rnn_layer(&input, &rnn_weights, &output_weights, &mut state);
        println!("output: {:?}", output);
        println!("state: {:?}", state[2]);
    }

    #[bench]
    fn conv30_bench(b: &mut Bencher) {
        let params = [[[0u64; C0]; 9]; C1 * 64];
        let image = [[[0u64; C0]; 30]; 30];
        b.iter(|| {
            conv30(&image, &params);
        });
    }
    #[bench]
    fn pool30_bench(b: &mut Bencher) {
        let image = [[[0u64; C0]; 30]; 30];
        b.iter(|| {
            pool30(&image);
        });
    }
    #[bench]
    fn flatten30_bench(b: &mut Bencher) {
        let image = [[[0u64; C1]; 30]; 30];
        b.iter(|| {
            flatten(&image);
        });
    }
}
