#![feature(int_log)]
#![feature(generic_const_exprs)]
#![feature(avx512_target_feature)]
#![feature(stdsimd)]

use bnn::bits::{b32, b64, b8, BitArray, GetBit};
use bnn::bitslice::{
    bit_add, bit_add_wrapping, bit_splat, comparator, equality, extend, ragged_array_popcount,
    transpose_8, BitSlice, BlockTranspose,
};
//use bnn::matrix::{block_transpose_256, block_transpose_512};
use bnn::ecc::{decode_byte, encode_byte};
use bnn::shape::flatten_2d;
use rayon::prelude::*;
use std::arch::x86_64::{__m256i, __m512i, _mm256_setzero_si256};
use std::convert::TryFrom;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::mem;
use std::time::Instant;

fn main() {
    let bytes = std::fs::read("/big/temp/books_txt/Delphi Complete Works of Charles Dickens (Illustrated) - Charles Dickens.txt").unwrap();
    //let bytes: Vec<u8> = (0..(2usize.pow(25) + 5)).map(|i| (i % 50) as u8).collect();

    let expanded: Vec<[b64; 4]> = bytes.par_iter().map(|&b| encode_byte(b)).collect();
    //let expanded: Vec<[b64; 4]> = (0..(2usize.pow(24))).map(|_| [b64(0); 4]).collect();

    let window_start = Instant::now();
    let (mut input, mut target): (Vec<[b64; 8]>, Vec<[b64; 4]>) = expanded
        .par_windows(3)
        .map(|slice| {
            let input = *flatten_2d::<b64, 2, 4>(<&[[b64; 4]; 2]>::try_from(&slice[0..2]).unwrap());
            let mut target = slice[2];
            target[0].0 ^= input[0].0;
            target[1].0 ^= input[1].0 << 13;
            target[2].0 ^= input[3].0;
            target[3].0 ^= input[2].0;
            (input, target)
        })
        .take(2usize.pow(8))
        .unzip();

    //let input = __m256i::block_transpose(<&[[b64; 8]; 256]>::try_from(&input[0..]).unwrap());
    let block = __m256i::block_transpose(<&[[b64; 4]; 256]>::try_from(&target[0..]).unwrap());
    //dbg!(&input);
    //for i in 0..256 {
    //    let row: [b64; 4] = unsafe { mem::transmute(target[i]) };
    //    println!("{} {:?}", i, row);
    //}

    /*
    for threshold in 50..200 {
        //let threshold = 125;

        let sum: [__m256i; 10] = ragged_array_popcount(&block[0..]);
        let expanded_threshold: [__m256i; 10] = bit_splat(threshold);
        //dbg!(expanded_threshold);
        let (lt, _, gt) = comparator(&sum, &expanded_threshold);
        //dbg!(lt);
        //dbg!(gt);
        let count = gt.count_bits();
        dbg!(count);

        let real_count: u32 = target
            .iter()
            .map(|x| {
                let sum = x.iter().map(|x| x.0.count_ones()).sum::<u32>();
                //dbg!(sum);
                (sum > threshold) as u32
            })
            .sum();
        assert_eq!(count, real_count);
    }
    */
    let foo = b8(0b_11001111_u8);
    dbg!(foo.get_bit(1));
}
