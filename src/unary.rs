use crate::bits::{b32, b8, BitStates, BitWord};
use std::marker::PhantomData;

macro_rules! to_unary {
    ($name:ident, $b_type:ty, $len:expr) => {
        pub fn $name(input: u8) -> $b_type {
            !(<$b_type>::ONES << (input / (256 / $len) as u8) as usize)
        }
    };
}

to_unary!(to_10, b32, 10);
to_unary!(to_32, b32, 32);

pub trait Preprocess<Input> {
    type Output;
    fn preprocess(input: &Input) -> Self::Output;
}

pub struct Normalize<P> {
    post_process: PhantomData<P>,
}

impl<P: Preprocess<[u8; 3]>> Preprocess<[[[u8; 3]; 3]; 3]> for Normalize<P>
where
    P::Output: Default,
{
    type Output = [[P::Output; 3]; 3];
    fn preprocess(input: &[[[u8; 3]; 3]; 3]) -> [[P::Output; 3]; 3] {
        let mut mins = [255_u8; 3];
        for x in 0..3 {
            for y in 0..3 {
                for c in 0..3 {
                    mins[c] = input[x][y][c].min(mins[c]);
                }
            }
        }
        let mut target = <[[P::Output; 3]; 3]>::default();
        for x in 0..3 {
            for y in 0..3 {
                let mut pixel = [0_u8; 3];
                for c in 0..3 {
                    pixel[c] = input[x][y][c] - mins[c];
                }
                target[x][y] = P::preprocess(&pixel);
            }
        }
        target
    }
}

pub struct Unary<T> {
    output_type: PhantomData<T>,
}

impl<I, O, const L: usize> Preprocess<[I; L]> for Unary<[O; L]>
where
    Unary<O>: Preprocess<I, Output = O>,
    [O; L]: Default,
{
    type Output = [O; L];
    fn preprocess(input: &[I; L]) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = Unary::<O>::preprocess(&input[i]);
        }
        target
    }
}

impl Preprocess<u8> for Unary<b32> {
    type Output = b32;
    fn preprocess(&input: &u8) -> b32 {
        to_32(input)
    }
}

impl Preprocess<[u8; 3]> for Unary<b32> {
    type Output = b32;
    fn preprocess(input: &[u8; 3]) -> b32 {
        to_10(input[0]) | (to_10(input[1]) << 10) | (to_10(input[2]) << 20)
    }
}

pub struct Identity();

impl<I: Copy> Preprocess<I> for Identity {
    type Output = I;
    #[inline]
    fn preprocess(&input: &I) -> I {
        input
    }
}

pub struct Edges();

impl Preprocess<[[[u8; 3]; 3]; 3]> for Edges {
    type Output = b32;
    fn preprocess(input: &[[[u8; 3]; 3]; 3]) -> b32 {
        edges_from_patch(input)
    }
}

// Given a 3x3 patch of 3 color channels, extract the 32 edges.
// Each edge compares two sets of three pixels.
// For example horizontal is:
// |a|a|a|
// | | | |
// |b|b|b|
// while one diagonal is:
// |a|a| |
// |a| |b|
// | |b|b|
// center is ignored by all edges.
fn edges_from_patch(patch: &[[[u8; 3]; 3]; 3]) -> b32 {
    let mut target = b32::default();
    // horizontal
    target |= extract_partition(patch, [(0, 0), (0, 1), (0, 2)], [(2, 0), (2, 1), (2, 2)]) << 24;
    // vertical
    target |= extract_partition(patch, [(0, 0), (1, 0), (2, 0)], [(0, 2), (1, 2), (2, 2)]) << 16;
    // diagonal
    target |= extract_partition(patch, [(0, 2), (0, 1), (1, 2)], [(2, 0), (2, 1), (1, 2)]) << 8;
    // other diagonal
    target |= extract_partition(patch, [(0, 0), (1, 0), (0, 1)], [(2, 2), (1, 2), (2, 1)]) << 0;
    target
}

// Given the patch and the indices of the two sides of an edge, extract the 8 colors.
// It returns a b32, but only the first 8 bits are used.
fn extract_partition(
    patch: &[[[u8; 3]; 3]; 3],
    a: [(usize, usize); 3],
    b: [(usize, usize); 3],
) -> b32 {
    b32(color_features_from_partition(
        elementwise_sum_3(
            patch[a[0].0][a[0].1],
            patch[a[1].0][a[1].1],
            patch[a[2].0][a[2].1],
        ),
        elementwise_sum_3(
            patch[b[0].0][b[0].1],
            patch[b[1].0][b[1].1],
            patch[b[2].0][b[2].1],
        ),
    )
    .0 as u32)
}

// Sum together each of the three color channels.
// The sum is u16s to avoid overflows.
fn elementwise_sum_3(a: [u8; 3], b: [u8; 3], c: [u8; 3]) -> [u16; 3] {
    let mut sums = [0u16; 3];
    for i in 0..3 {
        sums[i] += a[i] as u16 + b[i] as u16 + c[i] as u16;
    }
    sums
}

// for each of the 8 combinations of the three color channels, compare the two sides of the edge.
// 0, which is black, is degenerate.
fn color_features_from_partition(a: [u16; 3], b: [u16; 3]) -> b8 {
    let mut target = b8::default();
    for i in 0..8 {
        // mask is a bit mask of three bits, one for each of red, green, and blue.
        let mask = b8(i as u8);
        target |= b8((masked_sum(mask, a) > masked_sum(mask, b)) as u8) << i;
    }
    target
}

// mask is a set of 3 bits indicating what subset of the color channels to use.
// sum the colors iff the corresponding bit is set.
fn masked_sum(mask: b8, values: [u16; 3]) -> u16 {
    let mut sum = 0u16;
    for i in 0..3 {
        sum += values[i] * (mask.bit(i) as u16);
    }
    sum
}
