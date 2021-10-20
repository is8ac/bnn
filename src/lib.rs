#![feature(avx512_target_feature)]
#![feature(int_log)]
#![feature(stdsimd)]
#![forbid(unreachable_pub)]
#![feature(test)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

#[macro_use]
extern crate lazy_static;

pub mod bench;
pub mod bits;
pub mod bitslice;
pub mod count;
pub mod count_bits;
pub mod ecc;
pub mod layer;
pub mod search;
pub mod shape;
