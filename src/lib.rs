#![feature(int_log)]
#![forbid(unreachable_pub)]
#![forbid(unsafe_code)]
#![feature(test)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

#[macro_use]
extern crate lazy_static;

pub mod bench;
pub mod bits;
pub mod bitslice;
pub mod count_bits;
pub mod ecc;
pub mod layer;
pub mod search;
pub mod shape;
