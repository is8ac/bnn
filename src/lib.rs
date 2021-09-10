#![feature(stdsimd)]
#![forbid(unreachable_pub)]
#![feature(test)]
#![feature(adt_const_params)]
#![feature(generic_const_exprs)]

#[macro_use]
extern crate lazy_static;

pub mod bits;
pub mod count;
pub mod count_bits;
pub mod datasets;
pub mod ecc;
pub mod matrix;
pub mod search;
pub mod shape;
pub mod unary;
