#![feature(stdsimd)]
#![forbid(unreachable_pub)]
#![feature(test)]
#![feature(const_generics)]
#![feature(const_evaluatable_checked)]

#[macro_use]
extern crate lazy_static;

pub mod bits;
pub mod count;
pub mod datasets;
//pub mod descend;
//pub mod image2d;
//pub mod layers;
pub mod count_bits;
pub mod ecc;
pub mod matrix;
pub mod random_tables;
pub mod shape;
pub mod unary;
