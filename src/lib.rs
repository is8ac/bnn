#![forbid(unsafe_code)]
#![feature(const_generics)]
#![deny(unreachable_pub)]

extern crate bincode;
extern crate rayon;
extern crate serde;
extern crate time;

pub mod bits;
pub mod count;
pub mod datasets;
pub mod image2d;
pub mod layer;
pub mod shape;
pub mod unary;
pub mod weight;
