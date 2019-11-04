#![feature(const_generics)]
#![feature(test)]

extern crate bincode;
extern crate rand;
extern crate rayon;
extern crate serde;
extern crate serde_derive;
extern crate time;

pub mod bits;
pub mod count;
pub mod datasets;
pub mod image2d;
pub mod layer;
pub mod shape;
pub mod unary;
pub mod weight;
