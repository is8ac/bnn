#![feature(const_generics)]
use bitnn::bits::{b16, b32, b8, BitMul, BitWord, HammingDistance};
use bitnn::shape::{Array, Element, Map, MapMut, Shape, ZipFold};
use std::fmt;
use std::num::Wrapping;

type InputShape = [([[(); 3]; 4], [(); 4]); 5];
//type InputType = ([[u16; 3]; 4], [u16; 4]);
type InputType = <u8 as Element<InputShape>>::Array;
type InputElem = <InputType as Array<InputShape>>::Element;

fn main() {
    let foo = b8::splat(false);
    let bar = b8::splat(true);
    let baz = b32::splat(false);
    dbg!(bar.bit(5));
    println!("{}", foo);
    dbg!(foo ^ bar);
    dbg!(baz);
    dbg!(std::any::type_name::<InputShape>());
    dbg!(std::any::type_name::<<u8 as Element<InputShape>>::Array>());
    dbg!(std::any::type_name::<
        <InputType as Array<InputShape>>::Element,
    >());
    //let mut counters = <u32 as Element<InputShape>>::Array::default();
    //foo.increment_counters(&mut counters);
    //dbg!(counters);
    let weights = [(b16::splat(true), 5u32); 8];
    let output: b8 = weights.bit_mul(&b16::splat(false));
    dbg!(output);
}
