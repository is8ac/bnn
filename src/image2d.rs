use crate::shape::{Shape, ZipMap};
use rand::Rng;
use std::fmt;
use std::hash::{Hash, Hasher};

pub trait ImageShape {}

impl<const X: usize, const Y: usize> ImageShape for [[(); Y]; X] {}

pub trait PixelPack<P> {
    type I;
}

pub struct DynamicImageShape {
    x: usize,
    y: usize,
}

impl<P: Sized, const X: usize, const Y: usize> PixelPack<P> for [[(); Y]; X] {
    type I = [[P; Y]; X];
}

pub trait PixelMap<I, O>
where
    Self: PixelPack<I> + PixelPack<O>,
{
    fn map<F: Fn(I) -> O>(
        input: &<Self as PixelPack<I>>::I,
        map_fn: F,
    ) -> <Self as PixelPack<O>>::I;
}

impl<I: Copy, O, const X: usize, const Y: usize> PixelMap<I, O> for [[(); Y]; X]
where
    [[(); Y]; X]: ImageShape,
    [[O; Y]; X]: Default,
{
    fn map<F: Fn(I) -> O>(input: &[[I; Y]; X], map_fn: F) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..X {
            for y in 0..Y {
                target[x][y] = map_fn(input[x][y]);
            }
        }
        target
    }
}

pub trait PixelFold<B, P, const PX: usize, const PY: usize>
where
    Self: ImageShape + PixelPack<P>,
{
    fn pixel_fold<F: Fn(B, &P) -> B>(input: &<Self as PixelPack<P>>::I, acc: B, fold_fn: F) -> B;
}

impl<P: Copy, B, const X: usize, const Y: usize, const PX: usize, const PY: usize>
    PixelFold<B, P, PX, PY> for [[(); Y]; X]
{
    fn pixel_fold<F: Fn(B, &P) -> B>(input: &[[P; Y]; X], mut acc: B, fold_fn: F) -> B {
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(X - (PY / 2) * 2) {
                acc = fold_fn(acc, &input[x + PX / 2][y + PY / 2]);
            }
        }
        acc
    }
}

pub trait Conv<I, O, const PX: usize, const PY: usize>
where
    Self: ImageShape + PixelPack<I> + PixelPack<O>,
{
    fn conv<F: Fn([[I; PY]; PX]) -> O>(
        input: &<Self as PixelPack<I>>::I,
        map_fn: F,
    ) -> <Self as PixelPack<O>>::I;
}

impl<I: Copy, O, const PY: usize, const PX: usize, const X: usize, const Y: usize>
    Conv<I, O, PX, PY> for [[(); Y]; X]
where
    [[(); Y]; X]: ImageShape,
    [[O; Y]; X]: Default,
    [[I; PY]; PX]: Default,
{
    fn conv<F: Fn([[I; PY]; PX]) -> O>(input: &[[I; Y]; X], map_fn: F) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(Y - (PY / 2) * 2) {
                let mut patch = <[[I; PY]; PX]>::default();
                for px in 0..PX {
                    for py in 0..PY {
                        patch[px][py] = input[x + px][y + py];
                    }
                }
                target[x + PX / 2][y + PY / 2] = map_fn(patch);
            }
        }
        target
    }
}

/*
pub trait PatchFold<B, PatchShape: Shape>
where
    Self: Image2D,
    Self::PixelType: Element<PatchShape>,
{
    fn patch_fold<F: Fn(B, &<Self::PixelType as Element<PatchShape>>::Array) -> B>(&self, acc: B, fold_fn: F) -> B;
}

impl<P: Copy, B, const X: usize, const Y: usize, const PX: usize, const PY: usize> PatchFold<B, [[(); PY]; PX]> for [[P; Y]; X]
where
    [[P; PY]; PX]: Default,
{
    fn patch_fold<F: Fn(B, &<P as Element<[[(); PY]; PX]>>::Array) -> B>(&self, mut acc: B, fold_fn: F) -> B {
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(X - (PY / 2) * 2) {
                let mut patch = <[[P; PY]; PX]>::default();
                for px in 0..PX {
                    for py in 0..PY {
                        patch[px][py] = self[x + px][y + py];
                    }
                }
                acc = fold_fn(acc, &patch);
            }
        }
        acc
    }
}

pub trait PixelMap<O: Element<Self::ImageShape>>
where
    Self: Image2D,
{
    fn pixel_map<F: Fn(&Self::PixelType) -> O>(&self, map_fn: F) -> <O as Element<Self::ImageShape>>::Array;
}

//impl<I, O: Element<StaticImage<(), X, Y>, Array = StaticImage<O, X, Y>>, const X: usize, const Y: usize> PixelMap<O> for StaticImage<I, X, Y>
//where
//    StaticImage<O, X, Y>: Default,
//{
//    fn pixel_map<F: Fn(&I) -> O>(&self, map_fn: F) -> StaticImage<O, X, Y> {
//        let mut target = StaticImage::default();
//        for x in 0..X {
//            for y in 0..Y {
//                target.image[x][y] = map_fn(&self.image[x][y]);
//            }
//        }
//        target
//    }
//}

pub trait GlobalPool<O> {
    fn global_pool(&self) -> O;
}

pub trait PixelFold<B, PatchShape>
where
    Self: Image2D,
{
    fn pixel_fold<F: Fn(B, &Self::PixelType) -> B>(&self, acc: B, fold_fn: F) -> B;
}

//impl<P: Copy, B, const X: usize, const Y: usize, const PX: usize, const PY: usize> PixelFold<B, [[(); PY]; PX]> for StaticImage<P, X, Y> {
//    fn pixel_fold<F: Fn(B, &P) -> B>(&self, mut acc: B, fold_fn: F) -> B {
//        for x in 0..(X - (PX / 2) * 2) {
//            for y in 0..(X - (PY / 2) * 2) {
//                acc = fold_fn(acc, &self.image[x + PX / 2][y + PY / 2]);
//            }
//        }
//        acc
//    }
//}

pub trait RandomPatch<PatchShape: Shape>
where
    Self: Image2D,
    Self::PixelType: Element<PatchShape>,
{
    fn random_patch<RNG: Rng>(&self, rng: &mut RNG) -> <Self::PixelType as Element<PatchShape>>::Array;
}

//impl<P: Copy, const X: usize, const Y: usize, const PX: usize, const PY: usize> RandomPatch<[[(); PY]; PX]> for StaticImage<P, X, Y>
//where
//    Self: Image2D<PixelType = P>,
//    [[P; PY]; PX]: Default,
//{
//    fn random_patch<RNG: Rng>(&self, rng: &mut RNG) -> [[P; PY]; PX] {
//        let x = rng.gen_range(0, X - (PX / 2) * 2);
//        let y = rng.gen_range(0, Y - (PY / 2) * 2);
//        let mut patch = <[[P; PY]; PX]>::default();
//        for px in 0..PX {
//            for py in 0..PY {
//                patch[px][py] = self.image[x + px][y + py];
//            }
//        }
//        patch
//    }
//}

pub trait Poolable2D<const X: usize, const Y: usize> {
    type Pooled;
}

macro_rules! impl_poolable {
    ($x:expr, $y:expr) => {
        impl Poolable2D<$x, $y> for [[(); $y]; $x] {
            type Pooled = [[(); $y / 2]; $x / 2];
        }
    };
}

impl_poolable!(32, 32);
impl_poolable!(16, 16);
impl_poolable!(8, 8);
impl_poolable!(4, 4);
*/
