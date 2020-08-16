use crate::count::IncrementCounters;
use crate::shape::{Element, Merge, Shape, ZipMap};
use rand::Rng;
use std::fmt;
use std::hash::{Hash, Hasher};

pub struct StaticImage<P, const X: usize, const Y: usize> {
    pub image: [[P; Y]; X],
}

impl<P, const X: usize, const Y: usize> Default for StaticImage<P, X, Y>
where
    [[P; Y]; X]: Default,
{
    fn default() -> Self {
        StaticImage {
            image: <[[P; Y]; X]>::default(),
        }
    }
}

impl<P: fmt::Debug, const X: usize, const Y: usize> fmt::Debug for StaticImage<P, X, Y> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..Y {
            for x in 0..X {
                writeln!(f, "{:?}", self.image[x][y])?
            }
            writeln!(f, "{}", y)?
        }
        Ok(())
    }
}

//impl<P, const X: usize, const Y: usize> fmt::Display for StaticImage<P, X, Y> {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        for b in 0..P::BIT_LEN {
//            for y in 0..Y {
//                for x in 0..X {
//                    write!(f, "{}", if self.image[x][y].bit(b) { 1 } else { 0 })?
//                }
//                writeln!(f)?
//            }
//            writeln!(f, "{}", b)?
//        }
//        Ok(())
//    }
//}

pub trait Conv2D<PatchShape: Shape, O>
where
    Self: Image2D,
    Self::PixelType: Element<PatchShape>,
{
    type OutputType;
    fn conv2d<F: Fn(&<Self::PixelType as Element<PatchShape>>::Array) -> O>(
        &self,
        map_fn: F,
    ) -> Self::OutputType;
}

impl<I: Copy, O, const X: usize, const Y: usize, const PX: usize, const PY: usize>
    Conv2D<[[(); PY]; PX], O> for StaticImage<I, X, Y>
where
    [[I; PY]; PX]: Default,
    StaticImage<O, X, Y>: Default,
{
    type OutputType = StaticImage<O, X, Y>;
    fn conv2d<F: Fn(&[[I; PY]; PX]) -> O>(&self, map_fn: F) -> Self::OutputType {
        let mut target = StaticImage::<O, X, Y>::default();
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(Y - (PY / 2) * 2) {
                let mut patch = <[[I; PY]; PX]>::default();
                for px in 0..PX {
                    for py in 0..PY {
                        patch[px][py] = self.image[x + px][y + py];
                    }
                }
                target.image[x + (PX / 2)][y + (PY / 2)] = map_fn(&patch);
            }
        }
        target
    }
}

impl<P, const X: usize, const Y: usize> Hash for StaticImage<P, X, Y>
where
    [[P; Y]; X]: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.image.hash(state);
    }
}

impl<Pixel: Copy, P, Accumulator, const X: usize, const Y: usize>
    IncrementCounters<[[(); 3]; 3], P, Accumulator> for StaticImage<Pixel, X, Y>
where
    [[Pixel; 3]; 3]: Default + IncrementCounters<(), P, Accumulator>,
{
    fn increment_counters(&self, class: usize, counters: &mut Accumulator) {
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                {
                    let mut patch = <[[Pixel; 3]; 3]>::default();
                    for px in 0..3 {
                        for py in 0..3 {
                            patch[px][py] = self.image[x + px][y + py]
                        }
                    }
                    patch
                }
                .increment_counters(class, counters);
            }
        }
    }
}

pub trait Concat<A, B> {
    fn concat(a: &A, b: &B) -> Self;
}

impl<A, B, O: Merge<A, B>, const X: usize, const Y: usize>
    Concat<StaticImage<A, X, Y>, StaticImage<B, X, Y>> for StaticImage<O, X, Y>
where
    [[(); Y]; X]: ZipMap<A, B, O>,
{
    fn concat(a: &StaticImage<A, X, Y>, b: &StaticImage<B, X, Y>) -> Self {
        StaticImage {
            image: <[[(); Y]; X] as ZipMap<A, B, O>>::zip_map(&a.image, &b.image, |a, b| {
                O::merge(a, b)
            }),
        }
    }
}

pub trait Image2D
where
    Self::ImageShape: Shape,
{
    type PixelType;
    type ImageShape;
}

impl<P, const X: usize, const Y: usize> Image2D for StaticImage<P, X, Y> {
    type PixelType = P;
    type ImageShape = StaticImage<(), X, Y>;
}

pub struct ImageIndexIter {}

impl<const X: usize, const Y: usize> Shape for StaticImage<(), X, Y> {
    const N: usize = X * Y;
    type Index = [usize; 2];
    type IndexIter = ImageIndexIter;
    fn indices() -> ImageIndexIter {
        ImageIndexIter {}
    }
}

impl<P, const X: usize, const Y: usize> Element<StaticImage<(), X, Y>> for P {
    type Array = StaticImage<P, X, Y>;
}

impl<P, const X: usize, const Y: usize> Image2D for [[P; Y]; X] {
    type PixelType = P;
    type ImageShape = [[(); Y]; X];
}

pub trait ImagePixel<Shape> {
    type ImageType;
}

impl<T, const X: usize, const Y: usize> ImagePixel<[[(); Y]; X]> for T {
    type ImageType = [[T; Y]; X];
}

pub trait BitPool {
    type Input;
    fn andor_pool(input: &Self::Input) -> Self;
}

pub trait AvgPool {
    type Pooled;
    fn avg_pool(&self) -> Self::Pooled;
}

/*
macro_rules! andorpool {
    ($x_size:expr, $y_size:expr) => {
        impl<Pixel: Copy + AndOr + Default> BitPool
            for StaticImage<[[Pixel; $y_size / 2]; $x_size / 2]>
        {
            type Input = StaticImage<[[Pixel::Val; $y_size]; $x_size]>;
            fn andor_pool(StaticImage { image }: &Self::Input) -> Self {
                let mut target = StaticImage {
                    image: [[Pixel::default(); $y_size / 2]; $x_size / 2],
                };
                for x in 0..$x_size / 2 {
                    let x_index = x * 2;
                    for y in 0..$y_size / 2 {
                        let y_index = y * 2;
                        target.image[x][y] = Pixel::IDENTITY
                            .andor(&image[x_index + 0][y_index + 0])
                            .andor(&image[x_index + 0][y_index + 1])
                            .andor(&image[x_index + 1][y_index + 0])
                            .andor(&image[x_index + 1][y_index + 1]);
                    }
                }
                target
            }
        }
    };
}

andorpool!(32, 32);
andorpool!(16, 16);
andorpool!(8, 8);
andorpool!(4, 4);


macro_rules! impl_avgpool {
    ($x_size:expr, $y_size:expr) => {
        impl AvgPool for StaticImage<[[[u8; 3]; $y_size]; $x_size]> {
            type Pooled = StaticImage<[[[u8; 3]; $y_size / 2]; $x_size / 2]>;
            fn avg_pool(&self) -> Self::Pooled {
                let mut target = StaticImage {
                    image: [[[0_u8; 3]; $y_size / 2]; $x_size / 2],
                };
                for x in 0..$x_size / 2 {
                    let x_index = x * 2;
                    for y in 0..$y_size / 2 {
                        let y_index = y * 2;
                        for c in 0..3 {
                            let sum = self.image[x_index + 0][y_index + 0][c] as u16
                                + self.image[x_index + 0][y_index + 1][c] as u16
                                + self.image[x_index + 1][y_index + 0][c] as u16
                                + self.image[x_index + 1][y_index + 1][c] as u16;
                            target.image[x][y][c] = (sum / 4) as u8;
                        }
                    }
                }
                target
            }
        }
    };
}

impl_avgpool!(32, 32);
impl_avgpool!(16, 16);
impl_avgpool!(8, 8);
impl_avgpool!(4, 4);
*/

pub trait PatchFold<B, PatchShape: Shape>
where
    Self: Image2D,
    Self::PixelType: Element<PatchShape>,
{
    fn patch_fold<F: Fn(B, &<Self::PixelType as Element<PatchShape>>::Array) -> B>(
        &self,
        acc: B,
        fold_fn: F,
    ) -> B;
}

impl<P: Copy, B, const X: usize, const Y: usize, const PX: usize, const PY: usize>
    PatchFold<B, [[(); PY]; PX]> for StaticImage<P, X, Y>
where
    [[P; PY]; PX]: Default,
{
    fn patch_fold<F: Fn(B, &<P as Element<[[(); PY]; PX]>>::Array) -> B>(
        &self,
        mut acc: B,
        fold_fn: F,
    ) -> B {
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(X - (PY / 2) * 2) {
                let mut patch = <[[P; PY]; PX]>::default();
                for px in 0..PX {
                    for py in 0..PY {
                        patch[px][py] = self.image[x + px][y + py];
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
    fn pixel_map<F: Fn(&Self::PixelType) -> O>(
        &self,
        map_fn: F,
    ) -> <O as Element<Self::ImageShape>>::Array;
}

impl<
        I,
        O: Element<StaticImage<(), X, Y>, Array = StaticImage<O, X, Y>>,
        const X: usize,
        const Y: usize,
    > PixelMap<O> for StaticImage<I, X, Y>
where
    StaticImage<O, X, Y>: Default,
{
    fn pixel_map<F: Fn(&I) -> O>(&self, map_fn: F) -> StaticImage<O, X, Y> {
        let mut target = StaticImage::default();
        for x in 0..X {
            for y in 0..Y {
                target.image[x][y] = map_fn(&self.image[x][y]);
            }
        }
        target
    }
}

pub trait GlobalPool<O> {
    fn global_pool(&self) -> O;
}

pub trait PixelFold<B, PatchShape>
where
    Self: Image2D,
{
    fn pixel_fold<F: Fn(B, &Self::PixelType) -> B>(&self, acc: B, fold_fn: F) -> B;
}

impl<P: Copy, B, const X: usize, const Y: usize, const PX: usize, const PY: usize>
    PixelFold<B, [[(); PY]; PX]> for StaticImage<P, X, Y>
{
    fn pixel_fold<F: Fn(B, &P) -> B>(&self, mut acc: B, fold_fn: F) -> B {
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(X - (PY / 2) * 2) {
                acc = fold_fn(acc, &self.image[x + PX / 2][y + PY / 2]);
            }
        }
        acc
    }
}

pub trait RandomPatch<PatchShape: Shape>
where
    Self: Image2D,
    Self::PixelType: Element<PatchShape>,
{
    fn random_patch<RNG: Rng>(
        &self,
        rng: &mut RNG,
    ) -> <Self::PixelType as Element<PatchShape>>::Array;
}

impl<P: Copy, const X: usize, const Y: usize, const PX: usize, const PY: usize>
    RandomPatch<[[(); PY]; PX]> for StaticImage<P, X, Y>
where
    Self: Image2D<PixelType = P>,
    [[P; PY]; PX]: Default,
{
    fn random_patch<RNG: Rng>(&self, rng: &mut RNG) -> [[P; PY]; PX] {
        let x = rng.gen_range(0, X - (PX / 2) * 2);
        let y = rng.gen_range(0, Y - (PY / 2) * 2);
        let mut patch = <[[P; PY]; PX]>::default();
        for px in 0..PX {
            for py in 0..PY {
                patch[px][py] = self.image[x + px][y + py];
            }
        }
        patch
    }
}
