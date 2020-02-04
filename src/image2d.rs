use crate::bits::{AndOr, BitArray, BitWord, Classify, IncrementFracCounters};
use crate::count::IncrementCounters;
use crate::float::FFFVMM;
use crate::layer::Apply;
use crate::shape::{Element, Map, Merge, Shape, ZipMap};
use std::fmt;
use std::hash::{Hash, Hasher};

pub struct StaticImage<Image> {
    pub image: Image,
}

impl<Image: Default> Default for StaticImage<Image> {
    fn default() -> Self {
        StaticImage {
            image: Image::default(),
        }
    }
}

impl<P: fmt::Debug, const X: usize, const Y: usize> fmt::Debug for StaticImage<[[P; Y]; X]> {
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

impl<P: BitWord, const X: usize, const Y: usize> fmt::Display for StaticImage<[[P; Y]; X]> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in 0..P::BIT_LEN {
            for y in 0..Y {
                for x in 0..X {
                    write!(f, "{}", if self.image[x][y].bit(b) { 1 } else { 0 })?
                }
                writeln!(f)?
            }
            writeln!(f, "{}", b)?
        }
        Ok(())
    }
}

impl<const X: usize, const Y: usize> fmt::Display for StaticImage<[[[u8; 3]; Y]; X]> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in 0..3 {
            for y in 0..Y {
                for x in 0..X {
                    write!(f, "{}", if self.image[x][y][b] > 128 { 1 } else { 0 })?
                }
                writeln!(f)?
            }
            writeln!(f, "{}", b)?
        }
        Ok(())
    }
}

impl<
        Preprocessor,
        T: Apply<[[IP; PY]; PX], (), Preprocessor, OP>,
        IP: Default + Copy,
        OP,
        const X: usize,
        const Y: usize,
        const PX: usize,
        const PY: usize,
    > Apply<StaticImage<[[IP; Y]; X]>, [[(); PY]; PX], Preprocessor, StaticImage<[[OP; Y]; X]>>
    for T
where
    [[IP; PY]; PX]: Default,
    [[OP; Y]; X]: Default,
{
    fn apply(&self, image: &StaticImage<[[IP; Y]; X]>) -> StaticImage<[[OP; Y]; X]> {
        let mut target = StaticImage {
            image: <[[OP; Y]; X]>::default(),
        };
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(Y - (PY / 2) * 2) {
                let mut patch = <[[IP; PY]; PX]>::default();
                for px in 0..PX {
                    for py in 0..PY {
                        patch[px][py] = image.image[x + px][y + py]
                    }
                }
                target.image[x + (PX / 2)][y + (PY / 2)] = self.apply(&patch);
            }
        }
        target
    }
}

impl<P, const X: usize, const Y: usize> Hash for StaticImage<[[P; Y]; X]>
where
    [[P; Y]; X]: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.image.hash(state);
    }
}

impl<Pixel: Copy, P, Accumulator, const X: usize, const Y: usize>
    IncrementCounters<[[(); 3]; 3], P, Accumulator> for StaticImage<[[Pixel; Y]; X]>
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
    Concat<StaticImage<[[A; Y]; X]>, StaticImage<[[B; Y]; X]>> for StaticImage<[[O; Y]; X]>
where
    [[(); Y]; X]: ZipMap<A, B, O>,
{
    fn concat(a: &StaticImage<[[A; Y]; X]>, b: &StaticImage<[[B; Y]; X]>) -> Self {
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

impl<P, const X: usize, const Y: usize> Image2D for StaticImage<[[P; Y]; X]> {
    type PixelType = P;
    type ImageShape = StaticImage<[[(); Y]; X]>;
}

impl<const X: usize, const Y: usize> Shape for StaticImage<[[(); Y]; X]> {
    const N: usize = X * Y;
    type Index = [usize; 2];
}

impl<P, const X: usize, const Y: usize> Element<StaticImage<[[(); Y]; X]>> for P {
    type Array = StaticImage<[[P; X]; Y]>;
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

pub trait AvgPool {
    type Pooled;
    fn avg_pool(&self) -> Self::Pooled;
}

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

impl<P: BitArray + IncrementFracCounters, const X: usize, const Y: usize, const C: usize>
    Classify<StaticImage<[[P; Y]; X]>, [(); C]> for [<f32 as Element<P::BitShape>>::Array; C]
where
    Self: FFFVMM<[f32; C], InputType = <f32 as Element<P::BitShape>>::Array>,
    [f32; C]: Default,
    P::BitShape: Map<u32, f32>,
    f32: Element<P::BitShape>,
    u32: Element<P::BitShape>,
    <u32 as Element<P::BitShape>>::Array: Default,
{
    fn max_class(&self, input: &StaticImage<[[P; Y]; X]>) -> usize {
        let channel_counts = {
            let mut counts = <(usize, <u32 as Element<P::BitShape>>::Array)>::default();
            for x in 0..X {
                for y in 0..Y {
                    input.image[x][y].increment_frac_counters(&mut counts);
                }
            }
            counts
        };
        let n = channel_counts.0 as f32;
        let float_hidden =
            <<P as BitArray>::BitShape as Map<u32, f32>>::map(&channel_counts.1, |&count| {
                count as f32 / n
            });
        let activations = self.fffvmm(&float_hidden);
        let mut max_act = 0_f32;
        let mut max_class = 0_usize;

        for c in 0..C {
            if activations[c] >= max_act {
                max_act = activations[c];
                max_class = c;
            }
        }
        max_class
    }
}

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
    PatchFold<B, [[(); PY]; PX]> for StaticImage<[[P; X]; Y]>
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
        O: Element<StaticImage<[[(); Y]; X]>, Array = StaticImage<[[O; Y]; X]>>,
        const X: usize,
        const Y: usize,
    > PixelMap<O> for StaticImage<[[I; Y]; X]>
where
    StaticImage<[[O; Y]; X]>: Default,
{
    fn pixel_map<F: Fn(&I) -> O>(&self, map_fn: F) -> StaticImage<[[O; Y]; X]> {
        let mut target = StaticImage::default();
        for x in 0..X {
            for y in 0..Y {
                target.image[x][y] = map_fn(&self.image[x][y]);
            }
        }
        target
    }
}
