use crate::bits::{
    ArrayBitAnd, ArrayBitOr, BitArray, BitMul, BitWord, Classify, Distance, IncrementFracCounters,
    IncrementHammingDistanceMatrix,
};
use crate::count::IncrementCounters;
use crate::shape::{Element, Shape};
use crate::unary::NormalizeAndBitpack;
use crate::layer::Apply;
use std::hash::{Hash, Hasher};

pub struct StaticImage<Pixel, const X: usize, const Y: usize> {
    pub image: [[Pixel; Y]; X],
}

impl<T: BitMul, IP: Default + Copy, const X: usize, const Y: usize> Apply<StaticImage<IP, {X}, {Y}>, [[IP; 3]; 3], T::Input> for T
where
    [[IP; 3]; 3]: NormalizeAndBitpack<T::Input>,
    [[T::Target; Y]; X]: Default,
{
    type Output = StaticImage<T::Target, {X}, {Y}>;
    fn apply(&self, image: &StaticImage<IP, {X}, {Y}>) -> StaticImage<T::Target, {X}, {Y}> {
        let mut target = StaticImage{image: <[[T::Target; Y]; X]>::default()};
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                let mut patch = [[IP::default(); 3]; 3];
                for px in 0..3 {
                    for py in 0..3 {
                        patch[px][py] = image.image[x + px][y + py]
                    }
                }
                target.image[x + 1][y + 1] = self.bit_mul(&patch.normalize_and_bitpack());
            }
        }
        target
    }
}


impl<P, const X: usize, const Y: usize> Hash for StaticImage<P, {X}, {Y}> where [[P; Y]; X]: Hash {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.image.hash(state);
    }
}


impl<
        Pixel: Copy,
        Patch: BitArray + IncrementFracCounters + IncrementHammingDistanceMatrix<Patch>,
        const X: usize,
        const Y: usize,
        const C: usize,
    > IncrementCounters<[[Pixel; 3]; 3], Patch, { C }> for StaticImage<Pixel, { X }, { Y }>
where
    [[Pixel; 3]; 3]: NormalizeAndBitpack<Patch> + Default,
    u32: Element<Patch::BitShape>,
    u32: Element<<Patch as BitArray>::BitShape>,
    <u32 as Element<Patch::BitShape>>::Array: Element<Patch::BitShape>,
{
    fn increment_counters(
        &self,
        class: usize,
        value_counters: &mut [(usize, <u32 as Element<Patch::BitShape>>::Array); C],
        counters_matrix: &mut <<u32 as Element<Patch::BitShape>>::Array as Element<
            Patch::BitShape,
        >>::Array,
        n_examples: &mut usize,
    ) {
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                *n_examples += 1;
                let mut patch = <[[Pixel; 3]; 3]>::default();
                for px in 0..3 {
                    for py in 0..3 {
                        patch[px][py] = self.image[x + px][y + py]
                    }
                }
                let normalized = patch.normalize_and_bitpack();
                normalized.increment_frac_counters(&mut value_counters[class]);
                normalized.increment_hamming_distance_matrix(counters_matrix, &normalized);
            }
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

impl<P, const X: usize, const Y: usize> Image2D for StaticImage<P, { X }, { Y }> {
    type PixelType = P;
    type ImageShape = [[(); Y]; X];
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

// An image shape that can be 2x2 pooled
pub trait Poolable {
    type Pooled;
}

macro_rules! impl_poolable {
    ($x:expr, $y:expr) => {
        impl Poolable for [[(); $y]; $x] {
            type Pooled = [[(); $y / 2]; $x / 2];
        }
    };
}

impl_poolable!(32, 32);
impl_poolable!(16, 16);
impl_poolable!(8, 8);
impl_poolable!(4, 4);

//pub trait ConvIncrementCounters<P: BitArray, const C: usize>
//where
//    Self: Image2D,
//    [[Self::PixelType; 3]; 3]: Normalize2D<P>,
//    u32: Element<P::BitShape>,
//    u32: Element<<P as BitArray>::BitShape>,
//    <u32 as Element<P::BitShape>>::Array: Element<P::BitShape>,
//{
//    fn conv_increment_counters(
//        &self,
//        class: usize,
//        value_counters: &mut [(usize, <u32 as Element<P::BitShape>>::Array); C],
//        counters_matrix: &mut <<u32 as Element<P::BitShape>>::Array as Element<P::BitShape>>::Array,
//        n_examples: &mut usize,
//    );
//}
//
//impl<
//        Patch: BitArray + IncrementFracCounters + IncrementHammingDistanceMatrix<Patch>,
//        IP: Copy + Default,
//        const X: usize,
//        const Y: usize,
//        const C: usize,
//    > ConvIncrementCounters<Patch, { C }> for [[IP; Y]; X]
//where
//    Self: Image2D<PixelType = IP>,
//    [[IP; 3]; 3]: Normalize2D<Patch>,
//    u32: Element<Patch::BitShape>,
//    u32: Element<<Patch as BitArray>::BitShape>,
//    <u32 as Element<Patch::BitShape>>::Array: Element<Patch::BitShape>,
//{
//    fn conv_increment_counters(
//        &self,
//        class: usize,
//        value_counters: &mut [(usize, <u32 as Element<Patch::BitShape>>::Array); C],
//        counters_matrix: &mut <<u32 as Element<Patch::BitShape>>::Array as Element<
//            Patch::BitShape,
//        >>::Array,
//        n_examples: &mut usize,
//    ) {
//        for x in 0..X - 2 {
//            for y in 0..Y - 2 {
//                *n_examples += 1;
//                let mut patch = [[IP::default(); 3]; 3];
//                for px in 0..3 {
//                    for py in 0..3 {
//                        patch[px][py] = self[x + px][y + py]
//                    }
//                }
//                let normalized = patch.normalize_2d();
//                normalized.increment_frac_counters(&mut value_counters[class]);
//                normalized.increment_hamming_distance_matrix(counters_matrix, &normalized);
//            }
//        }
//    }
//}

pub trait BitPool
where
    Self: Image2D,
    Self::ImageShape: Poolable,
    <<Self as Image2D>::ImageShape as Poolable>::Pooled: Shape,
    <Self as Image2D>::PixelType: Element<<<Self as Image2D>::ImageShape as Poolable>::Pooled>,
{
    fn or_pool(
        &self,
    ) -> <<Self as Image2D>::PixelType as Element<
        <<Self as Image2D>::ImageShape as Poolable>::Pooled,
    >>::Array;
    fn and_pool(
        &self,
    ) -> <<Self as Image2D>::PixelType as Element<
        <<Self as Image2D>::ImageShape as Poolable>::Pooled,
    >>::Array;
}

macro_rules! impl_bitpool {
    ($x_size:expr, $y_size:expr) => {
        impl<Pixel: ArrayBitOr + ArrayBitAnd + Default + Copy> BitPool
            for [[Pixel; $y_size]; $x_size]
        {
            fn or_pool(&self) -> [[Pixel; $y_size / 2]; $x_size / 2] {
                let mut target = [[Pixel::default(); $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let x_index = x * 2;
                    for y in 0..$y_size / 2 {
                        let y_index = y * 2;
                        target[x][y] = self[x_index + 0][y_index + 0]
                            .bit_or(&self[x_index + 0][y_index + 1])
                            .bit_or(&self[x_index + 1][y_index + 0])
                            .bit_or(&self[x_index + 1][y_index + 1]);
                    }
                }
                target
            }
            fn and_pool(&self) -> [[Pixel; $y_size / 2]; $x_size / 2] {
                let mut target = [[Pixel::default(); $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let x_index = x * 2;
                    for y in 0..$y_size / 2 {
                        let y_index = y * 2;
                        target[x][y] = self[x_index + 0][y_index + 0]
                            .bit_and(&self[x_index + 0][y_index + 1])
                            .bit_and(&self[x_index + 1][y_index + 0])
                            .bit_and(&self[x_index + 1][y_index + 1]);
                    }
                }
                target
            }
        }
    };
}

impl_bitpool!(32, 32);
impl_bitpool!(16, 16);
impl_bitpool!(8, 8);
impl_bitpool!(4, 4);

pub trait AvgPool
where
    Self: Image2D,
    Self::ImageShape: Poolable,
    <<Self as Image2D>::ImageShape as Poolable>::Pooled: Shape,
    <Self as Image2D>::PixelType: Element<<<Self as Image2D>::ImageShape as Poolable>::Pooled>,
{
    fn avg_pool(
        &self,
    ) -> <<Self as Image2D>::PixelType as Element<
        <<Self as Image2D>::ImageShape as Poolable>::Pooled,
    >>::Array;
}

macro_rules! impl_avgpool {
    ($x_size:expr, $y_size:expr) => {
        impl AvgPool for [[[u8; 3]; $y_size]; $x_size]
        where
            [[(); $x_size]; $y_size]: Poolable<Pooled = [[(); $y_size / 2]; $y_size / 2]>,
        {
            fn avg_pool(&self) -> [[[u8; 3]; $y_size / 2]; $x_size / 2] {
                let mut target = [[[0u8; 3]; $y_size / 2]; $x_size / 2];
                for x in 0..$x_size / 2 {
                    let x_index = x * 2;
                    for y in 0..$y_size / 2 {
                        let y_index = y * 2;
                        for c in 0..3 {
                            let sum = self[x_index + 0][y_index + 0][c] as u16
                                + self[x_index + 0][y_index + 1][c] as u16
                                + self[x_index + 1][y_index + 0][c] as u16
                                + self[x_index + 1][y_index + 1][c] as u16;
                            target[x][y][c] = (sum / 4) as u8;
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

pub trait Conv2D<Image: Image2D>
where
    Image::ImageShape: Shape,
    Self: BitMul,
    Self::Target: Element<Image::ImageShape>,
{
    fn conv2d(&self, image: &Image) -> <Self::Target as Element<Image::ImageShape>>::Array;
}

impl<T: BitMul, IP: Default + Copy, const X: usize, const Y: usize> Conv2D<[[IP; Y]; X]> for T
where
    [[IP; 3]; 3]: NormalizeAndBitpack<T::Input>,
    [[Self::Target; Y]; X]: Default,
{
    fn conv2d(&self, image: &[[IP; Y]; X]) -> [[Self::Target; Y]; X] {
        let mut target = <[[Self::Target; Y]; X]>::default();
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                let mut patch = [[IP::default(); 3]; 3];
                for px in 0..3 {
                    for py in 0..3 {
                        patch[px][py] = image[x + px][y + py]
                    }
                }
                target[x + 1][y + 1] = self.bit_mul(&patch.normalize_and_bitpack());
            }
        }
        target
    }
}

pub trait PrintImage {
    fn print_channels(&self);
}

impl<B: BitWord, const W: usize, const X: usize, const Y: usize> PrintImage for [[[B; W]; Y]; X] {
    fn print_channels(&self) {
        for w in 0..W {
            for b in 0..B::BIT_LEN {
                for x in 0..X {
                    for y in 0..Y {
                        print!("{}", self[x][y][w].bit(b) as u8);
                    }
                    print!("\n");
                }
                println!("-------");
            }
        }
    }
}

impl<B: BitWord, const X: usize, const Y: usize> PrintImage for [[(B, B); Y]; X] {
    fn print_channels(&self) {
        for b in 0..B::BIT_LEN {
            for x in 0..X {
                for y in 0..Y {
                    print!("{}", self[x][y].0.bit(b) as u8);
                }
                print!(" ");
                for y in 0..Y {
                    print!("{}", self[x][y].1.bit(b) as u8);
                }
                print!("\n");
            }
            println!("---");
        }
    }
}

pub trait NCorrectConv<Image> {
    fn n_correct(&self, image: &Image, class: usize) -> (usize, u64);
}

impl<
        T: Classify<Input = Patch, ClassesShape = [(); C]>,
        Patch: Distance<Rhs = Patch>,
        IP: Default + Copy,
        const X: usize,
        const Y: usize,
        const C: usize,
    > NCorrectConv<[[IP; Y]; X]> for T
where
    [[IP; 3]; 3]: NormalizeAndBitpack<Patch>,
{
    fn n_correct(&self, image: &[[IP; Y]; X], class: usize) -> (usize, u64) {
        let mut n_correct = 0u64;
        let mut total = 0usize;
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                let mut patch = [[IP::default(); 3]; 3];
                for px in 0..3 {
                    for py in 0..3 {
                        patch[px][py] = image[x + px][y + py]
                    }
                }
                total += 1;
                n_correct += (self.max_class(&patch.normalize_and_bitpack()) == class) as u64;
            }
        }
        (total, n_correct)
    }
}

pub trait ClassifyAvgPooled<Image> {
    fn n_correct(&self, image: &Image, class: usize) -> (usize, u64);
}

impl<
        T: Classify<Input = Patch, ClassesShape = [(); C]>,
        Patch: Distance<Rhs = Patch>,
        IP: Default + Copy,
        const X: usize,
        const Y: usize,
        const C: usize,
    > ClassifyAvgPooled<[[IP; Y]; X]> for T
where
    [[IP; 3]; 3]: NormalizeAndBitpack<Patch>,
{
    fn n_correct(&self, image: &[[IP; Y]; X], class: usize) -> (usize, u64) {
        let mut n_correct = 0u64;
        let mut total = 0usize;
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                let mut patch = [[IP::default(); 3]; 3];
                for px in 0..3 {
                    for py in 0..3 {
                        patch[px][py] = image[x + px][y + py]
                    }
                }
                total += 1;
                n_correct += (self.max_class(&patch.normalize_and_bitpack()) == class) as u64;
            }
        }
        (total, n_correct)
    }
}
