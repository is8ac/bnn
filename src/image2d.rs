use crate::bits::{
    b32, b8, AndOr, BitArray, BitMul, BitWord, Classify, Distance, IncrementFracCounters,
    IncrementHammingDistanceMatrix,
};
use crate::count::IncrementCounters;
use crate::layer::Apply;
use crate::shape::{Element, Merge, Shape, ZipMap};
use crate::unary::NormalizeAndBitpack;
use std::fmt;
use std::hash::{Hash, Hasher};

pub trait ExtractEdges {
    type OutputImage;
    fn extract_edges(&self) -> Self::OutputImage;
}

impl<const X: usize, const Y: usize> ExtractEdges for StaticImage<[[[u8; 3]; Y]; X]>
where
    [[b32; Y]; X]: Default,
{
    type OutputImage = StaticImage<[[b32; Y]; X]>;
    fn extract_edges(&self) -> Self::OutputImage {
        let mut target = StaticImage {
            image: <[[b32; Y]; X]>::default(),
        };
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                let mut patch = [[[0_u8; 3]; 3]; 3];
                for px in 0..3 {
                    for py in 0..3 {
                        patch[px][py] = self.image[x + px][y + py]
                    }
                }
                target.image[x + 1][y + 1] = edges_from_patch(&patch);
            }
        }
        target
    }
}

// Given a 3x3 patch of 3 color channels, extract the 32 edges.
// Each edge compares two sets of three pixels.
// For example horizontal is:
// |a|a|a|
// | | | |
// |b|b|b|
// while one diagonal is:
// |a|a| |
// |a| |b|
// | |b|b|
// center is ignores by all edges.
fn edges_from_patch(patch: &[[[u8; 3]; 3]; 3]) -> b32 {
    let mut target = b32::default();
    // horizontal
    target |= extract_partition(patch, [(0, 0), (0, 1), (0, 2)], [(2, 0), (2, 1), (2, 2)]) << 24;
    // vertical
    target |= extract_partition(patch, [(0, 0), (1, 0), (2, 0)], [(0, 2), (1, 2), (2, 2)]) << 16;
    // diagonal
    target |= extract_partition(patch, [(0, 2), (0, 1), (1, 2)], [(2, 0), (2, 1), (1, 2)]) << 8;
    // other diagonal
    target |= extract_partition(patch, [(0, 0), (1, 0), (0, 1)], [(2, 2), (1, 2), (2, 1)]) << 0;
    target
}

// Given the patch and the indices of the two sides of an edge, extract the 8 colors.
// It returns a b32, but only the first 8 bits are used.
fn extract_partition(
    patch: &[[[u8; 3]; 3]; 3],
    a: [(usize, usize); 3],
    b: [(usize, usize); 3],
) -> b32 {
    b32(color_features_from_partition(
        elementwise_sum_3(
            patch[a[0].0][a[0].1],
            patch[a[1].0][a[1].1],
            patch[a[2].0][a[2].1],
        ),
        elementwise_sum_3(
            patch[b[0].0][b[0].1],
            patch[b[1].0][b[1].1],
            patch[b[2].0][b[2].1],
        ),
    )
    .0 as u32)
}

// Sum together each of the three color channels.
// The sum is u16s to avoid overflows.
fn elementwise_sum_3(a: [u8; 3], b: [u8; 3], c: [u8; 3]) -> [u16; 3] {
    let mut sums = [0u16; 3];
    for i in 0..3 {
        sums[i] += a[i] as u16 + b[i] as u16 + c[i] as u16;
    }
    sums
}

// for each of the 8 combinations of the three color channels, compare the two sides of the edge.
fn color_features_from_partition(a: [u16; 3], b: [u16; 3]) -> b8 {
    let mut target = b8::default();
    for i in 0..8 {
        // mask is a bit mask of three bits, one for each of red, green, and blue.
        let mask = b8(i as u8);
        target |= b8((masked_sum(mask, a) > masked_sum(mask, b)) as u8) << i;
    }
    target
}

// mask is a set of 3 bits indicating what subset of the color channels to use.
// sum the colors iff the corresponding bit is set.
fn masked_sum(mask: b8, values: [u16; 3]) -> u16 {
    let mut sum = 0u16;
    for i in 0..3 {
        sum += values[i] * (mask.bit(i) as u16);
    }
    sum
}

pub struct StaticImage<Image> {
    pub image: Image,
}

impl<P: fmt::Debug, const X: usize, const Y: usize> fmt::Debug for StaticImage<[[P; Y]; X]> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for x in 0..X {
            for y in 0..Y {
                writeln!(f, "{:?}", self.image[x][y])?
            }
            writeln!(f, "{}", x)?
        }
        Ok(())
    }
}

impl<P: BitWord, const X: usize, const Y: usize> fmt::Display for StaticImage<[[P; Y]; X]> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in 0..P::BIT_LEN {
            for x in 0..X {
                for y in 0..Y {
                    write!(f, "{}", if self.image[x][y].bit(b) { 1 } else { 0 })?
                }
                writeln!(f)?
            }
            writeln!(f, "{}", b)?
        }
        Ok(())
    }
}

impl<T: BitMul, IP: Default + Copy, const X: usize, const Y: usize>
    Apply<StaticImage<[[IP; Y]; X]>, [[IP; 3]; 3], T::Input> for T
where
    [[IP; 3]; 3]: NormalizeAndBitpack<T::Input>,
    [[T::Target; Y]; X]: Default,
{
    type Output = StaticImage<[[T::Target; Y]; X]>;
    fn apply(&self, image: &StaticImage<[[IP; Y]; X]>) -> StaticImage<[[T::Target; Y]; X]> {
        let mut target = StaticImage {
            image: <[[T::Target; Y]; X]>::default(),
        };
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

impl<P, const X: usize, const Y: usize> Hash for StaticImage<[[P; Y]; X]>
where
    [[P; Y]; X]: Hash,
{
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
    > IncrementCounters<[[Pixel; 3]; 3], Patch, { C }> for StaticImage<[[Pixel; Y]; X]>
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

impl<IP: Distance, const X: usize, const Y: usize, const C: usize>
    Classify<StaticImage<[[IP::Rhs; Y]; X]>> for [([[IP; 3]; 3], u32); C]
where
    IP::Rhs: Default + Copy,
    [u32; C]: Default,
{
    const N_CLASSES: usize = C;
    type ClassesShape = [(); C];
    fn activations(&self, StaticImage { image }: &StaticImage<[[IP::Rhs; Y]; X]>) -> [u32; C] {
        let mut sums = <[u32; C]>::default();
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                let mut patch = [[IP::Rhs::default(); 3]; 3];
                for px in 0..3 {
                    for py in 0..3 {
                        patch[px][py] = image[x + px][y + py]
                    }
                }
                for c in 0..C {
                    sums[c] += self[c].0.distance(&patch) + self[c].1;
                }
            }
        }
        sums
    }
    fn max_class(&self, input: &StaticImage<[[IP::Rhs; Y]; X]>) -> usize {
        let activations = self.activations(input);
        let mut max_act = 0_u32;
        let mut max_class = 0_usize;
        for c in 0..C {
            let act = activations[c];
            if act >= max_act {
                max_act = act;
                max_class = c;
            }
        }
        max_class
    }
}
