use crate::bits::{BitArray, BFMA};
use crate::shape::{Element, Shape};

pub trait FMA {
    fn fma(&self, rhs: &Self) -> f32;
}

impl FMA for f32 {
    fn fma(&self, rhs: &Self) -> f32 {
        self * rhs
    }
}

impl<T: FMA, const L: usize> FMA for [T; L] {
    fn fma(&self, rhs: &[T; L]) -> f32 {
        let mut sum = 0f32;
        for i in 0..L {
            sum += self[i].fma(&rhs[i]);
        }
        sum
    }
}

// bit input, float weights and float output.
pub trait FloatMul<S: Shape>
where
    Self: BitArray,
    f32: Element<Self::BitShape> + Element<S>,
    <f32 as Element<Self::BitShape>>::Array: Element<S>,
{
    fn float_mul(
        &self,
        weights: &<<f32 as Element<Self::BitShape>>::Array as Element<S>>::Array,
    ) -> <f32 as Element<S>>::Array;
}

impl<T: BitArray + BFMA> FloatMul<()> for T
where
    f32: Element<T::BitShape>,
{
    fn float_mul(&self, weights: &<f32 as Element<T::BitShape>>::Array) -> f32 {
        self.bfma(weights)
    }
}

impl<S: Shape, T: BitArray + FloatMul<S>, const L: usize> FloatMul<[S; L]> for T
where
    f32: Element<T::BitShape> + Element<S>,
    <f32 as Element<T::BitShape>>::Array: Element<S>,
    [<f32 as Element<S>>::Array; L]: Default,
{
    fn float_mul(
        &self,
        weights: &[<<f32 as Element<Self::BitShape>>::Array as Element<S>>::Array; L],
    ) -> [<f32 as Element<S>>::Array; L] {
        let mut target = <[<f32 as Element<S>>::Array; L]>::default();
        for i in 0..L {
            target[i] = self.float_mul(&weights[i]);
        }
        target
    }
}
