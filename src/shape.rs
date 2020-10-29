use std::convert::TryInto;

/// A shape.
/// This trait has no concept of what it contains, just the shape.
/// It is implemented for arrays.
pub trait Shape
where
    Self::IndexIter: Iterator<Item = Self::Index>,
    Self::Index: Copy,
{
    /// The number of elements in the shape.
    const N: usize;
    /// The type used to index into the shape.
    type Index;
    type IndexIter;
    fn indices() -> Self::IndexIter;
}

#[derive(Debug)]
pub struct EmptyIndexIter {
    done: bool,
}

impl Iterator for EmptyIndexIter {
    type Item = ();
    fn next(&mut self) -> Option<()> {
        Some(()).filter(|_| {
            let done = self.done;
            self.done = true;
            !done
        })
    }
}

impl Shape for () {
    const N: usize = 1;
    type Index = ();
    type IndexIter = EmptyIndexIter;
    fn indices() -> EmptyIndexIter {
        EmptyIndexIter { done: false }
    }
}

#[derive(Debug)]
pub struct ShapeIndicesIter<T: Shape, const L: usize> {
    index: u8,
    inner: T::IndexIter,
}

impl<T: Shape, const L: usize> Iterator for ShapeIndicesIter<T, L>
where
    T::IndexIter: Iterator<Item = T::Index>,
{
    type Item = (u8, T::Index);
    fn next(&mut self) -> Option<(u8, T::Index)> {
        self.inner
            .next()
            .or_else(|| {
                self.index += 1;
                if self.index < L.try_into().unwrap() {
                    self.inner = T::indices();
                    self.inner.next()
                } else {
                    None
                }
            })
            .map(|x| (self.index, x))
    }
}

impl<T: Shape, const L: usize> Shape for [T; L] {
    const N: usize = T::N * L;
    type Index = (u8, T::Index);
    type IndexIter = ShapeIndicesIter<T, L>;
    fn indices() -> ShapeIndicesIter<T, L> {
        ShapeIndicesIter::<T, L> {
            index: 0,
            inner: T::indices(),
        }
    }
}

// inserts a self inside the W
pub trait Wrap<W> {
    type Wrapped;
    fn wrap(self, w: W) -> Self::Wrapped;
}

impl<T: Copy> Wrap<()> for T {
    type Wrapped = T;
    fn wrap(self, _: ()) -> T {
        self
    }
}

impl<T: Wrap<W>, W> Wrap<(u8, W)> for T {
    type Wrapped = (u8, <T as Wrap<W>>::Wrapped);
    fn wrap(self, (i, w): (u8, W)) -> (u8, <T as Wrap<W>>::Wrapped) {
        (i, <T as Wrap<W>>::wrap(self, w))
    }
}

/// Given an element and a shape, get the array.
pub trait Element<S: Shape> {
    /// The type of the Shape `S` when filled with `Element`s of `Self`.
    type Array;
}

impl<T: Sized> Element<()> for T {
    type Array = T;
}

impl<T: Element<S>, S: Shape, const L: usize> Element<[S; L]> for T {
    type Array = [T::Array; L];
}

pub trait Map<I: Element<Self>, O: Element<Self>>
where
    Self: Shape + Sized,
{
    fn map<F: Fn(&I) -> O>(
        input: &<I as Element<Self>>::Array,
        map_fn: F,
    ) -> <O as Element<Self>>::Array;
}

impl<I, O> Map<I, O> for () {
    fn map<F: Fn(&I) -> O>(input: &I, map_fn: F) -> O {
        map_fn(input)
    }
}

impl<S: Shape + Map<I, O>, I: Element<S>, O: Element<S>, const L: usize> Map<I, O> for [S; L]
where
    <O as Element<Self>>::Array: LongDefault,
{
    fn map<F: Fn(&I) -> O>(
        input: &[<I as Element<S>>::Array; L],
        map_fn: F,
    ) -> <O as Element<Self>>::Array {
        let mut target = <<O as Element<Self>>::Array>::long_default();
        for i in 0..L {
            target[i] = <S as Map<I, O>>::map(&input[i], &map_fn);
        }
        target
    }
}

pub trait IndexMap<O: Element<Self>, W: Shape>
where
    Self: Shape + Sized + Element<W>,
    <Self as Element<W>>::Array: Shape,
{
    fn index_map<F: Fn(<<Self as Element<W>>::Array as Shape>::Index) -> O>(
        outer_index: W::Index,
        map_fn: F,
    ) -> <O as Element<Self>>::Array;
}

impl<O, W: Shape> IndexMap<O, W> for ()
where
    (): Element<W, Array = W>,
{
    fn index_map<F: Fn(W::Index) -> O>(outer_index: W::Index, map_fn: F) -> O {
        map_fn(outer_index)
    }
}

impl<
        O: Element<S>,
        W: Shape,
        S: Shape + IndexMap<O, <[(); L] as Element<W>>::Array>,
        const L: usize,
    > IndexMap<O, W> for [S; L]
where
    [S; L]: Element<W, Array = <S as Element<<[(); L] as Element<W>>::Array>>::Array>,
    <[S; L] as Element<W>>::Array: Shape,
    [(); L]: Element<W>,
    <[(); L] as Element<W>>::Array: Shape,
    <S as Element<<[(); L] as Element<W>>::Array>>::Array: Shape,
    [<O as Element<S>>::Array; L]: LongDefault,
    (u8, ()): Wrap<W::Index, Wrapped = <<[(); L] as Element<W>>::Array as Shape>::Index>,
    W::Index: Copy,
{
    fn index_map<F: Fn(<<[S; L] as Element<W>>::Array as Shape>::Index) -> O>(
        outer_index: W::Index,
        map_fn: F,
    ) -> [<O as Element<S>>::Array; L] {
        let mut target = <[<O as Element<S>>::Array; L]>::long_default();
        for i in 0..L {
            target[i] = <S as IndexMap<O, <[(); L] as Element<W>>::Array>>::index_map(
                (i as u8, ()).wrap(outer_index),
                &map_fn,
            );
        }
        target
    }
}

pub trait IndexGet<I>
where
    Self: Sized,
{
    type Element;
    fn index_get(&self, i: I) -> &Self::Element;
    fn index_get_mut(&mut self, i: I) -> &mut Self::Element;
    fn index_set(mut self, i: I, val: Self::Element) -> Self {
        *self.index_get_mut(i) = val;
        self
    }
}

impl<T> IndexGet<()> for T {
    type Element = T;
    fn index_get(&self, _: ()) -> &T {
        self
    }
    fn index_get_mut(&mut self, _: ()) -> &mut T {
        self
    }
}

impl<I, T: IndexGet<I>, const L: usize> IndexGet<(u8, I)> for [T; L] {
    type Element = T::Element;
    fn index_get(&self, (i, ii): (u8, I)) -> &T::Element {
        self[i as usize].index_get(ii)
    }
    fn index_get_mut(&mut self, (i, ii): (u8, I)) -> &mut T::Element {
        self[i as usize].index_get_mut(ii)
    }
}

pub trait Fold<B, I: Element<Self>>
where
    Self: Shape + Sized,
{
    fn fold<F: Fn(B, &I) -> B>(input: &<I as Element<Self>>::Array, acc: B, fold_fn: F) -> B;
}

impl<B, I: Element<(), Array = I> + Sized> Fold<B, I> for () {
    fn fold<F: Fn(B, &I) -> B>(input: &I, acc: B, fold_fn: F) -> B {
        fold_fn(acc, input)
    }
}

impl<S: Shape + Fold<B, I>, B, I: Element<S> + Sized, const L: usize> Fold<B, I> for [S; L] {
    fn fold<F: Fn(B, &I) -> B>(input: &[<I as Element<S>>::Array; L], mut acc: B, fold_fn: F) -> B {
        for i in 0..L {
            acc = <S as Fold<B, I>>::fold(&input[i], acc, &fold_fn);
        }
        acc
    }
}

pub trait ZipMap<A: Element<Self>, B: Element<Self>, O: Element<Self>>
where
    Self: Shape + Sized,
{
    fn zip_map<F: Fn(&A, &B) -> O>(
        a: &<A as Element<Self>>::Array,
        b: &<B as Element<Self>>::Array,
        map_fn: F,
    ) -> <O as Element<Self>>::Array;
}

impl<A: Element<(), Array = A>, B: Element<(), Array = B>, O> ZipMap<A, B, O> for () {
    fn zip_map<F: Fn(&A, &B) -> O>(a: &A, b: &B, map_fn: F) -> O {
        map_fn(a, b)
    }
}

impl<S: Shape + ZipMap<A, B, O>, A: Element<S>, B: Element<S>, O: Element<S>, const L: usize>
    ZipMap<A, B, O> for [S; L]
where
    [<O as Element<S>>::Array; L]: LongDefault,
{
    fn zip_map<F: Fn(&A, &B) -> O>(
        a: &<A as Element<[S; L]>>::Array,
        b: &<B as Element<[S; L]>>::Array,
        map_fn: F,
    ) -> [<O as Element<S>>::Array; L] {
        let mut target = <[<O as Element<S>>::Array; L]>::long_default();
        for i in 0..L {
            target[i] = S::zip_map(&a[i], &b[i], &map_fn);
        }
        target
    }
}

pub trait LongDefault {
    fn long_default() -> Self;
}

macro_rules! impl_long_default_for_array {
    ($len:expr) => {
        impl<T: Copy + LongDefault> LongDefault for [T; $len] {
            fn long_default() -> [T; $len] {
                [T::long_default(); $len]
            }
        }
    };
}

impl_long_default_for_array!(1);
impl_long_default_for_array!(2);
impl_long_default_for_array!(3);
impl_long_default_for_array!(4);
impl_long_default_for_array!(5);
impl_long_default_for_array!(6);
impl_long_default_for_array!(7);
impl_long_default_for_array!(8);
impl_long_default_for_array!(9);
impl_long_default_for_array!(10);
impl_long_default_for_array!(11);
impl_long_default_for_array!(12);
impl_long_default_for_array!(13);
impl_long_default_for_array!(14);
impl_long_default_for_array!(15);
impl_long_default_for_array!(16);
impl_long_default_for_array!(32);
impl_long_default_for_array!(64);
impl_long_default_for_array!(128);
impl_long_default_for_array!(256);
impl_long_default_for_array!(512);
impl_long_default_for_array!(1024);

pub trait Flatten<T: Copy + Element<Self>>
where
    Self: Shape + Sized,
{
    fn from_vec(slice: &[T]) -> <T as Element<Self>>::Array;
    fn to_vec(array: &<T as Element<Self>>::Array, slice: &mut [T]);
}

impl<T: Copy + Element<(), Array = T>> Flatten<T> for () {
    fn from_vec(slice: &[T]) -> T {
        assert_eq!(slice.len(), 1);
        slice[0]
    }
    fn to_vec(&array: &T, slice: &mut [T]) {
        assert_eq!(slice.len(), 1);
        slice[0] = array;
    }
}

impl<S: Shape + Flatten<T>, T: Element<S> + Copy, const L: usize> Flatten<T> for [S; L]
where
    [<T as Element<S>>::Array; L]: LongDefault,
{
    fn from_vec(slice: &[T]) -> [<T as Element<S>>::Array; L] {
        assert_eq!(slice.len(), S::N * L);
        let mut target = <[<T as Element<S>>::Array; L]>::long_default();
        for i in 0..L {
            target[i] = S::from_vec(&slice[S::N * i..S::N * (i + 1)]);
        }
        target
    }
    fn to_vec(array: &<T as Element<Self>>::Array, slice: &mut [T]) {
        assert_eq!(slice.len(), S::N * L);
        for i in 0..L {
            S::to_vec(&array[i], &mut slice[S::N * i..S::N * (i + 1)]);
        }
    }
}

#[cfg(test)]
mod test {
    use super::{Flatten, Shape};
    type TestShape = [[(); 2]; 3];
    #[test]
    fn from_vec() {
        let array1 = [[0u8, 1], [2, 3], [4, 5]];
        let mut flat = vec![0u8; TestShape::N];
        TestShape::to_vec(&array1, &mut flat);
        let array2 = TestShape::from_vec(&flat);
        assert_eq!(array1, array2);
    }
}

pub trait Merge<A, B> {
    fn merge(a: &A, b: &B) -> Self;
}

impl<T: Copy> Merge<T, T> for [T; 2] {
    fn merge(&a: &T, &b: &T) -> Self {
        [a, b]
    }
}

impl<T: Copy> Merge<[[T; 2]; 2], [T; 2]> for [T; 6] {
    fn merge(&a: &[[T; 2]; 2], &b: &[T; 2]) -> Self {
        [a[0][0], a[0][1], a[1][0], a[1][1], b[0], b[1]]
    }
}

macro_rules! impl_array_array_merge {
    ($a:expr, $b:expr) => {
        impl<T: Copy + LongDefault> Merge<[T; $a], [T; $b]> for [T; $a + $b] {
            fn merge(&a: &[T; $a], &b: &[T; $b]) -> Self {
                let mut target = <[T; $a + $b]>::long_default();
                for i in 0..$a {
                    target[i] = a[i];
                }
                for i in 0..$b {
                    target[$a + i] = b[i];
                }
                target
            }
        }
    };
}

impl_array_array_merge!(1, 1);
impl_array_array_merge!(2, 1);

impl_array_array_merge!(1, 2);
impl_array_array_merge!(2, 2);

impl_array_array_merge!(1, 3);
impl_array_array_merge!(2, 3);
