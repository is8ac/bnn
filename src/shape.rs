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
pub trait Pack<E>
where
    Self: Shape,
{
    /// The type of the Shape `Self` when filled with `Element`s of `E`.
    type T;
}

impl<E: Sized> Pack<E> for () {
    type T = E;
}

impl<E, S: Shape + Pack<E>, const L: usize> Pack<E> for [S; L] {
    type T = [<S as Pack<E>>::T; L];
}

pub trait Map<I, O>
where
    Self: Shape + Sized + Pack<I> + Pack<O>,
{
    fn map<F: Fn(&I) -> O>(input: &<Self as Pack<I>>::T, map_fn: F) -> <Self as Pack<O>>::T;
    fn map_mut<F: Fn(&I, &mut O)>(
        input: &<Self as Pack<I>>::T,
        target: &mut <Self as Pack<O>>::T,
        map_fn: F,
    );
}

impl<I, O> Map<I, O> for () {
    fn map<F: Fn(&I) -> O>(input: &I, map_fn: F) -> O {
        map_fn(input)
    }
    fn map_mut<F: Fn(&I, &mut O)>(
        input: &<Self as Pack<I>>::T,
        target: &mut <Self as Pack<O>>::T,
        map_fn: F,
    ) {
        map_fn(input, target);
    }
}

impl<S, I, O, const L: usize> Map<I, O> for [S; L]
where
    S: Shape + Map<I, O> + Pack<I> + Pack<O>,
    <Self as Pack<O>>::T: LongDefault,
{
    fn map<F: Fn(&I) -> O>(input: &[<S as Pack<I>>::T; L], map_fn: F) -> <Self as Pack<O>>::T {
        let mut target = <<Self as Pack<O>>::T>::long_default();
        for i in 0..L {
            target[i] = <S as Map<I, O>>::map(&input[i], &map_fn);
        }
        target
    }
    fn map_mut<F: Fn(&I, &mut O)>(
        input: &[<S as Pack<I>>::T; L],
        target: &mut [<S as Pack<O>>::T; L],
        map_fn: F,
    ) {
        for i in 0..L {
            <S as Map<I, O>>::map_mut(&input[i], &mut target[i], &map_fn);
        }
    }
}

pub trait IndexMap<O, W: Shape>
where
    Self: Shape + Sized + Pack<O>,
    W: Pack<Self>,
    <W as Pack<Self>>::T: Shape,
{
    fn index_map<F: Fn(<<W as Pack<Self>>::T as Shape>::Index) -> O>(
        outer_index: W::Index,
        map_fn: F,
    ) -> <Self as Pack<O>>::T;
}

impl<O, W> IndexMap<O, W> for ()
where
    W: Shape + Pack<(), T = W>,
{
    fn index_map<F: Fn(W::Index) -> O>(outer_index: W::Index, map_fn: F) -> O {
        map_fn(outer_index)
    }
}

/*
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
    [<O as Element<S>>::Array; L]: Default,
    (u8, ()): Wrap<W::Index, Wrapped = <<[(); L] as Element<W>>::Array as Shape>::Index>,
    W::Index: Copy,
{
    fn index_map<F: Fn(<<[S; L] as Element<W>>::Array as Shape>::Index) -> O>(
        outer_index: W::Index,
        map_fn: F,
    ) -> [<O as Element<S>>::Array; L] {
        let mut target = <[<O as Element<S>>::Array; L]>::default();
        for i in 0..L {
            target[i] = <S as IndexMap<O, <[(); L] as Element<W>>::Array>>::index_map(
                (i as u8, ()).wrap(outer_index),
                &map_fn,
            );
        }
        target
    }
}
*/

pub trait IndexGet<E>
where
    Self: Shape + Pack<E>,
{
    fn index_get(array: &<Self as Pack<E>>::T, i: Self::Index) -> &E;
    fn index_get_mut(array: &mut <Self as Pack<E>>::T, i: Self::Index) -> &mut E;
    fn index_set(mut array: <Self as Pack<E>>::T, i: Self::Index, val: E) -> <Self as Pack<E>>::T {
        *<Self as IndexGet<E>>::index_get_mut(&mut array, i) = val;
        array
    }
}

impl<E> IndexGet<E> for () {
    fn index_get(array: &E, _: ()) -> &E {
        array
    }
    fn index_get_mut(array: &mut E, _: ()) -> &mut E {
        array
    }
}

impl<S, E, const L: usize> IndexGet<E> for [S; L]
where
    S: Shape + IndexGet<E>,
{
    fn index_get(array: &[<S as Pack<E>>::T; L], (i, tail): (u8, S::Index)) -> &E {
        S::index_get(&array[i as usize], tail)
    }
    fn index_get_mut(array: &mut [<S as Pack<E>>::T; L], (i, tail): (u8, S::Index)) -> &mut E {
        S::index_get_mut(&mut array[i as usize], tail)
    }
}

pub trait Fold<B, E>
where
    Self: Shape + Sized + Pack<E>,
{
    fn fold<F: Fn(B, &E) -> B>(input: &<Self as Pack<E>>::T, acc: B, fold_fn: F) -> B;
}

impl<B, E> Fold<B, E> for () {
    fn fold<F: Fn(B, &E) -> B>(input: &E, acc: B, fold_fn: F) -> B {
        fold_fn(acc, input)
    }
}

impl<S, B, E, const L: usize> Fold<B, E> for [S; L]
where
    S: Shape + Fold<B, E> + Pack<E>,
{
    fn fold<F: Fn(B, &E) -> B>(input: &[<S as Pack<E>>::T; L], mut acc: B, fold_fn: F) -> B {
        for i in 0..L {
            acc = <S as Fold<B, E>>::fold(&input[i], acc, &fold_fn);
        }
        acc
    }
}

pub trait ZipMap<A, B, O>
where
    Self: Shape + Sized + Pack<A> + Pack<B> + Pack<O>,
{
    fn zip_map<F: Fn(&A, &B) -> O>(
        a: &<Self as Pack<A>>::T,
        b: &<Self as Pack<B>>::T,
        map_fn: F,
    ) -> <Self as Pack<O>>::T;
    fn zip_map_mut<F: Fn(&A, &B, &mut O)>(
        a: &<Self as Pack<A>>::T,
        b: &<Self as Pack<B>>::T,
        target: &mut <Self as Pack<O>>::T,
        map_fn: F,
    );
}

impl<A, B, O> ZipMap<A, B, O> for () {
    fn zip_map<F: Fn(&A, &B) -> O>(a: &A, b: &B, map_fn: F) -> O {
        map_fn(a, b)
    }
    fn zip_map_mut<F: Fn(&A, &B, &mut O)>(a: &A, b: &B, target: &mut O, map_fn: F) {
        map_fn(a, b, target)
    }
}

impl<S, A, B, O, const L: usize> ZipMap<A, B, O> for [S; L]
where
    S: Shape + ZipMap<A, B, O> + Pack<A> + Pack<B> + Pack<O>,
    [<S as Pack<O>>::T; L]: LongDefault,
{
    fn zip_map<F: Fn(&A, &B) -> O>(
        a: &[<S as Pack<A>>::T; L],
        b: &[<S as Pack<B>>::T; L],
        map_fn: F,
    ) -> [<S as Pack<O>>::T; L] {
        let mut target = <[<S as Pack<O>>::T; L]>::long_default();
        for i in 0..L {
            target[i] = S::zip_map(&a[i], &b[i], &map_fn);
        }
        target
    }
    fn zip_map_mut<F: Fn(&A, &B, &mut O)>(
        a: &[<S as Pack<A>>::T; L],
        b: &[<S as Pack<B>>::T; L],
        target: &mut [<S as Pack<O>>::T; L],
        map_fn: F,
    ) {
        for i in 0..L {
            S::zip_map_mut(&a[i], &b[i], &mut target[i], &map_fn);
        }
    }
}

pub trait LongDefault {
    fn long_default() -> Self;
}

impl LongDefault for f32 {
    fn long_default() -> Self {
        0f32
    }
}

impl LongDefault for i32 {
    fn long_default() -> Self {
        0i32
    }
}
impl LongDefault for usize {
    fn long_default() -> Self {
        0usize
    }
}

impl<A: LongDefault, B: LongDefault> LongDefault for (A, B) {
    fn long_default() -> Self {
        (A::long_default(), B::long_default())
    }
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
impl_long_default_for_array!(25);
impl_long_default_for_array!(32);
impl_long_default_for_array!(64);
impl_long_default_for_array!(128);
impl_long_default_for_array!(256);
impl_long_default_for_array!(512);
impl_long_default_for_array!(1024);
