/// A shape.
/// This trait has no concept of what it contains, just the shape.
/// It is implemented for nested arrays and pairs.
pub trait Shape {
    /// The number of elements in the shape.
    const N: usize;
    /// The type used to index into the shape.
    type Index;
}

impl Shape for () {
    const N: usize = 1;
    type Index = ();
}

impl<T: Shape, const L: usize> Shape for [T; L] {
    const N: usize = T::N * L;
    type Index = (usize, T::Index);
}

impl<T: Shape> Shape for Box<T> {
    const N: usize = T::N;
    type Index = T::Index;
}

/// Given an element and a shape, get the array.
///
/// # Example
///
/// ```
/// type Counters = <u32 as Element<MyShape>>::Array
/// ```
///
pub trait Element<S: Shape>
where
    Self: Sized,
{
    /// The type of the Shape `S` when filled with `Element`s of `Self`.
    type Array;
}

impl<T: Sized> Element<()> for T {
    type Array = T;
}

impl<T: Element<S>, S: Shape, const L: usize> Element<[S; L]> for T {
    type Array = [T::Array; L];
}

impl<T: Element<S>, S: Shape> Element<Box<S>> for T {
    type Array = Box<T::Array>;
}

pub trait MapMut<I: Element<Self>, O: Element<Self>>
where
    Self: Shape + Sized,
{
    fn map_mut<F: Fn(&mut O, &I)>(
        target: &mut <O as Element<Self>>::Array,
        input: &<I as Element<Self>>::Array,
        map_fn: F,
    );
}

impl<I, O> MapMut<I, O> for () {
    fn map_mut<F: Fn(&mut O, &I)>(target: &mut O, input: &I, map_fn: F) {
        map_fn(target, input)
    }
}

impl<S: Shape + MapMut<I, O>, I: Element<S>, O: Element<S>, const L: usize> MapMut<I, O>
    for [S; L]
{
    fn map_mut<F: Fn(&mut O, &I)>(
        target: &mut <O as Element<Self>>::Array,
        input: &[<I as Element<S>>::Array; L],
        map_fn: F,
    ) {
        for i in 0..L {
            <S as MapMut<I, O>>::map_mut(&mut target[i], &input[i], &map_fn);
        }
    }
}

impl<S: Shape + MapMut<I, O>, I: Element<S>, O: Element<S>> MapMut<I, O> for Box<S> {
    fn map_mut<F: Fn(&mut O, &I)>(
        target: &mut <O as Element<Self>>::Array,
        input: &<I as Element<Self>>::Array,
        map_fn: F,
    ) {
        <S as MapMut<I, O>>::map_mut(target, &input, &map_fn);
    }
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

impl<S: MapMut<I, O> + Shape, I: Element<S>, O: Element<S>> Map<I, O> for S
where
    <O as Element<S>>::Array: Default,
{
    fn map<F: Fn(&I) -> O>(
        input: &<I as Element<S>>::Array,
        map_fn: F,
    ) -> <O as Element<S>>::Array {
        let mut target = <O as Element<S>>::Array::default();
        S::map_mut(&mut target, input, |t, i| *t = map_fn(i));
        target
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

pub trait ZipFold<Acc, A: Element<Self>, B: Element<Self>>
where
    Self: Shape + Sized,
{
    fn zip_fold<F: Fn(Acc, &A, &B) -> Acc>(
        a: &<A as Element<Self>>::Array,
        b: &<B as Element<Self>>::Array,
        acc: Acc,
        fold_fn: F,
    ) -> Acc;
}

impl<Acc, A, B> ZipFold<Acc, A, B> for () {
    fn zip_fold<F: Fn(Acc, &A, &B) -> Acc>(a: &A, b: &B, acc: Acc, fold_fn: F) -> Acc {
        fold_fn(acc, a, b)
    }
}

impl<
        S: Shape + ZipFold<Acc, A, B>,
        Acc,
        A: Element<S> + Sized,
        B: Element<S> + Sized,
        const L: usize,
    > ZipFold<Acc, A, B> for [S; L]
{
    fn zip_fold<F: Fn(Acc, &A, &B) -> Acc>(
        a: &[<A as Element<S>>::Array; L],
        b: &[<B as Element<S>>::Array; L],
        mut acc: Acc,
        fold_fn: F,
    ) -> Acc {
        for i in 0..L {
            acc = <S as ZipFold<Acc, A, B>>::zip_fold(&a[i], &b[i], acc, &fold_fn);
        }
        acc
    }
}

pub trait ZipMapMut<A: Element<Self>, B: Element<Self>, O: Element<Self>>
where
    Self: Shape + Sized,
{
    fn zip_map_mut<F: Fn(&mut O, &A, &B)>(
        target: &mut <O as Element<Self>>::Array,
        a: &<A as Element<Self>>::Array,
        b: &<B as Element<Self>>::Array,
        map_fn: F,
    );
}

impl<A: Element<(), Array = A> + Copy, B: Element<(), Array = B> + Copy, O> ZipMapMut<A, B, O>
    for ()
{
    fn zip_map_mut<F: Fn(&mut O, &A, &B)>(target: &mut O, a: &A, b: &B, map_fn: F) {
        map_fn(target, a, b)
    }
}

impl<
        S: Shape + ZipMapMut<A, B, O>,
        A: Element<S>,
        B: Element<S>,
        O: Element<S>,
        const L: usize,
    > ZipMapMut<A, B, O> for [S; L]
{
    fn zip_map_mut<F: Fn(&mut O, &A, &B)>(
        target: &mut <O as Element<Self>>::Array,
        a: &<A as Element<[S; L]>>::Array,
        b: &<B as Element<[S; L]>>::Array,
        map_fn: F,
    ) {
        for i in 0..L {
            S::zip_map_mut(&mut target[i], &a[i], &b[i], &map_fn);
        }
    }
}
impl<S: Shape + ZipMapMut<A, B, O>, A: Element<S>, B: Element<S>, O: Element<S>> ZipMapMut<A, B, O>
    for Box<S>
{
    fn zip_map_mut<F: Fn(&mut O, &A, &B)>(
        target: &mut <O as Element<Self>>::Array,
        a: &<A as Element<Self>>::Array,
        b: &<B as Element<Self>>::Array,
        map_fn: F,
    ) {
        S::zip_map_mut(target, &a, &b, &map_fn);
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

impl<S: Shape + ZipMapMut<A, B, O>, A: Element<S>, B: Element<S>, O: Element<S>> ZipMap<A, B, O>
    for S
where
    <O as Element<S>>::Array: Default,
{
    fn zip_map<F: Fn(&A, &B) -> O>(
        a: &<A as Element<S>>::Array,
        b: &<B as Element<S>>::Array,
        map_fn: F,
    ) -> <O as Element<S>>::Array {
        let mut target = <<O as Element<S>>::Array>::default();
        S::zip_map_mut(&mut target, &a, &b, |t, x, y| *t = map_fn(x, y));
        target
    }
}

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
    [<T as Element<S>>::Array; L]: Default,
{
    fn from_vec(slice: &[T]) -> [<T as Element<S>>::Array; L] {
        assert_eq!(slice.len(), S::N * L);
        let mut target = <[<T as Element<S>>::Array; L]>::default();
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
