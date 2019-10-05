#![feature(const_generics)]

trait Shape {
    const N: usize;
    type Index;
    const SHAPE: Self::Index;
}

impl Shape for () {
    const N: usize = 1;
    type Index = ();
    const SHAPE: Self::Index = ();
}

impl<T: Shape, const L: usize> Shape for [T; L] {
    const N: usize = T::N * L;
    type Index = (usize, T::Index);
    const SHAPE: Self::Index = (L, T::SHAPE);
}

trait Array<S: Shape> {
    type Array;
}

impl<T: Sized> Array<()> for T {
    type Array = T;
}

impl<T: Array<S>, S: Shape, const L: usize> Array<[S; L]> for T {
    type Array = [T; L];
}

trait Map<S: Shape, I: Array<S, Array = Self>, O: Array<S>>
where
    Self: Sized,
{
    fn map<F: Fn(&I) -> O>(&self, map_fn: F) -> O::Array;
}

impl<I, O> Map<(), I, O> for I {
    fn map<F: Fn(&I) -> O>(&self, map_fn: F) -> O {
        map_fn(self)
    }
}

impl<S: Shape, I: Array<S>, O: Array<S>, const L: usize> Map<[S; L], I, O>
    for [<I as Array<S>>::Array; L]
where
    [<O as Array<S>>::Array; L]: Default,
    <I as Array<S>>::Array: Map<S, I, O>,
    I: Array<[S; L], Array = [<I as Array<S>>::Array; L]>,
    O: Array<[S; L], Array = [<O as Array<S>>::Array; L]>,
{
    fn map<F: Fn(&I) -> O>(&self, map_fn: F) -> [<O as Array<S>>::Array; L] {
        let mut target = <[<O as Array<S>>::Array; L]>::default();
        for i in 0..L {
            target[i] = <<I as Array<S>>::Array as Map<S, I, O>>::map(&self[i], &map_fn);
        }
        target
    }
}

fn main() {
    let a = [5u8; 3];
    let b = <[u8; 3] as Map<[(); 3], u8, f64>>::map(&a, |&x| x as f64 + 1f64);
    dbg!(b);
}
