pub trait ImageShape {}

pub struct Iter2D {
    max_x: usize,
    min_x: usize,
    cur_x: usize,
    max_y: usize,
    min_y: usize,
    cur_y: usize,
}

impl Iter2D {
    fn new(min_x: usize, max_x: usize, min_y: usize, max_y: usize) -> Self {
        Iter2D {
            max_x: max_x,
            min_x: min_x,
            cur_x: min_x,
            max_y: max_y,
            min_y: min_y,
            cur_y: min_y,
        }
    }
}

impl Iterator for Iter2D {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<(usize, usize)> {
        if self.cur_x < self.max_x {
            if self.cur_y < self.max_y {
                let y = self.cur_y;
                self.cur_y += 1;
                Some((self.cur_x, y))
            } else {
                self.cur_y = self.min_y;
                self.cur_x += 1;
                self.next()
            }
        } else {
            None
        }
    }
}

impl<const X: usize, const Y: usize> ImageShape for [[(); Y]; X] {}

pub trait PixelIndexSGet<I>
where
    Self: PixelPack<I>,
{
    fn get_pixel(input: &<Self as PixelPack<I>>::I, index: (usize, usize)) -> I;
    fn set_pixel_in_place(input: &mut <Self as PixelPack<I>>::I, index: (usize, usize), value: I);
    fn set_pixel(
        mut input: <Self as PixelPack<I>>::I,
        index: (usize, usize),
        value: I,
    ) -> <Self as PixelPack<I>>::I {
        Self::set_pixel_in_place(&mut input, index, value);
        input
    }
}

impl<I: Copy, const X: usize, const Y: usize> PixelIndexSGet<I> for [[(); Y]; X] {
    fn get_pixel(input: &[[I; Y]; X], (x, y): (usize, usize)) -> I {
        input[x][y]
    }
    fn set_pixel_in_place(input: &mut [[I; Y]; X], (x, y): (usize, usize), value: I) {
        input[x][y] = value;
    }
}

pub trait PixelPack<P> {
    type I;
}

impl<P: Sized, const X: usize, const Y: usize> PixelPack<P> for [[(); Y]; X] {
    type I = [[P; Y]; X];
}

pub trait PixelMap<I, O>
where
    Self: PixelPack<I> + PixelPack<O>,
{
    fn map<F: Fn(&I) -> O>(
        input: &<Self as PixelPack<I>>::I,
        map_fn: F,
    ) -> <Self as PixelPack<O>>::I;
}

impl<I, O, const X: usize, const Y: usize> PixelMap<I, O> for [[(); Y]; X]
where
    [[(); Y]; X]: ImageShape,
    [[O; Y]; X]: Default,
{
    fn map<F: Fn(&I) -> O>(input: &[[I; Y]; X], map_fn: F) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..X {
            for y in 0..Y {
                target[x][y] = map_fn(&input[x][y]);
            }
        }
        target
    }
}

pub trait PixelZipMap<A, B, O>
where
    Self: PixelPack<A> + PixelPack<B> + PixelPack<O>,
{
    fn zip_map<F: Fn(A, B) -> O>(
        a: &<Self as PixelPack<A>>::I,
        b: &<Self as PixelPack<B>>::I,
        map_fn: F,
    ) -> <Self as PixelPack<O>>::I;
}

impl<A: Copy, B: Copy, O, const X: usize, const Y: usize> PixelZipMap<A, B, O> for [[(); Y]; X]
where
    [[(); Y]; X]: ImageShape,
    [[O; Y]; X]: Default,
{
    fn zip_map<F: Fn(A, B) -> O>(a: &[[A; Y]; X], b: &[[B; Y]; X], map_fn: F) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..X {
            for y in 0..Y {
                target[x][y] = map_fn(a[x][y], b[x][y]);
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
    fn indices() -> Iter2D;
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
    fn indices() -> Iter2D {
        Iter2D::new(
            0 + PX / 2,
            (X - (PX / 2) * 2) + PX / 2,
            0 + PY / 2,
            (Y - (PY / 2) * 2) + PY / 2,
        )
    }
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
