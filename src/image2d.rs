pub trait ImageShape {
    fn dims() -> (usize, usize);
}

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

impl<const X: usize, const Y: usize> ImageShape for [[(); Y]; X] {
    fn dims() -> (usize, usize) {
        (X, Y)
    }
}

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

pub trait PixelMap<I, O, const PX: usize, const PY: usize>
where
    Self: PixelPack<I> + PixelPack<O>,
{
    fn map<F: Fn(&I) -> O>(
        input: &<Self as PixelPack<I>>::I,
        map_fn: F,
    ) -> <Self as PixelPack<O>>::I;
}

impl<I, O, const PX: usize, const PY: usize, const X: usize, const Y: usize> PixelMap<I, O, PX, PY>
    for [[(); Y]; X]
where
    [[(); Y]; X]: ImageShape,
    [[O; Y]; X]: Default,
{
    fn map<F: Fn(&I) -> O>(input: &[[I; Y]; X], map_fn: F) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(X - (PY / 2) * 2) {
                target[x + PX / 2][y + PY / 2] = map_fn(&input[x + PX / 2][y + PY / 2]);
            }
        }
        target
    }
}

pub trait PixelZipMap<A, B, O, const PX: usize, const PY: usize>
where
    Self: PixelPack<A> + PixelPack<B> + PixelPack<O>,
{
    fn zip_map<F: Fn(A, B) -> O>(
        a: &<Self as PixelPack<A>>::I,
        b: &<Self as PixelPack<B>>::I,
        map_fn: F,
    ) -> <Self as PixelPack<O>>::I;
}

impl<A: Copy, B: Copy, O, const PX: usize, const PY: usize, const X: usize, const Y: usize>
    PixelZipMap<A, B, O, PX, PY> for [[(); Y]; X]
where
    [[(); Y]; X]: ImageShape,
    [[O; Y]; X]: Default,
{
    fn zip_map<F: Fn(A, B) -> O>(a: &[[A; Y]; X], b: &[[B; Y]; X], map_fn: F) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..(X - (PX / 2) * 2) {
            for y in 0..(X - (PY / 2) * 2) {
                target[x + PX / 2][y + PY / 2] =
                    map_fn(a[x + PX / 2][y + PY / 2], b[x + PX / 2][y + PY / 2]);
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

pub trait SegmentedPixelFold<
    B,
    P,
    const SX: usize,
    const SY: usize,
    const PX: usize,
    const PY: usize,
> where
    Self: ImageShape + PixelPack<P>,
{
    fn seg_fold<F: Fn(B, &P) -> B>(
        input: &<Self as PixelPack<P>>::I,
        sx: usize,
        sy: usize,
        acc: B,
        fold_fn: F,
    ) -> B;
    fn indexed_seg_fold<F: Fn(B, (usize, usize), &P) -> B>(
        input: &<Self as PixelPack<P>>::I,
        sx: usize,
        sy: usize,
        acc: B,
        fold_fn: F,
    ) -> B;
}

impl<
        P: Copy,
        B,
        const X: usize,
        const Y: usize,
        const SX: usize,
        const SY: usize,
        const PX: usize,
        const PY: usize,
    > SegmentedPixelFold<B, P, SX, SY, PX, PY> for [[(); Y]; X]
{
    fn seg_fold<F: Fn(B, &P) -> B>(
        input: &[[P; Y]; X],
        sx: usize,
        sy: usize,
        mut acc: B,
        fold_fn: F,
    ) -> B {
        assert!(sx < SX);
        assert!(sy < SY);
        let x_seg = (X - (PX / 2) * 2) / SX;
        let y_seg = (Y - (PY / 2) * 2) / SY;
        //dbg!((x_seg * sx, x_seg * (sx + 1)));
        //dbg!((y_seg * sy, y_seg * (sy + 1)));
        for x in x_seg * sx..x_seg * (sx + 1) {
            for y in y_seg * sy..y_seg * (sy + 1) {
                acc = fold_fn(acc, &input[x + PX / 2][y + PY / 2]);
            }
        }
        acc
    }
    fn indexed_seg_fold<F: Fn(B, (usize, usize), &P) -> B>(
        input: &[[P; Y]; X],
        sx: usize,
        sy: usize,
        mut acc: B,
        fold_fn: F,
    ) -> B {
        assert!(sx < SX);
        assert!(sy < SY);
        let x_seg = (X - (PX / 2) * 2) / SX;
        let y_seg = (Y - (PY / 2) * 2) / SY;
        for x in x_seg * sx..x_seg * (sx + 1) {
            for y in y_seg * sy..y_seg * (sy + 1) {
                acc = fold_fn(
                    acc,
                    (x + PX / 2, y + PY / 2),
                    &input[x + PX / 2][y + PY / 2],
                );
            }
        }
        acc
    }
}

pub trait SegmentedConvFold<
    B,
    P,
    const SX: usize,
    const SY: usize,
    const PX: usize,
    const PY: usize,
> where
    Self: ImageShape + PixelPack<P>,
{
    fn seg_conv_fold<F: Fn(B, [[P; PY]; PX]) -> B>(
        input: &<Self as PixelPack<P>>::I,
        sx: usize,
        sy: usize,
        acc: B,
        fold_fn: F,
    ) -> B;
}

impl<
        P: Copy,
        B,
        const X: usize,
        const Y: usize,
        const SX: usize,
        const SY: usize,
        const PX: usize,
        const PY: usize,
    > SegmentedConvFold<B, P, SX, SY, PX, PY> for [[(); Y]; X]
where
    [[P; PY]; PX]: Default,
{
    fn seg_conv_fold<F: Fn(B, [[P; PY]; PX]) -> B>(
        input: &[[P; Y]; X],
        sx: usize,
        sy: usize,
        mut acc: B,
        fold_fn: F,
    ) -> B {
        assert!(sx < SX);
        assert!(sy < SY);
        let x_seg = (X - (PX / 2) * 2) / SX;
        let y_seg = (Y - (PY / 2) * 2) / SY;
        for x in x_seg * sx..x_seg * (sx + 1) {
            for y in y_seg * sy..y_seg * (sy + 1) {
                let mut patch = <[[P; PY]; PX]>::default();
                for px in 0..PX {
                    for py in 0..PY {
                        patch[px][py] = input[x + px][y + py];
                    }
                }
                acc = fold_fn(acc, patch);
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
