pub fn flatten_2d<T: Copy + Sized + Default, const A: usize, const B: usize>(input: &[[T; B]; A]) -> [T; A * B] {
    let mut target = [T::default(); A * B];
    for i in 0..A * B {
        target[i] = input[i / B][i % B];
    }
    target
}
