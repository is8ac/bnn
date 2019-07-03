#![feature(const_generics)]

macro_rules! patch_3x3 {
    ($input:expr, $x:expr, $y:expr) => {
        [
            [
                $input[$x + 0][$y + 0],
                $input[$x + 0][$y + 1],
                $input[$x + 0][$y + 2],
            ],
            [
                $input[$x + 1][$y + 0],
                $input[$x + 1][$y + 1],
                $input[$x + 1][$y + 2],
            ],
            [
                $input[$x + 2][$y + 0],
                $input[$x + 2][$y + 1],
                $input[$x + 2][$y + 2],
            ],
        ]
    };
}
trait Wrap<T> {
    type Wrap;
}

impl<T, I: Wrap<T>, const L: usize> Wrap<[T; L]> for I {
    type Wrap = [<I as Wrap<T>>::Wrap; L];
}

pub trait BitLen: Sized {
    const BIT_LEN: usize;
}

pub trait BitMul<I, O> {
    fn bit_mul(&self, input: &I) -> O;
}

pub trait HammingDistance {
    fn hamming_distance(&self, other: &Self) -> u32;
}

trait NestedIndex {
    type IndexType;
    fn flip_bit_index(&mut self, index: &Self::IndexType);
}

impl<T: UInt> NestedIndex for T {
    type IndexType = usize;
    fn flip_bit_index(&mut self, &index: &Self::IndexType) {
        self.flip_bit(index);
    }
}

trait IncrementCounters
where
    Self: NestedIndex,
{
    type BitCounterType;
    fn increment_counters(&self, counters: &mut Self::BitCounterType);
    fn top1_index(
        &self,
        a: &Self::BitCounterType,
        b: &Self::BitCounterType,
    ) -> (Option<Self::IndexType>, u32);
    fn compare_and_bitpack(
        counters_0: &Self::BitCounterType,
        counters_1: &Self::BitCounterType,
    ) -> Self;
}

trait UInt {
    const UINT_BIT_LEN: usize;
    fn flip_bit(&mut self, b: usize);
    fn bit(&self, i: usize) -> bool;
}

macro_rules! impl_bitlen_for_uint {
    ($type:ty, $len:expr) => {
        impl UInt for $type {
            const UINT_BIT_LEN: usize = $len;
            fn flip_bit(&mut self, index: usize) {
                *self ^= 1 << index
            }
            fn bit(&self, i: usize) -> bool {
                ((self >> i) & 1) == 1
            }
        }
        impl IncrementCounters for $type {
            type BitCounterType = [u32; <$type>::BIT_LEN];
            fn increment_counters(&self, counters: &mut Self::BitCounterType) {
                for b in 0..<$type>::BIT_LEN {
                    counters[b] += ((self >> b) & 1) as u32
                }
            }
            fn top1_index(
                &self,
                a: &Self::BitCounterType,
                b: &Self::BitCounterType,
            ) -> (Option<Self::IndexType>, u32) {
                let mut max_diff = 0u32;
                let mut index = None;
                for i in 0..$len {
                    let grad_sign = a[i] > b[i];
                    let bit_sign = self.bit(i);
                    // if the current sign is not the same as the gradient, then we can update.
                    if bit_sign ^ grad_sign {
                        let diff = a[i].saturating_sub(b[i]) | b[i].saturating_sub(a[i]);
                        if diff > max_diff {
                            max_diff = diff;
                            index = Some(i);
                        }
                    }
                }
                (index, max_diff)
            }
            fn compare_and_bitpack(
                counters_0: &Self::BitCounterType,
                counters_1: &Self::BitCounterType,
            ) -> Self {
                let mut target = <$type>::default();
                for b in 0..$len {
                    target |= ((counters_0[b] > counters_1[b]) as $type) << b;
                }
                target
            }
        }
        impl<I> Wrap<$type> for I {
            type Wrap = [I; <$type>::BIT_LEN];
        }
        impl BitLen for $type {
            const BIT_LEN: usize = $len;
        }
        impl<I: HammingDistance + BitLen> BitMul<I, $type> for [I; $len] {
            fn bit_mul(&self, input: &I) -> $type {
                let mut target = <$type>::default();
                for i in 0..$len {
                    target |=
                        ((self[i].hamming_distance(input) > (I::BIT_LEN as u32 / 2)) as $type) << i;
                }
                target
            }
        }
        impl HammingDistance for $type {
            #[inline(always)]
            fn hamming_distance(&self, other: &$type) -> u32 {
                (self ^ other).count_ones()
            }
        }
        impl<Input: IncrementCounters + BitLen + HammingDistance>
            IncrementMatrixCounters<Input, $type> for [Input; $len]
        {
            type MatrixBitCounterType = [(Input::BitCounterType, Input::BitCounterType); $len];
            // No grad is:
            //      _____
            //     |
            // ____|
            // current is:
            //       _____
            //      /
            // ____/
            // where the width is adjustable.
            fn increment_matrix_counters(
                &self,
                counters: &mut Self::MatrixBitCounterType,
                input: &Input,
                target: &$type,
                tanh_width: u32,
            ) {
                for b in 0..<$type>::BIT_LEN {
                    let activation = self[b].hamming_distance(&input);
                    let threshold = Input::BIT_LEN as u32 / 2;
                    // this patch only gets to vote if it is within tanh_width.
                    let diff =
                        activation.saturating_sub(threshold) | threshold.saturating_sub(activation);
                    if diff < tanh_width {
                        if target.bit(b) {
                            input.increment_counters(&mut counters[b].0);
                        } else {
                            input.increment_counters(&mut counters[b].1);
                        }
                    }
                }
            }
            fn matrix_top1_index(
                &self,
                counters: &Self::MatrixBitCounterType,
            ) -> (Option<Self::IndexType>, u32) {
                let mut max_diff = 0u32;
                let mut index = None;
                for i in 0..$len {
                    let (sub_index, diff) = self[i].top1_index(&counters[i].0, &counters[i].1);
                    if diff > max_diff {
                        max_diff = diff;
                        index = sub_index.map(|x| (x, i));
                    }
                }
                (index, max_diff)
            }
        }
        impl<Input: IncrementCounters + HammingDistance + BitLen> Backprop<Input, $type>
            for [Input; $len]
        where
            Input::BitCounterType: Default,
        {
            fn backprop(&self, input: &Input, output: &$type, tanh_width: u32) -> Input {
                let mut counters_0 = Input::BitCounterType::default();
                let mut counters_1 = Input::BitCounterType::default();
                let threshold = Input::BIT_LEN as u32 / 2;
                for i in 0..<$type>::BIT_LEN {
                    let activation = input.hamming_distance(&self[i]);
                    // this output only gets to vote if it is within tanh_width.
                    let diff =
                        activation.saturating_sub(threshold) | threshold.saturating_sub(activation);
                    if diff < tanh_width {
                        if output.bit(i) {
                            self[i].increment_counters(&mut counters_0);
                        } else {
                            self[i].increment_counters(&mut counters_1);
                        }
                    }
                }
                Input::compare_and_bitpack(&counters_0, &counters_1)
            }
        }
    };
}

impl_bitlen_for_uint!(u8, 8);
impl_bitlen_for_uint!(u16, 16);
impl_bitlen_for_uint!(u32, 32);
impl_bitlen_for_uint!(u64, 64);
impl_bitlen_for_uint!(u128, 128);

macro_rules! array_bit_len {
    ($len:expr) => {
        impl<T: BitLen> BitLen for [T; $len] {
            const BIT_LEN: usize = $len * T::BIT_LEN;
        }
    };
}

array_bit_len!(1);
array_bit_len!(2);
array_bit_len!(3);
array_bit_len!(4);
array_bit_len!(5);
array_bit_len!(6);
array_bit_len!(7);
array_bit_len!(8);
array_bit_len!(13);
array_bit_len!(16);
array_bit_len!(32);

impl<T: NestedIndex, const L: usize> NestedIndex for [T; L] {
    type IndexType = (T::IndexType, usize);
    fn flip_bit_index(&mut self, (sub_index, i): &Self::IndexType) {
        self[*i].flip_bit_index(sub_index);
    }
}

impl<T: IncrementCounters, const L: usize> IncrementCounters for [T; L]
where
    Self: Default,
{
    type BitCounterType = [T::BitCounterType; L];
    fn increment_counters(&self, counters: &mut [T::BitCounterType; L]) {
        for i in 0..L {
            self[i].increment_counters(&mut counters[i]);
        }
    }
    fn top1_index(
        &self,
        a: &Self::BitCounterType,
        b: &Self::BitCounterType,
    ) -> (Option<Self::IndexType>, u32) {
        let mut max_diff = 0u32;
        let mut index = None;
        for i in 0..L {
            let (sub_index, diff) = self[i].top1_index(&a[i], &b[i]);
            if diff > max_diff {
                max_diff = diff;
                index = sub_index.map(|x| (x, i));
            }
        }
        (index, max_diff)
    }
    fn compare_and_bitpack(
        counters_0: &Self::BitCounterType,
        counters_1: &Self::BitCounterType,
    ) -> Self {
        let mut target = Self::default();
        for i in 0..L {
            target[L] = T::compare_and_bitpack(&counters_0[i], &counters_1[i]);
        }
        target
    }
}

impl<T: HammingDistance, const L: usize> HammingDistance for [T; L] {
    fn hamming_distance(&self, other: &[T; L]) -> u32 {
        let mut distance = 0u32;
        for i in 0..L {
            distance += self[i].hamming_distance(&other[i]);
        }
        distance
    }
}

trait IncrementMatrixCounters<Input, Target>
where
    Self: NestedIndex,
{
    type MatrixBitCounterType;
    fn increment_matrix_counters(
        &self,
        counters: &mut Self::MatrixBitCounterType,
        input: &Input,
        target: &Target,
        tanh_width: u32,
    );
    fn matrix_top1_index(
        &self,
        counters: &Self::MatrixBitCounterType,
    ) -> (Option<Self::IndexType>, u32);
}

impl<
        Input,
        Target,
        MatrixBits: IncrementMatrixCounters<Input, Target> + Copy + Default,
        const L: usize,
    > IncrementMatrixCounters<Input, [Target; L]> for [MatrixBits; L]
{
    type MatrixBitCounterType = [MatrixBits::MatrixBitCounterType; L];
    fn increment_matrix_counters(
        &self,
        counters: &mut Self::MatrixBitCounterType,
        input: &Input,
        target: &[Target; L],
        tanh_width: u32,
    ) {
        for i in 0..L {
            self[i].increment_matrix_counters(&mut counters[i], input, &target[i], tanh_width);
        }
    }
    fn matrix_top1_index(
        &self,
        counters: &Self::MatrixBitCounterType,
    ) -> (Option<Self::IndexType>, u32) {
        let mut max_diff = 0u32;
        let mut index = None;
        for i in 0..L {
            let (sub_index, diff) = self[i].matrix_top1_index(&counters[i]);
            if diff > max_diff {
                max_diff = diff;
                index = sub_index.map(|x| (x, i));
            }
        }
        (index, max_diff)
    }
}

impl<I, O: Default + Copy, T: BitMul<I, O>, const L: usize> BitMul<I, [O; L]> for [T; L]
where
    [O; L]: Default,
{
    fn bit_mul(&self, input: &I) -> [O; L] {
        let mut target = <[O; L]>::default();
        for i in 0..L {
            target[i] = self[i].bit_mul(input);
        }
        target
    }
}

trait Conv2D<I, O> {
    fn conv2d(&self, input: &I) -> O;
}

impl<I: Copy, O: Default + Copy, W: BitMul<[[I; 3]; 3], O>, const X: usize, const Y: usize>
    Conv2D<[[I; Y]; X], [[O; Y]; X]> for W
where
    [[O; Y]; X]: Default,
{
    fn conv2d(&self, input: &[[I; Y]; X]) -> [[O; Y]; X] {
        let mut target = <[[O; Y]; X]>::default();
        for x in 0..X - 2 {
            for y in 0..Y - 2 {
                target[x + 1][y + 1] = self.bit_mul(&patch_3x3!(input, x, y));
            }
        }
        target
    }
}

trait Backprop<I, O> {
    fn backprop(&self, input: &I, output: &O, tanh_width: u32) -> I;
}

trait AutoencoderSample<Embedding>
where
    Self: Wrap<Embedding> + Sized,
    Embedding: Wrap<Self> + Sized,
    <Self as Wrap<Embedding>>::Wrap: IncrementMatrixCounters<Self, Embedding>,
    <Embedding as Wrap<Self>>::Wrap: IncrementMatrixCounters<Embedding, Self>,
{
    fn update_counters(
        &self,
        encoder: &<Self as Wrap<Embedding>>::Wrap,
        encoder_counter: &mut <<Self as Wrap<Embedding>>::Wrap as IncrementMatrixCounters<
            Self,
            Embedding,
        >>::MatrixBitCounterType,
        decoder: &<Embedding as Wrap<Self>>::Wrap,
        decoder_counter: &mut <<Embedding as Wrap<Self>>::Wrap as IncrementMatrixCounters<
            Embedding,
            Self,
        >>::MatrixBitCounterType,
    );
}

impl<Input, Embedding> AutoencoderSample<Embedding> for Input
where
    Self: Wrap<Embedding> + Sized,
    Embedding: Wrap<Self> + Sized,
    <Self as Wrap<Embedding>>::Wrap:
        IncrementMatrixCounters<Self, Embedding> + BitMul<Input, Embedding>,
    <Embedding as Wrap<Self>>::Wrap:
        IncrementMatrixCounters<Embedding, Self> + BitMul<Embedding, Self>,
{
    fn update_counters(
        &self,
        encoder: &<Self as Wrap<Embedding>>::Wrap,
        encoder_counter: &mut <<Self as Wrap<Embedding>>::Wrap as IncrementMatrixCounters<
            Self,
            Embedding,
        >>::MatrixBitCounterType,
        decoder: &<Embedding as Wrap<Self>>::Wrap,
        decoder_counter: &mut <<Embedding as Wrap<Self>>::Wrap as IncrementMatrixCounters<
            Embedding,
            Self,
        >>::MatrixBitCounterType,
    ) {
        let mut embedding = encoder.bit_mul(self);
        let actual = decoder.bit_mul(&embedding);
    }
}

fn main() {
    {
        let example = [[0b11001011u8; 3]; 3];
        let encoder = [[[[0b11010101u8; 3]; 3]; 16]; 2];
        let decoder = [[[[0b1101001010101010u16; 2]; 8]; 3]; 3];
        let mut encoder_counters = [[([[[0u32; 8]; 3]; 3], [[[0u32; 8]; 3]; 3]); 16]; 2];
        let mut decoder_counters = [[[([[0u32; 16]; 2], [[0u32; 16]; 2]); 8]; 3]; 3];

        <[[u8; 3]; 3] as AutoencoderSample<[u16; 2]>>::update_counters(
            &example,
            &encoder,
            &mut encoder_counters,
            &decoder,
            &mut decoder_counters,
        );
    }

    let mut weights = [[[[0b1100_1101u8; 3]; 3]; 16]; 5];
    let input = [[0b11001u8; 3]; 3];
    let output = weights.bit_mul(&input);

    let mut counters = [[[0u32; 8]; 3]; 3];
    input.increment_counters(&mut counters);

    let mut matrix_counters = [[([[[0u32; 8]; 3]; 3], [[[0u32; 8]; 3]; 3]); 16]; 5];
    weights.increment_matrix_counters(&mut matrix_counters, &input, &output, 7);
    weights.increment_matrix_counters(&mut matrix_counters, &input, &[0b10010001010010u16; 5], 7);
    weights.increment_matrix_counters(
        &mut matrix_counters,
        &[[0b10101010u8; 3]; 3],
        &[0b10010001010010u16; 5],
        7,
    );
    weights.increment_matrix_counters(
        &mut matrix_counters,
        &[[0b1100_1010u8; 3]; 3],
        &[0b1000101010001u16; 5],
        7,
    );
    weights.increment_matrix_counters(
        &mut matrix_counters,
        &[[0b1100_1010u8; 3]; 3],
        &[0b1000101010001u16; 5],
        7,
    );

    let (index, diff) = weights.matrix_top1_index(&matrix_counters);
    weights.flip_bit_index(&index.unwrap());

    let a: [[[u32; 8]; 3]; 3] = [[[5, 0, 2, 8, 6, 2, 0, 4]; 3]; 3];
    let b: [[[u32; 8]; 3]; 3] = [[[0, 7, 0, 0, 4, 7, 20, 50]; 3]; 3];
    let mut target = [[0b1001_0101u8; 3]; 3];
    let (index, diff) = target.top1_index(&a, &b);
    dbg!(diff);
    dbg!(index);
    println!("{:08b}", target[0][0]);
    target.flip_bit_index(&index.unwrap());
    println!("{:08b}", target[0][0]);
    let (index, diff) = target.top1_index(&a, &b);
    target.flip_bit_index(&index.unwrap());
    println!("{:08b}", target[0][0]);

    {
        let matrix = [
            0b1100_0000_1100_0010u16,
            0b0011_1010_0011_1011u16,
            0b0011_1010_0011_1100u16,
            0b1110_1101_1110_1101u16,
            0b1111_1100_0011_1100u16,
            0b0011_1010_0011_1010u16,
            0b1001_1100_1111_1100u16,
            0b0011_1010_0011_1010u16,
        ];
        let input = 0b0001_0011_0001_0011u16;
        let target = 0b1101_0101u8;
        let actual: u8 = matrix.bit_mul(&input);
        println!("actual: {:08b}", actual);
        dbg!(target.hamming_distance(&actual));
        let result = matrix.backprop(&input, &target, 3);
        println!("result: {:016b}", result);
        let new_output = matrix.bit_mul(&result);
        println!("new:    {:016b}", new_output);
        dbg!(target.hamming_distance(&actual));
    }
}

// only one bit of embedding has gradient
// only incrament one output of the encoder matrix counters.
// Backprop on decoder return a bit and an index.
// Then encoder matrix update is only for one output

// for each example in minibatch
// value -> encoder_matrix -> example_embedding
// example_embedding -> decoder_matrix -> value_activations
// filter value activations to close to center
// outputs that are close to center incrament minibatch decoder matrix counters _and_ example_embedding counters
// Take top mag example embedding index and use it to incrament encoder matrix counters.
// This looses information, but allows us to move slowly, allowing small minibatches.
