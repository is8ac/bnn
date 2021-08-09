use std::convert::TryInto;

fn quantize<const I: usize>(
    n: u64,
    t: u64,
    i: &[u64; I],
    i_xor_t: &[u64; I],
    i_cv: &[[u64; I]; I],
) -> ([Option<bool>; I], u32) {
    let cv_probs: [[f64; I]; I] = i_cv
        .iter()
        .map(|x| {
            x.iter()
                .map(|&c| -((c as f64 / n as f64) * 2f64 - 1f64).abs())
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap()
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mut targ_magns: Vec<f64> = i_xor_t
        .iter()
        .map(|&x| ((x as f64 / n as f64) * 2f64 - 1f64).abs())
        .collect();
    let mut indices: Vec<usize> = Vec::new();
    let mut cost = 0f64;
    while cost < 1f64 {
        let mut values: Vec<(usize, f64)> = cv_probs
            .iter()
            .zip(targ_magns.iter())
            .map(|(cv, p)| indices.iter().map(|&index| cv[index]).product::<f64>() * p)
            .enumerate()
            .collect();
        dbg!(&values);
        values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let (i, weight) = values.pop().unwrap();
        dbg!(i);
        indices.push(i);
        cost += targ_magns[i];
        dbg!(cost);
    }
    ([None; I], 0u32)
}

fn main() {
    let examples = vec![
        (0b_10001110_u8, true),
        (0b_11001110_u8, true),
        (0b_00011111_u8, true),
        (0b_10100001_u8, true),
        (0b_01001100_u8, true),
        (0b_01001110_u8, false),
        (0b_00001110_u8, false),
        (0b_11100011_u8, false),
        (0b_00111111_u8, false),
        (0b_10101010_u8, false),
    ];
    let counts = examples.iter().fold(
        (0u64, 0u64, [[0u64; 8]; 8], [0u64; 8], [0u64; 8]),
        |mut acc, (i, t)| {
            acc.0 += 1;
            acc.1 += *t as u64;
            for b in 0..8 {
                let b_bit = ((i >> b) & 1u8) == 1;
                acc.3[b] += b_bit as u64;
                acc.4[b] += (b_bit ^ t) as u64;
                for a in 0..8 {
                    let a_bit = ((i >> a) & 1u8) == 1;
                    acc.2[a][b] += (a_bit ^ b_bit) as u64;
                }
            }
            acc
        },
    );
    //dbg!(&counts);
    quantize::<8>(counts.0, counts.1, &counts.3, &counts.4, &counts.2);
}
