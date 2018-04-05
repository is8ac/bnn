#[macro_use]
extern crate bitnn;
extern crate rand;

fn main() {
    let mut params = [[0u64; 3]; 2 * 64];
    for o in 0..2 * 64 {
        for i in 0..3 {
            params[o][i] = rand::random::<u64>()
        }
    }
    let layer1 = dense_bits2bits!(3, 2);
    let layer1_grad = dense_bits2bits_grad!(3, 2);
    let empirical_grad = dense_empirical_grad!(3, 2);
    let inputs = [rand::random::<u64>(), rand::random::<u64>(), rand::random::<u64>()];

    let target_grads = ([rand::random::<u64>(), rand::random::<u64>()], [!0b0u64, !0b0u64]);
    println!("target: {:064b}", target_grads.0[0]);
    println!("input: {:064b}", inputs[0]);
    let mut actuals = [0u64; 2];
    layer1(&mut actuals, &params, &inputs);
    let grads = layer1_grad(&target_grads, &params, &inputs);

    let layer_fn = |input: &[u64; 3]| -> [u64; 2] {
        let mut outputs = [0u64; 2];
        layer1(&mut outputs, &params, &input);
        outputs
    };
    let eg = empirical_grad(&inputs, &target_grads, &layer_fn);
    println!("{:?}", eg);

    println!("grads {:064b}", grads.0[0]);
    println!("grads {:?}", grads.0);
    println!("mask {:064b}", grads.1[0]);
    print!("targets:");
    for target in target_grads.0.iter() {
        print!(" {:064b}", target);
    }
    print!("\n",);
    print!("actuals:");
    for target in actuals.iter() {
        print!(" {:064b}", target);
    }
    print!("\n",);
    print!("inputs:  ");
    for input in inputs.iter() {
        print!(" {:064b}", input);
    }
    print!("\n",);
    print!("eg       ");
    for g in eg.0.iter() {
        print!(" {:064b}", g);
    }
    print!("\n",);
    print!("eg masks:");
    for m in eg.1.iter() {
        print!(" {:064b}", m);
    }
    print!("\n",);
    print!("cg       ");
    for g in grads.0.iter() {
        print!(" {:064b}", g);
    }
    print!("\n",);
    print!("cg masks:");
    for m in grads.1.iter() {
        print!(" {:064b}", m);
    }
    print!("\n",);
}
