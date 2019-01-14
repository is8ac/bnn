extern crate bitnn;
extern crate rayon;
extern crate time;
use time::PreciseTime;

use bitnn::datasets::cifar;
//use bitnn::featuregen;
use bitnn::layers::unary;
use bitnn::layers::{
    ExtractPatchesNotched3x3, ExtractPixels, ExtractPixelsPadded, Layer, NewFromSplit,
    ObjectiveHead, OrPool2x2, Patch3x3NotchedConv, PatchMap, PixelMap, PoolOrLayer, VecApply,
};
use std::marker::PhantomData;

fn load_data() -> Vec<(usize, [[[u8; 3]; 32]; 32])> {
    let size: usize = 10000;
    let paths = vec![
        String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_1.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_2.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_3.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_4.bin"),
        //String::from("/home/isaac/big/cache/datasets/cifar-10-batches-bin/data_batch_5.bin"),
    ];
    let mut images = Vec::with_capacity(size * paths.len());
    for path in &paths {
        let mut batch = cifar::load_images_10(&path, size);
        images.append(&mut batch);
    }
    images
}

fn rgb_to_u16(pixels: [u8; 3]) -> u16 {
    unary::to_5(pixels[0]) as u16
        | ((unary::to_5(pixels[1]) as u16) << 5)
        | ((unary::to_6(pixels[2]) as u16) << 10)
}

//let mut model = Layer::<
//    [[u16; 32]; 32],
//    Patch3x3NotchedConv<u16, u128, [u128; 16], u16>,
//    [[u16; 32]; 32],
//    Layer<
//        [[u16; 32]; 32],
//        OrPool2x2<u16>,
//        [[u16; 16]; 16],
//        Layer<[[u16; 16]; 16], ExtractPatchesNotched3x3<u128>, u128, [u128; 10]>,
//    >,
//>::new_from_split(&unary_examples);
const L1: [u128; 16] = [
    320016523005042211355119935188195368960,
    6516431673409738999337428812657304080,
    254036999083193581517253470237661152263,
    336133496246653883067095084278149610462,
    337722524559230170681001184751793594368,
    145885001100845490215854175261391189788,
    165491400532389605783882490841355318040,
    339581396149071114831840923326447946849,
    173129400969661700255613818043630861278,
    195317737294471175610502146359340040288,
    6734521927373976340048401644337480086,
    151404534436709549959942561664159971552,
    37296837211650027928249234744434130248,
    103515209688501761246109999006646336478,
    163653267706160522005026499464409432592,
    271746926018881639012806747997744455137,
];

//let mut model = Layer::<
//    [[u16; 16]; 16],
//    Patch3x3NotchedConv<u16, u128, [u128; 32], u32>,
//    [[u32; 16]; 16],
//    Layer<
//        [[u32; 16]; 16],
//        OrPool2x2<u32>,
//        [[u32; 8]; 8],
//        Layer<[[u32; 8]; 8], ExtractPatchesNotched3x3<[u128; 2]>, [u128; 2], [[u128; 2]; 10]>,
//    >,
//>::new_from_split(&pooled_examples);
//const L2: [u128; 32] = [
//    320203608929588404607200284801098727012,
//    115584235957274597758935862420216313582,
//    318495114852301792394132269907091588952,
//    105120554281504422028490746154207013278,
//    190131793473998248728621366583425295786,
//    142656922012894147395293278226292992441,
//    190173512631224601257316020015111915413,
//    269464282169741487130976690102521653417,
//    237842000315782917125517575832023276867,
//    18944987524367789953778890440386637905,
//    246782603332350584401217548633863676017,
//    262923476933728450849888038130550162497,
//    231616541675328265407309980563456102246,
//    118451781670332778268331767133218757650,
//    188805848706291760795320533126875912842,
//    162305532739797242330786210169620881661,
//    126702068663395025966729888338479009456,
//    161994395570393248232064655263408670679,
//    328253740924640298690688555643562533805,
//    29243462127171015820458069004696819394,
//    54374562432173583665230946860175653103,
//    195254069537716779549014904561801229378,
//    183740462540316832572584019030793850698,
//    124098355454123659308580549739307999130,
//    315705586395645459207322157580878579445,
//    160937714900783196826417517683374690055,
//    104166688776715718010495230782991765008,
//    105111789235704510287930210240279481233,
//    326989924592087137963117514930394412650,
//    8318120579881123713235281705974253233,
//    54400520391436822223893657321930221806,
//    32905296666444912688768946184517684736,
//];

//let mut model = Layer::<
//    [[u16; 16]; 16],
//    Patch3x3NotchedConv<u16, u128, [u128; 16], u16>,
//    [[u16; 16]; 16],
//    Layer<
//        [[u16; 16]; 16],
//        OrPool2x2<u16>,
//        [[u16; 8]; 8],
//        Layer<[[u16; 8]; 8], ExtractPatchesNotched3x3<u128>, u128, [u128; 10]>,
//    >,
//>::new_from_split(&pooled_examples);
const L2: [u128; 16] = [
    330876875396407982019747843250858592650,
    30489625614124926951260209473964784232,
    317524621847877517652819410016095717277,
    126387585473424365469974119553896050568,
    18669859987915313954738678313280917382,
    159652583644906840035317341129312552827,
    105496635806490282938906150802183130190,
    167634603681508128900198666530996317975,
    237841980350614769304067849461684673804,
    2991168639094990020978869643280474961,
    139021025451701771653177175745157719543,
    114936468083744746384022544327931939617,
    231595429182202721850620391637542095676,
    109143897577164067836624234425376818708,
    232753449198719358512060005017738930672,
    117651863021502371043324564570918842621,
];

const MOD: usize = 7;

fn main() {
    for i in &L1 {
        println!("{:0128b}", i);
    }
    //let start = PreciseTime::now();
    let examples = load_data();
    let unary_examples: Vec<(usize, [[u16; 32]; 32])> = examples
        .iter()
        .map(|(class, image)| {
            (
                *class,
                image.patch_map(&|input, output: &mut u16| *output = rgb_to_u16(*input)),
            )
        })
        .collect();

    let layer1 = Patch3x3NotchedConv::new_from_parameters(L1);
    let l1_examples = layer1.vec_apply(&unary_examples);
    let pooled_examples = OrPool2x2::<u16>::new_from_split(&l1_examples).vec_apply(&l1_examples);

    let layer2 = Patch3x3NotchedConv::new_from_parameters(L2);
    let l2_examples = layer2.vec_apply(&pooled_examples);
    let pooled_examples2 = OrPool2x2::<u16>::new_from_split(&l2_examples).vec_apply(&l2_examples);

    let mut model = Layer::<
        [[u16; 8]; 8],
        Patch3x3NotchedConv<u16, u128, [u128; 16], u16>,
        [[u16; 8]; 8],
        Layer<
            [[u16; 8]; 8],
            OrPool2x2<u16>,
            [[u16; 4]; 4],
            Layer<[[u16; 4]; 4], ExtractPatchesNotched3x3<u128>, u128, [u128; 10]>,
        >,
    >::new_from_split(&pooled_examples2);
    //model.data.map_fn = L2;

    println!("starting optimize");
    let start = PreciseTime::now();
    for i in 0..3 {
        let acc = model.optimize(&pooled_examples2, MOD);
        println!(
            "mod: {}, acc: {:?}%, time: {}",
            MOD,
            acc * 100f64,
            start.to(PreciseTime::now())
        );
    }
    println!("{:?}", model.data.map_fn);
}

// 1000
// mod: 15, acc: 19.29795918367347%, time: PT44.177613621S
// mod: 10, acc: 17.87704081632653%, time: PT50.994113824S
// mod: 10, acc: 17.87704081632653%, time: PT52.008456309S

// 10000
// mod: 10, acc: 20.36387755102041%, time: PT532.975591321S
// mod: 19, acc: 19.745714285714286%, time: PT364.981708549S
// mod: 20, acc: 20.51673469387755%, time: PT329.119202843S
// mod: 20, acc: 20.51673469387755%, time: PT338.748518223S
// mod: 21, acc: 19.70841836734694%, time: PT330.213937246S
// mod: 25, acc: 19.856020408163268%, time: PT334.489838723S
// mod: 30, acc: 19.86622448979592%, time: PT269.020753183S

// 32%
