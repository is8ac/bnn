use bnn::bench::{PerfResults, PerfTest};
use plotters::prelude::*;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args();
    args.next();
    let y_scale: u32 = args
        .next()
        .expect("you must pass a scale number")
        .parse()
        .expect("first arg must be a number");

    let input_path: String = args.next().expect("you must pass an input path");
    let output_path: String = args.next().expect("you must pass an output path");

    let mut r = File::open(&Path::new(&input_path)).unwrap();
    let perf_results: PerfResults = serde_json::from_reader(r).unwrap();

    let root = BitMapBackend::new(&output_path, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "word size sum perf, {} {}",
                perf_results.machine_type, perf_results.cpu_arch
            ),
            ("sans-serif", 30).into_font(),
        )
        .margin(30)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(0f32..32f32, 0f32..y_scale as f32)?;

    chart.configure_mesh().draw()?;

    [(3, RED), (4, GREEN), (5, BLUE)]
        .iter()
        .for_each(|(b, color)| {
            chart
                .draw_series(PointSeries::of_element(
                    (0..32).filter_map(|i| {
                        perf_results
                            .tests
                            .iter()
                            .find(|test| {
                                test.algorithm == format!("bitslice-BitArray64x{}_u{}", i, b)
                            })
                            .map(|x| (i, x))
                    }),
                    2,
                    color.clone(),
                    &|(i, test), s, st| {
                        let c = (
                            i as f32,
                            (test.exp_nanos + test.unit_nanos) as f32 / test.n_examples as f32,
                        );
                        EmptyElement::at(c)
                            + Circle::new((0, 0), s, st.filled())
                            + Text::new(
                                format!("64x{}", i),
                                (10, 0),
                                ("sans-serif", 15.0).into_font(),
                            )
                    },
                ))
                .unwrap()
                .label(format!("{} bits", b))
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
        });

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
