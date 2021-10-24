use bnn::bench::{PerfResults, PerfTest};
use plotters::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut r = File::open(&Path::new("perf_results.json")).unwrap();
    let perf_results: PerfResults = serde_json::from_reader(r).unwrap();

    let root = BitMapBackend::new("word_perf_plot.png", (1024, 1024)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(format!("word size sum perf, {}", perf_results.machine_type), ("sans-serif", 50).into_font())
        .margin(30)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(0f32..32f32, 0f32..3000f32)?;

    chart.configure_mesh().draw()?;

    (0..32).for_each(|i| {
        (0..8).for_each(|b| {
            chart
                .draw_series(PointSeries::of_element(
                    perf_results.tests.iter().filter(|test| test.algorithm == format!("bitslice-BitArray64x{}_u{}", i, b)),
                    2,
                    &RED,
                    &|test, s, st| {
                        let c = (i as f32, (test.exp_nanos + test.unit_nanos) as f32 / test.n_examples as f32);
                        EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()) + Text::new(format!("64x{}_u{}", i, b), (10, 0), ("sans-serif", 15.0).into_font())
                    },
                ))
                .unwrap();
        });
    });

    chart.configure_series_labels().background_style(&WHITE.mix(0.8)).border_style(&BLACK).draw()?;

    Ok(())
}
