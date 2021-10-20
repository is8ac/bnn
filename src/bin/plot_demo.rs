use bnn::bench::{PerfResults, PerfTest};
use plotters::prelude::*;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut r = File::open(&Path::new("perf_results_zen2.json")).unwrap();
    let perf_results: PerfResults = serde_json::from_reader(r).unwrap();

    let root = BitMapBackend::new("demoplot.png", (1024, 1024)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "unit grad perf, 48 thread, zen2",
            ("sans-serif", 50).into_font(),
        )
        .margin(30)
        .x_label_area_size(20)
        .y_label_area_size(20)
        .build_cartesian_2d(0f32..4096f32, 0f32..3200f32)?;

    chart.configure_mesh().draw()?;

    /*
    chart.draw_series(PointSeries::of_element(
        perf_results.tests.iter().filter(|test| test.algorithm == "popcnt"),
        2,
        &BLACK,
        &|test, s, st| {
            let c = (test.chunk_size as f32, test.exp_nanos as f32 / test.n_examples as f32);
            return EmptyElement::at(c)
                + Circle::new((0, 0), s, st.filled())
                + Text::new(format!("{} {}", test.chunk_size, test.n_examples), (10, 0), ("sans-serif", 15.0).into_font());
        },
    ))?;
    */

    [
        ("bitslice-BitArray64x1", RGBColor(0, 0, 0)),
        ("bitslice-BitArray64x2", RGBColor(255, 0, 0)),
        ("bitslice-BitArray64x4", RGBColor(0, 255, 0)),
        ("bitslice-BitArray64x8", RGBColor(255, 255, 0)),
        ("bitslice-BitArray64x16", RGBColor(0, 0, 255)),
        ("bitslice-BitArray64x32", RGBColor(255, 128, 0)),
        ("bitslice-BitArray64x64", RGBColor(128, 0, 128)),
        ("bitslice-u64", RGBColor(128, 128, 128)),
        ("bitslice-u128", RGBColor(128, 128, 0)),
        ("bitslice-avx2", RGBColor(255, 0, 255)),
        ("bitslice-avx512", RGBColor(0, 128, 128)),
        ("bitslice-neon", RGBColor(0, 255, 255)),
    ]
    .iter()
    .cloned()
    .for_each(|(name, color)| {
        chart
            .draw_series(PointSeries::of_element(
                perf_results
                    .tests
                    .iter()
                    .filter(|test| test.algorithm == *name)
                    .filter(|test| test.n_threads == 48),
                2,
                color,
                &|test, s, st| {
                    let c = (
                        test.chunk_size as f32,
                        test.unit_nanos as f32 / test.n_examples as f32,
                    );
                    EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
                },
            ))
            .unwrap()
            .label(name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    });

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
