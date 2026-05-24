//! Phase-level timing benchmark — runs chudnovsky at multiple sizes and
//! prints per-phase wall times.  Used for performance analysis only; not
//! part of the shipped CLI.

use std::io::Write;
use std::time::Instant;

use pi_core::{
    algorithm::{chudnovsky::Chudnovsky, PiAlgorithm},
    DigitSink, Phase, ProgressReporter,
};

struct LogProgress {
    current: Option<(String, Instant)>,
}

impl LogProgress {
    fn new() -> Self {
        Self { current: None }
    }
}

impl ProgressReporter for LogProgress {
    fn set_phases(&mut self, _: &[Phase]) {}
    fn start_phase(&mut self, name: &str, _: u64) {
        self.current = Some((name.to_string(), Instant::now()));
    }
    fn tick(&mut self) {}
    fn end_phase(&mut self) {
        if let Some((name, start)) = self.current.take() {
            let elapsed = start.elapsed();
            eprintln!("    phase {:<20} {:>10.3?}", name, elapsed);
        }
    }
}

struct NullSink;
impl DigitSink for NullSink {
    fn write_integer_part(&mut self, _: &str) -> Result<(), std::io::Error> {
        Ok(())
    }
    fn write_fractional_digits(&mut self, _: &str) -> Result<(), std::io::Error> {
        Ok(())
    }
    fn finish(&mut self) -> Result<(), std::io::Error> {
        Ok(())
    }
}

fn main() {
    let sizes: Vec<u64> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("usage: phase_bench <digits1> <digits2> ..."))
        .collect();
    let sizes = if sizes.is_empty() {
        vec![100_000, 500_000, 1_000_000, 5_000_000]
    } else {
        sizes
    };

    for d in sizes {
        eprintln!("=== {} digits ===", d);
        let start = Instant::now();
        let mut sink = NullSink;
        let mut prog = LogProgress::new();
        Chudnovsky.compute(d, &mut sink, &mut prog).unwrap();
        eprintln!("    total                {:>10.3?}", start.elapsed());
        let _ = std::io::stderr().flush();
    }
}
