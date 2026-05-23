//! `pi` — CLI for computing pi to a chosen number of decimal digits.
//!
//! The CLI's job is to parse arguments, wire up a [`DigitSink`] and a
//! [`ProgressReporter`], hand them to a [`PiAlgorithm`], and time the run.
//! All compute logic lives in `pi-core`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};

use pi_core::output::{file_sink, stdout_sink};
use pi_core::progress::NoopProgress;
use pi_core::{AlgorithmKind, DigitSink, ProgressReporter};

#[derive(Parser, Debug)]
#[command(
    name = "pi",
    about = "Compute pi to a chosen number of decimal digits.",
    version
)]
struct Cli {
    /// Number of decimal digits to compute (counting the leading "3").
    #[arg(short = 'd', long, default_value_t = 1_000_000)]
    digits: u64,

    /// File to write digits to.  Use `-` for stdout.
    #[arg(short = 'o', long, default_value = "-")]
    output: String,

    /// Algorithm to use.
    #[arg(long, value_enum, default_value_t = AlgorithmFlag::Chudnovsky)]
    algorithm: AlgorithmFlag,

    /// Suppress the progress reporter.  Progress is always written to
    /// stderr, so this is only useful when stderr is a terminal you'd
    /// rather keep clean.
    #[arg(long)]
    no_progress: bool,

    /// After computing, compare the first 100 digits against a hardcoded
    /// reference value and report whether it matches.  Requires `-o
    /// <FILE>` (we read the output back to verify).
    #[arg(long)]
    verify: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AlgorithmFlag {
    Chudnovsky,
}

impl From<AlgorithmFlag> for AlgorithmKind {
    fn from(f: AlgorithmFlag) -> Self {
        match f {
            AlgorithmFlag::Chudnovsky => AlgorithmKind::Chudnovsky,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let algorithm = AlgorithmKind::from(cli.algorithm).build();
    let writing_to_stdout = cli.output == "-";

    let mut sink: Box<dyn DigitSink> = if writing_to_stdout {
        Box::new(stdout_sink())
    } else {
        Box::new(
            file_sink(&cli.output)
                .with_context(|| format!("creating output file `{}`", cli.output))?,
        )
    };

    let mut progress: Box<dyn ProgressReporter> = if cli.no_progress {
        Box::new(NoopProgress)
    } else {
        Box::new(CliProgress::new())
    };

    eprintln!(
        "computing {} digits of pi via {}",
        cli.digits,
        algorithm.name()
    );
    let start = Instant::now();
    algorithm.compute(cli.digits, &mut *sink, &mut *progress)?;
    let elapsed = start.elapsed();
    eprintln!("done in {:.3?}", elapsed);

    if cli.verify {
        if writing_to_stdout {
            eprintln!("--verify requires writing to a file; rerun with -o <FILE>");
        } else {
            verify_known_prefix(&PathBuf::from(&cli.output), cli.digits)?;
        }
    }

    Ok(())
}

// First 100 digits of pi, counting the leading `3` — i.e. one integer
// digit and 99 fractional digits.  Used by `--verify` to sanity-check the
// computed output.
const FIRST_100: &str = "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067";

fn verify_known_prefix(path: &Path, digits_requested: u64) -> Result<()> {
    use std::fs::File;
    use std::io::{BufReader, Read};

    // We can check up to 100 digits.  An n-digit output occupies n + 1 chars
    // in the file (one for the decimal point) when n >= 2, or just 1 char
    // when n = 1.
    let n = (digits_requested as usize).min(100);
    let want_chars = if n <= 1 { n } else { n + 1 };
    let expected = &FIRST_100[..want_chars];

    let mut reader = BufReader::new(
        File::open(path).with_context(|| format!("opening `{}` for verification", path.display()))?,
    );
    let mut buf = vec![0u8; want_chars];
    reader.read_exact(&mut buf)?;
    let got = std::str::from_utf8(&buf)?;

    if got == expected {
        eprintln!("verify: first {} digits match the known reference", n);
        Ok(())
    } else {
        eprintln!("verify: MISMATCH in first {} chars", want_chars);
        eprintln!("  expected: {}", expected);
        eprintln!("  got     : {}", got);
        anyhow::bail!("verification failed");
    }
}

/// `indicatif`-backed [`ProgressReporter`].
///
/// We only update the bar's position about 100 times per phase regardless of
/// how often `tick` is called, to keep the inner loop cheap.  Indicatif also
/// rate-limits redraws to ~15 Hz, so this is mostly belt-and-suspenders.
struct CliProgress {
    bar: Option<ProgressBar>,
    counter: u64,
    tick_every: u64,
}

impl CliProgress {
    fn new() -> Self {
        Self { bar: None, counter: 0, tick_every: 1 }
    }
}

impl ProgressReporter for CliProgress {
    fn start_phase(&mut self, name: &str, total: u64) {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template(
                "{prefix:>20} [{bar:30.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        bar.set_prefix(name.to_string());
        self.tick_every = (total / 100).max(1);
        self.counter = 0;
        self.bar = Some(bar);
    }

    fn tick(&mut self) {
        self.counter += 1;
        if let Some(bar) = self.bar.as_ref() {
            if self.counter.is_multiple_of(self.tick_every) {
                bar.set_position(self.counter);
            }
        }
    }

    fn end_phase(&mut self) {
        if let Some(bar) = self.bar.take() {
            bar.set_position(self.counter);
            bar.finish_and_clear();
        }
    }
}
