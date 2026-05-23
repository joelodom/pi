//! `pi` — CLI for computing pi and verifying digit files.
//!
//! Two modes, picked by which flags are given:
//!   * Compute (`--digits`, `-o`, ...): drive a [`PiAlgorithm`] to produce
//!     N digits and stream them through a [`DigitSink`].
//!   * Verify (`--verify A B`): byte-by-byte compare two digit files,
//!     ignoring trailing whitespace.  The shorter file's content must
//!     match the matching prefix of the longer file's content.
//!
//! Running `pi` with no arguments prints the help text.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
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
    about = "Compute pi to N decimal digits, or verify a digit file against a reference.",
    version,
    arg_required_else_help = true
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

    /// Suppress the progress reporter (progress goes to stderr).
    #[arg(long)]
    no_progress: bool,

    /// Verify two digit files against each other instead of computing.
    /// Trims trailing whitespace on both, then byte-by-byte compares the
    /// shorter file's content against the matching prefix of the longer
    /// file's content.  Skips computation entirely.
    #[arg(long, value_names = ["FILE_A", "FILE_B"], num_args = 2)]
    verify: Option<Vec<PathBuf>>,
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

    if let Some(files) = cli.verify.as_deref() {
        // clap's `num_args = 2` guarantees the slice has exactly two
        // entries, but assert defensively so the indexing never panics
        // silently if that constraint ever changes.
        assert_eq!(files.len(), 2, "--verify takes exactly two file arguments");
        return run_verify(&files[0], &files[1]);
    }

    run_compute(cli)
}

// ---------------------------------------------------------------------------
// Compute mode.
// ---------------------------------------------------------------------------

fn run_compute(cli: Cli) -> Result<()> {
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

    if !writing_to_stdout {
        eprintln!("output written to {}", cli.output);
        eprintln!(
            "to verify byte-by-byte against a reference file:\n  pi --verify {} <REFERENCE>",
            cli.output
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Verify mode.
// ---------------------------------------------------------------------------

/// Byte-by-byte compare two digit files, ignoring trailing whitespace.
///
/// Trims trailing `' '`, `'\t'`, `'\n'`, `'\r'` from both files, then
/// compares the shorter content against the matching prefix of the
/// longer content.  Returns `Ok(())` on match; bails with the first
/// differing byte offset on mismatch.
fn run_verify(file_a: &Path, file_b: &Path) -> Result<()> {
    let a_len = file_content_length(file_a)
        .with_context(|| format!("scanning `{}`", file_a.display()))?;
    let b_len = file_content_length(file_b)
        .with_context(|| format!("scanning `{}`", file_b.display()))?;
    let common = a_len.min(b_len);

    let mut a = BufReader::with_capacity(1 << 20, File::open(file_a)?);
    let mut b = BufReader::with_capacity(1 << 20, File::open(file_b)?);

    const CHUNK: usize = 64 * 1024;
    let mut buf_a = vec![0u8; CHUNK];
    let mut buf_b = vec![0u8; CHUNK];
    let mut pos: u64 = 0;

    while pos < common {
        let to_read = ((common - pos).min(CHUNK as u64)) as usize;
        a.read_exact(&mut buf_a[..to_read])?;
        b.read_exact(&mut buf_b[..to_read])?;

        if buf_a[..to_read] != buf_b[..to_read] {
            for i in 0..to_read {
                if buf_a[i] != buf_b[i] {
                    let offset = pos + i as u64;
                    eprintln!("verify: differ at byte offset {}", offset);
                    eprintln!(
                        "  {}: 0x{:02x} {}",
                        file_a.display(),
                        buf_a[i],
                        describe_byte(buf_a[i])
                    );
                    eprintln!(
                        "  {}: 0x{:02x} {}",
                        file_b.display(),
                        buf_b[i],
                        describe_byte(buf_b[i])
                    );
                    anyhow::bail!("verify failed at offset {}", offset);
                }
            }
        }
        pos += to_read as u64;
    }

    if a_len == b_len {
        eprintln!(
            "verify: {} and {} are identical ({} content bytes)",
            file_a.display(),
            file_b.display(),
            common
        );
    } else {
        let (short_path, short_len, long_path, long_len) = if a_len < b_len {
            (file_a, a_len, file_b, b_len)
        } else {
            (file_b, b_len, file_a, a_len)
        };
        eprintln!(
            "verify: all {} content bytes of {} match the prefix of {} ({} content bytes total)",
            short_len,
            short_path.display(),
            long_path.display(),
            long_len
        );
    }
    Ok(())
}

/// Length of `path` excluding any trailing run of whitespace (` `, `\t`,
/// `\n`, `\r`).
///
/// Scans backwards in 4 KiB chunks so the whole file never has to fit in
/// memory — important when verifying against the 100M-digit reference.
fn file_content_length(path: &Path) -> Result<u64> {
    let mut f = File::open(path).with_context(|| format!("opening `{}`", path.display()))?;
    let total_len = f.metadata()?.len();
    if total_len == 0 {
        return Ok(0);
    }

    const CHUNK: u64 = 4096;
    let mut buf = vec![0u8; CHUNK as usize];
    let mut end = total_len;

    while end > 0 {
        let read_size = CHUNK.min(end);
        let start = end - read_size;
        f.seek(SeekFrom::Start(start))?;
        f.read_exact(&mut buf[..read_size as usize])?;

        for i in (0..read_size as usize).rev() {
            let c = buf[i];
            if !matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                return Ok(start + i as u64 + 1);
            }
        }

        end = start;
    }

    Ok(0)
}

fn describe_byte(b: u8) -> String {
    match b {
        b' ' => "(space)".to_string(),
        b'\t' => "(tab)".to_string(),
        b'\n' => "(LF)".to_string(),
        b'\r' => "(CR)".to_string(),
        c if c.is_ascii_graphic() => format!("('{}')", c as char),
        c => format!("(non-printable {})", c),
    }
}

// ---------------------------------------------------------------------------
// Progress reporting.
// ---------------------------------------------------------------------------

/// `indicatif`-backed [`ProgressReporter`].
///
/// We only update the bar's position about 100 times per phase regardless
/// of how often `tick` is called, to keep the inner loop cheap.  Indicatif
/// also rate-limits redraws to ~15 Hz, so this is mostly belt-and-braces.
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
