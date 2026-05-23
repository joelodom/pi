//! `pi` — CLI for computing pi and verifying digit files.
//!
//! Three modes, picked by which flags are given:
//!   * Compute (`--digits`, `-o`, ...): drive a [`PiAlgorithm`] to produce
//!     N digits and stream them through a [`DigitSink`].
//!   * Verify (`--verify A B`): byte-by-byte compare two digit files,
//!     ignoring trailing whitespace.  The shorter file's content must
//!     match the matching prefix of the longer file's content.
//!   * Verify-hex (`--verify-hex HEX_FILE [--from-decimal DEC_FILE]`):
//!     spot-check hex digits of pi using the BBP formula as an
//!     independent oracle.  See `verify_hex` module.
//!
//! Running `pi` with no arguments prints the help text.

mod verify_hex;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use pi_core::output::{file_sink, stdout_sink};
use pi_core::progress::NoopProgress;
use pi_core::{AlgorithmKind, DigitSink, Phase, ProgressReporter};

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

    /// Verify the hex digits of a pi file using BBP as an independent
    /// oracle.  Use an existing converted hex file if it exists, or pass
    /// `--from-decimal <FILE>` to create it from a decimal pi file.
    /// Runs until interrupted (Ctrl-C) or a mismatch is detected.
    #[arg(long, value_name = "HEX_FILE")]
    verify_hex: Option<PathBuf>,

    /// Decimal pi file to convert when `--verify-hex` is given and the
    /// hex file doesn't yet exist.  Ignored otherwise.
    #[arg(long, value_name = "DECIMAL_FILE", requires = "verify_hex")]
    from_decimal: Option<PathBuf>,

    /// `--verify-hex`: BBP samples per random window (each BBP call
    /// verifies 8 consecutive hex digits).
    #[arg(long, default_value_t = 100, value_name = "M", requires = "verify_hex")]
    samples_per_window: usize,

    /// `--verify-hex`: BBP samples per sanity region (first/middle/last 1M).
    #[arg(long, default_value_t = 100, value_name = "N", requires = "verify_hex")]
    sanity_samples: usize,

    /// `--verify-hex`: number of rayon worker threads.  Defaults to all
    /// available cores.
    #[arg(long, value_name = "J", requires = "verify_hex")]
    jobs: Option<usize>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum AlgorithmFlag {
    Chudnovsky,
    #[value(name = "gauss-legendre")]
    GaussLegendre,
}

impl From<AlgorithmFlag> for AlgorithmKind {
    fn from(f: AlgorithmFlag) -> Self {
        match f {
            AlgorithmFlag::Chudnovsky => AlgorithmKind::Chudnovsky,
            AlgorithmFlag::GaussLegendre => AlgorithmKind::GaussLegendre,
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

    if let Some(hex_path) = cli.verify_hex.as_deref() {
        return verify_hex::run(
            hex_path,
            cli.from_decimal.as_deref(),
            cli.samples_per_window,
            cli.sanity_samples,
            cli.jobs,
        );
    }

    run_compute(cli)
}

// ---------------------------------------------------------------------------
// Compute mode.
// ---------------------------------------------------------------------------

fn run_compute(cli: Cli) -> Result<()> {
    let algorithm = AlgorithmKind::from(cli.algorithm).build();
    let writing_to_stdout = cli.output == "-";

    // Fail fast if a regular file already exists at the output path, so a
    // typo in `-o` doesn't clobber an expensive previous run.  We only
    // guard against existing regular files — special destinations like
    // `/dev/null` still work.
    if !writing_to_stdout && Path::new(&cli.output).is_file() {
        anyhow::bail!(
            "output file `{}` already exists; refusing to overwrite (delete it or pick a different -o)",
            cli.output
        );
    }

    // Print the plan before opening anything so a typo in --digits or -o
    // can be spotted (and ^C'd) before the slow work starts.
    eprintln!("computing pi:");
    eprintln!("  digits:    {}", fmt_thousands(cli.digits));
    eprintln!("  algorithm: {}", algorithm.name());
    eprintln!(
        "  output:    {}",
        if writing_to_stdout { "stdout" } else { &cli.output }
    );

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

    let start = Instant::now();
    algorithm.compute(cli.digits, &mut *sink, &mut *progress)?;
    let elapsed = start.elapsed();
    eprintln!(
        "done: {} digits in {:.3?}",
        fmt_thousands(cli.digits),
        elapsed
    );

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
                    eprintln!("verify: differ at byte offset {}", fmt_thousands(offset));
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
                    anyhow::bail!("verify failed at offset {}", fmt_thousands(offset));
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
            fmt_thousands(common)
        );
    } else {
        let (short_path, short_len, long_path, long_len) = if a_len < b_len {
            (file_a, a_len, file_b, b_len)
        } else {
            (file_b, b_len, file_a, a_len)
        };
        eprintln!(
            "verify: all {} content bytes of {} match the prefix of {} ({} content bytes total)",
            fmt_thousands(short_len),
            short_path.display(),
            long_path.display(),
            fmt_thousands(long_len)
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

/// Render `n` with `,` as a thousands separator (e.g. `1_234_567` -> `"1,234,567"`).
pub(crate) fn fmt_thousands(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(b as char);
    }
    out
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
/// When the algorithm declares its phases via `set_phases`, one
/// progress bar is created per phase and stacked into a [`MultiProgress`]
/// up front, so the user sees the whole pipeline (pending phases,
/// running phase, completed phases) at a glance.
///
/// If the algorithm doesn't pre-declare phases, bars are added lazily
/// the first time each phase starts — the old single-bar behavior in
/// "scrolling" form.
///
/// The inner loop only updates the bar's position about 100 times per
/// phase regardless of how often `tick` is called, to keep the loop
/// cheap.  Indicatif also rate-limits redraws to ~15 Hz internally.
struct CliProgress {
    multi: MultiProgress,
    bars: Vec<ProgressBar>,
    name_to_idx: HashMap<String, usize>,
    current: Option<usize>,
    counter: u64,
    tick_every: u64,
}

impl CliProgress {
    fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            bars: Vec::new(),
            name_to_idx: HashMap::new(),
            current: None,
            counter: 0,
            tick_every: 1,
        }
    }

    fn pending_style() -> ProgressStyle {
        // Plain (no color) bar with a `{msg}` slot for "(pending)".
        ProgressStyle::with_template(
            "{prefix:<22} [{bar:30}] {human_pos:>13}/{human_len:<13} {msg}",
        )
        .unwrap()
        .progress_chars("##-")
    }

    fn active_style() -> ProgressStyle {
        // Cyan-on-blue running bar with live ETA.
        ProgressStyle::with_template(
            "{prefix:<22} [{bar:30.cyan/blue}] {human_pos:>13}/{human_len:<13} eta {eta}",
        )
        .unwrap()
        .progress_chars("##-")
    }

    fn done_style() -> ProgressStyle {
        // Green completed bar with the total time the phase took.  We
        // reset the elapsed counter at `start_phase`, so `{elapsed}`
        // here is per-phase, not pipeline-cumulative.
        ProgressStyle::with_template(
            "{prefix:<22} [{bar:30.green}] {human_pos:>13}/{human_len:<13} done in {elapsed}",
        )
        .unwrap()
        .progress_chars("##-")
    }

    fn install_bar(&mut self, name: &str, total: u64) -> usize {
        let bar = self.multi.add(ProgressBar::new(total));
        bar.set_prefix(name.to_string());
        self.bars.push(bar);
        let idx = self.bars.len() - 1;
        self.name_to_idx.insert(name.to_string(), idx);
        idx
    }
}

impl ProgressReporter for CliProgress {
    fn set_phases(&mut self, phases: &[Phase]) {
        for p in phases {
            let idx = self.install_bar(p.name, p.total);
            let bar = &self.bars[idx];
            bar.set_style(Self::pending_style());
            bar.set_message("(pending)");
            // Force an initial draw so all phases are visible before the
            // first one starts ticking.
            bar.tick();
        }
    }

    fn start_phase(&mut self, name: &str, total: u64) {
        let idx = match self.name_to_idx.get(name).copied() {
            Some(i) => i,
            None => self.install_bar(name, total),
        };
        let bar = &self.bars[idx];
        bar.set_style(Self::active_style());
        bar.set_length(total);
        bar.set_position(0);
        // Per-phase elapsed/eta timing.  Bars created at `set_phases`
        // time would otherwise report elapsed time relative to pipeline
        // start, not to when their own phase began.
        bar.reset_elapsed();
        bar.reset_eta();
        self.current = Some(idx);
        self.counter = 0;
        self.tick_every = (total / 100).max(1);
    }

    fn tick(&mut self) {
        if let Some(idx) = self.current {
            self.counter += 1;
            if self.counter.is_multiple_of(self.tick_every) {
                self.bars[idx].set_position(self.counter);
            }
        }
    }

    fn end_phase(&mut self) {
        if let Some(idx) = self.current.take() {
            let bar = &self.bars[idx];
            let total = bar.length().unwrap_or(self.counter);
            bar.set_position(total);
            bar.set_style(Self::done_style());
            bar.finish();
        }
    }
}
