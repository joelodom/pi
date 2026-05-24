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
use pi_core::{AlgorithmKind, DigitSink, PerfRecorder, Phase, ProgressReporter};

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

    /// Append-only JSONL file to receive performance instrumentation.
    /// One `run-start` event is emitted at the top, followed by
    /// `phase-start` / `phase-end` markers and periodic `sample`
    /// records (memory, effective CPU cores).  Omit the flag to
    /// disable instrumentation entirely (the recorder becomes a NOP).
    #[arg(long, value_name = "FILE")]
    performance_file: Option<PathBuf>,

    /// Interval between periodic perf samples, in milliseconds.  Only
    /// meaningful when `--performance-file` is set.  Small values
    /// give finer time resolution at the cost of more lines in the
    /// output; the sampler itself is cheap (< 0.1% of one core at
    /// 500 ms).  Defaults to `perf.default_sample_ms` from the loaded
    /// config (laptop default: 500 ms).
    #[arg(long, value_name = "MS")]
    performance_sample_ms: Option<u64>,

    /// Path to a TOML configuration file with per-machine performance
    /// tuning (NTT / Karatsuba thresholds, parallelism breakpoints,
    /// rayon thread pool size, etc.).  When omitted, laptop-class
    /// defaults are used.  See `config/laptop.toml`,
    /// `config/server-128gb.toml`, and `config/server-massive.toml`
    /// for examples and per-knob documentation.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

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
    #[arg(long, default_value_t = 10, value_name = "M", requires = "verify_hex")]
    samples_per_window: usize,

    /// `--verify-hex`: BBP samples per sanity region (first/middle/last 1M).
    #[arg(long, default_value_t = 100, value_name = "N", requires = "verify_hex")]
    sanity_samples: usize,

    /// `--verify-hex`: maximum rayon worker threads (shared across all
    /// parallel verification phases — first/middle/last sanity regions
    /// plus the random loop).  Defaults to all available cores.  Rayon
    /// work-stealing redistributes workers across phases as sanity
    /// phases finish, so the random phase naturally absorbs freed
    /// capacity without manual reallocation.
    #[arg(long, value_name = "J", requires = "verify_hex")]
    max_jobs: Option<usize>,
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

// ---------------------------------------------------------------------------
// Config-file loader.  Parses the optional TOML file into a struct, then
// hands the values off to `bignum::config::apply` and
// `pi_core::config::apply`.  Missing fields fall back to the laptop
// defaults baked into each crate.  Setting `runtime.threads` to a
// non-zero value resizes the rayon global thread pool.
// ---------------------------------------------------------------------------

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    runtime: RuntimeSection,
    bignum: BignumSection,
    chudnovsky: ChudnovskySection,
    decimal_conversion: DecimalSection,
    perf: PerfSection,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RuntimeSection {
    /// Number of rayon worker threads.  Zero = autodetect (one per
    /// logical core).  Non-zero overrides the pool size.
    threads: usize,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct BignumSection {
    karatsuba_threshold: Option<usize>,
    parallel_karatsuba_threshold: Option<usize>,
    ntt_threshold: Option<usize>,
    newton_div_threshold: Option<usize>,
    ntt: NttSection,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct NttSection {
    target_task_size: Option<usize>,
    parallel_pack_threshold: Option<usize>,
    parallel_pointwise_threshold: Option<usize>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ChudnovskySection {
    parallel_split_threshold: Option<u64>,
    sequential_top_threshold: Option<u64>,
    parallel_final_assembly: Option<bool>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DecimalSection {
    to_string_dc_threshold: Option<usize>,
    parallel_to_string_threshold: Option<usize>,
}

#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PerfSection {
    default_sample_ms: Option<u64>,
}

fn load_and_apply_config(path: Option<&Path>) -> Result<()> {
    let file_cfg: FileConfig = match path {
        None => FileConfig::default(),
        Some(p) => {
            let text = std::fs::read_to_string(p)
                .with_context(|| format!("reading config `{}`", p.display()))?;
            toml::from_str(&text)
                .with_context(|| format!("parsing config `{}`", p.display()))?
        }
    };

    // Build the per-crate Config structs starting from each crate's
    // laptop defaults and overlaying any fields the file specifies.
    let mut bn = bignum::config::Config::default();
    let mut pc = pi_core::config::Config::default();

    if let Some(v) = file_cfg.bignum.karatsuba_threshold {
        bn.karatsuba_threshold = v;
    }
    if let Some(v) = file_cfg.bignum.parallel_karatsuba_threshold {
        bn.parallel_karatsuba_threshold = v;
    }
    if let Some(v) = file_cfg.bignum.ntt_threshold {
        bn.ntt_threshold = v;
    }
    if let Some(v) = file_cfg.bignum.newton_div_threshold {
        bn.newton_div_threshold = v;
    }
    if let Some(v) = file_cfg.bignum.ntt.target_task_size {
        bn.ntt.target_task_size = v;
    }
    if let Some(v) = file_cfg.bignum.ntt.parallel_pack_threshold {
        bn.ntt.parallel_pack_threshold = v;
    }
    if let Some(v) = file_cfg.bignum.ntt.parallel_pointwise_threshold {
        bn.ntt.parallel_pointwise_threshold = v;
    }
    if let Some(v) = file_cfg.decimal_conversion.to_string_dc_threshold {
        bn.to_string_dc_threshold = v;
    }
    if let Some(v) = file_cfg.decimal_conversion.parallel_to_string_threshold {
        bn.parallel_to_string_threshold = v;
    }
    if let Some(v) = file_cfg.chudnovsky.parallel_split_threshold {
        pc.chudnovsky.parallel_split_threshold = v;
    }
    if let Some(v) = file_cfg.chudnovsky.sequential_top_threshold {
        pc.chudnovsky.sequential_top_threshold = v;
    }
    if let Some(v) = file_cfg.chudnovsky.parallel_final_assembly {
        pc.chudnovsky.parallel_final_assembly = v;
    }
    if let Some(v) = file_cfg.perf.default_sample_ms {
        pc.perf.default_sample_ms = v;
    }

    bignum::config::apply(&bn);
    pi_core::config::apply(&pc);

    if file_cfg.runtime.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(file_cfg.runtime.threads)
            .build_global()
            .with_context(|| {
                format!(
                    "setting rayon thread pool to {} threads",
                    file_cfg.runtime.threads
                )
            })?;
    }

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load and apply the perf config (or defaults).  This MUST run
    // before any rayon work — if the config requests a specific
    // thread count, we set it once via `ThreadPoolBuilder::build_global`.
    load_and_apply_config(cli.config.as_deref())?;

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
            cli.max_jobs,
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

    let inner_progress: Box<dyn ProgressReporter> = if cli.no_progress {
        Box::new(NoopProgress)
    } else {
        Box::new(CliProgress::new())
    };

    // Open the perf recorder (no-op when `--performance-file` is unset).
    // Hold the sampler guard for the lifetime of the run so it stops
    // and joins on drop.
    let recorder = match cli.performance_file.as_deref() {
        Some(path) => PerfRecorder::open(path, cli.digits, algorithm.name())
            .with_context(|| {
                format!("opening performance file `{}`", path.display())
            })?,
        None => PerfRecorder::disabled(),
    };
    // CLI flag overrides; otherwise use the perf default from config.
    let sample_ms = cli
        .performance_sample_ms
        .unwrap_or_else(pi_core::config::perf_default_sample_ms);
    let _sampler = recorder.start_sampler(sample_ms);

    let mut progress: Box<dyn ProgressReporter> =
        Box::new(PerfWrappedProgress::new(inner_progress, recorder.clone()));

    let start = Instant::now();
    algorithm.compute(cli.digits, &mut *sink, &mut *progress)?;
    let elapsed = start.elapsed();
    recorder.run_end();
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

// ---------------------------------------------------------------------------
// PerfWrappedProgress
// ---------------------------------------------------------------------------
//
// Wraps an inner ProgressReporter (CliProgress or NoopProgress) and a
// PerfRecorder.  Forwards every reporter call to both, so the
// algorithm only sees the ProgressReporter interface and remains
// oblivious to perf instrumentation.  When the recorder is disabled,
// the forwarding calls are no-ops in the recorder's hot path.

struct PerfWrappedProgress {
    inner: Box<dyn ProgressReporter>,
    recorder: PerfRecorder,
}

impl PerfWrappedProgress {
    fn new(inner: Box<dyn ProgressReporter>, recorder: PerfRecorder) -> Self {
        Self { inner, recorder }
    }
}

impl ProgressReporter for PerfWrappedProgress {
    fn set_phases(&mut self, phases: &[Phase]) {
        self.inner.set_phases(phases);
    }

    fn start_phase(&mut self, name: &str, total: u64) {
        self.inner.start_phase(name, total);
        self.recorder.phase_start(name);
    }

    fn tick(&mut self) {
        self.inner.tick();
    }

    fn end_phase(&mut self) {
        self.recorder.phase_end();
        self.inner.end_phase();
    }
}
