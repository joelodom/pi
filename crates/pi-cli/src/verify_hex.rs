//! `--verify-hex` mode: convert a decimal pi file to hex (one-time,
//! cached on disk), then BBP-spot-check the hex file at deterministic
//! sanity positions followed by an open-ended cryptographically-random
//! sampling loop.
//!
//! Progress is rendered as a stack of multi-progress bars (one per
//! phase), declared up front so pending phases are visible alongside
//! the running and completed ones — matching the look of the compute
//! pipeline.  The conversion sub-phases and the unbounded random loop
//! are shown as spinners; the bounded sanity regions are shown as
//! finite bars that tick once per BBP call.
//!
//! See `--verify-hex` notes in the top-level README.

use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::rngs::OsRng;
use rand::Rng;
use rayon::prelude::*;
use rug::{Assign, Integer};

use pi_core::bbp;

use crate::fmt_thousands;

// =========================================================================
// Top-level entry point
// =========================================================================

pub fn run(
    hex_path: &Path,
    from_decimal: Option<&Path>,
    samples_per_window: usize,
    sanity_samples: usize,
    jobs: Option<usize>,
) -> Result<()> {
    // ---------------------------------------------------------------------
    // Pre-flight: figure out whether to convert.
    // ---------------------------------------------------------------------
    let hex_exists = hex_path.is_file();
    match (hex_exists, from_decimal) {
        (false, None) => bail!(
            "hex file `{}` does not exist and no `--from-decimal` was provided; \
             pass `--from-decimal <FILE>` to create it",
            hex_path.display()
        ),
        (true, Some(dec)) => bail!(
            "hex file `{}` already exists; remove it or omit `--from-decimal {}` \
             to reuse it",
            hex_path.display(),
            dec.display()
        ),
        _ => {}
    }

    // ---------------------------------------------------------------------
    // Plan and declare all phase bars up front so pending work is visible.
    // ---------------------------------------------------------------------
    let multi = MultiProgress::new();
    let conv = if hex_exists {
        None
    } else {
        Some(ConversionBars::create(&multi))
    };
    let sanity = SanityBars::create(&multi, sanity_samples as u64);
    let random = PhaseBar::new_spinner(&multi, "random sampling");

    // ---------------------------------------------------------------------
    // Rayon pool sized per `--jobs`.
    // ---------------------------------------------------------------------
    let pool = match jobs {
        Some(j) => rayon::ThreadPoolBuilder::new()
            .num_threads(j)
            .build()
            .context("building rayon pool")?,
        None => rayon::ThreadPoolBuilder::new()
            .build()
            .context("building rayon pool")?,
    };

    // ---------------------------------------------------------------------
    // SIGINT handling.  The handler flips an AtomicBool; the random loop
    // and parallel-iter closures check it between BBP calls.
    // ---------------------------------------------------------------------
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let i = Arc::clone(&interrupted);
        // `set_handler` is per-process; the binary only ever calls it once.
        let _ = ctrlc::set_handler(move || {
            i.store(true, Ordering::SeqCst);
        });
    }

    // ---------------------------------------------------------------------
    // Convert if needed.
    // ---------------------------------------------------------------------
    if let (false, Some(dec)) = (hex_exists, from_decimal) {
        let conv = conv.as_ref().expect("conversion bars present when converting");
        convert_decimal_to_hex(dec, hex_path, conv)?;
    } else {
        // `multi.suspend` runs the closure with bars paused (so the
        // message lands above them in tty mode), and is a plain no-op
        // wrapper around the closure in non-tty mode.
        multi.suspend(|| {
            eprintln!("reusing existing hex file: {}", hex_path.display());
        });
    }

    // ---------------------------------------------------------------------
    // Open the hex file and figure out content extents.
    // ---------------------------------------------------------------------
    let file = File::open(hex_path)
        .with_context(|| format!("opening `{}`", hex_path.display()))?;
    let (data_offset, n_hex_digits) = scan_hex_file(&file)?;
    multi.suspend(|| {
        eprintln!(
            "hex file: {} ({} hex digits past \"3.\"), {} rayon thread(s)",
            hex_path.display(),
            fmt_thousands(n_hex_digits),
            pool.current_num_threads()
        );
    });

    // ---------------------------------------------------------------------
    // Sanity sweep (bounded, fails fast on mismatch).
    // ---------------------------------------------------------------------
    let sanity_completed = pool.install(|| {
        run_sanity_phase(
            &file,
            data_offset,
            n_hex_digits,
            sanity_samples,
            &sanity,
            &interrupted,
        )
    })?;
    if !sanity_completed {
        multi.suspend(|| eprintln!("interrupted during sanity phase."));
        return Ok(());
    }

    // ---------------------------------------------------------------------
    // Random sampling loop (unbounded until SIGINT or mismatch).
    // ---------------------------------------------------------------------
    pool.install(|| {
        run_random_phase(
            &file,
            data_offset,
            n_hex_digits,
            samples_per_window,
            &random,
            &interrupted,
        )
    })
}

// =========================================================================
// Phase bar machinery
// =========================================================================

/// One progress row in the verify-hex pipeline.  Either a finite bar
/// (sanity regions) or a spinner (conversion sub-phases, random loop).
struct PhaseBar {
    bar: ProgressBar,
    is_spinner: bool,
}

impl PhaseBar {
    fn new_finite(multi: &MultiProgress, prefix: &str, total: u64) -> Self {
        let bar = multi.add(ProgressBar::new(total));
        bar.set_prefix(prefix.to_string());
        bar.set_style(pending_bar_style());
        bar.set_message("(pending)");
        bar.tick(); // initial render
        Self { bar, is_spinner: false }
    }

    fn new_spinner(multi: &MultiProgress, prefix: &str) -> Self {
        let bar = multi.add(ProgressBar::new_spinner());
        bar.set_prefix(prefix.to_string());
        bar.set_style(pending_spinner_style());
        bar.tick();
        Self { bar, is_spinner: true }
    }

    fn activate(&self) {
        self.bar.set_position(0);
        self.bar.reset_elapsed();
        self.bar.reset_eta();
        if self.is_spinner {
            self.bar.set_style(active_spinner_style());
            self.bar.enable_steady_tick(Duration::from_millis(100));
        } else {
            self.bar.set_style(active_bar_style());
            self.bar.set_message("");
        }
    }

    /// Thread-safe increment.  `indicatif::ProgressBar::inc` uses an
    /// internal mutex, so rayon workers can all call this concurrently.
    fn inc(&self, n: u64) {
        self.bar.inc(n);
    }

    fn complete(&self) {
        if self.is_spinner {
            self.bar.disable_steady_tick();
            // Make the spinner render as a "full" green bar at finish,
            // taking the final position as the length.  This gives a
            // consistent done state visually with the finite bars.
            let pos = self.bar.position();
            self.bar.set_length(pos.max(1));
        }
        self.bar.set_style(done_style());
        self.bar.finish();
    }
}

fn pending_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:<24} [{bar:30}] {human_pos:>13}/{human_len:<13} {msg}",
    )
    .unwrap()
    .progress_chars("##-")
}

fn pending_spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix:<24} (pending)").unwrap()
}

fn active_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:<24} [{bar:30.cyan/blue}] {human_pos:>13}/{human_len:<13} eta {eta}",
    )
    .unwrap()
    .progress_chars("##-")
}

fn active_spinner_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:<24} {spinner:.cyan}  {human_pos} so far | elapsed {elapsed}",
    )
    .unwrap()
}

fn done_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:<24} [{bar:30.green}] {human_pos:>13}/{human_len:<13} done in {elapsed}",
    )
    .unwrap()
    .progress_chars("##-")
}

/// Spinner bars for the four conversion sub-phases.  Created in
/// declaration order so the user sees the full plan up front.
struct ConversionBars {
    parse: PhaseBar,
    scale: PhaseBar,
    muldiv: PhaseBar,
    write: PhaseBar,
}

impl ConversionBars {
    fn create(multi: &MultiProgress) -> Self {
        Self {
            parse: PhaseBar::new_spinner(multi, "convert: parse decimal"),
            scale: PhaseBar::new_spinner(multi, "convert: build scales"),
            muldiv: PhaseBar::new_spinner(multi, "convert: mul + div"),
            write: PhaseBar::new_spinner(multi, "convert: write hex"),
        }
    }
}

/// Finite bars for the three sanity regions.  Each ticks once per BBP
/// call (which covers 8 consecutive hex digits).
struct SanityBars {
    first: PhaseBar,
    middle: PhaseBar,
    last: PhaseBar,
}

impl SanityBars {
    fn create(multi: &MultiProgress, samples_per_region: u64) -> Self {
        Self {
            first: PhaseBar::new_finite(multi, "sanity: first 1M", samples_per_region),
            middle: PhaseBar::new_finite(multi, "sanity: middle 1M", samples_per_region),
            last: PhaseBar::new_finite(multi, "sanity: last 1M", samples_per_region),
        }
    }
}

// =========================================================================
// Conversion (decimal -> hex), driving the four conversion bars
// =========================================================================

fn convert_decimal_to_hex(
    decimal_path: &Path,
    hex_path: &Path,
    bars: &ConversionBars,
) -> Result<()> {
    // --- Phase 1: read + parse to Integer ---------------------------------
    bars.parse.activate();
    let content = fs::read_to_string(decimal_path)
        .with_context(|| format!("reading `{}`", decimal_path.display()))?;
    let trimmed = content.trim();
    if !trimmed.starts_with("3.") {
        bail!("`{}` doesn't start with \"3.\"", decimal_path.display());
    }
    let frac_str = &trimmed[2..];
    if !frac_str.chars().all(|c| c.is_ascii_digit()) {
        bail!(
            "`{}` has non-digit characters after \"3.\"",
            decimal_path.display()
        );
    }
    // The numerator is `pi · 10^(D-1)` as an integer: "3" followed by the
    // fractional decimal digits.
    let digit_string = format!("3{frac_str}");
    let d_minus_1: u32 = frac_str
        .len()
        .try_into()
        .context("fractional decimal digit count exceeds u32::MAX")?;
    let mut numerator = Integer::new();
    numerator.assign(Integer::parse_radix(&digit_string, 10)?);
    drop(content);
    drop(digit_string);
    bars.parse.inc(1);
    bars.parse.complete();

    // --- Phase 2: build 16^H and 10^(D-1) ---------------------------------
    bars.scale.activate();
    // H = floor(D_minus_1 · log_16(10)) hex digits of precision.
    // log_16(10) = log(10) / log(16) = log_2(10) / 4 ≈ 0.8304.
    let log_16_10 = (10.0_f64.log2()) / 4.0;
    let h: u32 = ((d_minus_1 as f64) * log_16_10) as u32;
    let scale_hex = Integer::from(Integer::u_pow_u(16, h));
    let scale_dec = Integer::from(Integer::u_pow_u(10, d_minus_1));
    bars.scale.inc(1);
    bars.scale.complete();

    // --- Phase 3: numerator · 16^H / 10^(D-1) -----------------------------
    bars.muldiv.activate();
    let product = numerator * scale_hex;
    let m = product / scale_dec;
    bars.muldiv.inc(1);
    bars.muldiv.complete();

    // --- Phase 4: write hex output atomically -----------------------------
    bars.write.activate();
    let hex_str = m.to_string_radix(16);
    if !hex_str.starts_with('3') {
        bail!(
            "internal: hex conversion didn't produce a leading '3' (got `{}…`)",
            &hex_str[..hex_str.len().min(16)]
        );
    }
    let tmp_path = hex_path.with_extension("tmp");
    {
        let mut out = File::create(&tmp_path)
            .with_context(|| format!("creating `{}`", tmp_path.display()))?;
        out.write_all(b"3.")?;
        out.write_all(&hex_str.as_bytes()[1..])?;
        out.write_all(b"\n")?;
        out.sync_all()?;
    }
    fs::rename(&tmp_path, hex_path).with_context(|| {
        format!(
            "renaming `{}` -> `{}`",
            tmp_path.display(),
            hex_path.display()
        )
    })?;
    bars.write.inc(1);
    bars.write.complete();
    Ok(())
}

// =========================================================================
// File scanning
// =========================================================================

fn scan_hex_file(file: &File) -> Result<(u64, u64)> {
    let total_len = file.metadata()?.len();
    if total_len < 2 {
        bail!("hex file is too short");
    }

    // Header
    let mut header = [0_u8; 2];
    read_exact_at(file, &mut header, 0)?;
    if &header != b"3." {
        bail!("hex file doesn't start with \"3.\"");
    }
    let data_offset = 2;

    // Strip trailing whitespace by scanning the tail.
    let tail_size = 4096_u64.min(total_len);
    let tail_start = total_len - tail_size;
    let mut tail = vec![0_u8; tail_size as usize];
    read_exact_at(file, &mut tail, tail_start)?;
    let mut content_end = total_len;
    while content_end > data_offset {
        let idx = (content_end - 1 - tail_start) as usize;
        if matches!(tail[idx], b' ' | b'\t' | b'\n' | b'\r') {
            content_end -= 1;
        } else {
            break;
        }
    }

    let n_hex_digits = content_end - data_offset;
    Ok((data_offset, n_hex_digits))
}

// =========================================================================
// Sanity sweep
// =========================================================================

/// Returns `Ok(true)` on completion, `Ok(false)` on interrupt, `Err` on
/// mismatch.
fn run_sanity_phase(
    file: &File,
    data_offset: u64,
    n_hex_digits: u64,
    samples_per_region: usize,
    bars: &SanityBars,
    interrupted: &AtomicBool,
) -> Result<bool> {
    let one_m = 1_000_000_u64.min(n_hex_digits);
    let half = n_hex_digits / 2;

    // (label, range, bar) for each region.
    let regions: [(std::ops::Range<u64>, &PhaseBar); 3] = [
        (0..one_m, &bars.first),
        (
            half..(half + one_m).min(n_hex_digits),
            &bars.middle,
        ),
        (
            n_hex_digits.saturating_sub(one_m)..n_hex_digits,
            &bars.last,
        ),
    ];

    for (range, bar) in regions {
        if interrupted.load(Ordering::SeqCst) {
            return Ok(false);
        }
        bar.activate();
        let range_len = range.end - range.start;
        if range_len < 8 {
            // Region too small to fit an 8-digit BBP window; mark done.
            bar.complete();
            continue;
        }
        let max_start = range.end - 8;
        let positions: Vec<u64> = (0..samples_per_region)
            .map(|i| {
                let span = max_start - range.start;
                range.start
                    + (span * i as u64)
                        / (samples_per_region.saturating_sub(1).max(1) as u64)
            })
            .collect();
        check_positions(file, data_offset, &positions, bar, interrupted)?;
        if interrupted.load(Ordering::SeqCst) {
            return Ok(false);
        }
        bar.complete();
    }
    Ok(true)
}

// =========================================================================
// Random sampling
// =========================================================================

const WINDOW_BYTES: u64 = 1_000_000;

fn run_random_phase(
    file: &File,
    data_offset: u64,
    n_hex_digits: u64,
    samples_per_window: usize,
    bar: &PhaseBar,
    interrupted: &AtomicBool,
) -> Result<()> {
    if n_hex_digits < 8 {
        bail!("hex file has fewer than 8 hex digits; nothing to sample");
    }
    bar.activate();

    while !interrupted.load(Ordering::SeqCst) {
        let mut rng = OsRng;
        // Window start uniform in [0, n_hex_digits - 8] inclusive.
        let max_window_start = n_hex_digits - 8;
        let p = rng.gen_range(0..=max_window_start);
        let window_end = (p + WINDOW_BYTES).min(n_hex_digits);
        let window_span = window_end - p;
        let max_sample_offset = window_span.saturating_sub(8);
        let n_samples = if max_sample_offset == 0 {
            1
        } else {
            samples_per_window
        };
        let positions: Vec<u64> = (0..n_samples)
            .map(|_| {
                if max_sample_offset == 0 {
                    p
                } else {
                    p + rng.gen_range(0..=max_sample_offset)
                }
            })
            .collect();

        check_positions(file, data_offset, &positions, bar, interrupted)?;
    }

    bar.complete();
    Ok(())
}

// =========================================================================
// Parallel check workhorse
// =========================================================================

/// Run BBP on each position in parallel, comparing the 8 hex digits
/// returned against the bytes at the corresponding file offset.  On
/// mismatch, returns the error from the first detected mismatch (rayon
/// `try_for_each` short-circuits).  On interrupt, returns Ok early.
fn check_positions(
    file: &File,
    data_offset: u64,
    positions: &[u64],
    bar: &PhaseBar,
    interrupted: &AtomicBool,
) -> Result<()> {
    positions
        .par_iter()
        .try_for_each(|&pos| -> Result<()> {
            // Soft-stop: if the user has Ctrl-C'd, skip remaining work
            // without doing the BBP call.  Closures already in-flight
            // will still complete their call (~30s at deep n).
            if interrupted.load(Ordering::SeqCst) {
                return Ok(());
            }
            check_position(file, data_offset, pos)?;
            bar.inc(1);
            Ok(())
        })
}

fn check_position(file: &File, data_offset: u64, pos: u64) -> Result<()> {
    let bbp_digits = bbp::hex_digits_at(pos);

    let mut buf = [0_u8; 8];
    read_exact_at(file, &mut buf, data_offset + pos)
        .with_context(|| format!("reading hex file at position {pos}"))?;

    let mut file_digits: u32 = 0;
    for (i, &c) in buf.iter().enumerate() {
        let v = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => bail!(
                "non-hex character 0x{:02x} at file byte position {}",
                c,
                fmt_thousands(data_offset + pos + i as u64)
            ),
        };
        file_digits = (file_digits << 4) | v as u32;
    }

    if file_digits != bbp_digits {
        bail!(
            "MISMATCH at hex position {}:\n  file: 0x{:08x}\n  bbp:  0x{:08x}",
            fmt_thousands(pos),
            file_digits,
            bbp_digits
        );
    }
    Ok(())
}

// =========================================================================
// Portable positional reads
// =========================================================================

/// Read `buf.len()` bytes from `file` at `offset`.  Wraps `pread`-style
/// positional I/O so concurrent reads across rayon workers don't fight
/// over a shared file cursor.
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
            .map_err(|e| anyhow!("read_exact_at failed at offset {offset}: {e}"))
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut remaining = buf;
        let mut off = offset;
        while !remaining.is_empty() {
            let n = file
                .seek_read(remaining, off)
                .map_err(|e| anyhow!("seek_read failed at offset {off}: {e}"))?;
            if n == 0 {
                return Err(anyhow!("unexpected EOF at offset {off}"));
            }
            remaining = &mut remaining[n..];
            off += n as u64;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        // Slow fallback for exotic platforms (not safe for concurrent calls).
        use std::io::{Read, Seek, SeekFrom};
        let mut file = file.try_clone()?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf).map_err(|e| anyhow!("{e}"))
    }
}
