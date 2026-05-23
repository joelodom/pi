//! `--verify-hex` mode: convert a decimal pi file to hex (one-time,
//! cached on disk), then BBP-spot-check the hex file.
//!
//! After conversion (if needed), the three sanity regions (first /
//! middle / last 1M) and the unbounded random-sampling loop all run
//! **concurrently** on a shared rayon pool.  Each phase has its own
//! progress bar so the user sees all four in flight side-by-side.  As
//! sanity phases finish, their CPU naturally redistributes to the
//! remaining phases through rayon's work-stealing.
//!
//! Coordination across phases is a single `AtomicBool` "stop" flag.
//! It is set by the SIGINT handler (clean shutdown, exit 0) or by any
//! phase that detects a mismatch (error captured into a shared slot,
//! exit 1).  The flag is threaded all the way into
//! `bbp::hex_digits_at_interruptible` so an in-flight BBP call at deep
//! n bails within a few milliseconds rather than the ~minutes one call
//! would otherwise take.

use std::fs;
use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
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
    max_jobs: Option<usize>,
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
    // Rayon pool sized per `--max-jobs`.  All four verification phases
    // share this pool; rayon work-stealing redistributes workers as
    // sanity phases finish.
    // ---------------------------------------------------------------------
    let pool = match max_jobs {
        Some(j) => rayon::ThreadPoolBuilder::new()
            .num_threads(j)
            .build()
            .context("building rayon pool")?,
        None => rayon::ThreadPoolBuilder::new()
            .build()
            .context("building rayon pool")?,
    };

    // ---------------------------------------------------------------------
    // Coordination: a single `stop` flag (set by SIGINT or by a mismatch),
    // and a single slot for the first error.  Both phases and BBP itself
    // poll `stop`.
    // ---------------------------------------------------------------------
    let stop = Arc::new(AtomicBool::new(false));
    let first_error: Arc<Mutex<Option<anyhow::Error>>> = Arc::new(Mutex::new(None));

    {
        let s = Arc::clone(&stop);
        let _ = ctrlc::set_handler(move || {
            s.store(true, Ordering::SeqCst);
        });
    }

    // ---------------------------------------------------------------------
    // Convert if needed.  (Conversion stays sequential — each step
    // depends on the previous.)
    // ---------------------------------------------------------------------
    if let (false, Some(dec)) = (hex_exists, from_decimal) {
        let conv = conv.as_ref().expect("conversion bars present when converting");
        convert_decimal_to_hex(dec, hex_path, conv)?;
    } else {
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
            "hex file: {} ({} hex digits past \"3.\"), max-jobs {}",
            hex_path.display(),
            fmt_thousands(n_hex_digits),
            pool.current_num_threads()
        );
    });

    // ---------------------------------------------------------------------
    // Build the three sanity ranges and launch all four phases in
    // parallel via `pool.scope`.
    // ---------------------------------------------------------------------
    let range_first = 0..1_000_000_u64.min(n_hex_digits);
    let half = n_hex_digits / 2;
    let range_middle = half..(half + 1_000_000).min(n_hex_digits);
    let range_last = n_hex_digits.saturating_sub(1_000_000)..n_hex_digits;

    let file_ref = &file;
    let stop_ref: &AtomicBool = &stop;
    let error_slot_ref: &Mutex<Option<anyhow::Error>> = &first_error;
    let first_bar = &sanity.first;
    let middle_bar = &sanity.middle;
    let last_bar = &sanity.last;
    let random_bar = &random;

    pool.scope(|s| {
        s.spawn(move |_| {
            run_sanity_region(
                file_ref,
                data_offset,
                range_first,
                sanity_samples,
                first_bar,
                stop_ref,
                error_slot_ref,
            );
        });
        s.spawn(move |_| {
            run_sanity_region(
                file_ref,
                data_offset,
                range_middle,
                sanity_samples,
                middle_bar,
                stop_ref,
                error_slot_ref,
            );
        });
        s.spawn(move |_| {
            run_sanity_region(
                file_ref,
                data_offset,
                range_last,
                sanity_samples,
                last_bar,
                stop_ref,
                error_slot_ref,
            );
        });
        s.spawn(move |_| {
            run_random_loop(
                file_ref,
                data_offset,
                n_hex_digits,
                samples_per_window,
                random_bar,
                stop_ref,
                error_slot_ref,
            );
        });
    });

    // ---------------------------------------------------------------------
    // Decide exit status from the shared error slot.  An error (mismatch
    // detected) propagates as anyhow::Error; otherwise, if `stop` was set
    // it's a user-requested SIGINT and we print a summary line.
    // ---------------------------------------------------------------------
    if let Some(err) = first_error.lock().unwrap().take() {
        return Err(err);
    }
    if stop.load(Ordering::SeqCst) {
        eprintln!("verify-hex: interrupted.");
    }
    Ok(())
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
        bar.tick();
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
    /// internal mutex, so rayon workers can call this concurrently.
    fn inc(&self, n: u64) {
        self.bar.inc(n);
    }

    fn complete(&self) {
        if self.is_spinner {
            self.bar.disable_steady_tick();
            let pos = self.bar.position();
            self.bar.set_length(pos.max(1));
        }
        self.bar.set_style(done_style());
        // Force a redraw with the new style before marking the bar
        // finished — without this, a bar that arrived at its final
        // position with the old (active) style still showing on screen
        // will not pick up the new style (the implicit draw from
        // `finish()` is sometimes skipped when the position doesn't
        // change).  This shows up as a sanity bar that completed at
        // 100/100 but stays "eta 0s" instead of "done in X".
        self.bar.tick();
        self.bar.finish();
    }

    /// Mark the phase as halted (SIGINT or a mismatch in another phase).
    /// Position is frozen wherever the work happened to be — no jump to
    /// the bar's full length — and the style switches to yellow with
    /// "interrupted at {elapsed}".
    fn interrupted(&self) {
        if self.is_spinner {
            self.bar.disable_steady_tick();
            self.bar.set_style(interrupted_spinner_style());
        } else {
            self.bar.set_style(interrupted_bar_style());
        }
        self.bar.tick();
        // `abandon` finalizes the bar without modifying its position
        // (unlike `finish`, which jumps it to `length`).
        self.bar.abandon();
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

fn interrupted_bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:<24} [{bar:30.yellow}] {human_pos:>13}/{human_len:<13} interrupted at {elapsed}",
    )
    .unwrap()
    .progress_chars("##-")
}

fn interrupted_spinner_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:<24} {human_pos:>13} so far                | interrupted at {elapsed}",
    )
    .unwrap()
}

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
// Conversion (decimal -> hex)
// =========================================================================

fn convert_decimal_to_hex(
    decimal_path: &Path,
    hex_path: &Path,
    bars: &ConversionBars,
) -> Result<()> {
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

    bars.scale.activate();
    let log_16_10 = (10.0_f64.log2()) / 4.0;
    let h: u32 = ((d_minus_1 as f64) * log_16_10) as u32;
    let scale_hex = Integer::from(Integer::u_pow_u(16, h));
    let scale_dec = Integer::from(Integer::u_pow_u(10, d_minus_1));
    bars.scale.inc(1);
    bars.scale.complete();

    bars.muldiv.activate();
    let product = numerator * scale_hex;
    let m = product / scale_dec;
    bars.muldiv.inc(1);
    bars.muldiv.complete();

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
    let mut header = [0_u8; 2];
    read_exact_at(file, &mut header, 0)?;
    if &header != b"3." {
        bail!("hex file doesn't start with \"3.\"");
    }
    let data_offset = 2;
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
// Sanity phase (one region per call; called in parallel for the three
// regions inside `pool.scope`).
// =========================================================================

fn run_sanity_region(
    file: &File,
    data_offset: u64,
    range: Range<u64>,
    samples: usize,
    bar: &PhaseBar,
    stop: &AtomicBool,
    error_slot: &Mutex<Option<anyhow::Error>>,
) {
    // Stop was already set before this phase could start: transition the
    // bar pending → interrupted so the user doesn't see a "(pending)"
    // row sitting under a "verify-hex: interrupted" summary.
    if stop.load(Ordering::SeqCst) {
        bar.interrupted();
        return;
    }
    bar.activate();

    let range_len = range.end - range.start;
    if range_len < 8 {
        bar.complete();
        return;
    }
    let max_start = range.end - 8;
    let positions: Vec<u64> = (0..samples)
        .map(|i| {
            let span = max_start - range.start;
            let denom = (samples.saturating_sub(1)).max(1) as u64;
            range.start + (span * i as u64) / denom
        })
        .collect();

    let result = positions.par_iter().try_for_each(|&pos| -> Result<()> {
        if stop.load(Ordering::SeqCst) {
            return Ok(());
        }
        check_position(file, data_offset, pos, stop, bar)
    });

    if let Err(e) = result {
        record_error(error_slot, e);
        stop.store(true, Ordering::SeqCst);
    }

    if stop.load(Ordering::SeqCst) {
        bar.interrupted();
    } else {
        bar.complete();
    }
}

// =========================================================================
// Random sampling phase (unbounded, runs concurrently with the sanity
// regions on the same pool).
// =========================================================================

const WINDOW_BYTES: u64 = 1_000_000;

fn run_random_loop(
    file: &File,
    data_offset: u64,
    n_hex_digits: u64,
    samples_per_window: usize,
    bar: &PhaseBar,
    stop: &AtomicBool,
    error_slot: &Mutex<Option<anyhow::Error>>,
) {
    if stop.load(Ordering::SeqCst) {
        bar.interrupted();
        return;
    }
    if n_hex_digits < 8 {
        record_error(
            error_slot,
            anyhow!("hex file has fewer than 8 hex digits; nothing to sample"),
        );
        stop.store(true, Ordering::SeqCst);
        bar.interrupted();
        return;
    }
    bar.activate();

    while !stop.load(Ordering::SeqCst) {
        let mut rng = OsRng;
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

        let result = positions.par_iter().try_for_each(|&pos| -> Result<()> {
            if stop.load(Ordering::SeqCst) {
                return Ok(());
            }
            check_position(file, data_offset, pos, stop, bar)
        });

        if let Err(e) = result {
            record_error(error_slot, e);
            stop.store(true, Ordering::SeqCst);
            break;
        }
    }

    // Random sampling is unbounded — exiting always means stop was set
    // (either by SIGINT or by a mismatch somewhere).
    bar.interrupted();
}

// =========================================================================
// Single-position BBP check + file compare
// =========================================================================

fn check_position(
    file: &File,
    data_offset: u64,
    pos: u64,
    stop: &AtomicBool,
    bar: &PhaseBar,
) -> Result<()> {
    let bbp_digits = match bbp::hex_digits_at_interruptible(pos, stop) {
        Some(d) => d,
        None => return Ok(()), // BBP bailed mid-call on stop; no compare done.
    };

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

    bar.inc(1);
    Ok(())
}

/// Record the *first* error any phase reports; subsequent reports are
/// dropped (the user only sees one).
fn record_error(slot: &Mutex<Option<anyhow::Error>>, err: anyhow::Error) {
    if let Ok(mut g) = slot.lock() {
        if g.is_none() {
            *g = Some(err);
        }
    }
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
        use std::io::{Read, Seek, SeekFrom};
        let mut file = file.try_clone()?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf).map_err(|e| anyhow!("{e}"))
    }
}
