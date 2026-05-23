//! `--verify-hex` mode: convert a decimal pi file to hex (one-time,
//! cached on disk), then BBP-spot-check the hex file at deterministic
//! sanity positions followed by an open-ended cryptographically-random
//! sampling loop.
//!
//! See `verify-hex` design notes in the README.

use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use rand::rngs::OsRng;
use rand::Rng;
use rayon::prelude::*;
use rug::{Assign, Integer};

use pi_core::bbp;

use crate::fmt_thousands;

/// Top-level entry point.
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
        (false, Some(dec)) => convert_decimal_to_hex(dec, hex_path)?,
        (true, None) => {
            eprintln!("reusing existing hex file: {}", hex_path.display());
        }
    }

    // ---------------------------------------------------------------------
    // Open hex file and figure out content extents.
    // ---------------------------------------------------------------------
    let file = File::open(hex_path)
        .with_context(|| format!("opening `{}`", hex_path.display()))?;
    let (data_offset, n_hex_digits) = scan_hex_file(&file)?;
    eprintln!(
        "hex file: {} ({} hex digits past \"3.\")",
        hex_path.display(),
        fmt_thousands(n_hex_digits)
    );

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
    eprintln!("rayon threads: {}", pool.current_num_threads());

    // ---------------------------------------------------------------------
    // SIGINT handling.  The handler flips an AtomicBool; the random loop
    // checks it between windows.
    // ---------------------------------------------------------------------
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let i = Arc::clone(&interrupted);
        // `set_handler` is per-process; in tests we'd skip this.  For the
        // CLI binary it's fine.
        let _ = ctrlc::set_handler(move || {
            i.store(true, Ordering::SeqCst);
        });
    }

    // ---------------------------------------------------------------------
    // Sanity sweep (bounded, fails fast on mismatch).
    // ---------------------------------------------------------------------
    pool.install(|| sanity_phase(&file, data_offset, n_hex_digits, sanity_samples))?;

    // ---------------------------------------------------------------------
    // Random sampling loop (unbounded until Ctrl-C or mismatch).
    // ---------------------------------------------------------------------
    pool.install(|| {
        random_sampling_phase(
            &file,
            data_offset,
            n_hex_digits,
            samples_per_window,
            &interrupted,
        )
    })
}

// =========================================================================
// Conversion (decimal -> hex)
// =========================================================================

fn convert_decimal_to_hex(decimal_path: &Path, hex_path: &Path) -> Result<()> {
    let overall = Instant::now();
    eprintln!(
        "converting decimal -> hex:\n  in:  {}\n  out: {}",
        decimal_path.display(),
        hex_path.display()
    );

    // --- Phase 1: read + parse to Integer ---------------------------------
    let phase = Instant::now();
    eprintln!("  [1/4] reading + parsing decimal");
    let content = fs::read_to_string(decimal_path)
        .with_context(|| format!("reading `{}`", decimal_path.display()))?;
    let trimmed = content.trim();
    if !trimmed.starts_with("3.") {
        bail!("`{}` doesn't start with \"3.\"", decimal_path.display());
    }
    let frac_str = &trimmed[2..];
    if !frac_str.chars().all(|c| c.is_ascii_digit()) {
        bail!("`{}` has non-digit characters after \"3.\"", decimal_path.display());
    }
    // The numerator is `pi · 10^(D-1)` as an integer: "3" followed by the
    // fractional decimal digits.
    let digit_string = format!("3{frac_str}");
    let d_minus_1: u32 = frac_str
        .len()
        .try_into()
        .context("fractional decimal digit count exceeds u32::MAX")?;
    let mut numerator = Integer::new();
    numerator
        .assign(Integer::parse_radix(&digit_string, 10)?);
    eprintln!(
        "        {} fractional decimal digits parsed in {:.1?}",
        fmt_thousands(d_minus_1 as u64),
        phase.elapsed()
    );
    // Free the string copy now that we have the Integer.
    drop(content);
    drop(digit_string);

    // --- Phase 2: build 16^H and 10^(D-1) ---------------------------------
    let phase = Instant::now();
    // H = floor(D_minus_1 · log_16(10)) hex digits of precision.
    // log_16(10) = log(10) / log(16) = log_2(10) / 4 ≈ 0.8304.
    let log_16_10 = (10.0_f64.log2()) / 4.0;
    let h: u32 = ((d_minus_1 as f64) * log_16_10) as u32;
    eprintln!(
        "  [2/4] computing 16^{} and 10^{}",
        fmt_thousands(h as u64),
        fmt_thousands(d_minus_1 as u64)
    );
    let scale_hex = Integer::from(Integer::u_pow_u(16, h));
    let scale_dec = Integer::from(Integer::u_pow_u(10, d_minus_1));
    eprintln!("        built scale factors in {:.1?}", phase.elapsed());

    // --- Phase 3: numerator · 16^H / 10^(D-1) -----------------------------
    let phase = Instant::now();
    eprintln!("  [3/4] multiplying and dividing");
    let product = numerator * scale_hex;
    let m = product / scale_dec;
    eprintln!("        computed `pi · 16^H` in {:.1?}", phase.elapsed());

    // --- Phase 4: write hex output atomically -----------------------------
    let phase = Instant::now();
    eprintln!("  [4/4] writing hex file (via .tmp + rename)");
    let hex_str = m.to_string_radix(16);
    // The result should be "3" followed by H hex digits — the leading "3"
    // is the integer part of pi, and the rest is the fractional expansion.
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
    fs::rename(&tmp_path, hex_path)
        .with_context(|| format!("renaming `{}` -> `{}`", tmp_path.display(), hex_path.display()))?;
    eprintln!(
        "        wrote {} bytes in {:.1?}",
        fmt_thousands((hex_str.len() + 3) as u64),
        phase.elapsed()
    );

    eprintln!("conversion done in {:.1?}", overall.elapsed());
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

fn sanity_phase(
    file: &File,
    data_offset: u64,
    n_hex_digits: u64,
    samples_per_region: usize,
) -> Result<()> {
    eprintln!("sanity phase: 3 regions × {} samples × 8 hex digits each",
              fmt_thousands(samples_per_region as u64));

    let one_m = 1_000_000_u64.min(n_hex_digits);
    let half = n_hex_digits / 2;
    let regions: [(&str, std::ops::Range<u64>); 3] = [
        ("first ", 0..one_m),
        (
            "middle",
            half..(half + one_m).min(n_hex_digits),
        ),
        (
            "last  ",
            n_hex_digits.saturating_sub(one_m)..n_hex_digits,
        ),
    ];

    let started = Instant::now();
    for (name, range) in regions {
        let region_start = Instant::now();
        let range_len = range.end - range.start;
        if range_len < 8 {
            eprintln!("  {name}: region < 8 hex digits, skipping");
            continue;
        }
        // Pick samples_per_region positions evenly spaced through the region.
        // Each sample's BBP call covers positions p..p+8, so clamp the
        // ranges to ensure the 8-digit window stays inside the region.
        let max_start = range.end - 8;
        let positions: Vec<u64> = (0..samples_per_region)
            .map(|i| {
                let span = max_start - range.start;
                range.start + (span * i as u64) / (samples_per_region.saturating_sub(1).max(1) as u64)
            })
            .collect();

        check_positions(file, data_offset, &positions)?;
        eprintln!(
            "  {name} [{}..{}): {} samples ok in {:.1?}",
            fmt_thousands(range.start),
            fmt_thousands(range.end),
            fmt_thousands(samples_per_region as u64),
            region_start.elapsed()
        );
    }
    eprintln!("sanity phase complete in {:.1?}", started.elapsed());
    Ok(())
}

// =========================================================================
// Random sampling
// =========================================================================

const WINDOW_BYTES: u64 = 1_000_000;

fn random_sampling_phase(
    file: &File,
    data_offset: u64,
    n_hex_digits: u64,
    samples_per_window: usize,
    interrupted: &Arc<AtomicBool>,
) -> Result<()> {
    eprintln!(
        "random sampling: {} samples per window × 8 hex digits, OsRng (Ctrl-C to stop)",
        fmt_thousands(samples_per_window as u64)
    );

    let started = Instant::now();
    let mut windows: u64 = 0;
    let mut samples: u64 = 0;

    while !interrupted.load(Ordering::SeqCst) {
        let mut rng = OsRng;

        // Window start uniformly in [0, max(n_hex_digits - 8, 1)).
        if n_hex_digits < 8 {
            bail!("hex file has fewer than 8 hex digits, can't sample");
        }
        let max_window_start = n_hex_digits - 8;
        let p = rng.gen_range(0..=max_window_start);
        let window_end = (p + WINDOW_BYTES).min(n_hex_digits);
        let window_span = window_end - p;

        // Pick positions in [p, p + window_span - 8] so the 8-digit window
        // around each BBP call stays inside the window (and the file).
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

        check_positions(file, data_offset, &positions)?;

        windows += 1;
        samples += n_samples as u64;

        if windows.is_multiple_of(10) {
            let elapsed = started.elapsed();
            let positions_checked = samples * 8;
            eprintln!(
                "  windows: {} | samples: {} | positions: {} | elapsed: {:.0?}",
                fmt_thousands(windows),
                fmt_thousands(samples),
                fmt_thousands(positions_checked),
                elapsed
            );
        }
    }

    let elapsed = started.elapsed();
    eprintln!("interrupted.");
    eprintln!(
        "verified {} windows, {} samples, {} positions in {:.1?}",
        fmt_thousands(windows),
        fmt_thousands(samples),
        fmt_thousands(samples * 8),
        elapsed
    );
    Ok(())
}

// =========================================================================
// Parallel check workhorse
// =========================================================================

/// Run BBP on each position in parallel, comparing the 8 hex digits
/// returned against the bytes at the corresponding file offset.  Returns
/// `Err` on the first detected mismatch (`find_map_any` short-circuits).
fn check_positions(file: &File, data_offset: u64, positions: &[u64]) -> Result<()> {
    let mismatch = positions
        .par_iter()
        .find_map_any(|&pos| check_position(file, data_offset, pos).err());
    if let Some(e) = mismatch {
        return Err(e);
    }
    Ok(())
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
        // Slow fallback: lock + seek + read.  Not safe for concurrent
        // calls; only used on exotic platforms.
        let mut file = file.try_clone()?;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf).map_err(|e| anyhow!("{e}"))
    }
}
