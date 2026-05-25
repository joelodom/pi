//! Hardware-aware config TOML generator.
//!
//! Output is a complete `pi --config` file with per-knob justification
//! comments explaining *why* the picked value, not just what it does.
//! Same TOML schema accepted by `load_and_apply_config` — generated
//! configs can be loaded back directly.

use crate::hardware::HardwareProfile;
use std::fmt::Write;

/// Memory-mode tier picked from `estimated_peak_full_gb / ram_gb`.
#[derive(Debug, Clone, Copy, PartialEq)]
enum MemoryMode {
    /// Plenty of headroom (peak < 50% of RAM).  Speed-tuned defaults.
    Speed,
    /// Tight (peak 50–80% of RAM).  Sequential FA only.
    Moderate,
    /// Over budget (peak > 80% of RAM).  Sequential FA + sequential
    /// binary-split top.  Expects some swap activity.
    Conservative,
}

pub fn generate(digits: u64, hw: &HardwareProfile) -> String {
    let cores = hw.logical_cores;
    let ram_gb = hw.ram_bytes as f64 / 1024.0_f64.powi(3);

    // ---- Memory model ------------------------------------------------
    //
    // Empirically: a D-digit pi run peaks at roughly
    //   peak_rss ≈ 5.4 GB × (D / 100M)        (fully parallel)
    // measured at 100M on both 11-core (5.0 GB) and 64-core (5.0 GB)
    // hosts.  Peak is dominated by the top-of-tree integers (Q, T,
    // denom, recip, pi_numer) and 1-2 concurrent NTT scratch buffers
    // in FA — neither scales with core count.  An earlier version
    // multiplied by `cores/11` and overestimated by ~10× on
    // many-core hosts (predicted 314 GB at 1B on 64 cores, actual
    // peak 31 GB), forcing disk-backing to engage on instances with
    // ample RAM.  Linear-in-digits, flat-in-cores matches the data.
    let est_peak_full_gb = 5.4 * (digits as f64) / 1.0e8;

    let memory_mode = if hw.ram_bytes == 0 {
        // Unknown RAM — be conservative.
        MemoryMode::Conservative
    } else if est_peak_full_gb < ram_gb * 0.5 {
        MemoryMode::Speed
    } else if est_peak_full_gb < ram_gb * 0.8 {
        MemoryMode::Moderate
    } else {
        MemoryMode::Conservative
    };

    // ---- Knob picks --------------------------------------------------

    // Parallel-Karatsuba: lower threshold on many-core boxes so finer
    // Karatsuba sub-multiplications get rayon-dispatched.
    let parallel_karatsuba_threshold = match cores {
        0..=12 => 512,
        13..=32 => 256,
        33..=64 => 192,
        _ => 128,
    };

    // NTT task size: smaller chunks fan out work across more cores.
    let ntt_target_task_size = match cores {
        0..=12 => 65_536,
        13..=32 => 32_768,
        _ => 16_384,
    };

    // chudnovsky.parallel_split_threshold: lower on many-core boxes.
    let parallel_split_threshold = match cores {
        0..=12 => 64,
        13..=32 => 32,
        _ => 16,
    };

    // decimal_conversion.parallel_to_string_threshold: same shape.
    let parallel_to_string_threshold = match cores {
        0..=12 => 256,
        13..=32 => 128,
        _ => 64,
    };

    // Decide disk-backing first; it changes how aggressive the
    // memory-saving knobs below need to be.
    let disk_limb_threshold = pick_disk_limb_threshold(digits, est_peak_full_gb, ram_gb);
    let disk_engaged = disk_limb_threshold != usize::MAX;

    // sequential_top_threshold (terms): only meaningful when memory
    // is tight; sized so the top few levels of the tree serialise.
    // Two regimes:
    //   * Disk-backed: only need to bound *concurrent NTT scratch*.
    //     Integer memory spills to disk, so we only need to keep the
    //     handful of largest-N NTT calls serialised.  Empirically
    //     ~digits/30 catches just the top 1–2 levels at billion scale.
    //   * No disk-backing: must bound both NTT scratch AND total
    //     concurrent Integer state.  Much more aggressive — digits/1000.
    let sequential_top_threshold = match memory_mode {
        MemoryMode::Speed => 0,
        MemoryMode::Moderate => 0,
        MemoryMode::Conservative if disk_engaged => digits / 30,
        MemoryMode::Conservative => digits / 1000,
    };

    // With disk-backing, the FA Float intermediates spill to disk too,
    // so concurrent recip+sqrt chains don't double live memory the way
    // they did in pure-RAM mode.  Keep parallel FA on when disk handles
    // the pressure.
    let parallel_final_assembly = match memory_mode {
        MemoryMode::Speed | MemoryMode::Moderate => true,
        MemoryMode::Conservative if disk_engaged => true,
        MemoryMode::Conservative => false,
    };

    // Sample interval: longer runs warrant coarser sampling.
    let default_sample_ms = match digits {
        0..=10_000_000 => 250,
        10_000_001..=200_000_000 => 500,
        200_000_001..=5_000_000_000 => 1_000,
        _ => 2_000,
    };

    // ---- TOML output -------------------------------------------------

    let mut s = String::new();
    writeln!(s, "# =============================================================================").unwrap();
    writeln!(s, "# Generated by `pi --generate-config {digits}`").unwrap();
    writeln!(s, "# Host: {} cores logical ({} physical), {} RAM, {}/{}",
        cores,
        hw.physical_cores,
        if hw.ram_bytes > 0 { format!("{:.1} GB", ram_gb) } else { "unknown".into() },
        hw.os, hw.arch,
    ).unwrap();
    writeln!(s, "# Target: {} digits", fmt_thousands(digits)).unwrap();
    writeln!(s, "# Estimated peak RSS at full parallelism: {:.1} GB", est_peak_full_gb).unwrap();
    writeln!(s, "#   (linear extrapolation from a measured 5.4 GB at 100M; peak is set by").unwrap();
    writeln!(s, "#    top-of-tree integers + NTT scratch, which do not scale with core count)").unwrap();
    writeln!(s, "# Memory mode: {:?}", memory_mode).unwrap();
    write_memory_mode_rationale(&mut s, memory_mode, est_peak_full_gb, ram_gb);
    writeln!(s, "# =============================================================================").unwrap();
    writeln!(s).unwrap();

    // [runtime]
    writeln!(s, "[runtime]").unwrap();
    writeln!(s, "# 0 = autodetect (rayon will pick {cores}).  Set non-zero to leave headroom").unwrap();
    writeln!(s, "# for other processes or to confine to a NUMA node.").unwrap();
    writeln!(s, "threads = 0").unwrap();
    writeln!(s).unwrap();

    // [bignum]
    writeln!(s, "[bignum]").unwrap();
    writeln!(s, "# Algorithmic crossover, doesn't shift with hardware.").unwrap();
    writeln!(s, "karatsuba_threshold = 32").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "# Parallel-Karatsuba sub-multiplications begin at this size.").unwrap();
    writeln!(s, "# Picked {parallel_karatsuba_threshold} for {cores}-core hosts: smaller chunks fan").unwrap();
    writeln!(s, "# out work better; the laptop default 512 is right for ≤12 cores.").unwrap();
    writeln!(s, "parallel_karatsuba_threshold = {parallel_karatsuba_threshold}").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "# NTT engages at this limb count.  Algorithmic crossover (≈Karatsuba ↔ NTT").unwrap();
    writeln!(s, "# transition) and shifts little with hardware.").unwrap();
    writeln!(s, "ntt_threshold = 8192").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "# Newton-Raphson reciprocal division above this size.  Algorithmic.").unwrap();
    writeln!(s, "newton_div_threshold = 64").unwrap();
    writeln!(s).unwrap();
    if disk_limb_threshold == usize::MAX {
        writeln!(s, "# Disk-backed integer storage is disabled — peak RSS is").unwrap();
        writeln!(s, "# expected to fit comfortably in RAM.  Set lower to push").unwrap();
        writeln!(s, "# the biggest Integers to mmap'd scratch files (see").unwrap();
        writeln!(s, "# bignum::storage), at a wall-time cost from SSD access.").unwrap();
        writeln!(s, "# disk_limb_threshold = 18446744073709551615   # usize::MAX = disabled").unwrap();
    } else {
        writeln!(s, "# MEMORY-CONSERVATIVE: Integer limb buffers ≥ {} limbs",
            fmt_thousands(disk_limb_threshold as u64)).unwrap();
        writeln!(s, "# (~{} MB each) are allocated via `mmap`-backed scratch",
            disk_limb_threshold * 8 / (1024 * 1024)).unwrap();
        writeln!(s, "# files in `$TMPDIR` instead of heap memory.  Active because").unwrap();
        writeln!(s, "# estimated peak ({:.1} GB) > 70% of available RAM ({:.1} GB).",
            est_peak_full_gb, ram_gb).unwrap();
        writeln!(s, "# Run wall time will rise (SSD bandwidth replaces RAM); the").unwrap();
        writeln!(s, "# trade-off is what makes 1B-10B runs feasible on this box.").unwrap();
        writeln!(s, "disk_limb_threshold = {disk_limb_threshold}").unwrap();
    }
    writeln!(s).unwrap();

    // [bignum.ntt]
    writeln!(s, "[bignum.ntt]").unwrap();
    writeln!(s, "# Per-rayon-task work in butterfly passes (u64 elements).  Sized so").unwrap();
    writeln!(s, "# each task fits in L2 cache; smaller values spread work across more").unwrap();
    writeln!(s, "# cores at the cost of more task overhead.  Picked {ntt_target_task_size}").unwrap();
    writeln!(s, "# for a {cores}-core host (≈{} KB per task).", ntt_target_task_size * 8 / 1024).unwrap();
    writeln!(s, "target_task_size = {ntt_target_task_size}").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "# Pack / pointwise-mul parallelism cross-overs.  Defaults work").unwrap();
    writeln!(s, "# across a wide core range.").unwrap();
    writeln!(s, "parallel_pack_threshold = 1024").unwrap();
    writeln!(s, "parallel_pointwise_threshold = 1024").unwrap();
    writeln!(s).unwrap();

    // [chudnovsky]
    writeln!(s, "[chudnovsky]").unwrap();
    writeln!(s, "# Binary-split sub-trees with ≥ this many terms run in parallel via").unwrap();
    writeln!(s, "# rayon::join.  Picked {parallel_split_threshold} for {cores} cores.").unwrap();
    writeln!(s, "parallel_split_threshold = {parallel_split_threshold}").unwrap();
    writeln!(s).unwrap();
    if sequential_top_threshold > 0 {
        writeln!(s, "# MEMORY-CONSERVATIVE: sub-trees at or above {} terms (the top",
            fmt_thousands(sequential_top_threshold)).unwrap();
        writeln!(s, "# few levels of the tree) serialise instead of parallelising, so").unwrap();
        writeln!(s, "# at most one huge NTT scratch pair is live at the top.  Active").unwrap();
        writeln!(s, "# because estimated peak ({:.1} GB) > 80% of available RAM ({:.1} GB).",
            est_peak_full_gb, ram_gb).unwrap();
        writeln!(s, "sequential_top_threshold = {sequential_top_threshold}").unwrap();
    } else {
        writeln!(s, "# 0 = no upper cap; binary-split fully parallel at every level.").unwrap();
        writeln!(s, "# Estimated peak fits comfortably in RAM — no sequential cap needed.").unwrap();
        writeln!(s, "sequential_top_threshold = 0").unwrap();
    }
    writeln!(s).unwrap();
    if !parallel_final_assembly {
        writeln!(s, "# MEMORY-CONSERVATIVE: sqrt-chain and reciprocal run sequentially").unwrap();
        writeln!(s, "# so only one chain's Float intermediates are live at a time.").unwrap();
        writeln!(s, "# Costs ~10% wall in final assembly; saves multi-GB at billion+ scale.").unwrap();
        writeln!(s, "parallel_final_assembly = false").unwrap();
    } else {
        writeln!(s, "# Parallel chains in final assembly — they fill each other's NTT").unwrap();
        writeln!(s, "# serial pockets (~10% faster, more concurrent Float intermediates).").unwrap();
        writeln!(s, "parallel_final_assembly = true").unwrap();
    }
    writeln!(s).unwrap();

    // [decimal_conversion]
    writeln!(s, "[decimal_conversion]").unwrap();
    writeln!(s, "# Algorithmic crossover, unchanged with hardware.").unwrap();
    writeln!(s, "to_string_dc_threshold = 32").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "# D&C base-conversion parallelism crossover.  Picked {parallel_to_string_threshold}").unwrap();
    writeln!(s, "# for {cores} cores.").unwrap();
    writeln!(s, "parallel_to_string_threshold = {parallel_to_string_threshold}").unwrap();
    writeln!(s).unwrap();

    // [perf]
    writeln!(s, "[perf]").unwrap();
    writeln!(s, "# Default `--performance-sample-ms`.  Picked {default_sample_ms} ms for a").unwrap();
    writeln!(s, "# {}-digit run — coarser samples for longer runs keep the JSONL small.",
        fmt_thousands(digits)).unwrap();
    writeln!(s, "default_sample_ms = {default_sample_ms}").unwrap();

    s
}

fn write_memory_mode_rationale(s: &mut String, mode: MemoryMode, est_gb: f64, ram_gb: f64) {
    if ram_gb <= 0.0 {
        writeln!(s, "# (RAM unknown — defaulting to Conservative mode.)").unwrap();
        return;
    }
    let pct = (est_gb / ram_gb) * 100.0;
    match mode {
        MemoryMode::Speed => {
            writeln!(s, "#   Estimated peak {est_gb:.1} GB is {pct:.0}% of RAM (< 50%) —").unwrap();
            writeln!(s, "#   speed-tuned defaults; no memory-conservative knobs active.").unwrap();
        }
        MemoryMode::Moderate => {
            writeln!(s, "#   Estimated peak {est_gb:.1} GB is {pct:.0}% of RAM (50–80%) —").unwrap();
            writeln!(s, "#   sequential final-assembly to keep one less GB-class Float chain").unwrap();
            writeln!(s, "#   live at a time.").unwrap();
        }
        MemoryMode::Conservative => {
            writeln!(s, "#   Estimated peak {est_gb:.1} GB exceeds 80% of RAM ({ram_gb:.1} GB) —").unwrap();
            writeln!(s, "#   sequential binary-split top AND sequential final-assembly to").unwrap();
            writeln!(s, "#   bound concurrent NTT scratch.  Some swap activity expected; the").unwrap();
            writeln!(s, "#   run completes but watch `major_faults` in --performance-file.").unwrap();
        }
    }
}

/// Pick `disk_limb_threshold` for the host.
///
/// If estimated peak RSS fits comfortably in RAM (< 70%), disable
/// disk-backing entirely (`usize::MAX`).  Otherwise pick a threshold
/// just above the typical mid-tree Integer size, so the few largest
/// Integers (q, t, denom_int, pi mantissa, top-of-tree merges) go to
/// mmap while small/medium ones stay in RAM.
///
/// Heuristic: the largest live Integer is roughly `digits / 19` limbs
/// (full mantissa).  Threshold = `digits / 200` catches anything
/// roughly 10% the size of the mantissa or bigger.  Tunable later via
/// `--config`.
fn pick_disk_limb_threshold(digits: u64, est_peak_gb: f64, ram_gb: f64) -> usize {
    if ram_gb <= 0.0 || est_peak_gb < ram_gb * 0.7 {
        usize::MAX
    } else {
        // Cap at ~10M limbs so we never disk-back tiny intermediates
        // even on extreme runs.  At 1B digits this gives 5M; at 10B
        // it gives 50M (capped to 10M).
        ((digits / 200) as usize).clamp(50_000, 10_000_000)
    }
}

fn fmt_thousands(n: u64) -> String {
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
