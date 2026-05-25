//! Runtime-tunable performance parameters for the bignum crate.
//!
//! Every value here defaults to a setting that works well on a
//! consumer laptop (8–16 cores, 16–32 GB RAM).  Embedding programs
//! that need to tune for different hardware should build a
//! [`Config`], populate the fields they want to override, and call
//! [`apply`].  The bignum crate itself never parses the TOML file —
//! that's the embedder's job; we just receive the parsed values.
//!
//! The values live in `AtomicUsize`s so reads inside dispatch
//! functions are cheap (one relaxed load) and the entire crate can
//! be tuned from the outside without rebuilding.  Setting happens
//! once at program startup; the atomics avoid having to use a
//! `OnceLock` and a panic-on-double-init pattern.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Performance knobs exposed by the bignum crate.  Field-for-field
/// match against TOML sections `[bignum]` and `[bignum.ntt]` in
/// `pi-cli`.
#[derive(Debug, Clone)]
pub struct Config {
    /// Limb count below which schoolbook (`O(N^2)`) multiplication
    /// beats Karatsuba.  Lower values pull more inputs into Karatsuba;
    /// higher values keep more on schoolbook.
    pub karatsuba_threshold: usize,
    /// Limb count above which Karatsuba's three sub-multiplications
    /// run on separate rayon worker threads.  Lower values spawn
    /// rayon tasks for smaller inputs (more parallelism, more
    /// overhead); higher values keep medium-sized multiplications
    /// sequential.
    pub parallel_karatsuba_threshold: usize,
    /// Limb count above which `mul_mag` dispatches to the Goldilocks
    /// NTT.  Lower = NTT for more sizes (better asymptotics but
    /// worse small-size constants).
    pub ntt_threshold: usize,
    /// Limb count above which integer division uses Newton–Raphson
    /// reciprocal (built on Karatsuba/NTT mul) instead of Knuth
    /// Algorithm D.  Lower = more divisions take the NR path.
    pub newton_div_threshold: usize,
    /// Limb count below which the divide-and-conquer base-10
    /// conversion falls back to the naive divide-by-10^19 leaf.
    pub to_string_dc_threshold: usize,
    /// Limb count above which the D&C base-10 conversion's hi/lo
    /// recursion forks onto separate rayon workers.  Lower = more
    /// fine-grained parallelism, more task overhead.
    pub parallel_to_string_threshold: usize,
    pub ntt: NttConfig,
    /// Integer limb buffers with at least this many `u64`s are
    /// allocated via `mmap`-backed scratch files instead of in heap
    /// memory.  The OS page cache then handles paging between RAM
    /// and SSD on access.  Set to `usize::MAX` (the default) to
    /// disable disk-backing entirely — every Integer stays in
    /// `Vec<u64>`.  Tuning down to (say) `5_000_000` pushes large
    /// integers (~40 MB+) to disk for runs whose total live state
    /// would exceed RAM.
    pub disk_limb_threshold: usize,
}

#[derive(Debug, Clone)]
pub struct NttConfig {
    /// Target u64-element count per rayon task in butterfly passes.
    /// Sized so each task fits in L2.  Smaller = finer parallelism
    /// (good when core count exceeds the L2-budget breakdown).
    pub target_task_size: usize,
    /// Limb count above which packing limbs into NTT coefficients
    /// runs in parallel.
    pub parallel_pack_threshold: usize,
    /// NTT element count above which the pointwise multiply in
    /// `mul_mag_ntt` runs in parallel.
    pub parallel_pointwise_threshold: usize,
    /// NTT transform length `N` at or above which `mul_mag_ntt`
    /// dispatches to the four-step (Bailey matrix Fourier)
    /// decomposition.  Below this we use the straight radix-2 NTT.
    /// Four-step trades extra arithmetic (transpose + cross-twiddle)
    /// for L2-resident sub-FFTs; it wins once `N` is large enough
    /// that radix-2's late butterfly passes stride beyond cache and
    /// every access misses.  Set to `2^32` (effectively the max
    /// possible N) to disable four-step entirely.
    ///
    /// **Empirical note:** measured on a 2-core Graviton 2 host at
    /// N = 2^21 (10M-digit pi top-of-tree mults), four-step is
    /// ~28 % *slower* than radix-2 — the constant overhead of
    /// 2·√N sub-FFTs plus the rectangular-N transpose outweighs the
    /// modest cache savings when only the last few radix-2 passes
    /// were actually cache-bound.  The default is therefore set
    /// conservatively at 2^25; production hosts with high core
    /// counts and runs at N ≥ 2^27 will see real benefit and may
    /// want to lower it via `--config`.
    pub four_step_threshold: usize,
}


impl Default for Config {
    /// Laptop defaults (8–16 cores).  Matches the constants the crate
    /// shipped with before runtime tuning was introduced.
    fn default() -> Self {
        Self {
            karatsuba_threshold: 32,
            parallel_karatsuba_threshold: 512,
            ntt_threshold: 8192,
            newton_div_threshold: 64,
            to_string_dc_threshold: 32,
            parallel_to_string_threshold: 256,
            ntt: NttConfig::default(),
            disk_limb_threshold: usize::MAX,
        }
    }
}

impl Default for NttConfig {
    fn default() -> Self {
        Self {
            target_task_size: 1 << 16, // 64 K u64s = 512 KB per task
            parallel_pack_threshold: 1024,
            parallel_pointwise_threshold: 1024,
            // 2^25 = ~33 M elements (256 MiB working set).  See the
            // empirical note on the field doc — at smaller N our
            // four-step impl loses to radix-2 because the per-call
            // overhead (transposes + 2·√N sub-FFTs) outweighs the
            // late-pass cache savings.  Production hosts can lower
            // this via `--config` once they have benchmark evidence
            // it pays off at their N and core count.
            four_step_threshold: 1 << 25,
        }
    }
}

// =====================================================================
// Backing atomics — initial values match `Config::default()`.
// =====================================================================

static KARATSUBA_THRESHOLD: AtomicUsize = AtomicUsize::new(32);
static PARALLEL_KARATSUBA_THRESHOLD: AtomicUsize = AtomicUsize::new(512);
static NTT_THRESHOLD: AtomicUsize = AtomicUsize::new(8192);
static NEWTON_DIV_THRESHOLD: AtomicUsize = AtomicUsize::new(64);
static TO_STRING_DC_THRESHOLD: AtomicUsize = AtomicUsize::new(32);
static PARALLEL_TO_STRING_THRESHOLD: AtomicUsize = AtomicUsize::new(256);
static NTT_TARGET_TASK_SIZE: AtomicUsize = AtomicUsize::new(1 << 16);
static NTT_PARALLEL_PACK_THRESHOLD: AtomicUsize = AtomicUsize::new(1024);
static NTT_PARALLEL_POINTWISE_THRESHOLD: AtomicUsize = AtomicUsize::new(1024);
static NTT_FOUR_STEP_THRESHOLD: AtomicUsize = AtomicUsize::new(1 << 25);
static DISK_LIMB_THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Push a [`Config`] into the live atomics.  Call this once at
/// program startup, before any compute path runs.  Repeat calls just
/// overwrite the previous values.
pub fn apply(c: &Config) {
    KARATSUBA_THRESHOLD.store(c.karatsuba_threshold, Ordering::Relaxed);
    PARALLEL_KARATSUBA_THRESHOLD
        .store(c.parallel_karatsuba_threshold, Ordering::Relaxed);
    NTT_THRESHOLD.store(c.ntt_threshold, Ordering::Relaxed);
    NEWTON_DIV_THRESHOLD.store(c.newton_div_threshold, Ordering::Relaxed);
    TO_STRING_DC_THRESHOLD.store(c.to_string_dc_threshold, Ordering::Relaxed);
    PARALLEL_TO_STRING_THRESHOLD
        .store(c.parallel_to_string_threshold, Ordering::Relaxed);
    NTT_TARGET_TASK_SIZE.store(c.ntt.target_task_size, Ordering::Relaxed);
    NTT_PARALLEL_PACK_THRESHOLD
        .store(c.ntt.parallel_pack_threshold, Ordering::Relaxed);
    NTT_PARALLEL_POINTWISE_THRESHOLD
        .store(c.ntt.parallel_pointwise_threshold, Ordering::Relaxed);
    NTT_FOUR_STEP_THRESHOLD.store(c.ntt.four_step_threshold, Ordering::Relaxed);
    DISK_LIMB_THRESHOLD.store(c.disk_limb_threshold, Ordering::Relaxed);
}

/// Snapshot the currently-applied configuration by reading the live
/// atomics into a fresh `Config`.  Useful for `pi-core::perf` to
/// capture what was active for the run; also useful in tests that
/// want to assert post-`apply` state.
impl Config {
    pub fn current() -> Self {
        Self {
            karatsuba_threshold: KARATSUBA_THRESHOLD.load(Ordering::Relaxed),
            parallel_karatsuba_threshold: PARALLEL_KARATSUBA_THRESHOLD
                .load(Ordering::Relaxed),
            ntt_threshold: NTT_THRESHOLD.load(Ordering::Relaxed),
            newton_div_threshold: NEWTON_DIV_THRESHOLD.load(Ordering::Relaxed),
            to_string_dc_threshold: TO_STRING_DC_THRESHOLD.load(Ordering::Relaxed),
            parallel_to_string_threshold: PARALLEL_TO_STRING_THRESHOLD
                .load(Ordering::Relaxed),
            ntt: NttConfig {
                target_task_size: NTT_TARGET_TASK_SIZE.load(Ordering::Relaxed),
                parallel_pack_threshold: NTT_PARALLEL_PACK_THRESHOLD
                    .load(Ordering::Relaxed),
                parallel_pointwise_threshold: NTT_PARALLEL_POINTWISE_THRESHOLD
                    .load(Ordering::Relaxed),
                four_step_threshold: NTT_FOUR_STEP_THRESHOLD.load(Ordering::Relaxed),
            },
            disk_limb_threshold: DISK_LIMB_THRESHOLD.load(Ordering::Relaxed),
        }
    }
}

#[inline]
pub(crate) fn disk_limb_threshold() -> usize {
    DISK_LIMB_THRESHOLD.load(Ordering::Relaxed)
}

// =====================================================================
// Crate-internal getters — these replace the former `const`s in
// integer.rs and ntt.rs.  All inlined; the load is a single mov on
// x86_64 / ldar on aarch64.
// =====================================================================

#[inline]
pub(crate) fn karatsuba_threshold() -> usize {
    KARATSUBA_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn parallel_karatsuba_threshold() -> usize {
    PARALLEL_KARATSUBA_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn ntt_threshold() -> usize {
    NTT_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn newton_div_threshold() -> usize {
    NEWTON_DIV_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn to_string_dc_threshold() -> usize {
    TO_STRING_DC_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn parallel_to_string_threshold() -> usize {
    PARALLEL_TO_STRING_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn ntt_target_task_size() -> usize {
    NTT_TARGET_TASK_SIZE.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn ntt_parallel_pack_threshold() -> usize {
    NTT_PARALLEL_PACK_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn ntt_parallel_pointwise_threshold() -> usize {
    NTT_PARALLEL_POINTWISE_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub(crate) fn ntt_four_step_threshold() -> usize {
    NTT_FOUR_STEP_THRESHOLD.load(Ordering::Relaxed)
}

/// Test-only setter for the four-step threshold; lets tests force the
/// four-step path on small inputs.  Restore the previous value before
/// the test ends to keep other tests independent.
#[cfg(test)]
pub(crate) fn set_ntt_four_step_threshold_for_test(v: usize) -> usize {
    NTT_FOUR_STEP_THRESHOLD.swap(v, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_atomics_at_startup() {
        let d = Config::default();
        assert_eq!(d.karatsuba_threshold, karatsuba_threshold());
        assert_eq!(d.ntt_threshold, ntt_threshold());
        assert_eq!(d.ntt.target_task_size, ntt_target_task_size());
    }

    #[test]
    fn apply_round_trips() {
        let mut c = Config::default();
        c.ntt_threshold = 9999;
        c.ntt.target_task_size = 32_768;
        apply(&c);
        assert_eq!(ntt_threshold(), 9999);
        assert_eq!(ntt_target_task_size(), 32_768);
        // Restore defaults so we don't pollute later tests.
        apply(&Config::default());
    }
}
