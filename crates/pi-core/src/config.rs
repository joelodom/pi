//! Runtime-tunable performance parameters for `pi-core`.
//!
//! Same pattern as [`bignum::config`]: laptop-friendly defaults
//! backed by `AtomicUsize`s, with a public `apply` entry point so an
//! embedding program can override values once at startup.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Pi-core performance knobs.
#[derive(Debug, Clone)]
pub struct Config {
    pub chudnovsky: ChudnovskyConfig,
    pub perf: PerfDefaults,
}

#[derive(Debug, Clone)]
pub struct ChudnovskyConfig {
    /// Range size (in Chudnovsky terms) below which `binary_split`
    /// stays sequential; above it the two halves fork onto separate
    /// rayon workers.  Smaller = more parallel tasks at the leaves
    /// (good when you have many cores hungry for work).
    pub parallel_split_threshold: u64,
}

#[derive(Debug, Clone)]
pub struct PerfDefaults {
    /// Default value for the `--performance-sample-ms` CLI flag.
    /// Sampling overhead is tiny (well under 0.1% of one core at
    /// 500 ms); raise on long runs where 100 ms granularity is wasted.
    pub default_sample_ms: u64,
}

impl Default for Config {
    /// Laptop defaults (8–16 cores).
    fn default() -> Self {
        Self {
            chudnovsky: ChudnovskyConfig::default(),
            perf: PerfDefaults::default(),
        }
    }
}

impl Default for ChudnovskyConfig {
    fn default() -> Self {
        Self {
            parallel_split_threshold: 64,
        }
    }
}

impl Default for PerfDefaults {
    fn default() -> Self {
        Self {
            default_sample_ms: 500,
        }
    }
}

// =====================================================================
// Backing atomics
// =====================================================================

static CHUDNOVSKY_PARALLEL_SPLIT_THRESHOLD: AtomicU64 = AtomicU64::new(64);
static PERF_DEFAULT_SAMPLE_MS: AtomicU64 = AtomicU64::new(500);
// Placeholder for usize-typed knobs we'll add later (currently empty).
#[allow(dead_code)]
static _RESERVED_USIZE: AtomicUsize = AtomicUsize::new(0);

pub fn apply(c: &Config) {
    CHUDNOVSKY_PARALLEL_SPLIT_THRESHOLD
        .store(c.chudnovsky.parallel_split_threshold, Ordering::Relaxed);
    PERF_DEFAULT_SAMPLE_MS.store(c.perf.default_sample_ms, Ordering::Relaxed);
}

#[inline]
pub(crate) fn chudnovsky_parallel_split_threshold() -> u64 {
    CHUDNOVSKY_PARALLEL_SPLIT_THRESHOLD.load(Ordering::Relaxed)
}

#[inline]
pub fn perf_default_sample_ms() -> u64 {
    PERF_DEFAULT_SAMPLE_MS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_round_trips() {
        let mut c = Config::default();
        c.chudnovsky.parallel_split_threshold = 128;
        c.perf.default_sample_ms = 250;
        apply(&c);
        assert_eq!(chudnovsky_parallel_split_threshold(), 128);
        assert_eq!(perf_default_sample_ms(), 250);
        apply(&Config::default());
    }
}
