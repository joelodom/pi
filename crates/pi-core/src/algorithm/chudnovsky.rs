//! Chudnovsky algorithm with binary splitting.
//!
//! The Chudnovsky brothers' formula is the fastest known practical series
//! for pi:
//!
//! ```text
//! 1/π = 12 · Σ_{k=0}^{∞}  (-1)^k (6k)! (Bk + A)
//!                          ──────────────────────
//!                          (3k)! (k!)^3 · C^{3k+3/2}
//! ```
//!
//! with `A = 13_591_409`, `B = 545_140_134`, `C = 640_320`.  Each term
//! contributes about 14.18 decimal digits.
//!
//! Binary splitting evaluates the partial sum
//! `Σ_{k=1}^{N} M_k L_k = T(1, N+1) / Q(1, N+1)` exactly using integer
//! arithmetic, where
//!
//! * `p_k = -(6k - 5)(6k - 1)(2k - 1)`
//! * `q_k = k³ · C³ / 24`
//! * `L_k = A + B·k`
//! * `M_k = M_{k-1} · p_k / q_k`,   `M_0 = 1`
//!
//! Including the `k = 0` term (which is just `A`) gives
//! `S = (A · Q + T) / Q`, so
//!
//! ```text
//! π = (426_880 · √10005 · Q) / (A · Q + T)
//! ```
//!
//! The integers `P`, `Q`, `T` grow to roughly `D` decimal digits each by
//! the top of the recursion, so the dominant cost is a single GMP
//! multiplication of two `D`-digit numbers — O(M(D) · log D) total.

use anyhow::Result;
use bignum::{Float, Integer, Pow};

use crate::output::DigitSink;
use crate::precision::PrecisionPlan;
use crate::progress::{Phase, ProgressReporter};

use super::util::write_decimal_digits;
use super::PiAlgorithm;

/// Decimal digits added per Chudnovsky term (≈ log10(C³ / 24) / 3).
pub const DIGITS_PER_TERM: f64 = 14.181_647_462_725_478;

/// Extra series terms beyond the asymptotic estimate, to absorb the
/// discrete rounding when converting from a per-term digit count to an
/// integer count.
const SAFETY_TERMS: u64 = 8;

const A: u32 = 13_591_409;
const B: u32 = 545_140_134;
// C = 640_320.  We only ever need C³ / 24 = 10_939_058_860_032_000 as a
// constant.
const C3_OVER_24: u64 = 10_939_058_860_032_000;

const PHASE_BINARY_SPLITTING: &str = "binary splitting";
const PHASE_FINAL_ASSEMBLY: &str = "final assembly";
const PHASE_DECIMAL_CONVERSION: &str = "decimal conversion";

#[derive(Default, Debug, Clone, Copy)]
pub struct Chudnovsky;

impl PiAlgorithm for Chudnovsky {
    fn name(&self) -> &'static str {
        "chudnovsky"
    }

    fn compute(
        &self,
        digits: u64,
        sink: &mut dyn DigitSink,
        progress: &mut dyn ProgressReporter,
    ) -> Result<()> {
        if digits == 0 {
            sink.finish()?;
            return Ok(());
        }

        let plan = PrecisionPlan::for_digits(digits)?;

        // Number of binary-splitting terms.  Each Chudnovsky term
        // contributes `DIGITS_PER_TERM` digits; round up and add a
        // small safety margin for the discrete-to-continuous gap.
        let n_terms = (digits as f64 / DIGITS_PER_TERM).ceil() as u64 + SAFETY_TERMS;

        progress.set_phases(&[
            Phase { name: PHASE_BINARY_SPLITTING, total: n_terms },
            Phase { name: PHASE_FINAL_ASSEMBLY, total: 4 },
            Phase { name: PHASE_DECIMAL_CONVERSION, total: 1 },
        ]);

        // Wrap the binary-splitting + final-assembly work in its own
        // scope so the multi-gigabyte `_p`, `q`, `t`, and `denom_int`
        // get dropped before the decimal-conversion phase allocates
        // its own large buffers.  At 10B digits this single change
        // saves ~16 GB of peak memory.
        let pi = {
            progress.start_phase(PHASE_BINARY_SPLITTING, n_terms);
            let (_p, q, t) = binary_split(1, n_terms + 1, progress);
            progress.end_phase();

            progress.start_phase(PHASE_FINAL_ASSEMBLY, 4);
            // S = (A · Q + T) / Q, where T/Q sums terms k = 1..N and A
            // is the k = 0 contribution.
            let denom_int = Integer::from(A) * &q + &t;
            progress.tick();

            let mut pi = Float::with_val_64(plan.precision_bits, 10_005);
            pi.sqrt_mut();
            progress.tick();

            pi *= 426_880_u32;
            pi *= &q;
            progress.tick();

            pi /= &denom_int;
            progress.tick();
            progress.end_phase();
            pi
            // _p, q, t, denom_int all dropped here.
        };

        progress.start_phase(PHASE_DECIMAL_CONVERSION, 1);
        write_decimal_digits(pi, digits, sink)?;
        progress.tick();
        progress.end_phase();

        Ok(())
    }
}

/// Binary splitting over the half-open range `[a, b)` of Chudnovsky terms.
///
/// Returns `(P, Q, T)` such that, with `M_0 = 1`,
/// `Σ_{k=a}^{b-1} M_k L_k = (M_{a-1} · T) / Q`.
///
/// The top-level caller invokes this with `a = 1`, so `M_{a-1} = M_0 = 1`
/// and the partial sum is exactly `T / Q`.
fn binary_split(
    a: u64,
    b: u64,
    progress: &mut dyn ProgressReporter,
) -> (Integer, Integer, Integer) {
    let result = binary_split_pure(a, b);
    // Ticks happen at leaves; with parallel execution we can't easily
    // ferry per-leaf progress out of the rayon scope.  Batch the tick
    // count: bar stays empty during the parallel section, then jumps
    // to 100% when this returns.  Acceptable trade for the speedup —
    // binary-split is no longer the slow phase anyway.
    for _ in a..b {
        progress.tick();
    }
    result
}

/// Subtree sizes below this run sequentially.  Above it, the two halves
/// of the range are forked via `rayon::join` so the recursion exposes
/// parallelism at every level until the granularity shrinks below the
/// per-task overhead.  64 ≈ 6 levels of doubling above the leaves.
const PARALLEL_SPLIT_THRESHOLD: u64 = 64;

/// Pure (no-progress, `Send`-safe) variant of binary splitting.  Spawns
/// `rayon::join` for the two recursive calls once the range is large
/// enough that the rayon task overhead is amortized.
fn binary_split_pure(a: u64, b: u64) -> (Integer, Integer, Integer) {
    debug_assert!(a < b, "binary_split called with empty/reversed range [{a}, {b})");
    if b - a == 1 {
        let k = Integer::from(a);
        let six_k = Integer::from(&k * 6_u32);
        // p_k = -(6k - 5)(6k - 1)(2k - 1).
        let factor1: Integer = Integer::from(&six_k - 5_u32);
        let factor2: Integer = Integer::from(&six_k - 1_u32);
        let factor3: Integer = Integer::from(&k * 2_u32) - 1_u32;
        let product: Integer = factor1 * factor2 * factor3;
        let p: Integer = -product;
        // q_k = k³ · (C³ / 24)
        let q: Integer = Integer::from(C3_OVER_24) * k.clone().pow(3_u32);
        // t_k = p_k · (A + B·k)
        let l: Integer = Integer::from(A) + Integer::from(&k * B);
        let t: Integer = Integer::from(&p * &l);
        (p, q, t)
    } else {
        let m = (a + b) / 2;
        let ((p_l, q_l, t_l), (p_r, q_r, t_r)) = if b - a >= PARALLEL_SPLIT_THRESHOLD {
            rayon::join(
                || binary_split_pure(a, m),
                || binary_split_pure(m, b),
            )
        } else {
            (binary_split_pure(a, m), binary_split_pure(m, b))
        };
        // Combine.  Order matters: we need to use `&q_r` and `&p_l` to
        // build `t` before consuming them in `p` and `q`.
        let t = t_l * &q_r + &p_l * t_r;
        let p = p_l * p_r;
        let q = q_l * q_r;
        (p, q, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::util::test_support::{StringSink, FIRST_100};
    use crate::progress::NoopProgress;

    #[test]
    fn one_digit() {
        let mut sink = StringSink::new();
        Chudnovsky.compute(1, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, "3");
    }

    #[test]
    fn fifty_digits() {
        let mut sink = StringSink::new();
        Chudnovsky.compute(50, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, &FIRST_100[..51]);
    }

    #[test]
    fn one_hundred_digits() {
        let mut sink = StringSink::new();
        Chudnovsky.compute(100, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, FIRST_100);
    }

    #[test]
    fn one_thousand_digits_match_known_prefix() {
        let mut sink = StringSink::new();
        Chudnovsky.compute(1_000, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out.len(), 1_001);
        assert_eq!(&sink.out[..101], FIRST_100);
    }
}
