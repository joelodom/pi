//! Gauss-Legendre / Brent-Salamin algorithm.
//!
//! Quadratically-convergent AGM iteration.  Starting from
//!
//! ```text
//! a₀ = 1            b₀ = 1/√2            t₀ = 1/4            p₀ = 1
//! ```
//!
//! repeat
//!
//! ```text
//! aₙ₊₁ = (aₙ + bₙ) / 2
//! bₙ₊₁ = √(aₙ · bₙ)
//! tₙ₊₁ = tₙ − pₙ · (aₙ − aₙ₊₁)²
//! pₙ₊₁ = 2 · pₙ
//! ```
//!
//! and after enough iterations
//!
//! ```text
//! π ≈ (aₙ + bₙ)² / (4 · tₙ).
//! ```
//!
//! Convergence is quadratic, so the number of correct binary digits
//! roughly doubles per iteration.  To get the working precision
//! `P` bits we therefore need ≈ ⌈log₂ P⌉ + small safety iterations.
//!
//! This is implemented primarily as an *independent* second algorithm
//! to cross-check Chudnovsky's output: any bug specific to one (formula
//! constants, the binary-splitting recurrence, the AGM iteration logic)
//! cannot survive a byte-by-byte agreement on D digits with the other.

use anyhow::Result;
use bignum::{Float, Integer};

use crate::output::DigitSink;
use crate::precision::PrecisionPlan;
use crate::progress::{Phase, ProgressReporter};

use super::util::write_decimal_digits;
use super::PiAlgorithm;

const PHASE_INIT: &str = "initialization";
const PHASE_ITERATIONS: &str = "agm iterations";
const PHASE_FINAL_ASSEMBLY: &str = "final assembly";
const PHASE_DECIMAL_CONVERSION: &str = "decimal conversion";

/// Extra iterations past `ceil(log2(precision_bits))`.  Each iteration
/// after convergence is wasted work and starts losing precision in the
/// `(a − a')²` subtraction, so the safety here is intentionally small.
const SAFETY_ITERATIONS: u32 = 2;

#[derive(Default, Debug, Clone, Copy)]
pub struct GaussLegendre;

impl PiAlgorithm for GaussLegendre {
    fn name(&self) -> &'static str {
        "gauss-legendre"
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

        // Iterations needed to reach `precision_bits` correct binary
        // digits, given quadratic convergence (one bit at iteration 0
        // doubles to `2^n` bits after n iterations).
        let n_iterations =
            (plan.precision_bits as f64).log2().ceil() as u32 + SAFETY_ITERATIONS;
        let prec = plan.precision_bits;

        progress.set_phases(&[
            Phase { name: PHASE_INIT, total: 1 },
            Phase { name: PHASE_ITERATIONS, total: n_iterations as u64 },
            Phase { name: PHASE_FINAL_ASSEMBLY, total: 1 },
            Phase { name: PHASE_DECIMAL_CONVERSION, total: 1 },
        ]);

        // Run initialization, iterations, and final assembly inside a
        // scope so `a`, `b`, `t`, `p`, and the per-iteration scratch
        // Floats all get dropped before the decimal-conversion phase
        // allocates its own large buffers.  At billion-plus digits each
        // Float is multi-GB, so this scope plus the pre-allocated
        // scratch (avoiding 4 fresh Float allocations per iteration)
        // are what let the working set stay below the binary-splitting
        // peak instead of stacking on top of it.
        let pi = {
            // --- Initialization -------------------------------------------
            progress.start_phase(PHASE_INIT, 1);
            let mut a = Float::with_val_64(prec, 1);
            // b = 1/√2 = √(1/2).  One sqrt is cheaper than sqrt then reciprocal.
            let mut b = Float::with_val_64(prec, 0.5_f64);
            b.sqrt_mut();
            let mut t = Float::with_val_64(prec, 0.25_f64);
            let mut p = Integer::from(1);
            // Pre-allocated scratch Floats reused every iteration.  At
            // 10B digits each Float is ~4 GB, so 4 scratch Floats are
            // ~16 GB of live mantissa per iteration; without reuse we'd
            // be allocating and freeing all 16 GB every step (~512 GB of
            // cumulative allocator traffic across ~32 iterations).
            let mut a_new = Float::with_val_64(prec, 0);
            let mut b_new = Float::with_val_64(prec, 0);
            let mut diff = Float::with_val_64(prec, 0);
            let mut diff_sq = Float::with_val_64(prec, 0);
            progress.tick();
            progress.end_phase();

            // --- AGM iterations -------------------------------------------
            progress.start_phase(PHASE_ITERATIONS, n_iterations as u64);
            for _ in 0..n_iterations {
                // a_new = (a + b) / 2
                a_new.assign(&a + &b);
                a_new /= 2_u32;

                // b_new = √(a · b)
                b_new.assign(&a * &b);
                b_new.sqrt_mut();

                // t -= p · (a − a_new)²
                diff.assign(&a - &a_new);
                diff_sq.assign(&diff * &diff);
                diff_sq *= &p;
                t -= &diff_sq;

                // Promote a_new → a (and similarly b_new → b) by
                // swapping — the previous `a` (now in `a_new`) gets
                // overwritten by the next iteration's `assign`, so no
                // copy is needed.
                std::mem::swap(&mut a, &mut a_new);
                std::mem::swap(&mut b, &mut b_new);
                p <<= 1_u32;

                progress.tick();
            }
            progress.end_phase();

            // --- Final assembly: π = (a + b)² / (4 · t) -------------------
            progress.start_phase(PHASE_FINAL_ASSEMBLY, 1);
            let mut pi = Float::with_val_64(prec, &a + &b);
            pi.square_mut();
            pi /= &t;
            pi /= 4_u32;
            progress.tick();
            progress.end_phase();
            pi
            // a, b, t, p, a_new, b_new, diff, diff_sq all dropped here.
        };

        // --- Decimal conversion --------------------------------------------
        progress.start_phase(PHASE_DECIMAL_CONVERSION, 1);
        write_decimal_digits(pi, digits, sink)?;
        progress.tick();
        progress.end_phase();

        Ok(())
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
        GaussLegendre.compute(1, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, "3");
    }

    #[test]
    fn fifty_digits() {
        let mut sink = StringSink::new();
        GaussLegendre.compute(50, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, &FIRST_100[..51]);
    }

    #[test]
    fn one_hundred_digits() {
        let mut sink = StringSink::new();
        GaussLegendre.compute(100, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, FIRST_100);
    }

    #[test]
    fn one_thousand_digits_match_known_prefix() {
        let mut sink = StringSink::new();
        GaussLegendre.compute(1_000, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out.len(), 1_001);
        assert_eq!(&sink.out[..101], FIRST_100);
    }
}
