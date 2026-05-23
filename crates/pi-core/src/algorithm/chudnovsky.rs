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

use anyhow::{anyhow, Result};
use rug::float::Round;
use rug::ops::{Pow, PowAssign};
use rug::{Float, Integer};

use crate::output::DigitSink;
use crate::precision::PrecisionPlan;
use crate::progress::ProgressReporter;

use super::PiAlgorithm;

/// Decimal digits added per Chudnovsky term (≈ log10(C³ / 24) / 3).
pub const DIGITS_PER_TERM: f64 = 14.181_647_462_725_478;

const A: u32 = 13_591_409;
const B: u32 = 545_140_134;
// C = 640_320.  We only ever need C³ / 24 = 10_939_058_860_032_000 as a
// constant.
const C3_OVER_24: u64 = 10_939_058_860_032_000;

#[derive(Default, Debug, Clone, Copy)]
pub struct Chudnovsky;

impl PiAlgorithm for Chudnovsky {
    fn name(&self) -> &'static str {
        "chudnovsky"
    }

    fn digits_per_term(&self) -> f64 {
        DIGITS_PER_TERM
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

        let plan = PrecisionPlan::for_digits(digits, DIGITS_PER_TERM);

        progress.start_phase("binary splitting", plan.terms);
        let (_p, q, t) = binary_split(1, plan.terms + 1, progress);
        progress.end_phase();

        progress.start_phase("final assembly", 4);
        // S = (A · Q + T) / Q, where T/Q sums terms k = 1..N and A is the
        // k = 0 contribution.
        let denom_int = Integer::from(A) * &q + &t;
        progress.tick();

        let mut pi = Float::with_val(plan.precision_bits, 10_005);
        pi.sqrt_mut();
        progress.tick();

        pi *= 426_880_u32;
        pi *= &q;
        progress.tick();

        pi /= &denom_int;
        progress.tick();
        progress.end_phase();

        progress.start_phase("decimal conversion", 1);
        write_decimal_digits(&pi, digits, sink)?;
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
    if b - a == 1 {
        let k = Integer::from(a);
        let six_k = Integer::from(&k * 6_u32);
        // p_k = -(6k - 5)(6k - 1)(2k - 1).  Literal suffixes are needed
        // because `rug` provides `Sub<T> for &Integer` for many primitive T
        // and the inference can't pick one without a hint.
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
        progress.tick();
        (p, q, t)
    } else {
        let m = (a + b) / 2;
        let (p_l, q_l, t_l) = binary_split(a, m, progress);
        let (p_r, q_r, t_r) = binary_split(m, b, progress);
        // Combine.  Order matters: we need to use `&q_r` and `&p_l` to
        // build `t` before consuming them in `p` and `q`.
        let t = t_l * &q_r + &p_l * t_r;
        let p = p_l * p_r;
        let q = q_l * q_r;
        (p, q, t)
    }
}

/// Render `pi` to `digits` decimal digits and stream them to `sink`.
fn write_decimal_digits(pi: &Float, digits: u64, sink: &mut dyn DigitSink) -> Result<()> {
    // Multiply by 10^(digits - 1) so the truncated value is an integer with
    // exactly `digits` decimal digits.  The working precision (see
    // `PrecisionPlan`) is wide enough to represent 10^(digits - 1) exactly,
    // so the multiplication is lossless to many more bits than we need.
    let prec = pi.prec();
    let exp = Integer::from(digits) - 1_u32;
    let mut scale = Float::with_val(prec, 10);
    scale.pow_assign(&exp);
    let scaled = Float::with_val(prec, pi * &scale);

    // Truncate (round toward -∞ — for positive pi the same as floor and
    // the same as `trunc`).  `Float::to_integer` rounds to nearest, which
    // would give the wrong answer for "the first N digits of pi" whenever
    // the digit just past the cut is ≥ 5.
    let (int_part, _ord) = scaled
        .to_integer_round(Round::Down)
        .ok_or_else(|| anyhow!("pi scaled to integer was non-finite"))?;

    let mut s = int_part.to_string();
    let want = digits as usize;
    match s.len().cmp(&want) {
        std::cmp::Ordering::Less => {
            // Should not happen for pi (always starts with "3"), but pad
            // defensively to keep the contract clear for future algorithms.
            let pad = want - s.len();
            s = "0".repeat(pad) + &s;
        }
        std::cmp::Ordering::Greater => s.truncate(want),
        std::cmp::Ordering::Equal => {}
    }

    sink.write_integer_part(&s[..1])?;
    if digits > 1 {
        sink.write_fractional_digits(&s[1..])?;
    }
    sink.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::DigitSink;
    use crate::progress::NoopProgress;
    use std::io;

    struct StringSink {
        out: String,
        wrote_dot: bool,
    }

    impl StringSink {
        fn new() -> Self {
            Self { out: String::new(), wrote_dot: false }
        }
    }

    impl DigitSink for StringSink {
        fn write_integer_part(&mut self, digits: &str) -> io::Result<()> {
            self.out.push_str(digits);
            Ok(())
        }
        fn write_fractional_digits(&mut self, digits: &str) -> io::Result<()> {
            if !self.wrote_dot {
                self.out.push('.');
                self.wrote_dot = true;
            }
            self.out.push_str(digits);
            Ok(())
        }
        fn finish(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // First 100 digits of pi, counting the leading `3` (so 99 fractional
    // digits, ending at the `7` that comes before the `9` at fractional
    // position 100).
    const FIRST_100: &str = "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067";

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
