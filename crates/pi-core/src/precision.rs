//! Precision planning.
//!
//! Decides how many series terms and how many bits of working precision to
//! use, given a target number of decimal digits.  Errors out explicitly if
//! the request would push the f64-based planning arithmetic, or rug/MPFR
//! itself, past the point where the resulting computation could be trusted.

use std::f64::consts::LOG2_10;

use anyhow::{bail, Result};

/// Bits of slack added to working precision.  Generous — the final scaled
/// multiplication and a handful of GMP roundings each lose only O(1) bits.
const SAFETY_BITS: u64 = 256;

/// Extra series terms beyond the asymptotic estimate, to absorb the
/// discrete rounding when converting from a per-term digit count to an
/// integer count.
const SAFETY_TERMS: u64 = 8;

/// Largest digit count we'll accept.
///
/// `digits as f64` is exact for values up to 2^53 (~9·10^15) — beyond
/// that, the planning arithmetic loses precision through f64 rounding,
/// which would silently produce a plan with the wrong number of terms or
/// the wrong precision.  Bail explicitly instead.  (This ceiling is
/// vastly larger than anything that fits on real hardware in any case.)
pub const MAX_DIGITS: u64 = 1 << 53;

/// Planning output for a single pi computation.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionPlan {
    /// Target decimal digits of pi (counting the leading `3`).
    pub digits: u64,
    /// Number of series terms to evaluate.
    pub terms: u64,
    /// Working precision (mantissa bits) for [`rug::Float`] values.
    pub precision_bits: u64,
}

impl PrecisionPlan {
    /// Build a plan for `digits` decimal digits, where each series term
    /// contributes roughly `digits_per_term` decimal digits.
    ///
    /// Returns an error if `digits` is too large for the planning
    /// arithmetic to remain exact, or if the resulting working precision
    /// would exceed MPFR's `prec_max_64()`.
    pub fn for_digits(digits: u64, digits_per_term: f64) -> Result<Self> {
        // The compute pipeline short-circuits on `digits == 0`, but
        // produce a well-formed plan anyway so callers don't have to
        // special-case it.
        if digits == 0 {
            return Ok(Self { digits: 0, terms: 0, precision_bits: 0 });
        }

        if digits > MAX_DIGITS {
            bail!(
                "{} digits exceeds the supported maximum ({}); larger D would lose precision \
                 in the f64-based planning arithmetic",
                digits,
                MAX_DIGITS
            );
        }

        // `digits as f64` is exact in this range; the arithmetic below
        // stays well inside f64's exponent range too.
        let terms = (digits as f64 / digits_per_term).ceil() as u64 + SAFETY_TERMS;
        let precision_bits = (digits as f64 * LOG2_10).ceil() as u64 + SAFETY_BITS;

        let mpfr_cap = rug::float::prec_max_64();
        if precision_bits > mpfr_cap {
            bail!(
                "{} digits requires {} bits of working precision, which exceeds MPFR's cap ({})",
                digits,
                precision_bits,
                mpfr_cap
            );
        }

        Ok(Self { digits, terms, precision_bits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGITS_PER_TERM: f64 = 14.181_647_462_725_478;

    #[test]
    fn zero_digits_gives_empty_plan() {
        let plan = PrecisionPlan::for_digits(0, DIGITS_PER_TERM).unwrap();
        assert_eq!(plan.digits, 0);
        assert_eq!(plan.terms, 0);
        assert_eq!(plan.precision_bits, 0);
    }

    #[test]
    fn one_million_digits_fits_in_u64_precision() {
        let plan = PrecisionPlan::for_digits(1_000_000, DIGITS_PER_TERM).unwrap();
        assert!(plan.precision_bits > 3_321_928); // > 1M * log2(10)
        assert!(plan.precision_bits < 3_322_500);
        assert!(plan.terms > 70_500);
        assert!(plan.terms < 70_600);
    }

    #[test]
    fn ten_billion_digits_planned_without_u32_overflow() {
        // Before u64: this would saturate `as u32` and silently truncate
        // precision_bits to u32::MAX, producing wrong digits.  Now it
        // just works.
        let plan = PrecisionPlan::for_digits(10_000_000_000, DIGITS_PER_TERM).unwrap();
        assert!(plan.precision_bits > 33_219_280_000); // > 10B * log2(10)
        assert!(plan.precision_bits as u128 > u32::MAX as u128);
    }

    #[test]
    fn too_many_digits_is_a_clean_error() {
        let err = PrecisionPlan::for_digits(MAX_DIGITS + 1, DIGITS_PER_TERM).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("exceeds the supported maximum"), "got: {msg}");
    }
}
