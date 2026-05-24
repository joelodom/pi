//! Precision planning.
//!
//! Computes how many bits of working precision a pi computation needs to
//! produce a chosen number of decimal digits.  Errors out explicitly if
//! the request would push the f64-based planning arithmetic, or
//! rug/MPFR itself, past the point where the resulting computation
//! could be trusted.
//!
//! Series-vs-iteration term counts are algorithm-specific and live in
//! the algorithm modules, not here.

use std::f64::consts::LOG2_10;

use anyhow::{bail, Result};

/// Bits of slack added to working precision.  Generous — the final
/// scaled multiplication and the handful of GMP roundings each lose
/// only O(1) bits.
const SAFETY_BITS: u64 = 256;

/// Largest digit count we'll accept.
///
/// `digits as f64` is exact for values up to 2^53 (≈9·10^15); beyond
/// that, the planning arithmetic loses precision through f64 rounding
/// and would silently produce a plan with the wrong precision.  Bail
/// explicitly instead.  (This ceiling is vastly larger than anything
/// that fits on real hardware anyway.)
pub const MAX_DIGITS: u64 = 1 << 53;

/// Planning output for a single pi computation.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionPlan {
    /// Target decimal digits of pi (counting the leading `3`).
    pub digits: u64,
    /// Working precision (mantissa bits) for [`rug::Float`] values.
    pub precision_bits: u64,
}

impl PrecisionPlan {
    /// Build a plan for `digits` decimal digits.
    ///
    /// Returns an error if `digits` is too large for the planning
    /// arithmetic to remain exact, or if the resulting working
    /// precision would exceed MPFR's `prec_max_64()`.
    pub fn for_digits(digits: u64) -> Result<Self> {
        // The compute pipeline short-circuits on `digits == 0`, but
        // produce a well-formed plan anyway so callers don't have to
        // special-case it.
        if digits == 0 {
            return Ok(Self { digits: 0, precision_bits: 0 });
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
        let precision_bits = (digits as f64 * LOG2_10).ceil() as u64 + SAFETY_BITS;

        let mpfr_cap = bignum::float::prec_max_64();
        if precision_bits > mpfr_cap {
            bail!(
                "{} digits requires {} bits of working precision, which exceeds MPFR's cap ({})",
                digits,
                precision_bits,
                mpfr_cap
            );
        }

        Ok(Self { digits, precision_bits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_digits_gives_empty_plan() {
        let plan = PrecisionPlan::for_digits(0).unwrap();
        assert_eq!(plan.digits, 0);
        assert_eq!(plan.precision_bits, 0);
    }

    #[test]
    fn one_million_digits_fits_in_u64_precision() {
        let plan = PrecisionPlan::for_digits(1_000_000).unwrap();
        assert!(plan.precision_bits > 3_321_928); // > 1M * log2(10)
        assert!(plan.precision_bits < 3_322_500);
    }

    #[test]
    fn ten_billion_digits_planned_without_u32_overflow() {
        // Before u64: this would saturate `as u32` and silently truncate
        // precision_bits to u32::MAX, producing wrong digits.  Now it
        // just works.
        let plan = PrecisionPlan::for_digits(10_000_000_000).unwrap();
        assert!(plan.precision_bits > 33_219_280_000); // > 10B * log2(10)
        assert!(plan.precision_bits as u128 > u32::MAX as u128);
    }

    #[test]
    fn too_many_digits_is_a_clean_error() {
        let err = PrecisionPlan::for_digits(MAX_DIGITS + 1).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("exceeds the supported maximum"), "got: {msg}");
    }
}
