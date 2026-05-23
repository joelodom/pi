//! Precision planning.
//!
//! Decides how many series terms and how many bits of working precision to
//! use, given a target number of decimal digits.

use std::f64::consts::LOG2_10;

/// Bits of slack added to working precision.  Generous — the final scaled
/// multiplication and the handful of GMP roundings only lose O(1) bits each.
const SAFETY_BITS: u32 = 256;

/// Extra series terms beyond the asymptotic estimate, to absorb the discrete
/// rounding when converting from a per-term digit count to an integer count.
const SAFETY_TERMS: u64 = 8;

/// Planning output for a single pi computation.
#[derive(Debug, Clone, Copy)]
pub struct PrecisionPlan {
    /// Target decimal digits of pi (counting the leading `3`).
    pub digits: u64,
    /// Number of series terms to evaluate.
    pub terms: u64,
    /// Working precision (mantissa bits) for [`rug::Float`] values.
    pub precision_bits: u32,
}

impl PrecisionPlan {
    pub fn for_digits(digits: u64, digits_per_term: f64) -> Self {
        let terms = (digits as f64 / digits_per_term).ceil() as u64 + SAFETY_TERMS;
        let precision_bits = (digits as f64 * LOG2_10).ceil() as u32 + SAFETY_BITS;
        Self { digits, terms, precision_bits }
    }
}
