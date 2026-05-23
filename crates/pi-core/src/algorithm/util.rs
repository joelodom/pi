//! Shared low-level helpers used by every algorithm in this module.
//!
//! Lives here, not inside any single algorithm file, because both
//! Chudnovsky and Gauss-Legendre (and any future algorithm) need to
//! widen MPFR's exponent range before they start producing huge
//! intermediate magnitudes, and both convert the same final `Float` pi
//! value into a stream of decimal digits.

use anyhow::{anyhow, bail, Result};
use rug::float::Round;
use rug::ops::PowAssign;
use rug::{Float, Integer};

use crate::output::DigitSink;

/// Push MPFR's allowed exponent range out to its hardware maximum so
/// the O(10^D)-magnitude intermediates inside the algorithms don't
/// trigger silent overflow to +∞.  The change is process-global, but a
/// widened range is harmless for any Float with a small magnitude.
pub(super) fn widen_mpfr_exponent_range_for(digits: u64) -> Result<()> {
    use gmp_mpfr_sys::mpfr;
    use std::f64::consts::LOG2_10;

    // Conservative estimate of the largest binary exponent we'll need:
    // log2(10) · D, plus generous margin for intermediate products.
    let needed_exp = (digits as f64 * LOG2_10).ceil() as i128 + 1024;

    // Safety: every gmp_mpfr_sys call here only reads or writes MPFR's
    // global emin/emax through the documented MPFR API.  None of them
    // dereferences any pointer we hand in.
    unsafe {
        let emax_max = mpfr::get_emax_max();
        let emin_min = mpfr::get_emin_min();
        if needed_exp > emax_max as i128 {
            bail!(
                "{} digits requires an MPFR exponent up to {}, exceeding the platform cap ({})",
                digits,
                needed_exp,
                emax_max
            );
        }
        if mpfr::set_emax(emax_max) != 0 {
            bail!("failed to widen MPFR emax to {}", emax_max);
        }
        if mpfr::set_emin(emin_min) != 0 {
            bail!("failed to widen MPFR emin to {}", emin_min);
        }
    }
    Ok(())
}

/// Render `pi` to exactly `digits` decimal digits (counting the leading
/// `3`) and stream them through `sink`.
///
/// Takes `pi` **by value** and operates on it in place so that the
/// scale Float, the scaled Float, and `pi` itself are not all in memory
/// at once.  At a billion-plus digits each of those is multi-gigabyte;
/// the explicit `drop`s along the way claw back ~2 full-precision
/// Floats compared to the obvious "create a new scaled value" version.
///
/// `pi` must already hold the value of π to enough working precision
/// (see [`crate::precision::PrecisionPlan`]) that all `digits` decimal
/// digits are correct.
pub(super) fn write_decimal_digits(
    mut pi: Float,
    digits: u64,
    sink: &mut dyn DigitSink,
) -> Result<()> {
    // Build the scale 10^(digits - 1) at the same working precision.
    let prec = pi.prec_64();
    let mut scale = Float::with_val_64(prec, 10);
    let exp = Integer::from(digits) - 1_u32;
    scale.pow_assign(&exp);
    drop(exp);

    // Scale `pi` in place — `pi *= &scale` reuses pi's mantissa
    // allocation, then we immediately drop `scale` (full-precision
    // Float, multi-GB at scale).
    pi *= &scale;
    drop(scale);

    // Truncate (round toward -∞ — for positive pi the same as floor and
    // the same as `trunc`).  Float::to_integer rounds to nearest, which
    // would give the wrong answer for "the first N digits of pi"
    // whenever the digit just past the cut is ≥ 5.
    let (int_part, _ord) = pi
        .to_integer_round(Round::Down)
        .ok_or_else(|| anyhow!("pi scaled to integer was non-finite"))?;
    drop(pi);

    let mut s = int_part.to_string();
    drop(int_part);

    let want = digits as usize;
    match s.len().cmp(&want) {
        std::cmp::Ordering::Less => {
            // Shouldn't happen for pi (always starts with "3"), but
            // pad defensively to keep the contract clear for future
            // algorithms whose first digit could be zero.
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
pub(super) mod test_support {
    use crate::output::DigitSink;
    use std::io;

    /// Sink that just collects whatever bytes the algorithm emits into a
    /// single `String`, with the decimal point reinserted on the first
    /// fractional chunk.  Convenient for unit tests.
    pub struct StringSink {
        pub out: String,
        wrote_dot: bool,
    }

    impl StringSink {
        pub fn new() -> Self {
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

    /// First 100 digits of pi (one integer digit + 99 fractional digits,
    /// ending at the `7` that comes before the `9` at fractional position
    /// 100).  Used by every algorithm's test suite as the canonical
    /// reference.
    pub const FIRST_100: &str =
        "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067";
}
