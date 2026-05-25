//! Shared low-level helpers used by every algorithm in this module.
//!
//! Lives here, not inside any single algorithm file, because both
//! Chudnovsky and Gauss-Legendre (and any future algorithm) convert
//! the same final `Float` pi value into a stream of decimal digits.

use anyhow::{anyhow, Result};
use bignum::{Float, Integer, PowAssign, Round};

use crate::output::DigitSink;
use crate::progress::ProgressReporter;

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
    progress: &mut dyn ProgressReporter,
) -> Result<()> {
    // Build the scale 10^(digits - 1) at the same working precision.
    progress.milestone("dc.scale_build.start");
    let prec = pi.prec_64();
    let mut scale = Float::with_val_64(prec, 10);
    let exp = Integer::from(digits) - 1_u32;
    scale.pow_assign(&exp);
    drop(exp);
    progress.milestone("dc.scale_build.end");

    // Scale `pi` in place — `pi *= &scale` reuses pi's mantissa
    // allocation, then we immediately drop `scale` (full-precision
    // Float, multi-GB at scale).
    progress.milestone("dc.scale_mul.start");
    pi *= &scale;
    drop(scale);
    progress.milestone("dc.scale_mul.end");

    // Truncate (round toward -∞ — for positive pi the same as floor and
    // the same as `trunc`).  Float::to_integer rounds to nearest, which
    // would give the wrong answer for "the first N digits of pi"
    // whenever the digit just past the cut is ≥ 5.
    progress.milestone("dc.to_integer.start");
    let (int_part, _ord) = pi
        .to_integer_round(Round::Down)
        .ok_or_else(|| anyhow!("pi scaled to integer was non-finite"))?;
    drop(pi);
    progress.milestone("dc.to_integer.end");

    progress.milestone("dc.to_string.start");
    let mut s = int_part.to_string();
    drop(int_part);
    progress.milestone("dc.to_string.end");

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
