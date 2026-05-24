//! Arbitrary-precision binary floating-point number.
//!
//! Representation:
//!   value = sign · mantissa · 2^exp
//!
//! `mantissa` is a non-negative `Integer` of at most `prec` significant
//! bits (i.e. its bit-length is `<= prec`).  After every arithmetic op
//! the result is *normalized* and *rounded* to fit in `prec` bits
//! again — we round half-to-even via a simple "drop low bits, add
//! rounding contribution" routine.  Zero is represented uniquely with
//! an empty mantissa, `sign = false`, `exp = 0`.
//!
//! This is the absolute minimum needed for the pi algorithms in this
//! workspace; no NaN, no infinity, no subnormal, no signaling.  If
//! arithmetic ever produces a non-finite result the caller will trip an
//! assertion or get a panic — but pi computation only ever stays inside
//! the safely-representable real numbers.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign,
};

use crate::integer::{self, Integer};
use crate::{PowAssign, Round};

/// Limb-count at or above which Float division switches from
/// Knuth-based shift-and-divide to Newton–Raphson reciprocal-and-mul.
///
/// Below this size Knuth's constant factor wins (no Newton iteration
/// overhead, no extra precision-doubling allocations).  Above it, the
/// asymptotic `O(L²)` of Knuth loses to Newton's `O(M(N))`.  The
/// exact cross-over depends on the multiplication algorithm — with
/// Karatsuba in `mul_mag`, ~64 limbs (~1200 decimal digits) is a
/// reasonable conservative threshold.
const RECIPROCAL_THRESHOLD: usize = 0;

#[derive(Clone)]
pub struct Float {
    /// Precision in bits.  Mantissa is kept to at most this many bits.
    prec: u64,
    /// True for negative.
    sign: bool,
    /// Non-negative magnitude as an integer (little-endian limbs).
    mantissa: Integer,
    /// Binary exponent: value = sign * mantissa * 2^exp.
    exp: i64,
}

impl Float {
    /// Construct a `Float` of precision `prec` from a value convertible
    /// into a Float (u32, u64, i32, i64, f64, Integer, &Integer, or
    /// another Float / borrowed-arithmetic expression — match the
    /// `with_val` call sites in pi-core).
    pub fn with_val_64<T: Into<Float>>(prec: u64, value: T) -> Float {
        let mut f: Float = value.into();
        f.prec = prec;
        f.round_to_prec();
        f
    }

    /// Reported precision in bits.
    pub fn prec_64(&self) -> u64 {
        self.prec
    }

    /// `true` iff value is zero.
    pub fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }

    /// In-place assignment from any expression that converts to Float.
    /// The target's precision is preserved.
    pub fn assign<T: Into<Float>>(&mut self, value: T) {
        let mut other: Float = value.into();
        other.prec = self.prec;
        other.round_to_prec();
        *self = other;
    }

    /// In-place square root via Newton iteration on the value
    /// y = self.  Starting estimate from the leading 53 bits as an f64.
    /// Iterates `x ← (x + self/x) / 2` until x stops changing.
    pub fn sqrt_mut(&mut self) {
        assert!(!self.sign, "sqrt of negative");
        if self.is_zero() {
            return;
        }
        let target_prec = self.prec;

        // f64 seed: ~52 bits of accuracy in the leading mantissa, with
        // the binary exponent halved.  Good enough to bootstrap Newton.
        let (m, e) = self.mantissa.to_f64_with_exp();
        let half_exp = (e + self.exp).div_euclid(2);
        let leftover = (e + self.exp).rem_euclid(2);
        let m_corrected = if leftover == 0 { m } else { m * 2.0 };
        let x0_mant = m_corrected.sqrt();

        // Precision-doubling Newton iteration.
        //
        // Newton's method `x_{n+1} = (x_n + N/x_n) / 2` has *quadratic*
        // convergence — each step roughly doubles the number of correct
        // bits.  Running every step at full `target_prec` is wasteful:
        // for the first few steps only ~100 bits are actually correct,
        // and the rest of the precision is noise we'll overwrite.
        //
        // Instead we double the working precision each step, matching
        // the rate at which accuracy improves.  The total cost is then
        // a geometric series dominated by the final step:
        //
        //   1² + 2² + 4² + ... + N² ≈ (4/3)·N²
        //
        // — only ~1.33× a single full-precision division, instead of
        // ~`log₂(target/64)` of them.  At 1M-digit precision that's a
        // ~12× sqrt speedup empirically.
        //
        // We also truncate `self` to the current working precision
        // before each division.  Otherwise `div_at_prec` operates on
        // the full target-precision mantissa internally and reintroduces
        // the cost we're trying to avoid (the shift-and-divide inside
        // `div_at_prec` is proportional to operand size, not just to
        // the requested output precision).
        let mut current_prec = 64_u64;
        let mut x = Float::from_f64_at_prec(x0_mant, current_prec);
        x.exp += half_exp;
        x.round_to_prec();

        // Loop runs at progressively higher precision until current_prec
        // hits the cap (target+16), then runs *one more* iteration at
        // that cap before stopping.  The extra iteration is essential:
        // the doubling iter that first hits the cap runs at full target
        // precision but its INPUT only has accuracy ~target/2 (limited
        // by the previous iter's precision), so its output is accurate
        // to only ~target/2 bits.  One more pass with full-accuracy
        // input is what actually delivers full target accuracy.
        //
        // Without the extra iteration the result silently loses
        // ~target/2 bits of accuracy whenever log₂(target/52) isn't a
        // whole number — for pi-compute that bit it hard above ~3M
        // digits (5M lost ~900K decimal digits, 10M lost ~1.4M).
        let mut full_prec_iters = 0;
        loop {
            current_prec = (current_prec * 2).min(target_prec + 16);

            // Truncate N to current_prec for the division — only the
            // top ~current_prec/64 limbs are copied, not the full
            // target-precision mantissa.
            let n_at_prec = self.truncated_to_prec(current_prec);
            // Bring x up to the new working precision (its information
            // content is unchanged; we're just relabeling the slot it
            // occupies so the next division produces a wider result).
            x.prec = current_prec;
            x.round_to_prec();

            let div = n_at_prec.div_at_prec(&x, current_prec + 8);
            let mut sum = x.add_at_prec(&div, current_prec + 8);
            sum.exp -= 1; // divide by 2
            sum.prec = current_prec;
            sum.round_to_prec();
            x = sum;

            if current_prec >= target_prec + 16 {
                full_prec_iters += 1;
                if full_prec_iters >= 2 {
                    break;
                }
            }
        }

        x.prec = target_prec;
        x.round_to_prec();
        *self = x;
    }

    /// In-place square.  mantissa <- mantissa^2; exp <- 2·exp.
    pub fn square_mut(&mut self) {
        if self.is_zero() {
            return;
        }
        let prec = self.prec;
        self.mantissa = &self.mantissa * &self.mantissa;
        self.exp = self
            .exp
            .checked_mul(2)
            .expect("Float::square_mut exponent overflow");
        self.sign = false;
        self.prec = prec;
        self.round_to_prec();
    }

    /// Convert to (Integer, Ordering) by truncation toward -∞
    /// (`Round::Down`).  Returns `None` if the value can't be converted
    /// (we never hit this case since Float never holds NaN/inf).
    ///
    /// Matches `rug::Float::to_integer_round(&self, …)` — takes self by
    /// reference, so callers can keep the Float around.  pi-core
    /// immediately drops it via an explicit `drop(pi)`.
    pub fn to_integer_round(&self, round: Round) -> Option<(Integer, Ordering)> {
        match round {
            Round::Down => {
                if self.is_zero() {
                    return Some((Integer::new(), Ordering::Equal));
                }
                // value = sign * mantissa * 2^exp
                // For exp >= 0, the value is already an integer.
                // For exp < 0, shift right by -exp.  Truncation toward -∞:
                // if there are nonzero discarded bits AND value is negative,
                // we subtract one from the magnitude (equivalent to floor).
                if self.exp >= 0 {
                    let mut m = self.mantissa.clone();
                    if self.exp > 0 {
                        m.limbs = integer::shl_mag(&m.limbs, self.exp as u64);
                    }
                    if self.sign && !m.is_zero() {
                        m.negative = true;
                    }
                    Some((m, Ordering::Equal))
                } else {
                    let shift = (-self.exp) as u64;
                    let truncated = integer::shr_mag(&self.mantissa.limbs, shift);
                    // Detect any nonzero discarded bit.
                    let any_low = has_low_bits(&self.mantissa.limbs, shift);
                    let mut int_part = Integer { limbs: truncated, negative: false };
                    if self.sign && (!int_part.is_zero() || any_low) {
                        // Negative + nonzero discarded bits => floor goes one lower.
                        if any_low {
                            int_part = &int_part + &Integer::from(1_u32);
                        }
                        int_part.negative = !int_part.is_zero();
                    }
                    let ord = if any_low {
                        Ordering::Less
                    } else {
                        Ordering::Equal
                    };
                    Some((int_part, ord))
                }
            }
        }
    }

    /// In-place `self <- self^exp` for an Integer exp.  Uses square-and-
    /// multiply.  Only non-negative exponents are supported (panics
    /// otherwise — pi-core only calls this with positive Integers).
    pub fn pow_assign_integer(&mut self, exp: &Integer) {
        assert!(!exp.negative, "Float::pow_assign with negative exponent");
        if exp.is_zero() {
            *self = Float::with_val_64(self.prec, 1_u32);
            return;
        }
        // Walk exp's bits from MSB to LSB.
        let prec = self.prec;
        let mut result = Float::with_val_64(prec, 1_u32);
        let mut base = self.clone();
        // Extract exponent bits.  Highest limb first.
        let bits = exp.bits();
        for i in 0..bits {
            let limb = (i / 64) as usize;
            let off = (i % 64) as u32;
            let bit = (exp.limbs[limb] >> off) & 1;
            if bit == 1 {
                result = result.mul_at_prec(&base, prec);
            }
            if i + 1 < bits {
                base = base.mul_at_prec(&base, prec);
            }
        }
        *self = result;
    }

    // -----------------------------------------------------------------
    // Internal helpers: arithmetic at a chosen precision
    // -----------------------------------------------------------------

    /// `self + rhs`, computed at `prec` bits and returned as a fresh Float.
    fn add_at_prec(&self, rhs: &Float, prec: u64) -> Float {
        if self.is_zero() {
            let mut r = rhs.clone();
            r.prec = prec;
            r.round_to_prec();
            return r;
        }
        if rhs.is_zero() {
            let mut r = self.clone();
            r.prec = prec;
            r.round_to_prec();
            return r;
        }
        // Align to the smaller exponent.
        let (mut a_mant, a_sign, mut a_exp) =
            (self.mantissa.clone(), self.sign, self.exp);
        let (mut b_mant, b_sign, mut b_exp) =
            (rhs.mantissa.clone(), rhs.sign, rhs.exp);
        match a_exp.cmp(&b_exp) {
            Ordering::Greater => {
                let diff = (a_exp - b_exp) as u64;
                a_mant.limbs = integer::shl_mag(&a_mant.limbs, diff);
                a_exp -= diff as i64;
            }
            Ordering::Less => {
                let diff = (b_exp - a_exp) as u64;
                b_mant.limbs = integer::shl_mag(&b_mant.limbs, diff);
                b_exp -= diff as i64;
            }
            Ordering::Equal => {}
        }
        let _ = b_exp;
        // Signed sum on the aligned mantissas.
        a_mant.negative = a_sign && !a_mant.is_zero();
        b_mant.negative = b_sign && !b_mant.is_zero();
        let sum = &a_mant + &b_mant;
        let sign = sum.negative;
        let mut mant_abs = sum;
        mant_abs.negative = false;
        let mut out = Float { prec, sign, mantissa: mant_abs, exp: a_exp };
        out.round_to_prec();
        out
    }

    /// `self - rhs`.
    fn sub_at_prec(&self, rhs: &Float, prec: u64) -> Float {
        let mut neg = rhs.clone();
        if !neg.is_zero() {
            neg.sign = !neg.sign;
        }
        self.add_at_prec(&neg, prec)
    }

    /// `self * rhs`.
    pub fn mul_at_prec(&self, rhs: &Float, prec: u64) -> Float {
        if self.is_zero() || rhs.is_zero() {
            return Float { prec, sign: false, mantissa: Integer::new(), exp: 0 };
        }
        let mantissa = &self.mantissa * &rhs.mantissa;
        let exp = self
            .exp
            .checked_add(rhs.exp)
            .expect("Float multiply exponent overflow");
        let sign = self.sign ^ rhs.sign;
        let mut out = Float { prec, sign, mantissa, exp };
        out.round_to_prec();
        out
    }

    /// `self / rhs`.
    ///
    /// Two implementations live behind this entry point.  Knuth-based
    /// division (shift `self.mantissa` so the integer quotient has
    /// `prec + guard` bits, then divide via `integer::div_rem_mag`) is
    /// O(L²) for L-limb operands — fast when L is small but
    /// quadratically painful once L gets large.  Newton–Raphson
    /// reciprocal-then-multiply replaces that with two Karatsuba
    /// multiplications worth of work, which is O(M(N)) ≈ O(N^1.585).
    ///
    /// We dispatch on operand size: below `RECIPROCAL_THRESHOLD` limbs
    /// the constants on Knuth win; above it Newton's asymptotic edge
    /// dominates.  The threshold was tuned by running pi-compute at
    /// 100K / 500K / 1M digits and picking the cross-over.
    fn div_at_prec(&self, rhs: &Float, prec: u64) -> Float {
        if rhs.is_zero() {
            panic!("Float division by zero");
        }
        if self.is_zero() {
            return Float { prec, sign: false, mantissa: Integer::new(), exp: 0 };
        }
        let max_limbs = self.mantissa.limbs.len().max(rhs.mantissa.limbs.len());
        if max_limbs < RECIPROCAL_THRESHOLD {
            return self.div_at_prec_knuth(rhs, prec);
        }
        let recip = rhs.reciprocal_at_prec(prec + 16);
        self.mul_at_prec(&recip, prec)
    }

    /// Original shift-then-Knuth-divide path.  Kept as the small-operand
    /// fallback (and as a reference implementation that's much easier to
    /// audit than the Newton reciprocal).
    fn div_at_prec_knuth(&self, rhs: &Float, prec: u64) -> Float {
        let guard: u64 = 64;
        let want_bits = prec + guard;
        // We want `(self.mantissa * 2^shift) / rhs.mantissa` to have
        // at least `want_bits` bits.
        let a_bits = self.mantissa.bits();
        let b_bits = rhs.mantissa.bits();
        // Roughly, the quotient has `a_bits + shift - b_bits` bits.
        let needed = want_bits as i64 + b_bits as i64 - a_bits as i64;
        let shift = needed.max(1) as u64;
        let mut a_shifted = self.mantissa.clone();
        a_shifted.limbs = integer::shl_mag(&a_shifted.limbs, shift);
        let (q, _r) = integer::div_rem_mag(&a_shifted.limbs, &rhs.mantissa.limbs);
        let mantissa = Integer { limbs: q, negative: false };
        let exp = self
            .exp
            .checked_sub(rhs.exp)
            .expect("Float divide exponent underflow")
            .checked_sub(shift as i64)
            .expect("Float divide exponent underflow");
        let sign = self.sign ^ rhs.sign;
        let mut out = Float { prec, sign, mantissa, exp };
        out.round_to_prec();
        out
    }

    /// Newton–Raphson reciprocal: compute `1/self` at `target_prec`.
    ///
    /// Uses the standard `x_{k+1} = x_k · (2 - self · x_k)` iteration,
    /// which has *quadratic* convergence and uses only multiplication
    /// and subtraction (no division — that's the whole point).  We
    /// start from an f64 estimate (~52 bits of seed) and
    /// precision-double per iteration so the total cost is geometric
    /// and dominated by the final pass.  Combined with Karatsuba
    /// multiplication, the asymptotic cost is `O(M(N))` rather than
    /// the `O(N²)` of the Knuth-based divide it replaces.
    ///
    /// Returns a Float whose value is `1/self` to `target_prec` bits.
    /// Self must be nonzero.
    pub fn reciprocal_at_prec(&self, target_prec: u64) -> Float {
        assert!(!self.is_zero(), "reciprocal of zero");

        // Work on |self| and reapply sign at the end so the Newton
        // iteration runs in non-negative arithmetic and the `2 - …`
        // step always produces a positive intermediate.
        let mag = Float { sign: false, ..self.clone() };

        // f64 seed for 1/|self|: extract a normalized f64 from the
        // mantissa's top bits, invert it, and adjust the binary
        // exponent.  Gives ~52 correct bits to bootstrap Newton.
        let (m, e) = mag.mantissa.to_f64_with_exp();
        let inv_m = 1.0 / m;
        let neg_exp = -(e + mag.exp);
        let mut x = Float::from_f64_at_prec(inv_m, 64);
        x.exp += neg_exp;
        x.round_to_prec();

        // See the matching comment in `sqrt_mut`: the doubling sequence
        // alone falls one iteration short of full target accuracy
        // whenever log₂(target/52) isn't a whole number.  We force one
        // extra iteration at the cap to bring accuracy from ~target/2
        // up to full target.
        let mut current_prec = 64_u64;
        let mut full_prec_iters = 0;
        loop {
            current_prec = (current_prec * 2).min(target_prec + 16);

            // Truncate `self` to current_prec for the same reason as
            // in `sqrt_mut`: otherwise the multiplication operates on
            // the full original mantissa and reintroduces the cost we
            // were trying to dodge.
            let b_at_prec = mag.truncated_to_prec(current_prec);
            x.prec = current_prec;
            x.round_to_prec();

            // bx = b * x.   For a converged x, bx ≈ 1.
            let bx = b_at_prec.mul_at_prec(&x, current_prec + 8);
            // two_minus_bx = 2 - bx.   Near convergence this is ~1
            // plus a tiny correction term that captures the error in x.
            let two = Float::from_f64_at_prec(2.0, current_prec + 8);
            let two_minus_bx = two.sub_at_prec(&bx, current_prec + 8);
            // x_new = x * (2 - bx).   Roughly doubles the correct bits.
            x = x.mul_at_prec(&two_minus_bx, current_prec);

            if current_prec >= target_prec + 16 {
                full_prec_iters += 1;
                if full_prec_iters >= 2 {
                    break;
                }
            }
        }

        x.prec = target_prec;
        x.round_to_prec();
        x.sign = self.sign;
        x
    }

    /// Return a copy of `self` whose mantissa holds at most `prec`
    /// bits.  Functionally equivalent to
    /// `{ let mut t = self.clone(); t.prec = prec; t.round_to_prec(); t }`,
    /// but avoids cloning the full mantissa Vec when `self` is much
    /// larger than `prec` — only the top ~prec/64 limbs are copied.
    ///
    /// Used in the Newton iters of `sqrt_mut` / `reciprocal_at_prec`,
    /// where every iteration needs a `prec`-truncated view of an
    /// operand whose mantissa is target-precision (much larger).
    pub(crate) fn truncated_to_prec(&self, prec: u64) -> Float {
        if self.mantissa.is_zero() {
            return Float { prec, sign: false, mantissa: Integer::new(), exp: 0 };
        }
        let mantissa_bits = self.mantissa.bits();
        if mantissa_bits <= prec {
            // Already small enough — just clone and relabel the prec.
            let mut out = self.clone();
            out.prec = prec;
            return out;
        }
        // Keep enough limbs to hold `prec + 1` bits so `round_to_prec`
        // can examine the round bit just below the cut.  Mantissa is
        // little-endian, so we drop low limbs and keep the high ones.
        let keep_limbs = ((prec + 64) / 64) as usize + 1;
        let total_limbs = self.mantissa.limbs.len();
        if keep_limbs >= total_limbs {
            // Nothing meaningful to trim — fall back to the simple path.
            let mut out = self.clone();
            out.prec = prec;
            out.round_to_prec();
            return out;
        }
        let drop_limbs = total_limbs - keep_limbs;
        let new_limbs: Vec<u64> = self.mantissa.limbs[drop_limbs..].to_vec();
        let mut out = Float {
            prec,
            sign: self.sign,
            mantissa: Integer { limbs: new_limbs, negative: self.mantissa.negative },
            exp: self.exp + (drop_limbs as i64) * 64,
        };
        out.round_to_prec();
        out
    }

    /// Normalize: ensure mantissa has at most `prec` bits, rounding
    /// (half-to-even-ish, but we don't fuss — schoolbook truncation +
    /// round-bit add is fine for our purposes since we always carry a
    /// 256-bit safety margin in PrecisionPlan).
    fn round_to_prec(&mut self) {
        if self.mantissa.is_zero() {
            self.sign = false;
            self.exp = 0;
            return;
        }
        let bits = self.mantissa.bits();
        if bits <= self.prec {
            return;
        }
        let drop = bits - self.prec;
        // Examine the bit just below the cut (the "round bit") to do
        // simple round-half-up.
        let round_bit = bit_at(&self.mantissa.limbs, drop - 1);
        let mut new_limbs = integer::shr_mag(&self.mantissa.limbs, drop);
        if round_bit == 1 {
            // Increment new_limbs by 1.  This can grow the bit length
            // by one; if so, shift right one more and bump exp.
            let mut carry: u128 = 1;
            for limb in new_limbs.iter_mut() {
                let s = *limb as u128 + carry;
                *limb = s as u64;
                carry = s >> 64;
                if carry == 0 {
                    break;
                }
            }
            if carry > 0 {
                new_limbs.push(carry as u64);
            }
        }
        self.mantissa = Integer { limbs: new_limbs, negative: false };
        self.exp = self
            .exp
            .checked_add(drop as i64)
            .expect("Float round_to_prec exponent overflow");
        // Re-normalize: rounding-up at the top boundary may have grown
        // the mantissa by one bit, which is fine if it's still within
        // prec, but if not we drop one more bit.
        let new_bits = self.mantissa.bits();
        if new_bits > self.prec {
            let extra = new_bits - self.prec;
            self.mantissa.limbs = integer::shr_mag(&self.mantissa.limbs, extra);
            self.exp += extra as i64;
        }
    }

    /// Build a Float from an f64 at the given precision.  Handles
    /// finite values only (we never use non-finite in this crate).
    fn from_f64_at_prec(v: f64, prec: u64) -> Float {
        assert!(v.is_finite(), "Float::from_f64_at_prec with non-finite");
        if v == 0.0 {
            return Float { prec, sign: false, mantissa: Integer::new(), exp: 0 };
        }
        let bits = v.to_bits();
        let sign = (bits >> 63) & 1 == 1;
        let raw_exp = ((bits >> 52) & 0x7FF) as i64;
        let raw_mant = bits & ((1_u64 << 52) - 1);
        let (mantissa_u, exp): (u64, i64) = if raw_exp == 0 {
            // Subnormal: mantissa = raw_mant, exp = -1074
            (raw_mant, -1074)
        } else {
            // Normal: mantissa = raw_mant | (1<<52), exp = raw_exp - 1023 - 52
            (raw_mant | (1_u64 << 52), raw_exp - 1023 - 52)
        };
        let mantissa = Integer::from(mantissa_u);
        let mut f = Float { prec, sign, mantissa, exp };
        f.round_to_prec();
        f
    }
}

/// Whether bit `i` of a magnitude (0 = LSB) is set.
fn bit_at(a: &[u64], i: u64) -> u64 {
    let limb = (i / 64) as usize;
    if limb >= a.len() {
        return 0;
    }
    (a[limb] >> (i % 64)) & 1
}

/// Whether any of the lowest `n` bits of `a` are set.
fn has_low_bits(a: &[u64], n: u64) -> bool {
    if n == 0 || a.is_empty() {
        return false;
    }
    let full_limbs = (n / 64) as usize;
    for i in 0..full_limbs.min(a.len()) {
        if a[i] != 0 {
            return true;
        }
    }
    let rem = (n % 64) as u32;
    if rem > 0 {
        if let Some(&v) = a.get(full_limbs) {
            let mask = (1_u64 << rem) - 1;
            if v & mask != 0 {
                return true;
            }
        }
    }
    false
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        // Two zeros compare equal regardless of sign/exp.
        if self.is_zero() && other.is_zero() {
            return true;
        }
        self.sign == other.sign
            && self.mantissa == other.mantissa
            && self.exp == other.exp
    }
}

impl Eq for Float {}

// =====================================================================
// Conversions into Float (for `with_val_64`)
// =====================================================================

impl From<u32> for Float {
    fn from(v: u32) -> Self {
        Float { prec: 64, sign: false, mantissa: Integer::from(v), exp: 0 }
    }
}
impl From<u64> for Float {
    fn from(v: u64) -> Self {
        Float { prec: 64, sign: false, mantissa: Integer::from(v), exp: 0 }
    }
}
impl From<i32> for Float {
    fn from(v: i32) -> Self {
        Float::from(v as i64)
    }
}
impl From<i64> for Float {
    fn from(v: i64) -> Self {
        let sign = v < 0;
        let mag = (v as i128).unsigned_abs() as u64;
        let mantissa = Integer::from(mag);
        Float { prec: 64, sign, mantissa, exp: 0 }
    }
}
impl From<f64> for Float {
    fn from(v: f64) -> Self {
        // 53-bit precision is enough to hold any finite f64 exactly.
        Float::from_f64_at_prec(v, 64)
    }
}
impl From<Integer> for Float {
    fn from(v: Integer) -> Self {
        let sign = v.negative;
        let bits = v.bits().max(64);
        let mantissa = Integer { limbs: v.limbs, negative: false };
        Float { prec: bits, sign, mantissa, exp: 0 }
    }
}
impl<'a> From<&'a Integer> for Float {
    fn from(v: &'a Integer) -> Self {
        let sign = v.negative;
        let bits = v.bits().max(64);
        let mantissa = Integer { limbs: v.limbs.clone(), negative: false };
        Float { prec: bits, sign, mantissa, exp: 0 }
    }
}
impl<'a> From<&'a Float> for Float {
    fn from(v: &'a Float) -> Self {
        v.clone()
    }
}

// =====================================================================
// Arithmetic ops
// =====================================================================

macro_rules! impl_binop_float {
    ($trait:ident, $method:ident, $inner:ident) => {
        impl $trait<Float> for Float {
            type Output = Float;
            fn $method(self, rhs: Float) -> Float {
                let prec = self.prec.max(rhs.prec);
                self.$inner(&rhs, prec)
            }
        }
        impl<'a> $trait<&'a Float> for Float {
            type Output = Float;
            fn $method(self, rhs: &'a Float) -> Float {
                let prec = self.prec.max(rhs.prec);
                self.$inner(rhs, prec)
            }
        }
        impl<'a> $trait<Float> for &'a Float {
            type Output = Float;
            fn $method(self, rhs: Float) -> Float {
                let prec = self.prec.max(rhs.prec);
                self.$inner(&rhs, prec)
            }
        }
        impl<'a, 'b> $trait<&'b Float> for &'a Float {
            type Output = Float;
            fn $method(self, rhs: &'b Float) -> Float {
                let prec = self.prec.max(rhs.prec);
                self.$inner(rhs, prec)
            }
        }
    };
}

impl_binop_float!(Add, add, add_at_prec);
impl_binop_float!(Sub, sub, sub_at_prec);
impl_binop_float!(Mul, mul, mul_at_prec);
impl_binop_float!(Div, div, div_at_prec);

// Float ⊕ Integer / &Integer
macro_rules! impl_binop_float_int {
    ($trait:ident, $method:ident, $inner:ident) => {
        impl $trait<Integer> for Float {
            type Output = Float;
            fn $method(self, rhs: Integer) -> Float {
                let prec = self.prec;
                let rhs_f: Float = rhs.into();
                self.$inner(&rhs_f, prec)
            }
        }
        impl<'a> $trait<&'a Integer> for Float {
            type Output = Float;
            fn $method(self, rhs: &'a Integer) -> Float {
                let prec = self.prec;
                let rhs_f: Float = rhs.into();
                self.$inner(&rhs_f, prec)
            }
        }
        impl<'a> $trait<Integer> for &'a Float {
            type Output = Float;
            fn $method(self, rhs: Integer) -> Float {
                let prec = self.prec;
                let rhs_f: Float = rhs.into();
                self.$inner(&rhs_f, prec)
            }
        }
        impl<'a, 'b> $trait<&'b Integer> for &'a Float {
            type Output = Float;
            fn $method(self, rhs: &'b Integer) -> Float {
                let prec = self.prec;
                let rhs_f: Float = rhs.into();
                self.$inner(&rhs_f, prec)
            }
        }
    };
}

impl_binop_float_int!(Add, add, add_at_prec);
impl_binop_float_int!(Sub, sub, sub_at_prec);
impl_binop_float_int!(Mul, mul, mul_at_prec);
impl_binop_float_int!(Div, div, div_at_prec);

// Float ⊕ primitives — only the ones actually called in pi-core
macro_rules! impl_binop_float_prim {
    ($trait:ident, $method:ident, $inner:ident, $prim:ty) => {
        impl $trait<$prim> for Float {
            type Output = Float;
            fn $method(self, rhs: $prim) -> Float {
                let prec = self.prec;
                let rhs_f: Float = rhs.into();
                self.$inner(&rhs_f, prec)
            }
        }
        impl<'a> $trait<$prim> for &'a Float {
            type Output = Float;
            fn $method(self, rhs: $prim) -> Float {
                let prec = self.prec;
                let rhs_f: Float = rhs.into();
                self.$inner(&rhs_f, prec)
            }
        }
    };
}

impl_binop_float_prim!(Add, add, add_at_prec, u32);
impl_binop_float_prim!(Sub, sub, sub_at_prec, u32);
impl_binop_float_prim!(Mul, mul, mul_at_prec, u32);
impl_binop_float_prim!(Div, div, div_at_prec, u32);

// =====================================================================
// Assignment ops
// =====================================================================

macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $inner:ident, $rhs:ty) => {
        impl $trait<$rhs> for Float {
            fn $method(&mut self, rhs: $rhs) {
                let prec = self.prec;
                let r: Float = rhs.into();
                let out = self.$inner(&r, prec);
                *self = out;
            }
        }
    };
}

impl_assign_op!(AddAssign, add_assign, add_at_prec, Float);
impl_assign_op!(SubAssign, sub_assign, sub_at_prec, Float);
impl_assign_op!(MulAssign, mul_assign, mul_at_prec, Float);
impl_assign_op!(DivAssign, div_assign, div_at_prec, Float);

impl AddAssign<&Float> for Float {
    fn add_assign(&mut self, rhs: &Float) {
        let prec = self.prec;
        let out = self.add_at_prec(rhs, prec);
        *self = out;
    }
}
impl SubAssign<&Float> for Float {
    fn sub_assign(&mut self, rhs: &Float) {
        let prec = self.prec;
        let out = self.sub_at_prec(rhs, prec);
        *self = out;
    }
}
impl MulAssign<&Float> for Float {
    fn mul_assign(&mut self, rhs: &Float) {
        let prec = self.prec;
        let out = self.mul_at_prec(rhs, prec);
        *self = out;
    }
}
impl DivAssign<&Float> for Float {
    fn div_assign(&mut self, rhs: &Float) {
        let prec = self.prec;
        let out = self.div_at_prec(rhs, prec);
        *self = out;
    }
}

impl AddAssign<Integer> for Float {
    fn add_assign(&mut self, rhs: Integer) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.add_at_prec(&r, prec);
        *self = out;
    }
}
impl SubAssign<Integer> for Float {
    fn sub_assign(&mut self, rhs: Integer) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.sub_at_prec(&r, prec);
        *self = out;
    }
}
impl MulAssign<Integer> for Float {
    fn mul_assign(&mut self, rhs: Integer) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.mul_at_prec(&r, prec);
        *self = out;
    }
}
impl DivAssign<Integer> for Float {
    fn div_assign(&mut self, rhs: Integer) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.div_at_prec(&r, prec);
        *self = out;
    }
}

impl AddAssign<&Integer> for Float {
    fn add_assign(&mut self, rhs: &Integer) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.add_at_prec(&r, prec);
        *self = out;
    }
}
impl SubAssign<&Integer> for Float {
    fn sub_assign(&mut self, rhs: &Integer) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.sub_at_prec(&r, prec);
        *self = out;
    }
}
impl MulAssign<&Integer> for Float {
    fn mul_assign(&mut self, rhs: &Integer) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.mul_at_prec(&r, prec);
        *self = out;
    }
}
impl DivAssign<&Integer> for Float {
    fn div_assign(&mut self, rhs: &Integer) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.div_at_prec(&r, prec);
        *self = out;
    }
}

impl MulAssign<u32> for Float {
    fn mul_assign(&mut self, rhs: u32) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.mul_at_prec(&r, prec);
        *self = out;
    }
}
impl DivAssign<u32> for Float {
    fn div_assign(&mut self, rhs: u32) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.div_at_prec(&r, prec);
        *self = out;
    }
}
impl AddAssign<u32> for Float {
    fn add_assign(&mut self, rhs: u32) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.add_at_prec(&r, prec);
        *self = out;
    }
}
impl SubAssign<u32> for Float {
    fn sub_assign(&mut self, rhs: u32) {
        let prec = self.prec;
        let r: Float = rhs.into();
        let out = self.sub_at_prec(&r, prec);
        *self = out;
    }
}

// PowAssign<&Integer>
impl PowAssign<&Integer> for Float {
    fn pow_assign(&mut self, rhs: &Integer) {
        self.pow_assign_integer(rhs);
    }
}

// =====================================================================
// Cap matching `rug::float::prec_max_64()`
// =====================================================================

/// Maximum precision (bits) supported.  In our representation precision
/// is only bounded by RAM, but expose a finite cap so callers can
/// preserve their existing "is the planned precision too large?" check.
pub fn prec_max_64() -> u64 {
    // Use the same enormous value the codebase tolerates.  1 << 40 ≈ 1 TiB
    // of mantissa bits — far beyond any feasible pi run.
    1_u64 << 40
}

impl fmt::Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Float {{ prec: {}, sign: {}, mantissa_bits: {}, exp: {} }}",
            self.prec,
            self.sign,
            self.mantissa.bits(),
            self.exp,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Round;

    fn approx_eq(a: &Float, b: &Float, tol_ulp: u64) -> bool {
        // Equal-or-near at the smaller precision.
        if a == b {
            return true;
        }
        let diff = a.sub_at_prec(b, a.prec.max(b.prec));
        if diff.is_zero() {
            return true;
        }
        // bit length of diff vs smaller of |a|,|b|.
        let mag_a = a.mantissa.bits() as i64 + a.exp;
        let mag_b = b.mantissa.bits() as i64 + b.exp;
        let mag_min = mag_a.min(mag_b);
        let mag_diff = diff.mantissa.bits() as i64 + diff.exp;
        // |diff| <= 2^(mag_min - prec + tol_ulp_bits)
        let prec = a.prec.min(b.prec) as i64;
        mag_diff <= mag_min - prec + tol_ulp as i64
    }

    #[test]
    fn from_int() {
        let a = Float::with_val_64(64, 1_u32);
        let b = Float::with_val_64(64, 2_u32);
        assert!(!a.is_zero());
        assert!(!b.is_zero());
    }

    #[test]
    fn add_simple() {
        let a = Float::with_val_64(64, 1_u32);
        let b = Float::with_val_64(64, 2_u32);
        let c = &a + &b;
        let three = Float::with_val_64(64, 3_u32);
        assert!(approx_eq(&c, &three, 1));
    }

    #[test]
    fn sub_simple() {
        let a = Float::with_val_64(64, 5_u32);
        let b = Float::with_val_64(64, 2_u32);
        let c = &a - &b;
        let three = Float::with_val_64(64, 3_u32);
        assert!(approx_eq(&c, &three, 1));
    }

    #[test]
    fn mul_simple() {
        let a = Float::with_val_64(64, 6_u32);
        let b = Float::with_val_64(64, 7_u32);
        let c = &a * &b;
        let exp = Float::with_val_64(64, 42_u32);
        assert!(approx_eq(&c, &exp, 1));
    }

    #[test]
    fn div_simple() {
        let a = Float::with_val_64(128, 1_u32);
        let b = Float::with_val_64(128, 4_u32);
        let c = &a / &b;
        let exp = Float::with_val_64(128, 0.25_f64);
        assert!(approx_eq(&c, &exp, 2));
    }

    #[test]
    fn sqrt_simple() {
        let mut a = Float::with_val_64(256, 2_u32);
        a.sqrt_mut();
        // square should give back 2 within tolerance
        let mut sq = a.clone();
        sq.square_mut();
        let two = Float::with_val_64(256, 2_u32);
        assert!(approx_eq(&sq, &two, 4));
    }

    #[test]
    fn div_large_exercises_newton_path() {
        // Precision chosen so mantissas exceed RECIPROCAL_THRESHOLD (64
        // limbs ≈ 4096 bits) and `div_at_prec` routes through
        // `reciprocal_at_prec` instead of the Knuth fallback.
        let prec = 8192_u64;
        let a = Float::with_val_64(prec, 17_u32);
        let b = Float::with_val_64(prec, 3_u32);
        // Compute (a/b) * b and verify ≈ a within a few ULPs.  This
        // round-trip catches sign mistakes, exponent drift, and
        // Newton-convergence shortfalls that any unit value test would
        // miss.
        let q = a.div_at_prec(&b, prec);
        let back = q.mul_at_prec(&b, prec);
        assert!(
            approx_eq(&back, &a, 4),
            "round-trip a/b*b at NR path lost precision: got {back:?}, want {a:?}",
        );
    }

    #[test]
    fn reciprocal_round_trip_at_large_prec() {
        let prec = 8192_u64;
        // 1/7 reciprocated twice should land back ≈ 7.
        let seven = Float::with_val_64(prec, 7_u32);
        let one_seventh = seven.reciprocal_at_prec(prec);
        let back = one_seventh.reciprocal_at_prec(prec);
        assert!(
            approx_eq(&back, &seven, 4),
            "reciprocate-twice didn't land back on the original",
        );
    }

    #[test]
    fn sqrt_large() {
        let mut a = Float::with_val_64(512, 10_005_u32);
        a.sqrt_mut();
        let mut sq = a.clone();
        sq.square_mut();
        let exp = Float::with_val_64(512, 10_005_u32);
        assert!(approx_eq(&sq, &exp, 4));
    }

    #[test]
    fn sqrt_at_off_by_one_bug_precision() {
        // The precision-doubling Newton sqrt previously fell one
        // iteration short of full accuracy whenever frac(log₂(prec/64))
        // > ~0.7 — the final doubling iter ran at full precision but
        // its input had only target/2 bits of accuracy, so the output
        // also had only target/2 bits.  At pi-compute scale this
        // started producing wrong digits past ~5M decimal digits.
        //
        // 110000-bit precision triggers the same condition at small
        // scale: log₂(110000/64) ≈ 10.75, fractional part 0.75 > 0.7.
        // Without the fix, sqrt(N).square() differs from N by far more
        // than the 4-ULP test tolerance.
        let prec = 110_000_u64;
        let mut a = Float::with_val_64(prec, 10_005_u32);
        a.sqrt_mut();
        let mut sq = a.clone();
        sq.square_mut();
        let exp = Float::with_val_64(prec, 10_005_u32);
        assert!(
            approx_eq(&sq, &exp, 8),
            "sqrt didn't converge to full precision at 110000 bits",
        );
    }

    #[test]
    fn reciprocal_at_off_by_one_bug_precision() {
        // Same off-by-one as `sqrt_at_off_by_one_bug_precision`, but for
        // the Newton reciprocal: the doubling sequence falls one iter
        // short of full accuracy at the same precision threshold.
        let prec = 110_000_u64;
        let three = Float::with_val_64(prec, 3_u32);
        let recip = three.reciprocal_at_prec(prec);
        // 3 * (1/3) should be ≈ 1 to ~full precision.
        let product = three.mul_at_prec(&recip, prec);
        let one = Float::with_val_64(prec, 1_u32);
        assert!(
            approx_eq(&product, &one, 8),
            "reciprocal didn't converge to full precision at 110000 bits",
        );
    }

    #[test]
    fn square_mut_works() {
        let mut a = Float::with_val_64(64, 13_u32);
        a.square_mut();
        let exp = Float::with_val_64(64, 169_u32);
        assert!(approx_eq(&a, &exp, 1));
    }

    #[test]
    fn pow_assign_works() {
        let mut a = Float::with_val_64(128, 10_u32);
        a.pow_assign(&Integer::from(5_u32));
        let exp = Float::with_val_64(128, 100_000_u32);
        assert!(approx_eq(&a, &exp, 1));
    }

    #[test]
    fn to_integer_round_down_positive() {
        let a = Float::with_val_64(64, 0.5_f64); // exactly 1/2
        let f2 = Float::with_val_64(64, 5_u32);
        let v = &a * &f2; // 2.5
        let (i, _) = (&v).to_integer_round(Round::Down).unwrap();
        assert_eq!(i.to_string_radix(10), "2");
    }

    #[test]
    fn assign_replaces_value() {
        let mut a = Float::with_val_64(128, 1_u32);
        let b = Float::with_val_64(128, 100_u32);
        a.assign(&b + &b);
        let exp = Float::with_val_64(128, 200_u32);
        assert!(approx_eq(&a, &exp, 1));
    }
}
