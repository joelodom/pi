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
        let prec = self.prec;
        // Initial f64 estimate.
        let (m, e) = self.mantissa.to_f64_with_exp();
        // self ≈ m * 2^(e + exp).  Sqrt: x0 ≈ sqrt(m) * 2^((e+exp)/2).
        let half_exp = (e + self.exp).div_euclid(2);
        let leftover = (e + self.exp).rem_euclid(2);
        let m_corrected = if leftover == 0 { m } else { m * 2.0 };
        let x0_mant = m_corrected.sqrt();
        let mut x = Float::from_f64_at_prec(x0_mant, prec);
        x.exp += half_exp;
        x.round_to_prec();

        // Newton iterations.  We iterate until two consecutive iterates
        // round to the same Float value at `prec` bits — at quadratic
        // convergence this needs roughly log2(prec) steps from a good
        // seed but we cap defensively and use convergence detection.
        let max_iters = (prec as f64).log2().ceil() as u32 + 10;
        for _ in 0..max_iters {
            // x_new = (x + self/x) / 2
            let div = (&*self).div_at_prec(&x, prec + 8);
            let mut sum = x.add_at_prec(&div, prec + 8);
            sum.exp -= 1; // divide by 2
            sum.prec = prec;
            sum.round_to_prec();
            if sum == x {
                break;
            }
            x = sum;
        }
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
    fn mul_at_prec(&self, rhs: &Float, prec: u64) -> Float {
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

    /// `self / rhs`.  Computes `mantissa_a * 2^k / mantissa_b` for a
    /// large enough `k` that the integer quotient has `prec + guard`
    /// bits, then rounds back to `prec`.
    fn div_at_prec(&self, rhs: &Float, prec: u64) -> Float {
        if rhs.is_zero() {
            panic!("Float division by zero");
        }
        if self.is_zero() {
            return Float { prec, sign: false, mantissa: Integer::new(), exp: 0 };
        }
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
    fn sqrt_large() {
        let mut a = Float::with_val_64(512, 10_005_u32);
        a.sqrt_mut();
        let mut sq = a.clone();
        sq.square_mut();
        let exp = Float::with_val_64(512, 10_005_u32);
        assert!(approx_eq(&sq, &exp, 4));
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
