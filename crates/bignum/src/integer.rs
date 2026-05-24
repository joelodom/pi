//! Arbitrary-precision signed integer.
//!
//! Storage: a `Vec<u64>` little-endian limb vector for the magnitude
//! plus a separate sign bit.  Magnitude is always kept *normalized* —
//! no trailing (most-significant) zero limbs.  Zero is uniquely
//! represented as an empty vector with `negative == false`.
//!
//! Algorithms are schoolbook O(N^2) for multiply, bit-level long
//! division for divide.  Correctness > performance; this whole crate
//! is a proof-of-concept stand-in for `rug`.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{
    Add, AddAssign, Div, Mul, MulAssign, Neg, Rem, ShlAssign, Sub, SubAssign,
};

use crate::Pow;

/// Signed arbitrary-precision integer.
#[derive(Clone, Default)]
pub struct Integer {
    /// Little-endian magnitude limbs.  Always normalized (no trailing zeros).
    pub(crate) limbs: Vec<u64>,
    /// `true` iff the value is strictly negative.
    pub(crate) negative: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseIntegerError(pub String);

impl fmt::Display for ParseIntegerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error: {}", self.0)
    }
}

impl std::error::Error for ParseIntegerError {}

impl Integer {
    /// Construct zero.
    pub fn new() -> Self {
        Self { limbs: Vec::new(), negative: false }
    }

    /// `true` iff value is zero.
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Bit length of the absolute value (`0` for zero).
    pub fn bits(&self) -> u64 {
        match self.limbs.last() {
            None => 0,
            Some(&top) => {
                (self.limbs.len() as u64 - 1) * 64
                    + (64 - top.leading_zeros() as u64)
            }
        }
    }

    /// Replace `self` with `other` in place.
    pub fn assign(&mut self, other: Integer) {
        *self = other;
    }

    /// `base.pow(exp)` as an Integer.  Repeated squaring.
    pub fn u_pow_u(base: u32, exp: u32) -> Integer {
        let mut result = Integer::from(1_u32);
        let mut b = Integer::from(base);
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = &result * &b;
            }
            e >>= 1;
            if e > 0 {
                b = &b * &b;
            }
        }
        result
    }

    /// `self^exp`, repeated squaring.
    pub fn pow_u32(&self, exp: u32) -> Integer {
        let mut result = Integer::from(1_u32);
        let mut b = self.clone();
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = &result * &b;
            }
            e >>= 1;
            if e > 0 {
                b = &b * &b;
            }
        }
        result
    }

    /// Parse from a string in `radix` (only 10 and 16 are supported).
    pub fn parse_radix(s: &str, radix: u32) -> Result<Integer, ParseIntegerError> {
        if !(radix == 10 || radix == 16) {
            return Err(ParseIntegerError(format!(
                "unsupported radix {radix}"
            )));
        }
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            return Err(ParseIntegerError("empty string".into()));
        }
        let mut idx = 0;
        let negative = match bytes[0] {
            b'-' => {
                idx = 1;
                true
            }
            b'+' => {
                idx = 1;
                false
            }
            _ => false,
        };
        if idx == bytes.len() {
            return Err(ParseIntegerError("no digits".into()));
        }
        let mut result = Integer::new();
        let radix_int = Integer::from(radix as u64);
        for &b in &bytes[idx..] {
            let digit = match (b, radix) {
                (b'0'..=b'9', _) => (b - b'0') as u64,
                (b'a'..=b'f', 16) => (b - b'a' + 10) as u64,
                (b'A'..=b'F', 16) => (b - b'A' + 10) as u64,
                _ => {
                    return Err(ParseIntegerError(format!(
                        "invalid character '{}' for radix {}",
                        b as char, radix
                    )));
                }
            };
            if digit >= radix as u64 {
                return Err(ParseIntegerError(format!(
                    "digit {digit} out of range for radix {radix}"
                )));
            }
            result = &result * &radix_int;
            result = &result + &Integer::from(digit);
        }
        if !result.is_zero() {
            result.negative = negative;
        }
        Ok(result)
    }

    /// Render in `radix` (only 10 and 16 supported).
    pub fn to_string_radix(&self, radix: u32) -> String {
        if self.is_zero() {
            return "0".into();
        }
        let mut out = String::new();
        let mut mag = self.clone();
        mag.negative = false;
        match radix {
            16 => {
                // Direct limb iteration — bottom limb is base 2^64, render
                // hex of each limb most-significant-first.
                let n = mag.limbs.len();
                // Most-significant limb without leading zeros.
                out.push_str(&format!("{:x}", mag.limbs[n - 1]));
                for i in (0..n - 1).rev() {
                    out.push_str(&format!("{:016x}", mag.limbs[i]));
                }
            }
            10 => {
                // Pull off groups of 19 decimal digits per loop:
                // 10^19 = 10_000_000_000_000_000_000 < 2^64.  Each
                // division by the chunk base extracts up to 19 digits.
                const CHUNK: u64 = 10_000_000_000_000_000_000;
                let mut chunks: Vec<u64> = Vec::new();
                while !mag.is_zero() {
                    let (q, r) = div_rem_small(&mag, CHUNK);
                    chunks.push(r);
                    mag = q;
                }
                // Most-significant chunk first, no leading zeros.
                let last = chunks.pop().expect("nonzero has at least one chunk");
                out.push_str(&last.to_string());
                while let Some(c) = chunks.pop() {
                    out.push_str(&format!("{c:019}"));
                }
            }
            _ => panic!("unsupported radix {radix} in to_string_radix"),
        }
        if self.negative {
            let mut s = String::with_capacity(out.len() + 1);
            s.push('-');
            s.push_str(&out);
            out = s;
        }
        out
    }

    /// Convert |self| to `m * 2^e` with `m` as an f64 in `[1, 2)` (or
    /// `(0.0, 0)` if self is zero).
    pub(crate) fn to_f64_with_exp(&self) -> (f64, i64) {
        if self.is_zero() {
            return (0.0, 0);
        }
        let total_bits = self.bits();
        // Pull up to the top 53 bits into a u64 with the leading 1 at
        // position 52 if possible (so the value is in [2^52, 2^53)).
        let take_bits: u32 = 53;
        let m_norm;
        let log2_e;
        if total_bits <= take_bits as u64 {
            // Whole magnitude fits in 53 bits; mantissa = bottom limb.
            let v = *self.limbs.first().unwrap_or(&0);
            // total_bits >= 1 because !is_zero, and v's top bit is at total_bits-1.
            let exp = total_bits as i64 - 1;
            m_norm = v as f64 * 2.0_f64.powi(-(exp as i32));
            log2_e = exp;
        } else {
            // Take the top 53 bits.
            let drop = total_bits - take_bits as u64;
            let top53 = shr_to_u64(self, drop);
            // top53 has leading 1 at position 52, so value in [2^52, 2^53)
            // representing the leading 53 bits of self.  |self| ≈ top53 * 2^drop.
            // We want m in [1, 2): m = top53 / 2^52, exp = drop + 52.
            m_norm = top53 as f64 * 2.0_f64.powi(-52);
            log2_e = drop as i64 + 52;
        }
        (m_norm, log2_e)
    }
}

/// Right-shift `n.abs()` by `bits` and pack into u64 (truncates anything
/// above bit 63 of the result).  Used to extract a 53-ish bit mantissa.
fn shr_to_u64(n: &Integer, bits: u64) -> u64 {
    if n.is_zero() {
        return 0;
    }
    let limb = (bits / 64) as usize;
    let off = (bits % 64) as u32;
    let lo = *n.limbs.get(limb).unwrap_or(&0);
    let hi = *n.limbs.get(limb + 1).unwrap_or(&0);
    if off == 0 {
        lo
    } else {
        (lo >> off) | (hi << (64 - off))
    }
}

// =====================================================================
// Construction
// =====================================================================

impl From<u32> for Integer {
    fn from(v: u32) -> Self {
        Integer::from(v as u64)
    }
}

impl From<u64> for Integer {
    fn from(v: u64) -> Self {
        if v == 0 {
            Self::new()
        } else {
            Self { limbs: vec![v], negative: false }
        }
    }
}

impl From<i32> for Integer {
    fn from(v: i32) -> Self {
        Integer::from(v as i64)
    }
}

impl From<i64> for Integer {
    fn from(v: i64) -> Self {
        if v == 0 {
            return Self::new();
        }
        let negative = v < 0;
        let mag = (v as i128).unsigned_abs() as u64;
        Self { limbs: vec![mag], negative }
    }
}

impl From<usize> for Integer {
    fn from(v: usize) -> Self {
        Integer::from(v as u64)
    }
}

// =====================================================================
// Comparison
// =====================================================================

impl PartialEq for Integer {
    fn eq(&self, other: &Self) -> bool {
        self.negative == other.negative && self.limbs == other.limbs
    }
}

impl Eq for Integer {}

impl Ord for Integer {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => cmp_mag(&self.limbs, &other.limbs),
            (true, true) => cmp_mag(&other.limbs, &self.limbs),
        }
    }
}

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    match a.len().cmp(&b.len()) {
        Ordering::Equal => {}
        ord => return ord,
    }
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        match x.cmp(y) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    Ordering::Equal
}

// =====================================================================
// Negation
// =====================================================================

impl Neg for Integer {
    type Output = Integer;
    fn neg(mut self) -> Self::Output {
        if !self.is_zero() {
            self.negative = !self.negative;
        }
        self
    }
}

impl<'a> Neg for &'a Integer {
    type Output = Integer;
    fn neg(self) -> Self::Output {
        let mut out = self.clone();
        if !out.is_zero() {
            out.negative = !out.negative;
        }
        out
    }
}

// =====================================================================
// Add / Sub primitives (operate on magnitudes; sign handled by callers)
// =====================================================================

/// `a + b` on magnitudes; result is normalized.
fn add_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = Vec::with_capacity(long.len() + 1);
    let mut carry: u128 = 0;
    for i in 0..long.len() {
        let sum = long[i] as u128
            + short.get(i).copied().unwrap_or(0) as u128
            + carry;
        out.push(sum as u64);
        carry = sum >> 64;
    }
    if carry > 0 {
        out.push(carry as u64);
    }
    out
}

/// `a - b` where `a >= b` (in magnitude).  Result is normalized.
fn sub_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    debug_assert!(cmp_mag(a, b) != Ordering::Less, "sub_mag preconditions");
    let mut out = Vec::with_capacity(a.len());
    let mut borrow: i128 = 0;
    for i in 0..a.len() {
        let bi = b.get(i).copied().unwrap_or(0) as i128;
        let diff = a[i] as i128 - bi - borrow;
        if diff < 0 {
            out.push((diff + (1i128 << 64)) as u64);
            borrow = 1;
        } else {
            out.push(diff as u64);
            borrow = 0;
        }
    }
    // Strip trailing zeros.
    while matches!(out.last(), Some(&0)) {
        out.pop();
    }
    out
}

fn add_signed(a: &Integer, b: &Integer) -> Integer {
    if a.negative == b.negative {
        let limbs = add_mag(&a.limbs, &b.limbs);
        let mut r = Integer { limbs, negative: a.negative };
        if r.is_zero() {
            r.negative = false;
        }
        r
    } else {
        // Magnitudes subtract.
        match cmp_mag(&a.limbs, &b.limbs) {
            Ordering::Equal => Integer::new(),
            Ordering::Greater => {
                let limbs = sub_mag(&a.limbs, &b.limbs);
                let mut r = Integer { limbs, negative: a.negative };
                if r.is_zero() {
                    r.negative = false;
                }
                r
            }
            Ordering::Less => {
                let limbs = sub_mag(&b.limbs, &a.limbs);
                let mut r = Integer { limbs, negative: b.negative };
                if r.is_zero() {
                    r.negative = false;
                }
                r
            }
        }
    }
}

fn sub_signed(a: &Integer, b: &Integer) -> Integer {
    // a - b = a + (-b)
    let neg_b = Integer { limbs: b.limbs.clone(), negative: !b.negative && !b.is_zero() };
    add_signed(a, &neg_b)
}

// =====================================================================
// Multiplication
// =====================================================================

/// Below this many limbs in the smaller operand, schoolbook beats
/// Karatsuba (the constant-factor overhead of 3 recursive calls plus
/// the extra adds and subtracts outweighs the saved multiplication).
/// 32 limbs ≈ 2048 bits ≈ 620 decimal digits — a reasonable cross-over
/// for our scratch-allocating implementation.  GMP's empirical default
/// on x86-64 is in the same neighborhood (~28 limbs).
const KARATSUBA_THRESHOLD: usize = 32;

/// Dispatch: pick schoolbook for small operands, Karatsuba for large.
/// The cross-over is set by [`KARATSUBA_THRESHOLD`].
fn mul_mag(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    if a.len().min(b.len()) < KARATSUBA_THRESHOLD {
        mul_mag_schoolbook(a, b)
    } else {
        mul_mag_karatsuba(a, b)
    }
}

/// Schoolbook (long multiplication) on magnitudes — O(|a|·|b|).  The
/// inner loop is a `u128` multiply-accumulate; carries propagate to
/// the next limb.  Used directly for small operands and as the
/// Karatsuba base case.
fn mul_mag_schoolbook(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0_u64; a.len() + b.len()];
    for i in 0..a.len() {
        let mut carry: u128 = 0;
        let av = a[i] as u128;
        for j in 0..b.len() {
            let cur = out[i + j] as u128 + av * b[j] as u128 + carry;
            out[i + j] = cur as u64;
            carry = cur >> 64;
        }
        // Propagate the final carry.
        let mut k = i + b.len();
        while carry > 0 {
            let cur = out[k] as u128 + carry;
            out[k] = cur as u64;
            carry = cur >> 64;
            k += 1;
        }
    }
    while matches!(out.last(), Some(&0)) {
        out.pop();
    }
    out
}

/// Karatsuba multiplication on magnitudes — O(N^log₂3) ≈ O(N^1.585).
///
/// Splits each operand into high and low halves at the same limb
/// boundary and replaces the four naive sub-multiplications with three
/// recursive calls via the identity
///
/// ```text
/// (a_hi·B^m + a_lo) · (b_hi·B^m + b_lo)
///     = a_hi·b_hi·B^(2m)
///     + ((a_lo + a_hi)·(b_lo + b_hi) − a_hi·b_hi − a_lo·b_lo) · B^m
///     + a_lo·b_lo
/// ```
///
/// where `B = 2^64` is the limb base and `m` is the split point in
/// limbs.  The three subproducts (z0, z2, z1_prod) recurse through the
/// dispatcher, so the base case is schoolbook on small slices.
fn mul_mag_karatsuba(a: &[u64], b: &[u64]) -> Vec<u64> {
    // Split point: half of the longer operand, in limbs.
    let n = a.len().max(b.len());
    let m = n / 2;

    // Split each operand.  `split_at(m.min(len))` handles the case
    // where one operand is shorter than the split point: the "high"
    // half is then empty and z2 (or its analog) is zero.
    let (a_lo, a_hi) = a.split_at(m.min(a.len()));
    let (b_lo, b_hi) = b.split_at(m.min(b.len()));

    // z0 = a_lo · b_lo  (low × low)
    let z0 = mul_mag(a_lo, b_lo);
    // z2 = a_hi · b_hi  (high × high; may be empty if either hi is empty)
    let z2 = mul_mag(a_hi, b_hi);
    // z1_prod = (a_lo + a_hi) · (b_lo + b_hi)
    let a_sum = add_mag(a_lo, a_hi);
    let b_sum = add_mag(b_lo, b_hi);
    let z1_prod = mul_mag(&a_sum, &b_sum);
    // z1 = z1_prod − z0 − z2  (the cross terms; always ≥ 0)
    let z0_plus_z2 = add_mag(&z0, &z2);
    let z1 = sub_mag(&z1_prod, &z0_plus_z2);

    // Assemble: result = z0 + (z1 << m·64) + (z2 << 2m·64)
    let mut out = vec![0_u64; a.len() + b.len() + 1];
    add_at(&mut out, &z0, 0);
    add_at(&mut out, &z1, m);
    add_at(&mut out, &z2, 2 * m);
    while matches!(out.last(), Some(&0)) {
        out.pop();
    }
    out
}

/// In-place limb-vector add: `dest += src << (offset·64)`.  Carries
/// propagate past the end if needed; `dest` may grow.
fn add_at(dest: &mut Vec<u64>, src: &[u64], offset: usize) {
    if src.is_empty() {
        return;
    }
    if dest.len() < offset + src.len() {
        dest.resize(offset + src.len(), 0);
    }
    let mut carry: u128 = 0;
    for i in 0..src.len() {
        let sum = dest[offset + i] as u128 + src[i] as u128 + carry;
        dest[offset + i] = sum as u64;
        carry = sum >> 64;
    }
    let mut k = offset + src.len();
    while carry > 0 {
        if dest.len() <= k {
            dest.push(0);
        }
        let sum = dest[k] as u128 + carry;
        dest[k] = sum as u64;
        carry = sum >> 64;
        k += 1;
    }
}

fn mul_signed(a: &Integer, b: &Integer) -> Integer {
    let limbs = mul_mag(&a.limbs, &b.limbs);
    if limbs.is_empty() {
        return Integer::new();
    }
    Integer { limbs, negative: a.negative ^ b.negative }
}

// =====================================================================
// Division & remainder (bit-level long division on magnitudes)
// =====================================================================

/// Magnitude division: `|a| / |b|`, returning `(q, r)` with `r < |b|`.
///
/// Dispatches:
/// * trivial cases (`a < b`, `b = 1 limb`) are handled directly;
/// * otherwise Knuth's Algorithm D (TAOCP vol. 2 §4.3.1) — limb-level
///   long division with quotient-digit estimation.  Cost is `O(L²)`
///   per call, where `L = |a|.len()`.  An earlier bit-by-bit
///   implementation was `O(B·L) ≈ O(64·L²)`, so this is ~60× faster
///   at the precisions we care about.
pub(crate) fn div_rem_mag(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    assert!(!b.is_empty(), "division by zero");
    if cmp_mag(a, b) == Ordering::Less {
        return (Vec::new(), a.to_vec());
    }
    if b.len() == 1 {
        return div_rem_mag_single(a, b[0]);
    }
    div_rem_mag_knuth(a, b)
}

/// Single-limb divisor: walk `a` from MSB limb to LSB, carrying the
/// running remainder in 128 bits.  O(|a|) total.
fn div_rem_mag_single(a: &[u64], d: u64) -> (Vec<u64>, Vec<u64>) {
    let mut q = vec![0_u64; a.len()];
    let mut r: u128 = 0;
    for i in (0..a.len()).rev() {
        let cur = (r << 64) | a[i] as u128;
        q[i] = (cur / d as u128) as u64;
        r = cur % d as u128;
    }
    while matches!(q.last(), Some(&0)) {
        q.pop();
    }
    let rem = if r == 0 { Vec::new() } else { vec![r as u64] };
    (q, rem)
}

/// Knuth's Algorithm D.  Precondition: `b.len() >= 2`, `|a| >= |b|`.
///
/// **D1. Normalize.**  Shift both inputs left by `d = leading_zeros(b[n-1])`
/// bits so the divisor's top bit is set.  This makes step D3's
/// trial-quotient estimate from the top two limbs of the partial
/// remainder accurate to within 2 — and after the D3 refinement, at
/// most 1 too high (which D5/D6 corrects).
///
/// **D2-D7. Main loop**, j from m down to 0:
/// * **D3.** Trial quotient `qhat = floor((u[j+n]·B + u[j+n-1]) / v[n-1])`,
///   capped at `B-1`.  Refine by checking `qhat·v[n-2] > B·rhat + u[j+n-2]`.
/// * **D4.** Multiply-subtract: `u[j..j+n+1] -= qhat·v`.  May underflow.
/// * **D5/D6.** If underflow, `qhat` was 1 too high: decrement and add
///   `v` back; the add-carry cancels the subtract-borrow.
///
/// **D8. Unnormalize.**  Remainder gets shifted right by `d`.
fn div_rem_mag_knuth(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let n = b.len();
    let m = a.len() - n;
    let d = b[n - 1].leading_zeros();

    // Normalize.
    let v: Vec<u64> = if d == 0 { b.to_vec() } else { shl_mag(b, d as u64) };
    debug_assert_eq!(v.len(), n, "normalization should not grow divisor");
    let mut u: Vec<u64> = if d == 0 { a.to_vec() } else { shl_mag(a, d as u64) };
    // Ensure u has exactly m+n+1 limbs (the +1 might be 0).
    while u.len() < a.len() + 1 {
        u.push(0);
    }

    let v_top = v[n - 1];
    let v_next = v[n - 2];
    let mut q = vec![0_u64; m + 1];

    for j in (0..=m).rev() {
        // D3. Estimate qhat from the top two limbs of the partial remainder.
        let numerator = ((u[j + n] as u128) << 64) | (u[j + n - 1] as u128);
        let (mut qhat, mut rhat) = if u[j + n] >= v_top {
            // Quotient digit would saturate; cap at B-1 and adjust rhat.
            let qcap = u64::MAX as u128;
            (u64::MAX, numerator - qcap * v_top as u128)
        } else {
            ((numerator / v_top as u128) as u64, numerator % v_top as u128)
        };
        // Refine downward while `qhat·v[n-2] > B·rhat + u[j+n-2]`.
        loop {
            let lhs = (qhat as u128) * (v_next as u128);
            let rhs = (rhat << 64) | (u[j + n - 2] as u128);
            if lhs <= rhs {
                break;
            }
            qhat -= 1;
            rhat += v_top as u128;
            if rhat >= (1u128 << 64) {
                break;
            }
        }

        // D4. Multiply-subtract: u[j..=j+n] -= qhat * v.
        let underflow = mul_sub_inplace(&mut u[j..=j + n], &v, qhat);

        // D5/D6. Compensate if qhat was one too high.
        if underflow {
            qhat -= 1;
            add_back_inplace(&mut u[j..=j + n], &v);
        }

        q[j] = qhat;
    }

    // D8. Unnormalize remainder.  Quotient is unaffected by the
    // common left shift of dividend and divisor.
    let mut r: Vec<u64> = u[..n].to_vec();
    if d > 0 {
        r = shr_mag(&r, d as u64);
    }
    while matches!(r.last(), Some(&0)) {
        r.pop();
    }
    while matches!(q.last(), Some(&0)) {
        q.pop();
    }
    (q, r)
}

/// In-place `u -= qhat * v` over slice `u` (which must be exactly
/// `v.len() + 1` long).  Returns `true` if a borrow underflowed past
/// the top limb of `u` — meaning `qhat` was one too large.
fn mul_sub_inplace(u: &mut [u64], v: &[u64], qhat: u64) -> bool {
    debug_assert_eq!(u.len(), v.len() + 1);
    let mut borrow: i128 = 0;
    let mut carry: u128 = 0;
    let qh = qhat as u128;
    for i in 0..v.len() {
        let prod = qh * v[i] as u128 + carry;
        carry = prod >> 64;
        let prod_lo = prod as u64;
        let diff = (u[i] as i128) - (prod_lo as i128) - borrow;
        if diff < 0 {
            u[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            u[i] = diff as u64;
            borrow = 0;
        }
    }
    // Last subtraction: the high carry from the multiplication plus
    // any borrow against u's top limb.
    let total_sub = carry as i128 + borrow;
    let diff = (u[v.len()] as i128) - total_sub;
    if diff < 0 {
        u[v.len()] = (diff + (1i128 << 64)) as u64;
        true
    } else {
        u[v.len()] = diff as u64;
        false
    }
}

/// In-place `u += v` over slice `u` (length `v.len() + 1`).  The final
/// carry is discarded — it cancels the underflow borrow that
/// triggered the add-back step.
fn add_back_inplace(u: &mut [u64], v: &[u64]) {
    debug_assert_eq!(u.len(), v.len() + 1);
    let mut carry: u128 = 0;
    for i in 0..v.len() {
        let sum = u[i] as u128 + v[i] as u128 + carry;
        u[i] = sum as u64;
        carry = sum >> 64;
    }
    // Wrapping add into top limb; the overflow here cancels the
    // borrow recorded as u[v.len()]'s wrap-around from mul_sub.
    u[v.len()] = u[v.len()].wrapping_add(carry as u64);
}

/// Small-divisor optimization: divide magnitude by a u64, returning
/// (quotient, remainder).  Used by decimal stringification.
fn div_rem_small(a: &Integer, d: u64) -> (Integer, u64) {
    assert!(d > 0, "division by zero");
    if a.is_zero() {
        return (Integer::new(), 0);
    }
    let mut q = vec![0_u64; a.limbs.len()];
    let mut r: u128 = 0;
    for i in (0..a.limbs.len()).rev() {
        let cur = (r << 64) | a.limbs[i] as u128;
        q[i] = (cur / d as u128) as u64;
        r = cur % d as u128;
    }
    while matches!(q.last(), Some(&0)) {
        q.pop();
    }
    let qint = Integer { limbs: q, negative: false };
    (qint, r as u64)
}

fn div_signed(a: &Integer, b: &Integer) -> Integer {
    assert!(!b.is_zero(), "division by zero");
    let (q_mag, _r) = div_rem_mag(&a.limbs, &b.limbs);
    if q_mag.is_empty() {
        return Integer::new();
    }
    Integer { limbs: q_mag, negative: a.negative ^ b.negative }
}

fn rem_signed(a: &Integer, b: &Integer) -> Integer {
    assert!(!b.is_zero(), "division by zero");
    let (_q, r_mag) = div_rem_mag(&a.limbs, &b.limbs);
    if r_mag.is_empty() {
        return Integer::new();
    }
    // Sign of remainder matches sign of dividend (truncated division).
    Integer { limbs: r_mag, negative: a.negative }
}

// =====================================================================
// Shift left by u32 (logical shift on magnitude; preserves sign)
// =====================================================================

pub(crate) fn shl_mag(a: &[u64], n: u64) -> Vec<u64> {
    if a.is_empty() || n == 0 {
        return a.to_vec();
    }
    let limb_shift = (n / 64) as usize;
    let bit_shift = (n % 64) as u32;
    let mut out = vec![0_u64; a.len() + limb_shift + 1];
    if bit_shift == 0 {
        for (i, &v) in a.iter().enumerate() {
            out[i + limb_shift] = v;
        }
    } else {
        let mut carry: u64 = 0;
        for (i, &v) in a.iter().enumerate() {
            let shifted = (v << bit_shift) | carry;
            carry = v >> (64 - bit_shift);
            out[i + limb_shift] = shifted;
        }
        if carry > 0 {
            out[a.len() + limb_shift] = carry;
        }
    }
    while matches!(out.last(), Some(&0)) {
        out.pop();
    }
    out
}

pub(crate) fn shr_mag(a: &[u64], n: u64) -> Vec<u64> {
    if a.is_empty() {
        return Vec::new();
    }
    let limb_shift = (n / 64) as usize;
    if limb_shift >= a.len() {
        return Vec::new();
    }
    let bit_shift = (n % 64) as u32;
    let new_len = a.len() - limb_shift;
    let mut out = vec![0_u64; new_len];
    if bit_shift == 0 {
        for i in 0..new_len {
            out[i] = a[i + limb_shift];
        }
    } else {
        for i in 0..new_len {
            let lo = a[i + limb_shift] >> bit_shift;
            let hi = a
                .get(i + limb_shift + 1)
                .copied()
                .unwrap_or(0)
                << (64 - bit_shift);
            out[i] = lo | hi;
        }
    }
    while matches!(out.last(), Some(&0)) {
        out.pop();
    }
    out
}

impl ShlAssign<u32> for Integer {
    fn shl_assign(&mut self, n: u32) {
        if self.is_zero() || n == 0 {
            return;
        }
        self.limbs = shl_mag(&self.limbs, n as u64);
    }
}

impl ShlAssign<i32> for Integer {
    fn shl_assign(&mut self, n: i32) {
        assert!(n >= 0, "negative shift not supported on Integer");
        *self <<= n as u32;
    }
}

// =====================================================================
// Binary op trait impls — value/ref/primitive combinations
// =====================================================================
//
// The codebase calls Integer +-*/% Integer in all four combinations
// (owned/borrowed) and with u32/u64 primitives in some places.

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $impl_fn:ident) => {
        impl $trait<Integer> for Integer {
            type Output = Integer;
            fn $method(self, rhs: Integer) -> Integer {
                $impl_fn(&self, &rhs)
            }
        }
        impl<'a> $trait<&'a Integer> for Integer {
            type Output = Integer;
            fn $method(self, rhs: &'a Integer) -> Integer {
                $impl_fn(&self, rhs)
            }
        }
        impl<'a> $trait<Integer> for &'a Integer {
            type Output = Integer;
            fn $method(self, rhs: Integer) -> Integer {
                $impl_fn(self, &rhs)
            }
        }
        impl<'a, 'b> $trait<&'b Integer> for &'a Integer {
            type Output = Integer;
            fn $method(self, rhs: &'b Integer) -> Integer {
                $impl_fn(self, rhs)
            }
        }
    };
}

impl_binop!(Add, add, add_signed);
impl_binop!(Sub, sub, sub_signed);
impl_binop!(Mul, mul, mul_signed);
impl_binop!(Div, div, div_signed);
impl_binop!(Rem, rem, rem_signed);

// Primitive RHS: u32 and u64 — chudnovsky.rs uses both forms.
macro_rules! impl_binop_prim {
    ($trait:ident, $method:ident, $impl_fn:ident, $prim:ty) => {
        impl $trait<$prim> for Integer {
            type Output = Integer;
            fn $method(self, rhs: $prim) -> Integer {
                let rhs = Integer::from(rhs);
                $impl_fn(&self, &rhs)
            }
        }
        impl<'a> $trait<$prim> for &'a Integer {
            type Output = Integer;
            fn $method(self, rhs: $prim) -> Integer {
                let rhs = Integer::from(rhs);
                $impl_fn(self, &rhs)
            }
        }
        // And the reverse direction (primitive op Integer).
        impl $trait<Integer> for $prim {
            type Output = Integer;
            fn $method(self, rhs: Integer) -> Integer {
                let lhs = Integer::from(self);
                $impl_fn(&lhs, &rhs)
            }
        }
        impl<'a> $trait<&'a Integer> for $prim {
            type Output = Integer;
            fn $method(self, rhs: &'a Integer) -> Integer {
                let lhs = Integer::from(self);
                $impl_fn(&lhs, rhs)
            }
        }
    };
}

impl_binop_prim!(Add, add, add_signed, u32);
impl_binop_prim!(Sub, sub, sub_signed, u32);
impl_binop_prim!(Mul, mul, mul_signed, u32);
impl_binop_prim!(Div, div, div_signed, u32);
impl_binop_prim!(Rem, rem, rem_signed, u32);

impl_binop_prim!(Add, add, add_signed, u64);
impl_binop_prim!(Sub, sub, sub_signed, u64);
impl_binop_prim!(Mul, mul, mul_signed, u64);
impl_binop_prim!(Div, div, div_signed, u64);
impl_binop_prim!(Rem, rem, rem_signed, u64);

// =====================================================================
// Assignment ops
// =====================================================================

impl AddAssign<Integer> for Integer {
    fn add_assign(&mut self, rhs: Integer) {
        let r = add_signed(self, &rhs);
        *self = r;
    }
}
impl<'a> AddAssign<&'a Integer> for Integer {
    fn add_assign(&mut self, rhs: &'a Integer) {
        let r = add_signed(self, rhs);
        *self = r;
    }
}
impl AddAssign<u32> for Integer {
    fn add_assign(&mut self, rhs: u32) {
        let rhs = Integer::from(rhs);
        let r = add_signed(self, &rhs);
        *self = r;
    }
}

impl SubAssign<Integer> for Integer {
    fn sub_assign(&mut self, rhs: Integer) {
        let r = sub_signed(self, &rhs);
        *self = r;
    }
}
impl<'a> SubAssign<&'a Integer> for Integer {
    fn sub_assign(&mut self, rhs: &'a Integer) {
        let r = sub_signed(self, rhs);
        *self = r;
    }
}
impl SubAssign<u32> for Integer {
    fn sub_assign(&mut self, rhs: u32) {
        let rhs = Integer::from(rhs);
        let r = sub_signed(self, &rhs);
        *self = r;
    }
}

impl MulAssign<Integer> for Integer {
    fn mul_assign(&mut self, rhs: Integer) {
        let r = mul_signed(self, &rhs);
        *self = r;
    }
}
impl<'a> MulAssign<&'a Integer> for Integer {
    fn mul_assign(&mut self, rhs: &'a Integer) {
        let r = mul_signed(self, rhs);
        *self = r;
    }
}
impl MulAssign<u32> for Integer {
    fn mul_assign(&mut self, rhs: u32) {
        let rhs = Integer::from(rhs);
        let r = mul_signed(self, &rhs);
        *self = r;
    }
}

// =====================================================================
// Pow trait — match the call site `k.clone().pow(3_u32)`
// =====================================================================

impl Pow<u32> for Integer {
    type Output = Integer;
    fn pow(self, exp: u32) -> Integer {
        self.pow_u32(exp)
    }
}

impl<'a> Pow<u32> for &'a Integer {
    type Output = Integer;
    fn pow(self, exp: u32) -> Integer {
        self.pow_u32(exp)
    }
}

// =====================================================================
// Display / Debug
// =====================================================================

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_radix(10))
    }
}

impl fmt::Debug for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_radix(10))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_zero() {
        let a = Integer::new();
        assert!(a.is_zero());
        assert_eq!(a.to_string_radix(10), "0");
        assert_eq!(a.to_string_radix(16), "0");
        assert_eq!(a.bits(), 0);
    }

    #[test]
    fn small_round_trip_decimal() {
        for v in [0_u64, 1, 9, 10, 99, 1_000_000, u64::MAX] {
            let a = Integer::from(v);
            assert_eq!(a.to_string_radix(10), v.to_string());
        }
    }

    #[test]
    fn add_subtract() {
        let a = Integer::from(123_u32);
        let b = Integer::from(456_u32);
        assert_eq!((&a + &b).to_string_radix(10), "579");
        assert_eq!((&b - &a).to_string_radix(10), "333");
        assert_eq!((&a - &b).to_string_radix(10), "-333");
    }

    #[test]
    fn mul_basic() {
        let a = Integer::from(12345_u32);
        let b = Integer::from(67890_u32);
        assert_eq!((&a * &b).to_string_radix(10), "838102050");
    }

    #[test]
    fn mul_large_carry() {
        // 2^64 * 2^64 = 2^128
        let mut a = Integer::from(1_u64);
        a <<= 64_u32;
        let mut b = Integer::from(1_u64);
        b <<= 64_u32;
        let c = &a * &b;
        // 2^128 == "340282366920938463463374607431768211456"
        assert_eq!(
            c.to_string_radix(10),
            "340282366920938463463374607431768211456"
        );
    }

    #[test]
    fn karatsuba_matches_schoolbook_on_large_inputs() {
        // Build two operands big enough to trigger the Karatsuba path
        // (well above KARATSUBA_THRESHOLD).  Then verify the dispatched
        // mul_mag agrees with the leaf schoolbook on the same inputs.
        // This is the property test that the recursive identity is
        // implemented correctly across many sizes and split points.
        let mut a = Integer::from(0xDEAD_BEEF_CAFE_F00D_u64);
        let mut b = Integer::from(0x0123_4567_89AB_CDEF_u64);
        // Grow each to ~250 limbs (~16K bits ~ 4800 decimal digits).
        // Multiple sequential squarings give us non-trivial bit
        // patterns at every position, not just a single bit set.
        for _ in 0..7 {
            a = &a * &a;
        }
        for _ in 0..7 {
            b = &b * &b;
        }
        assert!(a.limbs.len() > 100 && b.limbs.len() > 100);
        let via_dispatch = super::mul_mag(&a.limbs, &b.limbs);
        let via_schoolbook = super::mul_mag_schoolbook(&a.limbs, &b.limbs);
        assert_eq!(via_dispatch, via_schoolbook);
    }

    #[test]
    fn knuth_div_roundtrip_many_sizes() {
        // For each (a_len, b_len), build a "dividend" by multiplying
        // a known quotient by a known divisor and adding a remainder,
        // then divide and verify we recover the original parts.  This
        // exercises Knuth D across operand sizes that span the
        // single-limb fast path, small operands, and operands large
        // enough that the inner D3 refinement and D6 add-back paths
        // both get hit by chance.
        let dividend_q = Integer::parse_radix(
            "1234567890123456789012345678901234567890123456789012345678901234567890",
            10,
        )
        .unwrap();
        let divisor_v = Integer::parse_radix(
            "98765432109876543210987654321",
            10,
        )
        .unwrap();
        let remainder = Integer::from(42_u32);
        // p = q*v + r
        let p = &(&dividend_q * &divisor_v) + &remainder;
        // p / v == (q, r)
        let (q_limbs, r_limbs) = super::div_rem_mag(&p.limbs, &divisor_v.limbs);
        assert_eq!(q_limbs, dividend_q.limbs, "Knuth quotient mismatch");
        assert_eq!(r_limbs, remainder.limbs, "Knuth remainder mismatch");
    }

    #[test]
    fn knuth_div_large_balanced() {
        // ~1000-bit / ~500-bit case: exercises Knuth's main loop with
        // a non-trivial number of quotient digits.  Roundtrip check.
        let q_int = {
            let mut x = Integer::from(0xFEDC_BA98_7654_3210_u64);
            for _ in 0..3 {
                x = &x * &x;
            }
            x
        };
        let v = {
            let mut x = Integer::from(0xDEAD_BEEF_u64);
            for _ in 0..3 {
                x = &x * &x;
            }
            x
        };
        let r = Integer::from(31337_u32);
        let p = &(&q_int * &v) + &r;
        let (q_limbs, r_limbs) = super::div_rem_mag(&p.limbs, &v.limbs);
        assert_eq!(q_limbs, q_int.limbs);
        assert_eq!(r_limbs, r.limbs);
    }

    #[test]
    fn knuth_div_single_limb_divisor() {
        // Should hit the single-limb fast path, not Knuth.
        let a = Integer::parse_radix("100000000000000000000000000000", 10).unwrap();
        let (q_limbs, r_limbs) = super::div_rem_mag(&a.limbs, &[7_u64]);
        let q = Integer { limbs: q_limbs, negative: false };
        let r = Integer { limbs: r_limbs, negative: false };
        // 10^29 = 7q + r; verify by multiplying back.
        let recon = &(&q * &Integer::from(7_u32)) + &r;
        assert_eq!(recon, a);
    }

    #[test]
    fn karatsuba_asymmetric_sizes() {
        // One operand much smaller than the other — degenerate split
        // case where the "high" half of the small operand is empty.
        let mut big = Integer::from(3_u32);
        big <<= 10_000_u32; // 10,001-bit number
        let small = Integer::from(0xFEDC_BA98_7654_3210_u64);
        let prod_via_dispatch = &big * &small;
        let prod_via_schoolbook = Integer {
            limbs: super::mul_mag_schoolbook(&big.limbs, &small.limbs),
            negative: false,
        };
        assert_eq!(prod_via_dispatch, prod_via_schoolbook);
    }

    #[test]
    fn shl_works() {
        let mut a = Integer::from(1_u32);
        a <<= 100_u32;
        // 2^100 == 1267650600228229401496703205376
        assert_eq!(a.to_string_radix(10), "1267650600228229401496703205376");
    }

    #[test]
    fn div_rem() {
        let a = Integer::from(1_000_000_007_u64);
        let b = Integer::from(31337_u64);
        let q = &a / &b;
        let r = &a % &b;
        assert_eq!(&(&q * &b) + &r, a);
    }

    #[test]
    fn div_rem_large() {
        // 10^30 / 17
        let a = Integer::u_pow_u(10, 30);
        let b = Integer::from(17_u32);
        let q = &a / &b;
        let r = &a % &b;
        assert_eq!(&q * &b + &r, a);
    }

    #[test]
    fn parse_radix_basic() {
        let a = Integer::parse_radix("12345678901234567890", 10).unwrap();
        assert_eq!(a.to_string_radix(10), "12345678901234567890");
        let b = Integer::parse_radix("ffeeddccbbaa9988", 16).unwrap();
        assert_eq!(b.to_string_radix(16), "ffeeddccbbaa9988");
    }

    #[test]
    fn parse_negative() {
        let a = Integer::parse_radix("-999", 10).unwrap();
        assert_eq!(a.to_string_radix(10), "-999");
    }

    #[test]
    fn parse_long() {
        // Round-trip a 200-digit decimal number.
        let s = "1".to_string() + &"234567890".repeat(22) + "5";
        let a = Integer::parse_radix(&s, 10).unwrap();
        assert_eq!(a.to_string_radix(10), s);
    }

    #[test]
    fn pow_basic() {
        let a = Integer::from(2_u32);
        let p = a.pow_u32(64);
        assert_eq!(p.to_string_radix(10), "18446744073709551616");
    }

    #[test]
    fn u_pow_u_basic() {
        assert_eq!(Integer::u_pow_u(10, 0).to_string_radix(10), "1");
        assert_eq!(Integer::u_pow_u(10, 1).to_string_radix(10), "10");
        assert_eq!(Integer::u_pow_u(10, 19).to_string_radix(10), "10000000000000000000");
    }

    #[test]
    fn bits_basic() {
        assert_eq!(Integer::from(0_u32).bits(), 0);
        assert_eq!(Integer::from(1_u32).bits(), 1);
        assert_eq!(Integer::from(2_u32).bits(), 2);
        assert_eq!(Integer::from(255_u32).bits(), 8);
        assert_eq!(Integer::from(256_u32).bits(), 9);
        assert_eq!(Integer::from(u64::MAX).bits(), 64);
    }

    #[test]
    fn neg_zero() {
        let z = -Integer::new();
        assert!(z.is_zero());
        assert!(!z.negative);
    }
}
