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
        let mag = if self.negative {
            Integer { limbs: self.limbs.clone(), negative: false }
        } else {
            self.clone()
        };
        if self.negative {
            out.push('-');
        }
        match radix {
            16 => {
                // Direct limb iteration — bottom limb is base 2^64, render
                // hex of each limb most-significant-first.
                let n = mag.limbs.len();
                out.push_str(&format!("{:x}", mag.limbs[n - 1]));
                for i in (0..n - 1).rev() {
                    out.push_str(&format!("{:016x}", mag.limbs[i]));
                }
            }
            10 => to_decimal_top(&mag, &mut out),
            _ => panic!("unsupported radix {radix} in to_string_radix"),
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

/// Subproblems at or above this size (in limbs of the smaller operand)
/// get their three Karatsuba sub-multiplications dispatched to rayon
/// via nested `join`.  Below the threshold the sub-mults are still
/// pure Karatsuba (down to the schoolbook leaves) but run sequentially
/// — rayon's task overhead would otherwise eat any saving.
///
/// 512 limbs ≈ 32K bits ≈ 10K decimal digits.  At top-level mults of
/// 80K+ limbs this gives us several levels of 3-way parallelism before
/// switching back to sequential.
const PARALLEL_KARATSUBA_THRESHOLD: usize = 512;

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
///
/// At top recursion levels (operands ≥ `PARALLEL_KARATSUBA_THRESHOLD`)
/// the three sub-mults are independent and we run them in parallel via
/// nested `rayon::join`.  Each level of the recursion that meets the
/// threshold gets 3× parallelism, so a few levels deep we have plenty
/// of work for any reasonable core count.
fn mul_mag_karatsuba(a: &[u64], b: &[u64]) -> Vec<u64> {
    // Split point: half of the longer operand, in limbs.
    let n = a.len().max(b.len());
    let m = n / 2;

    // Split each operand.  `split_at(m.min(len))` handles the case
    // where one operand is shorter than the split point: the "high"
    // half is then empty and z2 (or its analog) is zero.
    let (a_lo, a_hi) = a.split_at(m.min(a.len()));
    let (b_lo, b_hi) = b.split_at(m.min(b.len()));

    // The three sub-mults are independent and dominate the cost.  At
    // large sizes, dispatch them across rayon worker threads.
    let go_parallel = a.len().min(b.len()) >= PARALLEL_KARATSUBA_THRESHOLD;

    let (z0, z2, z1_prod) = if go_parallel {
        // Pre-compute the addends so they can be moved into the rayon
        // closures (which need 'static lifetimes via owned data).
        let a_sum = add_mag(a_lo, a_hi);
        let b_sum = add_mag(b_lo, b_hi);
        // Three-way fork: rayon::join is 2-way, so nest once.  The
        // outermost call gives us `z0` in parallel with `(z2, z1_prod)`
        // which itself is `z2` in parallel with `z1_prod`.
        let (z0, (z2, z1_prod)) = rayon::join(
            || mul_mag(a_lo, b_lo),
            || {
                rayon::join(
                    || mul_mag(a_hi, b_hi),
                    || mul_mag(&a_sum, &b_sum),
                )
            },
        );
        (z0, z2, z1_prod)
    } else {
        let z0 = mul_mag(a_lo, b_lo);
        let z2 = mul_mag(a_hi, b_hi);
        let a_sum = add_mag(a_lo, a_hi);
        let b_sum = add_mag(b_lo, b_hi);
        let z1_prod = mul_mag(&a_sum, &b_sum);
        (z0, z2, z1_prod)
    };

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
        //
        // `rhat` is conceptually a u64 (the running remainder) but we
        // hold it in u128 to allow the `rhat += v_top` accumulation.
        // CRITICAL: check `rhat >= B` *before* forming `rhat << 64`,
        // because that shift would silently overflow u128 (and lose
        // the high bits, corrupting the comparison).  Knuth guarantees
        // that once `rhat >= B`, the refinement condition can't hold
        // anyway, so an early break here is correct.
        loop {
            if rhat >= (1u128 << 64) {
                break;
            }
            let lhs = (qhat as u128) * (v_next as u128);
            let rhs = (rhat << 64) | (u[j + n - 2] as u128);
            if lhs <= rhs {
                break;
            }
            qhat -= 1;
            rhat += v_top as u128;
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
// Decimal-stringification: divide-and-conquer base conversion
// =====================================================================
//
// The naive approach to radix-10 conversion peels off 19 digits at a
// time by dividing by 10^19, which is `O(L²)` `u128 / u64` operations
// for an L-limb input.  On aarch64 `u128 / u64` is a software-emulated
// routine in compiler-builtins; profiling at 1M digits showed 83% of
// runtime sitting in that single inner loop.
//
// Divide-and-conquer base conversion replaces that with `O(M(N)·log N)`:
// split `n` by a power of ten roughly half its decimal length, recurse
// on the two halves, then concatenate (with leading-zero padding on
// the low half).  The expensive operation at each level is one
// Newton-Raphson Float-backed Knuth division — way cheaper than
// hundreds of millions of `u128 / u64`s.

/// Below this many limbs in the input, the naive divide-by-10^19 loop
/// beats the D&C overhead (a single Knuth divide + a `Integer::pow`
/// for the splitter, plus two recursive calls).  Tuned by inspection;
/// 32 limbs ≈ 600 decimal digits.
const TO_STRING_DC_THRESHOLD: usize = 32;

/// Decimal CHUNK = 10^19, the largest power of ten that fits in u64.
const DECIMAL_CHUNK_DIGITS: u32 = 19;
const DECIMAL_CHUNK: u64 = 10_000_000_000_000_000_000;

/// Top-level entry: append `n` (which must be `≥ 0` and nonzero) to
/// `out` with no leading zeros.
fn to_decimal_top(n: &Integer, out: &mut String) {
    debug_assert!(!n.is_zero() && !n.negative);
    if n.limbs.len() <= TO_STRING_DC_THRESHOLD {
        write_decimal_naive(n, out);
        return;
    }
    // Estimate decimal digit count from bit length: D ≈ bits · log₁₀ 2.
    // The estimate may be off by 1 either way, but our recursion is
    // structured so the natural digit count of `hi` is whatever it is —
    // we don't rely on the estimate being exact for `hi`, only for
    // picking the split point.  Lo's padded length is the split, so it
    // is exact by construction.
    let est_digits = decimal_digits_estimate(n.bits());
    let m = est_digits / 2;
    if m == 0 {
        write_decimal_naive(n, out);
        return;
    }
    let splitter = Integer::u_pow_u(10, m);
    let (hi, lo) = div_rem_signed(n, &splitter);
    // Hi: just emit it (no padding — caller wants the natural-length
    // top half).
    to_decimal_top(&hi, out);
    // Lo: must contribute exactly `m` characters.
    to_decimal_padded(&lo, m as usize, out);
}

/// Append exactly `want_len` decimal characters representing `n`,
/// padding with leading zeros if needed.  Precondition: `n < 10^want_len`.
fn to_decimal_padded(n: &Integer, want_len: usize, out: &mut String) {
    if n.is_zero() {
        for _ in 0..want_len {
            out.push('0');
        }
        return;
    }
    if n.limbs.len() <= TO_STRING_DC_THRESHOLD {
        let mut buf = String::new();
        write_decimal_naive(n, &mut buf);
        debug_assert!(buf.len() <= want_len, "padded leaf overflowed want_len");
        for _ in buf.len()..want_len {
            out.push('0');
        }
        out.push_str(&buf);
        return;
    }
    let m = want_len / 2;
    let splitter = Integer::u_pow_u(10, m as u32);
    let (hi, lo) = div_rem_signed(n, &splitter);
    to_decimal_padded(&hi, want_len - m, out);
    to_decimal_padded(&lo, m, out);
}

/// Existing peel-by-10^19 loop — used as the D&C base case once the
/// input is small enough that this is cheaper than another recursion.
fn write_decimal_naive(n: &Integer, out: &mut String) {
    use std::fmt::Write;
    debug_assert!(!n.is_zero() && !n.negative);
    let mut mag = n.clone();
    let mut chunks: Vec<u64> = Vec::new();
    while !mag.is_zero() {
        let (q, r) = div_rem_small(&mag, DECIMAL_CHUNK);
        chunks.push(r);
        mag = q;
    }
    let last = chunks.pop().expect("nonzero has at least one chunk");
    write!(out, "{last}").unwrap();
    while let Some(c) = chunks.pop() {
        write!(out, "{c:0w$}", w = DECIMAL_CHUNK_DIGITS as usize).unwrap();
    }
}

/// Quotient and remainder of `a / b` as `Integer`s.  Helper for D&C
/// stringification; works on any non-negative `a` and `b`.
///
/// Dispatches: large operands use Newton–Raphson reciprocal division
/// (O(M(N)) via Karatsuba); small ones stay on Knuth's Algorithm D,
/// whose constants win below the crossover.
fn div_rem_signed(a: &Integer, b: &Integer) -> (Integer, Integer) {
    debug_assert!(!a.negative && !b.negative && !b.is_zero());
    if b.limbs.len() >= NEWTON_DIV_THRESHOLD
        && a.limbs.len() >= NEWTON_DIV_THRESHOLD
        && cmp_mag(&a.limbs, &b.limbs) != Ordering::Less
    {
        return div_rem_newton(a, b);
    }
    let (q, r) = div_rem_mag(&a.limbs, &b.limbs);
    let qi = if q.is_empty() {
        Integer::new()
    } else {
        Integer { limbs: q, negative: false }
    };
    let ri = if r.is_empty() {
        Integer::new()
    } else {
        Integer { limbs: r, negative: false }
    };
    (qi, ri)
}

/// Crossover above which `div_rem_signed` uses Newton–Raphson reciprocal
/// division instead of Knuth Algorithm D.  Below this size Knuth's
/// lower constants (no Float setup, no reciprocal iteration) win; above
/// it NR's O(M(N)) asymptotic dominates Knuth's O(N²).  Picked to land
/// well above Karatsuba's own threshold so the multiplications inside
/// NR are themselves subquadratic.
const NEWTON_DIV_THRESHOLD: usize = 64;

/// Newton–Raphson reciprocal integer division for non-negative operands.
/// Returns `(a / b, a mod b)` exactly.
///
/// Strategy:
/// 1. Compute `1/b` as a Float at `q_bits + guard` precision via the
///    existing precision-doubling NR reciprocal.
/// 2. `q ≈ floor(a · recip)`.  With enough guard bits the floored
///    quotient is exact, or off by ±1.
/// 3. Compute `r = a − q·b` exactly and correct `q` by ±1 if needed.
///
/// Cost is dominated by two Karatsuba multiplies plus one reciprocal
/// (itself a geometric series of multiplies summing to O(M(N))).  That
/// is asymptotically O(M(N)) ≈ O(N^1.585) versus Knuth's O(N²); the
/// crossover sits around `NEWTON_DIV_THRESHOLD` limbs.
fn div_rem_newton(a: &Integer, b: &Integer) -> (Integer, Integer) {
    debug_assert!(!a.negative && !b.negative && !b.is_zero());
    debug_assert!(cmp_mag(&a.limbs, &b.limbs) != Ordering::Less);

    let a_bits = a.bits();
    let b_bits = b.bits();
    let q_bits = a_bits - b_bits + 1;

    // Working precision.  64 guard bits soaks up:
    // (a) the residual error of the final Newton iteration in
    //     `reciprocal_at_prec` (a few ulps), and
    // (b) the rounding of `a` into a `prec`-bit Float mantissa before
    //     the multiplication.
    // Combined they leave the floored quotient off by at most ±1, which
    // the correction loop handles.
    let prec = q_bits + 64;

    // Reciprocal of b at `prec` bits.
    let b_float: crate::float::Float = b.into();
    let recip = b_float.reciprocal_at_prec(prec);

    // a · recip ≈ a / b.  Truncate `a` to `prec` bits first so the
    // multiplication doesn't operate on a's full mantissa — we don't
    // need its low bits for an integer quotient of ~q_bits magnitude.
    let a_float: crate::float::Float = a.into();
    let a_trunc = a_float.truncated_to_prec(prec);
    let q_float = a_trunc.mul_at_prec(&recip, prec);

    // Floor.  q_float ≥ 0 (both inputs non-negative), so Round::Down
    // is plain truncation toward zero.
    let (mut q, _) = q_float
        .to_integer_round(crate::Round::Down)
        .expect("to_integer_round returned None for finite Float");

    // Exact remainder, then correct q by ±1 if the Float-based estimate
    // was off.  With prec = q_bits + 64 the loops fire zero or one
    // time in practice; written as loops as defence in depth.
    let one = Integer::from(1_u32);
    let mut r = a - &(&q * b);
    while r.negative && !r.is_zero() {
        q = &q - &one;
        r = &r + b;
    }
    while !r.negative && cmp_mag(&r.limbs, &b.limbs) != Ordering::Less {
        q = &q + &one;
        r = &r - b;
    }

    debug_assert!(!r.negative);
    debug_assert!(r.is_zero() || cmp_mag(&r.limbs, &b.limbs) == Ordering::Less);

    (q, r)
}

/// Decimal-digit count estimate from bit count using `log₁₀ 2`.
/// Off-by-one in either direction is fine for split-point selection
/// (top half's natural length absorbs the slack).
fn decimal_digits_estimate(bits: u64) -> u32 {
    // log₁₀ 2 = 0.301029995663981...
    ((bits as f64) * 0.301_029_995_663_981).ceil() as u32
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

    #[test]
    fn to_string_dc_matches_naive_on_known_values() {
        // 10^k - 1 for various k: hits a range straddling the D&C
        // threshold and stresses leading-zero / boundary cases.
        for k in [10, 50, 100, 500, 1000, 5000].iter().copied() {
            let mut n = Integer::u_pow_u(10, k);
            n = &n - &Integer::from(1_u32);
            let s = n.to_string_radix(10);
            // 10^k - 1 is k nines.
            let expected: String = std::iter::repeat('9').take(k as usize).collect();
            assert_eq!(s, expected, "10^{k} - 1 stringification");
        }
    }

    #[test]
    fn to_string_dc_round_trip_large() {
        // Build a non-trivial large integer (no special structure), then
        // verify parse_radix(to_string_radix(n)) == n.  The size (~10K
        // decimal digits) puts us well past the D&C threshold so the
        // recursive path is exercised end-to-end.
        let mut n = Integer::from(0xDEAD_BEEF_CAFE_F00D_u64);
        for _ in 0..10 {
            n = &n * &n;
        }
        // n now has ~16K bits ≈ ~4900 decimal digits.
        let s = n.to_string_radix(10);
        let back = Integer::parse_radix(&s, 10).unwrap();
        assert_eq!(back, n, "decimal round-trip failed");
    }

    #[test]
    fn to_string_dc_lo_zero_pads_correctly() {
        // 10^200 itself: in any split the low half becomes exactly zero,
        // which must produce m leading zeros — a common bug-magnet in
        // D&C base conversion.
        let n = Integer::u_pow_u(10, 200);
        let s = n.to_string_radix(10);
        let mut expected = String::from("1");
        for _ in 0..200 {
            expected.push('0');
        }
        assert_eq!(s, expected);
    }

    // ----- Newton-Raphson division ------------------------------------

    /// Cheap deterministic Integer generator for property-style tests.
    /// Produces a value of roughly `n_limbs` limbs from a seed.
    fn make_int(seed: u64, n_limbs: usize) -> Integer {
        let mut limbs = Vec::with_capacity(n_limbs);
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for _ in 0..n_limbs {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            limbs.push(s);
        }
        // Ensure top limb is nonzero so we hit `n_limbs` exactly.
        if let Some(top) = limbs.last_mut() {
            *top |= 1u64 << 63;
        } else {
            limbs.push(1);
        }
        Integer { limbs, negative: false }
    }

    fn assert_divmod_matches_knuth(a: &Integer, b: &Integer) {
        let (q_nr, r_nr) = div_rem_newton(a, b);
        let (q_k, r_k) = div_rem_mag(&a.limbs, &b.limbs);
        let q_k = Integer { limbs: q_k, negative: false };
        let r_k = Integer { limbs: r_k, negative: false };
        assert_eq!(q_nr.to_string_radix(16), q_k.to_string_radix(16),
            "quotient mismatch: a={} b={}", a.to_string_radix(16), b.to_string_radix(16));
        assert_eq!(r_nr.to_string_radix(16), r_k.to_string_radix(16),
            "remainder mismatch");
        // Sanity: q·b + r == a.
        let recon = &(&q_nr * b) + &r_nr;
        assert_eq!(recon.to_string_radix(16), a.to_string_radix(16),
            "q·b + r != a");
    }

    #[test]
    fn newton_div_matches_knuth_small() {
        // Just above the threshold so the NR path is exercised.
        for seed in 0..16u64 {
            let a = make_int(seed * 2 + 1, 200);
            let b = make_int(seed * 2 + 2, 100);
            assert_divmod_matches_knuth(&a, &b);
        }
    }

    #[test]
    fn newton_div_matches_knuth_unbalanced() {
        // Very large numerator, small (but above threshold) divisor.
        let a = make_int(7, 2000);
        let b = make_int(11, 70);
        assert_divmod_matches_knuth(&a, &b);
    }

    #[test]
    fn newton_div_matches_knuth_near_equal() {
        // a only slightly larger than b — quotient is small but the
        // Float estimate must still land within ±1.
        let b = make_int(13, 500);
        let a = &(&b * &Integer::from(7_u32)) + &Integer::from(3_u32);
        assert_divmod_matches_knuth(&a, &b);
    }

    #[test]
    fn newton_div_power_of_ten_splitter() {
        // The exact pattern hit by D&C base conversion: divide a large
        // random integer by 10^m, where m makes 10^m straddle the
        // threshold.
        for digits in [500_u32, 1500, 4000] {
            let b = Integer::u_pow_u(10, digits);
            let a = make_int(digits as u64, b.limbs.len() * 2 + 5);
            assert_divmod_matches_knuth(&a, &b);
        }
    }

    #[test]
    fn newton_div_large() {
        // ~64K limb (~4M bit) numerator over a ~32K limb divisor —
        // representative of the top-level division in 1M-digit base
        // conversion.  Confirms correctness at production sizes.
        let b = make_int(101, 32_000);
        let a = make_int(103, 64_000);
        assert_divmod_matches_knuth(&a, &b);
    }
}
