//! Bailey-Borwein-Plouffe (BBP) hexadecimal digit extractor for π.
//!
//! Computes 8 consecutive hexadecimal digits of π's fractional expansion
//! at any position `n` without computing the preceding digits.  Used as
//! an independent oracle for verifying `pi.txt` files: an algorithm bug
//! in Chudnovsky or Gauss-Legendre cannot survive byte-by-byte agreement
//! with BBP at a few thousand randomly-chosen positions.
//!
//! ## Formula
//!
//! ```text
//! π  =  Σ_{k=0}^∞ (1/16^k) [ 4/(8k+1) − 2/(8k+4) − 1/(8k+5) − 1/(8k+6) ]
//! ```
//!
//! For position `n`, multiply through by `16^n` and take the fractional
//! part.  The terms split at `k = n`:
//!
//! * `k ≤ n`: contribute `(16^(n-k) mod (8k+r)) / (8k+r)`  — modular
//!   exponentiation on small denominators, fits in `u64`/`u128`.
//! * `k > n`: contribute `1 / (16^(k-n) · (8k+r))`  — vanishes rapidly,
//!   we sum these in `f64` and convert to fixed-point at the end.
//!
//! The running sum is held in `u64` fixed-point representing a fraction
//! in `[0, 1)` (the value `v` represents `v / 2^64`).  This gives full
//! 64-bit precision instead of f64's 53 bits, leaving plenty of room
//! for the top 32 bits (8 hex digits) to remain correct at the deep
//! positions we care about (~10^9).
//!
//! Cost: O(n) modular exponentiations of O(log n) multiplications each.

/// Extract 8 consecutive hex digits of π's fractional expansion starting
/// at position `n`, packed into a `u32` with the digit at position `n`
/// in the most significant nibble.
///
/// `n = 0` returns `0x243F_6A88` (the first 8 hex digits of π, since
/// π = 3.243F6A88…).
///
/// Reliability: at `n` up to ~10^9 (1 billion), all 8 hex digits are
/// expected to be correct; the accumulated error in the u64 fixed-point
/// sum is roughly `n` ULPs out of `2^64`, so the top `64 − log2(n)` bits
/// are safe.  At `n = 10^9` that leaves ~34 reliable bits.
pub fn hex_digits_at(n: u64) -> u32 {
    let s1 = series_fixed(1, n);
    let s4 = series_fixed(4, n);
    let s5 = series_fixed(5, n);
    let s6 = series_fixed(6, n);

    // Combined = 4·S(1) − 2·S(4) − S(5) − S(6), mod 2^64 (= mod 1 in
    // fixed-point — wrapping arithmetic does the modular reduction for us).
    let combined = (4_u64.wrapping_mul(s1))
        .wrapping_sub(2_u64.wrapping_mul(s4))
        .wrapping_sub(s5)
        .wrapping_sub(s6);

    // Top 32 bits = 8 hex digits starting at position n.
    (combined >> 32) as u32
}

/// Fractional part of `Σ_{k=0..∞} 16^(n-k) / (8k+j)`, in u64
/// fixed-point (the returned value `v` represents `v / 2^64`).
fn series_fixed(j: u64, n: u64) -> u64 {
    let mut sum: u64 = 0;

    // Modular sum: `k` in `0..=n`, contributing `(16^(n-k) mod (8k+j)) /
    // (8k+j)` as a fixed-point fraction.
    for k in 0..=n {
        let denom = 8 * k + j;
        let pow = mod_pow_16(n - k, denom);
        // (pow / denom) as a fraction times 2^64.  `pow < denom`, so the
        // u128 result fits in u64.
        let term_fixed = ((pow as u128) << 64) / (denom as u128);
        sum = sum.wrapping_add(term_fixed as u64);
    }

    // Tail: `k > n`.  Each term is bounded by `1/16^(k-n)` so the series
    // converges geometrically; ~20 terms gets us below the u64 fixed-point
    // representable threshold.
    let mut tail = 0.0_f64;
    for offset in 1_i32..=20 {
        let k = n + offset as u64;
        let denom = (8 * k + j) as f64;
        let factor = 16.0_f64.powi(offset);
        let term = 1.0 / (factor * denom);
        if term == 0.0 {
            break;
        }
        tail += term;
    }
    let tail_fixed = (tail * 2.0_f64.powi(64)) as u64;
    sum.wrapping_add(tail_fixed)
}

/// `16^exp mod m` via binary exponentiation in `u128`.
///
/// `m` can be anything up to `u64::MAX`; `u128` multiplication holds
/// `m * m` without overflow.
fn mod_pow_16(exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let m128 = m as u128;
    let mut result: u128 = 1;
    let mut base: u128 = 16 % m128;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = (result * base) % m128;
        }
        e >>= 1;
        if e > 0 {
            base = (base * base) % m128;
        }
    }
    result as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First 16 hex digits of π's fractional expansion: `243F6A8885A308D3`.
    /// (Anyone can spot-check: π = 3.243F6A8885A308D3…)
    const PI_HEX_PREFIX: [u8; 16] = [
        0x2, 0x4, 0x3, 0xF, 0x6, 0xA, 0x8, 0x8, 0x8, 0x5, 0xA, 0x3, 0x0, 0x8, 0xD, 0x3,
    ];

    #[test]
    fn first_8_hex_digits_packed() {
        // 0x243F6A88 = first 8 hex digits of π.
        assert_eq!(hex_digits_at(0), 0x243F_6A88);
    }

    #[test]
    fn next_8_hex_digits_packed() {
        // 0x85A308D3 = hex digits at positions 8..16.
        assert_eq!(hex_digits_at(8), 0x85A3_08D3);
    }

    #[test]
    fn top_nibble_matches_each_of_first_16_positions() {
        for (n, &expected) in PI_HEX_PREFIX.iter().enumerate() {
            let d = hex_digits_at(n as u64);
            let top = ((d >> 28) & 0xf) as u8;
            assert_eq!(
                top, expected,
                "position {n}: got {top:x}, expected {expected:x}"
            );
        }
    }

    #[test]
    fn consecutive_positions_overlap() {
        // The bottom 7 hex digits of hex_digits_at(n) should equal the
        // top 7 hex digits of hex_digits_at(n + 1).
        for n in 0..16_u64 {
            let dn = hex_digits_at(n);
            let dn1 = hex_digits_at(n + 1);
            let dn_low7 = dn & 0x0FFF_FFFF;
            let dn1_top7 = dn1 >> 4;
            assert_eq!(dn_low7, dn1_top7, "overlap mismatch at n={n}");
        }
    }

    #[test]
    fn deeper_position_exercises_loop() {
        // A medium-depth position; the cross-check against pi-hex.txt
        // later is the real test, but this confirms the loop executes
        // and the modular arithmetic doesn't overflow.
        let d = hex_digits_at(10_000);
        let _ = d;
    }
}
