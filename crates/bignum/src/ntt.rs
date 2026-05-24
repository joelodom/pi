//! Number-theoretic transform (NTT) over the Goldilocks prime
//! `P = 2^64 - 2^32 + 1`, used for asymptotically fast integer
//! multiplication via cyclic convolution.
//!
//! Why Goldilocks: `P - 1 = 2^32 · (2^32 - 1)`, so the multiplicative
//! group has a `2^32`-element subgroup of roots of unity.  That lets a
//! single-prime NTT handle transform lengths up to `2^32`, which —
//! combined with 16-bit-per-coefficient packing — covers operands up
//! to roughly `2^33` bytes ≈ `2 · 10^10` decimal digits.  Beyond that
//! we will either drop to 8-bit packing (extending range to roughly
//! `10^11` digits) or move to a multi-prime NTT with CRT recombination.
//! Both are future work; the current single-prime implementation
//! comfortably covers the in-memory range the user cares about
//! (100M – 10B digits).
//!
//! Data-structure note for trillion-digit work: the NTT array is one
//! large `Vec<u64>`; pack/unpack and pointwise multiply iterate over
//! contiguous chunks, and the early butterfly passes are also
//! chunk-local.  A future disk-backed variant should replace
//! [`Vec<u64>`] with a `[u64]`-shaped buffer abstraction (mmap,
//! out-of-core paged chunks, etc.) and split the late, long-stride
//! butterfly passes into a six-step / four-step decomposition with
//! intermediate transpose — both are deferred until single-machine
//! memory becomes the limit.

#![allow(dead_code)]

use rayon::prelude::*;

// =====================================================================
// Goldilocks field: P = 2^64 - 2^32 + 1
// =====================================================================

/// The Goldilocks prime.
pub(crate) const P: u64 = 0xFFFF_FFFF_0000_0001;

/// `P - 2^64`, equivalently `2^32 - 1`.  Often appears in reductions
/// because `2^64 ≡ EPSILON (mod P)`.
const EPSILON: u64 = 0xFFFF_FFFF;

/// A multiplicative-group generator of `Z/PZ*`.  `7` is the standard
/// choice (smallest small prime that is a primitive root mod P).
const GENERATOR: u64 = 7;

/// Reduce `x mod P` for `x` in `[0, P + small)`.  Single conditional
/// subtraction.  No-op when `x < P`.
#[inline(always)]
fn canonicalize(x: u64) -> u64 {
    if x >= P { x - P } else { x }
}

/// `(a + b) mod P`, both inputs in `[0, P)`.
#[inline(always)]
fn add(a: u64, b: u64) -> u64 {
    let (sum, carry) = a.overflowing_add(b);
    // If we overflowed u64, the wrap value is `sum = a + b - 2^64`,
    // and `2^64 ≡ EPSILON (mod P)`, so add EPSILON to recover.
    let r = if carry { sum.wrapping_add(EPSILON) } else { sum };
    canonicalize(r)
}

/// `(a - b) mod P`, both inputs in `[0, P)`.
#[inline(always)]
fn sub(a: u64, b: u64) -> u64 {
    let (diff, borrow) = a.overflowing_sub(b);
    if borrow {
        // Underflow: wrap value is `a - b + 2^64`.  True mod-P value is
        // `a - b + P`, so subtract `2^64 - P = -EPSILON`, i.e. subtract
        // EPSILON from the wrapped result.  Provably safe (no second
        // underflow) given `a, b ∈ [0, P)`.
        diff.wrapping_sub(EPSILON)
    } else {
        diff
    }
}

/// `(a * b) mod P`, both inputs in `[0, P)`.
///
/// Uses the Goldilocks identity:
/// `x ≡ x_lo - x_hi_hi + x_hi_lo · (2^32 - 1) (mod P)`,
/// where `x` is split into 64-bit halves `(x_lo, x_hi)` and `x_hi` is
/// further split into 32-bit halves `(x_hi_lo, x_hi_hi)`.
#[inline(always)]
fn mul(a: u64, b: u64) -> u64 {
    reduce128((a as u128) * (b as u128))
}

#[inline(always)]
fn reduce128(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let x_hi_hi = x_hi >> 32;
    let x_hi_lo = (x_hi as u32) as u64;

    // Step 1: t0 = (x_lo - x_hi_hi) mod P.
    let (sub_raw, borrow) = x_lo.overflowing_sub(x_hi_hi);
    let t0 = if borrow {
        // `sub_raw` is `x_lo - x_hi_hi + 2^64`.  Adjusting by `-EPSILON`
        // gives the canonical value `x_lo - x_hi_hi + P`, in `[0, P)`.
        // The subtraction can't underflow because `sub_raw` lies in
        // `[2^64 - 2^32 + 1, 2^64)` when `borrow` fires.
        sub_raw - EPSILON
    } else {
        sub_raw
    };
    let t0 = canonicalize(t0);

    // Step 2: t1 = x_hi_lo * EPSILON.  Fits in u64 because both are
    // strictly less than 2^32.
    let t1 = x_hi_lo * EPSILON;
    // t1 ∈ [0, (2^32 - 1)^2] = [0, 2^64 - 2^33 + 1], which is < P.

    // Step 3: result = (t0 + t1) mod P.
    let (sum, carry) = t0.overflowing_add(t1);
    let result = if carry {
        // Overflow: wrap value is `t0 + t1 - 2^64`, mod P add EPSILON.
        // Per the bound on `t1`, this add can't itself overflow u64.
        sum + EPSILON
    } else {
        sum
    };
    canonicalize(result)
}

/// `base ^ exp (mod P)` via square-and-multiply.
fn pow(mut base: u64, mut exp: u64) -> u64 {
    let mut acc = 1_u64;
    while exp > 0 {
        if exp & 1 == 1 {
            acc = mul(acc, base);
        }
        exp >>= 1;
        if exp > 0 {
            base = mul(base, base);
        }
    }
    acc
}

/// Modular multiplicative inverse via Fermat's little theorem:
/// `a^(P-2) ≡ a^(-1) (mod P)`.  Slower than extended-Euclidean but
/// the simplicity is worth it — we only invert a constant per NTT.
fn inv(a: u64) -> u64 {
    debug_assert!(a != 0, "inverse of zero is undefined");
    pow(a, P - 2)
}

/// Return a primitive `n`-th root of unity in `Z/PZ`, where `n` must
/// be a power of two no larger than `2^32`.
fn primitive_root_of_unity(n: u64) -> u64 {
    debug_assert!(n.is_power_of_two() && n <= (1u64 << 32));
    // g^((P-1) / n) is an n-th root of unity, primitive when n divides
    // the order of g (which equals P-1 since g is a generator).
    pow(GENERATOR, (P - 1) / n)
}

// =====================================================================
// Radix-2 Cooley-Tukey NTT (decimation-in-time)
// =====================================================================

/// In-place bit-reversal permutation.  Length must be a power of two.
fn bit_reverse(a: &mut [u64]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let log_n = n.trailing_zeros();
    if log_n == 0 {
        return;
    }
    for i in 0..n {
        let j = (i as u64).reverse_bits() >> (64 - log_n);
        if (i as u64) < j {
            a.swap(i, j as usize);
        }
    }
}

/// Forward NTT in-place: `A[k] = Σ_i a[i] · ω^(i·k)` where ω is a
/// primitive `n`-th root of unity in Z/PZ.  Length must be a power of
/// two `n ≤ 2^32`.
pub(crate) fn ntt_forward(a: &mut [u64]) {
    let n = a.len();
    assert!(n.is_power_of_two() && n <= (1usize << 32));
    bit_reverse(a);
    butterflies(a, /*inverse=*/ false);
}

/// Inverse NTT in-place: `a[i] = (1/n) · Σ_k A[k] · ω^(-i·k)`.
pub(crate) fn ntt_inverse(a: &mut [u64]) {
    let n = a.len();
    assert!(n.is_power_of_two() && n <= (1usize << 32));
    bit_reverse(a);
    butterflies(a, /*inverse=*/ true);
    // Final scale by `1/n`.  Embarrassingly parallel.
    let n_inv = inv(n as u64);
    if n >= crate::config::ntt_parallel_pointwise_threshold() {
        a.par_iter_mut().for_each(|x| *x = mul(*x, n_inv));
    } else {
        for x in a.iter_mut() {
            *x = mul(*x, n_inv);
        }
    }
}

/// Body of the Cooley-Tukey iteration: log₂(n) passes of butterflies.
///
/// Each pass precomputes a `half`-element twiddle table so the inner
/// butterfly loop has no serial dependency between iterations.
///
/// Three parallel regimes, all wrapping the same butterfly body:
///   * Late passes (few groups but each is huge) — split lo/hi inside
///     each group and parallelize the inner index range.
///   * Mid passes (group size in the L2-resident range) — one rayon
///     task per group via `par_chunks_mut(len)`.
///   * Early passes (groups so small that one task = one group would
///     burn task overhead) — bundle many groups into each task.
///   * Very small inputs — sequential.
fn butterflies(a: &mut [u64], inverse: bool) {
    let n = a.len();
    // Read once per transform — the config is immutable during the
    // run, and one relaxed load per pass would still be cheap, but
    // caching keeps the inner branches comparing against a register.
    let target_task = crate::config::ntt_target_task_size();
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let omega_len_base = primitive_root_of_unity(len as u64);
        let omega_len = if inverse { inv(omega_len_base) } else { omega_len_base };

        // Twiddle table.  Serial chain of `half - 1` muls — cheap
        // relative to the `n / 2` muls in the actual pass.
        let twiddles = build_twiddles(omega_len, half);

        let groups = n / len;

        if half >= target_task && groups < 4 {
            // Late pass: only a handful of groups, but each is large
            // enough that intra-group parallelism is worthwhile.  Each
            // butterfly touches one element from each half of the
            // group, so splitting at `half` gives two disjoint mutable
            // slices we can iterate in parallel chunks.
            for group in a.chunks_mut(len) {
                let (lo, hi) = group.split_at_mut(half);
                let tw = &twiddles;
                lo.par_chunks_mut(target_task)
                    .zip(hi.par_chunks_mut(target_task))
                    .enumerate()
                    .for_each(|(chunk_idx, (lo_chunk, hi_chunk))| {
                        let base = chunk_idx * target_task;
                        for j in 0..lo_chunk.len() {
                            let u = lo_chunk[j];
                            let t = mul(tw[base + j], hi_chunk[j]);
                            lo_chunk[j] = add(u, t);
                            hi_chunk[j] = sub(u, t);
                        }
                    });
            }
        } else {
            // Standard: par_chunks_mut by (possibly bundled) groups.
            let task_size = if len >= target_task {
                len
            } else if n >= target_task {
                target_task
            } else {
                n
            };
            if n > task_size {
                a.par_chunks_mut(task_size).for_each(|big_chunk| {
                    for group in big_chunk.chunks_mut(len) {
                        butterfly_group(group, &twiddles, half);
                    }
                });
            } else {
                for group in a.chunks_mut(len) {
                    butterfly_group(group, &twiddles, half);
                }
            }
        }

        len *= 2;
    }
}

#[inline(always)]
fn butterfly_group(chunk: &mut [u64], twiddles: &[u64], half: usize) {
    for j in 0..half {
        let u = chunk[j];
        let t = mul(twiddles[j], chunk[j + half]);
        chunk[j] = add(u, t);
        chunk[j + half] = sub(u, t);
    }
}

fn build_twiddles(omega: u64, count: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(count);
    if count == 0 {
        return out;
    }
    let mut w = 1_u64;
    out.push(w);
    for _ in 1..count {
        w = mul(w, omega);
        out.push(w);
    }
    out
}

// =====================================================================
// Bit-packing: limbs ↔ NTT coefficients
// =====================================================================

/// Bits per NTT coefficient.  See module header for the size envelope
/// this implies.  16 keeps convolution sums comfortably below P for
/// any transform length up to `2^32`.
const BITS_PER_COEFF: usize = 16;
const COEFFS_PER_LIMB: usize = 64 / BITS_PER_COEFF; // 4

/// Pack little-endian `limbs` into the first `limbs.len() * 4`
/// coefficients of `coeffs`, zero-padding the rest.  Each coefficient
/// holds a 16-bit slice of the magnitude.  Parallel above
/// `PARALLEL_PACK_THRESHOLD` limbs.
fn pack(limbs: &[u64], coeffs: &mut [u64]) {
    let needed = limbs.len() * COEFFS_PER_LIMB;
    assert!(coeffs.len() >= needed, "coeffs buffer too small for pack");
    let (live, tail) = coeffs.split_at_mut(needed);
    if limbs.len() >= crate::config::ntt_parallel_pack_threshold() {
        live.par_chunks_mut(COEFFS_PER_LIMB)
            .zip(limbs.par_iter())
            .for_each(|(chunk, &limb)| {
                chunk[0] =  limb        & 0xFFFF;
                chunk[1] = (limb >> 16) & 0xFFFF;
                chunk[2] = (limb >> 32) & 0xFFFF;
                chunk[3] = (limb >> 48) & 0xFFFF;
            });
        tail.par_iter_mut().for_each(|c| *c = 0);
    } else {
        for (chunk, &limb) in live.chunks_mut(COEFFS_PER_LIMB).zip(limbs.iter()) {
            chunk[0] =  limb        & 0xFFFF;
            chunk[1] = (limb >> 16) & 0xFFFF;
            chunk[2] = (limb >> 32) & 0xFFFF;
            chunk[3] = (limb >> 48) & 0xFFFF;
        }
        for c in tail.iter_mut() {
            *c = 0;
        }
    }
}

// `parallel_pack_threshold` lives in crate::config.

/// Unpack convolution-output coefficients back into a little-endian
/// u64 limb vector.  Each input coefficient sits at bit position
/// `i · BITS_PER_COEFF`; values may be much larger than `2^16` (up to
/// roughly `N · 2^32`), so carries cascade across many limbs.  We use
/// a u128 running accumulator to absorb up to four shifted
/// contributions per output limb plus the high-half carry from the
/// previous limb.
fn unpack(coeffs: &[u64]) -> Vec<u64> {
    if coeffs.is_empty() {
        return Vec::new();
    }
    let total_bits = (coeffs.len() - 1) * BITS_PER_COEFF + 64;
    let approx_limbs = total_bits / 64 + 2;
    let mut result: Vec<u64> = Vec::with_capacity(approx_limbs);

    let mut carry: u128 = 0;
    let mut i = 0;
    while i < coeffs.len() {
        // Fold the next up-to-4 coefficients (at offsets 0, 16, 32, 48)
        // into the running u128 carry.  Worst case bound on carry after
        // this block: 2^(48 + 60) + 2^63 < 2^110, fits in u128.
        for offset_idx in 0..COEFFS_PER_LIMB {
            if i >= coeffs.len() {
                break;
            }
            let bit_offset = offset_idx * BITS_PER_COEFF;
            carry += (coeffs[i] as u128) << bit_offset;
            i += 1;
        }
        result.push(carry as u64);
        carry >>= 64;
    }
    while carry != 0 {
        result.push(carry as u64);
        carry >>= 64;
    }
    // Magnitude convention: no trailing zero limbs.
    while result.last() == Some(&0) {
        result.pop();
    }
    result
}

// =====================================================================
// Top-level multiplication
// =====================================================================

/// NTT-based magnitude multiplication.  Both inputs are little-endian
/// limb arrays representing non-negative integers; the result is the
/// limb array of `|a| · |b|`, trimmed of leading zeros.
///
/// Cost: `O(N log N)` field operations for `N ≈ 4 · max(|a|,|b|)`
/// coefficients, plus the linear pack/unpack passes.  The single-prime
/// Goldilocks transform requires `N ≤ 2^32`, which corresponds to
/// inputs up to roughly `2^33` bytes ≈ `2 · 10^10` decimal digits.
pub(crate) fn mul_mag_ntt(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let n_a_coeffs = a.len() * COEFFS_PER_LIMB;
    let n_b_coeffs = b.len() * COEFFS_PER_LIMB;
    let n_out = n_a_coeffs + n_b_coeffs - 1;
    let n = n_out.next_power_of_two();
    assert!(
        n <= (1usize << 32),
        "operand too large for single-prime Goldilocks NTT (need N={} > 2^32)",
        n,
    );

    let mut pa = vec![0u64; n];
    let mut pb = vec![0u64; n];
    // Pack both inputs in parallel — the two passes are independent and
    // each one is itself internally parallel above the pack threshold.
    rayon::join(|| pack(a, &mut pa), || pack(b, &mut pb));

    // Forward transforms — also independent, also internally parallel.
    rayon::join(|| ntt_forward(&mut pa), || ntt_forward(&mut pb));

    // Pointwise multiply: each lane independent, perfectly parallel.
    if n >= crate::config::ntt_parallel_pointwise_threshold() {
        pa.par_iter_mut().zip(pb.par_iter()).for_each(|(x, y)| {
            *x = mul(*x, *y);
        });
    } else {
        for (x, y) in pa.iter_mut().zip(pb.iter()) {
            *x = mul(*x, *y);
        }
    }
    // pb no longer needed.  Free its half-gigabyte mantissa space
    // before the inverse NTT does its own temporary allocations.
    drop(pb);

    ntt_inverse(&mut pa);

    unpack(&pa[..n_out])
}

// `parallel_pointwise_threshold` lives in crate::config.

// =====================================================================
// Tests for field arithmetic + NTT + pack/unpack + top-level multiply
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_basic() {
        assert_eq!(canonicalize(0), 0);
        assert_eq!(canonicalize(1), 1);
        assert_eq!(canonicalize(P - 1), P - 1);
        assert_eq!(canonicalize(P), 0);
        assert_eq!(canonicalize(P + 1), 1);
    }

    #[test]
    fn add_basic() {
        assert_eq!(add(0, 0), 0);
        assert_eq!(add(1, 2), 3);
        assert_eq!(add(P - 1, 1), 0);
        assert_eq!(add(P - 1, P - 1), P - 2);
        // Stress: values near 2^64.
        assert_eq!(add(P - 1, P - 1), P - 2);
    }

    #[test]
    fn sub_basic() {
        assert_eq!(sub(0, 0), 0);
        assert_eq!(sub(5, 3), 2);
        assert_eq!(sub(0, 1), P - 1);
        assert_eq!(sub(0, P - 1), 1);
    }

    #[test]
    fn mul_basic() {
        assert_eq!(mul(0, 12345), 0);
        assert_eq!(mul(1, 12345), 12345);
        assert_eq!(mul(2, 3), 6);
        // (-1) * (-1) = 1
        assert_eq!(mul(P - 1, P - 1), 1);
        // P/2 * 2 = P-1 ... wait, P is odd. Try (P+1)/2 * 2 = P + 1 ≡ 1.
        let half = (P + 1) / 2;
        assert_eq!(mul(half, 2), 1);
    }

    #[test]
    fn reduce128_matches_naive_mod() {
        // Compare against naive `% P` for a battery of inputs.
        let cases: &[u128] = &[
            0,
            1,
            P as u128 - 1,
            P as u128,
            P as u128 + 1,
            2 * P as u128,
            2 * P as u128 - 1,
            (P as u128) * (P as u128) - 1,
            (P as u128) * (P as u128 - 1),
            u128::MAX,
            u128::MAX - 1,
            // Some pseudo-random spreads.
            0x1234_5678_9ABC_DEF0_1122_3344_5566_7788,
            0xFFFF_0000_FFFF_0000_FFFF_0000_FFFF_0000,
            0xAAAA_BBBB_CCCC_DDDD_EEEE_FFFF_0000_1111,
        ];
        for &x in cases {
            let naive = (x % P as u128) as u64;
            let fast = reduce128(x);
            assert_eq!(fast, naive, "reduce128({x:#x}) mismatch");
        }
    }

    #[test]
    fn mul_matches_naive_mod() {
        // Compare a stress battery of mul(a, b) against (a * b) % P.
        let xs: &[u64] = &[
            0, 1, 2, 100, EPSILON, EPSILON + 1, P - 1,
            P / 2, P / 3, 0xDEAD_BEEF_CAFE_BABE & (P - 1),
            0x1234_5678_9ABC_DEF0 & (P - 1),
        ];
        for &a in xs {
            for &b in xs {
                let naive = ((a as u128) * (b as u128) % P as u128) as u64;
                let fast = mul(a, b);
                assert_eq!(fast, naive, "mul({a:#x}, {b:#x}) mismatch");
            }
        }
    }

    #[test]
    fn pow_fermat_identity() {
        // a^(P-1) ≡ 1 (mod P) for any a not divisible by P.
        for &a in &[2_u64, 3, 7, 12345, P - 1, P / 2] {
            assert_eq!(pow(a, P - 1), 1, "Fermat failed for a={a}");
        }
    }

    #[test]
    fn inv_round_trips() {
        for &a in &[1_u64, 2, 3, 7, 12345, P - 1, P / 2] {
            let i = inv(a);
            assert_eq!(mul(a, i), 1, "inv({a}) failed: a*inv(a) != 1");
        }
    }

    // ---- NTT tests --------------------------------------------------

    fn random_vec(n: usize, seed: u64) -> Vec<u64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                s % P
            })
            .collect()
    }

    fn naive_convolution(a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len() + b.len() - 1;
        let mut out = vec![0_u64; n];
        for i in 0..a.len() {
            for j in 0..b.len() {
                out[i + j] = add(out[i + j], mul(a[i], b[j]));
            }
        }
        out
    }

    #[test]
    fn bit_reverse_known() {
        // Length 8: index i (in binary 3 bits) maps to bit-reversed.
        // 0=000, 1=100, 2=010, 3=110, 4=001, 5=101, 6=011, 7=111
        let mut a = vec![0_u64, 1, 2, 3, 4, 5, 6, 7];
        bit_reverse(&mut a);
        assert_eq!(a, vec![0, 4, 2, 6, 1, 5, 3, 7]);
    }

    #[test]
    fn ntt_round_trip_small() {
        for log_n in [1_usize, 2, 3, 4, 8, 12] {
            let n = 1usize << log_n;
            let original = random_vec(n, log_n as u64);
            let mut a = original.clone();
            ntt_forward(&mut a);
            ntt_inverse(&mut a);
            assert_eq!(a, original, "round trip failed at n=2^{log_n}");
        }
    }

    #[test]
    fn ntt_convolution_matches_naive_small() {
        // Multiply [1, 2] * [3, 4] = [3, 10, 8]
        let a_orig: Vec<u64> = vec![1, 2];
        let b_orig: Vec<u64> = vec![3, 4];
        let expected = naive_convolution(&a_orig, &b_orig);

        let n_out = a_orig.len() + b_orig.len() - 1;
        let n = n_out.next_power_of_two();
        let mut a = a_orig.clone();
        a.resize(n, 0);
        let mut b = b_orig.clone();
        b.resize(n, 0);

        ntt_forward(&mut a);
        ntt_forward(&mut b);
        for (x, y) in a.iter_mut().zip(b.iter()) {
            *x = mul(*x, *y);
        }
        ntt_inverse(&mut a);

        assert_eq!(&a[..n_out], &expected[..]);
    }

    #[test]
    fn ntt_convolution_matches_naive_random() {
        // Larger random convolution at multiple lengths.
        for (in_len, seed) in [(7_usize, 1u64), (15, 2), (31, 3), (63, 4), (127, 5)] {
            let a_orig: Vec<u64> = random_vec(in_len, seed)
                .into_iter()
                // Keep values small enough that naive convolution sums stay <P.
                .map(|x| x & 0xFFFF)
                .collect();
            let b_orig: Vec<u64> = random_vec(in_len, seed ^ 0xFF)
                .into_iter()
                .map(|x| x & 0xFFFF)
                .collect();
            let expected = naive_convolution(&a_orig, &b_orig);

            let n_out = a_orig.len() + b_orig.len() - 1;
            let n = n_out.next_power_of_two();
            let mut a = a_orig.clone();
            a.resize(n, 0);
            let mut b = b_orig.clone();
            b.resize(n, 0);

            ntt_forward(&mut a);
            ntt_forward(&mut b);
            for (x, y) in a.iter_mut().zip(b.iter()) {
                *x = mul(*x, *y);
            }
            ntt_inverse(&mut a);

            assert_eq!(&a[..n_out], &expected[..],
                "convolution mismatch at in_len={in_len}");
        }
    }

    // ---- pack / unpack ----------------------------------------------

    #[test]
    fn pack_unpack_round_trip_one_limb() {
        let limbs: Vec<u64> = vec![0xDEAD_BEEF_CAFE_BABE];
        let mut coeffs = vec![0u64; limbs.len() * COEFFS_PER_LIMB + 3];
        pack(&limbs, &mut coeffs);
        let unpacked = unpack(&coeffs);
        assert_eq!(unpacked, limbs);
    }

    #[test]
    fn pack_unpack_round_trip_multi_limb() {
        let limbs: Vec<u64> = vec![
            0xDEAD_BEEF_CAFE_BABE,
            0x1234_5678_9ABC_DEF0,
            0x5555_5555_AAAA_AAAA,
            0xFFFF_FFFF_FFFF_FFFF,
            0x0000_0000_0000_0001,
        ];
        let mut coeffs = vec![0u64; limbs.len() * COEFFS_PER_LIMB + 5];
        pack(&limbs, &mut coeffs);
        let unpacked = unpack(&coeffs);
        assert_eq!(unpacked, limbs);
    }

    #[test]
    fn pack_unpack_random() {
        let limbs = random_vec(40, 12345);
        let mut coeffs = vec![0u64; limbs.len() * COEFFS_PER_LIMB + 10];
        pack(&limbs, &mut coeffs);
        let unpacked = unpack(&coeffs);
        // Strip trailing zeros from limbs too — unpack trims them.
        let mut expected = limbs.clone();
        while expected.last() == Some(&0) {
            expected.pop();
        }
        assert_eq!(unpacked, expected);
    }

    // ---- mul_mag_ntt vs schoolbook ----------------------------------

    /// Schoolbook magnitude multiply for the test oracle.  O(N^2).
    fn schoolbook(a: &[u64], b: &[u64]) -> Vec<u64> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0u64; a.len() + b.len()];
        for i in 0..a.len() {
            let mut carry: u64 = 0;
            for j in 0..b.len() {
                let prod = (a[i] as u128) * (b[j] as u128)
                    + (out[i + j] as u128)
                    + (carry as u128);
                out[i + j] = prod as u64;
                carry = (prod >> 64) as u64;
            }
            out[i + b.len()] = carry;
        }
        while out.last() == Some(&0) {
            out.pop();
        }
        out
    }

    #[test]
    fn ntt_mul_matches_schoolbook_tiny() {
        let a: Vec<u64> = vec![3];
        let b: Vec<u64> = vec![7];
        let want = schoolbook(&a, &b);
        let got = mul_mag_ntt(&a, &b);
        assert_eq!(got, want);
        assert_eq!(got, vec![21]);
    }

    #[test]
    fn ntt_mul_matches_schoolbook_small_random() {
        for (len_a, len_b, seed) in [
            (1_usize, 1_usize, 1_u64),
            (1, 4, 2),
            (4, 4, 3),
            (8, 5, 4),
            (32, 16, 5),
            (60, 60, 6),
            (200, 200, 7),
        ] {
            let a = random_vec(len_a, seed);
            let b = random_vec(len_b, seed ^ 0xFF);
            let want = schoolbook(&a, &b);
            let got = mul_mag_ntt(&a, &b);
            assert_eq!(got, want,
                "mismatch len_a={len_a} len_b={len_b} seed={seed}");
        }
    }

    #[test]
    fn ntt_mul_one_limb_max_values() {
        // u64::MAX * u64::MAX hits the largest possible limb values.
        let a: Vec<u64> = vec![u64::MAX];
        let b: Vec<u64> = vec![u64::MAX];
        let want = schoolbook(&a, &b);
        let got = mul_mag_ntt(&a, &b);
        assert_eq!(got, want);
    }

    #[test]
    fn omega_has_expected_order() {
        // omega_n has order exactly n for n a power of 2 ≤ 2^32.
        for log_n in [1_u32, 2, 4, 8, 16, 20] {
            let n = 1_u64 << log_n;
            let omega = primitive_root_of_unity(n);
            assert_eq!(pow(omega, n), 1, "omega^n != 1 (n=2^{log_n})");
            // Check primitivity: omega^(n/2) should be -1, not 1.
            if n > 1 {
                assert_eq!(pow(omega, n / 2), P - 1,
                    "omega^(n/2) != -1, omega not primitive (n=2^{log_n})");
            }
        }
    }
}
