//! Chudnovsky algorithm with binary splitting.
//!
//! The Chudnovsky brothers' formula is the fastest known practical series
//! for pi:
//!
//! ```text
//! 1/π = 12 · Σ_{k=0}^{∞}  (-1)^k (6k)! (Bk + A)
//!                          ──────────────────────
//!                          (3k)! (k!)^3 · C^{3k+3/2}
//! ```
//!
//! with `A = 13_591_409`, `B = 545_140_134`, `C = 640_320`.  Each term
//! contributes about 14.18 decimal digits.
//!
//! Binary splitting evaluates the partial sum
//! `Σ_{k=1}^{N} M_k L_k = T(1, N+1) / Q(1, N+1)` exactly using integer
//! arithmetic, where
//!
//! * `p_k = -(6k - 5)(6k - 1)(2k - 1)`
//! * `q_k = k³ · C³ / 24`
//! * `L_k = A + B·k`
//! * `M_k = M_{k-1} · p_k / q_k`,   `M_0 = 1`
//!
//! Including the `k = 0` term (which is just `A`) gives
//! `S = (A · Q + T) / Q`, so
//!
//! ```text
//! π = (426_880 · √10005 · Q) / (A · Q + T)
//! ```
//!
//! The integers `P`, `Q`, `T` grow to roughly `D` decimal digits each by
//! the top of the recursion, so the dominant cost is a single GMP
//! multiplication of two `D`-digit numbers — O(M(D) · log D) total.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use bignum::{Float, Integer, Pow};

use crate::output::DigitSink;
use crate::precision::PrecisionPlan;
use crate::progress::{Phase, ProgressReporter};

use super::util::write_decimal_digits;
use super::PiAlgorithm;

/// Decimal digits added per Chudnovsky term (≈ log10(C³ / 24) / 3).
pub const DIGITS_PER_TERM: f64 = 14.181_647_462_725_478;

/// Extra series terms beyond the asymptotic estimate, to absorb the
/// discrete rounding when converting from a per-term digit count to an
/// integer count.
const SAFETY_TERMS: u64 = 8;

const A: u32 = 13_591_409;
const B: u32 = 545_140_134;
// C = 640_320.  We only ever need C³ / 24 = 10_939_058_860_032_000 as a
// constant.
const C3_OVER_24: u64 = 10_939_058_860_032_000;

const PHASE_BINARY_SPLITTING: &str = "binary splitting";
const PHASE_FINAL_ASSEMBLY: &str = "final assembly";
const PHASE_DECIMAL_CONVERSION: &str = "decimal conversion";

#[derive(Default, Debug, Clone, Copy)]
pub struct Chudnovsky;

impl PiAlgorithm for Chudnovsky {
    fn name(&self) -> &'static str {
        "chudnovsky"
    }

    fn compute(
        &self,
        digits: u64,
        sink: &mut dyn DigitSink,
        progress: &mut dyn ProgressReporter,
    ) -> Result<()> {
        if digits == 0 {
            sink.finish()?;
            return Ok(());
        }

        let plan = PrecisionPlan::for_digits(digits)?;

        // Number of binary-splitting terms.  Each Chudnovsky term
        // contributes `DIGITS_PER_TERM` digits; round up and add a
        // small safety margin for the discrete-to-continuous gap.
        let n_terms = (digits as f64 / DIGITS_PER_TERM).ceil() as u64 + SAFETY_TERMS;

        progress.set_phases(&[
            Phase { name: PHASE_BINARY_SPLITTING, total: n_terms },
            Phase { name: PHASE_FINAL_ASSEMBLY, total: 4 },
            Phase { name: PHASE_DECIMAL_CONVERSION, total: 1 },
        ]);

        // Wrap the binary-splitting + final-assembly work in its own
        // scope so the multi-gigabyte `_p`, `q`, `t`, and `denom_int`
        // get dropped before the decimal-conversion phase allocates
        // its own large buffers.  At 10B digits this single change
        // saves ~16 GB of peak memory.
        let pi = {
            // The pi_numer chain in FA is `√10005 · 426880 · q`.  Only
            // the final `· q` depends on BS output; the `√10005 · 426880`
            // factor depends solely on the precision and can be computed
            // any time after we know `prec`.  At billion-plus digits the
            // sqrt (precision-doubling Newton iteration) is a multi-
            // minute chain of full-width multiplications; running it
            // during BS folds it into wall time we're already paying for.
            // The thread shares the rayon worker pool for its internal
            // NTT muls, co-scheduling naturally with BS — and the sqrt
            // chain is short relative to BS at scale, so it finishes
            // during BS's earlier (highly parallel) levels before BS
            // reaches its serialised top-of-tree combines.
            let prec = plan.precision_bits;
            let sqrt_handle = std::thread::Builder::new()
                .name("chudnovsky-sqrt".into())
                .spawn(move || {
                    let mut p = Float::with_val_64(prec, 10_005);
                    p.sqrt_mut();
                    p *= 426_880_u32;
                    p
                })
                .expect("spawning sqrt worker thread");

            progress.start_phase(PHASE_BINARY_SPLITTING, n_terms);
            let (q, t) = binary_split(1, n_terms + 1, progress);
            progress.end_phase();

            progress.start_phase(PHASE_FINAL_ASSEMBLY, 3);
            // S = (A · Q + T) / Q, where T/Q sums terms k = 1..N and A
            // is the k = 0 contribution, so
            //   π = (426880 · √10005 · Q) / (A·Q + T)
            //     = pi_numer · (1 / denom)
            //
            // The two factors `pi_numer` and `1/denom` are independent
            // — they share no intermediate values.  Both are O(M(N))
            // chains: the numerator is √10005 (precision-doubling
            // Newton) times two multiplies, and the reciprocal is
            // precision-doubling Newton on the integer denominator.
            // Running them on separate rayon worker threads halves the
            // serial chain at the top of final assembly.
            let denom_int = Integer::from(A) * &q + &t;
            // t is no longer needed — free its mantissa now (hundreds
            // of MB at billion-digit scale) before the NTT-heavy work
            // below allocates its scratch buffers.
            drop(t);
            progress.tick();

            // `denom_int` is consumed by `.into()` so its memory
            // is released when `denom_float` is built.
            let denom_float: Float = denom_int.into();

            // Wait for the background sqrt thread (started before BS).
            // At scale this returns immediately; the sqrt finished
            // during the parallel lower-tree levels of BS.
            let sqrt_426880 = sqrt_handle
                .join()
                .expect("chudnovsky-sqrt thread panicked");

            // Two chains compute pi_numer (sqrt · 426880 · q, where the
            // sqrt · 426880 factor is precomputed in the background
            // thread above — only `· q` remains here) and recip
            // (1 / denom).  Running them concurrently fills each
            // other's NTT serial pockets (~10% faster); running them
            // sequentially holds only one chain's Float intermediates
            // live at a time (saves multi-GB peak at billion-digit
            // scale).  Knob lives in chudnovsky.parallel_final_assembly.
            let (recip, pi_numer) = if crate::config::chudnovsky_parallel_final_assembly() {
                rayon::join(
                    || denom_float.reciprocal_at_prec(prec + 16),
                    || {
                        let mut p = sqrt_426880;
                        p *= &q;
                        p
                    },
                )
            } else {
                let recip = denom_float.reciprocal_at_prec(prec + 16);
                let mut p = sqrt_426880;
                p *= &q;
                (recip, p)
            };
            // Once both chains have returned, neither `q` nor
            // `denom_float` is referenced again.  Drop them before the
            // final multiply so its NTT buffers don't stack on top of
            // the now-dead mantissas.
            drop(q);
            drop(denom_float);
            progress.tick();

            let pi = pi_numer.mul_at_prec(&recip, prec);
            // Free the two operands of the final multiply now that the
            // result is built — keeps peak below `recip + pi_numer +
            // pi` (which is 3 full-precision Floats).
            drop(pi_numer);
            drop(recip);
            progress.tick();
            progress.end_phase();
            pi
            // q, t, denom_int all dropped here.
        };

        progress.start_phase(PHASE_DECIMAL_CONVERSION, 1);
        write_decimal_digits(pi, digits, sink)?;
        progress.tick();
        progress.end_phase();

        Ok(())
    }
}

/// Binary splitting over the half-open range `[a, b)` of Chudnovsky terms.
///
/// Returns `(Q, T)` such that the partial sum equals `T / Q` (with
/// `M_0 = 1` for the conventional top-level call `a = 1`).  The `P`
/// factor of the recursion *is* computed for every internal merge but
/// is intentionally *not* computed at the root: the caller throws it
/// away, and at billion-digit scale the root P multiply is a huge
/// NTT call (multi-GB scratch buffers) for a discarded result.  See
/// `binary_split_pure_root` for the no-P entry.
///
/// Progress: the parallel work runs on a worker thread (which dispatches
/// to the rayon pool internally), while this function polls a shared
/// atomic and emits ticks on behalf of completed merges.  Each merge
/// contributes a Karatsuba-weighted share of the total, so the bar
/// reflects real wall-time progress — fast early as small subtrees
/// finish, slowing into the few enormous root-level merges.
fn binary_split(
    a: u64,
    b: u64,
    progress: &mut dyn ProgressReporter,
) -> (Integer, Integer) {
    let n_terms = b - a;
    let total_weight = total_merge_weight(a, b).max(1);
    let done_weight = Arc::new(AtomicU64::new(0));
    let finished = Arc::new(AtomicBool::new(false));

    let worker = {
        let done_weight = Arc::clone(&done_weight);
        let finished = Arc::clone(&finished);
        std::thread::spawn(move || {
            let result = binary_split_pure_root(a, b, &done_weight);
            // Release: every fetch_add above happens-before this store,
            // so a reader that sees `finished == true` via Acquire is
            // guaranteed to see every weight contribution as well.
            finished.store(true, Ordering::Release);
            result
        })
    };

    let mut ticks_emitted: u64 = 0;
    loop {
        let done = done_weight.load(Ordering::Relaxed);
        let target = (((done as u128) * (n_terms as u128)) / total_weight as u128) as u64;
        let target = target.min(n_terms);
        while ticks_emitted < target {
            progress.tick();
            ticks_emitted += 1;
        }
        if finished.load(Ordering::Acquire) {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let result = worker.join().expect("binary_split worker panicked");
    // Guarantee the bar reaches 100% even if the weight model under-
    // counted by a unit somewhere (rounding) or a few merges finished
    // after the last poll.
    while ticks_emitted < n_terms {
        progress.tick();
        ticks_emitted += 1;
    }
    result
}

// Subtree sizes below this run sequentially.  Above it, the two halves
// of the range are forked via `rayon::join` so the recursion exposes
// parallelism at every level until the granularity shrinks below the
// per-task overhead.  Threshold lives in `crate::config`.

/// Karatsuba-aware cost estimate for one merge over `n` terms.
///
/// A merge over `n` terms multiplies two integers of size ~n/2 limbs, and
/// Karatsuba costs ~size^log₂3 ≈ size^1.585.  The constant factor is
/// irrelevant for the progress ratio; only the scaling matters.
fn merge_weight(n: u64) -> u64 {
    ((n as f64).powf(1.585)).max(1.0) as u64
}

/// Sum of `merge_weight` across every internal node of the splitting
/// tree for `[a, b)`.  Leaves contribute zero — their work is dominated
/// by even the smallest merge above them.
fn total_merge_weight(a: u64, b: u64) -> u64 {
    if b - a <= 1 {
        return 0;
    }
    let m = (a + b) / 2;
    merge_weight(b - a) + total_merge_weight(a, m) + total_merge_weight(m, b)
}

/// Decide whether a binary-split node of size `n` recurses in parallel
/// or sequentially.  Two thresholds:
/// * below `parallel_split_threshold` → sequential (rayon overhead
///   would outweigh the saved time);
/// * above `sequential_top_threshold` (when non-zero) → sequential,
///   to keep concurrent NTT scratch buffers bounded at huge sizes.
/// Between the two → parallel via `rayon::join`.
#[inline]
fn go_parallel(n: u64) -> bool {
    let lo = crate::config::chudnovsky_parallel_split_threshold();
    let hi = crate::config::chudnovsky_sequential_top_threshold();
    n >= lo && (hi == 0 || n < hi)
}

/// Root entry that returns only `(Q, T)` — skips the final `p_root =
/// p_l * p_r` multiplication that the regular recursion performs at
/// every internal node.  Justification: the caller of `binary_split`
/// discards `_p`, but in the all-paths-equal recursion that p_root
/// triggers a huge NTT call (at 1B-digit scale: ~4 GB scratch buffers)
/// for a result that's never read.  Both sub-recursions still go
/// through `binary_split_pure` so all internal merges compute p
/// correctly.  We also `drop` the two sub-p factors as soon as the
/// merge is done so their memory is reclaimed before the q multiply.
fn binary_split_pure_root(a: u64, b: u64, done_weight: &AtomicU64) -> (Integer, Integer) {
    debug_assert!(a < b);
    if b - a == 1 {
        // Single-term root — no big multiplies to skip.  Fall back to
        // the regular path and discard p.
        let (_, q, t) = binary_split_pure(a, b, done_weight);
        return (q, t);
    }
    let m = (a + b) / 2;
    let ((p_l, q_l, t_l), (p_r, q_r, t_r)) =
        if go_parallel(b - a) {
            rayon::join(
                || binary_split_pure(a, m, done_weight),
                || binary_split_pure(m, b, done_weight),
            )
        } else {
            (
                binary_split_pure(a, m, done_weight),
                binary_split_pure(m, b, done_weight),
            )
        };
    // t still needs p_l from the left subtree.  q needs neither.
    // After the t and q computations, p_l and p_r are unreferenced;
    // explicit drops free them before the q multiply runs.
    let t = t_l * &q_r + &p_l * t_r;
    drop(p_l);
    drop(p_r);
    let q = q_l * q_r;
    done_weight.fetch_add(merge_weight(b - a), Ordering::Relaxed);
    (q, t)
}

/// Pure (no-progress-callback, `Send`-safe) variant of binary splitting.
/// Spawns `rayon::join` for the two recursive calls once the range is
/// large enough that the rayon task overhead is amortized.  Each merge
/// publishes its Karatsuba-weighted share to `done_weight` so the
/// monitor in `binary_split` can render a smooth progress bar without
/// any per-merge synchronization cost beyond one relaxed `fetch_add`.
fn binary_split_pure(a: u64, b: u64, done_weight: &AtomicU64) -> (Integer, Integer, Integer) {
    debug_assert!(a < b, "binary_split called with empty/reversed range [{a}, {b})");
    if b - a == 1 {
        let k = Integer::from(a);
        let six_k = Integer::from(&k * 6_u32);
        // p_k = -(6k - 5)(6k - 1)(2k - 1).
        let factor1: Integer = Integer::from(&six_k - 5_u32);
        let factor2: Integer = Integer::from(&six_k - 1_u32);
        let factor3: Integer = Integer::from(&k * 2_u32) - 1_u32;
        let product: Integer = factor1 * factor2 * factor3;
        let p: Integer = -product;
        // q_k = k³ · (C³ / 24)
        let q: Integer = Integer::from(C3_OVER_24) * k.clone().pow(3_u32);
        // t_k = p_k · (A + B·k)
        let l: Integer = Integer::from(A) + Integer::from(&k * B);
        let t: Integer = Integer::from(&p * &l);
        (p, q, t)
    } else {
        let m = (a + b) / 2;
        let ((p_l, q_l, t_l), (p_r, q_r, t_r)) =
            if b - a >= crate::config::chudnovsky_parallel_split_threshold()
        {
            rayon::join(
                || binary_split_pure(a, m, done_weight),
                || binary_split_pure(m, b, done_weight),
            )
        } else {
            (
                binary_split_pure(a, m, done_weight),
                binary_split_pure(m, b, done_weight),
            )
        };
        // Combine.  Order matters: we need to use `&q_r` and `&p_l` to
        // build `t` before consuming them in `p` and `q`.
        let t = t_l * &q_r + &p_l * t_r;
        let p = p_l * p_r;
        let q = q_l * q_r;
        done_weight.fetch_add(merge_weight(b - a), Ordering::Relaxed);
        (p, q, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::util::test_support::{StringSink, FIRST_100};
    use crate::progress::NoopProgress;

    #[test]
    fn one_digit() {
        let mut sink = StringSink::new();
        Chudnovsky.compute(1, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, "3");
    }

    #[test]
    fn fifty_digits() {
        let mut sink = StringSink::new();
        Chudnovsky.compute(50, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, &FIRST_100[..51]);
    }

    #[test]
    fn one_hundred_digits() {
        let mut sink = StringSink::new();
        Chudnovsky.compute(100, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out, FIRST_100);
    }

    #[test]
    fn one_thousand_digits_match_known_prefix() {
        let mut sink = StringSink::new();
        Chudnovsky.compute(1_000, &mut sink, &mut NoopProgress).unwrap();
        assert_eq!(sink.out.len(), 1_001);
        assert_eq!(&sink.out[..101], FIRST_100);
    }
}
