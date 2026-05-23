# Implementation

How the code is laid out, what the moving pieces are, and where to
look for things. Assumes you can read Rust and have CS fundamentals
(data structures, parallelism, complexity).

For the *math* behind the algorithms, see [MATH.md](MATH.md). This
document is about the code.

## Cargo workspace

```text
pi/
├── Cargo.toml              workspace root
└── crates/
    ├── pi-core/            library: math, traits, BBP, precision plan
    │   └── src/
    │       ├── lib.rs
    │       ├── algorithm/
    │       │   ├── mod.rs            PiAlgorithm trait + AlgorithmKind registry
    │       │   ├── chudnovsky.rs     Chudnovsky + binary splitting
    │       │   ├── gauss_legendre.rs Brent-Salamin AGM iteration
    │       │   └── util.rs           widen_mpfr_exponent_range, write_decimal_digits
    │       ├── bbp.rs                Bailey-Borwein-Plouffe digit extractor
    │       ├── output.rs             DigitSink trait + WriterSink
    │       ├── precision.rs          PrecisionPlan
    │       └── progress.rs           ProgressReporter trait + Phase + NoopProgress
    └── pi-cli/             binary `pi`
        └── src/
            ├── main.rs               CLI + compute mode + --verify (file vs file)
            └── verify_hex.rs         --verify-hex orchestration
```

`pi-core` is the library: pure math, traits, no I/O or CLI. `pi-cli`
is the binary that wires up file sinks, progress bars, signal
handlers, and command-line arguments.

## Trait abstractions

Three small traits keep concerns separated:

```rust
pub trait PiAlgorithm {
    fn name(&self) -> &'static str;
    fn compute(
        &self,
        digits: u64,
        sink: &mut dyn DigitSink,
        progress: &mut dyn ProgressReporter,
    ) -> Result<()>;
}

pub trait DigitSink {
    fn write_integer_part(&mut self, digits: &str) -> io::Result<()>;
    fn write_fractional_digits(&mut self, digits: &str) -> io::Result<()>;
    fn finish(&mut self) -> io::Result<()>;
}

pub trait ProgressReporter {
    fn set_phases(&mut self, _phases: &[Phase]) {}
    fn start_phase(&mut self, name: &str, total: u64);
    fn tick(&mut self);
    fn end_phase(&mut self);
}
```

- Adding a new compute algorithm: implement `PiAlgorithm`, add to
  `AlgorithmKind`.
- New output destination: implement `DigitSink`. The current
  `WriterSink<W: Write>` covers stdout, files, and anything else with
  `std::io::Write`.
- New progress UI: implement `ProgressReporter`. Tests use
  `NoopProgress`; the CLI uses an `indicatif::MultiProgress`-backed
  one.

The CLI and the other modules don't change when you swap any of these.

## Numeric backend

Both compute algorithms run on **GMP** (for arbitrary-precision
integers) and **MPFR** (for arbitrary-precision floating-point),
wrapped by the [`rug`](https://crates.io/crates/rug) crate. GMP is the
best-in-the-world implementation of multi-precision integer
arithmetic; its FFT-based multiplication is what makes Chudnovsky's
inner loops practical at a billion digits.

The BBP code is **pure Rust** — no rug, no GMP. The denominators in
the BBP series stay small (`8k + r` for `r ∈ {1, 4, 5, 6}`), so `u64`
plus `u128` for safe multiplication is enough. Avoiding GMP here is a
small performance win and one less dependency to learn.

### Precision planning

`PrecisionPlan::for_digits(D)` returns the working precision in bits:

```rust
precision_bits = ceil(D · log₂(10)) + 256
```

The 256-bit safety margin covers the few ULPs lost to MPFR rounding
plus the final `pi × 10^(D-1)` scaling. `precision_bits` is `u64` (it
was `u32` early on; got lifted when 1B digits became the load-bearing
case).

The plan also bails explicitly when the request exceeds either f64's
ability to represent the digit count cleanly (`2^53`) or MPFR's
allowed precision range. Quiet truncation is the failure mode you
*never* want when you're computing a billion-digit answer.

### Widening MPFR's exponent range

`algorithm/util.rs::widen_mpfr_exponent_range_for(digits)` is called
at the top of every `compute` to push MPFR's allowed exponent range out
to the platform maximum. The reason: at `D` digits, the intermediate
`pi *= &q` step temporarily has magnitude `~10^D`, whose binary
exponent (`~3.3·D`) exceeds MPFR's *default* `emax` (`2^30 − 1`)
once `D` crosses a few hundred million. The widened range is a global
MPFR setting but harmless for any Float with a small magnitude.

## Chudnovsky

`algorithm/chudnovsky.rs`. The compute path:

1. **Precision plan.** Compute working precision and the number of
   binary-splitting terms (`D / 14.18 + 8`).
2. **Widen MPFR exponent range.** One call to the helper above.
3. **Binary splitting.** `binary_split(1, n_terms + 1)` returns
   `(P, Q, T)`. Each leaf computes one term's `(p_k, q_k, t_k)`
   directly; each internal node combines two children with three
   integer multiplications and an add.
4. **Final assembly.** `pi = 426880 · √10005 · Q / (A·Q + T)`. One
   square root, two multiplications, one division on full-precision
   Floats.
5. **Decimal conversion.** `write_decimal_digits(pi, D, sink)`. See
   below.

### The combine rule

```rust
let t = t_l * &q_r + &p_l * t_r;
let p = p_l * p_r;
let q = q_l * q_r;
```

The owned vs. borrowed pattern matters: `q_r` and `p_l` are borrowed
for the `t` computation so they're still available afterward for `q`
and `p` respectively. Compute `t` first because both `t_l` and `t_r`
are consumed there; then `p` (consuming `p_l` and `p_r`); then `q`.

### Memory layout

At the top of the recursion, `P`, `Q`, `T` are each roughly `D`
decimal digits, which is `~0.42 · D` bytes. Six of those (three from
each half) sit in memory at the top-level combine, plus FFT scratch
during each multiplication (~2× operand size). This peak is the
dominant memory cost of the whole algorithm.

To keep this peak from stacking on top of the decimal-conversion
phase, the body is wrapped in a `{ ... }` scope that returns just
`pi` — so `P`, `Q`, `T`, and `denom_int` are all dropped before the
decimal-conversion phase allocates its own large buffers.

### Decimal conversion

`algorithm/util.rs::write_decimal_digits` takes `pi` **by value** and
operates on it in place:

```rust
pi *= &scale;                 // pi := pi · 10^(D-1)
drop(scale);                  // free a full-precision Float
let (int_part, _) = pi.to_integer_round(Down).unwrap();
drop(pi);                     // free another full-precision Float
let s = int_part.to_string();
drop(int_part);
// emit s through the sink, with the decimal point inserted after "3"
```

The explicit drops claw back about two full-precision Floats compared
to the "create a new scaled Float, then convert" version. At billion-
plus digits each of those is multi-GB.

The rounding direction is `Round::Down` (toward `−∞`, equivalent to
floor and to truncation for positive `pi`). MPFR's default
`to_integer` rounds to nearest, which would give the wrong answer for
"the first N digits of π" whenever the digit just past the cut is `≥ 5`.

## Gauss-Legendre

`algorithm/gauss_legendre.rs`. Brent-Salamin iteration in
`ceil(log₂(precision_bits)) + 2` steps. Each step:

```rust
a_new.assign(&a + &b);
a_new /= 2_u32;

b_new.assign(&a * &b);
b_new.sqrt_mut();

diff.assign(&a - &a_new);
diff_sq.assign(&diff * &diff);
diff_sq *= &p;
t -= &diff_sq;

std::mem::swap(&mut a, &mut a_new);
std::mem::swap(&mut b, &mut b_new);
p <<= 1_u32;
```

To avoid 4 full-precision Float allocations per iteration (which would
be ~16 GB of allocator churn at 10B digits over ~32 iterations), the
scratch buffers `a_new`, `b_new`, `diff`, `diff_sq` are pre-allocated
outside the loop. `std::mem::swap` promotes `a_new → a` and
`b_new → b` without a copy — the old `a` now sits in `a_new` and gets
overwritten by the next iteration's `assign`.

Like Chudnovsky, the whole algorithm body is wrapped in a scope so the
working Floats drop before decimal conversion.

## BBP

`bbp.rs`. The hot loop:

```rust
fn series_fixed(j: u64, n: u64, stop: &AtomicBool) -> Option<u64> {
    let mut sum: u64 = 0;
    for k in 0..=n {
        if k % 10_000 == 0 && stop.load(Ordering::Relaxed) {
            return None;
        }
        let denom = 8 * k + j;
        let pow = mod_pow_16(n - k, denom);
        let term_fixed = ((pow as u128) << 64) / (denom as u128);
        sum = sum.wrapping_add(term_fixed as u64);
    }
    // ... fast-decaying tail summed in f64, then converted to fixed-point
}
```

Notes:

- The running sum is `u64` fixed-point: a value `v` represents the
  fraction `v / 2^64`. Modular arithmetic on `wrapping_add` keeps the
  sum implicitly modulo 1.
- `mod_pow_16` uses `u128` internally so the intermediate
  `result * base` never overflows.
- The `k > n` tail decays geometrically by a factor of 16 per step,
  so summing ~20 terms in `f64` is precise enough.
- The interrupt poll every 10,000 inner iterations costs about one
  atomic load per millisecond of CPU work — well under 0.1% overhead.
  Without it, an in-flight BBP call at deep `n` is uninterruptible for
  ~15–20 minutes; with it, SIGINT lands within milliseconds.

The two exported functions are `hex_digits_at(n) → u32` (test-friendly,
uninterruptible) and `hex_digits_at_interruptible(n, &AtomicBool) →
Option<u32>` (used by verify-hex).

## verify-hex orchestration

`pi-cli/src/verify_hex.rs`. The largest single file in the project,
because it's a complete sub-program in its own right.

### Phases

Up to five, depending on flags:

1. **Conversion** (skipped if the hex file already exists). Reads the
   decimal pi file, parses into a rug `Integer`, computes
   `m = numerator · 16^H / 10^(D−1)` where `H ≈ 0.83 · D` is the
   number of hex digits we get out of `D` decimal digits. Writes
   `"3." + m.to_string_radix(16) + "\n"` to a `.tmp` file, then
   atomically renames into place — so a Ctrl-C mid-conversion never
   leaves a half-written hex file behind.
2. **Sanity sweep** — three regions, deterministic positions:
   - First 1M hex digits, `--sanity-samples` BBP calls (default 100).
   - Middle 100K hex digits, `--sanity-samples / 10` calls.
   - Last 10K hex digits, `--sanity-samples / 100` calls.

   Sample counts scale down at deeper positions because each BBP call
   at position `n` costs `O(n log n)` — a call near the end of a
   billion-digit file is roughly *2,000 times* more expensive than one
   near the beginning. Without scaling, the last-region sanity sweep
   would dominate the runtime.
3. **Random sampling loop** (unbounded). Cryptographic RNG picks a
   uniform random window start in `[0, N − 8)`. `--samples-per-window`
   random positions are drawn inside that 1M-byte window. Each is
   BBP-checked.

All four phase tasks (three sanity, one random) run **concurrently**
on a shared rayon thread pool via `pool.scope(|s| { s.spawn(…) })`.
Rayon's work-stealing redistributes worker capacity as sanity phases
finish, so the random phase naturally absorbs freed CPU without
manual reallocation.

### IntervalSet

Every successful `check_position` records the 8-hex-digit interval
`[pos, pos + 8)` in a shared `IntervalSet` — a `BTreeMap<u64, u64>`
keyed by interval start, with overlap/adjacency merging on insert.
After each random window, the random bar's message gets refreshed
with the running coverage: total bytes verified, distinct intervals,
percent of the file covered.

The data structure is straightforward but worth glancing at: insert
finds candidates via a `range(..=end)` query, filters for actual
overlap, merges them all, removes the old, inserts the merged one.
The total-covered counter is maintained incrementally.

### Coordination across phases

A single `Arc<AtomicBool>` "stop" flag is the canonical "should I keep
going?" signal:

- The SIGINT handler sets it.
- Any phase that detects a mismatch sets it after storing the error in
  a shared `Mutex<Option<anyhow::Error>>` slot.
- The flag is threaded all the way down into
  `bbp::hex_digits_at_interruptible`, so an in-flight BBP call bails
  in milliseconds rather than after minutes of unrolled work.

After `pool.scope` returns, the orchestrator checks the error slot.
A `Some` propagates as the program's exit error (non-zero). A `None`
with `stop` set means the user interrupted (exit 0). A `None` with
`stop` unset shouldn't happen (the random loop is unbounded; the only
way it exits cleanly is via stop), but the code handles it.

### The conversion tail skip

The decimal→hex conversion has up to one unit of round-down error in
its last hex digit (the leftover precision `δ · 16^H / 10^(D−1)` is in
`[0, 1)`, so the floor of the scaled product can be one less than the
true value, occasionally with a short borrow chain into the previous
digit). To keep this from producing false-positive mismatches, every
range and the random sampling window is clamped to `safe_end =
n_hex_digits − TAIL_SKIP`, where `TAIL_SKIP = 32`.

The initial status line says so explicitly: "(last 32 hex digits
skipped to avoid conversion-boundary noise)."

## Progress UI

Built on `indicatif::MultiProgress`. Both the CLI's compute path and
verify-hex have their own `PhaseBar` types with a small state machine:

```text
pending ──► active ──► done
                  ╰─►  interrupted
```

Each state has its own style:

| state         | bar appearance                                           |
|---------------|----------------------------------------------------------|
| pending       | flat plain bar with `(pending)` message                  |
| active (bar)  | cyan/blue fill with `eta {eta}`                          |
| active (spinner) | cyan spinner with running counter                     |
| active (random) | cyan spinner with running coverage in `{msg}`         |
| done          | green full bar with `done in {elapsed}`                  |
| interrupted   | yellow bar at *current* position (no jump to full) with `interrupted at {elapsed}` |

The `interrupted` state uses `ProgressBar::abandon()` instead of
`finish()` so the position stays where it was, not jumping to the
bar's full length.

For multi-threaded progress (verify-hex's sanity bars), each rayon
worker increments its phase's bar directly via
`ProgressBar::inc(1)`, which is internally synchronized by indicatif.

Phases are declared up front via `ProgressReporter::set_phases`, so
the user sees the full pipeline at once rather than discovering one
phase at a time.

## Error handling

`anyhow::Result` is threaded through everything. Compute and verify
modes both return `Result<()>` from `main`, so anything propagated
becomes the program's exit error.

The atomic `stop` flag is the canonical "should I keep going?" signal
across threads, not exceptions. This matters for verify-hex: a
mismatch in one phase needs to wind down three other phases that
aren't on the call stack from the mismatching site.

SIGINT handling is `ctrlc::set_handler` setting the same flag. We
deliberately don't restore the previous handler — the process is
expected to exit shortly after.

---

For the math, see [MATH.md](MATH.md). For contributing, see
[DEVELOPING.md](DEVELOPING.md).
