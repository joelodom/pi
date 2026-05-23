# pi

A CLI program in Rust for computing pi to arbitrary precision, designed
to start at a million digits and scale upward.

Two compute algorithms plus three verification paths:

* **Chudnovsky** with binary splitting (default; fastest in practice).
* **Gauss-Legendre / Brent-Salamin** AGM iteration (independent
  cross-check algorithm).
* **`--verify FILE_A FILE_B`** — byte-by-byte compare of two digit
  files, ignoring trailing whitespace.
* **`--verify-hex HEX_FILE [--from-decimal DEC_FILE]`** — convert a
  decimal pi file to hex once and then spot-check thousands of
  cryptographically-randomly-chosen hex positions against the
  Bailey-Borwein-Plouffe (BBP) formula as an independent oracle.

Compute and BBP arithmetic both run on GMP/MPFR via the
[`rug`](https://crates.io/crates/rug) crate. A million digits
finishes in well under a second with either compute algorithm.

## Quickstart

```sh
# Build (the first build also compiles GMP and MPFR from source, so it
# takes a few minutes; subsequent builds are fast).
cargo build --release

# Show help (this is also what bare `pi` prints — no arguments => help).
./target/release/pi

# Compute 1,000,000 digits and write them to pi.txt.
./target/release/pi --digits 1000000 -o pi.txt

# Verify pi.txt against a trusted reference file.  Trailing whitespace is
# ignored on both sides, and the shorter file's content is compared
# against the matching prefix of the longer one — so a freshly computed
# 1M-digit pi.txt will succeed against a 100M-digit reference if it agrees
# on the first 1M digits.
./target/release/pi --verify pi.txt pi3-100-million-verified.txt

# Independent BBP-based verification.  First time, convert the decimal
# file to hex on disk; subsequent times, reuse it.  Runs until Ctrl-C
# or a mismatch.
./target/release/pi --verify-hex pi-hex.txt --from-decimal pi.txt
./target/release/pi --verify-hex pi-hex.txt   # reuse the existing hex file

# Compute 100 digits to stdout.
./target/release/pi --digits 100
```

Run `./target/release/pi --help` for the full flag list.

## CLI

`pi` has three modes, picked by which flags are given:

* **Compute** — `pi --digits N [-o FILE] [--algorithm ALG] [--no-progress]`.
  After writing to a file, the CLI prints the suggested `pi --verify ...`
  invocation.
* **Verify (file vs. file)** — `pi --verify FILE_A FILE_B`.  Trims trailing
  whitespace (`' '`, `'\t'`, `'\n'`, `'\r'`) from both files, then
  byte-by-byte compares the shorter content against the matching prefix
  of the longer content.  On mismatch it reports the first differing
  byte offset and exits non-zero.  Skips computation entirely.
* **Verify-hex (BBP spot-check)** — `pi --verify-hex HEX_FILE
  [--from-decimal DEC_FILE]`.  If `HEX_FILE` exists, reuses it; otherwise
  `--from-decimal` is required and converts the decimal pi file to hex
  on disk (atomic `.tmp` + rename).  Then BBP-spot-checks the hex file
  in two phases: a deterministic sanity sweep over the first / middle /
  last 1M hex digits, followed by an unbounded random-sampling loop
  (window starts via `OsRng`, 8 hex digits per BBP call, parallel via
  rayon).  Runs until SIGINT or a mismatch; on mismatch reports the
  position and the offending byte and exits non-zero.

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --digits N` | `1000000` | Decimal digits to compute (counting the leading `3`). |
| `-o, --output FILE` | `-` | File to write digits to. Use `-` for stdout. |
| `--algorithm NAME` | `chudnovsky` | `chudnovsky` (series + binary splitting) or `gauss-legendre` (AGM iteration). |
| `--no-progress` | | Suppress the progress bars (they go to stderr). |
| `--verify FILE_A FILE_B` | | Byte-by-byte compare two digit files, ignoring trailing whitespace. Runs instead of compute. |
| `--verify-hex HEX_FILE` | | BBP-based hex verification. Runs instead of compute. |
| `--from-decimal DEC_FILE` | | With `--verify-hex`, source decimal file to convert when `HEX_FILE` doesn't yet exist. |
| `--samples-per-window M` | `100` | With `--verify-hex`, BBP samples per random-window (each call covers 8 hex digits). |
| `--sanity-samples N` | `100` | With `--verify-hex`, BBP samples per sanity region (first/middle/last 1M). |
| `--jobs J` | ncpu | With `--verify-hex`, rayon worker threads. |

Running `pi` with no arguments prints the help text.

## Repository layout

```
pi/
├── Cargo.toml              # Workspace
└── crates/
    ├── pi-core/            # Library: algorithms, sinks, planning, progress, BBP
    │   └── src/
    │       ├── lib.rs
    │       ├── algorithm/
    │       │   ├── mod.rs            # PiAlgorithm trait + registry
    │       │   ├── chudnovsky.rs
    │       │   ├── gauss_legendre.rs
    │       │   └── util.rs           # widen_mpfr_exponent_range, write_decimal_digits
    │       ├── bbp.rs                # Bailey-Borwein-Plouffe hex digit extractor
    │       ├── output.rs             # DigitSink trait + WriterSink
    │       ├── precision.rs          # PrecisionPlan
    │       └── progress.rs           # ProgressReporter trait + Phase + NoopProgress
    └── pi-cli/             # Binary `pi`
        └── src/
            ├── main.rs               # CLI, compute mode, --verify mode
            └── verify_hex.rs         # --verify-hex orchestration
```

## Architecture

The library is organized around a few small traits so each concern can be
swapped out independently. Adding a new algorithm, output destination, or
progress backend does not touch the others.

### `PiAlgorithm` (`crates/pi-core/src/algorithm/mod.rs`)

```rust
pub trait PiAlgorithm {
    fn name(&self) -> &'static str;
    fn compute(
        &self,
        digits: u64,
        sink: &mut dyn DigitSink,
        progress: &mut dyn ProgressReporter,
    ) -> anyhow::Result<()>;
}
```

To add a new algorithm (Borwein quintic, an NTT-backed Chudnovsky, …)
implement this trait and add a variant to `AlgorithmKind`. Term/iteration
counts are algorithm-private; the shared `PrecisionPlan` only computes
the working precision. Nothing in the CLI or the sinks changes.

### `DigitSink` (`crates/pi-core/src/output.rs`)

```rust
pub trait DigitSink {
    fn write_integer_part(&mut self, digits: &str) -> io::Result<()>;
    fn write_fractional_digits(&mut self, digits: &str) -> io::Result<()>;
    fn finish(&mut self) -> io::Result<()>;
}
```

The integer part (just `"3"` for pi) is emitted first, then zero or more
chunks of fractional digits, then `finish`. The sink decides how to format
(decimal point insertion, wrapping, newlines, multi-file splitting, mmap,
network, …).

Two ready-to-use shapes are provided: `stdout_sink()` and
`file_sink(path)`, both backed by the generic `WriterSink<W: Write>` so any
`Write` implementor can be plugged in.

### `ProgressReporter` (`crates/pi-core/src/progress.rs`)

```rust
pub trait ProgressReporter {
    fn set_phases(&mut self, _phases: &[Phase]) {}
    fn start_phase(&mut self, name: &str, total: u64);
    fn tick(&mut self);
    fn end_phase(&mut self);
}

pub struct Phase { pub name: &'static str, pub total: u64 }
```

Algorithms call `set_phases` up front with the full ordered list of
phases they intend to run, so a multi-phase reporter (like the CLI's
`MultiProgress`-backed one) can render pending / running / completed
bars at a glance. Tests use `NoopProgress`. Any other backend
(structured logging, telemetry, web socket) plugs in the same way.

### `PrecisionPlan` (`crates/pi-core/src/precision.rs`)

Decides the working precision (mantissa bits for `rug::Float`) for a
given target digit count, with a `u64` precision field, a fixed 256-bit
safety margin, and explicit `bail!`s when the request would push the
f64-based planning arithmetic or MPFR's `prec_max_64()` past their
limits. Series-vs-iteration term counts are algorithm-specific and
live inside each algorithm module rather than in `PrecisionPlan`.

### `bbp` (`crates/pi-core/src/bbp.rs`)

Pure-Rust Bailey-Borwein-Plouffe hex digit extractor: `hex_digits_at(n)`
returns the 8 hex digits of pi's fractional expansion starting at
position `n`, packed in a `u32`. Uses `u64` fixed-point arithmetic in
the running sum and `u128` modular exponentiation in the inner loop —
no `rug` dependency. Cost per call is O(n log n) machine ops. Reliable
for `n` up to ~10^9 (the top ~30 bits of the accumulated sum remain
correct).

## Verification

Three independent checks; any combination of them gives high confidence:

**1. Compare against a trusted reference file.**

```sh
./target/release/pi --verify pi.txt pi3-100-million-verified.txt
```

Trims trailing whitespace on both sides. The shorter file's content
must match the matching prefix of the longer one, so a freshly
computed N-digit `pi.txt` succeeds against a much larger reference
when its first N digits agree.

**2. Cross-algorithm: Chudnovsky vs. Gauss-Legendre.**

The two compute algorithms share only the bignum backend, the
precision plan, and the decimal-conversion code; the actual pi
computation goes through two entirely independent code paths (series
+ binary splitting vs. quadratically-convergent AGM iteration with
sqrt). Any bug specific to one of them — wrong formula constants, an
off-by-one in the binary-splitting recurrence, a wrong initial
condition in the AGM — is caught by a byte-by-byte match on `D`
digits between the two algorithms.

```sh
./target/release/pi --digits 1000000000 -o pi-chud.txt
./target/release/pi --digits 1000000000 --algorithm gauss-legendre -o pi-gl.txt
./target/release/pi --verify pi-chud.txt pi-gl.txt
```

Gauss-Legendre is roughly 2–3× slower than Chudnovsky in practice, so
the cross-check costs roughly the time of one Chudnovsky run again.

**3. BBP spot-checks (`--verify-hex`).**

A third independent path that uses a completely different formula
(Bailey-Borwein-Plouffe) and produces hex digits, not decimal. Run
once to convert your decimal pi file to hex on disk, then sample
cryptographically-randomly-chosen positions until you're satisfied or
a mismatch is detected:

```sh
# First run: convert + verify (the conversion is cached on disk).
./target/release/pi --verify-hex pi-hex.txt --from-decimal pi.txt

# Subsequent runs reuse the converted file:
./target/release/pi --verify-hex pi-hex.txt
```

The conversion step is the long pole (10–30 min for a billion-digit
input); each subsequent run does only spot-checks and starts sampling
in seconds. The random phase runs until SIGINT or a mismatch, so let
it run as long as you want.

## Algorithm details

### Chudnovsky

Chudnovsky brothers' formula:

```
1/π = 12 · Σ_{k=0}^{∞}  (-1)^k (6k)! (Bk + A)
                        ──────────────────────
                        (3k)! (k!)^3 · C^{3k+3/2}
```

with `A = 13_591_409`, `B = 545_140_134`, `C = 640_320`. Each term
contributes about 14.18 decimal digits, so D digits needs ≈ D / 14.18
terms.

The sum is evaluated with **binary splitting** on
`[1, N+1)`. For the half-open range `[a, b)` the recursion returns
integers `(P, Q, T)` so that the partial sum `Σ_{k=a}^{b-1} M_k L_k`
equals `(M_{a-1} · T) / Q` (see `chudnovsky.rs` for the recurrences).
Calling it with `a = 1` makes `M_{a-1} = M_0 = 1`, and the `k = 0` term
(which is just `A`) is folded in at the top:

```
S = (A · Q + T) / Q
π = (426_880 · √10005 · Q) / (A · Q + T)
```

The integers `P`, `Q`, `T` grow to roughly `D` decimal digits each at the
top of the recursion, so the dominant cost is a single GMP multiplication
of two `D`-digit numbers — O(M(D) · log D) total work.

Working precision is `D · log2(10) + 256` bits, and we compute
`D / 14.18 + 8` terms. The 256-bit and 8-term margins are generous; only
the final scaled `pi × 10^(D-1)` multiplication and a handful of GMP
roundings consume any of them.

### Gauss-Legendre / Brent-Salamin

```text
a₀ = 1     b₀ = 1/√2     t₀ = 1/4     p₀ = 1

aₙ₊₁ = (aₙ + bₙ) / 2
bₙ₊₁ = √(aₙ · bₙ)
tₙ₊₁ = tₙ − pₙ · (aₙ − aₙ₊₁)²
pₙ₊₁ = 2 · pₙ

π  ≈  (aₙ + bₙ)² / (4 tₙ)
```

Quadratic convergence: the number of correct binary digits roughly
doubles per iteration, so reaching `P` bits needs ≈ ⌈log₂ P⌉
iterations (we add a small safety margin, currently 2).

The implementation lives in `crates/pi-core/src/algorithm/gauss_legendre.rs`.
It is provided primarily as an *independent* cross-check for Chudnovsky.

### BBP (verification only)

```text
π  =  Σ_{k=0}^∞ (1/16^k) [ 4/(8k+1) − 2/(8k+4) − 1/(8k+5) − 1/(8k+6) ]
```

For position `n`, multiply through by `16^n` and take the fractional
part. The sum splits at `k = n`: the `k ≤ n` half is computed with
modular exponentiation on small (`8k+r`) denominators (so each term
fits in a `u64`); the `k > n` tail decays by `1/16` per step and is
summed in `f64`. The accumulated fractional sum lives in a `u64`
fixed-point register (representing `v / 2^64`); the top 32 bits of
that register are the 8 hex digits we report. Implementation in
`crates/pi-core/src/bbp.rs`. Used by `--verify-hex`; never used to
produce digits.

## Roadmap to one trillion digits

The current implementation handles up to a few hundred million digits on
a workstation with enough RAM. Getting to a trillion needs several
upgrades; the architecture is built to absorb them piece by piece.

1. **Parallel binary splitting.** The left and right subtrees of
   `binary_split` are independent. A `rayon::join` at the top few levels
   gives a 4–8× speedup on a multi-core machine. (The progress reporter
   will need atomics if we go this route.)
2. **Disk-backed integer storage.** At a trillion digits each of `P`,
   `Q`, `T` is roughly 400 GB. We will need an `Integer`-like type that
   spills to disk, or a binary-splitting variant that computes one root-
   to-leaf path at a time and streams partial results.
3. **Faster multiplication.** GMP is excellent but a custom NTT (number-
   theoretic transform) multiplication on disk-resident integers is what
   y-cruncher uses to crack the world records.
4. **Chunked output.** `DigitSink` already supports streaming
   fractional digits in arbitrary-sized chunks, so the algorithm only
   needs to format and emit a slice at a time instead of materializing
   the whole expansion in one string.
5. **Checkpointing.** A trillion-digit run will take hours to days.
   Periodic checkpoints of `(P, Q, T)` (or of whichever bignum state
   replaces them) would let us restart on failure.

For now: a million digits in seconds, and a path to grow.

## Notes on dependencies

The `rug` crate links against GMP, MPFR, and MPC, which are built from
source by the `gmp-mpfr-sys` crate as part of the first `cargo build`.
This needs a C compiler, `m4`, and `make` on the host, which are present
in essentially every Rust developer setup.

