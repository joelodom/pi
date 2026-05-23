# pi

A CLI program in Rust for computing pi to arbitrary precision, designed
to start at a million digits and scale upward.

Two algorithms are implemented:

* **Chudnovsky** with binary splitting (default; fastest in practice).
* **Gauss-Legendre / Brent-Salamin** AGM iteration (independent
  cross-check algorithm).

Both run on GMP/MPFR via the [`rug`](https://crates.io/crates/rug)
crate. A million digits finishes in well under a second on a modern
laptop with either algorithm.

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
./target/release/pi --verify pi.txt pi3-100-million.txt

# Compute 100 digits to stdout.
./target/release/pi --digits 100
```

Run `./target/release/pi --help` for the full flag list.

## CLI

`pi` has two modes, picked by which flags are given:

* **Compute** — `pi --digits N [-o FILE] [--algorithm ALG] [--no-progress]`.
  After writing to a file, the CLI prints the suggested `pi --verify ...`
  invocation.
* **Verify** — `pi --verify FILE_A FILE_B`.  Trims trailing whitespace
  (`' '`, `'\t'`, `'\n'`, `'\r'`) from both files, then byte-by-byte
  compares the shorter content against the matching prefix of the longer
  content.  On mismatch it reports the first differing byte offset and
  exits non-zero.  Skips computation entirely.

| Flag | Default | Description |
|------|---------|-------------|
| `-d, --digits N` | `1000000` | Decimal digits to compute (counting the leading `3`). |
| `-o, --output FILE` | `-` | File to write digits to. Use `-` for stdout. |
| `--algorithm NAME` | `chudnovsky` | `chudnovsky` (series + binary splitting) or `gauss-legendre` (AGM iteration). |
| `--no-progress` | | Suppress the progress bar (it goes to stderr). |
| `--verify FILE_A FILE_B` | | Byte-by-byte compare two digit files, ignoring trailing whitespace. Runs instead of compute. |

Running `pi` with no arguments prints the help text.

## Repository layout

```
pi/
├── Cargo.toml              # Workspace
└── crates/
    ├── pi-core/            # Library: algorithms, sinks, planning, progress
    │   └── src/
    │       ├── lib.rs
    │       ├── algorithm/
    │       │   ├── mod.rs       # PiAlgorithm trait + registry
    │       │   └── chudnovsky.rs
    │       ├── output.rs        # DigitSink trait + WriterSink
    │       ├── precision.rs     # PrecisionPlan
    │       └── progress.rs      # ProgressReporter trait + NoopProgress
    └── pi-cli/             # Binary `pi`
        └── src/main.rs
```

## Architecture

The library is organized around a few small traits so each concern can be
swapped out independently. Adding a new algorithm, output destination, or
progress backend does not touch the others.

### `PiAlgorithm` (`crates/pi-core/src/algorithm/mod.rs`)

```rust
pub trait PiAlgorithm {
    fn name(&self) -> &'static str;
    fn digits_per_term(&self) -> f64;
    fn compute(
        &self,
        digits: u64,
        sink: &mut dyn DigitSink,
        progress: &mut dyn ProgressReporter,
    ) -> anyhow::Result<()>;
}
```

To add a new algorithm (Gauss-Legendre, BBP, Borwein quintic, …) implement
this trait and add a variant to `AlgorithmKind`. Nothing in the CLI or the
sinks changes.

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
    fn start_phase(&mut self, name: &str, total: u64);
    fn tick(&mut self);
    fn end_phase(&mut self);
}
```

The CLI implements an `indicatif`-backed reporter; tests use `NoopProgress`.
Any other backend (structured logging, telemetry, web socket) plugs in the
same way.

### `PrecisionPlan` (`crates/pi-core/src/precision.rs`)

Decides the working precision (mantissa bits for `rug::Float`) and the
number of series terms for a given target digit count. Each algorithm
declares its `digits_per_term`; the planner uses that to choose the term
count, then adds a fixed safety margin in both terms and bits.

## Cross-algorithm verification

The two algorithms share only the bignum backend, the precision plan,
and the decimal-conversion code; the actual pi computation goes through
two entirely independent code paths (series + binary splitting vs.
quadratically-convergent AGM iteration with sqrt). Any bug specific to
one of them — wrong formula constants, an off-by-one in the
binary-splitting recurrence, a wrong initial condition in the AGM — is
caught by a byte-by-byte match on `D` digits between the two
algorithms.

To cross-check a D-digit computation:

```sh
./target/release/pi --digits 1000000000 -o pi-chud.txt
./target/release/pi --digits 1000000000 --algorithm gauss-legendre -o pi-gl.txt
./target/release/pi --verify pi-chud.txt pi-gl.txt
```

Gauss-Legendre is roughly 2–3× slower than Chudnovsky in practice, so
the cross-check costs roughly the time of one Chudnovsky run again.

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

