# Developing

For anyone — person or LLM — contributing to this project.

## Setup

```sh
cargo build --release
cargo test --release
cargo clippy --release --all-targets -- -D warnings
```

The first build pulls and compiles GMP, MPFR, and MPC from source via
`gmp-mpfr-sys`. You need a C compiler, `m4`, and `make` on the host
(almost certainly already installed on any machine with a Rust
toolchain). Takes a few minutes the first time; everything subsequent
is incremental.

## Layout

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the architecture in
detail. At a glance:

- `crates/pi-core/` — the math (no I/O, no CLI).
- `crates/pi-cli/` — the binary, CLI, file sinks, progress bars,
  signal handlers.
- `MATH.md`, `IMPLEMENTATION.md` — companion docs.

## Tests

```sh
cargo test --release             # whole workspace
cargo test --release -p pi-core  # library tests only
cargo test --release -p pi-cli   # binary tests only (IntervalSet, …)
```

Tests are colocated with the code under `#[cfg(test)]` modules. The
core invariant covered by the suite is: both compute algorithms, the
BBP digit extractor, and the IntervalSet used by verify-hex all match
a hardcoded reference for the first 100 decimal / hex digits.

For changes that touch a compute algorithm, the bare minimum
acceptance test is "compute 1M digits with both algorithms, both
match the trusted 100M reference (`pi3-100-million-verified.txt`),
both produce byte-identical output to each other." This isn't an
automated test in the suite, but it's the smoke check before pushing.

## Conventions

- **Clippy passes with `-D warnings`.** No `#[allow(clippy::…)]`
  without a rationale comment.
- **Tests pass.** No `#[ignore]` on failing tests; no `--no-verify`
  on commits.
- **Small, focused commits.** Look at `git log --oneline` and the
  commit bodies for the established style — most commits explain the
  *why* and call out the trade-off, not just the *what*.
- **New public API in `pi-core` comes with at least one test** that
  exercises the contract.
- **Memory-conscious patterns matter** for this project specifically.
  Explicit `drop` calls, pre-allocated scratch Floats,
  `std::mem::swap` to promote values without copying — these add a
  line or two and save multi-GB of working set. Worth it.
- **No license file yet, intentionally.** If this project goes public,
  pick one then.

## Notes for AI agents

A few specific things if you're an LLM working in this repo:

- **Read this whole doc and the related ones first.** `MATH.md` and
  `IMPLEMENTATION.md` explain *why* the code is the way it is. The
  inline comments in the source explain subtler local decisions — the
  conversion tail skip in `verify_hex.rs`, the interrupt polling in
  `bbp.rs`, the per-phase rayon scope, the explicit `drop` calls in
  `write_decimal_digits` and `compute`. These are intentional and
  shouldn't be "cleaned up" away.
- **Follow existing patterns rather than introducing new ones.**
  Three small traits (`PiAlgorithm`, `DigitSink`, `ProgressReporter`)
  cover most concerns. New code should slot into them, not invent a
  parallel scheme.
- **Don't widen `pi-core`'s public API speculatively.** The trait
  surface is deliberately small. Adding methods because they *might*
  be useful adds maintenance burden you can't repay.
- **Commit messages are paragraphs, not one-liners.** The subject
  line is the headline; the body explains the trade-off, the
  alternative considered, the empirical reason for the choice. Look
  at the existing `git log` for the established style.
- **When in doubt, ask the user.** They'll tell you which way to go
  and why; that information then often becomes the commit message.

## Roadmap to one trillion digits

The current code computes about 6 billion digits comfortably on an
18 GB laptop and about 100 billion on a cloud VM with 256+ GB RAM
without architectural changes. The trillion-digit milestone needs
more.

### Phase 1 — current

- Single-process, all in-memory.
- Chudnovsky with binary splitting via GMP/MPFR.
- Cross-verified by Gauss-Legendre and by BBP spot-checks.
- Practical ceiling: ~6B on 18 GB, ~100B on 256+ GB.

### Phase 2 — incremental wins (worth doing)

- **Parallel binary splitting.** The left and right subtrees of
  `binary_split` are independent. A `rayon::join` at the top few
  levels of the recursion would give a 4–8× wall-clock speedup on a
  multi-core machine. The progress reporter would need atomic ticks;
  the rest of the algorithm is trivially parallel. Probably an
  afternoon of work for a real win.
- **Faster BBP via Bellard's variant.** About 43% fewer modular
  exponentiations than classical BBP. Drop-in replacement for
  `hex_digits_at`.
- **Streaming decimal output.** The current decimal-conversion code
  materializes the full ~D-digit string in memory. Chunked output
  (write 100 MB at a time, free that chunk, emit the next) would
  reduce peak memory by another ~`D` bytes. The `DigitSink` interface
  allows multiple `write_fractional_digits` calls, so the contract
  already permits this — only the caller (`write_decimal_digits`)
  needs to be reworked.

### Phase 3 — disk-backed bignum (necessary for >100B)

This is the big architectural change. Past ~100B digits, the
fundamental constraint is that each of the three top-level integers
(`P`, `Q`, `T`) is too big to fit in RAM alongside the others. The
fix is to spill them to disk during the inner combines and stream
chunks back in as the multiplication needs them.

The shape of this work:

- A `BigIntBackend` trait in `pi-core` with at least two
  implementations: the current `rug::Integer` wrapper (for everything
  that fits in RAM) and a new `DiskBackedInteger` that mmaps a temp
  file.
- The disk-backed type needs, at minimum, chunked multiplication
  (which in turn needs an FFT that operates on disk-resident data),
  addition, integer division, and the ability to be converted to a
  `Float` of arbitrary precision.
- Once that exists, the algorithm code stays the same — it just picks
  the right backend based on size.

This is a multi-week project. Don't introduce the trait
speculatively — design it once we have empirical data on what
disk-backed ops actually need to do efficiently.

### Phase 4 — distributed (for ~1T and beyond)

At a trillion digits, even disk-backed bignum on a single machine
gets uncomfortable — multiple TB of fast SSD, hours per top-level
multiplication. The endgame is distributed compute: shard the big
integers across multiple nodes, communicate via MPI or similar.

This is a fundamentally different project. y-cruncher gets to a
trillion-plus on a single (very large) machine; achieving the same
distributedly would be more interesting.

## Open items

- **Conversion is uninterruptible.** During the ~10–30-minute
  decimal→hex conversion for billion-digit verify-hex inputs, Ctrl-C
  doesn't interrupt the in-flight GMP operation; it just sets the
  flag, and the user waits for the current step to finish. Threading
  cooperative cancellation into GMP isn't really an option (GMP
  doesn't expose hooks). The pragmatic options are: break the
  conversion into smaller resumable chunks, or just live with it.
- **The hex file's last digit is unverifiable.** See `TAIL_SKIP` in
  `verify_hex.rs`. The conversion can be off by 1 in the LSB, and
  the only fix is more input precision (one extra decimal digit at
  conversion time). Currently mitigated by skipping the last 32 hex
  digits during verification.
- **No CI.** Tests pass locally; nothing is automated. A basic GitHub
  Actions workflow (run cargo test + clippy on push) would be cheap
  to add.

---

For the math, see [MATH.md](MATH.md). For how the code is laid out,
[IMPLEMENTATION.md](IMPLEMENTATION.md).
