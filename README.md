# pi

A command-line tool in Rust that computes π to a billion (and counting)
decimal digits, with three independent ways to prove the digits are
actually right.

I've been writing about the journey at
**[onetrilliondigits.blogspot.com][blog]** — the blog has the *why*
and the meandering detours; this repository has the *what*.

[blog]: https://onetrilliondigits.blogspot.com

## Try it

```sh
cargo build --release            # first build compiles GMP/MPFR; takes a few minutes
./target/release/pi --digits 100
```

A million digits is well under a second on a modern laptop. A billion
is roughly ten minutes. Ten billion wants a bigger machine.

Bare `pi` (no flags) prints the help text.

## What it does

`pi` is one binary with three modes, picked by which flags you pass.

**Compute** N digits of π, using either of two algorithms:

```sh
# Default: Chudnovsky series + binary splitting.  Same algorithm class
# y-cruncher uses for world records.
pi --digits 1000000 -o pi.txt

# Independent algorithm: Gauss-Legendre AGM iteration.
pi --digits 1000000 --algorithm gauss-legendre -o pi-gl.txt
```

**Compare two digit files** byte-by-byte, with sensible defaults:
trailing whitespace ignored on both sides, shorter file treated as a
prefix of the longer. So a freshly computed 1M-digit file verifies
cleanly against a 100M-digit reference.

```sh
pi --verify pi.txt some-trusted-reference.txt
```

**Independent spot-check** via the [Bailey-Borwein-Plouffe formula][bbp]
— a completely separate code path that computes individual hexadecimal
digits of π. Run once to convert your decimal file to hex; subsequent
runs reuse the converted file and start sampling immediately.

```sh
pi --verify-hex pi-hex.txt --from-decimal pi.txt   # first time
pi --verify-hex pi-hex.txt                          # later runs
```

The random-sampling phase runs until you Ctrl-C it or it catches a
mismatch.

[bbp]: https://en.wikipedia.org/wiki/Bailey%E2%80%93Borwein%E2%80%93Plouffe_formula

## Three independent paths

Two compute algorithms ([Chudnovsky][chud], [Gauss-Legendre][gl]) plus
one verification oracle (BBP). They share the bignum library and the
final decimal-output code, but every formula constant and every loop
body is different. Any algorithm-specific bug is caught by byte-by-byte
agreement with another path.

[chud]: https://en.wikipedia.org/wiki/Chudnovsky_algorithm
[gl]: https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_algorithm

## How big can I push it?

| Hardware                | Comfortable ceiling |
|-------------------------|---------------------|
| Laptop, 18 GB RAM       | ~6 billion digits   |
| Laptop, 64 GB RAM       | ~20–25 billion      |
| Cloud VM, 128 GB RAM    | ~50 billion         |
| Cloud VM, 256+ GB RAM   | ~100 billion        |

Past ~100B the natural next step is disk-backed bignum arithmetic. See
[DEVELOPING.md](DEVELOPING.md) for the roadmap to a trillion.

## Read more

- **[MATH.md](MATH.md)** — the mathematics behind the algorithms.
  Written for anyone with college-level calculus who finds the
  history and the formulas interesting. No PhD required.
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** — how the code is laid
  out and why. Aimed at someone with a CS background who wants to
  read or modify the source.
- **[DEVELOPING.md](DEVELOPING.md)** — what you need to know to
  contribute, plus the roadmap toward a trillion digits.

## Build notes

The `rug` crate links against GMP, MPFR, and MPC, which are built from
source by `gmp-mpfr-sys` on first `cargo build`. You need a C compiler,
`m4`, and `make` on the host (almost certainly already installed on any
machine with a Rust toolchain). The first build takes a few minutes;
subsequent rebuilds are fast.
