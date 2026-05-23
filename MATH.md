# The math

How do you actually *compute* a billion digits of π?

It's a fair question. π is the ratio of a circle's circumference to its
diameter — easy to define, awful to compute directly. No real circle
will hold a billion correct digits, and the draw-and-measure method
falls apart somewhere around the third.

This is the story of three formulas that make a billion digits not just
possible but quick.

## A quick history detour

People have been computing more digits of π for the same reason they
climb mountains: because they're there. The methods get more ingenious
as the digit counts grow.

- **Archimedes** (~250 BCE) used inscribed and circumscribed polygons
  to bound π. He got `3.1408 < π < 3.1429`, which is two correct
  digits.
- **Leibniz** (1674) discovered the achingly beautiful series
  `π/4 = 1 − 1/3 + 1/5 − 1/7 + …`. It converges so slowly that you
  need about five billion terms for just ten correct digits.
- **Machin** (1706) wrote `π/4 = 4 arctan(1/5) − arctan(1/239)` and
  computed 100 digits by hand.
- **Ramanujan** (1914) published a catalog of bizarre, fast-converging
  series, each one looking unmotivated until you understand modular
  forms.
- **Brent** and **Salamin** (independently, 1976) found an iteration
  based on the *arithmetic-geometric mean* that doubles the number of
  correct digits per step.
- **Chudnovsky brothers** (1988) refined Ramanujan's approach into
  the formula y-cruncher still uses for world records.
- **Bailey, Borwein, and Plouffe** (1995) discovered a formula that
  lets you compute the n-th hexadecimal digit of π *without computing
  any of the earlier ones*. They found it by running an integer-
  relation algorithm on high-precision data — a computer noticed it
  before any human suspected it existed.

This repository implements the last three.

---

## The Chudnovsky algorithm

The headline formula:

```text
1            ∞       (-1)^k · (6k)! · (545,140,134·k + 13,591,409)
─ = 12  ·    Σ      ─────────────────────────────────────────────
π           k=0          (3k)! · (k!)³ · 640,320^(3k + 3/2)
```

Each term contributes about **14.18 decimal digits** to π. For
comparison, each Leibniz term adds about one *bit*, badly. To get a
billion digits, you sum about 70 million Chudnovsky terms. That's not
many, by the standards of exotic series.

### Where does 14.18 come from?

After the (`(6k)!`, `(3k)!`, `k!`) factorials cancel, the ratio of
consecutive terms simplifies to

```text
|t_{k+1} / t_k|  ≈  1728 / 640,320³  ≈  6.6 × 10⁻¹⁵
```

So each new term shrinks the running tail by roughly fifteen orders of
magnitude, and `log₁₀(640,320³ / 1728) ≈ 14.18`. That's the convergence
rate.

### Where do those magic constants come from?

This is where it gets really cool.

The constants `640,320`, `545,140,134`, and `13,591,409` aren't
arbitrary. They come from the theory of *modular forms* — specifically,
from evaluating the j-invariant at the imaginary quadratic point
`τ = (1 + i√163) / 2`.

163 is the largest [Heegner number][heegner] — one of a special list of
integers `(1, 2, 3, 7, 11, 19, 43, 67, 163)` for which the ring of
integers in `ℚ(√−d)` has unique factorization. These numbers come up
all over deep number theory.

Their best-known party trick is **Ramanujan's constant**:

```text
e^(π√163) ≈ 262,537,412,640,768,743.99999999999925...
```

Twelve nines after the decimal point. Math is weird.

That near-integer is `744 + 640,320³` plus a tiny correction. The
"640,320" in Chudnovsky's formula is exactly that 640,320. Chudnovsky's
series is one of an infinite family of [Ramanujan–Sato series][rs],
parameterized by Heegner numbers; choosing the largest one (163) gives
the fastest convergence.

You don't need to understand any of that to use the formula, but it's
fun to know that "14.18 digits per term" comes from a chain that leads
back to elliptic curves with complex multiplication.

[heegner]: https://en.wikipedia.org/wiki/Heegner_number
[rs]: https://en.wikipedia.org/wiki/Ramanujan%E2%80%93Sato_series

### Binary splitting

The formula is great. The naive way to evaluate it isn't.

If you compute each term as a high-precision rational and add them up,
the cost is roughly `O(N · M(D))`, where `N` is the number of terms
and `M(D)` is the cost of multiplying two `D`-digit numbers. For a
billion digits that's about `N ≈ 7 × 10⁷` terms and `M(D) ≈ D · log² D`
ops per multiplication. Multiply through and you get something on the
order of 10¹⁹ operations — many years on a single core.

The trick is **binary splitting**, a fairly general technique for
summing rational series fast. The idea: terms of the form
`Σ a₀a₁⋯aₖ / b₀b₁⋯bₖ` (a "linearly recurrent" series) have a partial
sum that can be computed by combining two halves with a single
multiplication of products at each level of a binary tree.

For Chudnovsky, after some bookkeeping, the combine rule is:

```text
For a range [a, b):
    P_{a,b} = P_{a,m} · P_{m,b}
    Q_{a,b} = Q_{a,m} · Q_{m,b}
    T_{a,b} = T_{a,m} · Q_{m,b} + P_{a,m} · T_{m,b}
```

The atomic case (`b − a = 1`) computes one term's `(p_k, q_k, t_k)`
directly.

Why is this fast? At depth `ℓ` of the recursion tree, integers are
roughly `D / 2^ℓ` digits and there are `2^ℓ` combines. Using FFT
multiplication `M(n) = O(n log n log log n)`, the cost at each level
is roughly `O(D log D log log D)`, and there are `O(log D)` levels.
Total: `O(D · log² D · log log D)`.

So binary splitting turns the naive `O(D² · polylog)` summation into
`O(D · log² D · polylog)` — the difference between "computable in your
lifetime" and "computable while you make a sandwich."

---

## Gauss-Legendre / Brent-Salamin

Completely different approach. Same answer.

### The arithmetic-geometric mean

In 1799, when Gauss was 22, he was hand-computing the
**arithmetic-geometric mean** of two numbers. The recurrence:

```text
a₀ = some starting value          a_{n+1} = (a_n + b_n) / 2
b₀ = some starting value          b_{n+1} = √(a_n · b_n)
```

Both sequences converge to the same limit, called `M(a₀, b₀)` — the
AGM. Gauss tried `a₀ = 1, b₀ = √2`, computed `M(1, √2) ≈ 1.198140234…`,
and noticed it matched (to as many digits as he had) the inverse of
`(2/π) · ∫₀¹ dt / √(1 − t⁴)`, a *lemniscatic integral* from elliptic
function theory. He wrote in his diary that this observation "will
surely open up a wholly new field of analysis."

It did. Within a few years, the AGM became the foundation of the
theory of elliptic functions.

### Quadratic convergence

The AGM converges *quadratically*: the error halves *and squares* at
each step. Concretely, if `cₙ = (aₙ − bₙ) / 2`,

```text
c_{n+1}  ≈  cₙ² / (4 · M)
```

So each iteration roughly doubles the number of correct digits. About
33 iterations gets you to a billion correct digits. (Brent calls this
"halving the distance you have left to go, twice.")

For comparison: Chudnovsky's series adds *linearly* with the term
count — about 14 digits per term — and needs ~70 million terms for a
billion digits. The AGM gets there in 33. The per-iteration work is
heavier, but the iteration count is comically smaller.

### The Brent-Salamin formula

The connection from AGM to π goes through elliptic integrals. The
complete elliptic integral of the first kind,

```text
K(k)  =  ∫₀^(π/2)  dθ / √(1 − k² sin² θ),
```

satisfies the surprisingly clean identity `K(k) = π / (2 · AGM(1, √(1 − k²)))`.
Combine that with Legendre's relation between `K(k)`, `K(k')`, `E(k)`,
and `E(k')` (where `k' = √(1 − k²)`) and you can derive — by some
careful algebra — the following π-computing recurrence:

```text
a₀ = 1     b₀ = 1/√2     t₀ = 1/4     p₀ = 1

a_{n+1} = (a_n + b_n) / 2
b_{n+1} = √(a_n · b_n)
t_{n+1} = t_n − p_n · (a_n − a_{n+1})²
p_{n+1} = 2 · p_n

π  =  lim_{n→∞}  (a_n + b_n)² / (4 · t_n)
```

That's the whole algorithm — five recurrence lines. Brent (1976) and
Salamin (1976) discovered this independently and published almost
simultaneously. It's been a standard tool for high-precision π
computation ever since.

### Chudnovsky vs. AGM

Both are roughly `O(D · log² D)` for `D` digits, but with very
different constants. In practice Chudnovsky is faster by a factor of
two or three because binary splitting evaluates exactly using integer
arithmetic until a single division at the very end, while
Gauss-Legendre needs full-precision Float operations from iteration
one (the `√(aₙbₙ)` step).

We have both implementations because they're entirely independent. If
either has a bug, the other won't reproduce it.

---

## Bailey-Borwein-Plouffe

In 1995, David Bailey, Peter Borwein, and Simon Plouffe published this
identity:

```text
         ∞      1     ⎛   4         2         1         1   ⎞
π   =    Σ    ─── · ⎜ ─────  −  ─────  −  ─────  −  ───── ⎟
        k=0  16^k   ⎝ 8k+1     8k+4      8k+5      8k+6   ⎠
```

That's a fine series with a nice closed form. The headline, though, is
this: the formula lets you compute **the n-th hexadecimal digit of π
without computing any of the preceding digits**, in about the time it
takes a single Chudnovsky term to evaluate.

### How the spigot works

To get the n-th hex digit, compute the fractional part of `16^n · π`
and multiply by 16. The integer part of *that* is your digit.

Why does this work? Look at what happens when you multiply the BBP
series through by `16^n`:

```text
16^n · π  =  Σ_k  16^(n−k)  ·  [ 4/(8k+1)  −  2/(8k+4)  −  …  ]
```

Split the sum at `k = n`:

- **`k ≤ n`**: the factor `16^(n−k)` is a non-negative integer power
  of 16. We only care about the *fractional* part of the whole sum, so
  we can compute each term modulo 1. For each `8k+r` in the
  denominator, what matters is just `(16^(n−k) mod (8k+r)) / (8k+r)`.
  And `16^(n−k) mod (small integer)` is fast — binary modular
  exponentiation in `O(log n)` operations per term.

- **`k > n`**: the factor `16^(n−k)` is a tiny fraction (1/16,
  1/256, …). The terms decay geometrically; about 20 of them are
  enough to pin down the fractional part to full precision.

Cost: roughly `O(n)` modular exponentiations of small integers, each
`O(log n)` machine ops. Total: `O(n · log n)` machine ops per
8-hex-digit extraction. The asymptotic is friendly; the constant
factor is not — the inner loop is dominated by `u128 % m`, and in
this implementation a single 8-digit extraction at `n = 10⁹` takes
roughly 15–20 minutes on one core. That's why verification samples
deep positions sparingly.

### Why this is wild

There is no known formula like this for **base 10**.

Bellard (1997) found an algorithm that extracts individual *decimal*
digits, but it's `O(n²)` — at `n = 10⁹` that means weeks per decimal
digit, where hex BBP needs minutes. The asymptotic gap reflects a
deep structural fact: π has identities that play nicely with binary
arithmetic in a way it doesn't with base 10.

### How it was discovered

Maybe the most wonderful part: BBP was discovered **experimentally**.

Bailey, Borwein, and Plouffe were exploring identities for π using the
[PSLQ integer-relation algorithm][pslq]. PSLQ takes a list of real
numbers (computed to very high precision) and finds *integer*
coefficients such that their linear combination is approximately zero.
If the coefficients turn out to be small, it's strong empirical
evidence that an exact linear relation holds.

They computed high-precision values of `π` and of several series of the
form `Σ 1/(16^k · (8k + r))`, ran PSLQ, and the algorithm returned the
integer combination that became the BBP formula. Once you have a
candidate identity, you prove it analytically by integrating
appropriate rational functions over `[0, 1/√2]`.

This is *experimental mathematics* in the most literal sense: the
computer noticed a theorem and the humans proved it afterward.

[pslq]: https://en.wikipedia.org/wiki/Integer_relation_algorithm

---

## Why three is better than one

Each compute or verify path in this repository goes through the same
math, but via completely different formulas:

| Path             | Origin                              | Approach                                                                                  |
|------------------|-------------------------------------|-------------------------------------------------------------------------------------------|
| Chudnovsky       | Ramanujan-style series + binary splitting | Sum a fast-converging series exactly with integer arithmetic, divide once at the end |
| Gauss-Legendre   | AGM iteration                       | Iterate two means plus a running correction; take a ratio of Floats at the end            |
| BBP              | Spigot via modular exponentiation   | Compute individual hex digits independently; never produces decimal output                |

They share the bignum backend (GMP/MPFR) and the decimal-output code
that prints the final answer. *Every other thing* is different: every
formula constant, every recurrence, every loop body. A bug specific to
one algorithm cannot survive byte-by-byte agreement with another.

This isn't *proof* the digits are right — only a formal proof system
could give you that. But "three completely independent computations
agree on every byte of N billion digits" is the strongest empirical
correctness argument available short of a formal proof, and it's the
argument this repository runs on.

---

For how the algorithms are wired up in code, see
[IMPLEMENTATION.md](IMPLEMENTATION.md). For contributing,
[DEVELOPING.md](DEVELOPING.md).
