# 10B-digit run handoff

## Update 2026-05-25 (overnight session)

The 10B run from 2026-05-24 panicked partway through binary splitting
with `assert!(n <= (1usize << 32))` in `mul_mag_ntt`.  The next 10B
attempt and any 100B attempt need a rebuilt binary that includes the
overnight fixes below.

### Commits added this session

```
841c1d0 bignum: route oversized NTT multiplications through Karatsuba splitting
eb20dc8 bignum: scratch buffer pool for mul_mag_ntt to cut peak-RSS spikes
2ebf4ea bignum: parallelize build_twiddles for large counts
57f09ea bignum: cache NTT twiddle tables across mul_mag_ntt calls
4cfa63e bignum: parallel bit_reverse with raw-pointer swaps
424486b bignum: parallel unpack for large NTT outputs
```

### What each does (and why it matters at 1B+ digits)

* **NTT routing fix (841c1d0)** — operands too large for a single
  Goldilocks-prime NTT (N > 2^32 coeffs) used to hit a hard `assert!`.
  Now they're routed through `mul_mag_karatsuba`, which halves the
  operands recursively until each leaf fits.  At 100B digits this
  costs three Karatsuba levels (27 NTT leaves at N ~ 2^31).
  *Without this, 100B can't even start.*

* **Scratch buffer pool (eb20dc8)** — `mul_mag_ntt` no longer
  allocates fresh `Vec<u64>` workspace on every call.  An RAII handle
  pulls from a per-size pool (cap 4 ordinary, 1 huge ≥ 1 GiB).  At
  100B with 16 GB scratch per leaf NTT × 27 leaves, this is the
  difference between paging the allocator constantly and reusing
  pages.  Direct effect: lower peak-RSS variance → cheaper instance.

* **Parallel build_twiddles (2ebf4ea)** — twiddle recurrence is
  serial as written.  At N=2^31 the largest pass's twiddle build was
  a 5+ s serial chain (worst-case 10+ s) per NTT call.  Split into
  chunks with precomputed per-chunk starting omegas, runs across
  rayon workers.

* **Twiddle cache (57f09ea)** — twiddle tables are pure functions of
  `(count, inverse)`.  Built once per (size, direction) and reused
  via `Arc<Vec<u64>>`.  Binary splitting hits the cache constantly
  because most mults at any tree level share the same N.  Per-table
  cap of 2^28 entries (≤ 2 GiB) keeps any single cached table
  bounded.

* **Parallel bit_reverse (4cfa63e)** — bit-reverse is cache-hostile
  (`a[i]` swap with `a[rev(i)]`, distant in memory).  At N=2^31,
  ~1 B swaps × ~16 B per swap = ~16 GB of memory traffic that the
  serial loop runs in maybe 30 s.  Pairs `{i, rev(i)}` are disjoint
  so iterations are race-free; parallelized via a raw-pointer
  wrapper.  Engages above N = 2^18.

* **Parallel unpack (424486b)** — the last serial step in the NTT
  pipeline.  At N=2^31 the carry-chain loop processes ~2 B
  coefficients serially (~5–10 s wall time per call).  Partition
  into chunks at COEFFS_PER_LIMB block boundaries, each chunk runs
  the recurrence assuming `carry_in=0`, and a serial merge pass
  propagates inter-chunk carries.  Engages above 2^20 coeffs
  (≈ 250 K output limbs).  Together with the other parallelizations
  above, the entire `mul_mag_ntt` pipeline (pack, forward, pointwise,
  inverse, unpack) is now parallelized end-to-end.

### Tests added this session

15 new tests, all in `crates/bignum/src/`:

* `integer::tests::karatsuba_above_ntt_routes_correctly_on_oversize`
* `ntt::tests::pool_reuses_released_buffer_at_same_size`
* `ntt::tests::pool_caps_at_max_per_size_under_repeated_ntt`
* `ntt::tests::pool_reuse_preserves_correctness`
* `ntt::tests::build_twiddles_parallel_matches_serial`
* `ntt::tests::build_twiddles_at_threshold`
* `ntt::tests::build_twiddles_uneven_chunks`
* `ntt::tests::twiddle_cache_hit_returns_correct_table`
* `ntt::tests::twiddle_cache_forward_and_inverse_are_distinct`
* `ntt::tests::twiddle_cache_preserves_ntt_correctness`
* `ntt::tests::bit_reverse_parallel_matches_serial`
* `ntt::tests::bit_reverse_parallel_is_involution`
* `ntt::tests::bit_reverse_parallel_via_ntt_round_trip`
* `ntt::tests::unpack_parallel_matches_serial`
* `ntt::tests::unpack_parallel_cross_chunk_carry`

Full bignum suite: 81 passing (was 65 before this session).
### Smoke tests run on the downsized 2-vCPU box

| Digits | Wall    | Peak RSS | vs reference                   |
|--------|---------|----------|--------------------------------|
| 1M     | 13 s    | ~50 MB   | first 50 digits match canonical π |
| 2M     | n/a     | ~120 MB  | byte-identical with 5M overlap  |
| 3M     | 25 s    | ~190 MB  | byte-identical with 5M overlap  |
| 5M     | 36 s    | 329 MB   | byte-identical with 10M overlap |
| 10M    | 80 s    | 615 MB   | byte-identical with 5M overlap  |
| 20M    | 174 s   | 1.2 GB   | byte-identical with 10M overlap |

Memory scaling is roughly linear with digits (matches the generator's
5.4 GB / 100M model); wall time is roughly linear at this scale on
2 cores (no NTT cache effects yet because N stays under L3).

### Suggested next steps (not done — too risky for unattended)

1. **Four-step / cache-blocked NTT** (the deferred `√N × √N` matrix
   decomposition called out in `ntt.rs`'s module doc).  This is the
   single biggest theoretical win at N > 2^25 because late butterfly
   passes currently miss cache on essentially every access.  But the
   implementation is fiddly (transpose, twiddle correction, sub-FFT
   wiring) and the failure modes are subtle.  Recommend a focused
   daytime session with the user available.

2. **Radix-4 butterflies** — fewer total mul/add ops per pass, ~10–
   15 % NTT speedup.  Less risky than four-step but still a real
   rewrite of the butterfly inner loops.

3. **Parallel `unpack`** — currently serial.  At N=2^31 it's ~5–10 s
   per call.  Cross-chunk carry merge is the tricky bit; ~3 % NTT
   speedup, not the biggest fish.

4. **Re-run 10B** with the rebuilt binary on a smaller, cheaper
   instance.  The pool + cache reductions should let it succeed with
   the existing memory footprint (~240 GB peak); the wall-time
   improvements are speculative until measured.  Recommend a
   `c8g.metal-48xl` or step down to `c7g.16xlarge` with `--digits
   1_000_000_000` first to recalibrate the predicted-vs-actual peak
   RSS curve with the new pool behaviour.

### House rules (unchanged)

* **Don't commit without explicit permission** — overnight session
  was given blanket permission for the NTT improvement work; that
  authorization ended with this handoff.  Next change needs its own
  ask.
* **Run-size ceiling: 20M digits** without explicit instruction in
  the current turn.

---

## Original handoff (before the failed 10B run)



Context for picking up the pi-on-EC2 work in a fresh session.  Written
2026-05-24 before a 10B-digit attempt on a Graviton instance; revise
or delete once the 10B is complete and analysed.

## Where we are

Branch: `pure-rust-poc`.  Last four commits are the relevant arc:

```
595be8f perf: milestone events + Linux io/swap counters for 10B-run analysis
147897b pi: fix generator overestimate; overlap sqrt(10005) with binary splitting
50137d3 bignum: disk-backed limb storage for billion-digit runs
98b1f4a pi: auto-apply hardware-tuned config by default; embed config in perf log
```

Working tree is clean.  Everything below is committed; no uncommitted
WIP to recover.

## The immediate experiment

Run 10B digits on a c7g.16xlarge (64 vCPU Graviton3, 128 GB RAM) —
the user said they can scale the existing instance up to the biggest
c7g.  This is the first attempt at 10B on this codebase.

Command (no config override — generator now picks correctly):

```bash
./target/release/pi --digits 10000000000 \
    --output pi-10b.txt \
    --performance-file perf-10b-ec2.jsonl
```

Storage: pi-10b.txt is ~10 GB.  Need a ~200 GB gp3 root volume for
headroom.

Predicted wall on c7g.16xlarge: **~5–6 hours**.  Extrapolation from
the 1B at 30:17 on the same instance, adjusted for the ~20–30 min
saved by the new sqrt overlap.  Run may be killed before bedtime to
contain cost — instrumentation is built around that case.

Generator pick for 10B on 64c/128GB (verified locally):
* `disk_limb_threshold = 5_000_000` (engaged — 10B genuinely doesn't
  fit in 128 GB).
* `sequential_top_threshold = 333_333_333` (top 1–2 BS levels
  serialise, levels 3+ parallel).
* `parallel_final_assembly = true`.

## Why each recent commit matters

### 595be8f — perf instrumentation

Three additions aimed at making a partial perf log diagnose-able:

* **Milestone events** decompose FA into
  `fa.denom_construct`, `fa.sqrt_join`, `fa.parallel_chains`,
  `fa.final_mul`, and DC into `dc.scale_build`, `dc.scale_mul`,
  `dc.to_integer`, `dc.to_string`.  Subtract start.t_ms from end.t_ms
  for sub-phase wall.  See `crates/pi-core/src/algorithm/chudnovsky.rs`
  and `crates/pi-core/src/algorithm/util.rs`.
* **Linux /proc/self/io fields** on every sample:
  `io_rchar`, `io_wchar`, `io_read_bytes`, `io_write_bytes`.
  Distinguishes page-cache hits (rchar - read_bytes) from real SSD
  reads (read_bytes).  Critical for diagnosing disk-backing cost.
* **Linux VmSwap** as `swap_mb`.  Should remain 0 on EC2 (no swap).
  If it ever rises, the kernel is in trouble.

Overhead measured at ≤ noise floor (10M A/B on laptop: with-perf
runs averaged 9.64s, no-perf 9.88s — perf-enabled was *faster* on
the noisy laptop, confirming it's invisible).  At the 2000ms sample
cadence the generator picks for ≥ 5B, perf overhead on the 10B run
is order-of-seconds out of order-of-hours.

### 147897b — generator fix + sqrt overlap

* **Generator est_peak fix**: removed the bogus `cores/11` scaling
  factor that made the generator overestimate peak RSS by ~10× on
  many-core hosts (predicted 314 GB at 1B on 64 cores; actual 31 GB).
  This was forcing disk-backing to engage on big-RAM EC2 instances
  unnecessarily.  New formula is flat-in-cores, linear-in-digits:
  `5.4 GB × digits / 100M`.  Verified picks for 1B / 10B on
  c7g.16xlarge in the commit message.

* **sqrt(10005) overlap with BS**: a `std::thread` named
  `chudnovsky-sqrt` is spawned at the top of `Chudnovsky::compute`
  before BS starts.  It computes `Float::with_val_64(prec, 10_005)
  .sqrt_mut() * 426_880` in parallel with BS, sharing the rayon pool
  for its internal NTT muls.  FA waits on the handle (typically
  immediate at scale) and the pi_numer chain shrinks to just `*= &q`.
  Expected win on 10B: ~20–30 minutes off FA wall.

  The `fa.sqrt_join` milestone duration directly reports how much
  wait time the join cost.  At small N this will be non-zero (BS too
  short for sqrt to finish); at 10B it should round to zero.

### 50137d3 — disk-backed limb storage

`Integer.limbs` is now a `LimbStorage` enum: `Memory(Vec<u64>)` for
the common path and `Mapped { mmap, path, len, capacity }` for
buffers ≥ `disk_limb_threshold` u64s.  Each Mapped allocation uses
an unlinked scratch file in `std::env::temp_dir()` (or
`bignum.scratch_dir` if set), calls `madvise(SEQUENTIAL)`, and drops
its `File` handle immediately so the kernel keeps the inode alive
via the mmap (avoids FD exhaustion).  Atomic counters in
`bignum::storage` expose live/cumulative mmap bytes and counts;
these appear in every perf JSONL `sample` event.

See the "Disk-backed limb storage" section in CLAUDE.md for the
mmap shape, threshold heuristic, and the macOS major-fault caveat.

## The two parallelism bottlenecks to watch for at 10B

These are known weak spots from the 1B EC2 data.  If the 10B run
shows the same pattern, they're the next optimization targets.

### FA caps at ~12 cores

On the 1B EC2 run (64 cores available), FA mean cores was 11.9 —
**19% utilization**, vs 41% for BS and 49% for DC.  The internal
sequencing of FA ops is serial (denom_construct → sqrt_join →
parallel_chains → final_mul); each individual op uses inner
parallelism but only one runs at a time.

At 10B this becomes order-of-an-hour wall time on its own.  Future
work: overlap the chains across phase boundaries (e.g.  start
`denom_construct` during BS's final combine when `q` and `t` are
already available; start `final_mul` setup as soon as `recip` or
`pi_numer` is ready).

The new `fa.*` milestones let post-hoc analysis verify the
bottleneck is still where 1B data said it was.

### DC's to_string dominates

At 1M on the laptop, `dc.to_string` was 235ms out of 273ms of DC
total — **86% of DC wall**.  The `to_string` is
`bignum::Integer::to_string`, which converts an N-limb integer to
a decimal string.

If this fraction holds at 10B, DC's `to_string` is ~9% of total wall.
A divide-and-conquer to_string (we may have one — there's a
`to_string_dc_threshold` knob) likely already engages at billion
scale.  Verify in the perf log; if it's not engaging, that's a
single-knob fix.

## Optimization candidates beyond what's shipped

Ordered by estimated bang for buck:

1. **FA chain pipelining** (above) — the biggest remaining win, but
   non-trivial: requires breaking the strict
   denom→sqrt_join→parallel→final ordering.
2. **Six-step NTT** in `bignum::ntt` — would cut per-huge-mul wall
   1.5–3×.  Significant implementation effort.
3. **DC to_string algorithm audit** — verify D&C engages; if not, fix.
   Cheap.
4. **mlock the NTT scratch** so the OS never pages the hottest data.
   Useful when disk-backing is engaged and RAM is tight.
5. **`sqrt(10005)` cache file** — precompute at the largest precision
   ever used, mmap'd, top-N-bits slice on load.  Cheap; saves the few
   minutes of sqrt at every future run.  Was discussed and deferred:
   the 595be8f overlap already hides sqrt cost on long runs, so cache
   helps small-N runs more than large-N.

## How to analyse the perf log

Two-line config snapshot is on line 2:

```bash
jq -r 'select(.kind=="config")' perf-10b-ec2.jsonl
```

Per-phase wall:

```bash
jq -r 'select(.kind=="phase-end") | "\(.phase): \(.duration_ms/1000)s"' perf-10b-ec2.jsonl
```

FA sub-phase breakdown (paired start/end milestones):

```bash
jq -r 'select(.kind=="milestone")' perf-10b-ec2.jsonl
```

Core utilisation by phase:

```bash
jq -s '[.[] | select(.kind=="sample") |
        {phase, cpu_cores, rss_mb, mmap_count_live, swap_mb,
         io_read_bytes, mmap_bytes_total}] |
        group_by(.phase) |
        map({phase: .[0].phase,
             n: length,
             mean_cores: ([.[].cpu_cores] | add / length),
             max_rss_mb: ([.[].rss_mb] | max),
             max_swap_mb: ([.[].swap_mb] | max),
             max_mmap_live: ([.[].mmap_count_live] | max)})' \
   perf-10b-ec2.jsonl
```

Disk-backing efficiency (cache hit ratio):

```bash
# Final sample's io_read_bytes vs mmap_bytes_total tells us what
# fraction of mmap reads actually hit the SSD.  Anything > 50% is
# concerning — should be much lower if page cache is doing its job.
```

## House rules from CLAUDE.md (do not violate)

* **Don't commit without explicit permission.**  Each commit is
  single-use authorization.  `git add` also counts — don't stage
  without an ask.
* **Run-size ceiling: 20M digits.**  Don't run pi computations over
  20M digits without explicit instruction in the current turn.
  Larger runs tie up the user's hardware for minutes to hours.
* **Performance-knob maintenance contract**: see CLAUDE.md.  Adding
  a knob touches four places.
