# Notes for Claude

Short, focused — extending this costs tokens on every turn.

If `HANDOFF-10B.md` exists at repo root, read it first — it has
session-specific context (current experiment, recent commits, known
bottlenecks) that supersedes the defaults below.  Delete it once its
experiment is concluded.

## Performance-knob maintenance contract

Knobs live in two places: per-crate `Config` structs (in
`bignum::config` and `pi_core::config`) backed by atomics, and the
generator that emits a config for a (target digits, host) pair.

**When you add a new performance knob, update *all four* of these in
the same change:**

1. The crate's `Config` struct + atomic + getter + `apply` branch +
   the `Config::current()` snapshot constructor.
2. The `Section` struct + parsing in `crates/pi-cli/src/main.rs`
   `load_and_apply_config`.
3. The generator in `crates/pi-cli/src/config_gen.rs` — pick a value
   based on `HardwareProfile` + `digits` and emit it with a comment
   stating *why* that value was chosen.  This is the source of truth
   for "what the program will use by default."
4. The perf snapshot in `crates/pi-core/src/perf.rs`
   `write_config_snapshot` — add the field to the `config` JSONL
   event so post-hoc analysis records what was active for the run.

Skipping step 3 means runs without `--config` silently get the
default for that knob instead of a hardware-tuned value.  Skipping
step 4 means perf logs lose the value, and reproducing a run from
its JSONL becomes harder.

## Config flow at runtime

There is no checked-in example TOML.  The program does this in
`main`:

* `--generate-config <DIGITS>` → run the generator for the host, print
  the TOML, exit.  No computation.
* `--config <FILE>` → load that file, apply it, compute.
* (neither) → run the generator for the host with `cli.digits`, parse
  its TOML output back through the same loader, apply, compute.  This
  is the default path; users typically never pass `--config`.

Whatever ends up applied is captured in the perf JSONL `config`
event right after `run-start`, so a run can be reproduced or
analyzed without the original TOML file.

## Disk-backed limb storage (`bignum::storage`)

`Integer.limbs` is a `LimbStorage` enum: `Memory(Vec<u64>)` for the
common path and `Mapped { mmap, file, ... }` for buffers `≥
disk_limb_threshold` u64s.  Mapped storage uses a temp file in
`std::env::temp_dir()` (overridable via `bignum.scratch_dir` config),
unlinked on drop.  The OS page cache handles RAM/SSD eviction
transparently; sequential access patterns (Karatsuba, pack/unpack)
stay near RAM speed, random access (NTT bit_reverse on a disk-backed
buffer) thrashes — don't lower the threshold low enough to push NTT
*scratch* to disk.

Atomic counters expose live and cumulative mmap bytes/counts; they
appear in every perf JSONL `sample` event as `mmap_bytes_live`,
`mmap_count_live`, `mmap_bytes_total`, `mmap_count_total`.

Default is `usize::MAX` (disabled).  The generator picks a non-trivial
threshold only when `estimated_peak_rss > 70% of RAM`.

## Don't commit without explicit permission

Each commit is single-use authorization.  The user wants to test
changes before they're committed.  Even after a recent "commit", the
next change needs its own ask.  `git add` counts — don't stage
without an ask either.

## Run-size ceiling

Don't run pi computations over 20M digits without explicit
instruction in the current turn.  Larger runs tie up the user's
hardware for minutes to hours.  Benchmark in the 100K–20M range;
extrapolate larger sizes.
