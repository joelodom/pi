# Notes for Claude

Short, focused — extending this costs tokens on every turn.

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
