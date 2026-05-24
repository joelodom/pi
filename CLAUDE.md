# Notes for Claude

Short, focused — extending this costs tokens on every turn.

## Performance-knob maintenance contract

The codebase has two places knobs live: per-crate `Config` structs (in
`bignum::config` and `pi_core::config`) backed by atomics, and the
TOML files that ship as user-facing examples and generator output.

**When you add a new performance knob, update *all* of these in the
same change:**

1. The crate's `Config` struct + atomic + getter + `apply` branch.
2. The `Section` struct + parsing in `crates/pi-cli/src/main.rs`
   `load_and_apply_config`.
3. All three example configs under `config/` (`laptop.toml`,
   `server-128gb.toml`, `server-massive.toml`) with a comment
   explaining the knob and what to change for different hardware.
4. The generator in `crates/pi-cli/src/config_gen.rs` — pick a value
   based on `HardwareProfile` + `digits`, and emit it with a comment
   stating *why* that value was chosen for this hardware.

Skipping step 4 means `pi --generate-config` silently drops the new
knob and users get a config with default behavior they didn't ask for.

The generator is the *source of truth* for what we think is optimal
per-hardware. The example TOML files are explanatory snapshots; they
should agree with what the generator would emit for representative
hardware:

| TOML | should match generator output for |
|------|------------------------------------|
| `laptop.toml` | ~11 cores, ~18 GB RAM, current default target |
| `server-128gb.toml` | ~32 cores, ~128 GB RAM, a billion-digit target |
| `server-massive.toml` | ~128 cores, ~512 GB RAM, a billion-digit target |

If the generator's heuristics change, update the example TOMLs too.

## Don't commit without explicit permission

Each commit is single-use authorization. The user wants to test
changes before they're committed. Even if a recent "commit" landed
in this session, the next change needs its own ask. Stage staging
(`git add`) counts as preparing to commit — don't do it without an
ask either.

## Run-size ceiling

Don't run pi computations over 20M digits without explicit instruction
in the current turn. Larger runs tie up the user's hardware for minutes
to hours. Benchmark in the 100K–20M range; extrapolate larger sizes.
