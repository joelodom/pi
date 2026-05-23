# Runbook: 10B digits on AL2023

Step-by-step from a freshly-launched Amazon Linux 2023 EC2 instance
to a verified 10-billion-digit run of π. Roughly 16 hours of wall
time end-to-end if you let verification run overnight; ~$13 on
on-demand pricing.

## What you should already have

- An AL2023 instance running, with SSH access as `ec2-user`.
- **Recommended instance: `r7g.4xlarge`** (Graviton ARM, 16 vCPU,
  128 GB RAM, ~$0.85/hr on-demand US-East). The compute peak is
  ~40–50 GB so 128 GB gives comfortable headroom.
- Default root EBS volume (~8 GB) is fine. All the big files go on a
  separate data volume — provisioned in §1.

For instance-launch options other than `r7g.4xlarge`, see the
**Hardware notes** at the end.

## 1. Provision the data EBS volume

You want a separate ~60 GB gp3 volume. The root EBS doesn't have
room for two 10-GB-class files plus a ~4 GB `target/` directory.

> **A note on the `aws` CLI**: AL2023 ships with `aws` CLI v2
> preinstalled, but it needs credentials. Three options for this
> section:
>
> - **Console** (simplest, no setup): use the EC2 web console for
>   create + attach, then SSH in for the format + mount step.
> - **`aws` from your laptop**: assumes you've already run
>   `aws configure` locally.
> - **`aws` from inside the instance**: requires an IAM role with
>   EC2 permissions attached to the instance (`AmazonEC2FullAccess`
>   works, or scope it to `ec2:CreateVolume`, `ec2:AttachVolume`,
>   `ec2:DescribeInstances`, `ec2:DescribeVolumes`). Attach at
>   launch, or later via *EC2 console → Actions → Security → Modify
>   IAM role*.
>
> The CLI commands below work identically from either machine once
> creds are in place.

### Create the volume

In the EC2 console, **Elastic Block Store → Volumes → Create volume**:

- Volume type: **gp3**
- Size: **60 GiB**
- IOPS / throughput: defaults (3000 IOPS / 125 MB/s)
- Availability Zone: **must match your instance's AZ** (look at the
  instance in the console and copy the AZ exactly)
- Encryption: optional

CLI equivalent (run from anywhere you have `aws` configured):

```sh
# Find your instance's AZ:
aws ec2 describe-instances --instance-ids <i-xxxxxxxxxxxxx> \
    --query 'Reservations[].Instances[].Placement.AvailabilityZone' --output text

aws ec2 create-volume \
    --availability-zone <az-from-above> \
    --size 60 \
    --volume-type gp3 \
    --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=pi-data}]' \
    --query VolumeId --output text
```

Save the returned `vol-xxxxxxxxxxxxx` ID.

### Attach the volume to your instance

**Console**: select the new volume, **Actions → Attach volume**.
Pick your instance, set device name to `/dev/sdf`, click Attach.

**CLI**:

```sh
aws ec2 attach-volume \
    --volume-id <vol-xxxxxxxxxxxxx> \
    --instance-id <i-xxxxxxxxxxxxx> \
    --device /dev/sdf
```

The "device name" `/dev/sdf` is just what the AWS control plane uses
as a label — on Nitro instances like r7g.4xlarge the Linux kernel
exposes it as `/dev/nvme1n1` regardless. Pick any free `/dev/sdX`
that doesn't conflict with another attachment.

Attachment takes 5–15 seconds. Check the volume's state in the
console (it should transition `available` → `in-use`) or with:

```sh
aws ec2 describe-volumes --volume-ids <vol-xxxxxxxxxxxxx> \
    --query 'Volumes[].State' --output text
```

### Format and mount on the instance

SSH into the instance and:

```sh
# Confirm the device. nvme0n1 is the root; nvme1n1 should be your
# new 60 GB volume.
lsblk

# Format (XFS handles >10 GB files well and is shipped in AL2023).
sudo mkfs.xfs /dev/nvme1n1

sudo mkdir -p /data
sudo mount /dev/nvme1n1 /data
sudo chown ec2-user:ec2-user /data

# Persist across reboot (optional but tidy).
echo '/dev/nvme1n1 /data xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab
```

ext4 works fine too if you prefer it; the workload is heavy sequential
I/O either way.

## 2. Install build prerequisites

```sh
sudo dnf install -y gcc make m4 diffutils git tmux

# Rust via rustup.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
. "$HOME/.cargo/env"
```

`gcc make m4 diffutils` are what `gmp-mpfr-sys` needs to compile GMP
and MPFR from source. `tmux` is essential — your SSH session will
disconnect during a multi-hour run.

## 3. Clone and build

Put everything on the data volume so the root volume stays free:

```sh
cd /data
git clone <your-pi-repo-url> pi
cd pi

# Force cargo to put its target directory on /data too.
export CARGO_TARGET_DIR=/data/cargo-target
echo 'export CARGO_TARGET_DIR=/data/cargo-target' >> ~/.bashrc

cargo build --release
```

First build is 5–10 minutes (GMP and MPFR compile from source).
After this, the binary lives at `/data/cargo-target/release/pi`.

Smoke check:

```sh
/data/cargo-target/release/pi --digits 100
```

Should print the first 100 digits of π in under a second.

## 4. Compute 10B digits (Chudnovsky)

Use `tmux` so the run survives an SSH disconnect:

```sh
tmux new -s pi
# Inside tmux:
/data/cargo-target/release/pi \
    --digits 10000000000 \
    -o /data/pi-10b-chudnovsky.txt
```

Detach: `Ctrl-b d`. Reattach: `tmux attach -t pi`.

Expected:
- Wall time: ~2–3 hours.
- Peak RSS: ~40–50 GB.
- Output file: ~10 GB at `/data/pi-10b-chudnovsky.txt`.

In a second SSH session you can watch memory and disk:

```sh
watch -n 5 'free -h && df -h /data'
```

If RSS climbs past ~100 GB or the system starts swapping: kill the
run (`Ctrl-c` inside tmux) and either bump to `r7g.8xlarge` (256 GB)
or step the digit count down. AL2023 has no swap by default — an
OOM kill will be quick and clean.

## 5. Verify with BBP (no Gauss-Legendre needed)

This step replaces what would otherwise be a second compute run with
Gauss-Legendre. BBP is fully independent from Chudnovsky:

- Different formula (Bailey-Borwein-Plouffe, hex spigot).
- Different number type (pure-Rust `u64` / `u128`, no GMP in the
  digit-extraction code path).
- Different inner loop.

Any algorithmic bug in Chudnovsky would produce systemically wrong
digits across many positions; random BBP sampling catches systemic
bugs with effectively certain probability after a few thousand
samples.

The only step that does go through GMP is the one-time decimal→hex
conversion of the output file. A multiplication bug in GMP could in
principle corrupt both Chudnovsky's output and the conversion in the
same direction — but GMP is the most-battle-tested
arbitrary-precision library in existence (decades of use in
cryptography, Mathematica, SageMath, y-cruncher), so this is not
where a bug is going to live.

```sh
# Still inside the same tmux session.
/data/cargo-target/release/pi \
    --verify-hex /data/pi-10b-chudnovsky-hex.txt \
    --from-decimal /data/pi-10b-chudnovsky.txt \
    --sanity-samples 100 \
    --samples-per-window 10
```

This runs four phases concurrently after the conversion:

1. **Conversion** (one-time, ~30–60 min): writes
   `/data/pi-10b-chudnovsky-hex.txt` (~8.3 GB of hex). Atomic via
   `.tmp + rename`, so a Ctrl-C mid-conversion never leaves a
   half-written file.
2. **Sanity sweep — first 1M hex digits** (100 BBP calls, fast).
3. **Sanity sweep — middle 100K hex digits** (10 calls, slower).
4. **Sanity sweep — last 10K hex digits** (1 call, slowest — BBP at
   deep positions is ~15–20 min per call).
5. **Random sampling** (unbounded): 10 BBP samples per random 1M-byte
   window, indefinite until Ctrl-C or a mismatch.

How long to let random sampling run? Each window is ~10 samples;
near the start of the file each sample is milliseconds, near the end
each is 15–20 minutes. A reasonable target: **let it run overnight
(~12 hours)** — that accumulates enough deep samples that even a bug
affecting 0.0001% of digits is overwhelmingly likely to be caught.

The verifiable region excludes the last 32 hex digits (~38 decimal
digits) due to a known conversion-boundary precision issue
(`TAIL_SKIP` in `verify_hex.rs`). So the very last ~38 decimal digits
of your output file are not BBP-verified — accept that or extend the
compute to (10B + a small buffer) digits and treat only the first 10B
as the final result.

You can also Ctrl-C and resume later. The hex file persists on
`/data`:

```sh
# Resume sampling later, against the already-converted hex file:
/data/cargo-target/release/pi \
    --verify-hex /data/pi-10b-chudnovsky-hex.txt
```

When you're satisfied, Ctrl-C. The summary line shows total
coverage:

```text
verify-hex: covered 12,345,678 of 8,300,000,000 verifiable hex digits
(0.14876%) across 1234 disjoint intervals
```

## 6. Wind down

Choose one:

- **Keep the files between sessions**: take an EBS snapshot of the
  data volume:
  ```sh
  aws ec2 create-snapshot --volume-id <vol-id> \
      --description "pi 10B verified $(date +%Y-%m-%d)"
  ```
  Snapshot storage: ~$0.05/GB-month for actual used space (~30 GB =
  $1.50/month). Then stop or terminate the instance.

- **Throw it all away**: terminate the instance; if the data volume
  isn't set to delete-on-terminate, detach and delete it manually.

Stopping (not terminating) the instance keeps the EBS volumes alive
and detached; you can re-attach to a new instance later. Stopped
instances pay only EBS storage (~$0.08/GB-month for gp3).

## Cost summary

On-demand US-East pricing for `r7g.4xlarge` at $0.8467/hr:

| Phase                          | Duration | Cost     |
|--------------------------------|----------|----------|
| Build                          | ~10 min  | $0.14    |
| Chudnovsky 10B                 | ~2–3 hr  | $1.70–2.55 |
| Decimal→hex conversion         | ~1 hr    | $0.85    |
| Sanity sweep                   | ~30 min  | $0.42    |
| Random sampling (overnight)    | ~12 hr   | $10.16   |
| **Total**                      | ~16 hr   | **~$13.30** |

Drop the overnight sampling and you're at ~$4 for a one-day run with
sanity-only verification.

EBS storage is negligible (~$0.50 for one day of a 60 GB gp3 volume).

## Hardware notes

| Instance        | vCPU | RAM    | $/hr (OD US-E) | When to pick                          |
|-----------------|------|--------|----------------|---------------------------------------|
| r7g.2xlarge     | 8    | 64 GB  | $0.42          | Risky for 10B; fine for ≤5B           |
| **r7g.4xlarge** | 16   | 128 GB | $0.85          | **Recommended for 10B**               |
| r7g.8xlarge     | 32   | 256 GB | $1.69          | Headroom for 20–30B without changes   |
| r7i.4xlarge     | 16   | 128 GB | $1.06          | If you specifically need x86          |

`r7g` (Graviton) is ~20% cheaper than `r7i` (Intel) for equivalent
performance on this workload. GMP has well-tuned aarch64 paths.

**Spot pricing**: r7g.4xlarge spot is typically $0.20–0.30/hr. **Do
not use spot for the compute phase** — the output is materialized
fully in memory and only written at the end, so a spot termination
loses the entire run. Spot is fine for the random sampling phase
(the hex file is durable; just resume after a termination).

## Recovering from a failed run

- **OOM during compute**: bump to `r7g.8xlarge` or reduce digit count.
- **Disk full during compute**: the output file is written atomically
  via `WriterSink` buffering; if the disk fills mid-write, you lose
  the run. Verify free space on `/data` before launching
  (`df -h /data`).
- **Disk full during conversion**: the conversion writes
  `pi-10b-chudnovsky-hex.txt.tmp` first and renames on success. A
  partial `.tmp` is safe to delete and retry.
- **SSH disconnect during compute**: harmless if you used tmux.
  Reattach with `tmux attach -t pi`.
- **Random sampling caught a mismatch**: the program exits non-zero
  and prints the position. This is a real finding — investigate
  before trusting the file.
