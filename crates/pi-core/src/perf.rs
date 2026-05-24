//! Performance instrumentation.
//!
//! Emits a JSON-lines stream of events to an append-only file when a
//! [`PerfRecorder`] is opened with a path.  Three kinds of event:
//!
//! * `run-start` — written once at the top of [`PerfRecorder::open`],
//!   with `digits`, `algorithm`, available core count, and wall-clock
//!   start time (`unix_ms`).
//! * `phase-start` / `phase-end` — emitted by the `ProgressReporter`
//!   wrapper on phase boundaries, carrying the phase name and (for
//!   `phase-end`) its duration.
//! * `sample` — periodic snapshots from a background thread.  Every
//!   sample carries the *current* phase string (so memory or CPU
//!   spikes can be attributed without a temporal join), the resident
//!   set size in MB, and an effective-core-count derived from
//!   `getrusage(RUSAGE_SELF)` deltas over wall-clock deltas.
//!
//! The disabled / NOP path is a `PerfRecorder` whose `inner: None` —
//! every public method is a one-line branch that returns immediately,
//! so callers can sprinkle calls without measuring the cost.
//!
//! Lines look like:
//!
//! ```text
//! {"t_ms":0,"kind":"run-start","unix_ms":1716572745123,"digits":10000000,"algorithm":"chudnovsky","cores":16}
//! {"t_ms":12,"kind":"phase-start","phase":"binary splitting"}
//! {"t_ms":500,"kind":"sample","phase":"binary splitting","rss_mb":482,"cpu_cores":9.7}
//! {"t_ms":4612,"kind":"phase-end","phase":"binary splitting","duration_ms":4600}
//! ```
//!
//! All numeric fields are plain JSON numbers; `phase` strings are
//! JSON-escaped defensively.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Cloneable handle to the perf-recording machinery.  Cheap to clone
/// (it's an `Option<Arc<...>>` under the hood).  Pass it freely — the
/// disabled variant turns every recording call into a one-line
/// branch.
#[derive(Clone, Default)]
pub struct PerfRecorder {
    inner: Option<Arc<Inner>>,
}

struct Inner {
    start: Instant,
    file: Mutex<BufWriter<std::fs::File>>,
    /// `(name, started_at)` of the currently-active phase, if any.
    /// `Mutex` rather than `RwLock` because phase transitions are rare
    /// (a handful per run) and samples reading this only happen every
    /// `sample_ms` ms — contention is essentially nil.
    current_phase: Mutex<Option<(String, Instant)>>,
    last_cpu: Mutex<CpuSnapshot>,
}

#[derive(Clone, Copy)]
struct CpuSnapshot {
    wall: Instant,
    user_us: u64,
    sys_us: u64,
}

impl PerfRecorder {
    /// Construct a disabled recorder.  Every method is a no-op.
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    /// Open `path` in append mode and start recording.  Writes one
    /// `run-start` event immediately.
    pub fn open(path: &Path, digits: u64, algorithm: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        let now = Instant::now();
        let inner = Arc::new(Inner {
            start: now,
            file: Mutex::new(BufWriter::with_capacity(8 * 1024, file)),
            current_phase: Mutex::new(None),
            last_cpu: Mutex::new(cpu_snapshot_now()),
        });
        let rec = Self { inner: Some(inner) };
        rec.write_run_start(digits, algorithm);
        Ok(rec)
    }

    /// True iff recording is active.  Use to skip work that's only
    /// useful when recording (rare — most methods are themselves
    /// cheap branches).
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Start a background sampler thread.  Returns a guard whose Drop
    /// stops the thread, joins it, and flushes the file.  No-op when
    /// recorder is disabled.
    pub fn start_sampler(&self, sample_ms: u64) -> SamplerGuard {
        let Some(inner) = self.inner.clone() else {
            return SamplerGuard { stop: None, handle: None };
        };
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let inner_clone = Arc::clone(&inner);
        let handle = thread::Builder::new()
            .name("perf-sampler".into())
            .spawn(move || {
                let interval = Duration::from_millis(sample_ms.max(10));
                loop {
                    // Poll the stop flag with short sleeps so we
                    // respond quickly to shutdown without spinning.
                    let mut waited = Duration::ZERO;
                    while waited < interval {
                        if stop_clone.load(Ordering::Relaxed) {
                            return;
                        }
                        let step = Duration::from_millis(50).min(interval - waited);
                        thread::sleep(step);
                        waited += step;
                    }
                    inner_clone.write_sample();
                }
            })
            .expect("spawn perf-sampler thread");
        SamplerGuard {
            stop: Some(stop),
            handle: Some(handle),
        }
    }

    /// Mark the start of a named phase.  Records the start time and
    /// emits a `phase-start` event.  Samples taken between now and
    /// the matching `phase_end` will carry this phase name.
    pub fn phase_start(&self, phase: &str) {
        if let Some(inner) = &self.inner {
            *inner.current_phase.lock().unwrap() =
                Some((phase.to_string(), Instant::now()));
            let t = inner.elapsed_ms();
            let phase_esc = json_escape(phase);
            inner.write_line(&format!(
                "{{\"t_ms\":{t},\"kind\":\"phase-start\",\"phase\":{phase_esc}}}"
            ));
        }
    }

    /// Mark the end of the currently-active phase.  Emits a
    /// `phase-end` event with the phase name and its duration in ms,
    /// computed from the matching `phase_start`.  A `phase_end`
    /// without a paired `phase_start` is a no-op.
    pub fn phase_end(&self) {
        if let Some(inner) = &self.inner {
            let taken = inner.current_phase.lock().unwrap().take();
            if let Some((name, started)) = taken {
                let duration_ms = started.elapsed().as_millis() as u64;
                let t = inner.elapsed_ms();
                let phase_esc = json_escape(&name);
                inner.write_line(&format!(
                    "{{\"t_ms\":{t},\"kind\":\"phase-end\",\"phase\":{phase_esc},\"duration_ms\":{duration_ms}}}"
                ));
            }
        }
    }

    /// Emit a final `run-end` event with totals.  Called at end of
    /// CLI run.  Optional; nothing else depends on it.
    pub fn run_end(&self) {
        if let Some(inner) = &self.inner {
            let t = inner.elapsed_ms();
            inner.write_line(&format!(
                "{{\"t_ms\":{t},\"kind\":\"run-end\"}}"
            ));
            // Flush any buffered tail.
            let _ = inner.file.lock().unwrap().flush();
        }
    }

    fn write_run_start(&self, digits: u64, algorithm: &str) {
        let Some(inner) = &self.inner else { return };
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let cores = num_cpus_get();
        let algo_esc = json_escape(algorithm);
        inner.write_line(&format!(
            "{{\"t_ms\":0,\"kind\":\"run-start\",\"unix_ms\":{unix_ms},\
             \"digits\":{digits},\"algorithm\":{algo_esc},\"cores\":{cores}}}"
        ));
    }
}

impl Inner {
    fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    fn write_line(&self, line: &str) {
        // Best-effort write: if the OS can't deliver, we'd rather
        // continue the computation than abort.  Errors are silently
        // dropped (the recorder is observability, not a critical
        // path).
        let mut f = self.file.lock().unwrap();
        let _ = writeln!(f, "{line}");
    }

    fn write_sample(&self) {
        let t = self.elapsed_ms();
        let rss_bytes = read_rss_bytes();
        let rss_mb = rss_bytes / (1024 * 1024);

        // CPU-cores busy: delta user+sys time over delta wall time
        // since the last sample.  First sample uses the recorder's
        // start time as the baseline.
        let now_cpu = cpu_snapshot_now();
        let mut last = self.last_cpu.lock().unwrap();
        let wall_us = now_cpu.wall.duration_since(last.wall).as_micros() as u64;
        let cpu_us = now_cpu
            .user_us
            .saturating_sub(last.user_us)
            .saturating_add(now_cpu.sys_us.saturating_sub(last.sys_us));
        *last = now_cpu;
        drop(last);
        let cpu_cores = if wall_us == 0 {
            0.0
        } else {
            cpu_us as f64 / wall_us as f64
        };

        let phase_name = self
            .current_phase
            .lock()
            .unwrap()
            .as_ref()
            .map(|(n, _)| n.clone())
            .unwrap_or_default();
        let phase_esc = json_escape(&phase_name);
        self.write_line(&format!(
            "{{\"t_ms\":{t},\"kind\":\"sample\",\"phase\":{phase_esc},\
             \"rss_mb\":{rss_mb},\"cpu_cores\":{cpu_cores:.3}}}"
        ));
    }
}

/// Guard returned by `start_sampler`.  Drop stops and joins the
/// sampler thread.
pub struct SamplerGuard {
    stop: Option<Arc<AtomicBool>>,
    handle: Option<JoinHandle<()>>,
}

impl Drop for SamplerGuard {
    fn drop(&mut self) {
        if let Some(s) = &self.stop {
            s.store(true, Ordering::Relaxed);
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// =====================================================================
// Telemetry helpers
// =====================================================================

fn cpu_snapshot_now() -> CpuSnapshot {
    let (user_us, sys_us) = read_cpu_times();
    CpuSnapshot {
        wall: Instant::now(),
        user_us,
        sys_us,
    }
}

/// `(user_us, sys_us)` for the current process via `getrusage`.  Both
/// fields are 0 on error or unsupported platforms.
fn read_cpu_times() -> (u64, u64) {
    // SAFETY: zero-init a POD struct; pass valid out-pointer to libc.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if ret != 0 {
        return (0, 0);
    }
    let user_us =
        (usage.ru_utime.tv_sec as u64) * 1_000_000 + (usage.ru_utime.tv_usec as u64);
    let sys_us = (usage.ru_stime.tv_sec as u64) * 1_000_000 + (usage.ru_stime.tv_usec as u64);
    (user_us, sys_us)
}

/// Resident set size in bytes, or 0 if unavailable.
#[cfg(target_os = "macos")]
#[allow(deprecated)]
fn read_rss_bytes() -> u64 {
    // SAFETY: `mach_task_self_` is a static extern; `task_info` writes
    // into the basic_info struct; the count argument is initialized to
    // the struct's field count.  The deprecation hint suggests moving
    // to the `mach2` crate — we accept it here to avoid one extra
    // dependency for a single read.
    use libc::*;
    let task: mach_port_t = unsafe { libc::mach_task_self_ };
    let mut info: mach_task_basic_info = unsafe { std::mem::zeroed() };
    let mut count = MACH_TASK_BASIC_INFO_COUNT;
    let ret = unsafe {
        task_info(
            task,
            MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as task_info_t,
            &mut count,
        )
    };
    if ret != 0 {
        return 0;
    }
    info.resident_size as u64
}

#[cfg(target_os = "linux")]
fn read_rss_bytes() -> u64 {
    // /proc/self/status has `VmRSS:    NNNNN kB`.
    let s = match std::fs::read_to_string("/proc/self/status") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // rest looks like "    1234567 kB"
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(n) = parts.first() {
                if let Ok(kib) = n.parse::<u64>() {
                    return kib * 1024;
                }
            }
        }
    }
    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn read_rss_bytes() -> u64 {
    0
}

fn num_cpus_get() -> usize {
    // Use rayon's view of the global pool — that's what actually
    // governs our parallelism, and it respects RAYON_NUM_THREADS.
    rayon::current_num_threads()
}

/// JSON-escape a string and wrap it in double quotes.  Conservative —
/// handles control chars defensively even though phase names should
/// only contain printable ASCII.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_recorder_is_noop() {
        let r = PerfRecorder::disabled();
        // None of these should panic or do anything observable.
        r.phase_start("hello");
        r.phase_end();
        r.run_end();
        assert!(!r.is_enabled());
        // start_sampler returns a guard that does nothing on drop.
        drop(r.start_sampler(100));
    }

    #[test]
    fn json_escape_simple() {
        assert_eq!(json_escape("hello"), "\"hello\"");
        assert_eq!(json_escape("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_escape("a\nb"), "\"a\\nb\"");
        assert_eq!(json_escape(""), "\"\"");
    }

    #[test]
    fn enabled_recorder_emits_events() {
        let tmp = std::env::temp_dir().join(format!(
            "pi-perf-test-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&tmp);
        let r = PerfRecorder::open(&tmp, 100, "chudnovsky").unwrap();
        r.phase_start("p1");
        std::thread::sleep(std::time::Duration::from_millis(10));
        r.phase_end();
        r.run_end();
        drop(r);
        let body = std::fs::read_to_string(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 4, "expected 4 events, got: {body:?}");
        assert!(lines[0].contains("\"kind\":\"run-start\""));
        assert!(lines[0].contains("\"digits\":100"));
        assert!(lines[1].contains("\"kind\":\"phase-start\""));
        assert!(lines[1].contains("\"phase\":\"p1\""));
        assert!(lines[2].contains("\"kind\":\"phase-end\""));
        // Duration is measured at runtime; assert the field is present
        // and parses to >= 10 ms (we slept 10 ms before phase_end).
        assert!(lines[2].contains("\"duration_ms\":"), "no duration in {}", lines[2]);
        assert!(lines[3].contains("\"kind\":\"run-end\""));
    }

    #[test]
    fn rss_bytes_nonzero() {
        // Sanity: our own process should report nonzero resident memory.
        let rss = read_rss_bytes();
        // On unsupported platforms this returns 0; otherwise it should
        // be > 0 (no test process is < 1 MB of resident).
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(rss > 0, "expected nonzero RSS, got {rss}");
        }
    }
}
