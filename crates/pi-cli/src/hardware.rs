//! Best-effort host hardware detection for the
//! `--generate-config` code path.  All detection failures degrade
//! silently to sensible fallbacks (logical-cores-only, RAM=0), so the
//! generator can always produce *some* output even on platforms we
//! don't recognise.

use std::ffi::CString;

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    /// Logical cores (one per SMT thread on hyper-threaded systems).
    pub logical_cores: usize,
    /// Physical cores.  Equal to `logical_cores` when we can't detect
    /// them separately.
    pub physical_cores: usize,
    /// Total system RAM in bytes.  `0` on detection failure.
    pub ram_bytes: u64,
    pub os: &'static str,
    pub arch: &'static str,
}

pub fn detect() -> HardwareProfile {
    HardwareProfile {
        logical_cores: num_logical_cores(),
        physical_cores: num_physical_cores(),
        ram_bytes: ram_bytes(),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
    }
}

/// Rayon's view of the available threads — its global pool defaults
/// to one thread per logical core unless `RAYON_NUM_THREADS` overrides.
/// Using rayon's value rather than a raw `num_cpus` call avoids
/// surprises when the user is running under a thread-count override.
fn num_logical_cores() -> usize {
    rayon::current_num_threads()
}

#[cfg(target_os = "macos")]
fn num_physical_cores() -> usize {
    sysctl_u32("hw.physicalcpu")
        .map(|v| v as usize)
        .unwrap_or_else(num_logical_cores)
}

#[cfg(target_os = "linux")]
fn num_physical_cores() -> usize {
    use std::collections::HashSet;
    let s = match std::fs::read_to_string("/proc/cpuinfo") {
        Ok(s) => s,
        Err(_) => return num_logical_cores(),
    };
    // Count distinct (physical_id, core_id) pairs.  Falls back to
    // logical-core count if the file doesn't expose those fields
    // (e.g. some container environments).
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut pid: Option<String> = None;
    let mut cid: Option<String> = None;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("physical id") {
            pid = v.split(':').nth(1).map(|s| s.trim().to_string());
        } else if let Some(v) = line.strip_prefix("core id") {
            cid = v.split(':').nth(1).map(|s| s.trim().to_string());
        } else if line.is_empty() {
            if let (Some(p), Some(c)) = (pid.take(), cid.take()) {
                seen.insert((p, c));
            }
        }
    }
    if seen.is_empty() {
        num_logical_cores()
    } else {
        seen.len()
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn num_physical_cores() -> usize {
    num_logical_cores()
}

#[cfg(target_os = "macos")]
fn ram_bytes() -> u64 {
    sysctl_u64("hw.memsize").unwrap_or(0)
}

#[cfg(target_os = "linux")]
fn ram_bytes() -> u64 {
    if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if let Some(n) = parts.first() {
                    if let Ok(kib) = n.parse::<u64>() {
                        return kib.saturating_mul(1024);
                    }
                }
            }
        }
    }
    0
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn ram_bytes() -> u64 {
    0
}

// ---- macOS sysctl helpers --------------------------------------------

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let mut val: u64 = 0;
    let mut size = std::mem::size_of_val(&val);
    let cstr = CString::new(name).ok()?;
    // SAFETY: cstr lives through the call; val is a valid out-pointer
    // of the size `size` bytes; sysctlbyname is documented to write at
    // most `size` bytes and update `size` to the actual length.
    let ret = unsafe {
        libc::sysctlbyname(
            cstr.as_ptr(),
            &mut val as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 {
        Some(val)
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn sysctl_u32(name: &str) -> Option<u32> {
    let mut val: u32 = 0;
    let mut size = std::mem::size_of_val(&val);
    let cstr = CString::new(name).ok()?;
    // SAFETY: same as `sysctl_u64`.
    let ret = unsafe {
        libc::sysctlbyname(
            cstr.as_ptr(),
            &mut val as *mut _ as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 {
        Some(val)
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn sysctl_u64(_: &str) -> Option<u64> {
    None
}

#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
fn sysctl_u32(_: &str) -> Option<u32> {
    None
}
