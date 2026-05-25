//! Limb-buffer storage backend with a memory ↔ mmap-backed split.
//!
//! Replaces the `Vec<u64>` that `Integer` historically used for its
//! magnitude.  Below a configurable size threshold an `Integer` still
//! holds its limbs in a `Vec<u64>` (identical performance and
//! behaviour to before).  Above the threshold it allocates a temp
//! file in the scratch directory and `mmap`s it for `&mut [u64]`
//! access.  The temp file is unlinked when the storage drops.
//!
//! The threshold and scratch directory are read from
//! [`crate::config`].  Default threshold is `usize::MAX` (disabled),
//! so without explicit configuration the program behaves exactly as
//! before.  When the threshold is engaged, the OS page cache handles
//! eviction between RAM and SSD — sequential access patterns are
//! near-RAM speed; random access patterns (e.g. NTT bit-reverse on a
//! disk-backed buffer) thrash and should be avoided by pre-pinning
//! NTT scratch in `Vec<u64>` storage.
//!
//! Process-wide atomic counters expose the live and cumulative
//! mmap'd bytes / counts so the perf JSONL recorder can correlate
//! disk usage with phase-level slowdowns.

use std::fs::OpenOptions;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use memmap2::{Advice, MmapMut, MmapOptions};

/// Limb buffer.  Either a heap `Vec<u64>` or a mmap'd file viewed as
/// a `[u64]` slice.
pub enum LimbStorage {
    Memory(Vec<u64>),
    /// File-backed.  `len` is the *logical* element count (≤ the
    /// underlying mmap's element capacity); allows cheap truncation
    /// without resizing the file.
    Mapped {
        mmap: MmapMut,
        path: PathBuf,
        len: usize,
        capacity: usize,
    },
}

impl LimbStorage {
    /// Empty storage.  Always memory-backed (no allocation).
    pub fn new() -> Self {
        Self::Memory(Vec::new())
    }

    /// Allocate `n` zeroed `u64` elements.  Routes to mmap when
    /// `n >= disk_limb_threshold()`.
    pub fn allocate_zeroed(n: usize) -> Self {
        if n == 0 {
            return Self::new();
        }
        if n >= crate::config::disk_limb_threshold() {
            Self::allocate_mapped(n)
        } else {
            Self::Memory(vec![0u64; n])
        }
    }

    /// Wrap an existing `Vec<u64>` as storage.  If the Vec is large
    /// enough that the threshold says it should be on disk, migrate
    /// it: allocate mmap of the right size, copy the data, drop the
    /// Vec.  Memory peaks briefly during the copy.
    pub fn from_vec(v: Vec<u64>) -> Self {
        if v.len() >= crate::config::disk_limb_threshold() {
            let mut storage = Self::allocate_mapped(v.len());
            storage.as_mut_slice().copy_from_slice(&v);
            storage
        } else {
            Self::Memory(v)
        }
    }

    fn allocate_mapped(n: usize) -> Self {
        let dir = scratch_dir();
        if let Err(e) = std::fs::create_dir_all(&dir) {
            panic!(
                "scratch dir `{}` could not be created: {e}",
                dir.display()
            );
        }
        let id = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = dir.join(format!("bignum-{pid}-{id}.dat"));
        let byte_size = (n as u64) * 8;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap_or_else(|e| {
                panic!("opening scratch file `{}`: {e}", path.display())
            });
        file.set_len(byte_size).unwrap_or_else(|e| {
            panic!(
                "set_len({byte_size}) on scratch file `{}`: {e}",
                path.display()
            )
        });
        // SAFETY: file is freshly created and exclusively owned by us;
        // no other process can mutate it while the mmap is live.
        let mmap = unsafe {
            MmapOptions::new()
                .len(byte_size as usize)
                .map_mut(&file)
                .unwrap_or_else(|e| {
                    panic!("mmap of `{}`: {e}", path.display())
                })
        };
        // Hint to the kernel that we'll touch this region sequentially.
        // For our access patterns (Karatsuba sub-mults, NTT pack/unpack,
        // shift/add/sub) that's nearly always true, and the hint lets
        // the kernel prefetch ahead and release pages we've passed.
        // Without this, every first-access page is a "major" fault on
        // macOS (file-backed mmaps don't get the zero-fill fast path).
        // Failure of advise is non-fatal — drop the result.
        let _ = mmap.advise(Advice::Sequential);
        // Drop the File handle immediately.  On macOS and Linux,
        // mmap'd memory keeps the underlying inode alive even after
        // the descriptor is closed — the kernel holds its own
        // reference — so we don't need to keep the fd around just to
        // preserve the mapping.  Dropping early is essential: pi at
        // multi-million-digit scale produces thousands of intermediate
        // disk-backed Integers, and we would otherwise blow the
        // per-process file-descriptor limit (256 on default macOS).
        drop(file);
        // Zero is the default for new file content (set_len extends
        // with zeros), so we don't need an explicit memset.
        MMAP_BYTES_LIVE.fetch_add(byte_size, Ordering::Relaxed);
        MMAP_COUNT_LIVE.fetch_add(1, Ordering::Relaxed);
        MMAP_BYTES_TOTAL_ALLOCATED.fetch_add(byte_size, Ordering::Relaxed);
        MMAP_COUNT_TOTAL_ALLOCATED.fetch_add(1, Ordering::Relaxed);
        Self::Mapped {
            mmap,
            path,
            len: n,
            capacity: n,
        }
    }

    /// Logical element count (number of `u64`s currently held).
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Memory(v) => v.len(),
            Self::Mapped { len, .. } => *len,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` if this storage is mmap'd (disk-backed).
    #[inline]
    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped { .. })
    }

    /// Truncate to `new_len`, releasing any tail elements.  For
    /// memory-backed storage this is `Vec::truncate`.  For mmap'd
    /// storage we just lower the logical length — the file size
    /// stays the same until drop.
    pub fn truncate(&mut self, new_len: usize) {
        match self {
            Self::Memory(v) => v.truncate(new_len),
            Self::Mapped { len, .. } => {
                if new_len < *len {
                    *len = new_len;
                }
            }
        }
    }

    /// Pop the last element.  For mmap'd storage, fetches the value
    /// then decrements the logical length.
    pub fn pop(&mut self) -> Option<u64> {
        match self {
            Self::Memory(v) => v.pop(),
            Self::Mapped { mmap, len, .. } => {
                if *len == 0 {
                    return None;
                }
                let bytes = &mmap[..];
                // SAFETY: byte slice is u64-aligned (mmap is page-
                // aligned) and `capacity` is exact.
                let elems: &[u64] = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const u64,
                        bytes.len() / 8,
                    )
                };
                let v = elems[*len - 1];
                *len -= 1;
                Some(v)
            }
        }
    }

    /// Push an element.  Memory-backed only — mmap'd storage panics
    /// (callers should pre-size).  This kept simple deliberately:
    /// growing a mmap means truncating the file and remapping, which
    /// is rare enough in our workload that we'd rather hit a panic
    /// and fix the call site to pre-allocate.
    pub fn push(&mut self, v: u64) {
        match self {
            Self::Memory(vec) => vec.push(v),
            Self::Mapped { .. } => {
                panic!(
                    "LimbStorage::push on mmap-backed storage \
                     (callers must pre-allocate via allocate_zeroed)"
                )
            }
        }
    }

    /// Clear to length 0.  Storage type is preserved.
    pub fn clear(&mut self) {
        match self {
            Self::Memory(v) => v.clear(),
            Self::Mapped { len, .. } => *len = 0,
        }
    }

    /// Borrow as `&[u64]`.
    #[inline]
    pub fn as_slice(&self) -> &[u64] {
        match self {
            Self::Memory(v) => v.as_slice(),
            Self::Mapped { mmap, len, .. } => {
                let bytes = &mmap[..];
                // SAFETY: see `pop`.
                unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const u64,
                        *len,
                    )
                }
            }
        }
    }

    /// Borrow as `&mut [u64]`.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        match self {
            Self::Memory(v) => v.as_mut_slice(),
            Self::Mapped { mmap, len, .. } => {
                let bytes = &mut mmap[..];
                // SAFETY: see `pop`.
                unsafe {
                    std::slice::from_raw_parts_mut(
                        bytes.as_mut_ptr() as *mut u64,
                        *len,
                    )
                }
            }
        }
    }
}

impl Default for LimbStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for LimbStorage {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for LimbStorage {}

impl Clone for LimbStorage {
    /// Clones always produce a memory-backed copy — disk-backing is
    /// chosen at allocation time based on size, and a clone of a
    /// medium-large Mapped storage might prefer memory.  We just
    /// route through `from_vec` which redecides.
    fn clone(&self) -> Self {
        Self::from_vec(self.as_slice().to_vec())
    }
}

impl Drop for LimbStorage {
    fn drop(&mut self) {
        if let Self::Mapped { mmap, path, capacity, .. } = self {
            let bytes = (*capacity as u64) * 8;
            MMAP_BYTES_LIVE.fetch_sub(bytes, Ordering::Relaxed);
            MMAP_COUNT_LIVE.fetch_sub(1, Ordering::Relaxed);
            // The mmap drops on return (its inode reference is the
            // last thing keeping the file alive after we close it).
            // `path` is unlinked so the on-disk bytes are reclaimed.
            let _ = mmap; // mmap drops at end-of-scope; explicit no-op
            let _ = std::fs::remove_file(&path);
        }
    }
}

impl Deref for LimbStorage {
    type Target = [u64];
    #[inline]
    fn deref(&self) -> &[u64] {
        self.as_slice()
    }
}

impl DerefMut for LimbStorage {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u64] {
        self.as_mut_slice()
    }
}

impl From<Vec<u64>> for LimbStorage {
    fn from(v: Vec<u64>) -> Self {
        Self::from_vec(v)
    }
}

impl std::fmt::Debug for LimbStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tag = match self {
            Self::Memory(_) => "Memory",
            Self::Mapped { .. } => "Mapped",
        };
        write!(f, "LimbStorage::{tag}(len={})", self.len())
    }
}

// =====================================================================
// Process-wide counters + scratch-dir override
// =====================================================================

static MMAP_BYTES_LIVE: AtomicU64 = AtomicU64::new(0);
static MMAP_COUNT_LIVE: AtomicUsize = AtomicUsize::new(0);
static MMAP_BYTES_TOTAL_ALLOCATED: AtomicU64 = AtomicU64::new(0);
static MMAP_COUNT_TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(1);

static CUSTOM_SCRATCH_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Override the scratch directory used by `Mapped` storage.  Call
/// once at startup before any large allocations.  Subsequent
/// allocations will create files under `path`.
pub fn set_scratch_dir(path: PathBuf) {
    *CUSTOM_SCRATCH_DIR.lock().unwrap() = Some(path);
}

/// Current scratch directory.  Defaults to `std::env::temp_dir()`.
fn scratch_dir() -> PathBuf {
    CUSTOM_SCRATCH_DIR
        .lock()
        .unwrap()
        .clone()
        .unwrap_or_else(std::env::temp_dir)
}

/// Public view of the currently-configured scratch directory.  Returns
/// the override set via [`set_scratch_dir`] if one is active, otherwise
/// `std::env::temp_dir()`.  Used by the CLI to surface the value in
/// the startup banner.
pub fn current_scratch_dir() -> PathBuf {
    scratch_dir()
}
// ---- Public counter accessors (for perf logging) --------------------

#[inline]
pub fn mmap_bytes_live() -> u64 {
    MMAP_BYTES_LIVE.load(Ordering::Relaxed)
}

#[inline]
pub fn mmap_count_live() -> usize {
    MMAP_COUNT_LIVE.load(Ordering::Relaxed)
}

#[inline]
pub fn mmap_bytes_total_allocated() -> u64 {
    MMAP_BYTES_TOTAL_ALLOCATED.load(Ordering::Relaxed)
}

#[inline]
pub fn mmap_count_total_allocated() -> usize {
    MMAP_COUNT_TOTAL_ALLOCATED.load(Ordering::Relaxed)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate `DISK_LIMB_THRESHOLD` via
    // `config::apply`.  Without this, parallel test threads stomp on
    // each other's global state.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    // NOTE: storage tests are #[ignore] because they mutate the
    // global DISK_LIMB_THRESHOLD atomic, which races with OTHER tests
    // in this crate that may concurrently allocate Integers above
    // whatever threshold these tests left in place.  Run them
    // explicitly with `cargo test --release storage:: -- --ignored`
    // or `--test-threads=1`.

    #[test]
    #[ignore]
    fn memory_path_basics() {
        let _g = TEST_LOCK.lock().unwrap();
        crate::config::apply(&crate::config::Config::default());
        let mut s = LimbStorage::allocate_zeroed(5);
        assert_eq!(s.len(), 5);
        assert!(!s.is_mapped());
        for (i, v) in s.as_mut_slice().iter_mut().enumerate() {
            *v = i as u64;
        }
        assert_eq!(s.as_slice(), &[0, 1, 2, 3, 4]);
        s.truncate(3);
        assert_eq!(s.as_slice(), &[0, 1, 2]);
        s.push(99);
        assert_eq!(s.as_slice(), &[0, 1, 2, 99]);
        assert_eq!(s.pop(), Some(99));
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    #[ignore]
    fn mapped_path_basics() {
        let _g = TEST_LOCK.lock().unwrap();
        // Force the mmap path with a tiny threshold.
        crate::config::apply(&crate::config::Config {
            disk_limb_threshold: 2,
            ..Default::default()
        });
        let mut s = LimbStorage::allocate_zeroed(5);
        assert_eq!(s.len(), 5);
        assert!(s.is_mapped());
        for (i, v) in s.as_mut_slice().iter_mut().enumerate() {
            *v = (i as u64) * 10;
        }
        assert_eq!(s.as_slice(), &[0, 10, 20, 30, 40]);
        s.truncate(3);
        assert_eq!(s.as_slice(), &[0, 10, 20]);
        assert_eq!(s.pop(), Some(20));
        s.clear();
        assert_eq!(s.len(), 0);
        // Restore defaults so we don't pollute later tests.
        crate::config::apply(&crate::config::Config::default());
    }

    #[test]
    #[ignore]
    fn from_vec_migrates_when_large() {
        let _g = TEST_LOCK.lock().unwrap();
        crate::config::apply(&crate::config::Config {
            disk_limb_threshold: 4,
            ..Default::default()
        });
        // Below threshold → memory.
        let s = LimbStorage::from_vec(vec![1, 2, 3]);
        assert!(!s.is_mapped());
        // At threshold → mapped.
        let s = LimbStorage::from_vec(vec![1, 2, 3, 4]);
        assert!(s.is_mapped());
        assert_eq!(s.as_slice(), &[1, 2, 3, 4]);
        crate::config::apply(&crate::config::Config::default());
    }

    #[test]
    #[ignore]
    fn drop_unlinks_scratch_file() {
        let _g = TEST_LOCK.lock().unwrap();
        crate::config::apply(&crate::config::Config {
            disk_limb_threshold: 1,
            ..Default::default()
        });
        let path_holder;
        {
            let s = LimbStorage::allocate_zeroed(8);
            let LimbStorage::Mapped { path, .. } = &s else {
                panic!("expected Mapped");
            };
            path_holder = path.clone();
            assert!(path_holder.exists(), "file should exist while storage lives");
        }
        // After drop, file is unlinked.
        assert!(!path_holder.exists(), "file should be unlinked after drop");
        crate::config::apply(&crate::config::Config::default());
    }

    #[test]
    #[ignore]
    fn live_counters_track_lifetime() {
        let _g = TEST_LOCK.lock().unwrap();
        crate::config::apply(&crate::config::Config {
            disk_limb_threshold: 1,
            ..Default::default()
        });
        let before_live = mmap_count_live();
        let before_total = mmap_count_total_allocated();
        {
            let _s = LimbStorage::allocate_zeroed(16);
            assert_eq!(mmap_count_live(), before_live + 1);
            assert_eq!(mmap_count_total_allocated(), before_total + 1);
        }
        assert_eq!(mmap_count_live(), before_live);
        assert_eq!(mmap_count_total_allocated(), before_total + 1);
        crate::config::apply(&crate::config::Config::default());
    }
}
