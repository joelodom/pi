//! Progress reporting trait used by the algorithms.
//!
//! CLI and library callers plug in any backend (terminal progress bars,
//! structured logging, telemetry, no-op).  The algorithms call into this
//! trait between phases and at each unit of work, so all I/O for progress
//! lives outside `pi-core`.

/// Receives progress notifications from a running computation.
pub trait ProgressReporter {
    /// Called when a named phase begins, with the expected total units.
    fn start_phase(&mut self, name: &str, total: u64);

    /// Called once per unit of work within the current phase.
    fn tick(&mut self);

    /// Called when the current phase ends.
    fn end_phase(&mut self);
}

/// A [`ProgressReporter`] that does nothing.  Suitable for library callers
/// that have no UI and tests that don't care.
#[derive(Default, Debug, Clone, Copy)]
pub struct NoopProgress;

impl ProgressReporter for NoopProgress {
    fn start_phase(&mut self, _: &str, _: u64) {}
    fn tick(&mut self) {}
    fn end_phase(&mut self) {}
}
