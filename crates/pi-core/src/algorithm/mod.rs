//! Algorithm trait and registry.
//!
//! Currently registered: Chudnovsky (with binary splitting) and
//! Gauss-Legendre (Brent-Salamin AGM).  Future algorithms (Borwein
//! quintic, an NTT-backed Chudnovsky, …) implement [`PiAlgorithm`]
//! and add a variant to [`AlgorithmKind`].
//!
//! BBP is *not* in this registry — it cannot produce decimal digits
//! efficiently; it lives in [`crate::bbp`] as a verification oracle.

use std::fmt;

use anyhow::Result;

use crate::output::DigitSink;
use crate::progress::ProgressReporter;

pub mod chudnovsky;
pub mod gauss_legendre;
mod util;

/// Strategy for computing pi to a chosen number of decimal digits.
///
/// Implementations are responsible for picking their own series-vs-
/// iteration term count from a [`crate::precision::PrecisionPlan`].
/// The plan only computes working precision; nothing in the plan or
/// the trait is algorithm-specific.
pub trait PiAlgorithm {
    /// Stable identifier used in flags and logs.
    fn name(&self) -> &'static str;

    /// Compute `digits` decimal digits of pi (counting the leading `3`),
    /// streaming the result through `sink` and reporting progress to
    /// `progress`.  Implementations should call
    /// [`ProgressReporter::set_phases`] once at the start with their
    /// full phase list so multi-phase backends can render upcoming work.
    fn compute(
        &self,
        digits: u64,
        sink: &mut dyn DigitSink,
        progress: &mut dyn ProgressReporter,
    ) -> Result<()>;
}

/// Algorithm selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmKind {
    Chudnovsky,
    GaussLegendre,
}

impl AlgorithmKind {
    pub fn build(self) -> Box<dyn PiAlgorithm> {
        match self {
            AlgorithmKind::Chudnovsky => Box::new(chudnovsky::Chudnovsky),
            AlgorithmKind::GaussLegendre => Box::new(gauss_legendre::GaussLegendre),
        }
    }
}

impl fmt::Display for AlgorithmKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chudnovsky => f.write_str("chudnovsky"),
            Self::GaussLegendre => f.write_str("gauss-legendre"),
        }
    }
}
