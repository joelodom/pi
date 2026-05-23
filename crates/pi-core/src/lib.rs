//! Core library for computing pi to arbitrary precision.
//!
//! Components are exposed as small traits so the moving pieces can be
//! swapped independently as the target scale grows:
//!
//! * [`algorithm::PiAlgorithm`] — picks the math.  Chudnovsky (default,
//!   fastest) and Gauss-Legendre (independent cross-check) are
//!   implemented; future algorithms slot in here.
//! * [`output::DigitSink`] — picks where digits go.  Currently stdout
//!   and single-file backends are implemented; the algorithm never sees
//!   the concrete destination, so adding new ones (chunked file series,
//!   network, …) doesn't touch the compute code.
//! * [`progress::ProgressReporter`] — picks how progress is reported
//!   (silent, indicatif, log).  Algorithms call [`progress::ProgressReporter::set_phases`]
//!   up front so multi-phase reporters can show upcoming work.
//! * [`precision::PrecisionPlan`] — picks the working precision (bits)
//!   from a target digit count.  Series-vs-iteration term counts are
//!   algorithm-private.
//! * [`bbp`] — pure-Rust Bailey-Borwein-Plouffe hex-digit extractor used
//!   by `pi --verify-hex` as an independent oracle for spot-checking
//!   computed pi files.  Not used to produce digits.
//!
//! Big-number arithmetic uses GMP/MPFR via the `rug` crate; at Chudnovsky
//! scale this is by far the heaviest part of the computation, and GMP's
//! FFT multiplication is what makes a billion-digit run finish in minutes
//! and a million-digit run finish in well under a second.

pub mod algorithm;
pub mod bbp;
pub mod output;
pub mod precision;
pub mod progress;

pub use algorithm::{AlgorithmKind, PiAlgorithm};
pub use output::DigitSink;
pub use precision::PrecisionPlan;
pub use progress::{NoopProgress, Phase, ProgressReporter};
