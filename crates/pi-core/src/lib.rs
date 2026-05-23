//! Core library for computing pi to arbitrary precision.
//!
//! Components are exposed as small traits so the moving pieces can be
//! swapped independently as the target scale grows:
//!
//! * [`algorithm::PiAlgorithm`] — picks the math.  Currently only the
//!   Chudnovsky algorithm with binary splitting is implemented, but
//!   Gauss-Legendre, BBP, Borwein, etc. would slot in here.
//! * [`output::DigitSink`] — picks where digits go (stdout, file, chunked
//!   file series, mmap, network).  The algorithm never sees the concrete
//!   destination.
//! * [`progress::ProgressReporter`] — picks how progress is reported
//!   (silent, indicatif, log).
//! * [`precision::PrecisionPlan`] — picks the working precision and term
//!   count from a target digit count.
//!
//! Big-number arithmetic uses GMP/MPFR via the `rug` crate; at Chudnovsky
//! scale this is by far the heaviest part of the computation, and GMP's
//! FFT multiplication is what makes a million-digit run finish in seconds.

pub mod algorithm;
pub mod output;
pub mod precision;
pub mod progress;

pub use algorithm::{AlgorithmKind, PiAlgorithm};
pub use output::DigitSink;
pub use precision::PrecisionPlan;
pub use progress::{NoopProgress, Phase, ProgressReporter};
