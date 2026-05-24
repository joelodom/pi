//! `bignum`: a tiny, schoolbook, proof-of-concept replacement for the
//! `rug` crate (which wraps GMP/MPFR).
//!
//! The crate exposes just enough of the rug surface area to keep the
//! pi-core algorithms compiling and the existing test suite passing:
//! `Integer`, `Float`, the `Pow`/`PowAssign`/`Assign` traits, and a
//! single-variant `Round` enum.  Correctness > performance: multiplies
//! are O(N²) schoolbook, division is bit-level long division, etc.
//!
//! Only the standard library is used.

pub mod config;
pub mod float;
pub mod integer;
pub(crate) mod ntt;

pub use float::Float;
pub use integer::Integer;

/// Rounding mode for converting [`Float`] to [`Integer`].  Only
/// `Down` (toward -∞) is implemented because that's the single mode
/// used by pi-core's decimal-digit pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Round {
    Down,
}

/// Returns `base^rhs`.
pub trait Pow<Rhs> {
    type Output;
    fn pow(self, rhs: Rhs) -> Self::Output;
}

/// In-place exponentiation: `self <- self^rhs`.
pub trait PowAssign<Rhs> {
    fn pow_assign(&mut self, rhs: Rhs);
}

/// In-place assignment from a value or borrowed expression.
pub trait Assign<Src = Self> {
    fn assign(&mut self, src: Src);
}

// Inherent `Integer::assign(other)` is already defined; provide the
// trait flavor so `use bignum::Assign; n.assign(...)` resolves the same
// way as `use rug::Assign`.
impl Assign<Integer> for Integer {
    fn assign(&mut self, src: Integer) {
        Integer::assign(self, src);
    }
}

impl<'a> Assign<&'a Integer> for Integer {
    fn assign(&mut self, src: &'a Integer) {
        Integer::assign(self, src.clone());
    }
}

impl Assign<Float> for Float {
    fn assign(&mut self, src: Float) {
        Float::assign(self, src);
    }
}

impl<'a> Assign<&'a Float> for Float {
    fn assign(&mut self, src: &'a Float) {
        Float::assign(self, src.clone());
    }
}
