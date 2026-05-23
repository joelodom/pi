//! Sinks for the computed digits of pi.
//!
//! The algorithm streams its result through a [`DigitSink`], so swapping out
//! the destination (stdout, single file, chunked file series, mmap, network)
//! never touches the computation code.

use std::fs::File;
use std::io::{self, BufWriter, Stdout, Write};
use std::path::Path;

/// Destination for the decimal expansion of pi.
///
/// The producer first calls [`Self::write_integer_part`] with the leading
/// `"3"`, then zero or more [`Self::write_fractional_digits`] calls with
/// chunks of fractional digits, then [`Self::finish`].  Implementations are
/// responsible for inserting the decimal point and any wrapping or
/// formatting they need.
pub trait DigitSink {
    fn write_integer_part(&mut self, digits: &str) -> io::Result<()>;
    fn write_fractional_digits(&mut self, digits: &str) -> io::Result<()>;
    fn finish(&mut self) -> io::Result<()>;
}

/// Generic `DigitSink` over any [`Write`] implementor.  Inserts the decimal
/// point automatically and (optionally) appends a trailing newline.
pub struct WriterSink<W: Write> {
    out: W,
    wrote_dot: bool,
    trailing_newline: bool,
}

impl<W: Write> WriterSink<W> {
    pub fn new(out: W) -> Self {
        Self { out, wrote_dot: false, trailing_newline: true }
    }

    pub fn without_trailing_newline(mut self) -> Self {
        self.trailing_newline = false;
        self
    }
}

impl<W: Write> DigitSink for WriterSink<W> {
    fn write_integer_part(&mut self, digits: &str) -> io::Result<()> {
        self.out.write_all(digits.as_bytes())
    }

    fn write_fractional_digits(&mut self, digits: &str) -> io::Result<()> {
        if !self.wrote_dot {
            self.out.write_all(b".")?;
            self.wrote_dot = true;
        }
        self.out.write_all(digits.as_bytes())
    }

    fn finish(&mut self) -> io::Result<()> {
        if self.trailing_newline {
            self.out.write_all(b"\n")?;
        }
        self.out.flush()
    }
}

/// Construct a buffered stdout sink (the default for the CLI).
pub fn stdout_sink() -> WriterSink<BufWriter<Stdout>> {
    WriterSink::new(BufWriter::with_capacity(64 * 1024, io::stdout()))
}

/// Construct a buffered single-file sink.  Uses a 1 MiB write buffer because
/// the digits arrive in one big string at the end of the computation.
pub fn file_sink(path: impl AsRef<Path>) -> io::Result<WriterSink<BufWriter<File>>> {
    let file = File::create(path)?;
    Ok(WriterSink::new(BufWriter::with_capacity(1 << 20, file)))
}
