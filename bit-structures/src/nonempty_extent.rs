// Represents a non-empty inclusive range.
// The non-empty invariant is upheld upon construction
use num::PrimInt;
use std::{debug_assert, ops::Range};

#[derive(Copy, Clone, Debug)]
pub struct Extent<T: PrimInt> {
    start: T,
    end: T,
}

impl<T: PrimInt> Extent<T> {
    pub fn new(start: T, end: T) -> Self {
        assert!(start <= end);
        Self { start, end }
    }
    // Return true if self overlaps other
    pub fn overlaps(&self, other: Extent<T>) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    pub fn overlaps_range(&self, other: Range<T>) -> bool {
        // assumes non-degenerate range
        debug_assert!(other.end > other.start);
        self.start < other.end && other.start <= self.end
    }

    // Return true if self fully contains other
    pub fn fully_contains(&self, other: Extent<T>) -> bool {
        // if self starts before other, and self ends after other.
        self.start <= other.start && self.end >= other.end
    }

    pub fn fully_contains_range(&self, other: Range<T>) -> bool {
        // assumes non-degenerate range
        debug_assert!(other.end > other.start);
        // if self starts before other, and self ends after other.
        self.start <= other.start && self.end >= other.end - T::one()
    }

    pub fn start(&self) -> T {
        self.start
    }

    pub fn end(&self) -> T {
        self.end
    }
}

impl<N: PrimInt> From<Range<N>> for Extent<N> {
    fn from(r: Range<N>) -> Self {
        assert!(!r.is_empty());
        Self::new(r.start, r.end - N::one())
    }
}
