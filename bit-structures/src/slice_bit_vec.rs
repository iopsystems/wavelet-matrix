// SliceBitVec is a sparse bit vector backed by sorted integer slice. Allows multiplicity.

use crate::{bit_block::BitBlock, bit_vec::MultiBitVec};
use std::debug_assert;

use crate::bit_vec::BitVec;

#[derive(Debug)]
pub struct SliceBitVec<Ones: BitBlock> {
    ones: Box<[Ones]>,      // Sorted slice of values
    len: Ones,              // Maximum representable integer [todo: is it? or is it that plus one?]
    has_multiplicity: bool, // Whether any element is repeated more than once
}

impl<Ones: BitBlock> SliceBitVec<Ones> {
    // note: in the case of multiplicity-enabled bitvecs, `len` is a misnomer.
    // it is more like `max_one_index` or `universe`. still need to decide what
    // to do about Ones::max_value() – should we allow setting that bit?
    // todo: should we accept a multiplicity bool argument to know the user's intention?
    pub fn new(ones: &[Ones], len: Ones) -> Self {
        // len must be convertible to usize since it is used as an array index
        assert!(len.to_usize().is_some());

        // ones must be monotonically nondecreasing
        debug_assert!(ones.windows(2).all(|w| w[0] <= w[1]),);

        let has_multiplicity = ones.windows(2).any(|w| w[0] == w[1]);
        if !has_multiplicity {
            // there cannot be more set 1-bits than
            debug_assert!(ones.len() <= len.as_usize());
        }

        Self {
            ones: ones.into(),
            len,
            has_multiplicity,
        }
    }
}

impl<Ones: BitBlock> BitVec<Ones> for SliceBitVec<Ones> {
    fn rank1(&self, i: Ones) -> Ones {
        if i >= self.len() {
            return self.num_ones();
        }
        // Safety: n < num_ones and num_ones is derived from a array length, which is usize.
        self.num_ones()
            .partition_point(|n| self.ones[n.as_usize()] < i)
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_ones() {
            return None;
        }
        // Safety: n < num_ones and num_ones is derived from a array length, which is usize.
        Some(self.ones[n.as_usize()])
    }

    fn rank0(&self, index: Ones) -> Ones {
        debug_assert!(!self.has_multiplicity);
        self.default_rank0(index)
    }

    fn select0(&self, n: Ones) -> Option<Ones> {
        debug_assert!(!self.has_multiplicity);
        self.default_select0(n)
    }

    fn num_ones(&self) -> Ones {
        Ones::from_usize(self.ones.len())
    }

    fn len(&self) -> Ones {
        self.len
    }
}

impl<Ones: BitBlock> MultiBitVec<Ones> for SliceBitVec<Ones> {}

#[cfg(test)]
mod tests {
    // use super::*;
    // use crate::bitvector;

    #[test]
    fn test_bitvector() {
        // Test the naive bitvector for correctness in edge cases.
        // This is important because we use the naive case as a baseline for the others.
    }
}
