// SliceBitVec is a sparse bit vector backed by sorted integer slice. Allows multiplicity.
// todo: test this type independently from the comparison tests which check it against the other bitvecs.
// todo: implement batch_rank using multi-value binary search

use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::bit_vec::{BitVec, MultiBitVec};
use std::debug_assert;

#[derive(Debug)]
pub struct SliceBitVec<Ones: BitBlock> {
    ones: Box<[Ones]>,      // Sorted slice of values
    len: Ones,              // Maximum representable integer plus one
    has_multiplicity: bool, // Whether any element is repeated more than once
}

impl<Ones: BitBlock> bincode::Encode for SliceBitVec<Ones> {
    encode_impl!(ones, len, has_multiplicity);
}

impl<Ones: BitBlock> bincode::Decode for SliceBitVec<Ones> {
    decode_impl!(ones, len, has_multiplicity);
}
impl<'de, Ones: BitBlock> bincode::BorrowDecode<'de> for SliceBitVec<Ones> {
    borrow_decode_impl!(ones, len, has_multiplicity);
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

        if let Some(&one) = ones.last() {
            debug_assert!(one <= len); // todo: < or <=?
        }

        Self {
            ones: ones.into(),
            len,
            has_multiplicity,
        }
    }
}

impl<Ones: BitBlock> BitVec for SliceBitVec<Ones> {
    type Ones = Ones;

    fn rank1(&self, i: Ones) -> Ones {
        if i >= self.len() {
            return self.num_ones();
        }

        // Indexing safety: n < num_ones and num_ones is derived from an array length, which is usize.
        // (if we want to later change this to .get_unchecked)
        self.num_ones()
            .partition_point(|n| self.ones[n.as_usize()] < i)
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_ones() {
            return None;
        }
        // Indexing safety: n < num_ones and num_ones is derived from a array length, which is usize.
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

impl<Ones: BitBlock> MultiBitVec for SliceBitVec<Ones> {}

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
