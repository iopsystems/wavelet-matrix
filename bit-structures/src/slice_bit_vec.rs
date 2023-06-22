// Simple bit vector implemented as a slice-backed dense array containing sorted indices of set bits.
// Should allow multiplicity (if there is multiplicity then select0/rank0 should be disallowed)

use crate::bit_block::BitBlock;
use std::debug_assert;

use crate::bit_vec::BitVec;

#[derive(Debug)]
pub struct SliceBitVec<Ones: BitBlock> {
    ones: Box<[Ones]>,
    len: usize,
}

impl<Ones: BitBlock> SliceBitVec<Ones> {
    pub fn new(ones: &[Ones], len: Ones) -> Self {
        assert!(
            len.to_usize().is_some(),
            "len must be convertible to usize since it is used as an array index"
        );

        debug_assert!(
            ones.windows(2).all(|w| w[0] < w[1]),
            "ones must be monotonically increasing"
        );

        debug_assert!(ones.len() <= len.as_usize());

        Self {
            ones: ones.into(),
            len: len.usize(),
        }
    }
}

impl<Ones: BitBlock> BitVec<Ones> for SliceBitVec<Ones> {
    fn rank1(&self, i: Ones) -> Ones {
        if i >= self.len() {
            return self.num_ones();
        }
        // Safety: n < num_ones and num_ones is derived from a array length, which is usize,
        // so as_usize is always correct.
        self.num_ones()
            .partition_point(|n| self.ones[n.as_usize()] < i)
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_ones() {
            return None;
        }
        // Safety: n < num_ones and num_ones is derived from a array length, which is usize,
        // so as_usize is always correct.
        Some(self.ones[n.as_usize()])
    }

    fn num_ones(&self) -> Ones {
        Ones::from_usize(self.ones.len())
    }

    fn len(&self) -> Ones {
        Ones::from_usize(self.len)
    }
}

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
