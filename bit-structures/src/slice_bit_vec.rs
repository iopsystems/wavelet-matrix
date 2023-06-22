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
        // check that the length can be converted to usize (should we check u32 instead?)
        // assert!(usize::try_from(len).is_ok());

        debug_assert!(
            ones.windows(2).all(|w| w[0] < w[1]),
            "ones must be monotonically increasing"
        );
        // debug_assert!(ones.len() <= len); // duplicates are allowed
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
        self.num_ones()
            .partition_point(|n| self.ones[n.usize()] < i)
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_ones() {
            return None;
        }
        Some(self.ones[n.usize()])
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
