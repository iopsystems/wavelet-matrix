// Simple bit vector implemented as a slice-backed dense array containing sorted indices of set bits.
// Should allow multiplicity (if there is multiplicity then select0/rank0 should be disallowed)

use std::debug_assert;

use num::ToPrimitive;

use crate::bit_block::BitBlock;
use crate::bit_vec::{BitVec, Ones};
use crate::utils::PartitionPoint;

#[derive(Debug)]
pub struct SliceBitVec {
    ones: Box<[Ones]>,
    len: usize,
}

impl SliceBitVec {
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
            len: len.into_usize(),
        }
    }
}

impl BitVec for SliceBitVec {
    fn rank1(&self, i: Ones) -> Ones {
        if i >= self.len() {
            return self.num_ones();
        }
        self.num_ones()
            .partition_point(|n| self.ones[n.into_usize()] < i)
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_ones() {
            return None;
        }
        Some(self.ones[n.to_usize().unwrap()])
    }

    fn num_ones(&self) -> Ones {
        self.ones.len().try_into().unwrap()
    }

    fn len(&self) -> Ones {
        self.len.try_into().unwrap()
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
