// Simple bit vector implemented as a slice-backed dense array containing sorted indices of set bits.
// Should allow multiplicity (if there is multiplicity then select0/rank0 should be disallowed)

use std::debug_assert;

use crate::bit_block::BitBlock;
use crate::bit_vec::BitVec;
use crate::utils::partition_point;

#[derive(Debug)]
pub struct SliceBitVec<T: BitBlock> {
    ones: Box<[T]>,
    len: usize,
}

impl<T: BitBlock> SliceBitVec<T> {
    pub fn new(ones: &[T], len: usize) -> Self {
        // debug_assert!(
        //     ones.windows(2).all(|w| w[0] < w[1]),
        //     "ones must be monotonically increasing"
        // );
        debug_assert!(ones.len() <= len);
        Self {
            ones: ones.into(),
            len,
        }
    }
}

impl<T: BitBlock> BitVec for SliceBitVec<T> {
    type Ones = T;

    fn rank1(&self, i: T) -> T {
        if i >= self.len() {
            return self.num_ones();
        }
        partition_point(self.num_ones(), |n| self.ones[n.to_usize().unwrap()] < i)
    }

    fn select1(&self, n: T) -> Option<T> {
        if n >= self.num_ones() {
            return None;
        }
        Some(self.ones[n.to_usize().unwrap()])
    }

    fn num_ones(&self) -> T {
        self.ones.len()
    }

    fn len(&self) -> T {
        self.len
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
