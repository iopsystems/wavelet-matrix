// Simple bit vector implemented as a slice-backed dense array containing sorted indices of set bits.
// Should allow multiplicity (if there is multiplicity then select0/rank0 should be disallowed)

use std::debug_assert;

use crate::bit_vec::BitVec;
use crate::utils::partition_point;

#[derive(Debug)]
pub struct SliceBitVec {
    ones: Box<[usize]>,
    len: usize,
}

impl SliceBitVec {
    pub fn new(ones: &[usize], len: usize) -> Self {
        debug_assert!(
            ones.windows(2).all(|w| w[0] < w[1]),
            "ones must be monotonically increasing"
        );
        debug_assert!(ones.len() <= len);
        Self {
            ones: ones.into(),
            len,
        }
    }
}

impl BitVec for SliceBitVec {
    fn rank1(&self, i: usize) -> usize {
        if i >= self.len() {
            return self.num_ones();
        }
        partition_point(self.num_ones(), |n| self.ones[n] < i)
    }

    fn select1(&self, n: usize) -> Option<usize> {
        if n >= self.num_ones() {
            return None;
        }
        Some(self.ones[n])
    }

    fn num_ones(&self) -> usize {
        self.ones.len()
    }

    fn len(&self) -> usize {
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
