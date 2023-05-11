#![allow(dead_code)]

use crate::bitvector::BitVector;
use crate::utils::{binary_search_after, binary_search_after_by};

pub type OnesType = u32; // usize

#[derive(Debug, bincode::Encode, bincode::Decode)]
pub struct SparseBitVector {
    ones: Vec<OnesType>,
    len: usize,
}

impl SparseBitVector {
    pub fn new(ones: Vec<usize>, len: usize) -> SparseBitVector {
        SparseBitVector {
            // convert from usize to u32 (inefficient, but hopefully temporary)
            ones: ones.into_iter().map(|d| d.try_into().unwrap()).collect(),
            len,
        }
    }

    pub fn builder(len: usize) -> SparseBitVectorBuilder {
        SparseBitVectorBuilder {
            ones: Vec::new(),
            len,
        }
    }
}

impl BitVector for SparseBitVector {
    fn rank1(&self, index: usize) -> usize {
        binary_search_after(&self.ones, index.try_into().unwrap(), 0, self.ones.len())
    }

    fn rank0(&self, index: usize) -> usize {
        if index >= self.len {
            return self.num_zeros();
        }
        index - self.rank1(index) + 1
    }

    // Returns the value of the `i`-th bit as a bool.
    fn access(&self, index: usize) -> bool {
        assert!(index < self.len, "out of bounds");
        // Quick hack. Can do better.
        let ones_count = self.rank1(index) - self.rank1(index - 1);
        ones_count == 1
    }

    fn select1(&self, n: usize) -> Option<usize> {
        if n < 1 || n > self.num_ones() {
            return None;
        }
        Some(self.ones[n - 1] as usize)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        if n < 1 || n > self.num_zeros() {
            return None;
        }
        if self.num_ones() == 0 {
            return Some(n - 1);
        }
        // i is the index of the block containing the i-th zero (it's in the run preceding the 1).
        // There are i ones before it.
        let i = binary_search_after_by(|i| self.ones[i] as usize - i, n - 1, 0, self.ones.len());
        Some(n + i - 1)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn num_zeros(&self) -> usize {
        self.len - self.ones.len()
    }

    fn num_ones(&self) -> usize {
        self.ones.len()
    }
}

#[derive(Clone)]
pub struct SparseBitVectorBuilder {
    ones: Vec<OnesType>,
    len: usize,
}

impl SparseBitVectorBuilder {
    pub fn one(&mut self, index: usize) {
        assert!(index < self.len, "out of bounds");
        self.ones.push(index.try_into().unwrap());
    }

    pub fn build(mut self) -> SparseBitVector {
        self.ones.sort();
        SparseBitVector {
            ones: self.ones,
            len: self.len,
        }
    }
}
