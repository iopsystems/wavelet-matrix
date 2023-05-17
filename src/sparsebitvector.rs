#![allow(dead_code)]

use crate::bitvector::BitVector;
use crate::utils::{binary_search_after, binary_search_after_by};

pub type OnesType = u32; // usize; // u32

#[derive(Debug, bincode::Encode, bincode::Decode)]
pub struct SparseBitVector {
    ones: Vec<OnesType>,
    chunk_size: usize,
    rank_blocks: Vec<usize>,
    len: usize,
}

impl SparseBitVector {
    pub fn new(ones: Vec<OnesType>, len: usize) -> SparseBitVector {
        if let Some(&last) = ones.last() {
            assert!((last as usize) < len);
        }

        let n = len;
        let m = ones.len().max(1);
        let chunk_size = (n as f64 / m as f64).ceil() as usize;
        let num_chunks = div_ceil(n, chunk_size);
        // println!("initializing sparse vector with n = {} and m = {} with {} chunks of size {}", n, m, num_chunks, chunk_size);
        let mut rank_blocks: Vec<usize> = vec![];
        rank_blocks.resize(num_chunks, 0);
        for one in ones.iter().copied() {
            rank_blocks[one as usize / chunk_size] += 1;
        }

        // cumulate
        let mut acc = 0;
        for x in rank_blocks.iter_mut() {
            *x += acc;
            acc = *x;
        }
        rank_blocks.push(ones.len());

        SparseBitVector {
            ones,
            chunk_size,
            rank_blocks,
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
        let i = index / self.chunk_size;
        let offset_lo = if i == 0 { 0 } else { self.rank_blocks[i - 1] };
        let offset_hi = self.rank_blocks[i];
        binary_search_after(&self.ones, index.try_into().unwrap(), offset_lo, offset_hi)
        // binary_search_after(&self.ones, index.try_into().unwrap(), 0, self.ones.len())
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
        SparseBitVector::new(self.ones, self.len)
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
