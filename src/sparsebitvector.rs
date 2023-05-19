#![allow(dead_code)]

use crate::bitvector::BitVector;

use simple_sds::ops::{BitVec, Rank, Select, SelectZero};
use simple_sds::serialize::Serialize;
use simple_sds::sparse_vector::{SparseBuilder, SparseVector};

pub type OnesType = u32; // usize;

#[derive(Debug)]
pub struct SparseBitVector {
    ones: SparseVector,
    len: usize,
}

impl SparseBitVector {
    pub fn new(ones: Vec<OnesType>, len: usize) -> SparseBitVector {
        let mut b = SparseBuilder::new(len, ones.len()).unwrap();
        b.extend(ones.into_iter().map(|x| x as usize));
        assert!(b.is_full());
        let v = SparseVector::try_from(b).unwrap();

        println!("{},", 8 * v.size_in_bytes());

        SparseBitVector { ones: v, len }
    }

    pub fn builder(len: usize) -> SparseBitVectorBuilder {
        SparseBitVectorBuilder {
            ones: Vec::new(),
            len,
        }
    }
}

impl BitVector for SparseBitVector {
    // Returns the number of one bits at or below the bit index `index`.
    fn rank1(&self, index: usize) -> usize {
        self.ones.rank(index + 1)
    }

    // Returns the number of zero bits at or below index `i`.
    fn rank0(&self, index: usize) -> usize {
        if index >= self.len {
            return self.num_zeros();
        }
        index - self.rank1(index) + 1
    }

    // Returns the value of the `i`-th bit as a bool.
    fn access(&self, index: usize) -> bool {
        assert!(index < self.len, "out of bounds");
        self.ones.get(index)
    }

    // Returns the index of the n-th one, with n ranging from 1 to the number of ones.
    fn select1(&self, n: usize) -> Option<usize> {
        if n == 0 {
            return None;
        }
        self.ones.select(n - 1)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        if n == 0 {
            return None;
        }
        self.ones.select_zero(n - 1)
    }

    fn len(&self) -> usize {
        self.ones.len()
    }

    fn num_zeros(&self) -> usize {
        self.ones.count_zeros()
    }

    fn num_ones(&self) -> usize {
        self.ones.count_ones()
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
