#![allow(dead_code)]

use crate::bitvector::BitVector;
use crate::utils::{binary_search_after, binary_search_after_by};

use bio::data_structures::rank_select::RankSelect;
use bv::BitVec;
use bv::BitsMut;

pub type OnesType = u32; // usize; // u32

#[derive(Debug)]
pub struct SparseBitVector {
    ones: Vec<OnesType>,
    chunk_size: usize,
    rank_blocks: RankSelect, // Vec<usize>,
    len: usize,
}

impl SparseBitVector {
    pub fn new(ones: Vec<OnesType>, len: usize) -> SparseBitVector {
        if let Some(&last) = ones.last() {
            assert!((last as usize) < len);
        }

        // accelerate rank queries by subdividing the universe into equally-sized chunks
        // and storing the number of ones that appear in each chunk.
        // Picking n/m as the chunk size gives us approximately m chunks, so the size
        // of the acceleration index is proportional to the number of elements (ones.len()).
        let n = len; // universe size
        let m = ones.len().max(1); // number of ones
        let chunk_size = (n as f64 / m as f64).ceil() as usize;
        let num_chunks = div_ceil(n, chunk_size);
        // println!("initializing sparse vector with n = {} and m = {} with {} chunks of size {}", n, m, num_chunks, chunk_size);
        let mut rank_blocks: Vec<usize> = vec![];
        rank_blocks.resize(num_chunks, 1); // pre-fill with count = 1 so that we can use select queries later even if some blocks have count 0...
        for one in ones.iter().copied() {
            rank_blocks[one as usize / chunk_size] += 1;
        }

        // convert the rank block values to cumulative sums
        let mut acc = 0;
        for x in rank_blocks.iter_mut() {
            *x += acc;
            acc = *x;
        }

        let nbits = acc + 1; // more than ones.len() due to the extra 'spacer' ones
        let mut bits: BitVec<u8> = BitVec::new_fill(false, nbits as u64);
        for x in rank_blocks {
            bits.set_bit(x as u64, true);
        }
        let rs_k = 8;
        // (nbits.ilog2().pow(2) / 32) as usize;
        // println!("rs k = {}", rs_k);
        let rs = RankSelect::new(bits, rs_k);

        SparseBitVector {
            ones,
            chunk_size,
            rank_blocks: rs,
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
        // dbg!(&self.ones, index);

        let i = index / self.chunk_size;
        let offset_lo = self.rank_blocks.select_1(i as u64).unwrap_or(0) as usize - i;
        let offset_hi = self
            .rank_blocks
            .select_1((i + 1) as u64)
            .unwrap_or(self.ones.len() as u64) as usize
            - (i + 1);
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
