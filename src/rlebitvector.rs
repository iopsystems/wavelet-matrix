// for now
#![allow(dead_code)]

use log::info;

use crate::bitvector::BitVector;
use crate::sparsebitvector::SparseBitVector;
use crate::utils::binary_search_after_by;

// Represents a bitvector as a sequence of (run of zeros followed by a run of ones).
// Consecutive runs of the same digit are coalesced during constrution (in the builder).
// I think we rely on this
#[derive(Debug, bincode::Encode, bincode::Decode)]
pub struct RLEBitVector {
    // z[i]: Cumulative number of zeros before the start of the i-th 1-run;
    // can be thought of as pointing to the index of the first 1 in a 01-run.
    z: SparseBitVector,
    // zo[i]: Cumulative number of ones and zeros at the end of the i-th 01-run;
    // can be thought of as pointing just beyond the index of the last 1 in a 01-run.
    zo: SparseBitVector,
    len: usize,
    num_zeros: usize,
    num_ones: usize,
}

impl RLEBitVector {
    // todo: is there a way that we can make impossible-to-misuse versions of these functions?
    // eg. what are invariants that mean the index is always valid?
    // https://observablehq.com/@yurivish/bitvectors-with-runs-3
    // todo: think about what we actually store in Z and ZO - those are the numbers we want to retrieve.
    // ie. if aligned_rank1(select1(...)) is always valid => make a function aligned_rank1(nth_one)
    // todo: add [debug_]asserts to these aligned functions
    // each 0-run and 1-run. Results will be returned for other indices,
    // but may be wildly incorrect. So be very careful.
    pub fn aligned_rank0(&self, index: usize) -> usize {
        if index >= self.len {
            return self.num_zeros;
        };

        // Number of complete 01-runs up to virtual index i
        let j = self.zo.rank1(index);

        // Number of zeros preceding the (aligned) index i
        self.z.select1(j + 1).unwrap()
    }

    pub fn aligned_rank1(&self, index: usize) -> usize {
        if index >= self.len {
            return self.num_ones;
        };
        index - self.aligned_rank0(index) + 1
    }

    pub fn builder() -> RLEBitVectorBuilder {
        RLEBitVectorBuilder {
            z: Vec::new(),
            zo: Vec::new(),
            len: 0,
            num_zeros: 0,
            num_ones: 0,
        }
    }
}

impl BitVector for RLEBitVector {
    fn rank1(&self, index: usize) -> usize {
        if index >= self.len {
            return self.num_ones;
        };

        // Number of complete 01-runs up to and including the virtual index `index`
        let j = self.zo.rank1(index);

        // Number of zeros including the j-th block
        let num_cumulative_zeros = self.z.select1(j + 1).unwrap();

        // Number of zeros preceding the j-th block
        let num_preceding_zeros = if j == 0 {
            0
        } else {
            self.z.select1(j).unwrap()
        };

        // Number of zeros in the j-th block
        let num_zeros = num_cumulative_zeros - num_preceding_zeros;

        // Start index of the j-th block
        let block_start = if j == 0 {
            0
        } else {
            self.zo.select1(j).unwrap()
        };

        // Number of ones preceding the j-th block
        let num_preceding_ones = block_start - num_preceding_zeros;

        // Start index of ones in the j-th block
        let ones_start = block_start + num_zeros;

        // This used to be num_preceding_ones + 0.max(index - ones_start + 1),
        // but we need to prevent subtraction overflow.
        let adj = if index + 1 >= ones_start {
            index + 1 - ones_start
        } else {
            0
        };
        num_preceding_ones + adj
    }

    fn rank0(&self, index: usize) -> usize {
        if index >= self.len {
            return self.num_zeros;
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
        if n < 1 || n > self.num_ones {
            return None;
        }

        // The n-th one is in the j-th 01-block.
        let j = binary_search_after_by(
            |i| self.zo.select1(i + 1).unwrap() - self.z.select1(i + 1).unwrap(),
            n - 1,
            0,
            self.z.num_ones(),
        );

        // Number of zeros up to and including the jth block
        let num_cumulative_zeros = self.z.select1(j + 1).unwrap();

        Some(num_cumulative_zeros + n - 1)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        if n < 1 || n > self.num_zeros {
            return None;
        };

        // The n-th zero is in the j-th 01-block.
        let j = self.z.rank1(n - 1);

        // If we're in the first 01-block, the n-th zero is at index n - 1.
        if j == 0 {
            return Some(n - 1);
        };

        // Start index of the j-th block
        let block_start = self.zo.select1(j).unwrap();

        // Number of zeros preceding the jth block
        let num_preceding_zeros = self.z.select1(j).unwrap();

        // Return the index of the (n - numPrecedingZeros)th zero in the j-th 01-block.
        Some(block_start + (n - num_preceding_zeros) - 1)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn num_ones(&self) -> usize {
        self.num_ones
    }

    fn num_zeros(&self) -> usize {
        self.num_zeros
    }
}

#[derive(Clone)]
pub struct RLEBitVectorBuilder {
    z: Vec<usize>,
    zo: Vec<usize>,
    len: usize,
    num_zeros: usize,
    num_ones: usize,
}

impl RLEBitVectorBuilder {
    pub fn run(&mut self, num_zeros: usize, num_ones: usize) {
        if num_zeros == 0 && num_ones == 0 {
            return;
        }
        let len = self.z.len();
        self.num_zeros += num_zeros;
        self.num_ones += num_ones;
        self.len += num_zeros + num_ones;
        if num_zeros == 0 && len > 0 {
            // self run consists of only ones; coalesce it with the
            // previous run (since all runs contain ones at their end).
            self.zo[len - 1] += num_ones;
        } else if num_ones == 0 && self.last_block_contains_only_zeros() {
            // self run consists of only zeros; coalesce it with the
            // previous run (since it turns out to consist of only zeros).
            self.z[len - 1] += num_zeros;
            self.zo[len - 1] += num_zeros;
        } else {
            // No coalescing is possible; create a new block of runs.
            // Append the cumulative number of zeros to the Z array
            self.z.push(self.num_zeros);
            // Append the cumulative number of ones and zeros to the ZO array
            self.zo.push(self.len);
        }
    }

    pub fn build(self) -> RLEBitVector {
        info!("built RLEBitVector: {} runs", self.z.len());

        RLEBitVector {
            z: SparseBitVector::new(self.z, self.len),
            zo: SparseBitVector::new(self.zo, self.len),
            len: self.len,
            num_zeros: self.num_zeros,
            num_ones: self.num_ones,
        }
    }

    fn last_block_contains_only_zeros(&self) -> bool {
        let len = self.z.len();
        match len {
            0 => false,
            1 => self.z[0] == self.zo[0],
            _ => {
                let last_block_length = self.zo[len - 1] - self.zo[len - 2];
                let last_block_num_zeros = self.z[len - 1] - self.z[len - 2];
                last_block_length == last_block_num_zeros
            }
        }
    }
}
