// Compressed representation of bitvector with runs of 0-bits and 1-bits.
// The rank0/rank1 functions are efficient, and select0 is efficient; select1 requires a binary search over the full range.
// We could implement a "flipped" bitvector wrapper to make select1 efficient.

use crate::bit_block::BitBlock;
use crate::bit_vec::BitVec;
use crate::sparse_bit_vec::SparseBitVec;

pub struct RLEBitVec<Ones: BitBlock> {
    // z[i]: Cumulative number of zeros before the start of the i-th 1-run;
    // can be thought of as pointing to the index of the first 1 in a 01-run.
    z: SparseBitVec<Ones>,
    // zo[i]: Cumulative number of ones and zeros at the end of the i-th 01-run;
    // can be thought of as pointing just beyond the index of the last 1 in a 01-run.
    zo: SparseBitVec<Ones>,
    len: Ones,
    num_zeros: Ones,
    num_ones: Ones,
}

impl<Ones: BitBlock> RLEBitVec<Ones> {
    // todo: is there a way that we can make impossible-to-misuse versions of these functions?
    // eg. what are invariants that mean the index is always valid?
    // https://observablehq.com/@yurivish/bitvectors-with-runs-3
    // todo: think about what we actually store in Z and ZO - those are the numbers we want to retrieve.
    // ie. if aligned_rank1(select1(...)) is always valid => make a function aligned_rank1(nth_one)
    // todo: add [debug_]asserts to these aligned functions
    // each 0-run and 1-run. Results will be returned for other indices,
    // but may be wildly incorrect. So be very careful.
    // note: this assumes the old definitions of rank/select
    pub fn aligned_rank0(&self, index: Ones) -> Ones {
        if index >= self.len {
            return self.num_zeros;
        };

        // Number of complete 01-runs up to virtual index i
        let j = self.zo.rank1(index);

        // Number of zeros preceding the (aligned) index i
        self.z.select1(j + Ones::one()).unwrap()
    }

    pub fn aligned_rank1(&self, index: Ones) -> Ones {
        if index >= self.len {
            return self.num_ones;
        };
        index - self.aligned_rank0(index) + Ones::one()
    }

    pub fn builder() -> RLEBitVectorBuilder<Ones> {
        RLEBitVectorBuilder {
            z: Vec::new(),
            zo: Vec::new(),
            len: Ones::zero(),
            num_zeros: Ones::zero(),
            num_ones: Ones::zero(),
        }
    }
}

impl<Ones: BitBlock> BitVec<Ones> for RLEBitVec<Ones> {
    fn rank1(&self, index: Ones) -> Ones {
        if index >= self.len {
            return self.num_ones;
        };

        // Number of complete 01-runs up to the virtual index `index`
        let j = self.zo.rank1(index);

        // Number of zeros including the j-th block
        let num_cumulative_zeros = self.z.select1(j).unwrap();

        // Number of zeros preceding the j-th block
        // note: relies on the fact that our bitvectors cannot represent
        // Ones::max_value() values since that is reserved for the len
        let num_preceding_zeros = self
            .z
            .select1(j.wrapping_sub(&Ones::one()))
            .unwrap_or(Ones::zero());

        // Number of zeros in the j-th block
        let num_zeros = num_cumulative_zeros - num_preceding_zeros;

        // Start index of the j-th block
        let block_start = self
            .zo
            .select1(j.wrapping_sub(&Ones::one()))
            .unwrap_or(Ones::zero());

        // Number of ones preceding the j-th block
        let num_preceding_ones = block_start - num_preceding_zeros;

        // Start index of ones in the j-th block
        let ones_start = block_start + num_zeros;

        // This used to be num_preceding_ones + 0.max(index - ones_start + 1),
        // but we need to prevent subtraction overflow.
        let adj = (index).saturating_sub(ones_start);
        num_preceding_ones + adj
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_ones {
            return None;
        }

        // The n-th one is in the j-th 01-block.
        let j = self
            .z
            .num_ones()
            .partition_point(|i| self.zo.select1(i).unwrap() - self.z.select1(i).unwrap() <= n);

        // Number of zeros up to and including the j-th block
        let num_cumulative_zeros = self.z.select1(j).unwrap();

        Some(num_cumulative_zeros + n)
    }

    fn select0(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_zeros {
            return None;
        };

        // The n-th zero is in the j-th 01-block.
        let j = self.z.rank1(n + Ones::one());

        // If we're in the first 01-block, the n-th zero is at index n.
        if j.is_zero() {
            return Some(n);
        };

        // Start index of the j-th 01-block
        let block_start = self.zo.select1(j - Ones::one()).unwrap();

        // Number of zeros preceding the j-th 01-block
        let num_preceding_zeros = self.z.select1(j - Ones::one()).unwrap();

        // Return the index of the (n - num_preceding_zeros)-th zero in the j-th 01-block.
        Some(block_start + (n - num_preceding_zeros))
    }

    fn len(&self) -> Ones {
        self.len
    }

    fn num_ones(&self) -> Ones {
        self.num_ones
    }

    fn num_zeros(&self) -> Ones {
        self.num_zeros
    }
}

#[derive(Clone)]
pub struct RLEBitVectorBuilder<Ones> {
    z: Vec<Ones>,
    zo: Vec<Ones>,
    len: Ones,
    num_zeros: Ones,
    num_ones: Ones,
}

impl<Ones: BitBlock> RLEBitVectorBuilder<Ones> {
    pub fn run(&mut self, num_zeros: Ones, num_ones: Ones) {
        if num_zeros == Ones::zero() && num_ones == Ones::zero() {
            return;
        }
        let len = self.z.len();
        self.num_zeros += num_zeros;
        self.num_ones += num_ones;
        self.len += num_zeros + num_ones;
        if num_zeros == Ones::zero() && len > 0 {
            // this run consists of only ones; coalesce it with the
            // previous run (since all runs contain ones at their end).
            self.zo[len - 1] += num_ones;
        } else if num_ones.is_zero() && self.last_block_contains_only_zeros() {
            // this run consists of only zeros; coalesce it with the
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

    pub fn build(self) -> RLEBitVec<Ones> {
        RLEBitVec {
            z: SparseBitVec::new(&self.z, self.num_zeros + Ones::one()),
            zo: SparseBitVec::new(&self.zo, self.num_zeros + self.num_ones + Ones::one()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_vec;

    pub fn new<Ones: BitBlock>(ones: &[Ones], len: Ones) -> RLEBitVec<Ones> {
        let mut bb = RLEBitVec::<Ones>::builder();
        // use wrapping subtraction for the case when the first one is at index 0
        let mut prev = Ones::max_value();
        for &one in ones {
            bb.run(one.wrapping_sub(&prev) - Ones::one(), Ones::one());
            prev = one;
        }
        let extra_len = len.wrapping_sub(&prev) - Ones::one();
        bb.run(extra_len, Ones::zero());
        bb.build()
    }

    #[test]
    fn test_bitvector() {
        bit_vec::test_bitvector(new::<u32>);
        bit_vec::test_bitvector_vs_naive(new::<u32>);
    }

    #[test]
    fn test_single_run() {
        let mut bb = RLEBitVec::<u32>::builder();
        let num_zeros = 10;
        let num_ones = 12;
        bb.run(num_zeros, num_ones);
        let bv = bb.build();
        for i in 0..=num_zeros {
            assert_eq!(bv.rank1(i), 0);
        }
        for i in num_zeros + 1..=num_zeros + num_ones {
            assert_eq!(bv.rank1(i), i - num_zeros);
        }
    }

    #[test]
    fn test_runs_rank() {
        let mut bb = RLEBitVec::<u32>::builder();
        bb.run(1, 2);
        bb.run(3, 4);
        bb.run(2, 1);
        let bv = bb.build();
        let ans = [0, 0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 6, 7];
        for (i, x) in ans.into_iter().enumerate() {
            let i = i as u32;
            dbg!(i, bv.rank0(i), bv.rank1(i));
            assert_eq!(bv.rank1(i), x);
            assert_eq!(bv.rank0(i), i - x);
        }
    }

    #[test]
    fn test_runs_select1() {
        let mut bb = RLEBitVec::<u32>::builder();
        bb.run(1, 2);
        bb.run(3, 4);
        bb.run(2, 1);
        let bv = bb.build();
        let ans = [1, 2, 6, 7, 8, 9, 12];
        for (i, x) in ans.into_iter().enumerate() {
            let i = i as u32;
            assert_eq!(bv.select1(i).unwrap(), x);
        }
        assert_eq!(bv.select1(7), None);
    }

    #[test]
    fn test_runs_select0() {
        let mut bb = RLEBitVec::<u32>::builder();
        bb.run(1, 2);
        bb.run(3, 4);
        bb.run(2, 1);
        let bv = bb.build();
        let ans = [0, 3, 4, 5, 10, 11];
        for (i, x) in ans.into_iter().enumerate() {
            let i = i as u32;
            dbg!(i, bv.select0(i).unwrap());
            assert_eq!(bv.select0(i).unwrap(), x);
        }
        assert_eq!(bv.select1(14), None);
    }
}
