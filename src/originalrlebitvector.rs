// for now
#![allow(dead_code)]

// use log::info;

use crate::bitvector::BitVector;
use crate::sparsebitvector::SparseBitVector;
use crate::utils::binary_search_after_by;

#[derive(Debug)]
pub struct OriginalRLEBitVector {
    z: SparseBitVector,
    o: SparseBitVector,
    len: usize,
    num_zeros: usize,
    num_ones: usize,
}

impl OriginalRLEBitVector {
    pub fn builder() -> OriginalRLEBitVectorBuilder {
        OriginalRLEBitVectorBuilder {
            z: vec![],
            o: vec![],
            len: 0,
            num_zeros: 0,
            num_ones: 0,
        }
    }
}

impl BitVector for OriginalRLEBitVector {
    fn rank1(&self, index: usize) -> usize {
        if index >= self.len {
            return self.num_ones;
        }

        // j is the index of the run containing the `index`-th element
        // invariant: j + 1 is always a valid number to pass to select.
        let j = binary_search_after_by(
            |i| self.z.select1(i + 1).unwrap() + self.o.select1(i + 1).unwrap(),
            index,
            0,
            self.z.num_ones(),
        ) - 1;

        // we want to know whether `index` is pointing to a zero.
        // we can determine this by checking whether the number of ones in preceding runs
        // plus the number of zeros up to and including this run is >= index.
        let preceding_ones = self.o.select1(j + 1).unwrap_or(0);
        let preceding_zeros_inclusive = self.z.select1(j + 2).unwrap_or(self.num_zeros);
        let index_is_zero = preceding_ones + preceding_zeros_inclusive > index;

        if index_is_zero {
            preceding_ones
        } else {
            // number of elements up to and including index, minus number of preceding zeros
            index + 1 - preceding_zeros_inclusive
        }
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

        // the n-th one is part of the r-th one run.
        let r = self.o.rank1(n - 1);

        // there are r complete zero runs before it, which means that
        // there are i zeros before it.
        let i = self.z.select1(r + 1).unwrap_or(self.num_zeros);

        // dbg!("select1:", n, r, i, i + n - 1);
        Some(i + n - 1)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        if n < 1 || n > self.num_zeros {
            return None;
        };

        // the n-th zero is part of the r-th zero run.
        let r = self.z.rank1(n - 1);

        // there are r - 1 complete one runs before it, which means that
        // there are i ones before it.
        let i = self.o.select1(r).unwrap();

        // dbg!("select0:", n, r, i, i + n - 1);
        Some(i + n - 1)
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
pub struct OriginalRLEBitVectorBuilder {
    z: Vec<usize>,
    o: Vec<usize>,
    len: usize,
    num_zeros: usize,
    num_ones: usize,
}

impl OriginalRLEBitVectorBuilder {
    pub fn run(&mut self, num_zeros: usize, num_ones: usize) {
        if num_zeros == 0 && num_ones == 0 {
            return;
        }

        // Only allow a no-zero run at the very beginning of the bitvector
        assert!((num_zeros > 0 && num_ones > 0) || (num_zeros == 0 && self.len == 0));
        // No coalescing is possible; create a new block of runs.
        // Append the number of new zeros to the Z array
        if num_zeros > 0 {
            self.z.push(num_zeros);
        }
        // Append the number of new ones to the O array
        self.o.push(num_ones);

        self.num_zeros += num_zeros;
        self.num_ones += num_ones;
        self.len += num_zeros + num_ones;
    }

    pub fn build(mut self) -> OriginalRLEBitVector {
        // cumulate z and o to derive 1 bit positions
        let mut z_off = 0;
        for i in 0..self.z.len() {
            let v = self.z[i];
            self.z[i] = z_off;
            z_off += v;
        }

        let mut o_off = 0;
        for i in 0..self.o.len() {
            let v = self.o[i];
            self.o[i] = o_off;
            o_off += v;
        }

        OriginalRLEBitVector {
            z: SparseBitVector::new(
                self.z.iter().map(|&x| x.try_into().unwrap()).collect(),
                self.num_zeros,
            ),
            o: SparseBitVector::new(
                self.o.iter().map(|&x| x.try_into().unwrap()).collect(),
                self.num_ones,
            ),
            len: self.len,
            num_zeros: self.num_zeros,
            num_ones: self.num_ones,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::rlebitvector::*;

    use rand::Rng;

    #[test]
    fn compare_bitvecs() {
        let mut rng = rand::thread_rng();

        // 100 iterations of constructing and checking 2 bitvectors
        for _ in 0..100 {
            let mut bb1 = OriginalRLEBitVector::builder();
            let mut bb2 = RLEBitVector::builder();

            // create vectors of runs of zeros and ones
            for _ in 1..100 {
                let num_zeros = rng.gen_range(1..100);
                let num_ones = rng.gen_range(1..100);
                bb1.run(num_zeros, num_ones);
                bb2.run(num_zeros, num_ones);
            }
            let bv1 = bb1.build();
            let bv2 = bb2.build();

            for i in 0..bv1.len() {
                assert_eq!(bv1.rank1(i), bv2.rank1(i));
                assert_eq!(bv1.rank0(i), bv2.rank0(i));
            }

            let num_ones = bv1.rank1(bv1.len());
            for i in 1..=num_ones {
                assert_eq!(bv1.select1(i), bv2.select1(i));
            }

            let num_zeros = bv1.rank0(bv1.len());
            for i in 1..=num_zeros {
                assert_eq!(bv1.select0(i), bv2.select0(i));
            }
        }
    }

    #[test]
    fn test_runs_rank() {
        let mut bb = OriginalRLEBitVector::builder();
        // 011 0001111 001
        bb.run(1, 2);
        bb.run(3, 4);
        bb.run(2, 1);
        let bv = bb.build();
        let ans = [0, 1, 2, 2, 2, 2, 3, 4, 5, 6, 6, 6, 7];
        for (i, x) in ans.into_iter().enumerate() {
            assert_eq!(bv.rank1(i), x);
            assert_eq!(bv.rank0(i), i + 1 - x);
        }
    }

    #[test]
    fn test_runs_select1() {
        let mut bb = OriginalRLEBitVector::builder();
        // 000011111111 001 01111
        bb.run(4, 8);
        bb.run(2, 1);
        bb.run(1, 4);
        let bv = bb.build();

        let ans = [4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19];
        for (i, x) in ans.into_iter().enumerate() {
            // println!("{:?}, {:?} ?= {:?}", i + 1, bv.select1(i + 1), x);
            assert_eq!(bv.select1(i + 1).unwrap(), x);
        }
        assert_eq!(bv.select1(0), None);
        assert_eq!(bv.select1(20), None);
        // assert_eq!(false, true);
    }

    #[test]
    fn test_runs_select0() {
        let mut bb = OriginalRLEBitVector::builder();
        // 000011111111 001 01111
        bb.run(4, 8);
        bb.run(2, 1);
        bb.run(1, 4);
        let bv = bb.build();
        let ans = [0, 1, 2, 3, 12, 13, 15];
        for (i, x) in ans.into_iter().enumerate() {
            // println!("{:?}, {:?} ?= {:?}", i + 1, bv.select0(i + 1), x);
            assert_eq!(bv.select0(i + 1).unwrap(), x);
        }
        assert_eq!(bv.select0(0), None);
        assert_eq!(bv.select0(16), None);
    }

    #[test]
    fn test_single_run() {
        let mut bb = OriginalRLEBitVector::builder();
        let num_zeros = 10;
        let num_ones = 12;
        bb.run(num_zeros, num_ones);
        bb.run(num_zeros, num_ones);
        let bv = bb.build();
        for i in 0..num_zeros {
            assert_eq!(bv.rank1(i), 0);
        }
        for i in num_zeros..num_zeros + num_ones {
            assert_eq!(bv.rank1(i), i - (num_zeros - 1));
        }
    }
}
