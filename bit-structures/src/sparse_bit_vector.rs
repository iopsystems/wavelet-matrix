// Elias-Fano-encoded sparse bitvector

use std::debug_assert;

use crate::bitvector;
use crate::bitvector::BitVector;
use crate::dense_bit_vector::DenseBitVector;
use crate::int_vector::IntVector;
use crate::raw_bit_vector::RawBitVector;
use crate::utils::{one_mask, partition_point};

struct SparseBitVector {
    high: DenseBitVector,   // High bit buckets in unary encoding
    low: IntVector,         // Low bits in fixed-width encoding
    len: usize,             // Number of elements
    universe: usize,        // Maximum representable integer
    low_bits: usize,        // Number of low bits per element
    low_mask: usize,        // Mask with the low_bits lowest bits set to 1
    has_multiplicity: bool, // Whether any element is repeated more than once
}

impl SparseBitVector {
    // note: get, select0, rank0 will be incorrect when there is multiplicity.
    // because we rely on default impls, there is no room fo
    // debug_assert!(!self.has_multiplicity);

    pub fn new(len: usize, universe: usize, values: impl Iterator<Item = usize>) -> Self {
        // todo: understand the comments in the paper "On Elias-Fano for Rank Queries in FM-Indexes"
        // but for now do the more obvious thing. todo: explain.
        // this is nice because we don't need the number of high bits explicitly so can avoid computing them
        let low_bits = (universe / len).max(1).ilog2() as usize;
        let low_mask = one_mask(low_bits) as usize;

        // unary coding; 1 denotes values and 0 denotes separators
        let high_len = len + (universe >> low_bits);
        let mut high = RawBitVector::new(high_len);
        let mut low = IntVector::new(len, low_bits);
        let mut prev = 0;
        let mut has_multiplicity = false;
        for (n, value) in values.into_iter().enumerate() {
            debug_assert!(prev <= value);
            debug_assert!(value <= universe);

            // Track whether any element is repeated
            has_multiplicity |= n > 0 && value == prev;
            prev = value;

            // Encode element
            let quotient = value >> low_bits;
            let remainder = value & low_mask;
            high.set(n + quotient);
            low.write_int(remainder as u32);
        }

        // todo: allow tuning of the block parameters
        let high = DenseBitVector::new(high, 8, 8);

        Self {
            high,
            low,
            len,
            universe,
            low_bits,
            low_mask,
            has_multiplicity,
        }
    }

    // todo: use this in the constructor?
    fn quotient(&self, value: usize) -> usize {
        value >> self.low_bits
    }

    fn remainder(&self, value: usize) -> usize {
        value & self.low_mask
    }
}

impl BitVector for SparseBitVector {
    //     3: index of the first guy of the next group
    //  1: index of the first guy of this group
    // -1--33----7
    // 01234567890
    // o|oo||oooo|oo|ooooo|
    // 0 12  3456 78
    fn rank1(&self, index: usize) -> usize {
        if index >= self.len() {
            return self.num_ones();
        }

        let quotient = self.quotient(index);
        let (lower_bound, upper_bound) = if quotient == 0 {
            (0, self.high.select0(0).unwrap_or(self.len))
        } else {
            // compute the lower..upper range to search within the low bits
            let i = quotient - 1;
            let lower_bound = self.high.select0(i).map(|x| x - i).unwrap_or(0);

            let i = quotient;
            let upper_bound = self.high.select0(i).map(|x| x - i).unwrap_or(self.len);

            (lower_bound, upper_bound)
        };

        // count the number of elements in this bucket that are strictly below i, using just the low bits
        let remainder = self.remainder(index) as u32;
        let bucket_count = partition_point(upper_bound - lower_bound, |n| {
            let index = lower_bound + n;
            let value = self.low.get(index);
            value < remainder
        });

        lower_bound + bucket_count
    }

    fn rank0(&self, index: usize) -> usize {
        debug_assert!(!self.has_multiplicity);
        bitvector::default_rank0(self, index)
    }

    fn select1(&self, n: usize) -> Option<usize> {
        let quotient = self.high.rank0(self.high.select1(n)?);
        let remainder = self.low.get(n) as usize;
        Some((quotient << self.low_bits) + remainder)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        debug_assert!(!self.has_multiplicity);
        if n >= self.num_zeros() {
            return None;
        }
        // Binary search over ranks for select0.
        // Note: an alternative strategy that involves identifying the 0-run
        // containing the n-th 0-bit is used by simple-sds and may be more efficient.
        let index = partition_point(self.len(), |i| self.rank0(i) <= n);
        Some(index - 1)
    }

    fn get(&self, index: usize) -> bool {
        debug_assert!(!self.has_multiplicity);
        bitvector::default_get(self, index)
    }

    fn num_zeros(&self) -> usize {
        self.len() - self.num_ones()
    }

    fn num_ones(&self) -> usize {
        self.len
    }

    fn len(&self) -> usize {
        // note: can overflow if usize == u32 and the universe is the max.
        //       but we want to be able to represent the max value for eg.
        //       morton-order things.
        // todo: should we therefore use u64 for these properties?
        debug_assert!(self.universe < usize::MAX);
        self.universe + 1
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvector;

    use super::*;

    #[test]
    fn test_bitvector() {
        bitvector::test_bitvector_vs_naive(|ones, len| {
            SparseBitVector::new(ones.len(), len.saturating_sub(1), ones.iter().copied())
        });
    }

    #[test]
    fn test_rank1() {
        // todo: rewrite into a more compact and less arbitrary test case
        let ones = [1, 2, 5, 10, 32];
        let len = ones.len();
        let universe = ones.iter().max().copied().unwrap();
        let bv = SparseBitVector::new(len, universe, ones.iter().copied());

        assert_eq!(bv.rank1(0), 0);
        assert_eq!(bv.rank1(1), 0);
        assert_eq!(bv.rank1(2), 1);
        assert_eq!(bv.rank1(3), 2);
        assert_eq!(bv.rank1(4), 2);
        assert_eq!(bv.rank1(5), 2);
        assert_eq!(bv.rank1(9), 3);
        assert_eq!(bv.rank1(10), 3);
        assert_eq!(bv.rank1(31), 4);
        assert_eq!(bv.rank1(32), 4);

        assert_eq!(bv.rank0(0), 0);
        assert_eq!(bv.rank0(1), 1);
        assert_eq!(bv.rank0(2), 1);
        assert_eq!(bv.rank0(3), 1);
        assert_eq!(bv.rank0(4), 2);
        assert_eq!(bv.rank0(5), 3);
        assert_eq!(bv.rank0(9), 6);
        assert_eq!(bv.rank0(10), 7);
        assert_eq!(bv.rank0(31), 27);
        assert_eq!(bv.rank0(32), 28);
        assert_eq!(bv.rank0(320), 28);
    }
    #[test]
    fn test_select1() {
        // todo: rewrite into a more compact and less arbitrary test case
        let ones = [1, 2, 5, 10, 32];
        let len = ones.len();
        let universe = ones.iter().max().copied().unwrap();
        let bv = SparseBitVector::new(len, universe, ones.iter().copied());
        assert_eq!(bv.select1(0), Some(1));
        assert_eq!(bv.select1(1), Some(2));
        assert_eq!(bv.select1(2), Some(5));
        assert_eq!(bv.select1(3), Some(10));
        assert_eq!(bv.select1(4), Some(32));
        assert_eq!(bv.select1(5), None);
    }
}
