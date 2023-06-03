// Elias-Fano-encoded sparse bitvector

use std::{debug_assert, todo};

use crate::dense_bit_vector::DenseBitVector;
use crate::int_vector::IntVector;
use crate::raw_bit_vector::RawBitVector;
use crate::utils::{one_mask, partition_point};

struct SparseBitVector {
    high: DenseBitVector,
    low: IntVector,
    // Number of elements
    // In the sparse vector interpretation this is the number of ones (incl. multiplicities)
    len: usize,
    // Maximum representable integer
    universe: usize,
    low_bits: usize, // number of low bits
    low_mask: usize, // mask of low_bits 1s
    has_multiplicity: bool,
}

impl SparseBitVector {
    // todo: consider non-u32 value and universe
    pub fn new(len: usize, universe: usize, values: impl Iterator<Item = usize>) -> Self {
        // todo: understand the comments in the paper "On Elias-Fano for Rank Queries in FM-Indexes"
        // but for now do the more obvious thing. todo: explain.
        // this is nice because we don't need the number of high bits explicitly so can avoid computing them
        let low_bits = (universe / len).max(1).ilog2() as usize;

        // unary coding; 1 denotes values and 0 denotes separators
        let high_len = len + (universe >> low_bits);
        let mut high = RawBitVector::new(high_len);
        let mut low = IntVector::new(len, low_bits);
        let low_mask = one_mask(low_bits) as usize;
        let mut prev = 0;
        let mut has_multiplicity = false;
        for (n, value) in values.into_iter().enumerate() {
            debug_assert!(prev <= value);
            debug_assert!(value <= universe);

            has_multiplicity |= n > 0 && value == prev;
            prev = value;

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

    // ooo|oooo|oo|ooooo|
    pub fn rank1(&self, i: usize) -> usize {
        if i > self.universe {
            return self.len;
        }

        let quotient = self.quotient(i);

        // todo: make sure the lower bound is inclusive and upper is exclusive in both branches.

        // todo: explain the idea behind these bounds; the summary in "On Elias-Fano for Rank Queries in FM-Indexes" is
        // To perform rank(i,1) on X, we first find the id of the bucket i lies in,
        // and then perform two select operations on U, with consecutive arguments.
        // The first one gives the count of 1s up to the start of the bucket that xi lies in.
        // The second one tells us the size and endpoints of the subarray of L comprising this bucket.
        // Finally, we perform a rank operation within the bucket.
        let (lower_bound, upper_bound) = if quotient == 0 {
            (0, self.high.select0(0).unwrap_or(self.len))
        } else {
            // subtract the index to arrive at the actual value indices in the lower bits
            let i = quotient - 1;
            let lower_bound = self.high.select0(i).map(|x| x - i).unwrap_or(0);

            let i = quotient;
            let upper_bound = self.high.select0(i).map(|x| x - i).unwrap_or(self.len);

            (lower_bound, upper_bound)
        };

        // todo: think through the edge conditions - i think this isn't quite right
        // todo: remember, compensate for the marker elements...

        // no elements in this bucket; count all elements preceding it
        if lower_bound + 1 == upper_bound {
            return self.high.rank1(quotient);
        }

        // count the number of elements strictly below i using just the low bits
        let remainder = self.remainder(i) as u32;
        let bucket_count = partition_point(upper_bound - lower_bound, |n| {
            let index = lower_bound + n;
            let value = self.low.get(index);
            value < remainder
        });

        lower_bound + bucket_count
    }

    pub fn rank0(&self, i: usize) -> usize {
        debug_assert!(!self.has_multiplicity);
        if i > self.universe {
            return self.universe - self.len;
        }
        i - self.rank1(i)
    }

    pub fn select1(&self, n: usize) -> Option<usize> {
        let quotient = self.high.rank0(self.high.select1(n)?);
        let remainder = self.low.get(n) as usize;
        Some((quotient << self.low_bits) + remainder)
    }

    pub fn select0(&self, _n: usize) -> Option<usize> {
        debug_assert!(!self.has_multiplicity);
        todo!();
    }
}
