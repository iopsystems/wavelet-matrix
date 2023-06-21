// Elias-Fano-encoded sparse bitvector

use crate::bit_block::BitBlock;
use std::debug_assert;

use crate::bit_buf::BitBuf;
use crate::bit_vec;
use crate::bit_vec::BitVec;
use crate::dense_bit_vec::DenseBitVec;
use crate::int_vec::IntVec;
use crate::utils::partition_point;

pub struct SparseBitVec {
    high: DenseBitVec<u8>,  // High bit buckets in unary encoding
    low: IntVec,            // Low bits in fixed-width encoding
    num_ones: usize,        // Number of elements (n)
    len: usize,             // Maximum representable integer (u + 1)
    low_bits: usize,        // Number of low bits per element
    low_mask: usize,        // Mask with the low_bits lowest bits set to 1
    has_multiplicity: bool, // Whether any element is repeated more than once
}

impl SparseBitVec {
    // note: get, select0, rank0 will be incorrect when there is multiplicity.
    // because we rely on default impls, there is no room fo
    // debug_assert!(!self.has_multiplicity);
    // todo: figure out how to store all u32 elements including u32::MAX
    pub fn new(ones: &[usize], len: usize) -> Self {
        // todo: understand the comments in the paper "On Elias-Fano for Rank Queries in FM-Indexes"
        // but for now do the more obvious thing. todo: explain.
        // this is nice because we don't need the number of high bits explicitly so can avoid computing them
        let num_ones = ones.len();
        let bits_per_one = if num_ones == 0 { 0 } else { len / num_ones };
        let low_bits = bits_per_one.max(1).ilog2();
        let low_mask = usize::one_mask(low_bits);

        // unary coding; 1 denotes values and 0 denotes separators
        let high_len = num_ones + (len >> low_bits);
        let mut high = BitBuf::new(high_len);
        let mut low = IntVec::new(num_ones, low_bits as usize);
        let mut prev = 0;
        let mut has_multiplicity = false;

        for (n, one) in ones.iter().copied().enumerate() {
            debug_assert!(prev <= one);
            debug_assert!(one < len);

            // Track whether any element is repeated
            has_multiplicity |= n > 0 && one == prev;
            prev = one;

            // Encode element
            let quotient = one >> low_bits;
            high.set(n + quotient);
            let remainder = one & low_mask;
            low.write_int(remainder as u32);
        }

        // todo: allow tuning of the block parameters
        let high = DenseBitVec::new(high, 8, 8);

        Self {
            high,
            low,
            num_ones,
            len,
            // todo: this feels like an untidy consequences of
            // the way we approach u32/u64/usize – maybe it can
            // be removed through more careful use of types.
            low_bits: low_bits.try_into().unwrap(),
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

impl BitVec for SparseBitVec {
    type Ones = usize;

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
            (0, self.high.select0(0).unwrap_or(self.num_ones))
        } else {
            // compute the lower..upper range to search within the low bits
            let i = quotient - 1;
            let lower_bound = self.high.select0(i).map(|x| x - i).unwrap_or(0);

            let i = quotient;
            let upper_bound = self.high.select0(i).map(|x| x - i).unwrap_or(self.num_ones);

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
        bit_vec::default_rank0(self, index)
    }

    fn select1(&self, n: usize) -> Option<usize> {
        let quotient = self.high.rank0(self.high.select1(n)?);
        let remainder = self.low.get(n) as usize;
        Some((quotient << self.low_bits) + remainder)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        debug_assert!(!self.has_multiplicity);
        bit_vec::default_select0(self, n)
    }

    fn get(&self, index: usize) -> bool {
        debug_assert!(!self.has_multiplicity);
        bit_vec::default_get(self, index)
    }

    fn num_zeros(&self) -> usize {
        self.len() - self.num_ones()
    }

    fn num_ones(&self) -> usize {
        self.num_ones
    }

    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use crate::bit_vec;

    use super::*;

    #[test]
    fn test_bitvector() {
        bit_vec::test_bitvector(SparseBitVec::new);
        bit_vec::test_bitvector_vs_naive(SparseBitVec::new);
    }

    // todo: sparse-specific tests
}
