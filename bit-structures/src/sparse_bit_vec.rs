// Elias-Fano-encoded sparse bitvector

use crate::bincode_helpers::{
    bincode_borrow_decode_impl, bincode_decode_impl, bincode_encode_impl,
};
use crate::bit_block::BitBlock;
use crate::bit_buf::BitBuf;
use crate::bit_vec::{BitVec, MultiBitVec};
use crate::dense_bit_vec::DenseBitVec;
use crate::int_vec::IntVec;
use std::debug_assert;

#[derive(Debug)]
pub struct SparseBitVec<Ones: BitBlock> {
    high: DenseBitVec<Ones>, // High bit buckets in unary encoding
    low: IntVec,             // Low bits in fixed-width encoding
    num_ones: Ones,          // Number of elements (n)
    len: Ones,               // Maximum representable integer plus one
    low_bit_width: Ones,     // Number of low bits per element
    low_mask: Ones,          // Mask with the low_bit_width lowest bits set to 1
    has_multiplicity: bool,  // Whether any element is repeated more than once
}

impl<Ones: BitBlock> bincode::Encode for SparseBitVec<Ones> {
    bincode_encode_impl!(
        high,
        low,
        num_ones,
        len,
        low_bit_width,
        low_mask,
        has_multiplicity
    );
}

impl<Ones: BitBlock> bincode::Decode for SparseBitVec<Ones> {
    bincode_decode_impl!(
        high,
        low,
        num_ones,
        len,
        low_bit_width,
        low_mask,
        has_multiplicity
    );
}
impl<'de, Ones: BitBlock> bincode::BorrowDecode<'de> for SparseBitVec<Ones> {
    bincode_borrow_decode_impl!(
        high,
        low,
        num_ones,
        len,
        low_bit_width,
        low_mask,
        has_multiplicity
    );
}

impl<Ones: BitBlock> SparseBitVec<Ones> {
    // note: get, select0, rank0 will be incorrect when there is multiplicity.
    // because we rely on default impls, there is no room fo
    // debug_assert!(!self.has_multiplicity);
    // todo: figure out how to store all u32 elements including u32::MAX
    pub fn new(ones: &[Ones], len: Ones) -> Self {
        // todo: understand the comments in the paper "On Elias-Fano for Rank Queries in FM-Indexes"
        // but for now do the more obvious thing. todo: explain.
        // this is nice because we don't need the number of high bits explicitly so can avoid computing them
        let num_ones: Ones = Ones::from_usize(ones.len());
        let bits_per_one = if num_ones.is_zero() {
            Ones::zero()
        } else {
            len / num_ones
        };
        let low_bit_width = bits_per_one.max(Ones::one()).ilog2();
        let low_mask = Ones::one_mask(low_bit_width);

        // unary coding; 1 denotes values and 0 denotes separators
        let high_len = num_ones + (len >> low_bit_width);
        let mut high = BitBuf::new(high_len.usize());
        let mut low = IntVec::new(num_ones.usize(), low_bit_width as usize);
        let mut prev = Ones::zero();
        let mut has_multiplicity = false;

        for (n, one) in ones.iter().copied().enumerate() {
            debug_assert!(prev <= one);
            debug_assert!(one < len);

            // Track whether any element is repeated
            has_multiplicity |= n > 0 && one == prev;
            prev = one;

            // Encode element
            let quotient = one >> low_bit_width;
            high.set(n + quotient.usize());
            let remainder = one & low_mask;
            low.write_int(remainder.u32());
        }

        // todo: allow tuning of the block parameters
        let high = DenseBitVec::new(high, Ones::from_u32(10), Ones::from_u32(10));

        Self {
            high,
            low,
            num_ones,
            len,
            low_bit_width: Ones::from_u32(low_bit_width),
            low_mask,
            has_multiplicity,
        }
    }

    fn quotient(&self, value: Ones) -> Ones {
        value >> self.low_bit_width
    }

    fn remainder(&self, value: Ones) -> Ones {
        value & self.low_mask
    }
}

impl<Ones: BitBlock> BitVec for SparseBitVec<Ones> {
    type Ones = Ones;
    //     3: index of the first guy of the next group
    //  1: index of the first guy of this group
    // -1--33----7
    // 01234567890
    // o|oo||oooo|oo|ooooo|
    // 0 12  3456 78
    fn rank1(&self, index: Ones) -> Ones {
        if index >= self.len() {
            return self.num_ones();
        }

        let quotient = self.quotient(index);
        let (lower_bound, upper_bound) = if quotient.is_zero() {
            let lower_bound = Ones::zero();
            let upper_bound = self.high.select0(Ones::zero()).unwrap_or(self.num_ones);
            (lower_bound, upper_bound)
        } else {
            // compute the lower..upper range to search within the low bits
            // since the dense bitvec has Ones = u32, convert the quotient.
            // todo: try the equivalent of "as u32" if we can assert in the
            // constructor that we don't go beyond u32::MAX bits.
            let i = quotient - Ones::one();
            let lower_bound = self.high.select0(i).map(|x| x - i).unwrap_or(Ones::zero());
            let i = quotient;
            let upper_bound = self.high.select0(i).map(|x| x - i).unwrap_or(self.num_ones);
            (lower_bound, upper_bound)
        };

        // count the number of elements in this bucket that are strictly below i, using just the low bits
        let remainder = self.remainder(index);
        let bucket_count = (upper_bound - lower_bound).partition_point(|n| {
            let index = lower_bound + n;
            let value = Ones::from_u32(self.low.get(index.usize()));
            value < remainder
        });

        lower_bound + bucket_count
    }

    fn rank0(&self, index: Ones) -> Ones {
        debug_assert!(!self.has_multiplicity);
        self.default_rank0(index)
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        let quotient = self.high.rank0(self.high.select1(n)?);
        let remainder = Ones::from_u32(self.low.get(n.usize()));
        Some((quotient << self.low_bit_width) + remainder)
    }

    fn select0(&self, n: Ones) -> Option<Ones> {
        debug_assert!(!self.has_multiplicity);
        self.default_select0(n)
    }

    fn get(&self, index: Ones) -> bool {
        debug_assert!(!self.has_multiplicity);
        self.default_get(index)
    }

    fn num_ones(&self) -> Ones {
        self.num_ones
    }

    fn len(&self) -> Ones {
        self.len
    }
}

impl<Ones: BitBlock> MultiBitVec for SparseBitVec<Ones> {}

#[cfg(test)]
mod tests {
    use crate::bit_vec;

    use super::*;

    #[test]
    fn test_bitvector() {
        bit_vec::test_bitvector(SparseBitVec::<u32>::new);
        bit_vec::test_bitvector_vs_naive(SparseBitVec::<u32>::new);
    }

    // todo: sparse-specific tests
}
