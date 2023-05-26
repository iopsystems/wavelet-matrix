// Dense bit vector with rank and select, using the technique described
// in the paper "Fast, Small, Simple Rank/Select on Bitmaps".

use std::debug_assert;

use crate::raw_bitvector::RawBitVector;

type BT = u32; // Block type. TODO: Should we distinguish between types for rank vs. select blocks?

#[derive(Debug)]
pub struct DenseBitVector {
    data: RawBitVector, // bit data
    r: Box<[BT]>,       // rank samples
    s: Box<[BT]>,       // select samples
}

impl DenseBitVector {
    fn new(data: RawBitVector, sr_bits: u32, ss_bits: u32) -> Self {
        debug_assert!(sr_bits >= RawBitVector::block_bits().ilog2());
        debug_assert!(ss_bits >= RawBitVector::block_bits().ilog2());

        let ss = 1 << ss_bits; // Select sampling rate, in 1-bits
        let sr = 1 << sr_bits; // Rank sampling rate, in bits

        let mut r = vec![]; // rank samples
        let mut s = vec![]; // select samples

        let mut cumulative_ones = 0; // 1-bits preceding the current raw block
        let mut cumulative_bits = 0; // bits preceding the current raw block
        let mut one_threshold = 0; // take a select sample at the (one_threshold+1)th 1-bit

        // Raw blocks per rank block
        let raw_block_sr = sr >> RawBitVector::block_bits().ilog2();

        // Iterate one rank block at a time for convenient rank sampling
        for blocks in data.blocks().chunks(raw_block_sr) {
            r.push(cumulative_ones); // Take a rank sample
            for block in blocks.iter().copied() {
                let block_ones = block.count_ones();
                if cumulative_ones + block_ones >= one_threshold {
                    // Take a select sample, which consists of two parts:
                    // 1. The cumulative bits preceding this raw block
                    let high = cumulative_bits;
                    // 2. The bit offset of the (ss * i + 1)-th 1-bit
                    let low = one_threshold - cumulative_ones;
                    // High is a multiple of the raw block size so these
                    // two values should never overlap.
                    debug_assert!(high & low == 0);
                    // Add the select sample and bump the threshold.
                    s.push(high + low);
                    one_threshold += ss;
                }
                cumulative_ones += block_ones;
                cumulative_bits += RawBitVector::block_bits();
            }
        }

        Self {
            data,
            r: r.into_boxed_slice(),
            s: s.into_boxed_slice(),
        }
    }

    fn rank1(&self, _i: usize) -> usize {
        todo!()
    }

    fn rank0(&self, _i: usize) -> usize {
        todo!()
    }

    fn select1(&self, i: usize) -> Option<usize> {
        if i >= self.data.len() {
            return None;
        }
        todo!()
    }

    fn select0(&self, _i: usize) -> Option<usize> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_works() {
        let ones = [0, 1, 2, 5, 10, 32];
        let mut raw = RawBitVector::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        let bv = DenseBitVector::new(raw, 5, 5);
        for b in bv.data.blocks() {
            println!("{:b}", b);
        }
        // panic!("aaa");
    }
}
