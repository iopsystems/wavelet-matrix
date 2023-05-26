// Dense bit vector with rank and select, using the technique described
// in the paper "Fast, Small, Simple Rank/Select on Bitmaps".
// Uses a 32-bit universe size and so can store a little more than 4 billion bits.

use std::debug_assert;

use crate::raw_bitvector::RawBitVector;

#[derive(Debug)]
pub struct DenseBitVector {
    raw: RawBitVector, // bit data
    sr_bits: u32,      //
    ss_bits: u32,      //
    r: Box<[u32]>,     // rank samples
    s1: Box<[u32]>,    // select1 samples
}

impl DenseBitVector {
    fn new(data: RawBitVector, sr_bits: u32, ss_bits: u32) -> Self {
        let raw_block_bits = data.block_bits();
        debug_assert!(sr_bits >= raw_block_bits.ilog2());
        debug_assert!(ss_bits >= raw_block_bits.ilog2());

        let ss = 1 << ss_bits; // Select sampling rate: sample every `ss` 1-bits
        let sr = 1 << sr_bits; // Rank sampling rate: sample every `sr` bits

        let mut r = vec![]; // rank samples
        let mut s1 = vec![]; // select1 samples

        let mut cumulative_ones = 0; // 1-bits preceding the current raw block
        let mut cumulative_bits = 0; // bits preceding the current raw block
        let mut threshold = 0; // take a select sample at the (threshold+1)th 1-bit

        // Raw blocks per rank block
        let raw_block_sr = sr >> raw_block_bits.ilog2();

        // Iterate one rank block at a time for convenient rank sampling
        for blocks in data.blocks().chunks(raw_block_sr) {
            r.push(cumulative_ones); // Take a rank sample
            for block in blocks.iter().copied() {
                let block_ones = block.count_ones();
                if cumulative_ones + block_ones >= threshold {
                    // Take a select sample, which consists of two parts:
                    // 1. The cumulative bits preceding this raw block
                    let high = cumulative_bits;
                    // 2. The bit offset of the (ss * i + 1)-th 1-bit
                    let low = threshold - cumulative_ones;
                    // High is a multiple of the raw block size so these
                    // two values should never overlap.
                    debug_assert!(high & low == 0);
                    // Add the select sample and bump the threshold.
                    s1.push(high + low);
                    threshold += ss;
                }
                cumulative_ones += block_ones;
                cumulative_bits += raw_block_bits;
            }
        }

        Self {
            raw: data,
            sr_bits,
            ss_bits,
            r: r.into_boxed_slice(),
            s1: s1.into_boxed_slice(),
        }
    }

    // fn rank_block_index(i: usize) -> usize {

    // }

    // fn raw_block_index(i: usize) -> usize {

    // }

    fn rank1(&self, i: usize) -> usize {
        // let rank_block_index = i >> self.sr_bits;
        // let mut _rank = self.r[rank_block_index];

        // let j = 123;

        // let index = rank_block_index << self.sr_bits;
        // let raw_block_index = index >> RawBitVector::block_bits().ilog2();
        // let raw_blocks = ;

        // self.raw.blocks()

        // partition_point(self.r.len(), |ri| self.r[ri] < i);
        123
    }

    fn rank0(&self, _i: usize) -> usize {
        todo!()
    }

    fn select1(&self, i: usize) -> Option<usize> {
        if i >= self.raw.len() {
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
        for b in bv.raw.blocks() {
            println!("{:b}", b);
        }
        // panic!("aaa");
    }
}
