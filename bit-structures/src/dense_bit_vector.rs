// Dense bit vector with rank and select, based on the ideas described
// in the paper "Fast, Small, Simple Rank/Select on Bitmaps".
// We use an additional level of blocks provided by the RawBitVector, but the ideas are the same.
// Uses a 32-bit universe size and so can store a little more than 4 billion bits.

// todo:
// - use select as an acceleration index for rank
//   - benchmark the effect on nonuniformly distributed 1 bits; i bet it helps more when the data are clustered
// x change rank to return the count strictly below the input index i so that rank and select become inverses.
//   x we can possibly reuse the rank block indexing check
// - is 'dense' the right name for this? the raw one is dense, this just adds rank/select support.

use crate::bit_block::BitBlock;
use std::debug_assert;

use crate::{
    raw_bit_vector::{self, RawBitVector},
    utils::one_mask,
};

#[derive(Debug)]
pub struct DenseBitVector {
    raw: RawBitVector, // bit data
    sr_pow2: u32,      //
    ss_pow2: u32,      //
    r: Box<[u32]>,     // rank samples
    s1: Box<[u32]>,    // select1 samples
    num_ones: usize,
}

impl DenseBitVector {
    fn new(data: RawBitVector, sr_pow2: u32, ss_pow2: u32) -> Self {
        let raw = data;
        let raw_block_bits = raw.block_bits();
        debug_assert!(sr_pow2 >= raw_block_bits.ilog2());
        debug_assert!(ss_pow2 >= raw_block_bits.ilog2());

        let ss = 1 << ss_pow2; // Select sampling rate: sample every `ss` 1-bits
        let sr = 1 << sr_pow2; // Rank sampling rate: sample every `sr` bits

        let mut r = vec![]; // rank samples
        let mut s1 = vec![]; // select1 samples

        let mut cumulative_ones = 0; // 1-bits preceding the current raw block
        let mut cumulative_bits = 0; // bits preceding the current raw block
        let mut threshold = 0; // take a select sample at the (threshold+1)th 1-bit

        // Raw blocks per rank block
        let raw_block_sr = sr >> raw_block_bits.ilog2();

        // Iterate one rank block at a time for convenient rank sampling
        for blocks in raw.blocks().chunks(raw_block_sr) {
            r.push(cumulative_ones); // Take a rank sample
            for block in blocks.iter() {
                let block_ones = block.count_ones();
                if cumulative_ones + block_ones >= threshold {
                    // Take a select sample, which consists of two parts:
                    // 1. The cumulative bits preceding this raw block
                    let high = cumulative_bits;
                    // 2. The bit offset of the (ss * i + 1)-th 1-bit within this raw block
                    let low = threshold - cumulative_ones;
                    // High is a multiple of the raw block size so these
                    // two values should never overlap in their bit ranges.
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
            raw,
            sr_pow2,
            ss_pow2,
            r: r.into_boxed_slice(),
            s1: s1.into_boxed_slice(),
            num_ones: cumulative_ones as usize,
        }
    }

    /// Return a select1 block index for a given 1-bit index
    fn s1_index(&self, n: usize) -> usize {
        n >> self.ss_pow2
    }

    /// Return a rank block index for a given bit index
    fn r_index(&self, i: usize) -> usize {
        i >> self.sr_pow2
    }

    /// Return a rank-block-aligned bit index for a given bit index
    fn rank_aligned_bit_index(&self, i: usize) -> usize {
        self.r_index(i) << self.sr_pow2
    }

    /// Return a raw block index for a given bit index
    fn raw_index(&self, i: usize) -> usize {
        self.raw.block_index(i)
    }

    /// Return the bit offset within the raw block
    fn raw_offset(&self, i: usize) -> usize {
        self.raw.bit_offset(i)
    }

    fn rank1(&self, i: usize) -> usize {
        if i >= self.raw.len() {
            return self.num_ones;
        }

        // Start with the prefix count from the rank block
        let mut rank = self.r[self.r_index(i)];

        // // self.s[select_index] points somewhere before i
        // let select_index = self.select_index(rank);

        // Sequentially scan raw blocks from raw_start onwards
        // todo: use select blocks to increase raw_start by hopping through select blocks
        let raw_start = self.raw_index(self.rank_aligned_bit_index(i));
        let raw_end = self.raw_index(i);
        let raw_slice = &self.raw.blocks()[raw_start..=raw_end];
        if let Some((last_block, blocks)) = raw_slice.split_last() {
            // Add the ones in fully-covered raw blocks
            for block in blocks {
                rank += block.count_ones()
            }

            // Add any ones in the final partly-covered raw block
            let raw_bit_offset = self.raw_offset(i);
            if raw_bit_offset > 0 {
                let mask = one_mask(raw_bit_offset);
                rank += (last_block & mask).count_ones();
            }
        }

        rank as usize
    }

    fn rank0(&self, i: usize) -> usize {
        let len = self.raw.len();
        if i > len {
            return len - self.num_ones;
        }
        i - self.rank1(i)
    }

    fn select1(&self, n: usize) -> Option<usize> {
        if n >= self.num_ones {
            return None;
        }

        // Steps (todo):
        // 1. Use the select block for an initial position
        // 2. Use rank blocks to hop over many raw blocks
        // 3. Use raw blocks to hop over many bytes
        // 4. Use bytes to hop over many bits
        // 5. Return the target bit position within the byte
        // debug_assert!(usize::BITS >= u32::BITS);

        // note: we may go past the last rank block, but never past the last raw block
        // (unless n >= self.num_ones)

        let sample_index = self.s1_index(n);
        let sample = self.s1[sample_index];
        // The raw block index of the preceding block, and the number of 1-bits
        // on the current block to reach the select_ones'th one
        // A select sample consists of two parts:
        // 1. The cumulative bits preceding this raw block. Since these are shifted down,
        //    they represent the raw block index.
        // 2. The bit offset of the (ss * i + 1)-th 1-bit within this raw block
        let (raw_start, correction) = self.raw.bit_split(sample as usize);
        let select_ones = sample_index << self.ss_pow2; // num. of ones represented by this sample

        // want: next block value and index, and the preceding one count up to that block.
        let mut prev_ones = select_ones - correction;
        let mut cur_ones = prev_ones;
        let raw_blocks = self.raw.blocks()[raw_start..].iter().copied();
        let (count, mut block) = raw_blocks
            .enumerate()
            .find(|(_, block)| {
                prev_ones = cur_ones;
                cur_ones += block.count_ones() as usize;
                cur_ones > n
            })
            .unwrap();

        let shift = raw_bit_vector::BT::bits().ilog2() as usize;
        for _ in prev_ones..n {
            block &= block - 1; // unset extra zeros
        }
        let block_bits = (raw_start + count) << shift;
        let bit_offset = block.trailing_zeros() as usize;
        Some(block_bits + bit_offset)
    }

    fn select0(&self, _i: usize) -> Option<usize> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_block::BitBlock;
    use crate::raw_bit_vector;
    use rand::Rng;

    #[test]
    fn test_ranks() {
        // todo: rewrite into a more compact and less arbitrary test case
        let ones = [1, 2, 5, 10, 32];
        let mut raw = RawBitVector::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        let bv = DenseBitVector::new(raw, 5, 5);
        // for b in bv.raw.blocks() {
        //     println!("{:b}", b);
        // }
        // panic!("aaa");
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
        let mut raw = RawBitVector::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        let bv = DenseBitVector::new(raw, 5, 5);
        assert_eq!(bv.select1(0), Some(1));
        assert_eq!(bv.select1(1), Some(2));
        assert_eq!(bv.select1(2), Some(5));
        assert_eq!(bv.select1(3), Some(10));
        assert_eq!(bv.select1(4), Some(32));
        assert_eq!(bv.select1(5), None);
    }

    #[test]
    fn test_select1_rand() {
        let n_iters = 100;
        for _ in 1..n_iters {
            let mut ones = vec![];
            let mut rng = rand::thread_rng();
            let mut prev = 0;
            for _ in 1..100 {
                let one = prev + rng.gen_range(1..3 * raw_bit_vector::BT::bits()) as usize;
                prev = one;
                ones.push(one);
            }

            // let ones = vec![  9, 25, 61, 76, 96, 134, 163, 187, 265];
            let mut raw = RawBitVector::new(ones.iter().max().unwrap() + 1);
            for o in ones.iter().copied() {
                raw.set(o);
            }

            println!("ones {:?}", ones);
            let bv = DenseBitVector::new(raw, 5, 5);
            for (i, o) in ones.iter().copied().enumerate() {
                println!("testing index {:?} with one  {:?}", i, o);
                assert_eq!(bv.select1(i), Some(o));
            }
            assert_eq!(bv.select1(ones.len()), None);
            assert_eq!(bv.select1(2 * ones.len()), None);
        }
    }

    // test todo:
    // - zero-length bitvector
    // - rank at around the exact length/block boundary of the bitvec
}
