// Dense bit vector with rank and select, based on the ideas described
// in the paper "Fast, Small, Simple Rank/Select on Bitmaps".
// We use an additional level of blocks provided by the RawBitVec, but the ideas are the same.
// Uses a 32-bit universe size and so can store a little more than 4 billion bits.

// todo:
// - use select as an acceleration index for rank
//   - benchmark the effect on nonuniformly distributed 1 bits; i bet it helps more when the data are clustered
// x change rank to return the count strictly below the input index i so that rank and select become inverses.
//   x we can possibly reuse the rank block indexing check
// - is 'dense' the right name for this? the raw one is dense, this just adds rank/select support.

use crate::bit_block::BitBlock;
use crate::bit_vec::BitVec;
use std::debug_assert;

use crate::bit_buf::BitBuf;

type RawBlock = u8;

// todo: describe what each rank/select sample holds.

#[derive(Debug)]
pub struct DenseBitVec {
    raw: BitBuf<RawBlock>, // bit data
    sr_pow2: u32,          //
    ss_pow2: u32,          //
    r: Box<[u32]>,         // rank samples holding the number of preceding 1-bits
    s0: Box<[u32]>,        // select0 samples
    s1: Box<[u32]>,        // select1 samples
    num_ones: usize,
}

impl DenseBitVec {
    pub fn new(data: BitBuf<RawBlock>, sr_log2: u32, ss_log2: u32) -> Self {
        let raw = data;
        let raw_block_bits = RawBlock::BITS;
        let raw_block_pow2 = RawBlock::WIDTH;
        debug_assert!(sr_log2 >= raw_block_pow2);
        debug_assert!(ss_log2 >= raw_block_pow2);

        let ss = 1 << ss_log2; // Select sampling rate: sample every `ss` 1-bits
        let sr = 1 << sr_log2; // Rank sampling rate: sample every `sr` bits

        let mut r = vec![]; // rank samples
        let mut s0 = vec![]; // select0 samples
        let mut s1 = vec![]; // select1 samples

        let mut cumulative_ones = 0; // 1-bits preceding the current raw block
        let mut cumulative_bits = 0; // bits preceding the current raw block
        let mut zeros_threshold = 0; // take a select0 sample at the (zeros_threshold+1)th 1-bit
        let mut ones_threshold = 0; // take a select1 sample at the (ones_threshold+1)th 1-bit

        // Raw blocks per rank block
        let raw_block_sr = sr >> raw_block_pow2;

        // Iterate one rank block at a time for convenient rank sampling
        for blocks in raw.blocks().chunks(raw_block_sr) {
            r.push(cumulative_ones); // Take a rank sample
            for block in blocks.iter() {
                let block_ones = block.count_ones();
                if cumulative_ones + block_ones > ones_threshold {
                    // Take a select sample, which consists of two parts:
                    // 1. The cumulative bits preceding this raw block
                    let high = cumulative_bits;
                    // 2. The number of 1-bits before the (ss * i + 1)-th 1-bit within this raw block
                    let low = ones_threshold - cumulative_ones;
                    // High is a multiple of the raw block size so these
                    // two values should never overlap in their bit ranges.
                    debug_assert!(high & low == 0);
                    // Add the select sample and bump the ones_threshold.
                    s1.push(high + low);
                    ones_threshold += ss;
                }
                let block_zeros = raw_block_bits - block_ones;
                let cumulative_zeros = cumulative_bits - cumulative_ones;
                if cumulative_zeros + block_zeros > zeros_threshold {
                    // Take a select sample, which consists of two parts:
                    // 1. The cumulative bits preceding this raw block
                    let high = cumulative_bits;
                    // 2. The number of 0-bits before (ss * i + 1)-th 0-bit within this raw block
                    let low = zeros_threshold - cumulative_zeros;
                    // High is a multiple of the raw block size so these
                    // two values should never overlap in their bit ranges.
                    debug_assert!(high & low == 0);
                    // Add the select sample and bump the ones_threshold.
                    s0.push(high + low);
                    zeros_threshold += ss;
                }
                cumulative_ones += block_ones;
                cumulative_bits += raw_block_bits;
            }
        }

        Self {
            raw,
            sr_pow2: sr_log2,
            ss_pow2: ss_log2,
            r: r.into_boxed_slice(),
            s0: s0.into_boxed_slice(),
            s1: s1.into_boxed_slice(),
            num_ones: cumulative_ones as usize,
        }
    }

    /// Return a select1 block index for a given 1-bit index
    fn s_index(&self, n: usize) -> usize {
        n >> self.ss_pow2
    }

    /// Return a rank block index for a given bit index
    fn r_index(&self, index: usize) -> usize {
        index >> self.sr_pow2
    }

    /// Return a rank-block-aligned bit index for a given bit index
    fn rank_aligned_bit_index(&self, index: usize) -> usize {
        self.r_index(index) << self.sr_pow2
    }
}

impl BitVec for DenseBitVec {
    fn rank1(&self, index: usize) -> usize {
        if index >= self.len() {
            return self.num_ones();
        }

        // Start with the prefix count from the rank block
        let mut rank = self.r[self.r_index(index)];

        // // self.s[select_index] points somewhere before index
        // let select_index = self.select_index(rank);

        // Sequentially scan raw blocks from raw_start onwards
        // todo: use select blocks to increase raw_start by hopping through select blocks
        let raw_start = RawBlock::block_index(self.rank_aligned_bit_index(index));
        let raw_end = RawBlock::block_index(index);
        let raw_slice = &self.raw.blocks()[raw_start..=raw_end];
        if let Some((last_block, blocks)) = raw_slice.split_last() {
            // Add the ones in fully-covered raw blocks
            for block in blocks {
                rank += block.count_ones()
            }

            // Add any ones in the final partly-covered raw block
            let raw_bit_offset = RawBlock::bit_offset(index);
            if raw_bit_offset > 0 {
                let mask: RawBlock = RawBlock::one_mask(raw_bit_offset as u32);
                rank += (last_block & mask).count_ones();
            }
        }

        rank as usize
    }

    fn select1(&self, n: usize) -> Option<usize> {
        // our rank blocks hold u32 indices and it seems odd for us to support larger array sizes...
        debug_assert!(n <= u32::MAX as usize);

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

        // {
        //     let (raw_start, correction) = self.s1_block(n);
        //     let skipped_raw_blocks = self.raw.blocks()[raw_start..].iter().take_while(|raw_block|raw_block<n).count();
        // }

        // note: we may go past the last rank block, but never past the last raw block
        // (unless n >= self.num_ones)

        let sample_index = self.s_index(n);
        let sample = self.s1[sample_index];
        // The raw block index of the preceding block, and the number of 1-bits
        // on the current block to reach the select_ones'th one
        // A select sample consists of two parts:
        // 1. The cumulative bits preceding this raw block. Since these are shifted down,
        //    they represent the raw block index.
        // 2. The bit offset of the (ss * i + 1)-th 1-bit within this raw block
        let (mut raw_start, correction) = RawBlock::index_offset(sample as usize);
        let select_ones = sample_index << self.ss_pow2; // num. of ones represented by this sample

        let mut preceding_ones = select_ones - correction;

        // Speed past multiple raw blocks using rank blocks.
        // Convert the raw block index into a rank block index
        // note: we could do this all in one swoop by finding the last valid rank block
        // and using the number of blocks traversed and the final value of the block.
        // of course, only do that if the number of skipped rank blocks is greater than 0.
        // note: all these 'pow2's are misleading - this is the log2/bitwidth...
        let raw_blocks_per_rank_block_pow2 = self.sr_pow2 - RawBlock::WIDTH;
        // convert raw index -> rank index
        let preceding_rank_block = raw_start >> raw_blocks_per_rank_block_pow2;
        let rank_start = preceding_rank_block + 1; // start from the next block
        let rank_iter = self.r[rank_start..].iter().copied();

        for (i, r) in rank_iter.enumerate() {
            let r = r as usize;
            if r < n {
                preceding_ones = r;
                // convert rank index -> raw index
                raw_start = (rank_start + i) << raw_blocks_per_rank_block_pow2;
            } else {
                break;
            }
        }

        // want: the next raw block value and index, and the preceding one count up to that block.
        let mut cur_ones = preceding_ones;
        let raw_blocks = self.raw.blocks()[raw_start..].iter().copied();
        let (count, mut block) = raw_blocks
            .enumerate()
            .find(|(_, block)| {
                preceding_ones = cur_ones;
                cur_ones += block.count_ones() as usize;
                cur_ones > n
            })
            .unwrap();

        let shift = RawBlock::WIDTH as usize;
        for _ in preceding_ones..n {
            block &= block - 1; // unset extra zeros
        }
        let block_bits = (raw_start + count) << shift;
        let bit_offset = block.trailing_zeros() as usize;
        Some(block_bits + bit_offset)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        if n >= self.num_zeros() {
            return None;
        }
        let sample_index = self.s_index(n);
        let sample = self.s0[sample_index];

        let (raw_start, correction) = RawBlock::index_offset(sample as usize);
        let select_zeros = sample_index << self.ss_pow2; // num. of zeros represented by this sample
        let mut prev_zeros = select_zeros - correction;
        let mut cur_zeros = prev_zeros;
        let raw_blocks = self.raw.blocks()[raw_start..].iter().copied();
        let (count, mut block) = raw_blocks
            .enumerate()
            .find(|(_, block)| {
                prev_zeros = cur_zeros;
                cur_zeros += block.count_zeros() as usize;
                cur_zeros > n
            })
            .unwrap();
        let shift = RawBlock::WIDTH as usize;

        block = !block;
        for _ in prev_zeros..n {
            block &= block - 1; // unset extra zeros
        }
        block = !block;

        let block_bits = (raw_start + count) << shift;
        let bit_offset = block.trailing_ones() as usize;
        Some(block_bits + bit_offset)
    }

    fn num_ones(&self) -> usize {
        self.num_ones
    }

    fn len(&self) -> usize {
        self.raw.len()
    }

    fn get(&self, index: usize) -> bool {
        self.raw.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::bit_block::BitBlock;
    use crate::bit_vec;
    use rand::Rng;

    #[test]
    fn test_new() {
        let raw = BitBuf::new(100);
        let _ = DenseBitVec::new(raw, RawBlock::WIDTH, RawBlock::WIDTH);
    }

    #[test]
    fn test_bitvector() {
        let f = |ones: &[usize], len| {
            let mut raw = BitBuf::new(len);
            for one in ones.iter().copied() {
                raw.set(one);
            }
            DenseBitVec::new(raw, RawBlock::WIDTH, RawBlock::WIDTH)
        };
        bit_vec::test_bitvector(f);
        bit_vec::test_bitvector_vs_naive(f);
    }

    #[test]
    fn test_ranks() {
        // todo: rewrite into a more compact and less arbitrary test case
        let ones = [1, 2, 5, 10, 32];
        let mut raw = BitBuf::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        let bv = DenseBitVec::new(raw, RawBlock::WIDTH, RawBlock::WIDTH);
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
        let mut raw = BitBuf::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        let bv = DenseBitVec::new(raw, RawBlock::WIDTH, RawBlock::WIDTH);
        assert_eq!(bv.select1(0), Some(1));
        assert_eq!(bv.select1(1), Some(2));
        assert_eq!(bv.select1(2), Some(5));
        assert_eq!(bv.select1(3), Some(10));
        assert_eq!(bv.select1(4), Some(32));
        assert_eq!(bv.select1(5), None);
    }

    #[test]
    fn test_select0() {
        // todo: rewrite into a more compact and less arbitrary test case
        let ones = [1, 2, 5, 10];
        let mut raw = BitBuf::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        let bv = DenseBitVec::new(raw, RawBlock::WIDTH, RawBlock::WIDTH);
        assert_eq!(bv.select0(0), Some(0));
        assert_eq!(bv.select0(1), Some(3));
        assert_eq!(bv.select0(2), Some(4));
        assert_eq!(bv.select0(3), Some(6));
        assert_eq!(bv.select0(4), Some(7));
        assert_eq!(bv.select0(5), Some(8));
        assert_eq!(bv.select0(6), Some(9));
        assert_eq!(bv.select0(7), None);
        assert_eq!(bv.select0(8), None);
    }

    #[test]
    fn test_select1_rand() {
        let n_iters = 100;
        for _ in 1..n_iters {
            let mut ones = vec![];
            let mut rng = rand::thread_rng();
            let mut prev = 0;
            for _ in 1..100 {
                let one = prev + rng.gen_range(1..3 * RawBlock::BITS) as usize;
                prev = one;
                ones.push(one);
            }

            // let ones = vec![  9, 25, 61, 76, 96, 134, 163, 187, 265];
            let mut raw = BitBuf::new(ones.iter().max().unwrap() + 1);
            for o in ones.iter().copied() {
                raw.set(o);
            }

            println!("ones {:?}", ones);
            let bv = DenseBitVec::new(raw, RawBlock::WIDTH, RawBlock::WIDTH);
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
