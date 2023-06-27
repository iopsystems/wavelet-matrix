// Dense bit vector with rank and select, based on the ideas described
// in the paper "Fast, Small, Simple Rank/Select on Bitmaps".
// We use an additional level of blocks provided by the RawBitVec, but the ideas are the same.

// todo:
//  - benchmark the effect on nonuniformly distributed 1 bits; i bet it helps more when the data are clustered
//  - try split_last in select1
use crate::bit_block::BitBlock;
use crate::bit_buf::BitBuf;
use crate::bit_vec::BitVec;
use crate::utils::select1;
use std::debug_assert;

// todo: describe what each rank/select sample holds.

#[derive(Debug)]
pub struct DenseBitVec<Ones, Raw>
where
    // Type of the 1-bits stored in this BitVec
    Ones: BitBlock,
    // Block type for the underlying storage BitBuf
    Raw: BitBlock,
{
    raw: BitBuf<Raw>, // bit data
    sr_pow2: Ones,    // power of 2 of the rank sampling rate
    ss_pow2: Ones,    // power of 2 of the select sampling rate
    r: Box<[Ones]>,   // rank samples holding the number of preceding 1-bits
    s0: Box<[Ones]>,  // select0 samples
    s1: Box<[Ones]>,  // select1 samples
    num_ones: Ones,
}

impl<Ones: BitBlock, Raw: BitBlock> DenseBitVec<Ones, Raw> {
    pub fn new(data: BitBuf<Raw>, sr_pow2: Ones, ss_pow2: Ones) -> Self {
        let raw = data;
        let raw_block_bits = Ones::from_u32(Raw::BITS);
        let raw_block_pow2 = Ones::from_u32(Raw::BIT_WIDTH);
        debug_assert!(sr_pow2 >= raw_block_pow2);
        debug_assert!(ss_pow2 >= raw_block_pow2);

        let ss: Ones = Ones::one() << ss_pow2; // Select sampling rate: sample every `ss` 1-bits
        let sr: Ones = Ones::one() << sr_pow2; // Rank sampling rate: sample every `sr` bits

        let mut r = vec![]; // rank samples
        let mut s0 = vec![]; // select0 samples
        let mut s1 = vec![]; // select1 samples

        let mut cumulative_ones = Ones::zero(); // 1-bits preceding the current raw block
        let mut cumulative_bits = Ones::zero(); // bits preceding the current raw block
        let mut zeros_threshold = Ones::zero(); // take a select0 sample at the (zeros_threshold+1)th 1-bit
        let mut ones_threshold = Ones::zero(); // take a select1 sample at the (ones_threshold+1)th 1-bit

        // Raw blocks per rank block
        let raw_block_sr = sr >> raw_block_pow2;

        // Iterate one rank block at a time for convenient rank sampling
        for blocks in raw.blocks().chunks(raw_block_sr.usize()) {
            r.push(cumulative_ones); // Take a rank sample
            for block in blocks.iter() {
                let block_ones = Ones::from_u32(block.count_ones());
                if cumulative_ones + block_ones > ones_threshold {
                    // Take a select sample, which consists of two parts:
                    // 1. The cumulative bits preceding this raw block
                    let high = cumulative_bits;
                    // 2. The number of 1-bits before the (ss * i + 1)-th 1-bit within this raw block
                    let low = ones_threshold - cumulative_ones;
                    // High is a multiple of the raw block size so these
                    // two values should never overlap in their bit ranges.
                    debug_assert!((high & low).is_zero());
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
                    debug_assert!((high & low).is_zero());
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
            sr_pow2,
            ss_pow2,
            r: r.into_boxed_slice(),
            s0: s0.into_boxed_slice(),
            s1: s1.into_boxed_slice(),
            // num_ones is always <= u32::MAX by construction
            num_ones: cumulative_ones,
        }
    }

    /// Return a select1 block index for a given 1-bit index
    fn s_index(&self, n: Ones) -> usize {
        (n >> self.ss_pow2).usize()
    }

    /// Return a rank block index for a given bit index
    fn r_index(&self, index: Ones) -> usize {
        (index >> self.sr_pow2).usize()
    }

    /// Return a rank-block-aligned bit index for a given bit index
    fn rank_aligned_bit_index(&self, index: Ones) -> Ones {
        Ones::from_usize(self.r_index(index) << self.sr_pow2.as_usize())
    }

    fn select_sample(s: &[Ones], ss_pow2: Ones, n: Ones) -> (Ones, Ones) {
        // Select samples are sampled every j*2^ss_pow2 1-bits and stores
        // a value related to the bit position of the 2^ss_pow2-th bit.
        // For improved performance, rather than storing the position of
        // that bit directly, each select sample holds two separate values:
        // 1. The raw-block-aligned bit position of that bit, ie. the number
        // of bits preceding the raw block containing the 2^ss-pow2_th bit.
        // 2. The bit position of the (ss * i + 1)-th 1-bit within that raw block,
        //    which we can subtract from j*2^ss_pow2 to tell the number of 1-bits
        //    up to the raw-block-aligned bit position.
        let sample_index = n >> ss_pow2;
        let sample = s[sample_index.usize()];

        // bitmask with the Raw::BIT_WIDTH bottom bits set.
        let mask = Ones::from_u32(Raw::BITS - 1);
        let bit_pos = sample & !mask;
        let correction = sample & mask;

        // assert that bit pos is Raw-aligned
        debug_assert!(Raw::bit_offset(bit_pos.usize()) == 0);

        // num. of ones represented by this sample, up to the raw block boundary
        let num_ones = (sample_index << ss_pow2) - correction;

        (bit_pos, num_ones)
    }
}

impl<Ones: BitBlock, Raw: BitBlock> BitVec for DenseBitVec<Ones, Raw> {
    type Ones = Ones;

    fn rank1(&self, index: Ones) -> Ones {
        if index >= self.len() {
            return self.num_ones();
        }

        // Start with the prefix count from the rank block
        let mut rank = self.r[self.r_index(index)];
        let index_usize = index.usize();

        // use select blocks to increase raw_start by hopping through select blocks

        // Identify the largest select block pointing at or before the rank in the rank block
        // let (mut bit_pos, mut preceding_ones) = Self::select_sample(&self.s1, self.ss_pow2, n);
        let mask = Ones::from_u32(Raw::BITS - 1);

        let s_start = rank >> self.ss_pow2;
        let s_blocks = self.s1[self.s1.len().min(s_start.usize() + 1)..]
            .iter()
            .copied();
        let s_block = s_blocks
            .take_while(|&x| {
                // raw-block-aligned bit position represented by this select block
                let bit_pos = x & !mask;
                bit_pos <= index
            })
            .enumerate()
            .last();
        let raw_start = if let Some((i, sample)) = s_block {
            let i: Ones = Ones::from_usize(i);
            let bit_pos = sample & !mask;
            let correction = sample & mask;
            // the new rank is based on the select block index,
            // since we sample selects based on the rank.
            let select_block_index = s_start + i + Ones::one();
            rank = (select_block_index << self.ss_pow2) - correction;
            // the new raw block is based on the position of the 1-bit designated by the select block.
            Raw::block_index(bit_pos.usize())
        } else {
            Raw::block_index(self.rank_aligned_bit_index(index).usize())
        };

        // // self.s[select_index] points somewhere before index
        // let select_index = self.select_index(rank);

        // Sequentially scan raw blocks from raw_start onwards
        let raw_end = Raw::block_index(index_usize);
        let raw_slice = &self.raw.blocks()[raw_start..=raw_end];
        if let Some((last_block, blocks)) = raw_slice.split_last() {
            // Add the ones in fully-covered raw blocks
            for block in blocks {
                rank += Ones::from_u32(block.count_ones())
            }

            // Add any ones in the final partly-covered raw block
            let raw_bit_offset = Raw::bit_offset(index_usize);
            if raw_bit_offset > 0 {
                let mask: Raw = Raw::one_mask(raw_bit_offset as u32);
                rank += Ones::from_u32((*last_block & mask).count_ones());
            }
        }

        rank
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_ones {
            return None;
        }

        // Use the select block for an initial position
        let (mut bit_pos, mut preceding_ones) = Self::select_sample(&self.s1, self.ss_pow2, n);
        // Use rank blocks to hop over many raw blocks
        let r_start = self.r_index(bit_pos); // index of the preceding rank block
        let r_blocks = self.r[r_start + 1..].iter().copied();
        let r_block = r_blocks.take_while(|&x| x < n).enumerate().last();
        if let Some((i, ones)) = r_block {
            let r_start: Ones = Ones::from_usize(r_start);
            let i: Ones = Ones::from_usize(i);
            bit_pos = (r_start + i + Ones::one()) << self.sr_pow2; // bit pos depends on the index of the rank block
            preceding_ones = ones;
        }

        // we want the next raw block value and index, and the preceding one count up to that block.
        let mut cur_ones = preceding_ones;
        let raw_start = Raw::block_index(bit_pos.usize());
        let raw_blocks = self.raw.blocks()[raw_start..].iter().copied();
        let (count, block) = raw_blocks
            .enumerate()
            .find(|(_, block)| {
                preceding_ones = cur_ones;
                cur_ones += Ones::from_u32(block.count_ones());
                cur_ones > n
            })
            .unwrap();

        // clear the bottom 1-bits
        let shift = Raw::BIT_WIDTH as usize;
        let block_bits = Ones::from_usize((raw_start + count) << shift);
        let bit_offset = Ones::from_u32(select1(block, (n - preceding_ones).as_u32()));
        Some(block_bits + bit_offset)
    }

    // todo: use a common abstraction for select0 and select1.
    // right now this doesn't implement the optimization that hops over rank blocks.
    fn select0(&self, n: Ones) -> Option<Ones> {
        if n >= self.num_zeros() {
            return None;
        }
        let sample_index = self.s_index(n);
        let sample = self.s0[sample_index];

        let (raw_start, correction) = Raw::index_offset(sample.usize());
        let select_zeros = Ones::from_usize(sample_index) << self.ss_pow2; // num. of zeros represented by this sample
        let mut prev_zeros = select_zeros - Ones::from_usize(correction);
        let mut cur_zeros = prev_zeros;
        let raw_blocks = self.raw.blocks()[raw_start..].iter().copied();
        let (count, mut block) = raw_blocks
            .enumerate()
            .find(|(_, block)| {
                prev_zeros = cur_zeros;
                cur_zeros += Ones::from_u32(block.count_zeros());
                cur_zeros > n
            })
            .unwrap();
        let shift = Raw::BIT_WIDTH as usize;

        block = !block;
        for _ in num::iter::range(prev_zeros, n) {
            block &= block - Raw::one(); // unset extra zeros
        }
        block = !block;

        let block_bits = (raw_start + count) << shift;
        let bit_offset = block.trailing_ones();
        Some(Ones::from_usize(block_bits) + Ones::from_u32(bit_offset))
    }

    fn num_ones(&self) -> Ones {
        self.num_ones
    }

    fn len(&self) -> Ones {
        Ones::from_usize(self.raw.len())
    }

    fn get(&self, index: Ones) -> bool {
        self.raw.get(index.usize())
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
        type Ones = u32;
        type Raw = u8;
        let raw = BitBuf::new(100);
        let _ = DenseBitVec::<Ones, Raw>::new(raw, Raw::BIT_WIDTH, Raw::BIT_WIDTH);
    }

    #[test]
    fn test_bitvector() {
        let f = |ones: &[u32], len: u32| {
            type Raw = u8;
            let mut raw = BitBuf::<Raw>::new(len.try_into().unwrap());
            for one in ones.iter().copied() {
                raw.set(one.try_into().unwrap());
            }
            DenseBitVec::<u32, Raw>::new(raw, Raw::BIT_WIDTH, Raw::BIT_WIDTH)
        };
        bit_vec::test_bitvector(f);
        bit_vec::test_bitvector_vs_naive(f);
    }

    #[test]
    fn test_ranks() {
        // todo: rewrite into a more compact and less arbitrary test case
        let ones = [1, 2, 5, 10, 32];
        let mut raw = BitBuf::<u8>::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        type Raw = u8;
        let bv = DenseBitVec::new(raw, Raw::BIT_WIDTH, Raw::BIT_WIDTH);
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
        let mut raw = BitBuf::<u8>::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        type Raw = u8;
        let bv = DenseBitVec::new(raw, Raw::BIT_WIDTH, Raw::BIT_WIDTH);
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
        let mut raw = BitBuf::<u8>::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        type Raw = u8;
        let bv = DenseBitVec::new(raw, Raw::BIT_WIDTH, Raw::BIT_WIDTH);
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
        type Ones = u32;
        type Raw = u8;
        let n_iters = 100;
        for _ in 1..n_iters {
            let mut ones = vec![];
            let mut rng = rand::thread_rng();
            let mut prev = 0;
            for _ in 1..100 {
                let one = prev + rng.gen_range(1..3 * Raw::BITS) as Ones;
                prev = one;
                ones.push(one);
            }

            // let ones = vec![  9, 25, 61, 76, 96, 134, 163, 187, 265];
            let mut raw = BitBuf::<u8>::new((ones.iter().max().unwrap() + 1).usize());
            for o in ones.iter().copied() {
                raw.set(o.usize());
            }

            println!("ones {:?}", ones);
            let bv = DenseBitVec::new(raw, Raw::BIT_WIDTH, Raw::BIT_WIDTH);
            for (i, o) in ones.iter().copied().enumerate() {
                println!("testing index {:?} with one  {:?}", i, o);
                assert_eq!(bv.select1(i as Ones), Some(o));
            }
            assert_eq!(bv.select1(ones.len().try_into().unwrap()), None);
            assert_eq!(bv.select1((2 * ones.len()).try_into().unwrap()), None);
        }
    }

    // test todo:
    // - zero-length bitvector
    // - rank at around the exact length/block boundary of the bitvec
}
