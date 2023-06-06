// Dense bit vector with rank and select, using the technique described
// in the paper "Fast, Small, Simple Rank/Select on Bitmaps"
// We should find a way to make the rank/select structures optional to save space?
// Or maybe just set the configurable sampling parameters such that we basically
// sample only one element (or zero).
// That can extend to the sparse one, which uses a dense one underneath.
// We use u32 to represent sampled rank and select blocks, giving us the ability to represent
// bitvectors roughly 2^32 in length.
//
// note that wasm simd has 128-bit bitselect instruction, and also popcnt (which is effectively rank);
// use them for a simd-accelerated version of the library (ie. 128-bit basic blocks, and use bitselect
// for select correction in the final block).
// Of course, we should benchmark to see.
//
// TODO: rename this file to whatever the struct ends up being named
// https://observablehq.com/d/2654fee8107c15ab#SimpleSelectBitVec

use std::debug_assert;

use crate::raw_bitvector::RawBitVec;

type BT = u32; // Block type. TODO: Should we distinguish between types for rank vs. select blocks?

#[derive(Debug)]
pub struct DenseBitVec {
    data: RawBitVec,
    // Regularly spaced rank samples
    // Stores a sampling of rank answers from the beginning of B up to the end of each block: r[i] = rank1(B, i · Sr)
    r: Box<[BT]>,
    // Regularly spaced select samples (containing an equal number of ones, variable length).
    // We ensure that each block starts with a 1, to minimize unnecessary scanning.
    // Thus sampleSelect[j] = select1(B, j · Ss + 1) (for now)
    s: Box<[BT]>,
}

// todo: document that you can 'turn off' select or rank acceleration by setting
// the sr/ss values to very low (maybe should be options)

// todo: rank will use select indices to speed up but select won't use rank (maybe),
// unless it gives a speedup. How do we say which selects we want indexing for? ss0, ss1?
// A builder step of some sort?
// todo: consider not storing the prefix or suffix of zeros in the dense bitvector, keeping them notional
// ie. start_offset
impl DenseBitVec {
    // todo: accept a raw vector and iterate through its ones (popcount-accelerated)
    // sr_bits: power of 2 of rank sampling rate (in terms of ones)
    // ss_bits: power of 2 of select sampling rate (in terms of universe size)
    fn new(data: RawBitVec, sr_bits: u32, ss_bits: u32) -> Self {
        // todo: debug assert bounds on the sample rates
        // Assert that the sample rates must each exceed a single raw block
        debug_assert!(sr_bits >= RawBitVec::block_bits().ilog2());
        debug_assert!(ss_bits >= RawBitVec::block_bits().ilog2());

        // Select sampling rate (in terms of ones)
        let ss = 1 << ss_bits;
        // Rank sampling rate (in terms of universe size)
        let sr = 1 << sr_bits;

        // Number of raw bitvector blocks per rank block
        let raw_block_sr = sr >> RawBitVec::block_bits().ilog2();

        let mut r = vec![]; // rank samples
        let mut s = vec![]; // select samples // todo: call this s1 in case we also want s0

        // we don't need to do the trailing zeros thing here (todo: make it a utility function)
        // todo: we want s to point to the RawBitVec block containing the sampled 1-bit, plus correction information storing how many 1 bits come before it.

        // Number of ones seen in the raw bitvector so far
        let mut cumulative_ones = 0;
        // Track the number of bits processed so far
        let mut cumulative_bits = 0;

        // When we've seen more than ss * i 1-bits, collect a select sample.
        // This is a moving threshold that increments by ss each time we collect a sample.
        let mut select_threshold = 0;

        // Iterate raw bitvector blocks (also called "raw blocks") in chunks corresponding to rank blocks
        for blocks in data.blocks().chunks(raw_block_sr) {
            // The rank block r[i] is equal to the number of ones up to but not including the (sr * i)-th bit.
            r.push(cumulative_ones);

            // Iterate through raw blocks to accumulate the 1-bit count and add select samples
            for block in blocks.iter().copied() {
                let block_ones = block.count_ones();
                if cumulative_ones + block_ones >= select_threshold {
                    // Add a new select sample if we've passed the select threshold.

                    // Each select block encodes two numbers that together allow us to compute the
                    // position of the the bit of interest, the (ss*i + 1)-th 1-bit:
                    // 1. The number of bits preceding the raw block that holds the bit of interest.
                    //    This allows us to quickly navigate to that block-aligned index.
                    //    Because this is a multiple of block_bits (which is a power of two),
                    //    the lowest `log2(block_bits)` bits are zero, and used to store the
                    //    bit offset, which is a number in [0, block_bits).
                    //    The raw block index can be accessed as:
                    //    select_sample >> RawBitVec::block_bits().
                    let a = cumulative_bits;

                    // 2. The bit offset inside that block of the bit of interest.
                    //    This is the number of 1-bits in this block that precede
                    //    the (j*sr + 1)-th bit.
                    //    The number of preceding bits in the raw block can be accessed as:
                    //    select_sample & one_mask(RawBitVec::block_bits()).
                    let b = select_threshold - cumulative_ones;

                    // This differs from simply storing the position of the bit of interest,
                    // because the low bits are used to store not the offset of that bit within
                    // the block, but rather the number of 1-bits preceding it in the block.
                    //
                    // You can think of the select sample as pointing to the start of the raw block
                    // containing the bit of interest (1), along with correction information that
                    // tells us the exact rank up to that position.
                    //
                    // This allows the select block to tell us two things at once:
                    // - The rank up to the preceding block
                    // - The position of the bit of interest, which can be computed quickly
                    //   using some bit math to un-set the extra bits followed by a call
                    //   to trailing_zeroes().

                    // Assert that the the two numbers do not overlap.
                    debug_assert!(a & b == 0);

                    // Append the select sample and bump the select threshold
                    s.push(a + b);
                    select_threshold += ss;
                }
                cumulative_ones += block_ones;
                cumulative_bits += RawBitVec::block_bits();
            }
        }

        Self {
            data,
            r: r.into_boxed_slice(),
            s: s.into_boxed_slice(),
        }
    }

    /// Return the number of 1-bits at or below index `i`
    fn rank1(&self, _i: usize) -> usize {
        todo!()
    }

    /// Return the number of 0-bits at or below index `i`
    fn rank0(&self, _i: usize) -> usize {
        // if index >= self.len {
        //     return self.len - self.num_ones; // or return self.num_zeros;
        // }
        // index - self.rank1(i) + 1
        todo!()
    }

    /// Return an option with the index of `i`-h 1-bit if one exists
    fn select1(&self, i: usize) -> Option<usize> {
        if i >= self.data.len() {
            return None;
        }
        todo!()
    }

    /// Return an option with the index of `i`-h 0-bit if one exists
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
        let mut raw = RawBitVec::new(ones.iter().max().unwrap() + 1);
        for i in ones {
            raw.set(i);
        }
        let bv = DenseBitVec::new(raw, 5, 5);
        // dbg!(bv);
        for b in bv.data.blocks() {
            println!("{:b}", b);
        }
        panic!("aaa");
    }

    // todo
    // - test that you cannot write more ints than you have length
    // - test that 64-bit blocks (and other bit widths) all work
    // - test sequences with repeating elements, etc.
    // - test multi-block sequences
}
