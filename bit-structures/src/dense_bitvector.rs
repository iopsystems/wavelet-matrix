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

use std::debug_assert;

use crate::raw_bitvector::{self, RawBitVector};
use crate::utils::BitBlock;

type BT = u32; // Block type. TODO: Should we distinguish between types for rank vs. select blocks?

pub struct DenseBitVector {
    data: RawBitVector,
    // Regularly spaced rank samples
    // Stores a sampling of rank answers from the beginning of B up to the end of each block: r[i] = rank1(B, i · Sr)
    // r: Box<[BT]>,
    // Regularly spaced select samples (containing an equal number of ones, variable length).
    // We ensure that each block starts with a 1, to minimize unnecessary scanning.
    // Thus sampleSelect[j] = select1(B, j · Ss + 1) (for now)
    // s: Box<[BT]>,
}

// todo: rank will use select indices to speed up but select won't use rank (maybe),
// unless it gives a speedup. How do we say which selects we want indexing for? ss0, ss1?
// A builder step of some sort?
// todo: consider not storing the prefix or suffix of zeros in the dense bitvector, keeping them notional
// ie. start_offset
impl DenseBitVector {
    // todo: accept a raw vector and iterate through its ones (popcount-accelerated)
    fn new(data: RawBitVector, rank_sr_pow2: usize, select_sr_pow2: usize) -> Self {
        // todo: debug assert bounds on the sample rates
        let select_sr = 1 << select_sr_pow2;
        let rank_sr = 1 << rank_sr_pow2;
        let rank_block_sr = rank_sr >> raw_bitvector::BT::bits();
        debug_assert!(
            rank_block_sr > 0,
            "rank sample rate must be at least the size of a single raw bitvector block"
        );

        let mut r = vec![]; // rank samples
        let mut s = vec![]; // select samples

        // todo: we want s to point to the RawBitVector block containing the sampled 1-bit, plus correction information storing how many 1 bits come before it.

        let mut cumulative_ones = 0;
        let mut select_boundary = 1;
        for blocks in data.blocks().chunks(rank_block_sr) {
            r.push(cumulative_ones);
            for mut block in blocks.iter().copied() {
                let block_ones = block.count_ones();
                cumulative_ones += block_ones;

                // we don't need to do the trailing zeros thing here (todo: make it a utility function)

                if cumulative_ones >= select_boundary {
                    // k is the number of ones in this block preceding the selected one
                    let _k = cumulative_ones - select_boundary;
                    let _mask = (1 << raw_bitvector::BT::bits()) - 1;
                    // We want to store ... tbc
                    // raw_bitvector::BT::bits();

                    // we want to record some 1-bit inside this block as a select sample,
                    // but we don't yet know its precise position within the block.
                    // we need to un-set k bits before counting the position of the next bit as the select sample
                    let k = cumulative_ones - select_boundary;
                    for _ in 0..k {
                        block &= block - 1;
                    }
                    debug_assert!(block > 0); // assert that we have not un-set all available ones
                    let select_sample = r.len() * rank_sr + block.trailing_zeros() as usize;
                    s.push(select_sample);
                    select_boundary += select_sr;
                }
            }
        }

        // todo: document that you can 'turn off' select or rank acceleration by setting
        // the sr/ss values to very low (maybe should be options)
        // debug_assert!(
        //     data.len() < 1 << BT::bits(),
        //     "length cannot exceed the maximum representable rank/select block value"
        // );
        // sr: power of 2 of rank sampling rate
        // ss: power of 2 of select sampling rate
        // https://observablehq.com/d/2654fee8107c15ab#SimpleSelectBitVector
        Self { data }
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
