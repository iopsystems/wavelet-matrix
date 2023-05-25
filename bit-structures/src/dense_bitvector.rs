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

use crate::raw_bitvector::RawBitVector;
use crate::utils::BitBlock;

type BT = u32; // Block type. TODO: Should we distinguish between types for rank vs. select blocks?

pub struct RankSelectSupport {
    data: RawBitVector,
    // Regularly spaced rank samples
    // r: Box<[BT]>,
    // Regularly spaced select samples
    // s: Box<[BT]>,
}

// todo: rank will use select indices to speed up but select won't use rank (maybe),
// unless it gives a speedup. How do we say which selects we want indexing for? ss0, ss1?
// A builder step of some sort?
impl RankSelectSupport {
    fn new(data: RawBitVector, _sr: usize, _ss: usize) -> Self {
        debug_assert!(
            data.len() < 1 << BT::bits(),
            "length cannot exceed the maximum representable rank/select block value"
        );
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
