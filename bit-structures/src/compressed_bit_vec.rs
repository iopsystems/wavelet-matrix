/*
// Inspired by Roaring and BitMagic.
// Unlike Roaring, we store all containers, including empty ones, for faster rank queries.
// https://arxiv.org/pdf/1709.07821.pdf

// Store containers in a dense array. Each represents 2^16 elements.

use crate::dense_bit_vec::DenseBitVec;
use std::ops::Range;

#[derive(Debug)]
enum Container {
    Runs(Box<[Range<u16>]>), // Ranges representing runs of ones: start..start+length. We could actually use our RLE bitvector here.
    Sparse(Box<[u16]>), // lowest 16 bits of one positions. We could actually use our naïve bitvector here, if we parameterize the element type.
    Dense(Box<DenseBitVec<u32>>), // dense bit representation. We use our dense bitvector.
}

// Implement rank and select on each conatiner by delegating to its inner type in a match statement.
// Then for rank and select on the compressed bitvector, adjust based on the preceding containers.

struct CompressedBitVec {
    containers: Box<[Container]>,
    // todo: u32 rank blocks with the preceding count instead? still small overhead vs.
    // todo: can store a final rank block containing the sum total for the whole bitvector too,
    //       if it makes certain things easier (eg. get the count per block by subtracting cur from next)
    counts: Box<[u16]>,
    len: usize,
}

impl CompressedBitVec {
    fn new(ones: &[usize], len: usize) -> Self {
        for _one in ones {}
        let _ = len;

        // To decide how to compress each working set, compute the number of bits in each representation.
        let containers = vec![];
        let counts = vec![];
        Self {
            containers: containers.into_boxed_slice(),
            counts: counts.into_boxed_slice(),
            len,
        }
    }
    // rank1:
    // select1: binary search high blocks using partition_point, then
}

*/
