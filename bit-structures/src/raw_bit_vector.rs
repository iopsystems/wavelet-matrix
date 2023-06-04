// A plain fixed-size bitvector with no acceleration structures, backed by an array of integers.
// Supports random bit read and write. Intended as a data representation for dense bitvectors.
// Not designed for general fixed-width encoding; we can use the bitbuffer library for that.
// todo:
// - customizable backing integer type (feels useful to be able to change the block type)
//   - implement T.bits() for u16, u32, u64;
//   - see simple-sds split_at_index for how they handle it; maybe we want a T.mask() also for offset masking
//   - see also my similar block_index_and_offset
// - add as many debug_assert! s as is reasonable to do

use crate::bit_block::BitBlock;
use crate::utils::div_ceil;

// todo:
// - add a builder so that we can RLE the start and end of the bitvector, only storing the middle chunk.
//   - not entirely sure how best to do this
//     - a trim() method on RawBitVector? then we can't set beyond the trimmed after
//     - a builder that preallocates the full thing and trims on build?
//     - a builder that reallocates as you go?

// todo:
// - is there an elegant way to generalize this to arbitrary unsigned types?
// - is there a specific choice that works well with wasm and wasm simd?
// - current thoughts: we can use the BitBlock trait to centralize common functionality
//   across block types, and type aliases to fix the individual block types for each
//   individual bitvector type and block.
pub type BT = u32; // Block type

// Raw bits represented in an array of integer blocks.
// Immutable once constructed
#[derive(Debug)]
pub struct RawBitVector {
    blocks: Box<[BT]>,
    len: usize,
}

impl RawBitVector {
    pub fn new(len: usize) -> Self {
        // The number of blocks should be just enough to represent `len` bits.
        let num_blocks = div_ceil(len, BT::bits() as usize);
        // Initialize to zero so that any trailing bits in the last block will be zero.
        let data = vec![0; num_blocks].into_boxed_slice();
        Self { blocks: data, len }
    }

    /// Return the bool value of the bit at index `i`
    pub fn get(&self, i: usize) -> bool {
        let block = self.blocks[BT::block_index(i)];
        let bit = block & (1 << BT::bit_offset(i));
        bit != 0
    }

    /// Write a 1-bit to index `i`.
    // Since the data buffer is initialized to its final size at construction time
    // bits may be set in any order.
    pub fn set(&mut self, i: usize) {
        self.blocks[BT::block_index(i)] |= 1 << BT::bit_offset(i);
    }

    /// Return an immutable reference to the underlying data as a slice
    pub fn blocks(&self) -> &[BT] {
        &self.blocks
    }

    /// Bitvector length in bits.
    pub fn len(&self) -> usize {
        self.len
    }

    // todo: is there a way to not have to re-export these functions here, and instead
    // provide  access to an associated type or something?

    /// Number of bits in a raw block.
    pub fn block_bits(&self) -> u32 {
        BT::bits()
    }

    pub fn block_index(&self, i: usize) -> usize {
        BT::block_index(i)
    }

    pub fn bit_offset(&self, i: usize) -> usize {
        BT::bit_offset(i)
    }

    pub fn bit_split(&self, i: usize) -> (usize, usize) {
        BT::bit_split(i)
    }
}
