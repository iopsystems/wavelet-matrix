// A plain fixed-size bitvector with no acceleration structures, backed by an array of integers.
// Supports random bit read and write. Intended as a data representation for dense bitvectors.
// Not designed for general fixed-width encoding; we can use the bitbuffer library for that.
// todo:
// - customizable backing integer type (feels useful to be able to change the block type)
//   - implement T.bits() for u16, u32, u64;
//   - see simple-sds split_at_index for how they handle it; maybe we want a T.mask() also for offset masking
//   - see also my similar block_index_and_offset
// - add as many debug_assert! s as is reasonable to do

use crate::utils::{div_ceil, BitBlock};

// todo:
// - is there an elegant way to generalize this to arbitrary unsigned types?
// - is there a specific choice that works well with wasm and wasm simd?
// - current thoughts: we can use the BitBlock trait to centralize common functionality
//   across block types, and type aliases to fix the individual block types for each
//   individual bitvector type and block.
pub type BT = u64; // Block type

// Raw bits represented in an array of integer blocks.
// Immutable once constructed
pub struct RawBitVector {
    blocks: Box<[BT]>,
    len: usize,
}

impl RawBitVector {
    /// Return the bool value of the bit at index `i`
    pub fn get(&mut self, i: usize) -> bool {
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

    /// Return the length in bits
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn new(len: usize) -> Self {
        // The number of blocks should be just enough to represent `len` bits.
        let num_blocks = div_ceil(len, BT::bits() as usize);
        // Initialize to zero so that any trailing bits in the last block will be zero.
        let data = vec![0; num_blocks].into_boxed_slice();
        Self { blocks: data, len }
    }
}
