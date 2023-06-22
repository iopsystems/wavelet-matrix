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
//     - a trim() method on RawBitVec? then we can't set beyond the trimmed after
//     - a builder that preallocates the full thing and trims on build?
//     - a builder that reallocates as you go?
//     - or if the ones are sorted, can just look at the first and last.
// - [same idea, refined] implement a "default value" that we can compress out of the top and bottom – or maybe always zero.
//   ie., store only the middle section and return the default value outside it.

// todo:
// - does this need to use Ones as the input to eg. .get?
//   - we can't actually store more than usize::MAX elements in a slice...

#[derive(Debug)]
pub struct BitBuf<Block: BitBlock = u8> {
    blocks: Box<[Block]>,
    len: usize,
}

impl<Block: BitBlock> BitBuf<Block> {
    pub fn new(len: usize) -> Self {
        // The number of blocks should be just enough to represent `len` bits.
        let num_blocks = div_ceil(len, Block::BITS as usize);
        // Initialize to zero so that any trailing bits in the last block will be zero.
        let data = vec![Block::zero(); num_blocks].into_boxed_slice();
        Self { blocks: data, len }
    }

    /// Return the bool value of the bit at index `index`
    pub fn get(&self, index: usize) -> bool {
        let block = self.blocks[Block::block_index(index)];
        // let rhs = (B::one() << B::bit_offset(index));
        let bit = block & (Block::one() << Block::bit_offset(index)); // (1 << B::bit_offset(index));
        bit != Block::zero()
    }

    /// Write a 1-bit to index `index`.
    // Since the data buffer is initialized to its final size at construction time
    // bits may be set in any order.
    pub fn set(&mut self, index: usize) {
        let block_index = Block::block_index(index);
        let block = self.blocks[block_index];
        let set_bit = Block::one() << Block::bit_offset(index);
        self.blocks[block_index] = block | set_bit;
    }

    /// Return an immutable reference to the underlying data as a slice
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Bitvector length in bits.
    pub fn len(&self) -> usize {
        self.len
    }
}
