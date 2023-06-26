// A plain fixed-size bitvector with no acceleration structures, backed by an array of integers.
// Supports random bit read and write. Intended as a data representation for dense bitvectors.
// Not designed for general fixed-width encoding; we can use the bitbuffer library for that.

use crate::bit_block::BitBlock;
use crate::utils::div_ceil;

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
        let blocks = vec![Block::zero(); num_blocks].into();
        Self { blocks, len }
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
