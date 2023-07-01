// A plain fixed-size bitvector with no acceleration structures, backed by an array of integers.
// Supports random bit read and write. Intended as a data representation for dense bitvectors.
// Not designed for general fixed-width encoding; we can use the bitbuffer library for that.

use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::utils::div_ceil;
use get_size::GetSize;

#[derive(Debug, Clone, get_size_derive::GetSize)]
pub struct BitBuf<Block: BitBlock> {
    blocks: Box<[Block]>,
    universe_size: usize,
}

impl<Block: BitBlock> bincode::Encode for BitBuf<Block> {
    encode_impl!(blocks, universe_size);
}
impl<Block: BitBlock> bincode::Decode for BitBuf<Block> {
    decode_impl!(blocks, universe_size);
}
impl<'de, Block: BitBlock> bincode::BorrowDecode<'de> for BitBuf<Block> {
    borrow_decode_impl!(blocks, universe_size);
}

impl<Block: BitBlock> BitBuf<Block> {
    pub fn new(universe_size: usize) -> Self {
        // The number of blocks should be just enough to represent `universe_size` bits.
        let num_blocks = div_ceil(universe_size, Block::BITS as usize);
        // Initialize to zero so that any trailing bits in the last block will be zero.
        let blocks: Box<[Block]> = vec![Block::zero(); num_blocks].into();

        log::info!("bitbuf blocks: {:?}", blocks.get_size());

        Self {
            blocks,
            universe_size,
        }
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
    pub fn universe_size(&self) -> usize {
        self.universe_size
    }
}
