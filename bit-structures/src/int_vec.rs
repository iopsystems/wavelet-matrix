// Fixed-width unsigned integer vector that bit-packs fixed-width (<= 64 bits) integers into a u64 array.

use crate::bit_block::BitBlock;
use crate::utils::div_ceil;
use std::debug_assert;

type BT = u32; // Block type

pub struct IntVec {
    data: Box<[BT]>,
    /// Number of elements
    len: usize,
    /// Bits per number
    bit_width: usize,
    /// Points to the next available bit index
    write_cursor: usize,
}

impl IntVec {
    // todo: test with zero bit width
    pub fn new(len: usize, bit_width: usize) -> Self {
        assert!(bit_width <= BT::BITS.try_into().unwrap());
        // The number of blocks should be just enough to represent `len * bit_width` bits.
        let num_blocks = div_ceil(len * bit_width, BT::BITS as usize);
        // Initialize to zero so that any trailing bits in the last block will be zero.
        let data = vec![0; num_blocks].into_boxed_slice();
        Self {
            data,
            len,
            bit_width,
            write_cursor: 0,
        }
    }

    pub fn write_int(&mut self, value: BT) {
        debug_assert!(
            value < (1 << self.bit_width),
            "int value cannot exceed the maximum value representable by the bit width"
        );

        // If we have zero bit width, only allow writing zeros (and there's no need to write them!)
        if self.bit_width == 0 {
            assert!(value == 0);
            return;
        }

        let index = BT::block_index(self.write_cursor);
        let offset = BT::bit_offset(self.write_cursor);
        self.data[index] |= value << offset;

        // Number of available bits in the target block
        let num_available_bits = BT::BITS as usize - offset;

        // If needed, write the remaining bits to the next block
        if num_available_bits < self.bit_width {
            self.data[index + 1] = value >> num_available_bits;
        }
        self.write_cursor += self.bit_width;
    }

    pub fn get(&self, index: usize) -> BT {
        assert!(index < self.len);

        // If we have zero bit width, our vector is entirely full of zeros.
        if self.bit_width == 0 {
            return 0;
        }

        let bit_index = index * self.bit_width;
        let block_index = BT::block_index(bit_index);
        let offset = BT::bit_offset(bit_index);

        // Low bit mask with a number of ones equal to self.bit_width.
        let mask = u32::one_mask(self.bit_width as u32);

        // Extract the bits the value that are present in the target block
        let mut value = (self.data[block_index] & (mask << offset)) >> offset;

        // Number of available bits in the target block
        let num_available_bits = BT::BITS as usize - offset;

        // If needed, extract the remaining bits from the bottom of the next block
        if num_available_bits < self.bit_width {
            let num_remaining_bits = self.bit_width - num_available_bits;
            let high_bits = self.data[block_index + 1] & u32::one_mask(num_remaining_bits as u32);
            value |= high_bits << num_available_bits;
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed() {
        let seq = [5, 6, 2, 0, 7, 1, 11, 5, 10, 10, 2, 3, 4, 10, 20, 100, 5];
        let max = seq.iter().max().copied().unwrap();
        let n_bits = (max as f64).log2().ceil() as usize;
        let mut bv = IntVec::new(100, n_bits);
        for n in seq {
            bv.write_int(n);
        }
        for (index, n) in seq.iter().copied().enumerate() {
            assert_eq!(bv.get(index), n);
        }
    }

    #[test]
    #[should_panic]
    fn test_any_panic() {
        let mut bv = IntVec::new(100, 3);
        bv.write_int(8);
    }

    // todo
    // - test that you cannot write more ints than you have length
    // - test that 64-bit blocks (and other bit widths) all work
    // - test sequences with repeating elements, etc.
    // - test multi-block sequences
    // - test zero-bit-width entries
}
