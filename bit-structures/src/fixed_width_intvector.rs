// Fixed-width unsigned integer vector that bit-packs fixed-width (<= 64 bits) integers into a u64 array.

use std::debug_assert;

use crate::utils::{div_ceil, BitBlock};

type BT = u32; // Block type

pub struct FixedWidthIntVector {
    data: Box<[BT]>,
    /// Number of elements
    len: usize,
    /// Bits per number
    bit_width: usize,
    /// Points to the next available bit index
    write_cursor: usize,
}

impl FixedWidthIntVector {
    pub fn new(len: usize, bit_width: usize) -> Self {
        assert!(bit_width <= BT::bits().try_into().unwrap());
        // The number of blocks should be just enough to represent `len * bit_width` bits.
        let num_blocks = div_ceil(len * bit_width, BT::bits() as usize);
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
        let index = BT::block_index(self.write_cursor);
        let offset = BT::bit_offset(self.write_cursor);
        self.data[index] |= value << offset;

        // Number of available bits in the target block
        let num_available_bits = BT::bits() as usize - offset;

        // If needed, write the remaining bits to the next block
        if num_available_bits < self.bit_width {
            self.data[index + 1] = value >> num_available_bits;
        }
        self.write_cursor += self.bit_width;
    }

    pub fn read_int(&mut self, i: usize) -> BT {
        let bit_index = i * self.bit_width;
        let block_index = BT::block_index(bit_index);
        let offset = BT::bit_offset(bit_index);

        // Low bit mask with a number of ones equal to self.bit_width.
        // Eg. if bit_width is 3, then mask is 0b111.
        let mask = (1 << self.bit_width) - 1;

        // Extract the bits the value that are present in the target block
        let mut value = (self.data[block_index] & (mask << offset)) >> offset;

        // Number of available bits in the target block
        let num_available_bits = BT::bits() as usize - offset;

        // If needed, extract the remaining bits from the bottom of the next block
        if num_available_bits < self.bit_width {
            let num_remaining_bits = self.bit_width - num_available_bits;
            let high_bits = self.data[block_index + 1] & ((1 << num_remaining_bits) - 1);
            value |= high_bits << num_available_bits;
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_works() {
        let mut bv = FixedWidthIntVector::new(100, 3);
        bv.write_int(5);
        bv.write_int(6);
        bv.write_int(2);
        bv.write_int(0);
        bv.write_int(7);
        assert_eq!(bv.read_int(0), 5);
        assert_eq!(bv.read_int(1), 6);
        assert_eq!(bv.read_int(2), 2);
        assert_eq!(bv.read_int(3), 0);
        assert_eq!(bv.read_int(4), 7);
    }

    #[test]
    #[should_panic]
    fn test_any_panic() {
        let mut bv = FixedWidthIntVector::new(100, 3);
        bv.write_int(8);
    }

    // todo
    // - test that you cannot write more ints than you have length
    // - test that 64-bit blocks (and other bit widths) all work
    // - test sequences with repeating elements, etc.
}
