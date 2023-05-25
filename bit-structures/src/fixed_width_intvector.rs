// Fixed-width unsigned integer vector that bit-packs fixed-width (<= 64 bits) integers into a u64 array.

use std::debug_assert;

use crate::utils::{div_ceil, BitBlock};

type BT = u64; // Block type

struct FixedWidthIntVector {
    data: Box<[BT]>,
    len: usize,
    bits_per_number: usize,
    // Points to the next available bit index
    write_cursor: usize,
}

impl FixedWidthIntVector {
    pub fn new(len: usize, bits_per_number: usize) -> Self {
        // The number of blocks should be just enough to represent `len * bits_per_number` bits.
        let num_blocks = div_ceil(len * bits_per_number, BT::bits() as usize);
        // Initialize to zero so that any trailing bits in the last block will be zero.
        let data = vec![0; num_blocks].into_boxed_slice();
        Self {
            data,
            len,
            bits_per_number,
            write_cursor: 0,
        }
    }

    pub fn write_int(&mut self, value: u64) {
        debug_assert!(value < (1 << self.bits_per_number));
        let index = BT::block_index(self.write_cursor);
        let offset = BT::bit_offset(self.write_cursor);
        self.data[index] |= value << offset;

        // Number of available bits in the target block
        let num_available_bits = BT::bits() as usize - offset;

        // If needed, write the remaining bits to the next block
        if num_available_bits < self.bits_per_number {
            self.data[index + 1] = value >> num_available_bits;
        }
        self.write_cursor += self.bits_per_number;
    }

    pub fn read_int(&mut self, i: usize) -> u64 {
        let bit_index = i * self.bits_per_number;
        let block_index = BT::block_index(bit_index);
        let offset = BT::bit_offset(bit_index);

        // Low bit mask with a number of ones equal to self.bits_per_number.
        // Eg. if bits_per_number is 3, then mask is 0b111.
        let mask = (1 << self.bits_per_number) - 1;

        // Extract the bits the value that are present in the target block
        let mut value = (self.data[block_index] & (mask << offset)) >> offset;

        // Number of available bits in the target block
        let num_available_bits = BT::bits() as usize - offset;

        // If needed, extract the remaining bits from the bottom of the next block
        if num_available_bits < self.bits_per_number {
            let num_remaining_bits = self.bits_per_number - num_available_bits;
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
        // assert_eq!(utils::bit_floor(0), 0);
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

        let result = std::panic::catch_unwind(move || bv.write_int(8));
        assert!(result.is_err());
    }
}
