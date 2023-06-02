/// Trait representing an integer type that is being used as a block of bits.
// todo: can this require and implement basic math (shift, addition, subtraction, multiplication)?
// then we can allow generic implementations with customizable block sizes.
pub trait BitBlock {
    /// The number of bits in this integer representation
    fn bits() -> u32;

    /// Return a tuple consisting of the high bits (shifted down) and low bits of the bit block.
    fn bit_split(i: usize) -> (usize, usize) {
        (Self::block_index(i), Self::bit_offset(i))
    }

    /// Block index of the block containing the `i`-th bit
    fn block_index(i: usize) -> usize {
        // According to a quick Godbolt check, the call to ilog2
        // is optimized away in release mode (-C opt-level=2).
        i >> Self::bits().ilog2()
    }

    /// Bit index of the `i`-th bit within its block (mask off the high bits)
    fn bit_offset(i: usize) -> usize {
        i & (Self::bits() - 1) as usize
    }
}

impl BitBlock for u8 {
    fn bits() -> u32 {
        Self::BITS
    }
}

impl BitBlock for u16 {
    fn bits() -> u32 {
        Self::BITS
    }
}

impl BitBlock for u32 {
    fn bits() -> u32 {
        Self::BITS
    }
}

impl BitBlock for u64 {
    fn bits() -> u32 {
        Self::BITS
    }
}

impl BitBlock for u128 {
    fn bits() -> u32 {
        Self::BITS
    }
}

impl BitBlock for usize {
    fn bits() -> u32 {
        Self::BITS
    }
}
