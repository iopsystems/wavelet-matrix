/// Trait representing an integer type that is being used as a block of bits.
// todo: can this require and implement basic math (shift, addition, subtraction, multiplication)?
// then we can allow generic implementations with customizable block sizes.
pub trait BitBlock: Clone {
    const MIN: Self;
    const MAX: Self;
    const BITS: u32;

    /// Block index of the block containing the `i`-th bit
    fn bit_offset(i: usize) -> usize {
        i & (Self::BITS - 1) as usize
    }

    /// Bit index of the `i`-th bit within its block (mask off the high bits)
    fn block_index(i: usize) -> usize {
        i >> Self::bits_pow2()
    }

    /// Block index and bit offset of the `i`-th bit
    fn index_offset(i: usize) -> (usize, usize) {
        (Self::block_index(i), Self::bit_offset(i))
    }

    /// Power of 2 of the number of bits in this block type
    fn bits_pow2() -> u32 {
        Self::BITS.ilog2()
    }
}

impl BitBlock for u8 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u16 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u32 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u64 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u128 {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const BITS: u32 = Self::BITS;
}

impl BitBlock for usize {
    const MIN: Self = Self::MIN;
    const MAX: Self = Self::MAX;
    const BITS: u32 = Self::BITS;
}

// Adapter functions if we only have an instance of a block and not the type in hand
// pub fn bit_split<B: BitBlock>(_: B, i: usize) -> (usize, usize) {
//     B::bit_split(i)
// }

// pub fn bits_log2<B: BitBlock>(_: B) -> u32 {
//     B::bits_log2()
// }

// pub fn block_index<B: BitBlock>(_: B, i: usize) -> usize {
//     B::block_index(i)
// }

// pub fn bit_offset<B: BitBlock>(_: B, i: usize) -> usize {
//     B::bit_offset(i)
// }
