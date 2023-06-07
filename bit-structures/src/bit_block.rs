use num::traits::CheckedShr;
use num::traits::WrappingSub;
use num::PrimInt;
use num::ToPrimitive;
use num::Unsigned;

/// Trait representing an unsigned integer type used as a block of bits,
/// which allows our bit-based structures to be generic over block sizes.
pub trait BitBlock: PrimInt + ToPrimitive + Unsigned + WrappingSub + CheckedShr + Clone {
    const BITS: u32;

    /// Block index of the block containing the `i`-th bit
    fn bit_offset(i: usize) -> usize {
        i & (Self::BITS - 1) as usize
    }

    /// Bit index of the `i`-th bit within its block (mask off the high bits)
    fn block_index(i: usize) -> usize {
        i >> Self::bits_log2()
    }

    /// Block index and bit offset of the `i`-th bit
    fn index_offset(i: usize) -> (usize, usize) {
        (Self::block_index(i), Self::bit_offset(i))
    }

    /// Power of 2 of the number of bits in this block type
    fn bits_log2() -> u32 {
        Self::BITS.ilog2()
    }

    /// Return a bit mask with `n` 1-bits set in the low bits.
    fn one_mask(n: impl Into<u32>) -> Self {
        let max = Self::zero().wrapping_sub(&Self::one());
        max.checked_shr(Self::BITS - n.into())
            .unwrap_or(Self::zero())
    }
}

impl BitBlock for u8 {
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u16 {
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u32 {
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u64 {
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u128 {
    const BITS: u32 = Self::BITS;
}

impl BitBlock for usize {
    const BITS: u32 = Self::BITS;
}
