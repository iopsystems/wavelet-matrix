use core::ops::BitAndAssign;
use core::ops::Shr;
use num::traits::AsPrimitive;
use num::traits::CheckedShr;
use num::traits::WrappingSub;
use num::PrimInt;
use num::Unsigned;
use std::ops::BitAnd;
use std::ops::BitOrAssign;
use std::ops::Shl;
use std::ops::ShrAssign;
use std::ops::Sub;

/// Trait representing an unsigned integer type used as a block of bits,
/// which allows our bit-based structures to be generic over block sizes.
pub trait BitBlock:
    PrimInt
    + Unsigned
    + Sub
    + WrappingSub
    + CheckedShr
    + BitAndAssign
    + BitAnd
    + Clone
    + Shl<Output = Self>
    + Shr<Output = Self>
    + BitOrAssign
    + ShrAssign
    + AsPrimitive<u32>
{
    const BITS: u32; // number of bits in the representation of this type
    const BIT_WIDTH: u32 = Self::BITS.ilog2(); // bit width

    /// Bit index of the `i`-th bit within its block (mask off the high bits)
    fn bit_offset(i: usize) -> usize {
        i & (Self::BITS - 1) as usize
    }

    /// Block index of the block containing the `i`-th bit
    fn block_index(i: usize) -> usize {
        i >> Self::BIT_WIDTH
    }

    /// Block index and bit offset of the `i`-th bit
    fn index_offset(i: usize) -> (usize, usize) {
        (Self::block_index(i), Self::bit_offset(i))
    }
    /// Return a bit mask with `n` 1-bits set in the low bits.
    fn one_mask(n: impl Into<u32>) -> Self {
        let max = Self::zero().wrapping_sub(&Self::one());
        max.checked_shr(Self::BITS - n.into())
            .unwrap_or(Self::zero())
    }
    // panics if the value does not fit
    fn into_usize(self) -> usize {
        self.to_usize().unwrap()
    }
}

impl BitBlock for u8 {
    const BITS: u32 = Self::BITS;
}

impl BitBlock for u16 {
    const BITS: u32 = Self::BITS;
}

impl BitBlock for usize {
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
