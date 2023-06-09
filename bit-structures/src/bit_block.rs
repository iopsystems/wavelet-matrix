use num::traits::{CheckedShr, SaturatingSub, WrappingSub};
use num::{PrimInt, Unsigned};
use std::fmt::Debug;
use std::ops::{
    AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Shl, Shr, ShrAssign, SubAssign,
};

// todo:
// - audit uses of .usize() and switch to as_usize() where appropriate
// - consider using PrimInt for u8 and u16, keeping BitBlock to u32 and u64 only

/// Trait representing an unsigned integer type used as a block of bits,
/// which allows our bit-based structures to be generic over block sizes.
pub trait BitBlock:
    'static
    + private::Sealed
    + Unsigned
    + PrimInt
    + WrappingSub
    + CheckedShr
    + Shl<Output = Self>
    + Shr<Output = Self>
    + ShrAssign
    + AddAssign
    + SubAssign
    + BitAnd
    + BitOr
    + BitOrAssign
    + BitAndAssign
    + SaturatingSub
    + Sized
    + Copy
    + Clone
    + Debug
    + Into<u64>
    + bincode::Encode
    + bincode::Decode
    + for<'de> bincode::BorrowDecode<'de>
{
    const BITS: u32; // number of bits in the representation of this type

    // todo: this is not the bit width... BITS is the bit width. This is BITS_WIDTH... o_O
    const BIT_WIDTH: u32 = Self::BITS.ilog2(); // bit width

    fn size_in_bytes() -> usize {
        Self::BITS as usize
    }

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

    // The into_xxxx functions panic if the value does not fit
    fn usize(self) -> usize {
        self.to_usize().unwrap()
    }

    fn u32(self) -> u32 {
        self.to_u32().unwrap()
    }

    fn u64(self) -> u64 {
        self.to_u64().unwrap()
    }

    fn f64(self) -> f64 {
        self.to_f64().unwrap()
    }

    fn as_u32(self) -> u32;
    fn as_u64(self) -> u64;
    fn as_usize(self) -> usize;
    fn as_f64(self) -> f64;

    // will panic if the value does not fit
    fn from_u32(value: u32) -> Self;
    fn from_u64(value: u64) -> Self;
    fn from_usize(value: usize) -> Self;

    fn partition_point(self, pred: impl Fn(Self) -> bool) -> Self {
        let n = self;
        let mut b = Self::zero();
        let mut bit = Self::bit_floor(n);
        while bit != Self::zero() {
            let i = (b | bit) - Self::one();
            if i < n && pred(i) {
                b |= bit
            }
            bit >>= Self::one();
        }
        b
    }

    fn bit_floor(self) -> Self {
        let n = self;
        if n.is_zero() {
            Self::zero()
        } else {
            let msb = Self::BITS - 1 - n.leading_zeros();
            Self::one().unsigned_shl(msb)
        }
    }

    fn div_ceil(self, m: Self) -> Self {
        let n = self;
        (n + m - Self::one()) / m
    }

    fn ilog2(self) -> u32;
}

// we use a private trait to prevent BitBlock from being subtyped outside this package
mod private {
    pub trait Sealed {}
}

macro_rules! bit_block_impl {
     ($($t:ty)*) => ($(
        impl private::Sealed for $t {}

        impl BitBlock for $t {
            const BITS: u32 = Self::BITS;
            fn ilog2(self) -> u32 {
                self.ilog2()
            }
            fn as_u32(self) -> u32 {
                self as u32
            }
            fn as_u64(self) -> u64 {
                self as u64
            }
            fn as_f64(self) -> f64 {
                self as f64
            }
            fn as_usize(self) -> usize {
                self as usize
            }
            fn from_u32(value: u32) -> Self {
                <$t>::try_from(value).unwrap()
            }
            fn from_u64(value: u64) -> Self {
                <$t>::try_from(value).unwrap()
            }
            fn from_usize(value: usize) -> Self {
                <$t>::try_from(value).unwrap()
            }
        }

     )*)
 }

bit_block_impl! { u8 u16 u32 u64 }
