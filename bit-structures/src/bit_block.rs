use std::ops::AddAssign;
use std::ops::BitOr;
use std::ops::BitAndAssign;
use std::ops::Shr;
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
use std::fmt::Debug;

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
    // + Shl<u32, Output = Self>
    + Shr<Output = Self>
    + BitOrAssign
    + BitOr 
    + ShrAssign
    + AddAssign
    + AsPrimitive<u32>
    // + std::iter::Step
    +TryFrom<u32> + Debug
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

    // The into_xxxx functions panic if the value does not fit
    fn into_usize(self) -> usize {
        self.to_usize().unwrap()
    }

  fn into_u32(self) -> u32 {
        self.to_u32().unwrap()
    }

    fn into_u64(self) -> u64 {
        self.to_u64().unwrap()
    }

    // will panic if the value does not fit
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

    fn bit_floor(self) -> Self{
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


macro_rules! bit_block_impl {
     ($($t:ty)*) => ($(
        impl BitBlock for $t {
            const BITS: u32 = Self::BITS;
            fn from_usize(value: usize) -> Self {
               <$t>::try_from(value).unwrap()
            }
            fn ilog2(self) -> u32 {
                self.ilog2()
            }
        }
     )*)
 }

bit_block_impl! { u8 u16 u32 u64 }

pub trait LargeBitBlock: BitBlock + From<u32> {}

impl LargeBitBlock for u32 {}

impl LargeBitBlock for u64 {}

