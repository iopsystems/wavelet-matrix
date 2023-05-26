// For now.
#![allow(dead_code)]

/// Bitwise binary search the range [0, n] based on lower_bound_pad from this article:
///   https://orlp.net/blog/bitwise-binary-search/
///
/// Returns the index of the partition point according to the given predicate
/// (the index of the first element of the second partition).
///
/// The slice is assumed to be partitioned according to the given predicate.
/// This means that all elements for which the predicate returns true are at
/// the start of the slice and all elements for which the predicate returns
/// false are at the end.
///
/// If this slice is not partitioned, the returned result is unspecified
/// and meaningless, as this method performs a kind of binary search.
///
/// See https://doc.rust-lang.org/1.69.0/std/primitive.slice.html#method.partition_point
///
/// See the appendix (bottom of this file for a more elaborate but efficient implementation).
pub fn partition_point(n: usize, pred: impl Fn(usize) -> bool) -> usize {
    let mut b = 0;
    let mut bit = bit_floor(n);
    while bit != 0 {
        let i = (b | bit) - 1;
        if i < n && pred(i) {
            b |= bit
        }
        bit >>= 1;
    }
    b
}

pub fn bit_floor(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        let msb = usize::BITS - 1 - n.leading_zeros();
        1 << msb
    }
}

pub fn div_ceil(n: usize, m: usize) -> usize {
    (n + m - 1) / m
}

// Return a mask with the lowest `num_bits` bits set to 1.
// For example, one_mask(3) == 0b111/
// todo: how can we make this work for usize and u32?
// is there a version that works correctly for both 0 and 32?
pub fn one_mask(num_bits: usize) -> u32 {
    debug_assert!(num_bits > 0);
    u32::MAX.wrapping_shr(u32::BITS - num_bits as u32)
}

/// Trait representing an integer type that is being used as a block of bits.
// todo: move into its own file so we can use it in the other types
pub trait BitBlock {
    /// The number of bits in this integer representation
    fn bits() -> u32;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_floor() {
        assert_eq!(bit_floor(0), 0);
        assert_eq!(bit_floor(1), 1);
        assert_eq!(bit_floor(2), 2);
        assert_eq!(bit_floor(3), 2);
        assert_eq!(bit_floor(4), 4);
        assert_eq!(bit_floor(5), 4);
    }

    #[test]
    fn test_partition_point() {
        let n = 100;
        let target = 60;
        assert_eq!(partition_point(n, |i| i < target), target);
        assert_eq!(partition_point(target - 1, |i| i < target), target - 1);
    }
}

/*

Appendix: Potentially faster but more complex implementation of partition_point

// Prototype implementation of the more complex but faster overlap method
// from the same article. Is the overlap version from the orlp.net article
// with some optimizations described further down in the same article.
fn partition_point_overlap(n: usize, mut pred: impl FnMut(usize) -> bool) -> usize {
    if n == 0 {
        return 0;
    };

    let two_k = bit_floor(n);
    let begin = if pred(n / 2) { n - (two_k - 1) } else { 0 };

    let mut b = 0;
    let mut bit = two_k >> 1;
    while bit != 0 {
        if pred(begin + b + bit) {
            b += bit
        }
        bit >>= 1;
    }
    begin + b + 1
}

*/
