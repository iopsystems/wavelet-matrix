// Naive bit vector implemented as a dense ones array.
// Primarily used for testing purposes.

#[derive(Debug)]
pub struct NaiveBitVector {
    ones: Box<[usize]>,
    len: usize,
}

impl NaiveBitVector {
    pub fn new(ones: &[usize], len: usize) -> Self {
        // todo: assert ones are in sorted monotonically ascending order
        Self {
            ones: ones.into_boxed_slice(),
            len,
        }
    }
}

impl BitVector for NaiveBitVector {
    fn rank1(&self, i: usize) -> usize {
        if i >= self.len() {
            return self.num_ones();
        }
    }

    fn rank0(&self, i: usize) -> usize {
        if i >= self.len() {
            return self.num_zeros();
        }
        i - self.rank1(i)
    }

    fn select1(&self, n: usize) -> Option<usize> {
        if n >= self.num_ones {
            return None;
        }
        Some(1)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        if n >= self.num_zeros() {
            return None;
        }
        Some(1)
    }

    fn num_zeros(&self) -> usize {
        self.len() - self.num_ones()
    }

    fn num_ones(&self) -> usize {
        self.num_ones
    }

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, i: usize) -> bool {
        let ones_count = self.rank1(i) - self.rank1(i - 1);
        ones_count == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::bit_block::BitBlock;
    // use crate::{bitvector, raw_bit_vector};
    // use rand::Rng;

    #[test]
    fn test_new() {
        let raw = RawBitVector::new(100);
        let _ = DenseBitVector::new(raw, 5, 5);
    }
}
