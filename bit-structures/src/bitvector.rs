// Bitvector support traits for Access, Rank, Select.
// How can we make select and rank support configurable at zero cost? Type level constant params?
// todo: should this fromiterator type be u32, usize, or u64? or generic?
// todo: now can we implement from_iter in a way that allows us to also pass configuration options,
// eg. sample rates for rank/select samples?
// : FromIterator<u32>
// todo: decide whether to call these `index` and `n` or `i` and `n`
use crate::utils::partition_point;

// You should implement:
// - one of rank1 or rank0
// - one of num_ones or num_zeros
// - len
// todo: test these impls
pub trait BitVector {
    // note: could provide an impl in terms of rank0
    fn rank1(&self, index: usize) -> usize {
        if index >= self.len() {
            return self.num_ones();
        }
        index - self.rank0(index)
    }

    fn rank0(&self, index: usize) -> usize {
        if index >= self.len() {
            return self.num_zeros();
        }
        index - self.rank1(index)
    }

    fn get(&self, index: usize) -> bool {
        // This could be done more efficiently but is a reasonable default.
        let ones_count = self.rank1(index) - self.rank1(index - 1);
        ones_count == 1
    }

    /// Default impl of select1 using binary search over ranks
    fn select1(&self, n: usize) -> Option<usize> {
        if n >= self.num_ones() {
            return None;
        }
        let index = partition_point(self.len(), |i| self.rank1(i) <= n);
        Some(index - 1)
    }

    /// Default impl of select0 using binary search over ranks
    fn select0(&self, n: usize) -> Option<usize> {
        if n >= self.num_zeros() {
            return None;
        }
        let index = partition_point(self.len(), |i| self.rank0(i) <= n);
        Some(index - 1)
    }

    // note: could provide a default impl in terms of num_zeros and num_ones
    fn len(&self) -> usize;

    fn num_ones(&self) -> usize {
        self.len() - self.num_zeros()
    }

    fn num_zeros(&self) -> usize {
        self.len() - self.num_ones()
    }
}

#[cfg(test)]
// todo: split into rank1/select1/rank0/select0 so we can check the 1 functions
// on EF vectors with multiplicity
// special cases:
// - length-1 bitvectors
//
pub fn test_bitvector<T: BitVector>(new: impl Fn(&[usize], usize) -> T) {
    struct TestCase(Vec<usize>, usize);

    let test_cases = [
        TestCase(vec![0, 10], 100),
        TestCase(vec![1, 2, 5, 10, 32], 33),
    ];

    for TestCase(ones, len) in test_cases {
        let bv = new(&ones, len);
        test_bitvector_with_ones(&ones, bv)
    }
}

pub fn test_bitvector_with_ones<T: BitVector>(ones: &[usize], bv: T) {
    use std::collections::{hash_map::RandomState, HashSet};

    // test basic consistency
    assert!(bv.len() >= bv.num_ones());
    assert_eq!(bv.num_ones(), ones.len());
    assert_eq!(bv.num_zeros(), bv.len() - bv.num_ones());

    // test rank

    // test select

    for (n, one) in ones.iter().copied().enumerate() {
        assert_eq!(bv.select1(n), Some(one));
    }

    let zeros = {
        let mut zero_set: HashSet<usize, RandomState> = HashSet::from_iter(0..bv.len());
        for one in ones.iter() {
            zero_set.remove(one);
        }
        let mut zeros = Vec::from_iter(zero_set);
        zeros.sort();
        zeros
    };
    dbg!(ones);
    for (n, zero) in zeros.iter().copied().enumerate() {
        dbg!(n, zero);
        assert_eq!(bv.select0(n), Some(zero));
    }
}
