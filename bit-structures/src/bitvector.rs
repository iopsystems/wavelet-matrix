// Bitvector support traits for Access, Rank, Select.
// How can we make select and rank support configurable at zero cost? Type level constant params?
// todo: should this fromiterator type be u32, usize, or u64? or generic?
// todo: now can we implement from_iter in a way that allows us to also pass configuration options,
// eg. sample rates for rank/select samples?
// : FromIterator<u32>
// todo: decide whether to call these `index` and `n` or `i` and `n`
use crate::utils::partition_point;

// You should implement:
// - rank1 + num_ones, or rank0 + num_zeros
// - len

//
// todo: test these impls
// - SparseBitVector uses select0 and get
// - does anyone use select1?
pub trait BitVector {
    // note: could provide an impl in terms of rank0
    fn rank1(&self, index: usize) -> usize {
        default_rank1(self, index)
    }

    fn rank0(&self, index: usize) -> usize {
        default_rank0(self, index)
    }

    fn select1(&self, n: usize) -> Option<usize> {
        default_select1(self, n)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        default_select0(self, n)
    }

    fn get(&self, index: usize) -> bool {
        default_get(self, index)
    }

    fn len(&self) -> usize;

    fn num_ones(&self) -> usize {
        self.len() - self.num_zeros()
    }

    fn num_zeros(&self) -> usize {
        self.len() - self.num_ones()
    }
}

pub fn default_rank1<T: BitVector + ?Sized>(bv: &T, index: usize) -> usize {
    if index >= bv.len() {
        return bv.num_ones();
    }
    index - bv.rank0(index)
}

pub fn default_rank0<T: BitVector + ?Sized>(bv: &T, index: usize) -> usize {
    if index >= bv.len() {
        return bv.num_zeros();
    }
    index - bv.rank1(index)
}

/// Default impl of select1 using binary search over ranks
pub fn default_select0<T: BitVector + ?Sized>(bv: &T, n: usize) -> Option<usize> {
    if n >= bv.num_zeros() {
        return None;
    }
    let index = partition_point(bv.len(), |i| bv.rank0(i) <= n);
    Some(index - 1)
}

/// Default impl of select0 using binary search over ranks
pub fn default_select1<T: BitVector + ?Sized>(bv: &T, n: usize) -> Option<usize> {
    if n >= bv.num_ones() {
        return None;
    }
    let index = partition_point(bv.len(), |i| bv.rank1(i) <= n);
    Some(index - 1)
}

pub fn default_get<T: BitVector + ?Sized>(bv: &T, index: usize) -> bool {
    // This could be done more efficiently but is a reasonable default.
    let ones_count = bv.rank1(index + 1) - bv.rank1(index);
    ones_count == 1
}

#[cfg(test)]
// todo: split into rank1/select1/rank0/select0 so we can check the 1 functions
// on EF vectors with multiplicity
// special cases:
// - length-1 bitvectors
//
pub fn test_bitvector_vs_naive<T: BitVector>(new: impl Fn(&[usize], usize) -> T) {
    use crate::naive_bit_vector::NaiveBitVector;

    struct TestCase(Vec<usize>, usize);

    let test_cases = [
        TestCase(vec![0, 10], 100),
        TestCase(vec![1, 2, 5, 10, 32], 33),
        TestCase(vec![1, 2, 5, 10, 32], 33),
    ];

    for TestCase(ones, len) in test_cases {
        let bv = new(&ones, len);
        let nv = NaiveBitVector::new(&ones, len);

        // test basic properties
        assert_eq!(bv.num_ones(), bv.rank1(bv.len()));
        assert_eq!(bv.num_zeros(), bv.rank0(bv.len()));
        assert_eq!(bv.num_zeros() + bv.num_ones(), bv.len());
        assert_eq!(bv.len(), nv.len());

        // test rank0 and rank1
        for i in 0..bv.len() + 2 {
            assert_eq!(bv.rank0(i), nv.rank0(i));
            assert_eq!(bv.rank1(i), nv.rank1(i));
        }

        // test select0
        for n in 0..nv.num_zeros() {
            assert_eq!(bv.select0(n), nv.select0(n));
        }
        assert_eq!(bv.select0(nv.num_zeros()), None);
        assert_eq!(bv.select0(nv.num_zeros() + 1), None);

        // test select1
        for n in 0..nv.num_ones() {
            assert_eq!(bv.select1(n), nv.select1(n));
        }
        assert_eq!(bv.select1(nv.num_ones()), None);
        assert_eq!(bv.select1(nv.num_ones() + 1), None);

        // test get
        for i in 0..nv.len() {
            assert_eq!(bv.get(i), nv.get(i));
        }
    }
}
