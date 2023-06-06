// Bitvector support traits for Access, Rank, Select.
// How can we make select and rank support configurable at zero cost? Type level constant params?
// todo: should this fromiterator type be u32, usize, or u64? or generic?
// todo: now can we implement from_iter in a way that allows us to also pass configuration options,
// eg. sample rates for rank/select samples?
// : FromIterator<u32>
// todo: decide whether to call these `index` and `n` or `i` and `n`
// todo: rename this file to bit_vec.rs?
use crate::utils::partition_point;

// You should implement:
// - rank1 or rank0
// - num_ones or num_zeros
// - len

pub trait BitVec {
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

    fn num_ones(&self) -> usize {
        self.len() - self.num_zeros()
    }

    fn num_zeros(&self) -> usize {
        self.len() - self.num_ones()
    }

    fn len(&self) -> usize;

    // todo: batch_rank/select/get which collect into an existing vec (to reduce allocations)
    // fn batch_rank1(&self, index: impl Iterator<Item=usize>, out: Vec<usize>) {
    //     out.extend(index.map(|index| self.rank1(index)))
    // }
}

pub fn default_rank1<T: BitVec + ?Sized>(bv: &T, index: usize) -> usize {
    if index >= bv.len() {
        return bv.num_ones();
    }
    index - bv.rank0(index)
}

pub fn default_rank0<T: BitVec + ?Sized>(bv: &T, index: usize) -> usize {
    if index >= bv.len() {
        return bv.num_zeros();
    }
    index - bv.rank1(index)
}

/// Default impl of select1 using binary search over ranks
pub fn default_select0<T: BitVec + ?Sized>(bv: &T, n: usize) -> Option<usize> {
    if n >= bv.num_zeros() {
        return None;
    }
    let index = partition_point(bv.len(), |i| bv.rank0(i) <= n);
    Some(index - 1)
}

/// Default impl of select0 using binary search over ranks
pub fn default_select1<T: BitVec + ?Sized>(bv: &T, n: usize) -> Option<usize> {
    if n >= bv.num_ones() {
        return None;
    }
    let index = partition_point(bv.len(), |i| bv.rank1(i) <= n);
    Some(index - 1)
}

pub fn default_get<T: BitVec + ?Sized>(bv: &T, index: usize) -> bool {
    // This could be done more efficiently but is a reasonable default.
    let ones_count = bv.rank1(index + 1) - bv.rank1(index);
    ones_count == 1
}

#[cfg(test)]
pub fn test_bitvector<T: BitVec>(new: impl Fn(&[usize], usize) -> T) {
    let bv = new(&[1, 2, 3], 4);
    assert_eq!(bv.len(), 4);
    assert_eq!(bv.num_ones(), 3);
    assert_eq!(bv.num_zeros(), 1);
    assert_eq!(bv.rank0(0), 0);
    assert_eq!(bv.rank0(1), 1);
    assert_eq!(bv.rank0(2), 1);
    assert_eq!(bv.rank0(3), 1);
    assert_eq!(bv.rank0(4), 1);
    assert_eq!(bv.rank0(5), 1);

    assert_eq!(bv.rank1(0), 0);
    assert_eq!(bv.rank1(1), 0);
    assert_eq!(bv.rank1(2), 1);
    assert_eq!(bv.rank1(3), 2);
    assert_eq!(bv.rank1(4), 3);
    assert_eq!(bv.rank1(5), 3);

    assert_eq!(bv.select0(0), Some(0));
    assert_eq!(bv.select0(1), None);
    assert_eq!(bv.select0(2), None);

    assert_eq!(bv.select1(0), Some(1));
    assert_eq!(bv.select1(1), Some(2));
    assert_eq!(bv.select1(2), Some(3));
    assert_eq!(bv.select1(3), None);
    assert_eq!(bv.select1(4), None);
}

pub fn test_bitvector_vs_naive<T: BitVec>(new: impl Fn(&[usize], usize) -> T) {
    use exhaustigen::Gen;

    use crate::naive_bit_vec::NaiveBitVec;

    struct TestCase(Vec<usize>, usize);

    // we use a length larger than what we assume is
    // the largest RawBitVec block size (128)
    let len = 150;
    let mut test_cases = vec![
        TestCase(vec![], 0),
        TestCase(vec![], len),
        TestCase(vec![0], len),
        TestCase(vec![len - 1], len),
        TestCase(vec![0, 10], len),
        TestCase((0..len).collect(), len),
        TestCase((10..len).collect(), len),
        TestCase((0..len - 10).collect(), len),
        TestCase(vec![1, 2, 5, 10, 32], 33),
        TestCase(vec![1, 2, 5, 10, 32], 33),
    ];

    {
        // add test cases with a sparser bit pattern
        let input = vec![0, 10, 20, len - 1];
        let mut gen = Gen::new();
        while !gen.done() {
            let ones = gen.gen_subset(&input);
            test_cases.push(TestCase(ones.copied().collect(), len));
        }
    }

    {
        // Generate all 2^k subsets of the elements 0..k and
        // use them as the bitvector one positions
        let k = 10;
        let input: Vec<_> = (0..k).collect();
        let mut gen = Gen::new();
        while !gen.done() {
            let ones = gen.gen_subset(&input);
            test_cases.push(TestCase(ones.copied().collect(), k));
        }
    }

    for TestCase(ones, len) in test_cases {
        dbg!("test case", &ones, &len);
        let bv = new(&ones, len);
        let nv = NaiveBitVec::new(&ones, len);

        // test basic properties
        assert_eq!(bv.num_ones(), bv.rank1(bv.len()));
        assert_eq!(bv.num_zeros(), bv.rank0(bv.len()));
        assert_eq!(bv.num_zeros() + bv.num_ones(), bv.len());
        assert_eq!(bv.len(), nv.len(), "unequal lengths");

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