use crate::bit_block::BitBlock;
use num::One;
use num::Zero;

// todo
// - size_in_bytes() as part of the bitvec trait, and bitblock trait.
//   - not size_in_bits so that it can always fit in a usize
// - consider having MultiBitVec be a separate trait that removes all zero-related functionality?
//   - no, it makes sense to do rank0/select0 queries on some representations (though not all support it)
// - consider calling it MultiVec or something? it is not quite as bit-ish since now a bit can be set 'multiple times'...
// - current len() means universe size for bitvecs, but should mean num elements for multibitvecs (and wavelet matrix)
//   - len could be less than, equal to, or greater than universe size
// for some things you want the universe size (ie. max one index plus one)
// for other things you want length - num_zeros + num_ones.
// - what should len mean for multisets? number of one bits, or number of one plus zero bits? ie. universe size plus num extra elements...
// - should we remove num_zeros from the multibitvec trait?

// BitVec is the general vector trait. MultiBitVec is the trait for bitvectors with multiplicity.

// Implementers should implement:
// - rank1 or rank0
// - num_ones or num_zeros
// - len

// Note: We want to avoid usize since it depends on the environment and we want to
// serialize from 64-bit to 32-bit environments (WebAssembly)

// todo: consider removing default impl of rank1 since it is not used by any current subtypes

// note: the sorted ones constructor makes it natural to do zero compression, since we know
// the universe size and bounds of the 1 bits.

pub trait BitVecFromSorted: BitVec {
    fn from_sorted(ones: &[Self::Ones], len: Self::Ones) -> Self;
}

// pub trait MultiBitVecFromSorted: MultiBitVec {
//     // construct from (index, weight) pairs.
//     fn from_sorted(ones: &[Self::Ones], counts: &[Self::Ones], len: Self::Ones) -> Self;
// }

// note: the static bounds are to support deriving bincode implementations for all concrete subtypes.
// i don't fully understand this stuff yet so it may be a bad idea, but so far all bitvecs hold no references
// and therefore seem to be compatible with a 'static lifetime...
pub trait BitVec:
    'static + bincode::Encode + bincode::Decode + for<'de> bincode::BorrowDecode<'de>
{
    type Ones: BitBlock;

    // experimental construction interface;
    // todo:
    // - needs to be able to pass configuration options somehow (eg. rank/select sampling rates)
    // - does not make much sense for the RLE bitvec

    fn rank1(&self, index: Self::Ones) -> Self::Ones {
        self.default_rank1(index)
    }

    fn rank0(&self, index: Self::Ones) -> Self::Ones {
        self.default_rank0(index)
    }

    fn select1(&self, n: Self::Ones) -> Self::Ones {
        self.try_select1(n).unwrap()
    }

    fn select0(&self, n: Self::Ones) -> Self::Ones {
        self.try_select0(n).unwrap()
    }

    fn try_select1(&self, n: Self::Ones) -> Option<Self::Ones> {
        self.default_try_select1(n)
    }

    fn try_select0(&self, n: Self::Ones) -> Option<Self::Ones> {
        self.default_try_select0(n)
    }

    fn get(&self, index: Self::Ones) -> bool {
        self.default_get(index)
    }

    fn num_ones(&self) -> Self::Ones {
        self.universe_size() - self.num_zeros()
    }

    // todo: this is not valid in the face of multiplicity
    fn num_zeros(&self) -> Self::Ones {
        self.universe_size() - self.num_ones()
    }

    // num_ones() + num_zeros()
    // Note: Since `len` returns a value of type `Ones`,
    // the maximum length of a BitVec is 2^n-1 and the
    // maximum index is 2^n-2, with n = Ones::BITS.
    // This means that you cannot have a BitVec with its
    // (2^n-1)-th bit set even though that value is
    // representable by the Ones type (it is Ones::MAX).
    // This is a trade-off in favor of sanity: if we
    // allowed BitVecs of length 2^n, then there could
    // be 2^n 0-bits or 1-bits in an array, and all of
    // the relevant functions would need to use higher
    // bit widths for their return values and internal
    // computations. So we opt for sanity at the low level
    // and can compensate at higher levels if needed (e.g.
    // by storing the count of elements in the phantom
    // (2^n-1)-th position separately and perhaps using
    // a rank1p function that is analogous to log1p,
    // which would compute rank1(i+1) and work even when
    // i+1 and the resulting rank would exceed the bit width
    // of Ones.
    fn universe_size(&self) -> Self::Ones;

    // fn universe_size(&self) -> Self::Ones;

    // todo: return the total size in bytes using std::mem::size_of plus the
    // same recursively for all constituents behind a pointer?
    // this note (https://doc.rust-lang.org/std/mem/fn.size_of.html#size-of-structs)
    // seems odd since it refers to using field declaration order even though I don't think
    // Rust actually uses that to lay things out in memory by default.
    // See also: https://github.com/DKerp/get-size
    // fn size_in_bytes() -> Ones;

    // todo: batch_rank/select/get which collect into an existing vec (to reduce allocations)
    // fn batch_rank1(&self, index: impl Iterator<Item=Ones>, out: Vec<Ones>) {
    //     out.extend(index.map(|index| self.rank1(index)))
    // }

    /// Default impl of rank1 using rank0
    fn default_rank1(&self, index: Self::Ones) -> Self::Ones {
        if index >= self.universe_size() {
            return self.num_ones();
        }
        index - self.rank0(index)
    }

    /// Default impl of rank0 using rank1
    fn default_rank0(&self, index: Self::Ones) -> Self::Ones {
        if index >= self.universe_size() {
            return self.num_zeros();
        }
        index - self.rank1(index)
    }

    /// Default impl of select1 using binary search over ranks
    fn default_try_select0(&self, n: Self::Ones) -> Option<Self::Ones> {
        if n >= self.num_zeros() {
            return None;
        }
        let index = self.universe_size().partition_point(|i| self.rank0(i) <= n);
        Some(index - Self::Ones::one())
    }

    /// Default impl of select0 using binary search over ranks
    fn default_try_select1(&self, n: Self::Ones) -> Option<Self::Ones> {
        if n >= self.num_ones() {
            return None;
        }
        let index = self.universe_size().partition_point(|i| self.rank1(i) <= n);
        Some(index - Self::Ones::one())
    }

    fn default_get(&self, index: Self::Ones) -> bool {
        // This could be done more efficiently but is a reasonable default.
        let ones_count = self.rank1(index + Self::Ones::one()) - self.rank1(index);
        ones_count.is_one()
    }

    // shorthand impls for more concise zero/one invocations
    fn zero() -> Self::Ones {
        Self::Ones::zero()
    }

    fn one() -> Self::Ones {
        Self::Ones::one()
    }

    fn encode(&self) -> Vec<u8> {
        let config = bincode::config::standard().with_fixed_int_encoding();
        bincode::encode_to_vec(self, config).unwrap()
    }

    fn decode(data: Vec<u8>) -> Self {
        let config = bincode::config::standard().with_fixed_int_encoding();
        let (ret, _) = bincode::decode_from_slice(&data, config).unwrap();
        ret
    }
}

// For bitvector types that allow multiplicity
// TODO: For these types:
// - len/num_ones need not be of type Ones, ie. you could have Ones=u8 but have 1 billion elements.
// - there should be no rank0/select0/num_zeros unless they are specifically implemented to be multiplicity-aware.
// for now, from sorted
// - select_distinct1?
pub trait MultiBitVec: BitVecFromSorted {}

// We export these defaults so that implementors of this trait have the option of
// calling these functions, eg. after doing some bookkeeping work. For example,
// the sparse bitvec checks whether it contains multiplicity before calling select0 or rank0.

#[cfg(test)]
pub fn test_bitvector<T: BitVec<Ones = u32>>(new: impl Fn(&[u32], u32) -> T) {
    let bv = new(&[1, 2, 3], 4);
    assert_eq!(bv.universe_size(), 4);
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

    assert_eq!(bv.try_select0(0), Some(0));
    assert_eq!(bv.try_select0(1), None);
    assert_eq!(bv.try_select0(2), None);

    assert_eq!(bv.try_select1(0), Some(1));
    assert_eq!(bv.try_select1(1), Some(2));
    assert_eq!(bv.try_select1(2), Some(3));
    assert_eq!(bv.try_select1(3), None);
    assert_eq!(bv.try_select1(4), None);
}

#[cfg(test)]
pub fn test_bitvector_vs_naive<T: BitVec<Ones = u32>>(new: impl Fn(&[u32], u32) -> T) {
    use exhaustigen::Gen;

    use crate::slice_bit_vec::SliceBitVec;

    struct TestCase(Vec<u32>, u32);

    // we use a length larger than what we assume is
    // the largest RawBitVec block size (128)
    let len = 300;
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
        let input = vec![0, 10, 20, len - 5, len - 1];
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
        println!("test case: ones: {:?}\nlen: {:?}", &ones, &len);
        let bv = new(&ones, len);
        let nv = SliceBitVec::new(&ones, len);

        // test basic properties
        assert_eq!(bv.num_ones(), bv.rank1(bv.universe_size()));
        assert_eq!(bv.num_zeros(), bv.rank0(bv.universe_size()));
        assert_eq!(bv.num_zeros() + bv.num_ones(), bv.universe_size());
        assert_eq!(bv.universe_size(), nv.universe_size(), "unequal lengths");

        // test rank0 and rank1
        for i in 0..bv.universe_size() + 2 {
            assert_eq!(bv.rank0(i), nv.rank0(i));
            assert_eq!(bv.rank1(i), nv.rank1(i));
        }

        // test select0
        for n in 0..nv.num_zeros() {
            assert_eq!(bv.try_select0(n), nv.try_select0(n));
        }
        assert_eq!(bv.try_select0(nv.num_zeros()), None);
        assert_eq!(bv.try_select0(nv.num_zeros() + 1), None);

        // test select1
        for n in 0..nv.num_ones() {
            assert_eq!(bv.try_select1(n), nv.try_select1(n));
        }
        assert_eq!(bv.try_select1(nv.num_ones()), None);
        assert_eq!(bv.try_select1(nv.num_ones() + 1), None);

        // test get
        for i in 0..nv.universe_size() {
            assert_eq!(bv.get(i), nv.get(i));
        }
    }
}
