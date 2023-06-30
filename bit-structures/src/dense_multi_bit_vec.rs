use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::bit_vec::BitVec;
use crate::bit_vec::BitVecFromSorted;
use crate::bit_vec::MultiBitVec;
use crate::dense_bit_vec::DenseBitVec;
use crate::sparse_bit_vec::SparseBitVec;

// todo: figure out how to pass the right len to the occupancy/multiplicity constructors

#[derive(Debug)]
pub struct DenseMultiBitVec<Ones: BitBlock> {
    occupancy: DenseBitVec<Ones>,
    multiplicity: SparseBitVec<Ones>,
    num_ones: Ones,
    len: Ones,
}

impl<Ones: BitBlock> bincode::Encode for DenseMultiBitVec<Ones> {
    encode_impl!(occupancy, multiplicity, num_ones, len);
}
impl<Ones: BitBlock> bincode::Decode for DenseMultiBitVec<Ones> {
    decode_impl!(occupancy, multiplicity, num_ones, len);
}
impl<'de, Ones: BitBlock> bincode::BorrowDecode<'de> for DenseMultiBitVec<Ones> {
    borrow_decode_impl!(occupancy, multiplicity, num_ones, len);
}

impl<Ones: BitBlock> BitVecFromSorted for DenseMultiBitVec<Ones> {
    fn from_sorted(ones: &[Ones], len: Ones) -> Self {
        assert!(ones.windows(2).all(|w| w[0] <= w[1])); // assert sorted

        // collapse runs of the same one index
        let cumulative_sum_of_run_lengths: Vec<Ones> = ones
            .group_by(|a, b| a == b)
            .map(|g| g.len())
            .scan(Ones::zero(), |acc, x| {
                *acc += Ones::from_usize(x);
                Some(*acc)
            })
            .collect();
        let num_ones = Ones::from_usize(ones.len());
        Self {
            occupancy: DenseBitVec::from_sorted(ones, len),
            multiplicity: SparseBitVec::from_sorted(
                &cumulative_sum_of_run_lengths,
                num_ones + Ones::one(),
            ),
            num_ones,
            len,
        }
    }
}

impl<Ones: BitBlock> BitVec for DenseMultiBitVec<Ones> {
    type Ones = Ones;

    fn rank1(&self, index: Ones) -> Ones {
        let n = self.occupancy.rank1(index);
        if n.is_zero() {
            Ones::zero()
        } else {
            self.multiplicity.select1(n - Ones::one())
        }
    }

    fn try_select1(&self, n: Ones) -> Option<Ones> {
        let n = self.multiplicity.rank1(n + Ones::one());
        self.occupancy.try_select1(n)
    }

    // fn select0(&self, _n: Ones) -> Option<Ones> {
    //     unimplemented!()
    // }

    fn num_ones(&self) -> Ones {
        self.num_ones
    }

    fn len(&self) -> Ones {
        self.len
    }

    // fn get(&self, index: Ones) -> bool {
    //     index.is_zero()
    // }
}

impl<Ones: BitBlock> MultiBitVec for DenseMultiBitVec<Ones> {}

#[cfg(test)]
mod tests {
    use crate::bit_vec;

    use super::*;

    #[test]
    fn test_bitvector() {
        bit_vec::test_bitvector(DenseMultiBitVec::<u32>::from_sorted);
        bit_vec::test_bitvector_vs_naive(DenseMultiBitVec::<u32>::from_sorted);
    }
}
