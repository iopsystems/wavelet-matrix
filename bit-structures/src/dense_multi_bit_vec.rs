use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::bit_vec::BitVec;
use crate::bit_vec::BitVecFromSorted;
use crate::bit_vec::MultiBitVec;
use crate::dense_bit_vec::DenseBitVec;
use crate::sparse_bit_vec::SparseBitVec;

#[derive(Debug)]
pub struct DenseMultiBitVec<Ones: BitBlock> {
    occupancy: DenseBitVec<Ones>,
    multiplicity: SparseBitVec<Ones>,
    num_ones: Ones,
    universe_size: Ones,
}

impl<Ones: BitBlock> bincode::Encode for DenseMultiBitVec<Ones> {
    encode_impl!(occupancy, multiplicity, num_ones, universe_size);
}
impl<Ones: BitBlock> bincode::Decode for DenseMultiBitVec<Ones> {
    decode_impl!(occupancy, multiplicity, num_ones, universe_size);
}
impl<'de, Ones: BitBlock> bincode::BorrowDecode<'de> for DenseMultiBitVec<Ones> {
    borrow_decode_impl!(occupancy, multiplicity, num_ones, universe_size);
}

impl<Ones: BitBlock> BitVecFromSorted for DenseMultiBitVec<Ones> {
    fn from_sorted(ones: &[Ones], universe_size: Ones) -> Self {
        assert!(ones.windows(2).all(|w| w[0] <= w[1])); // assert sorted

        // collapse runs of the same one index into a vec of cumulative run lengths
        let mut cumulative_run_lengths = Vec::new();
        if let Some((&first, rest)) = ones.split_first() {
            let mut cur = first;
            let mut count = Ones::one();
            for next in rest.iter().copied() {
                if next != cur {
                    cumulative_run_lengths.push(count);
                    cur = next;
                }
                count += Ones::one()
            }
            cumulative_run_lengths.push(count);
        }

        let num_ones = Ones::from_usize(ones.len());
        Self {
            occupancy: DenseBitVec::from_sorted(ones, universe_size),
            multiplicity: SparseBitVec::from_sorted(
                &cumulative_run_lengths,
                num_ones + Ones::one(),
            ),
            num_ones,
            universe_size,
        }
    }
}

impl<Ones: BitBlock> BitVec for DenseMultiBitVec<Ones> {
    type Ones = Ones;

    // number of values in this multiset whose value is < index
    fn rank1(&self, index: Ones) -> Ones {
        let n = self.occupancy.rank1(index);
        if n.is_zero() {
            Ones::zero()
        } else {
            self.multiplicity.select1(n - Ones::one())
        }
    }

    // return the value of the n-th element
    fn try_select1(&self, n: Ones) -> Option<Ones> {
        let n = self.multiplicity.rank1(n + Ones::one());
        self.occupancy.try_select1(n)
    }

    // number of zeros whose value is < index
    fn rank0(&self, index: Ones) -> Ones {
        self.occupancy.rank0(index)
    }

    // value of the n-th zero
    fn try_select0(&self, n: Ones) -> Option<Ones> {
        self.occupancy.try_select0(n)
    }

    // return true if there is at least one value at the index
    fn get(&self, index: Ones) -> bool {
        self.occupancy.get(index)
    }

    fn num_ones(&self) -> Ones {
        self.num_ones
    }

    fn universe_size(&self) -> Ones {
        self.universe_size
    }
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
