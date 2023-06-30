// Dense bit vector with rank and select, based on the ideas described
// in the paper "Fast, Small, Simple Rank/Select on Bitmaps".
// We use an additional level of blocks provided by the RawBitVec, but the ideas are the same.

// todo:
//  - benchmark the effect on nonuniformly distributed 1 bits; i bet it helps more when the data are clustered
//  - try split_last in select1
use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::bit_vec::BitVec;
use crate::bit_vec::BitVecFromSorted;
use crate::dense_bit_vec::DenseBitVec;
use crate::sparse_bit_vec::SparseBitVec;

// todo: describe what each rank/select sample holds.

#[derive(Debug)]
pub struct DenseMultiBitVec<Ones: BitBlock> {
    occupancy: DenseBitVec<Ones>,
    multiplicity: SparseBitVec<Ones>,
}

impl<Ones: BitBlock> bincode::Encode for DenseMultiBitVec<Ones> {
    encode_impl!(occupancy, multiplicity);
}
impl<Ones: BitBlock> bincode::Decode for DenseMultiBitVec<Ones> {
    decode_impl!(occupancy, multiplicity);
}
impl<'de, Ones: BitBlock> bincode::BorrowDecode<'de> for DenseMultiBitVec<Ones> {
    borrow_decode_impl!(occupancy, multiplicity);
}

impl<Ones: BitBlock> BitVecFromSorted for DenseMultiBitVec<Ones> {
    fn from_sorted(ones: &[Ones], len: Ones) -> Self {
        Self {
            occupancy: DenseBitVec::from_sorted(ones, len),
            multiplicity: SparseBitVec::from_sorted(ones, len),
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
            self.multiplicity.select1(n - Ones::one()).unwrap()
        }
    }

    fn select1(&self, n: Ones) -> Option<Ones> {
        let n = self.multiplicity.rank1(n + Ones::one());
        self.occupancy.select1(n)
    }

    fn select0(&self, _n: Ones) -> Option<Ones> {
        unimplemented!()
    }

    fn num_ones(&self) -> Ones {
        Ones::one()
    }

    fn len(&self) -> Ones {
        Ones::one()
    }

    fn get(&self, index: Ones) -> bool {
        index.is_zero()
    }
}

#[cfg(test)]
mod tests {}
