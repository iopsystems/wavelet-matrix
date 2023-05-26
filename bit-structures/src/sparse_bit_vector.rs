// Elias-Fano-encoded sparse bitvector

use crate::dense_bit_vector::DenseBitVector;
use crate::int_vector::IntVector;

struct SparseBitVector {
    high: DenseBitVector,
    low: IntVector,
    len: usize,
    universe: usize,
}

impl SparseBitVector {
    //
}
