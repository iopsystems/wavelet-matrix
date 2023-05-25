// Elias-Fano-encoded sparse bitvector

use crate::dense_bitvector::DenseBitVector;
use crate::fixed_width_intvector::FixedWidthIntVector;

struct SparseBitVector {
    high: DenseBitVector,
    low: FixedWidthIntVector,
    len: usize,
    universe: usize,
}

impl SparseBitVector {
    //
}
