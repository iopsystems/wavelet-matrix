// https://github.com/pelikan-io/ccommon/blob/main/docs/modules/cc_histogram.rst
// I want three histograms:
// - naive raw pdf histogram backed by a CDF array, with an entry per bucket; can be incremented
//   - maybe this is a builder (though we don't have those for anything else)
//   - .to_dense() .to_sparse()
// - dense EF-compressed static histogram with repetitions for zero buckets; ie same as above, but with a sparse bitvec
// - sparse EF-compressed static histogram with a dense bitvector marking nonzero buckets (essentially a compact weighted multiset representation)

use crate::sparse_bit_vec::SparseBitVec;

struct Histogram {
    a: u32,
    b: u32,
    n: u32,
    cdf: SparseBitVec,
}

struct HistogramBuilder {
    // 2^a is the absolute error below the cutoff
    // 2^a is also the fixed bin width below the cutoff
    a: u32,
    // 2^-b is the relative error above the cutoff
    // 2^b is the number of bins per log segment above the cutoff
    b: u32,
    // c = a + b + 1 is the cutoff point below which are fixed-width bins
    // of width 2^a with absolute error, and above which are log segments
    // with absolute error of 2^-b by using 2^b bins per log segment
    c: u32,
    pdf: Box<[u32]>,
}

impl HistogramBuilder {
    fn new(a: u32, b: u32, n: u32) -> HistogramBuilder {
        let c = a + b + 1; // linear-log cutoff
        let _max_value = (1u64 << n) - 1;
        let len = 10;
        let pdf = vec![0, len].into_boxed_slice();

        HistogramBuilder { a, b, c, pdf }
    }

    // fn build() -> Histogram {
    //     // let cdf = SparseBitVec::new(ones, len);
    //     Histogram { a: 0, b: 0, cdf }
    // }

    pub fn bin_index(&self, value: u64) -> usize {
        let Self { a, b, c, .. } = self;
        if value < (1 << c) {
            // the bin width below the cutoff is 1 << a
            (value >> a) as usize
        } else {
            // the log segment the value belongs to
            let v = value.ilog2();

            // the number of bins per log segment (above the cutoff) = 2^b
            let bins_per_log_segment = 1 << b;

            // the total number of linear bins (below the cutoff) = 2^(c-a) = 2^(b+1) = 2*2^b
            let linear_bins = bins_per_log_segment << 1;

            // the number of log segments preceding the one containing `value`
            let preceding_log_segments = v - c;
            // could we compute bins_per_log_segment * preceding_log_segments directly
            // as (value >> c) >> b?

            // the bin offset within this log segment
            // - the expression `(value - (1 << v))` un-sets the largest bit, or equivalently,
            //   subtracts the portion of the value up to the start of the log segment,
            //   leaving just the offset value within the log segment. could also be phrased
            //   as `value & (value - 1)`. Or `value ^ (1 << b)`.
            // - the shift by `(v - b)`  extracts the top `b` bits of the value, which correspond
            //   to the bin index within this log segment (there are `2^b` bins per log segment).
            let bin_offset = ((value ^ (1 << v)) >> (v - b)) as u32;

            (linear_bins + bins_per_log_segment * preceding_log_segments + bin_offset) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let b = HistogramBuilder::new(1, 2, 6);
        let bins = [
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10,
            11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14,
            14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
        ];
        for (i, bin) in (0..64).zip(bins) {
            assert_eq!(b.bin_index(i), bin);
        }
    }
}
