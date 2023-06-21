// https://github.com/pelikan-io/ccommon/blob/main/docs/modules/cc_histogram.rst
// https://observablehq.com/d/35f0b601ed888da9
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
        let Self { a, b, c, .. } = *self;
        if value < (1 << c) {
            // the bin width below the cutoff is 1 << a
            (value >> a) as usize
        } else {
            // The log segment containing the value
            let v = value.ilog2();

            // We want to calculate the number of bins that precede the v-th log segment.
            // We can break this down into two components:
            // 1. The linear section has twice as many bins as an individual log segment above the cutoff,
            //    for a total of 2*2^b bins below the cutoff.
            // 2. Above the cutoff, there are `v - c` log segments before the v-th log segment, each with 2^b bins.
            // Taken together, there are (v - c + 2) * 2^b bins preceding the v-th log segment.
            let preceding_bins_before_log_segment = (v - c + 2) << b;

            // The bin offset within the v-th log segment.
            // - `value ^ (1 << v)` zeros the topmost (v-th) bit
            // - `>> (v - b)` extracts the top `b` bits of the value, corresponding
            //   to the bin index within the v-th log segment.
            let preceding_bins_in_log_segment = ((value ^ (1 << v)) >> (v - b)) as u32;

            // there are 2^(b+1) = 2*2^b bins below the cutoff, and (v-c)*2^b bins between the cutoff
            // and v-th log segment.
            (preceding_bins_before_log_segment + preceding_bins_in_log_segment) as usize
        }
    }

    // given a bin index, returns the lowest value that bin can contain.
    pub fn low(&self, i: u32) -> u64 {
        let Self { a, b, c, .. } = *self;
        let bins_in_linear_section = 2 << b;
        if i < bins_in_linear_section {
            (i << a) as u64
        } else {
            // the offset of the i-th bin inside the log section of the histogram
            let n = i - bins_in_linear_section;

            // the log segment containin the i-th bin.
            // there are c bins before the cutoff each log segment has 2^b bins wide.
            let log_segment = c + (n >> b); // mask off the low bits

            // the offset of this bin within its log segment
            let bin = n & ((1 << b) - 1); // keep only the low bits

            // The lowest value in this bin is given by:
            //   2^log_segment + bin * 2^(log_segment - b)
            // The first term represents the start of this log segment and
            // the second term represents the offset of the linear bin
            // within that log segment.
            ((1 << log_segment) + bin * (1 << (log_segment - b))) as u64
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
            if i > 14 {
                println!("--------------------");
                dbg!(i, bin, b.low(bin as u32));
            }
        }

        // dbg!(17, 8, b.low(8));

        panic!("histogram");
    }
}
