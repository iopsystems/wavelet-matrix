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
    // 2^a is the absolute error below the cutoff,
    // and is also the bin width below the cutoff.
    // there are 2^(b+1) bins below the cutoff, since
    // the cutoff is 2^(a+b+1) and the bin width is 2^a.
    a: u32,
    // 2^b is the number of bins per log segment above the cutoff, and
    // 2^-b is the relative error above the cutoff
    b: u32,
    // 2^c = 2^(a+b+1) is the cutoff point below which are 2^(b+1)
    // fixed-width bins of width 2^a with absolute error, and above which
    // are log segments each with 2^b bins and a relative error of 2^-b.
    c: u32,
    // 2^n - 1 is the maximum value this histogram can store.
    n: u32,
    // there are n log segments in this histogram, with
    // c of them below the cutoff point and n - c above it.
    // - below the cutoff, there are 2^(b+1) = 2*2^b bins in total
    // - above the cutoff, there are n-c log segments, with 2^b bins each
    // so the total number of bins in the histogram is (n-c+2) * 2^b.
    pdf: Box<[u32]>,
}

impl HistogramBuilder {
    fn new(a: u32, b: u32, n: u32) -> HistogramBuilder {
        let c = a + b + 1; // linear-log cutoff
        let num_bins = (n - c + 2) << b;
        let pdf = vec![0, num_bins].into_boxed_slice();

        HistogramBuilder { a, b, c, n, pdf }
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
            //    for a total of 2^(b+1) = 2*2^b bins below the cutoff.
            // 2. Above the cutoff, there are `v - c` log segments before the v-th log segment, each with 2^b bins.
            // Taken together, there are (v - c + 2) * 2^b bins preceding the v-th log segment.
            let preceding_bins_before_seg = (v - c + 2) << b;

            // The bin offset within the v-th log segment.
            // - `value ^ (1 << v)` zeros the topmost (v-th) bit
            // - `>> (v - b)` extracts the top `b` bits of the value, corresponding
            //   to the bin index within the v-th log segment.
            let preceding_bins_within_seg = ((value ^ (1 << v)) >> (v - b)) as u32;

            // there are 2^(b+1) = 2*2^b bins below the cutoff, and (v-c)*2^b bins between the cutoff
            // and v-th log segment.
            (preceding_bins_before_seg + preceding_bins_within_seg) as usize
        }
    }

    // given a bin index, returns the lowest value that bin can contain.
    pub fn low(&self, i: usize) -> u64 {
        let i = i as u32;
        let Self { a, b, c, .. } = *self;
        let bins_below_cutoff = 2 << b;
        if i < bins_below_cutoff {
            (i << a) as u64
        } else {
            // the number of bins in 0..i that are above the cutoff point
            let n = i - bins_below_cutoff;
            // the index of the log segment we're in: there are `c` log
            // segments below the cutoff and `n >> b` above, since each
            // one is divided into 2^b bins.
            let seg = c + (n >> b);
            // by definition, the lowest value in a log segment is 2^seg
            let seg_start = 1 << seg;
            // the bin we're in within that segment, given by the low bits of n:
            // the bit shifts remove the `b` lowest bits, leaving only the high
            // bits, which we then subtract from `n` to keep only the low bits.
            let bin = n - ((n >> b) << b);
            // the width of an individual bin within this log segment
            let bin_width = seg_start >> b;

            (seg_start + bin * bin_width) as u64
        }
    }

    pub fn high(&self, i: usize) -> u64 {
        if i == self.pdf.len() - 1 {
            self.max_value()
        } else {
            // the highest value of the i-th bin is the
            // integer right before the low of the next bin.
            self.low(i + 1) - 1
        }
    }

    pub fn max_value(&self) -> u64 {
        if self.n == 64 {
            u64::MAX
        } else {
            (1 << self.n) - 1
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
            14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16,
        ];
        for (i, bin) in (0..66).zip(bins) {
            assert_eq!(b.bin_index(i), bin);
            if i > 14 {
                println!("--------------------");
                dbg!(i, bin, b.low(bin));
            }
        }

        // dbg!(17, 8, b.low(8));

        panic!("histogram");
    }
}
