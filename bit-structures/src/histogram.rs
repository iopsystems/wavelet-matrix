// https://github.com/pelikan-io/ccommon/blob/main/docs/modules/cc_histogram.rst
// https://observablehq.com/d/35f0b601ed888da9
// https://github.com/pelikan-io/rustcommon/blob/main/histogram/src/histogram.rs
// I want three histograms:
// - naive raw pdf histogram backed by a CDF array, with an entry per bucket; can be incremented
//   - maybe this is a builder (though we don't have those for anything else)
//   - .to_dense() .to_sparse()
//     - or better, construct from another histogram w possibly different (but compatible) a, b parameters & block type
// - dense EF-compressed static histogram with repetitions for zero buckets; ie same as above, but with a sparse bitvec
// - sparse EF-compressed static histogram with a dense bitvector marking nonzero buckets (essentially a compact weighted multiset representation)
// - parameterizable storage maximum - eg. u32 or u64 (bitblock?)
// - sparse bitvec should also be parameterizable the same way so that we can store 64 or 128 bits if we want to, but not have to pay for it otherwise
// - make the "rank" / quantile function compatible with rank1(0) being inclusive; maybe make a rank1p which is the rank of the value plus one.
//   - like how log1p(x) = ln(x+1); rank1p(x) = rank1(x - 1) and rank0p(x) = rank0(x - 1) but will return 0 for x=0?

// todo:
// - consider reverting back to the nongeneric version; use u32 for a, b, c, n, and u64 for the value.
// - investigate if there are packages that allow you to "template" a type, implementing it for a finite enumerated set of types: xxx_impl!().

/*
https://www.reddit.com/r/rust/comments/7f7qgo/implementing_a_trait_for_multiple_concrete_structs/
https://github.com/rust-lang/rust/blob/59bf09d4d473c803609d3ad925a0ebf13bdbb0ab/src/libcore/cmp.rs#L759-L783

macro_rules! eq_impl {
    ($($t:ty)*) => ($(
        #[stable(feature = "rust1", since = "1.0.0")]
        impl Eq for $t {}
    )*)
}

eq_impl! { () bool char usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }

*/

use crate::{bit_vec::BitVec, slice_bit_vec::SliceBitVec};

struct Histogram<T: BitVec> {
    h: HistogramHelper,
    cdf: T,
}

struct HistogramBuilder {
    h: HistogramHelper,
    pdf: Box<[u64]>,
}

impl HistogramBuilder {
    pub fn new(a: u32, b: u32, n: u32) -> HistogramBuilder {
        let h = HistogramHelper::new(a, b, n);
        let pdf = vec![0; h.num_bins() as usize].into_boxed_slice();
        HistogramBuilder { h, pdf }
    }

    pub fn increment(&mut self, value: u64, count: u64) {
        let bin_index = self.h.bin_index(value);
        self.pdf[bin_index as usize] += count;
    }

    pub fn decrement(&mut self, value: u64, count: u64) {
        let bin_index = self.h.bin_index(value);
        self.pdf[bin_index as usize] -= count;
    }

    pub fn build(self) -> Histogram<SliceBitVec> {
        Histogram {
            h: self.h,
            cdf: SliceBitVec::new(&[], 10),
        }
    }
}

// pdf: Box<[u32]>,
// let num_bins = (n - c + 2) << b;
// let pdf = vec![num_bins; 0].into_boxed_slice();
// fn build() -> Histogram {
//     // let cdf = SparseBitVec::new(ones, len);
//     Histogram { a: 0, b: 0, cdf }
// }

// note: multiple histograms could concievably share a single helper,
// if space becomes an issue (it would have to be behind a pointer).
// note: a, b, c, n could be u8 except it causes type system annoyances...
struct HistogramHelper {
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
    // c of them below the cutoff point and n-c above it.
    // - below the cutoff, there are 2^(b+1) = 2*2^b bins in total
    // - above the cutoff, there are n-c log segments, with 2^b bins each
    // so the total number of bins in the histogram is (n-c+2) * 2^b.
    // while the number of bins could exceed u32::MAX, such as if the
    // histogram stores u64 values and consists entirely of linear bins
    // with a = 0, that is not a use case this library is designed to support
    // and so we limit the maximum number of bins to roughly 4 billion.
    // update: that's not true at the moment; should figure out the approach here.
    num_bins: u32,
}

impl HistogramHelper {
    fn new(a: u32, b: u32, n: u32) -> HistogramHelper {
        let c = a + b + 1;
        let num_bins = Self::bins_before_seg(n, c, b);
        HistogramHelper {
            a,
            b,
            c,
            n,
            num_bins,
        }
    }

    // seg: power of 2 of the log segment; also the index of the log segment
    // returns the number of bins up to the start of the seg-th log segment.
    // ie. if seg = n, returns the number of bins in the histogram (since
    // (the max value is 2^n-1 and the start of the n-th seg is at 2^n)
    fn bins_before_seg(seg: u32, c: u32, b: u32) -> u32 {
        // We want to calculate the number of bins that precede the v-th log segment.
        // We can break this down into two components:
        // 1. The linear section has twice as many bins as an individual log segment above the cutoff,
        //    for a total of 2^(b+1) = 2*2^b bins below the cutoff.
        // 2. Above the cutoff, there are `v - c` log segments before the v-th log segment, each with 2^b bins.
        // Taken together, there are (v - c + 2) * 2^b bins preceding the v-th log segment.
        // Note: if we knew there are less than 2^32 bins, we could use `((seg - c + 2) << b) as u64`
        (seg - c + 2) << b
    }

    pub fn max_value(&self) -> u64 {
        if self.n == u64::BITS {
            u64::max_value()
        } else {
            1 << self.n
        }
    }

    pub fn high(&self, i: u32) -> u64 {
        if i == self.num_bins {
            self.max_value()
        } else {
            // the highest value of the i-th bin is the
            // integer right before the low of the next bin.
            self.low(i + 1) - 1
        }
    }

    pub fn bin_index(&self, value: u64) -> u32 {
        let Self { a, b, c, .. } = *self;
        if value < (1 << c) {
            // the bin width below the cutoff is 1 << a
            (value >> a) as u32
        } else {
            // The log segment containing the value
            // Equivalent to value.ilog2() but compatible with traits from the Num crate
            // (T::BITS - 1 - value.leading_zeros());
            let v = value.ilog2();

            // The bin offset within the v-th log segment.
            // - `value ^ (1 << v)` zeros the topmost (v-th) bit
            // - `>> (v - b)` extracts the top `b` bits of the value, corresponding
            //   to the bin index within the v-th log segment.
            let bins_within_seg = (value ^ (1 << v)) >> (v - b);

            // there are 2^(b+1) = 2*2^b bins below the cutoff, and (v-c)*2^b bins between the cutoff
            // and v-th log segment.
            Self::bins_before_seg(v, c, b) + bins_within_seg as u32
        }
    }

    // given a bin index, returns the lowest value that bin can contain.
    pub fn low(&self, i: u32) -> u64 {
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
            let seg_start = 1u64 << seg;
            // the bin we're in within that segment, given by the low bits of n:
            // the bit shifts remove the `b` lowest bits, leaving only the high
            // bits, which we then subtract from `n` to keep only the low bits.
            let bin = n - ((n >> b) << b);
            // the width of an individual bin within this log segment
            let bin_width = seg_start >> b;
            // the lowest value represented by this bin is simple to compute:
            // start where the logarithmic segment begins, and increment by the
            // linear bin index within the segment times the bin width.
            seg_start + bin as u64 * bin_width
        }
    }

    pub fn a(&self) -> u32 {
        self.a
    }

    pub fn b(&self) -> u32 {
        self.b
    }

    pub fn c(&self) -> u32 {
        self.c
    }

    pub fn num_bins(&self) -> u32 {
        self.num_bins
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let b = HistogramHelper::new(1, 2, 6);
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
