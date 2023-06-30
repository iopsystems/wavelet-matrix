// Base-2 HDR histogram. Inspired and based in part on the Rezolus histogram implementation.
// Resources:
// Rezolus histogram documentation:
//   https://github.com/pelikan-io/ccommon/blob/main/docs/modules/cc_histogram.rst
// Rezolus histogram code:
//   https://github.com/pelikan-io/rustcommon/blob/main/histogram/src/histogram.rs
// Prototype histogram visualization:
//   https://observablehq.com/d/35f0b601ed888da9

use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::bit_vec::{BitVecFromSorted, MultiBitVec};
use num::One;
use num::Zero;
use std::debug_assert;

// Zero bins in the PDF manifest as repetitions in the CDF, so require a MultiBitVec
pub struct Histogram<V: MultiBitVec + BitVecFromSorted> {
    params: HistogramParams,

    // A bitvector representing the cumulative counts for each bin.
    cdf: V,

    // The total number of values added to this histogram
    count: V::Ones,
}

// Note: in this specific case, it seems like the following derives work, but they don't work for the sparse bitvec
// #[derive(bincode::Encode, bincode::Decode)]
// #[bincode(borrow_decode_bounds = "&'__de Histogram<V>: ::bincode::de::BorrowDecode<'__de> + '__de")]

impl<V: MultiBitVec> bincode::Encode for Histogram<V> {
    encode_impl!(params, cdf, count);
}
impl<V: MultiBitVec> bincode::Decode for Histogram<V> {
    decode_impl!(params, cdf, count);
}
impl<'de, V: MultiBitVec> bincode::BorrowDecode<'de> for Histogram<V> {
    borrow_decode_impl!(params, cdf, count);
}

impl<V: MultiBitVec> Histogram<V> {
    pub fn new(params: HistogramParams, cdf: V) -> Histogram<V> {
        let num_ones = cdf.num_ones();
        debug_assert!(num_ones == V::Ones::from_u32(params.num_bins()));
        let count = if num_ones.is_zero() {
            V::Ones::zero()
        } else {
            cdf.try_select1(num_ones - V::Ones::one()).unwrap()
        };
        Histogram { params, cdf, count }
    }

    /// Return an upper bound on the number of observations at or below `value`.
    pub fn cumulative_count(&self, value: V::Ones) -> V::Ones {
        // What is the index of the bin containing `value`?
        let bin_index = self.params.bin_index(value.into());
        // How many observations are in or below that bin?
        self.cdf.try_select1(V::Ones::from_u32(bin_index)).unwrap()
    }

    /// Return an upper bound on the value of the q-th quantile.
    pub fn quantile(&self, q: f64) -> V::Ones {
        // Number of observations at or below the q-th quantile
        let k = self.quantile_to_count(q);

        // Bin index of the bin containing the k-th observation
        let bin_index = self.cdf.rank1(k);

        // Maximum value in that bin
        let high = self.params.high(bin_index.u32());
        V::Ones::from_u64(high)
    }

    /// Return an upper bound on the number of observations that lie
    /// at or below the q-th quantile. E.g. if there are 2 observations,
    /// quantile_to_count(0) == 1, quantile_to_count(0.25) == 1,
    /// quantile_to_count(0.75) == 2, quantile_to_count(1.0) == 2
    /// todo: clarify this docstring - this returns a value in [1, count]
    pub fn quantile_to_count(&self, q: f64) -> V::Ones {
        debug_assert!((0.0..=1.0).contains(&q));
        if q == 0.0 {
            return V::one();
        }

        // Computes q*self.count rounded up by computing (1 - q)*count
        // rounded down, with rounding done implicitly by the cast to u64.
        // Using `as` to convert an `f64` into any integer type will
        // round towards zero inside representable range.
        let q = 1.0 - q;
        let count = (q * self.count.f64()) as u64;
        self.count - V::Ones::from_u64(count)
    }

    pub fn count(&self) -> V::Ones {
        self.count
    }

    pub fn params(&self) -> HistogramParams {
        self.params
    }

    pub fn builder(a: u32, b: u32, n: u32) -> HistogramBuilder<V> {
        HistogramBuilder::new(a, b, n)
    }

    pub fn encode(&self) -> Vec<u8> {
        let config = bincode::config::standard().with_fixed_int_encoding();
        bincode::encode_to_vec(self, config).unwrap()
    }

    pub fn decode(data: Vec<u8>) -> Self {
        let config = bincode::config::standard().with_fixed_int_encoding();
        let (ret, _) = bincode::decode_from_slice(&data, config).unwrap();
        ret
    }
}

pub struct HistogramBuilder<V: MultiBitVec> {
    params: HistogramParams,
    pdf: Box<[V::Ones]>,
}

impl<V: MultiBitVec> HistogramBuilder<V> {
    pub fn new(a: u32, b: u32, n: u32) -> HistogramBuilder<V> {
        let params = HistogramParams::new(a, b, n);
        let num_bins = params.num_bins();
        let pdf = vec![V::zero(); num_bins as usize].into();
        HistogramBuilder { params, pdf }
    }

    // Increment the count in a bin corresponding to a data value
    pub fn increment_value(&mut self, value: V::Ones, count: V::Ones) {
        let bin_index = self.params.bin_index(value.u64());
        self.pdf[bin_index as usize] += count;
    }

    // Increment the count in a bin at a particular index
    pub fn increment_index(&mut self, bin_index: usize, count: V::Ones) {
        self.pdf[bin_index] += count;
    }

    pub fn build(self) -> Histogram<V> {
        let mut acc = V::zero();
        let mut cdf = self.pdf;
        for x in cdf.iter_mut() {
            acc += *x;
            *x = acc;
        }
        // todo:
        Histogram::new(self.params, V::from_sorted(&cdf[..], acc + One::one()))
    }
}

#[derive(Copy, Clone, bincode::Encode, bincode::Decode)]
pub struct HistogramParams {
    // 2^a is the absolute error below the cutoff,
    // and is also the bin width below the cutoff.
    // there are 2^(b+1) bins below the cutoff, since
    // the cutoff is 2^(a+b+1) and the bin width is 2^a.
    a: u32,
    // 2^b is the number of bins per log segment above the cutoff, and
    // 2^-b is the relative error above the cutoff
    b: u32,
    // 2^n - 1 is the maximum value this histogram can store.
    n: u32,
    // 2^c = 2^(a+b+1) is the cutoff point below which are 2^(b+1)
    // fixed-width bins of width 2^a with absolute error, and above which
    // are log segments each with 2^b bins and a relative error of 2^-b.
    c: u32,
    // the number of bins in this histogram
    num_bins: u32,
}

impl HistogramParams {
    // note: in the classical parameterization, m = a and r = c, which implies that b = r - m - 1.
    fn new(a: u32, b: u32, n: u32) -> HistogramParams {
        let c = a + b + 1;
        // todo: assert that the number of bins is <= u32::MAX
        let num_bins = if n < c {
            // Each log segment is covered by bins of width 2^a and there are n log segments,
            // giving us 2^(n - a) bins in total. Also, we always want a minimum of 1 bin.
            1u32 << n.saturating_sub(a)
        } else {
            // See the comment in `bin_index` about `bins_below_seg` for a derivation
            (2 + n - c) << b
        };

        HistogramParams {
            a,
            b,
            c,
            n,
            num_bins,
        }
    }

    /// Return the bin index of the value given this histogram's parameters.
    pub fn bin_index(&self, value: u64) -> u32 {
        assert!(value <= self.max_value());
        let Self { a, b, c, .. } = *self;
        if value < (1 << c) {
            // We're below the cutoff.
            // The bin width below the cutoff is 1 << a
            (value >> a) as u32
        } else {
            // We're above the cutoff.

            // The log segment containing the value
            let v = value.ilog2();

            // The bin offset within the v-th log segment.
            // - `value - (1 << v)` zeros the topmost (v-th) bit.
            // - `>> (v - b)` extracts the top `b` bits of the value, corresponding
            //   to the bin index within the v-th log segment.
            let bins_within_seg = (value - (1 << v)) >> (v - b);

            // We want to calculate the number of bins that precede the v-th log segment.
            // 1. The linear section below the cutoff has twice as many bins as any log segment
            //    above the cutoff, for a total of 2^(b+1) = 2*2^b bins below the cutoff.
            // 2. Above the cutoff, there are `v - c` log segments before the v-th log segment,
            //    each with 2^b bins, for a total of (v - c) * 2^b bins above the cutoff.
            // Taken together, there are (v - c + 2) * 2^b bins preceding the v-th log segment.
            let bins_below_seg = (2 + v - c) << b;

            bins_below_seg + bins_within_seg as u32
        }
    }

    /// Given a bin index, returns the lowest value that bin can contain.
    pub fn low(&self, bin_index: u32) -> u64 {
        let Self { a, b, c, .. } = *self;
        let bins_below_cutoff = 2 << b;
        if bin_index < bins_below_cutoff {
            (bin_index << a) as u64
        } else {
            // the number of bins in 0..i that are above the cutoff point
            let n = bin_index - bins_below_cutoff;
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

    /// Given a bin index, returns the highest value that bin can contain.
    pub fn high(&self, bin_index: u32) -> u64 {
        if bin_index == self.num_bins() {
            self.max_value()
        } else {
            // the highest value of the i-th bin is the
            // integer right before the low of the next bin.
            self.low(bin_index + 1) - 1
        }
    }

    /// Return the maximum value representable by these histogram parameters.
    pub fn max_value(&self) -> u64 {
        if self.n == u64::BITS {
            u64::max_value()
        } else {
            (1 << self.n) - 1
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

    pub fn n(&self) -> u32 {
        self.n
    }

    pub fn num_bins(&self) -> u32 {
        self.num_bins
    }
}

// todo: randomized testing where we make a fully histogram with 0% error,
// and test that the error guarantees of the other histograms are achieved.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_multi_bit_vec::DenseMultiBitVec;
    use crate::slice_bit_vec::SliceBitVec;

    #[test]
    fn test_bin_index() {
        let params = HistogramParams::new(1, 2, 6);
        let bins = [
            // note: the last 2 values, currently 15, would be 16 if n was greater than 6.
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10,
            11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14,
            14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
        ];
        for (i, bin) in (0..64).zip(bins) {
            assert_eq!(params.bin_index(i), bin);
        }
        // todo: test error/panic if trying to bin a value that exceeds the max value
    }

    #[test]
    fn test_quantile_to_count() {
        let mut h = Histogram::<SliceBitVec<u32>>::builder(0, 1, 10);
        h.increment_value(0, 2);
        let h = h.build();
        assert_eq!(h.quantile_to_count(0.00), 1);
        assert_eq!(h.quantile_to_count(0.000001), 1);
        assert_eq!(h.quantile_to_count(0.49), 1);
        assert_eq!(h.quantile_to_count(0.50), 1);
        assert_eq!(h.quantile_to_count(1.00), 2);
    }

    #[test]
    fn percentiles_1() {
        let mut h = Histogram::<DenseMultiBitVec<u32>>::builder(0, 1, 10);
        for v in 1..1024 {
            h.increment_value(v, 1);
        }
        let h = h.build();
        let q_lo_hi = &[
            (0.01, 8, 11),
            (0.1, 96, 127),
            (0.25, 256, 383),
            (0.5, 512, 767),
            (0.75, 768, 1023),
            (0.9, 768, 1023),
            (0.99, 768, 1023),
            (0.9999, 768, 1023),
        ];

        for (q, _lo, hi) in q_lo_hi.iter().copied() {
            dbg!(q, hi);
            // assert_eq!(h.lowh.params.bin_index((q * h.count()) as u64), hi);
            assert_eq!(h.quantile(q), hi);
        }
    }

    #[test]
    fn percentiles_2() {
        let mut h = Histogram::<SliceBitVec<u32>>::builder(0, 4, 10);
        for v in 1..1024 {
            h.increment_value(v, 1);
        }
        let h = h.build();

        let q_lo_hi = &[
            (0.01, 11, 11),
            (0.1, 100, 103),
            (0.25, 256, 271),
            (0.5, 512, 543),
            (0.75, 768, 799),
            (0.9, 896, 927),
            (0.99, 992, 1023),
            (0.9999, 992, 1023),
        ];

        for (q, _lo, hi) in q_lo_hi.iter().copied() {
            dbg!(q, hi);
            // assert_eq!(h.lowh.params.bin_index((q * h.count()) as u64), hi);
            assert_eq!(h.quantile(q), hi);
        }
    }

    #[test]
    fn percentiles_3() {
        let mut h = Histogram::<SliceBitVec<u32>>::builder(0, 9, 30);
        h.increment_value(1, 1);
        h.increment_value(10_000_000, 1);
        let h = h.build();
        assert_eq!(h.quantile(0.0), 1);
        assert_eq!(h.quantile(0.25), 1);
        assert_eq!(h.quantile(0.75), 10010623);
        assert_eq!(h.quantile(1.0), 10010623);
    }

    #[test]
    fn num_bins() {
        // a scattershot of checks to ensure that the number of bins is computed correctly
        assert_eq!(HistogramParams::new(0, 0, 0).num_bins(), 1);
        assert_eq!(HistogramParams::new(0, 0, 6).num_bins(), 7);
        assert_eq!(HistogramParams::new(0, 0, 7).num_bins(), 8);
        assert_eq!(HistogramParams::new(0, 2, 6).num_bins(), 20);
        assert_eq!(HistogramParams::new(1, 2, 6).num_bins(), 16);
        assert_eq!(HistogramParams::new(1, 0, 6).num_bins(), 6);
        assert_eq!(HistogramParams::new(1, 1, 6).num_bins(), 10);
        assert_eq!(HistogramParams::new(0, 1, 6).num_bins(), 12);
        assert_eq!(HistogramParams::new(2, 0, 4).num_bins(), 3);
        assert_eq!(HistogramParams::new(2, 1, 4).num_bins(), 4);
        assert_eq!(HistogramParams::new(2, 2, 6).num_bins(), 12);
        assert_eq!(HistogramParams::new(2, 2, 5).num_bins(), 8);
        assert_eq!(HistogramParams::new(2, 2, 4).num_bins(), 4);
        assert_eq!(HistogramParams::new(2, 3, 3).num_bins(), 2);
        assert_eq!(HistogramParams::new(2, 3, 2).num_bins(), 1);
        assert_eq!(HistogramParams::new(2, 3, 1).num_bins(), 1);
        assert_eq!(HistogramParams::new(2, 3, 0).num_bins(), 1);

        assert_eq!(HistogramParams::new(32, 0, 32).num_bins(), 1);

        // todo: this should work, one day, or error.
        // all linear bins below a cutoff of 2^32.
        // assert_eq!(HistogramParams::new(0, 31, 32).num_bins(), u32::MAX);
    }
}
