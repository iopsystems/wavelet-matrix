// I want three histograms:
// - naive raw pdf histogram backed by a CDF array, with an entry per bucket; can be incremented
//   - maybe this is a builder (though we don't have those for anything else)
//   - .to_dense() .to_sparse()
// - dense EF-compressed static histogram with repetitions for zero buckets
// - sparse EF-compressed static histogram with a dense bitvector marking nonzero buckets (essentially a compact weighted multiset representation)

use crate::sparse_bit_vec::SparseBitVec;

struct Histogram {
    a: u32,
    b: u32,
    n: u32,
    cdf: SparseBitVec,
}

struct HistogramBuilder {
    a: u32,
    b: u32,
    pdf: Box<[u32]>,
}

impl HistogramBuilder {
    // fn new(a: u32, b: u32, n: u32) -> HistogramBuilder {
    //     // let c = b + a + 1;
    //     let bin_width_below_cutoff = 1 << a;
    //     let bins_per_log_segment = 1 << b;
    //     let linear_log_cutoff = 1 << (b + a + 1); // C
    //     let max_value = (1u64 << n) - 1;
    //     let len = 10;
    //     let pdf = vec![0, len].into_boxed_slice();
    //     HistogramBuilder { a, b, pdf }
    // }

    // fn build() -> Histogram {
    //     // let cdf = SparseBitVec::new(ones, len);
    //     Histogram { a: 0, b: 0, cdf }
    // }

    // fn bin_index(value: u64) {
    //     if value < linear_log_cutoff {
    //         value >> a
    //     } else {
    //         let h = value.ilog2();
    //         let num_preceding_log_bins = h - num_bins_below_cutoff;
    //         let offset_in_this_log_bin = (value - (1 << h)) >> (h - b);
    //         num_bins_below_cutoff + 0 + offset_in_this_log_bin
    //     }
    //     // https://observablehq.com/d/35f0b601ed888da9
    //     // index_base2 = (value) => {
    //     //   console.assert(base == 2);
    //     //   if (value < C) {
    //     //     return value >>> a;
    //     //   } else {
    //     //     const h = floor_log2(value);
    //     //     const numPrecedingLogBins = h - c;
    //     //     const offsetInThisLogBin = (value - (1 << h)) >>> (h - b);
    //     //     return (
    //     //       totalCountInLinearBins +
    //     //       numLinearStepsPerLogBin * numPrecedingLogBins +
    //     //       offsetInThisLogBin
    //     //     );
    //     //   }
    //     // }
    // }
}
