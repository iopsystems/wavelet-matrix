// I want three histograms:
// - naive raw pdf histogram backed by a CDF array, with an entry per bucket; can be incremented
//   - maybe this is a builder (though we don't have those for anything else)
//   - .to_dense() .to_sparse()
// - dense EF-compressed static histogram with repetitions for zero buckets
// - sparse EF-compressed static histogram with a dense bitvector marking nonzero buckets (essentially a compact weighted multiset representation)

struct Histogram {
    a: u32,
    b: u32,
}
