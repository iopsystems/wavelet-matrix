# Bit Structures

Simple implementations of basic bit-based data structures. Designed for use with static data (construct once, query many times)

# Questions

- This is intended for wasm use. So why not just specialize to u32 values everywhere?


## Potential future optimizations

- Improve the performance of binary search (see the appendix in utils.rs)
- 

## Limitations

The lowest-level bitvector types (`FixedWidthIntVec`, `RawBitVec`) assume that that the target architecture is little-endian.

## References

- [simple-sds](https://github.com/jltsiren/simple-sds/)

## To do

- add a column-oriented summed-area-table implementation for quick ranged access to EF- or potentially other-encoded bitvectors. This could be used for histograms if we want fast range access and don't want to use a wavelet matrix.
- add multiset support to the naïve bitvector which would be useful for the summed-area table if we want to store zeros.
- add a 'store the multiplicities in an EF-encoded bitvec, and the positions of nonzero elements in a dense bitvec' for relatively short sparse multisets.
- Optimize raw bitvector to allow for leading and trailing zeros
- Create a `FlippedBitVec<T>` that corresponds to the negated version of a bitvector; 1-bits become 0-bits and vice versa. Essentially, reroute select0/1 to 1/0 and the same for rank and count ones/zeros, and negate `get`.
- Port relevant tests from simple-sds, which is MIT-licensed
- make EF split point configurable
- prototype the sparse representation of a Rezolus histogram: store the cumulative weights for nonzero buckets in an EF vector + the indices of nonzero buckets in a dense bitvector.
- Consider making a PackedIntVec for fixed width integers (the lower half of EF). For now can use the bitbuffer library.
- Write some documentation about what (if any) assumptions we make about endianness in eg. intvector and bitvector
  - https://fgiesen.wordpress.com/2018/02/19/reading-bits-in-far-too-many-ways-part-1/:
    > LSB-first natural packing gives us the same bytes as LSB-first packing into a big integer then storing it in little-endian byte order
- Investigate whether performance is improved by making some blocks u8, eg. in `RawBitVec`. Maybe that reduces the memory bandwidth, though I'm not sure how unsafe it would be to try reinterpreting that memory into u32 or u64 or u128 blocks for popcounts/simd operations.
- figure out how to flexibly support any combination of select0 and select1
- Implement an RLE bitvec (probably mine, with fast rank)
- Consider a simple impl of Roaring as an elegant proof of concept, with just rank/select. Maybe this can be our 'compressed' variant.
- Implement a simple array-backed sparse bitvec
- Implement a 'dense multiset' where the universe is dense but each entry can be weighted, backed by a dense + sparse – dense to store nonzeros, sparse to store cumulative weights. Not a bitvector so can have different funcs than rank/select. This can then be the rezolus histogram repr. I think this was described in the RRR paper, and I originally implemented it with some "zero-compressed bitmap" scheme, but can now do it with EF.
- implement set ops (and, or, not) on raw bitvectors
- implement a general mechanism for constructing a bitvector of arbitrary type, accepting a sorted iterator of ones (sorted because some, eg. sparse, require it). Maybe part of the bitvector trait.
- test naive bitvector extensively, since it is our ground truth (can do this in the form of concrete tests against the bitvector trait, which would cover the naive bitvector as a special case)