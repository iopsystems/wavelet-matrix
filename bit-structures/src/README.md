# Bit Structures

## Potential future optimizations

- Improve the performance of binary search (see the appendix in utils.rs)
- 

## References

- [simple-sds](https://github.com/jltsiren/simple-sds/)

## To do

- Port relevant tests from simple-sds, which is MIT-licensed
- make EF split point configurable
- prototype the sparse representation of a Rezolus histogram: store the cumulative weights for nonzero buckets in an EF vector + the indices of nonzero buckets in a dense bitvector.
- Consider making a PackedIntVector for fixed width integers (the lower half of EF). For now can use the bitbuffer library.
- Write some documentation about what (if any) assumptions we make about endianness in eg. intvector and bitvector
  - https://fgiesen.wordpress.com/2018/02/19/reading-bits-in-far-too-many-ways-part-1/:
    > LSB-first natural packing gives us the same bytes as LSB-first packing into a big integer then storing it in little-endian byte order