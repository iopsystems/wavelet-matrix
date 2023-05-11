use crate::bitvector::BitVector;

// This is the integer type used for blocks of bits.
type BitBlockType = u64;

// A very simple bit vector supporting O(1) rank queries and O(log(n)) select queries.
#[derive(Debug, bincode::Encode, bincode::Decode)]
pub struct SimpleBitVector {
    // Store bits in blocks. Each block stores its bits in order from LSB to MSB.
    blocks: Vec<BitBlockType>,
    // Rank superblocks store cumulative ranks.
    // rank_superblocks[i] represents the rank up to and including the i-th block.
    // Right now, each rank superblock represents exactly one bit block, but a
    // future space improvement would be to configure the block sampling rate and have
    // rank_superblocks[i] store the rank up to and including the (sr * i)-th block.
    // See this paper for details: Fast, Small, Simple Rank/Select on Bitmaps
    // (https://users.dcc.uchile.cl/~gnavarro/ps/sea12.1.pdf)
    //
    // Note: Browser WebAssembly uses a 32-bit architecture, so these take 50% of
    // the space of blocks. We might consider changing this to a Vec<u32>, which would
    // save space on 64-bit architectures but limit us to representing bit vectors of
    // length up to 2^32-1.
    rank_superblocks: Vec<usize>,
    // Number of bits in this BitVector
    len: usize,
    // Number of one bits in this BitVector
    num_ones: usize,
    // Index of the bit beyond the rightmost bit in this bitvector.
    // If there are no ones, this value is zero. This means that we
    // cannot faithfully represent all bit vectors of length usize::MAX
    // so we disallow that length in the builder.
    max_one_index_plus_one: usize,
}

impl SimpleBitVector {
    // These select1 and select0 implementations use binary search over the array
    // without a select-based acceleration index, and are thus O(log(self.len)).
    // Both support hinted search: the search range can be specified through extra
    // input arguments for those cases where the sought-after bit is known to be
    // confined to a particular index range.
    // Optimization opportunities:
    // - Use sampled select blocks could cut down the search range
    // - Perform exponential rather than binary search
    // - Investigate using binary search on rank blocks to refine the hinted range prior
    //   to calling this function (or calling a different function to binary search bits
    //   within a block, or doing linear search, depending on the block distance between
    //   rank samples). I think we would use `partition_point`.
    // todo: accept a range of indices?
    // todo: document that this is [L, R)
    // todo: can we rewrite hinted select in terms of the binary search functions in utils?
    pub fn hinted_select1(&self, index: usize, mut left: usize, mut right: usize) -> Option<usize> {
        if index < 1 || index > self.num_ones {
            return None;
        }
        while left < right {
            let mid = (left + right) >> 1;
            if self.rank1(mid) < index {
                left = mid + 1
            } else {
                right = mid
            }
        }
        Some(left)
    }

    pub fn hinted_select0(&self, n: usize, mut left: usize, mut right: usize) -> Option<usize> {
        let num_zeros = self.len - self.num_ones;
        if n < 1 || n > num_zeros {
            return None;
        }
        while left < right {
            let mid = (left + right) >> 1;
            if self.rank0(mid) < n {
                left = mid + 1
            } else {
                right = mid
            }
        }
        Some(left)
    }

    // Use a builder pattern to construct a BitVector one by one.
    pub fn builder(len: usize) -> SimpleBitVectorBuilder {
        assert!(len < usize::MAX); // see comment on BitVector.max_one_index_plus_one
        SimpleBitVectorBuilder {
            blocks: Vec::new(),
            max_one_index_plus_one: 0,
            len,
        }
    }
}

impl BitVector for SimpleBitVector {
    // Returns the number of one bits at or below the bit index `index`.
    // It can be mathematically useful to take ranks beyond the length of
    // the vector, so we allow `index` to exceed `len`, such as when we
    // traverse power-of-two-sized nodes in the wavelet matrix.
    fn rank1(&self, index: usize) -> usize {
        // If we're past the last one index, we can avoid needing
        // to access the blocks in memory.
        // todo: investigate whether adding extra block_index < blocks.len()-style
        // checks to communicate to the compiler that bounds checks can be removed
        // is worth the performnce increase, if there is one.
        if index >= self.max_one_index_plus_one {
            return self.num_ones;
        }
        let (block_index, bit_offset) = block_index_and_offset(index);
        let block = self.blocks[block_index];

        // The rank superblock tells us how many ones there are in
        // all of the blocks up to and including this one.
        let rank_superblock = self.rank_superblocks[block_index];

        // But what we care about is how many ones there are specifically
        // up to and including our bit offset, which means we need to adjust the
        // number in the rank superblock by subtracting the number of ones inside
        // this block that lie beyond that bit index, which we can count with
        // this mask. Its low (bit_offset+1) bits are zero and the rest are ones.
        let mask = BitBlockType::MAX
            .checked_shl((bit_offset + 1) as u32)
            .unwrap_or(0);
        rank_superblock - (block & mask).count_ones() as usize
    }

    // Returns the number of zero bits at or below index `i`.
    fn rank0(&self, index: usize) -> usize {
        if index >= self.len {
            return self.len - self.num_ones;
        }
        index - self.rank1(index) + 1
    }

    // Returns the value of the `i`-th bit as a bool.
    fn access(&self, index: usize) -> bool {
        assert!(index < self.len, "out of bounds");
        // After the last one bit are only zeros.
        if index >= self.max_one_index_plus_one {
            return false;
        }
        let (block_index, bit_offset) = block_index_and_offset(index);
        let block = self.blocks[block_index];
        // Mask out the target bit
        let mask = 1 << bit_offset;
        block & mask > 0
    }

    fn select1(&self, n: usize) -> Option<usize> {
        // Use the max one index for the upper value of the range
        // when searching for one bits, since there are none after.
        self.hinted_select1(n, 0, self.max_one_index_plus_one)
    }

    fn select0(&self, n: usize) -> Option<usize> {
        self.hinted_select0(n, 0, self.len)
    }

    fn len(&self) -> usize {
        self.len
    }

    fn num_ones(&self) -> usize {
        self.num_ones
    }

    fn num_zeros(&self) -> usize {
        self.len - self.num_ones
    }
}

// BitVectorBuilder encodes partial state during construction so we never have an invalid BitVector.
#[derive(Clone)]
pub struct SimpleBitVectorBuilder {
    blocks: Vec<BitBlockType>,
    max_one_index_plus_one: usize,
    len: usize,
}

impl SimpleBitVectorBuilder {
    pub fn one(&mut self, index: usize) {
        assert!(index < self.len, "out of bounds");
        let (block_index, bit_offset) = block_index_and_offset(index);

        // Resize blocks if needed
        if self.blocks.len() <= block_index {
            self.blocks.resize(block_index + 1, 0);
        }

        // Update the value tracking the maximum one index if needed
        // Note: This could be computed in constant time in build by
        // taking at the total number of bits in this bitvector and
        // subtracting the number of leading ones in the final block.
        if index >= self.max_one_index_plus_one {
            self.max_one_index_plus_one = index + 1
        }

        // Set the one bit at the appropriate offset
        self.blocks[block_index] |= 1 << bit_offset;
    }

    pub fn build(self) -> SimpleBitVector {
        // Compute the cumulative sum of the number of one bits in each block
        let mut rank_superblocks = Vec::with_capacity(self.blocks.len());
        let mut num_ones = 0;
        for &block in self.blocks.iter() {
            num_ones += block.count_ones() as usize;
            rank_superblocks.push(num_ones);
        }

        SimpleBitVector {
            blocks: self.blocks,
            rank_superblocks,
            num_ones,
            max_one_index_plus_one: self.max_one_index_plus_one,
            len: self.len,
        }
    }
}

// Returns the block index and bit offset of the i-th bit.
fn block_index_and_offset(i: usize) -> (usize, usize) {
    // block index of the block containing the i-th bit
    let block_index = i >> BitBlockType::BITS.ilog2();

    // bit index of the bit within that block (mask off the high bits)
    let bit_offset = i & (BitBlockType::BITS - 1) as usize;

    (block_index, bit_offset)
}

// todo:
// - consider implementing BitVector::with_capacity to preallocate the appropriate number of blocks in advance
// - consider differences between initializing with ascending ones vs. in random order – should we try to optimize
//   the ascending case?
