use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::morton;
use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec};
use num::{One, PrimInt, Zero};
use std::{collections::VecDeque, ops::Range};

// todo
// - figure out if we ever recurse with empty ranges (can add an assert in traverse)
//   - i think we can check start.0 != end.0 and start.1 != end.1
// - verify that the intermediate traversals are indeed in ascending wavelet matrix order
// - consider a SymbolCount struct rather than returning tuples
// - ignore_bits
// - batch queries
// - set operations on multiple ranges: union, intersection, ...
// - functions that accept symbol or index ranges should accept .. and x.. and ..x
//   - i think this can be implemented with a trait that has an 'expand' method, or
//     by accepting a RangeBounds and writing a  fn that replaces unbounded with 0 or len/whatever.
// - document the 'wm as occupancy, plus external bit-reversed-bin-sorted cumulative sum array for
// weights at the lowest wm level' technique, as mentioned in the email thread with gonzalo navarro.

type Dense = DenseBitVec<u32, u8>;

#[derive(Debug, Copy, Clone)]
pub struct CountAll<T: BitBlock> {
    pub symbol: T,
    pub start: T,
    pub end: T,
}
impl<T: BitBlock> CountAll<T> {
    fn new(symbol: T, start: T, end: T) -> Self {
        Self { symbol, start, end }
    }
}

// Helper for traversing the wavelet matrix level by level,
// reusing space when possible and keeping the elements in
// sorted order with respect to the ordering of wavelet tree
// nodes in the wavelet matrix (all left nodes precede all
// right nodes).
#[derive(Debug)]
pub struct Traversal<T> {
    cur: VecDeque<KeyValue<T>>,
    next: VecDeque<KeyValue<T>>,
    num_left: usize,
}

// Traverse a wavelet matrix levelwise, at each level maintaining tree nodes
// in order they appear in the wavelet matrix (left children preceding right).
impl<T> Traversal<T> {
    fn new() -> Self {
        Self {
            cur: VecDeque::new(),
            next: VecDeque::new(),
            num_left: 0,
        }
    }

    fn init(&mut self, values: impl IntoIterator<Item = T>) {
        let iter = values.into_iter().enumerate().map(KeyValue::from_tuple);
        self.cur.clear();
        self.next.clear();
        self.next.extend(iter);
        self.num_left = 0;
    }

    fn traverse(&mut self, mut f: impl FnMut(&[KeyValue<T>], &mut Go<KeyValue<T>>)) {
        // precondition: `next` contains things to traverse.
        // postcondition: `next` has the next things to traverse, with (reversed)
        // left children followed by (non-reversed) right children, and num_left
        // indicating the number of left elements.

        // swap next into cur, then clear next
        std::mem::swap(&mut self.cur, &mut self.next);
        self.next.clear();

        // note: rather than reversing the left subtree in advance, here's an idea:
        // we could potentially call the callback twice per level, once with the
        // left iterator reversed, then the right iterator. this gets tricky in terms
        // of the types since the two iterators would be of different types.
        // If we do this, the left slice is cur[..self.num_left] and the right slice
        // is cur[self.num_left..].
        let cur = self.cur.make_contiguous();
        cur[..self.num_left].reverse();

        // for lifetime reasons (to avoid having to pass &mut self into f), create
        // an auxiliary structure to let f recurse left and right.
        let mut go = Go {
            next: &mut self.next,
            num_left: 0,
        };

        // invoke the traversal function with the current elements and the recursion helper
        // we pass an iterator rather than an element at a time so that f can do its own
        // batching if it wants to
        f(cur, &mut go);

        // update the number of nodes that went left based on the calls `f` made to `go`
        self.num_left = go.num_left;
    }

    pub fn results(&mut self) -> &mut [KeyValue<T>] {
        let slice = self.next.make_contiguous();
        // note: reverse only required if we want to return results in wm order,
        // which might be nice if we are eg. looking up associated data.
        slice[..self.num_left].reverse();

        self.num_left = 0; // update this so that calling results multiple times does not re-reverse the left
        slice
    }
}

struct Go<'a, T> {
    next: &'a mut VecDeque<T>,
    num_left: usize,
}

impl<T> Go<'_, T> {
    fn left(&mut self, kv: T) {
        // left children are appended to the front of the queue
        // which causes them to be in reverse order
        self.next.push_front(kv);
        self.num_left += 1;
    }
    fn right(&mut self, kv: T) {
        // right children are appended to the back of the queue
        self.next.push_back(kv);
    }
}
// The traversal order means that outputs do not appear in the same order as inputs and
// there may be multiple outputs per input (e.g. symbols found within a given index range)
// so associating each batch with an index allows us to track the association between inputs
// and outputs.
// The Key is (currently) the input index associated with this query, so we can track it through
// the tree.
#[derive(Debug, Copy, Clone, PartialEq, bincode::Encode, bincode::Decode)]
pub struct KeyValue<T> {
    pub key: usize,
    pub value: T,
}

// Associate a usize key to an arbitrary value; used for propagating the metadata
// of which original query element a partial query result is associated with as we
// traverse the wavelet tree
impl<T> KeyValue<T> {
    fn new(key: usize, value: T) -> KeyValue<T> {
        KeyValue { key, value }
    }
    // construct a BatchValue from an (key, value) tuple
    fn from_tuple((key, value): (usize, T)) -> KeyValue<T> {
        KeyValue { key, value }
    }
    fn map<U>(self, f: impl FnOnce(T) -> U) -> KeyValue<U> {
        KeyValue {
            key: self.key,
            value: f(self.value),
        }
    }
    // return a new KeyValue with the previous key and new value
    fn value(self, value: T) -> KeyValue<T> {
        KeyValue { value, ..self }
    }
}

#[derive(Debug)]
pub struct WaveletMatrix<V: BitVec> {
    levels: Vec<Level<V>>, // wm levels (bit planes)
    max_symbol: u32,       // maximum symbol value
    len: V::Ones,          // number of symbols
}

impl<V: BitVec> bincode::Encode for WaveletMatrix<V> {
    encode_impl!(levels, max_symbol, len);
}
impl<V: BitVec> bincode::Decode for WaveletMatrix<V> {
    decode_impl!(levels, max_symbol, len);
}
impl<'de, V: BitVec> bincode::BorrowDecode<'de> for WaveletMatrix<V> {
    borrow_decode_impl!(levels, max_symbol, len);
}

// store (rank0, rank1)
#[derive(Copy, Clone)]
struct Ranks<T>(T, T);

struct RankCache<V: BitVec> {
    end_index: Option<V::Ones>, // previous end index
    end_ranks: Ranks<V::Ones>,  // previous end ranks
    // note: we track these just out of interest;
    // we could enable only when profiling.
    num_hits: usize,   // number of cache hits
    num_misses: usize, // number of cache misses
}

impl<V: BitVec> RankCache<V> {
    fn new() -> Self {
        Self {
            end_index: None,
            end_ranks: Ranks(V::Ones::zero(), V::Ones::zero()),
            num_hits: 0,
            num_misses: 0,
        }
    }

    fn get(
        &mut self,
        start_index: V::Ones,
        end_index: V::Ones,
        level: &Level<V>,
    ) -> (Ranks<V::Ones>, Ranks<V::Ones>) {
        let start_ranks = if Some(start_index) == self.end_index {
            self.num_hits += 1;
            self.end_ranks
        } else {
            self.num_misses += 1;
            level.ranks(start_index)
        };
        self.end_index = Some(end_index);
        self.end_ranks = level.ranks(end_index);
        (start_ranks, self.end_ranks)
    }

    fn log_stats(&self) {
        log::info!(
            "cached {:.1}%: {:?} / {:?}",
            100.0 * self.num_hits as f64 / (self.num_hits + self.num_misses) as f64,
            self.num_hits,
            self.num_hits + self.num_misses,
        );
    }
}

impl WaveletMatrix<Dense> {
    pub fn new(data: Vec<u32>, max_symbol: u32) -> WaveletMatrix<Dense> {
        let num_levels = num_levels_for_symbol(max_symbol);
        // We implement two different wavelet matrix construction algorithms. One of them is more
        // efficient, but that algorithm does not scale well to large alphabets and also cannot
        // cannot handle element multiplicity because it constructs the bitvectors out-of-order.
        // It also requires O(2^num_levels) space. So, we check whether the number of data points
        // is less than 2^num_levels, and if so use the scalable algorithm, and otherise use the
        // the efficient algorithm.
        let levels = if num_levels <= (data.len().ilog2() as usize) {
            build_bitvecs(data, num_levels)
        } else {
            build_bitvecs_large_alphabet(data, num_levels)
        };

        WaveletMatrix::from_bitvecs(levels, max_symbol)
    }
}

// Wavelet matrix construction algorithm optimized for the case where we can afford to build a
// dense histogram that counts the number of occurrences of each symbol. Heuristically,
// this is roughly the case where the alphabet size does not exceed the number of data points.
// Implements Algorithm 1 (seq.pc) from the paper "Practical Wavelet Tree Construction".
fn build_bitvecs(data: Vec<u32>, num_levels: usize) -> Vec<Dense> {
    let mut levels = vec![BitBuf::new(data.len()); num_levels];
    let mut hist = vec![0; 1 << num_levels];
    let mut borders = vec![0; 1 << num_levels];
    let max_level = num_levels - 1;

    {
        // Count symbol occurrences and fill the first bitvector, whose bits
        // can be read from MSBs of the data in its original order.
        let level = &mut levels[0];
        let level_bit = 1 << max_level;
        for (i, &d) in data.iter().enumerate() {
            hist[d as usize] += 1;
            if d & level_bit > 0 {
                level.set(i);
            }
        }
    }

    // Construct the other levels bottom-up
    for l in (1..num_levels).rev() {
        // The number of wavelet tree nodes at this level
        let num_nodes = 1 << l;

        // Compute the histogram based on the previous level's histogram
        for i in 0..num_nodes {
            // Update the histogram in-place
            hist[i] = hist[2 * i] + hist[2 * i + 1];
        }

        // Get starting positions of intervals from the new histogram
        borders[0] = 0;
        for i in 1..num_nodes {
            // Update the positions in-place. The bit reversals map from wavelet tree
            // node order to wavelet matrix node order, with all left children preceding
            // the right children.
            let prev_index = reverse_low_bits(i - 1, l);
            borders[reverse_low_bits(i, l)] = borders[prev_index] + hist[prev_index];
        }

        // Fill the bit vector of the current level
        let level = &mut levels[l];
        let level_bit_index = max_level - l;
        let level_bit = 1 << level_bit_index;

        // This mask contains all ones except for the lowest level_bit_index bits.
        let bit_prefix_mask = usize::MAX
            .checked_shl((level_bit_index + 1) as u32)
            .unwrap_or(0);
        for &d in data.iter() {
            // Get and update position for bit by computing its bit prefix from the
            // MSB downwards which encodes the path from the root to the node at
            // this level that contains this bit
            let node_index = (d as usize & bit_prefix_mask) >> (level_bit_index + 1);
            let p = &mut borders[node_index];
            // Set the bit in the bitvector
            if d & level_bit > 0 {
                level.set(*p);
            }
            *p += 1;
        }
    }

    levels
        .into_iter()
        .map(|level| DenseBitVec::new(level, 10, 10))
        .collect()
}

// Returns an array of level bitvectors built from `data`.
// Handles the sparse case where the alphabet size exceeds the number of data points and
// building a histogram with an entry for each symbol is expensive
fn build_bitvecs_large_alphabet(mut data: Vec<u32>, num_levels: usize) -> Vec<Dense> {
    let mut levels = Vec::with_capacity(num_levels);
    let max_level = num_levels - 1;

    // For each level, stably sort the datapoints by their bit value at that level.
    // Elements with a zero bit get sorted left, and elements with a one bits
    // get sorted right, which is effectvely a bucket sort with two buckets.
    let mut right = Vec::new();

    for l in 0..max_level {
        let level_bit = 1 << (max_level - l);
        let mut bits = BitBuf::new(data.len());
        let mut index = 0;
        // Stably sort all elements with a zero bit at this level to the left, storing
        // the positions of all one bits at this level in `bits`.
        // We retain the elements that went left, then append those that went right.
        data.retain_mut(|d| {
            let value = *d;
            let go_left = value & level_bit == 0;
            if !go_left {
                bits.set(index);
                right.push(value);
            }
            index += 1;
            go_left
        });
        data.append(&mut right);
        levels.push(DenseBitVec::new(bits, 10, 10));
    }

    // For the last level we don'T need to do anything but build the bitvector
    {
        let mut bits = BitBuf::new(data.len());
        let level_bit = 1 << 0;
        for (index, d) in data.iter().enumerate() {
            if d & level_bit > 0 {
                bits.set(index);
            }
        }
        levels.push(DenseBitVec::new(bits, 10, 10));
    }

    levels
}

#[derive(Debug)]
struct Level<V: BitVec> {
    bv: V,
    num_zeros: V::Ones,
    // unsigned int with a single bit set signifying
    // the magnitude represented at that level.
    // e.g.  levels[0].bit == 1 << levels.len() - 1
    bit: V::Ones,
}

impl<V: BitVec> bincode::Encode for Level<V> {
    encode_impl!(bv, num_zeros, bit);
}
impl<V: BitVec> bincode::Decode for Level<V> {
    decode_impl!(bv, num_zeros, bit);
}
impl<'de, V: BitVec> bincode::BorrowDecode<'de> for Level<V> {
    borrow_decode_impl!(bv, num_zeros, bit);
}

impl<V: BitVec> Level<V> {
    // Returns (rank0(index), rank1(index))
    // This means that if x = ranks(index), x.0 is rank0 and x.1 is rank1.
    pub fn ranks(&self, index: V::Ones) -> Ranks<V::Ones> {
        if index.is_zero() {
            return Ranks(V::zero(), V::zero());
        }
        let num_ones = self.bv.rank1(index);
        let num_zeros = index - num_ones;
        Ranks(num_zeros, num_ones)
    }
}

/// Reverse the lowest `numBits` bits of `v`. For example:
///
/// assert!(reverse_low_bits(0b0000100100, 6) == 0b0000001001)
/// //                     ^^^^^^              ^^^^^^
///
// (todo: figure out how to import this function so the doctest
// can be run...)
fn reverse_low_bits(x: usize, num_bits: usize) -> usize {
    x.reverse_bits() >> (usize::BITS as usize - num_bits)
}

// todo: is this bit_ceil? seems to be bit_ceil(symbol-1)
pub fn num_levels_for_symbol(symbol: u32) -> usize {
    // Equivalent to max(1, ceil(log2(alphabet_size))), which ensures
    // that we always have at least one level even if all symbols are 0.
    (u32::BITS - symbol.leading_zeros())
        .max(1)
        .try_into()
        .unwrap()
}

impl<V: BitVec> WaveletMatrix<V> {
    pub fn from_bitvecs(levels: Vec<V>, max_symbol: u32) -> WaveletMatrix<V> {
        let max_level = levels.len() - 1;
        let len = levels.first().map(|level| level.universe_size()).unwrap();
        let levels: Vec<Level<V>> = levels
            .into_iter()
            .enumerate()
            .map(|(index, bits)| Level {
                num_zeros: bits.rank0(bits.universe_size()),
                bit: V::one() << (max_level - index),
                bv: bits,
            })
            .collect();
        WaveletMatrix {
            levels,
            max_symbol,
            len,
        }
    }

    // Locates a symbol on the virtual bottom level of the wavelet tree.
    // Returns two things, both restricted to the query range:
    // - the number of symbols preceding this one in sorted order (less than)
    // - the range of this symbol on the virtual bottom level
    // This function is designed for internal use, where knowing the precise
    // range on the virtual level can be useful, e.g. for select queries.
    // Since the range also tells us the count of this symbol in the range, we
    // can combine the two pieces of data together for a count-less-than-or-equal query.
    // We compute both of these in one function since it's pretty cheap to do so.
    fn locate(
        &self,
        symbol: V::Ones,
        range: Range<V::Ones>,
        ignore_bits: usize,
    ) -> (V::Ones, Range<V::Ones>) {
        let mut preceding_count = V::Ones::zero();
        let mut range = range;
        for level in self.levels(ignore_bits) {
            let start = level.ranks(range.start);
            let end = level.ranks(range.end);
            // check if the symbol's level bit is set to determine whether it should be mapped
            // to the left or right child node
            if (symbol & level.bit).is_zero() {
                // go left
                range = start.0..end.0;
            } else {
                // count the symbols in the left child before going right
                preceding_count += end.0 - start.0;
                range = level.num_zeros + start.1..level.num_zeros + end.1;
            }
        }
        (preceding_count, range)
    }

    // number of symbols less than this one, restricted to the query range
    pub fn preceding_count(&self, symbol: V::Ones, range: Range<V::Ones>) -> V::Ones {
        self.locate(symbol, range, 0).0
    }

    // number of times the symbol appears in the query range
    pub fn count(&self, symbol: V::Ones, range: Range<V::Ones>) -> V::Ones {
        let range = self.locate(symbol, range, 0).1;
        range.end - range.start
    }

    pub fn quantile(&self, k: V::Ones, range: Range<V::Ones>) -> (V::Ones, V::Ones) {
        assert!(k < range.end - range.start);
        let mut k = k;
        let mut range = range;
        let mut symbol = V::zero();
        for level in self.levels(0) {
            let start = level.ranks(range.start);
            let end = level.ranks(range.end);
            let left_count = end.0 - start.0;
            if k < left_count {
                // go left
                range = start.0..end.0;
            } else {
                // go right
                k -= left_count;
                symbol += level.bit;
                range = level.num_zeros + start.1..level.num_zeros + end.1;
            }
        }
        let count = range.end - range.start;
        (symbol, count)
    }

    pub fn select(&self, symbol: V::Ones, k: V::Ones, range: Range<V::Ones>) -> Option<V::Ones> {
        // track the symbol down to a range on the bottom level
        let range = self.locate(symbol, range, 0).1;

        // If there are fewer than `k+1` copies of `symbol` in the range, return early.
        if k < (range.end - range.start) {
            return None;
        }

        // track the k-th occurrence of the symbol up from the bottom-most virtual level
        let mut index = range.start + k;

        for level in self.levels.iter().rev() {
            let nz = level.num_zeros;
            // `index` represents an index on the level below this one, which may be
            // the bottom-most 'virtual' layer that contains all symbols in sorted order.
            //
            // We want to determine the position of the element represented by `index` on
            // this level, which we can do by "mapping" the index up to its parent node.
            //
            // `nz` tells us how many bits on the level below come from left children of
            // the wavelet tree represented by this wavelet matrix. If the index < nz, that
            // means that the index on the level below came from a left child on this level,
            // which means that it must be represented by a 0-bit on this level; specifically,
            // the `index`-th 0-bit, since the WT always represents a stable sort of its elements.
            //
            // On the other hand, if `index` came from a right child on this level, then it
            // is represented by a 1-bit on this level; specifically, the `index - nz`-th 1-bit.
            //
            // In either case, we can use bitvector select to compute the index on this level.
            if index < nz {
                // `index` represents a left child on this level, represented by the `index`-th 0-bit.
                index = level.bv.select0(index);
            } else {
                // `index` represents a right child on this level, represented by the `index-nz`-th 1-bit.
                index = level.bv.select1(index - nz);
            }
        }
        Some(index)
    }

    // Return the majority element, if one exists.
    // The majority element is one whose frequency is larger than 50% of the range.
    pub fn simple_majority(&self, range: Range<V::Ones>) -> Option<V::Ones> {
        let len = range.end - range.start;
        let half_len = len >> V::Ones::one();
        let (symbol, count) = self.quantile(half_len, range);
        if count > half_len {
            Some(symbol)
        } else {
            None
        }
    }

    // todo: fn k_majority(&self, k, range) { ... }
    // Returns the 1/k-majority. Ie. for k = 4, return the elements (if any) with
    // frequency larger than 1/4th (25%) of the specified index range.
    //   - note: could we use this with ignore_bits to check if eg. half of the values are in the bottom half/quarter?
    //   - ie. doing majority queries on the high bits lets us make some statements about the density of values across
    //     *ranges*. so rather than saying "these symbols have frequency >25%" we can say "these symbol ranges have
    //     frequency >25%", for power of two frequencies (or actually arbitrary ones, based on the quantiles...right?)
    // note: even more useful would be a k_majority_candidates function that returns all the samples, which can then be filtered down.

    pub fn get(&self, index: V::Ones) -> V::Ones {
        let mut index = index;
        let mut symbol = V::zero();
        for level in self.levels(0) {
            if !level.bv.get(index) {
                // go left
                index = level.bv.rank0(index);
            } else {
                // go right
                symbol += level.bit;
                index = level.num_zeros + level.bv.rank1(index);
            }
        }
        symbol
    }

    pub fn count_all(&self, range: Range<V::Ones>) -> Traversal<CountAll<V::Ones>> {
        self.count_all_batch(&[range])
    }

    // Count the number of occurrences of each symbol in the given index range.
    // Returns a vec of (input_index, symbol, start, end) tuples.
    // Returning (start, end) rather than a count `end - start` is helpful for
    // use cases that associate per-symbol data with each entry.
    // Note that (when ignore_bits is 0) the range is on the virtual bottom layer
    // of the wavelet matrix, where symbols are sorted in ascending bit-reversed order.
    // TODO: Is there a way to do ~half the number of rank queries for contiguous
    // ranges that share a midpoint, ie. [a..b, b..c, c..d]?
    pub fn count_all_batch(&self, ranges: &[Range<V::Ones>]) -> Traversal<CountAll<V::Ones>> {
        for range in ranges {
            assert!(range.end <= self.len());
        }

        let mut traversal = Traversal::new();
        traversal.init(ranges.iter().map(|range| CountAll {
            symbol: V::zero(),
            start: range.start,
            end: range.end,
        }));

        for level in self.levels(0) {
            traversal.traverse(|xs, go| {
                // Cache the most recent rank call in case the next one is the same.
                // This means caching the `end` of the previous range, and checking
                // if it is the same as the `start` of the current range.
                let mut rank_cache = RankCache::new();
                for x in xs {
                    let CountAll { symbol, start, end } = x.value;
                    // let start = level.ranks(start);
                    // let end = level.ranks(end);
                    let (start, end) = rank_cache.get(start, end, level);

                    // if there are any left children, go left
                    if start.0 != end.0 {
                        go.left(x.value(CountAll::new(symbol, start.0, end.0)));
                    }

                    // if there are any right children, set the level bit and go right
                    if start.1 != end.1 {
                        go.right(x.value(CountAll::new(
                            symbol + level.bit,
                            level.num_zeros + start.1,
                            level.num_zeros + end.1,
                        )));
                    }
                }
                rank_cache.log_stats();
            });
        }

        traversal
    }

    pub fn get_batch(&self, indices: &[V::Ones]) -> Traversal<(V::Ones, V::Ones)> {
        // stores (wm index, symbol) entries, each corresponding to an input index.
        let mut traversal = Traversal::new();
        // todo: struct for IndexSymbol?
        let iter = indices.iter().copied().map(|index| (index, V::zero()));
        traversal.init(iter);

        for level in self.levels(0) {
            traversal.traverse(|xs, go| {
                for x in xs {
                    let (index, symbol) = x.value;
                    if !level.bv.get(index) {
                        let index = level.bv.rank0(index);
                        go.left(x.value((index, symbol)));
                    } else {
                        let index = level.num_zeros + level.bv.rank1(index);
                        let symbol = symbol + level.bit;
                        go.right(x.value((index, symbol)));
                    }
                }
            });
        }

        traversal
    }

    // Returns an iterator over levels from the high bit downwards, ignoring the
    // bottom `ignore_bits` levels.
    fn levels(&self, ignore_bits: usize) -> impl Iterator<Item = &Level<V>> {
        self.levels.iter().take(self.levels.len() - ignore_bits)
    }

    pub fn len(&self) -> V::Ones {
        self.len
    }

    pub fn max_symbol(&self) -> u32 {
        self.max_symbol
    }

    pub fn num_levels(&self) -> usize {
        self.levels.len()
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

    fn foob(&self, symbol_range: Range<V::Ones>, range: Range<V::Ones>, dims: usize) -> V::Ones {
        assert!(!symbol_range.is_empty());
        let mut traversal = Traversal::new();
        // (skip, leftmost symbol of node, start, end)
        traversal.init([(0, V::zero(), range.start, range.end)]);
        let mut count = V::zero();
        let mut nodes_visited = 0;
        let nodes_skipped = 0;

        let x_mask = V::Ones::from_u32(morton::encode2(u32::MAX, 0));
        let y_mask = V::Ones::from_u32(morton::encode2(0, u32::MAX));
        let mask_range = |range: Range<V::Ones>, level_bit: V::Ones| {
            let code = if level_bit.trailing_zeros() % 2 == 1 {
                // is an x coord level
                // want to mask the non-x coords to zero on the start, and one on the end
                // (range.start & x_mask)..(range.end | !x_mask)
                morton::decode2x(range.start.u32())..morton::decode2x(range.end.u32())
            } else {
                // is a y coord
                // (range.start & y_mask)..(range.end | !y_mask)
                morton::decode2y(range.start.u32())..morton::decode2y(range.end.u32())
            };
            dbg!(code.clone());
            V::Ones::from_u32(code.start)..V::Ones::from_u32(code.end)
        };

        for level in self.levels(0) {
            dbg!(level.bit, level.bit.trailing_zeros());
            traversal.traverse(|xs, go| {
                for x in xs {
                    let symbol_range = mask_range(symbol_range.clone(), level.bit);

                    nodes_visited += 1;
                    let (skip, left_symbol, start, end) = x.value;
                    // this node represents left_symbol..right_symbol; the width of two children
                    // let node_range = left_symbol..left_symbol + level.bit + level.bit;
                    // println!("\nopen {:?}", node_range);

                    // if fully_contains(&symbol_range, &node_range) {
                    //     skip += 1;
                    // } else {
                    //     skip = 0;
                    // }

                    // if skip == dims {
                    //     println!("skip{:?} {:?}", dims, node_range);
                    //     count += end - start;
                    //     nodes_skipped += 1;
                    //     continue;
                    // }

                    // otherwise, recurse into the left and/or right, as both may be partly covered.
                    let start = level.ranks(start);
                    let end = level.ranks(end);

                    let left_child = mask_range(left_symbol..left_symbol + level.bit, level.bit);
                    let right_child =
                        mask_range(left_child.end..left_child.end + level.bit, level.bit);

                    if overlaps(&left_child, &symbol_range) {
                        // println!("left {:?} => {:?}", node_range, start.0..end.0);
                        go.left(x.value((skip, left_symbol, start.0, end.0)));
                    }

                    if overlaps(&right_child, &symbol_range) {
                        // println!(
                        //     "right {:?} => {:?}",
                        //     node_range,
                        //     level.num_zeros + start.1..level.num_zeros + end.1
                        // );
                        go.right(x.value((
                            skip,
                            left_symbol + level.bit,
                            level.num_zeros + start.1,
                            level.num_zeros + end.1,
                        )));
                    }
                }
            });
        }

        // add up all the nodes that we did not early-out from
        for x in traversal.results() {
            let (_skip, left_symbol, start, end) = x.value;
            println!(
                "for node representing {:?}..{:?}: +{:?}",
                left_symbol,
                left_symbol + V::one(),
                end - start
            );
            count += end - start;
        }

        dbg!(nodes_visited, nodes_skipped);
        count
    }
}

fn overlaps<T: BitBlock>(a: &Range<T>, b: &Range<T>) -> bool {
    a.start < b.end && b.start < a.end
}

// Return true if a fully contains b
fn fully_contains<T: BitBlock>(a: &Range<T>, b: &Range<T>) -> bool {
    // if a starts before b, and a ends after b.
    a.start <= b.start && a.end >= b.end
}

#[cfg(test)]
mod tests {
    use crate::morton;

    use super::*;

    #[test]
    fn test_get() {
        let mut symbols = vec![];
        for i in 0..64 {
            symbols.push(i) // morton::encode2(x, y)
        }

        let max_symbol = symbols.iter().max().copied().unwrap_or(0);
        dbg!(max_symbol);
        let wm = WaveletMatrix::new(symbols.clone(), max_symbol);
        for (i, sym) in symbols.iter().copied().enumerate() {
            assert_eq!(sym, wm.get(i as u32));
        }

        let x_range = 3..5;
        let y_range = 3..5;

        let start = morton::encode2(x_range.start, y_range.start);
        // inclusive x_range and y_range endpoints, but compute the exclusive end
        let end = morton::encode2(x_range.end - 1, y_range.end - 1) + 1;
        dbg!(start, end);
        let range = start..end;
        println!("{:?}", wm.foob(range, 0..wm.len(), 2));
        panic!("count_all");

        // dbg!(sym, wm.rank(sym, 0..wm.len()));
        // dbg!(i, wm.quantile(i as u32, 0..wm.len()));

        // println!("{:?}", wm.count_all(10..15));
        // dbg!(wm.select(10, 1, 0..wm.len()));
        // let indices = &[0, 1, 2, 1, 2, 0, 13];
        // dbg!(wm.get_batch(indices));
        // ;
        // panic!("get");
    }
}
