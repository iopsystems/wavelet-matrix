use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::morton;
use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec};
use num::{One, Zero};
use std::debug_assert;
use std::{collections::VecDeque, ops::Range};

// todo
// - consider using the extent crate for ranges: https://github.com/graydon/extent/blob/main/src/lib.rs
// - in the symbol count 64 test, try varying the range of the query so it is not always 1..wm.len().
// - audit whether we recurse into zero-width nodes in any cases
//   - i think we can check start.0 != end.0 and start.1 != end.1
//   - we fixed this in count_symbol_range_batch
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

// type representing the state of an individual traversal path down the wavelet tree
// during a count_symbol_range operation
#[derive(Copy, Clone, Debug)]
struct CountSymbolRange {
    acc: u32,   // mask accumulator
    left: u32,  // left symbol
    start: u32, // index  range start
    end: u32,   // index range end
}

impl CountSymbolRange {
    fn new(acc: u32, left: u32, start: u32, end: u32) -> Self {
        CountSymbolRange {
            acc,
            left,
            start,
            end,
        }
    }
}

// Helper for traversing the wavelet matrix level by level,
// reusing space when possible and keeping the elements in
// sorted order with respect to the ordering of wavelet tree
// nodes in the wavelet matrix (all left nodes precede all
// right nodes).
#[derive(Debug)]
pub struct Traversal<T> {
    cur: VecDeque<KeyVal<T>>,
    next: VecDeque<KeyVal<T>>,
    num_left: usize,
}

// Traverse a wavelet matrix levelwise, at each level maintaining tree nodes
// in order they appear in the wavelet matrix (left children preceding right).
impl<T> Traversal<T> {
    fn new(values: impl IntoIterator<Item = T>) -> Self {
        let mut traversal = Self {
            cur: VecDeque::new(),
            next: VecDeque::new(),
            num_left: 0,
        };
        traversal.init(values);
        traversal
    }

    fn init(&mut self, values: impl IntoIterator<Item = T>) {
        let iter = values.into_iter().enumerate().map(KeyVal::from_tuple);
        self.cur.clear();
        self.next.clear();
        self.next.extend(iter);
        self.num_left = 0;
    }

    fn traverse(&mut self, mut f: impl FnMut(&[KeyVal<T>], &mut Go<KeyVal<T>>)) {
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

    pub fn results(&mut self) -> &mut [KeyVal<T>] {
        let slice = self.next.make_contiguous();
        // note: reverse only required if we want to return results in wm order,
        // which might be nice if we are eg. looking up associated data.
        slice[..self.num_left].reverse();

        self.num_left = 0; // update this so that calling results multiple times does not re-reverse the left
        slice
    }

    // note: we check whether *next* is empty since that is what will be traversed next, since
    // `next` is swapped into `cur` in `traversal.traverse()`.
    pub fn is_empty(&self) -> bool {
        self.next.is_empty()
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
pub struct KeyVal<T> {
    pub key: usize,
    pub val: T,
}

// Associate a usize key to an arbitrary value; used for propagating the metadata
// of which original query element a partial query result is associated with as we
// traverse the wavelet tree
impl<T> KeyVal<T> {
    fn new(key: usize, value: T) -> KeyVal<T> {
        KeyVal { key, val: value }
    }
    // construct a BatchValue from an (key, value) tuple
    fn from_tuple((key, value): (usize, T)) -> KeyVal<T> {
        KeyVal { key, val: value }
    }
    fn map<U>(self, f: impl FnOnce(T) -> U) -> KeyVal<U> {
        KeyVal {
            key: self.key,
            val: f(self.val),
        }
    }
    // return a new KeyVal with the previous key and new value
    fn val(self, value: T) -> KeyVal<T> {
        KeyVal { val: value, ..self }
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

struct RangedRankCache<V: BitVec> {
    end_index: Option<V::Ones>, // previous end index
    end_ranks: Ranks<V::Ones>,  // previous end ranks
    // note: we track these just out of interest;
    // we could enable only when profiling.
    num_hits: usize,   // number of cache hits
    num_misses: usize, // number of cache misses
}

impl<V: BitVec> RangedRankCache<V> {
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
        println!(
            "cached {:.1}%: {:?} / {:?}",
            // note: can be nan
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

    pub fn count_symbol_range(
        &self,
        symbol_range: Range<u32>,
        range: Range<u32>,
        masks: &[u32],
    ) -> u32 {
        self.count_symbol_range_batch(&[symbol_range], range, masks)
            .first()
            .copied()
            .unwrap()
    }

    // Count the number of occurences of symbols in each of the symbol ranges,
    // returning a parallel array of counts.
    // Range is an index range.
    // Masks is a slice of bitmasks, one per level, indicating the bitmask operational
    // at that level, to enable multidimensional queries.
    // To search in 1d, pass std::iter::repeat(u32::MAX).take(wm.num_levels()).collect().
    pub fn count_symbol_range_batch(
        &self,
        symbol_ranges: &[Range<u32>],
        range: Range<u32>,
        masks: &[u32],
    ) -> Vec<u32> {
        // Union all bitmasks so we can tell when we the symbol range is fully contained within
        // the query range at a particular wavelet tree node, in order to avoid needless recursion.
        let all_masks = union_masks(masks);

        // The return vector of counts
        let mut counts = vec![0; symbol_ranges.len()];

        // Initialize a wavelet matrix traversal with one entry per symbol range we're searching.
        let init = CountSymbolRange::new(0, 0, range.start, range.end);
        let mut traversal = Traversal::new(std::iter::repeat(init).take(symbol_ranges.len()));

        for (level, mask) in self.levels.iter().zip(masks.iter().copied()) {
            traversal.traverse(|xs, go| {
                // Cache rank queries when the start of the current range is the same as the end of the previous range
                let mut rank_cache = RangedRankCache::new();
                for x in xs {
                    // The symbol range corresponding to the current query, masked to the relevant dimensions at this level
                    let symbol_range = mask_range(symbol_ranges[x.key].clone(), mask);

                    // Left, middle, and right symbol indices for the children of this node.
                    let (left, mid, right) = level.splits(x.val.left);

                    // Tuples representing the rank0/1 of start and rank0/1 of end.
                    let (start, end) = rank_cache.get(x.val.start, x.val.end, level);

                    // Check the left child if there are any elements there
                    if start.0 != end.0 {
                        // Determine whether we can short-circuit the recursion because the symbols
                        // represented by the left child are fully contained in symbol_range in all
                        // dimensions (ie. for all distinct masks). For example, if the masks represent
                        // a two-dimensional query, we need to check that (effectively) the quadtree
                        // node, represented by two contiguous dimensions, is contained. It's a bit subtle
                        // since we can early-out not only if a contiguous 'xy' range is detected, but also
                        // a contiguous 'yx' range – so long as the symbol range is contained in the most
                        // recent branching in all dimensions, we can stop the recursion early and count the
                        // elements in the node, since all children are contained within the query range.
                        //
                        // Each "dimension" is indicated by a different mask. So far, use cases have meant that
                        // each bit of the symbol is assigned to at most one mask.
                        //
                        // To accumulate a new mask to the accumulator, we will either set or un-set all the bits
                        // corresponding to this mask. We will set them if the symbol range represented by this node
                        // is fully contained in the query range, and un-set them otherwise.
                        //
                        // If the node is contained in all dimensions, then the accumulator will be equal to all_masks,
                        // and we can stop the recursion early.
                        let acc = accumulate_mask(left..mid, mask, &symbol_range, x.val.acc);
                        if acc == all_masks {
                            counts[x.key] += end.0 - start.0;
                        } else if overlaps(&symbol_range, &mask_range(left..mid, mask)) {
                            // We need to recurse into the left child. Do so with the new acc value.
                            go.left(x.val(CountSymbolRange::new(acc, left, start.0, end.0)));
                        }
                    }

                    // right child
                    if start.1 != end.1 {
                        // See the comments for the left node; the logical structure here is identical.
                        let acc = accumulate_mask(mid..right, mask, &symbol_range, x.val.acc);
                        if acc == all_masks {
                            counts[x.key] += end.1 - start.1;
                        } else if overlaps(&symbol_range, &mask_range(mid..right, mask)) {
                            go.right(x.val(CountSymbolRange::new(
                                acc,
                                mid,
                                level.num_zeros + start.1,
                                level.num_zeros + end.1,
                            )));
                        }
                    }
                }
            });
        }

        // For complete queries, the last iteration of the loop above finds itself recursing to the
        // virtual bottom level of the wavelet tree, each node representing an individual symbol,
        // so there should be no uncounted nodes left over. This is a bit subtle when masks are
        // involved but I think the same logic applies.
        if masks.len() == self.num_levels() {
            debug_assert!(traversal.is_empty());
        } else {
            // Count any nodes left over in the traversal if it didn't traverse all levels,
            // ie. some bottom levels were ignored.
            //
            // I'm not sure if this is actually the behavior we want – it means that symbols
            // outside the range will be counted...
            //
            // Yeah, let's comment this out for now and leave this note here to decide later.
            //
            // for x in traversal.results() {
            //     counts[x.key] += x.val.end - x.val.start;
            // }
        }

        counts
    }
}

// Return the union of set bits across all masks in `masks`
fn union_masks(masks: &[u32]) -> u32 {
    masks.iter().copied().reduce(set_bits).unwrap_or(0)
}

fn mask_range(range: Range<u32>, mask: u32) -> Range<u32> {
    (range.start & mask)..((range.end - 1) & mask) + 1
}

fn set_bits(value: u32, mask: u32) -> u32 {
    value | mask
}

fn unset_bits(value: u32, mask: u32) -> u32 {
    value & !mask
}

// given a current acc value, compute the acc value after visiting the node represented by `node_range`
// when the target search range is `symbol_range`.
// basically, decide whether to set or un-set the bits based on whether the node range is fully contained
// within symbol_range.
fn accumulate_mask(node_range: Range<u32>, mask: u32, symbol_range: &Range<u32>, acc: u32) -> u32 {
    toggle_bits(
        acc,
        mask,
        fully_contains(symbol_range, &mask_range(node_range, mask)),
    )
}

// acc represents an accumulated mask consisting of the set/unset
// bits resulting from previous calls to this function.
// the idea is that we want to toggle individual masks on and off
// such that we can detect if there is ever a time that all have
// been turned on.
// since mask bits are disjoint (eg. the x bits are distinct from
// y bits in 2d morton order), we can tell whether they're all set
// by checking equality with u32::MAX.
// This function conditionally toggles the bits in `acc` specified by `mask`
// on or off, based on the value of `cond`.
fn toggle_bits(acc: u32, mask: u32, cond: bool) -> u32 {
    if cond {
        set_bits(acc, mask)
    } else {
        unset_bits(acc, mask)
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

    // Given the start index of a left node on this level, return the split points
    // that cover the range:
    // - left is the start of the left node
    // - mid is the start of the right node
    // - right is one past the end of the right node
    pub fn splits(&self, left: V::Ones) -> (V::Ones, V::Ones, V::Ones) {
        (left, left + self.bit, left + self.bit + self.bit)
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

        let mut traversal = Traversal::new(ranges.iter().map(|range| CountAll {
            symbol: V::zero(),
            start: range.start,
            end: range.end,
        }));

        for level in self.levels(0) {
            traversal.traverse(|xs, go| {
                // Cache the most recent rank call in case the next one is the same.
                // This means caching the `end` of the previous range, and checking
                // if it is the same as the `start` of the current range.
                let mut rank_cache = RangedRankCache::new();
                for x in xs {
                    let CountAll { symbol, start, end } = x.val;
                    // let start = level.ranks(start);
                    // let end = level.ranks(end);
                    let (start, end) = rank_cache.get(start, end, level);

                    // if there are any left children, go left
                    if start.0 != end.0 {
                        go.left(x.val(CountAll::new(symbol, start.0, end.0)));
                    }

                    // if there are any right children, set the level bit and go right
                    if start.1 != end.1 {
                        go.right(x.val(CountAll::new(
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
        // todo: struct for IndexSymbol?
        let mut traversal = Traversal::new(indices.iter().copied().map(|index| (index, V::zero())));

        for level in self.levels(0) {
            traversal.traverse(|xs, go| {
                for x in xs {
                    let (index, symbol) = x.val;
                    if !level.bv.get(index) {
                        let index = level.bv.rank0(index);
                        go.left(x.val((index, symbol)));
                    } else {
                        let index = level.num_zeros + level.bv.rank1(index);
                        let symbol = symbol + level.bit;
                        go.right(x.val((index, symbol)));
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
}

fn overlaps<T: BitBlock>(a: &Range<T>, b: &Range<T>) -> bool {
    a.start < b.end && b.start < a.end
}

// Return true if a fully contains b
fn fully_contains<T: BitBlock>(a: &Range<T>, b: &Range<T>) -> bool {
    // if a starts before b, and a ends after b.
    a.start <= b.start && a.end >= b.end
}

pub fn morton_masks_for_dims(dims: u32, num_levels: usize) -> Vec<u32> {
    const S1: [u32; 1] = [u32::MAX];
    const S2: [u32; 2] = [morton::encode2(0, u32::MAX), morton::encode2(u32::MAX, 0)];
    const S3: [u32; 3] = [
        morton::encode3(0, 0, u32::MAX),
        morton::encode3(0, u32::MAX, 0),
        morton::encode3(u32::MAX, 0, 0),
    ];
    let masks = match dims {
        1 => &S1[..],
        2 => &S2[..],
        3 => &S3[..],
        _ => panic!("only 1-3 dimensions currently supported"),
    };
    masks.iter().copied().cycle().take(num_levels).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morton;
    use rand::{seq::SliceRandom, Rng};
    use std::time::{Duration, SystemTime};

    #[test]
    fn test_count_symbol_range_64() {
        let mut rng = rand::thread_rng();
        let mut symbols = vec![];

        let side_length: u32 = 8;
        let dims = 3;
        let n = side_length.pow(dims);
        for i in 0..n {
            let repetitions = i + 1;
            for _ in 0..repetitions {
                symbols.push(i)
            }
        }
        symbols.shuffle(&mut rng);

        let wm = WaveletMatrix::new(symbols.clone(), n - 1);
        let masks = morton_masks_for_dims(dims, wm.num_levels());

        // 8^4 = 4096
        // 8^6 = 262144
        // x2 and y2 are inclusive bounds
        // todo: figure out how to parameterize this by dimension
        let z_side_length = if false { side_length } else { 1 }; // make this effectively 2d
        for x1 in 0..side_length {
            for x2 in x1..side_length {
                for y1 in 0..side_length {
                    for y2 in y1..side_length {
                        for z1 in 0..z_side_length {
                            for z2 in z1..z_side_length {
                                // let start = morton::encode2(x1, y1);
                                // let end = morton::encode2(x2, y2) + 1;
                                let start = morton::encode3(x1, y1, z1);
                                let end = morton::encode3(x2, y2, z2) + 1;

                                let range = start..end;
                                let count = wm.count_symbol_range(range, 0..wm.len(), &masks);
                                let naive_count = symbols
                                    .iter()
                                    .copied()
                                    .filter(|&symbol| {
                                        let x = morton::decode3x(symbol);
                                        let y = morton::decode3y(symbol);
                                        let z = morton::decode3z(symbol);
                                        x1 <= x
                                            && x <= x2
                                            && y1 <= y
                                            && y <= y2
                                            && z1 <= z
                                            && z <= z2
                                    })
                                    .count()
                                    as u32;
                                assert_eq!(count, naive_count);
                            }
                        }
                    }
                }
            }
        }
    }

    fn ascend(x: Range<u32>) -> Range<u32> {
        // clear low bits to see if it affects query speed
        // let x = ((x.start >> 5) << 5)..((x.end >> 5) << 5);
        if x.start < x.end {
            x
        } else {
            x.end..x.start + 1
        }
    }

    #[test]
    fn test_count() {
        let mut rng = rand::thread_rng();
        let dims = 3;
        let base: u32 = 1 << 6; // bits per dimension
        let (n, k) = (100_000, base.pow(dims)); // n numbers in [0, k)

        let mut symbols = vec![];
        for _ in 0..n {
            symbols.push(rng.gen_range(0..k)); // / 1024) * 1024
        }
        let max_symbol = k - 1;
        let wm = WaveletMatrix::new(symbols.clone(), max_symbol);
        dbg!(wm.num_levels());

        let masks = morton_masks_for_dims(dims, wm.num_levels());

        let mut wm_duration = Duration::ZERO;

        let q = 1000;

        let mut wm_counts = vec![];
        let mut test_counts: Vec<u32> = vec![];
        let mut batch_counts = vec![];
        let mut queries = vec![];

        for _ in 0..q {
            // randomly sized boxes
            let x_range = ascend(rng.gen_range(0..base - 1)..rng.gen_range(0..base - 1));
            let y_range = ascend(rng.gen_range(0..base - 1)..rng.gen_range(0..base - 1));
            let z_range = ascend(rng.gen_range(0..base - 1)..rng.gen_range(0..base - 1));

            let start = match dims {
                1 => x_range.start,
                2 => morton::encode2(x_range.start, y_range.start),
                3 => morton::encode3(x_range.start, y_range.start, z_range.start),
                _ => panic!("as_dims must be 1, 2, or 3."),
            };

            // inclusive range endpoints within each dimensions, then compute the exclusive end
            let end = match dims {
                1 => x_range.end,
                2 => morton::encode2(x_range.end - 1, y_range.end - 1) + 1,
                3 => morton::encode3(x_range.end - 1, y_range.end - 1, z_range.end - 1) + 1,
                _ => panic!("as_dims must be 1, 2, or 3."),
            };

            assert!(end <= max_symbol + 1);

            let range = start..end;
            queries.push(range.clone());
            let test_count = symbols
                .iter()
                .copied()
                .filter(|&code| {
                    match dims {
                        1 => {
                            // 1d
                            start <= code && code < end
                        }
                        2 => {
                            // 2d
                            let x = morton::decode2x(code);
                            let y = morton::decode2y(code);
                            x_range.start <= x
                                && x < x_range.end
                                && y_range.start <= y
                                && y < y_range.end
                        }
                        3 => {
                            let x = morton::decode3x(code);
                            let y = morton::decode3y(code);
                            let z = morton::decode3z(code);
                            x_range.start <= x
                                && x < x_range.end
                                && y_range.start <= y
                                && y < y_range.end
                                && z_range.start <= z
                                && z < z_range.end
                        }
                        _ => panic!("as_dims must be 1, 2, or 3."),
                    }
                })
                .count() as u32;
            test_counts.push(test_count);
        }

        for query in &queries {
            // println!("\nnew query\n");
            let query_start_time = SystemTime::now();
            let wm_count = wm.count_symbol_range(query.clone(), 0..wm.len(), &masks);
            let query_end_time = SystemTime::now();
            let dur = query_end_time.duration_since(query_start_time).unwrap();
            wm_duration += dur;
            wm_counts.push(wm_count);
            batch_counts.push(wm_count); // todo: remove and do a single batch query to test
        }

        assert_eq!(test_counts, wm_counts);

        println!("total for {:?} queries: {:?}", q, wm_duration);

        // todo: implement batch queries for batches of symbol ranges in the same wm range
        let start_time = SystemTime::now();
        let res = wm.count_symbol_range_batch(&queries, 0..wm.len(), &masks);
        let end_time = SystemTime::now();
        println!(
            "time for batch query on {:?} inputs: {:?}",
            q,
            end_time.duration_since(start_time)
        );
        assert_eq!(wm_counts, res);
        // panic!("wheee");
    }

    // #[test]
    fn test_get() {
        let mut symbols = vec![];
        let pow = 6;
        let n = 2u32.pow(pow);
        let side = 2u32.pow(pow / 2);
        for i in 0..n {
            // if i % 2 == 1 {
            symbols.push(i)
            // }
        }
        let max_symbol = symbols.iter().max().copied().unwrap_or(0);
        let wm = WaveletMatrix::new(symbols.clone(), n - 1);
        for (i, sym) in symbols.iter().copied().enumerate() {
            assert_eq!(sym, wm.get(i as u32));
        }
        // caution: easy to go out of bounds here in either x or y alone
        let x_range = 1..side - 1;
        let y_range = 1..side - 1;
        let start = morton::encode2(x_range.start, y_range.start);
        // inclusive x_range and y_range endpoints, but compute the exclusive end
        let end = morton::encode2(x_range.end - 1, y_range.end - 1) + 1;
        assert!(end <= max_symbol + 1);
        dbg!(start, end);
        let range = start..end;
        let dims = 2;
        let masks = morton_masks_for_dims(dims, wm.num_levels());
        println!("{:?}", wm.count_symbol_range(range, 0..wm.len(), &masks));
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
