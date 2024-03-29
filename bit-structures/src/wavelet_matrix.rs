use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::bit_block::BitBlock;
use crate::morton;
use crate::nonempty_extent::Extent;
use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec};
use num::{Bounded, One, Zero};
use std::debug_assert;
use std::{collections::VecDeque, ops::Range};

// todo
// - figure out how to provide a fast family of related functions
//   - individual/range of symbols/indices. the trouble with unifying these is that
//     a range may recurse into both children, whereas as individual element can only go into one,
//     and if implemented the usual way this means less branching for the symbol case.
// - decide if symbols should be u32, u64, or V::Ones (or another generic), and be consistent throughout.
// - consider the (internal) use of inclusive Extents to simplify range handling
// - consider using the extent crate for ranges: https://github.com/graydon/extent/blob/main/src/lib.rs
// - in the symbol count 64 test, try varying the range of the query so it is not always 1..wm.len().
// - audit whether we recurse into zero-width nodes in any cases
//   - i think we can check start.0 != end.0 and start.1 != end.1
//   - we fixed this in count_batch
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
    pub symbol: T, // leftmost symbol in the node
    pub start: T,  // index  range start
    pub end: T,    // index range end
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
    acc: u32,   // mask accumulator for the levels traversed so far
    left: u32,  // leftmost symbol in the node
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

    fn traverse(&mut self, mut f: impl FnMut(&[KeyVal<T>], &mut Goer<KeyVal<T>>)) {
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
        let mut go = Goer {
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

// Passed into the traversal callback as a way to control the recursion.
// Goes left and/or right.
struct Goer<'a, T> {
    next: &'a mut VecDeque<T>,
    num_left: usize,
}

impl<T> Goer<'_, T> {
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

#[derive(Debug)]
pub struct WaveletMatrix<V: BitVec> {
    levels: Vec<Level<V>>, // wm levels (bit planes)
    max_symbol: u32,       // maximum symbol value
    len: V::Ones,          // number of symbols
    default_masks: Vec<u32>,
}

impl<V: BitVec> bincode::Encode for WaveletMatrix<V> {
    encode_impl!(levels, max_symbol, len, default_masks);
}
impl<V: BitVec> bincode::Decode for WaveletMatrix<V> {
    decode_impl!(levels, max_symbol, len, default_masks);
}
impl<'de, V: BitVec> bincode::BorrowDecode<'de> for WaveletMatrix<V> {
    borrow_decode_impl!(levels, max_symbol, len, default_masks);
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
        let len = data.len();
        let levels = if len == 0 {
            vec![]
        } else if num_levels <= (len.ilog2() as usize) {
            build_bitvecs(data, num_levels)
        } else {
            build_bitvecs_large_alphabet(data, num_levels)
        };

        WaveletMatrix::from_bitvecs(levels, max_symbol)
    }

    // Count the number of occurences of symbols in each of the symbol ranges,
    // returning a parallel array of counts.
    // Range is an index range.
    // Masks is a slice of bitmasks, one per level, indicating the bitmask operational
    // at that level, to enable multidimensional queries.
    // To search in 1d, pass std::iter::repeat(u32::MAX).take(wm.num_levels()).collect().
    pub fn count_batch(
        &self,
        range: Range<u32>,
        symbol_ranges: &[Range<u32>],
        masks: Option<&[u32]>,
    ) -> Vec<u32> {
        let masks = masks.unwrap_or(&self.default_masks);

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
                        // node's children, since that means all children are contained within the query range.
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
                                level.nz + start.1,
                                level.nz + end.1,
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
fn union_masks<T: BitBlock>(masks: &[T]) -> T {
    masks.iter().copied().reduce(set_bits).unwrap_or(T::zero())
}

fn mask_range<T: BitBlock>(range: Range<T>, mask: T) -> Range<T> {
    (range.start & mask)..((range.end - T::one()) & mask) + T::one()
}

fn mask_extent<T: BitBlock>(extent: Extent<T>, mask: u32) -> Extent<T> {
    let mask = T::from_u32(mask);
    Extent::new(extent.start() & mask, extent.end() & mask)
}

fn mask_symbol<T: BitBlock>(symbol: T, mask: T) -> T {
    symbol & mask
}

fn masked<T: BitBlock>(symbol: T, mask: T) -> T {
    symbol & mask
}

fn set_bits<T: BitBlock>(value: T, mask: T) -> T {
    value | mask
}

fn unset_bits<T: BitBlock>(value: T, mask: T) -> T {
    value & !mask
}

// given a current acc value, compute the acc value after visiting the node represented by `node_range`
// when the target search range is `symbol_range`.
// basically, decide whether to set or un-set the bits based on whether the node range is fully contained
// within symbol_range.
fn accumulate_mask<T: BitBlock>(
    node_range: Range<T>,
    mask: T,
    symbol_range: &Range<T>,
    accumulator: T,
) -> T {
    toggle_bits(
        accumulator,
        mask,
        fully_contains(symbol_range, &mask_range(node_range, mask)),
    )
}

// accumulator represents an accumulated mask consisting of the set/unset
// bits resulting from previous calls to this function.
// the idea is that we want to toggle individual masks on and off
// such that we can detect if there is ever a time that all have
// been turned on.
// since mask bits are disjoint (eg. the x bits are distinct from
// y bits in 2d morton order), we can tell whether they're all set
// by checking equality with u32::MAX.
// This function conditionally toggles the bits in `accumulator` specified by `mask`
// on or off, based on the value of `cond`.
fn toggle_bits<T: BitBlock>(accumulator: T, mask: T, cond: bool) -> T {
    if cond {
        set_bits(accumulator, mask)
    } else {
        unset_bits(accumulator, mask)
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

    // For the last level we don't need to do anything but build the bitvector
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
    // the number of zeros at this level (ie. bv.rank0(bv.universe_size())
    nz: V::Ones,
    // unsigned int with a single bit set signifying
    // the magnitude represented at that level.
    // e.g.  levels[0].bit == 1 << levels.len() - 1
    bit: V::Ones,
}

impl<V: BitVec> bincode::Encode for Level<V> {
    encode_impl!(bv, nz, bit);
}
impl<V: BitVec> bincode::Decode for Level<V> {
    decode_impl!(bv, nz, bit);
}
impl<'de, V: BitVec> bincode::BorrowDecode<'de> for Level<V> {
    borrow_decode_impl!(bv, nz, bit);
}

// Stores (rank0, rank1) as resulting from the Level::ranks function
#[derive(Copy, Clone)]
struct Ranks<T>(T, T);

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

    pub fn child_symbol_extents(
        &self,
        left: V::Ones,
        mask: u32,
    ) -> (Extent<V::Ones>, Extent<V::Ones>) {
        let (left, mid, right) = self.splits(left);
        (
            mask_extent(Extent::new(left, mid - V::one()), mask),
            mask_extent(Extent::new(mid, right - V::one()), mask),
        )
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
        let len = levels
            .first()
            .map(|level| level.universe_size())
            .unwrap_or(V::zero());
        let levels: Vec<Level<V>> = levels
            .into_iter()
            .enumerate()
            .map(|(index, bits)| Level {
                nz: bits.rank0(bits.universe_size()),
                bit: V::one() << (max_level - index),
                bv: bits,
            })
            .collect();
        let num_levels = levels.len();
        Self {
            levels,
            max_symbol,
            len,
            default_masks: std::iter::repeat(u32::max_value())
                .take(num_levels)
                .collect(),
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
        assert!(
            symbol <= V::Ones::from_u32(self.max_symbol),
            "symbol must not exceed max_symbol"
        );
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
                range = level.nz + start.1..level.nz + end.1;
            }
        }
        (preceding_count, range)
    }

    // Returns the index of the first symbol less than p in the index range `range`.
    // ("First" here is based on sequence order; we will return the leftmost such index).
    // Implements the following logic:
    // selectFirstLeq = (arr, p, lo, hi) => {
    //   let i = arr.slice(lo, hi).findIndex((x) => x <= p);
    //   return i === -1 ? undefined : lo + i;
    // }
    // note: since the left extent of the target is always zero, we could optimize the containment checks.
    //
    pub fn select_first_leq(&self, p: V::Ones, range: Range<V::Ones>) -> Option<V::Ones> {
        let mut range = range; // index range
        let mut symbol = V::zero(); // leftmost symbol in the currently-considered wavelet tree node
        let mut best = V::Ones::max_value();
        let mut found = false;
        let target = Extent::new(V::zero(), p);

        // todo: select_[first/last[_[leq/geq].
        // The idea is to return the minimum select position across all the nodes that could
        // potentially contain the first symbol <= p.
        //
        // We find the first left node that is fully contained in the [0, p] symbol range,
        // and then we recurse into the right child if it is partly contained, and repeat.

        for (i, level) in self.levels.iter().enumerate() {
            if range.is_empty() {
                break;
            }

            let ignore_bits = self.num_levels() - i; // ignore all levels below this one when selecting
            let (left, mid, right) = level.splits(symbol); // value split points of left/right children

            // if this wavelet tree node is fully contained in the target range, update best and return.
            if target.fully_contains_range(left..right) {
                let candidate = self.select_upwards(range.start, ignore_bits).unwrap();
                return Some(best.min(candidate));
            }

            let start = level.ranks(range.start);
            let end = level.ranks(range.end);

            // otherwise, we know that there are two possibilities:
            // 1. the left node is partly contained and the right node does not overlap the target
            // 2. the left node is fully contained and the right node may overlap the target
            if !target.fully_contains_range(left..mid) {
                // we're in case 1, so refine our search range by going left
                range = start.0..end.0;
            } else {
                // we're in case 2, so update the best so far from the left child node if it is nonempty, then go right.
                if start.0 != end.0 {
                    // since this select is happening on the child level, un-ignore that level.
                    let candidate = self.select_upwards(start.0, ignore_bits - 1).unwrap();
                    best = best.min(candidate);
                    found = true;
                }
                // go right
                symbol += level.bit;
                range = level.nz + start.1..level.nz + end.1;
            }
        }

        if found {
            Some(best)
        } else {
            None
        }
    }

    pub fn locate_batch(
        &self,
        ranges: &[Range<V::Ones>],
        symbols: &[V::Ones],
    ) -> Traversal<(V::Ones, V::Ones, V::Ones, V::Ones)> {
        let iter = symbols
            .iter()
            // todo: make a struct for this function: (symbol, preceding_count, start, end)
            .flat_map(|symbol| {
                assert!(
                    *symbol <= V::Ones::from_u32(self.max_symbol),
                    "symbol must not exceed max_symbol"
                );
                ranges
                    .iter()
                    .map(|range| (*symbol, V::zero(), range.start, range.end))
            });
        let mut traversal = Traversal::new(iter);
        for level in self.levels.iter() {
            traversal.traverse(|xs, go| {
                for x in xs {
                    let (symbol, preceding_count) = (x.val.0, x.val.1);
                    let (start, end) = (level.ranks(x.val.2), level.ranks(x.val.3));
                    if (symbol & level.bit).is_zero() {
                        go.left(x.val((symbol, preceding_count, start.0, end.0)));
                    } else {
                        go.right(x.val((
                            symbol,
                            preceding_count + end.0 - start.0,
                            level.nz + start.1,
                            level.nz + end.1,
                        )));
                    }
                }
            });
        }
        traversal
    }

    // number of symbols less than this one, restricted to the query range
    pub fn preceding_count(&self, symbol: V::Ones, range: Range<V::Ones>) -> V::Ones {
        self.locate(symbol, range, 0).0
    }

    // number of times the symbol appears in the query range
    pub fn count(&self, range: Range<V::Ones>, symbol: V::Ones) -> V::Ones {
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
                range = level.nz + start.1..level.nz + end.1;
            }
        }
        let count = range.end - range.start;
        (symbol, count)
    }

    pub fn select(
        &self,
        symbol: V::Ones,
        k: V::Ones,
        range: Range<V::Ones>,
        ignore_bits: usize,
    ) -> Option<V::Ones> {
        if symbol > V::Ones::from_u32(self.max_symbol) {
            return None;
        }

        // track the symbol down to a range on the bottom-most level we're interested in
        let range = self.locate(symbol, range, ignore_bits).1;
        let count = range.end - range.start;

        // If there are fewer than `k+1` copies of `symbol` in the range, return early.
        // `k` is zero-indexed, so our check includes equality.
        if count <= k {
            return None;
        }

        // track the k-th occurrence of the symbol up from the bottom-most virtual level
        // or higher, if ignore_bits is non-zero.
        let index = range.start + k;
        self.select_upwards(index, ignore_bits)
    }

    // same as select, but select the k-th instance from the back
    pub fn select_last(
        &self,
        symbol: V::Ones,
        k: V::Ones,
        range: Range<V::Ones>,
        ignore_bits: usize,
    ) -> Option<V::Ones> {
        if symbol > V::Ones::from_u32(self.max_symbol) {
            return None;
        }
        let range = self.locate(symbol, range, ignore_bits).1;
        let count = range.end - range.start;
        if count <= k {
            return None;
        }
        let index = range.end - k - V::one(); // - 1 because end is exclusive
        self.select_upwards(index, ignore_bits)
    }

    // This function abstracts the common second half of the select algorithm, once you've
    // identified an index on the "bottom" level and want to bubble it back up to translate
    // the "sorted" index from the bottom level to the index of that element in sequence order.
    // We make this a pub fn since it could allow eg. external users of `locate` to efficiently
    // select their chosen element. For example, perhaps we should remove `select_last`...
    pub fn select_upwards(&self, index: V::Ones, ignore_bits: usize) -> Option<V::Ones> {
        let mut index = index;
        for level in self.levels(ignore_bits).rev() {
            // `index` represents an index on the level below this one, which may be
            // the bottom-most 'virtual' layer that contains all symbols in sorted order.
            //
            // We want to determine the position of the element represented by `index` on
            // this level, which we can do by "mapping" the index up to its parent node.
            //
            // `level.nz` tells us how many bits on the level below come from left children of
            // the wavelet tree (represented by this wavelet matrix). If the index < nz, that
            // means that the index on the level below came from a left child on this level,
            // which means that it must be represented by a 0-bit on this level; specifically,
            // the `index`-th 0-bit, since the WT always represents a stable sort of its elements.
            //
            // On the other hand, if `index` came from a right child on this level, then it
            // is represented by a 1-bit on this level; specifically, the `index - nz`-th 1-bit.
            //
            // In either case, we can use bitvector select to compute the index on this level.
            if index < level.nz {
                // `index` represents a left child on this level, represented by the `index`-th 0-bit.
                index = level.bv.select0(index);
            } else {
                // `index` represents a right child on this level, represented by the `index-nz`-th 1-bit.
                index = level.bv.select1(index - level.nz);
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
                index = level.nz + level.bv.rank1(index);
            }
        }
        symbol
    }

    // Count the number of occurrences of each symbol in the given index range.
    // Returns a vec of (input_index, symbol, start, end) tuples.
    // Returning (start, end) rather than a count `end - start` is helpful for
    // use cases that associate per-symbol data with each entry.
    // Note that (when ignore_bits is 0) the range is on the virtual bottom layer
    // of the wavelet matrix, where symbols are sorted in ascending bit-reversed order.
    // TODO: Is there a way to do ~half the number of rank queries for contiguous
    // ranges that share a midpoint, ie. [a..b, b..c, c..d]?
    pub fn counts(
        &self,
        ranges: &[Range<V::Ones>],
        // note: this is inclusive because the requirement comes up in practice, eg.
        // a 32-bit integer can represent 3 10-bit integers, and you may want to query
        // for the 10-bit subcomponents separately, eg. 0..1025. But to represent 1025
        // in each dimension would require 33 bits. instead, inclusive extents allow
        // representing 0..1025 (11 bits) as 0..=1024 (10 bits).
        symbol_extent: Extent<V::Ones>,
        masks: Option<&[u32]>,
    ) -> Traversal<CountAll<V::Ones>> {
        let masks = masks.unwrap_or(&self.default_masks);

        for range in ranges {
            assert!(range.end <= self.len());
        }

        let mut traversal = Traversal::new(ranges.iter().map(|range| CountAll {
            symbol: V::zero(), // the leftmost symbol in the current node
            start: range.start,
            end: range.end,
        }));

        for (level, mask) in self.levels.iter().zip(masks.iter().copied()) {
            let symbol_extent = mask_extent(symbol_extent, mask);
            traversal.traverse(|xs, go| {
                // Cache the most recent rank call in case the next one is the same.
                // This means caching the `end` of the previous range, and checking
                // if it is the same as the `start` of the current range.
                let mut rank_cache = RangedRankCache::new();
                for x in xs {
                    let symbol = x.val.symbol;
                    let (left, right) = level.child_symbol_extents(symbol, mask);
                    let (start, end) = rank_cache.get(x.val.start, x.val.end, level);

                    // if there are any left children, go left
                    if start.0 != end.0 && symbol_extent.overlaps(left) {
                        go.left(x.val(CountAll::new(symbol, start.0, end.0)));
                    }

                    // if there are any right children, set the level bit and go right
                    if start.1 != end.1 && symbol_extent.overlaps(right) {
                        go.right(x.val(CountAll::new(
                            symbol + level.bit,
                            level.nz + start.1,
                            level.nz + end.1,
                        )));
                    }
                }
                rank_cache.log_stats();
            });
        }
        traversal
    }

    // todo: get with ranges rather than indices (in fact, indices could be represented as a special case,
    // though maybe it is faster this way and we should implement that separately)
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
                        let index = level.nz + level.bv.rank1(index);
                        let symbol = symbol + level.bit;
                        go.right(x.val((index, symbol)));
                    }
                }
            });
        }
        traversal
    }

    // todo: approximate_sum(ignore_levels) -> (Ones, Ones)
    // descend the tree, and at each level, the lower bound on the sum is the number
    // of elements that went right times the level bit for that level.
    // the upper bound is the lower bound, plus the total number of elements times
    // 0b00001111 with a number of 1s equal to the number of remaining levels (ie.
    // the upper bound is achieved if every element goes right on every remaining level).
    // there is a perf optimization on the last level to not actually project down to the
    // next level, since we don't need the elements on the virtual layer; just their sum.

    // Return the k-th element from the sorted concatenation of all the ranges.
    // note: i think this even works for overlapping or out-of-order ranges
    // todo: after each iteration,
    // 1. merge contiguous ranges
    // 2. remove empty ranges
    // since we don't care about how many we have and it is better to have less of them.
    // i think we can use retain_mut.
    // note: there is an optimization on the last level where we actually don't need to
    // project everyone down to the next level; we only need to check ranges until we
    // figure out the final bit of the returned symbol (and its index, which will be
    // in the final range that tells us that. Ie. does this just mean avoid the second
    // for loop after we determine go_left?
    pub fn multi_range_quantile(
        &self,
        k: V::Ones,
        ranges: &[Range<V::Ones>],
    ) -> (V::Ones, V::Ones) {
        assert!(k.u32() < ranges.iter().map(|x| (x.end - x.start).u32()).sum());
        let mut k = k;
        let mut symbol = V::zero();
        let mut ranges = Vec::from(ranges);
        // scratch space for start/end ranks
        let mut starts = vec![];
        let mut ends = vec![];

        for level in self.levels(0) {
            let mut total_left_count = V::zero();
            starts.clear();
            ends.clear();
            for range in &ranges {
                let start = level.ranks(range.start);
                let end = level.ranks(range.end);
                let left_count = end.0 - start.0;
                total_left_count += left_count;
                starts.push(start);
                ends.push(end);
            }

            // now determine whether we actually want to go left or right for the 'true' pass
            let go_left = k < total_left_count;

            // avoid iterating over the start/end and use the answers we just cached
            for ((range, start), end) in ranges.iter_mut().zip(starts.iter()).zip(ends.iter()) {
                if go_left {
                    *range = start.0..end.0;
                } else {
                    *range = level.nz + start.1..level.nz + end.1;
                }
            }

            if !go_left {
                symbol += level.bit;
                k -= total_left_count;
            }
        }

        let mut count = V::zero();
        // note: can compute the index on the virtual bottom level
        for ranges in ranges {
            count += ranges.end - ranges.start;
        }

        (symbol, count)
    }

    // Returns an iterator over levels from the high bit downwards, ignoring the
    // bottom `ignore_bits` levels.
    fn levels(&self, ignore_bits: usize) -> std::slice::Iter<Level<V>> {
        self.levels[..self.levels.len() - ignore_bits].iter()
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

    pub fn morton_masks_for_dims(&self, dims: u32) -> Vec<u32> {
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
        masks
            .iter()
            .copied()
            .cycle()
            .take(self.num_levels())
            .collect()
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
        let masks = wm.morton_masks_for_dims(dims);

        // 8^4 = 4096
        // 8^6 = 262144
        // x2 and y2 are inclusive bounds
        // todo: figure out how to parameterize this by dimension
        let z_side_length = if false { side_length } else { 0 }; // make this effectively 2d
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

                                let count = wm
                                    .count_batch(0..wm.len(), &[start..end], Some(&masks))
                                    .first()
                                    .copied()
                                    .unwrap();
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

        let masks = wm.morton_masks_for_dims(dims);

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
            let wm_count = wm
                .count_batch(0..wm.len(), &[query.clone()], Some(&masks))
                .first()
                .copied()
                .unwrap();
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
        let res = wm.count_batch(0..wm.len(), &queries, Some(&masks));
        let end_time = SystemTime::now();
        println!(
            "time for batch query on {:?} inputs: {:?}",
            q,
            end_time.duration_since(start_time)
        );
        assert_eq!(wm_counts, res);
        // panic!("wheee");
    }

    #[test]
    fn test_get() {
        let mut symbols = vec![];
        let pow = 6;
        let n = 2u32.pow(pow);
        let _side = 2u32.pow(pow / 2);
        for i in 0..n {
            // if i % 2 == 1 {
            symbols.push(i)
            // }
        }
        // let max_symbol = symbols.iter().max().copied().unwrap_or(0);
        let wm = WaveletMatrix::new(symbols.clone(), n - 1);
        for (i, sym) in symbols.iter().copied().enumerate() {
            assert_eq!(sym, wm.get(i as u32));
        }

        // test morton range querying
        // let x_range = 3..5;
        // let y_range = 3..5;
        // let start = morton::encode2(x_range.start, y_range.start);
        // let end = morton::encode2(x_range.end - 1, y_range.end - 1) + 1;
        // let masks = morton_masks_for_dims(2, wm.num_levels());
        // dbg!(start..end);
        // let mut xs = wm.counts(start..end, &[0..wm.len()], &masks);
        // for x in xs.results() {
        //     println!(
        //         "({:?}, {:?}): {:?}",
        //         morton::decode2x(x.val.symbol),
        //         morton::decode2y(x.val.symbol),
        //         x.val.end - x.val.start
        //     );
        // }
        // panic!("aah");

        // caution: easy to go out of bounds here in either x or y alone
        // let x_range = 1..side - 1;
        // let y_range = 1..side - 1;
        // let start = morton::encode2(x_range.start, y_range.start);
        // // inclusive x_range and y_range endpoints, but compute the exclusive end
        // let end = morton::encode2(x_range.end - 1, y_range.end - 1) + 1;
        // assert!(end <= max_symbol + 1);
        // dbg!(start, end);
        // let range = start..end;
        // let dims = 2;
        // let masks = morton_masks_for_dims(dims, wm.num_levels());
        // println!("{:?}", wm.count_symbol_range(range, 0..wm.len(), &masks));
        // panic!("count_all");
        // dbg!(sym, wm.rank(sym, 0..wm.len()));
        // dbg!(i, wm.quantile(i as u32, 0..wm.len()));
        // println!("{:?}", wm.count_all(10..15));
        // dbg!(wm.select(10, 1, 0..wm.len()));
        // let indices = &[0, 1, 2, 1, 2, 0, 13];
        // dbg!(wm.get_batch(indices));
        // ;
        // panic!("get");
        // */
    }
}
