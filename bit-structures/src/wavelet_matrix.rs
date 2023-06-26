use crate::bit_block::BitBlock;
use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec};
use std::ops::{Range, RangeInclusive};

#[derive(Debug)]
pub struct WaveletMatrix<Ones: BitBlock, BV: BitVec<Ones>> {
    levels: Vec<Level<Ones, BV>>,
    max_symbol: u32,
    len: Ones,
}

type Dense = DenseBitVec<u32, u8>;

// Wavelet matrix construction algorithm optimized for the case where we can afford to build a
// dense histogram that counts the number of occurrences of each symbol. Heuristically,
// this is roughly the case where the alphabet size does not exceed the number of data points.
// Implements Algorithm 1 (seq.pc) from the paper "Practical Wavelet Tree Construction".
fn build_levels(data: Vec<u32>, num_levels: usize) -> Vec<Dense> {
    let mut levels = vec![BitBuf::new(data.len()); num_levels];
    let mut hist = vec![0; 1 << num_levels];
    let mut borders = vec![0; 1 << num_levels];
    let max_level = num_levels - 1;

    {
        // Count symbol occurrences and fill the first bitvector, whose bits
        // can be read from MBSs of the data in its original order.
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

        // Compute the histogram based on the previous level's one
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
fn build_levels_large_alphabet(mut data: Vec<u32>, num_levels: usize) -> Vec<Dense> {
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

    // For the last level we don'BV need to do anything but build the bitvector
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
struct Level<Ones: BitBlock, BV: BitVec<Ones>> {
    bits: BV,
    num_zeros: Ones,
    // unsigned int with a single bit set signifying
    // the magnitude represented at that level.
    // e.g.  levels[0].bitmask == 1 << levels.len() - 1
    bitmask: u32, // todo: rename to mag?
}

impl<Ones: BitBlock, T: BitVec<Ones>> Level<Ones, T> {
    fn to_left_index(&self, index: Ones) -> Ones {
        index
    }

    fn to_right_index(&self, index: Ones) -> Ones {
        self.num_zeros + index
    }

    fn to_left_range(&self, range: Range<Ones>) -> Range<Ones> {
        range
    }

    fn to_right_range(&self, range: Range<Ones>) -> Range<Ones> {
        let nz = self.num_zeros;
        nz + range.start..nz + range.end
    }

    fn to_left_symbol(&self, symbol: u32) -> u32 {
        symbol
    }

    fn to_right_symbol(&self, symbol: u32) -> u32 {
        symbol | self.bitmask
    }

    fn overlaps_left_child(&self, range: &RangeInclusive<u32>, leftmost_symbol: u32) -> bool {
        let left_start = leftmost_symbol;
        let left_end_inclusive = left_start | (self.bitmask - 1);
        intervals_overlap_inclusive(left_start, left_end_inclusive, *range.start(), *range.end())
    }

    fn overlaps_right_child(&self, range: &RangeInclusive<u32>, leftmost_symbol: u32) -> bool {
        let right_start = leftmost_symbol | self.bitmask;
        let right_end_inclusive = right_start | (self.bitmask - 1);
        intervals_overlap_inclusive(
            right_start,
            right_end_inclusive,
            *range.start(),
            *range.end(),
        )
    }

    // Returns the rank0 and rank1 at `index `:
    // (rank0(index), rank1(index))
    pub fn ranks(&self, index: Ones) -> (Ones, Ones) {
        if index.is_zero() {
            return (Ones::zero(), Ones::zero());
        }
        let num_ones = self.bits.rank1(index);
        let num_zeros = index - num_ones;
        (num_zeros, num_ones)
    }
}

impl WaveletMatrix<u32, Dense> {
    pub fn from_data(data: Vec<u32>, max_symbol: u32) -> WaveletMatrix<u32, Dense> {
        let num_levels = num_levels_for_symbol(max_symbol);
        // We implement two different wavelet matrix construction algorithms. One of them is more
        // efficient, but that algorithm does not scale well to large alphabets and also cannot
        // cannot handle element multiplicity because it constructs the bitvectors out-of-order.
        // It also requires O(2^num_levels) space. So, we check whether the number of data points
        // is less than 2^num_levels, and if so use the scalable algorithm, and otherise use the
        // the efficient algorithm.
        let levels = if num_levels < (data.len().ilog2() as usize) {
            build_levels(data, num_levels)
        } else {
            build_levels_large_alphabet(data, num_levels)
        };

        WaveletMatrix::from_levels(levels, max_symbol)
    }
}

impl<Ones: BitBlock, BV: BitVec<Ones>> WaveletMatrix<Ones, BV> {
    pub fn from_levels(levels: Vec<BV>, max_symbol: u32) -> WaveletMatrix<Ones, BV> {
        let max_level = levels.len() - 1;
        let len = levels.first().map(|level| level.len()).unwrap();

        let levels: Vec<Level<Ones, BV>> = levels
            .into_iter()
            .enumerate()
            .map(|(index, bits)| Level {
                num_zeros: bits.rank0(bits.len()),
                bitmask: 1 << (max_level - index),
                bits,
            })
            .collect();

        WaveletMatrix {
            levels,
            max_symbol,
            len,
        }
    }

    // Returns an iterator over levels from the high bit downwards, ignoring the
    // bottom `ignore_bits` levels.
    fn levels(&self, ignore_bits: usize) -> impl Iterator<Item = &Level<Ones, BV>> {
        self.levels.iter().take(self.levels.len() - ignore_bits)
    }

    pub fn len(&self) -> Ones {
        self.len
    }

    pub fn max_symbol(&self) -> u32 {
        self.max_symbol
    }

    pub fn is_valid_symbol(&self, symbol: u32) -> bool {
        symbol <= self.max_symbol
    }

    pub fn is_valid_index(&self, index: Ones) -> bool {
        index < self.len
    }

    pub fn is_valid_index_range(&self, range: Range<Ones>) -> bool {
        range.start < self.len && range.end <= self.len && range.start <= range.end
    }

    pub fn is_valid_symbol_range(&self, range: Range<u32>) -> bool {
        range.start <= self.max_symbol && range.end <= self.max_symbol && range.start <= range.end
    }
}

// todo: is this bit_ceil?
pub fn num_levels_for_symbol(symbol: u32) -> usize {
    // Equivalent to max(1, ceil(log2(alphabet_size))), which ensures
    // that we always have at least one level even if all symbols are 0.
    (u32::BITS - symbol.leading_zeros())
        .max(1)
        .try_into()
        .unwrap()
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

fn intervals_overlap_inclusive(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    a_lo <= b_hi && b_lo <= a_hi
}

// The traversal order means that outputs do not appear in the same order as inputs and
// there may be multiple outputs per input (e.g. symbols found within a given index range)
// so associating each batch with an index allows us to track the association between inputs
// and outputs.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BatchValue<Ones: BitBlock, T> {
    pub index: Ones,
    pub payload: T,
}

impl<Ones: BitBlock, T> BatchValue<Ones, T> {
    fn new(input_index: Ones, value: T) -> BatchValue<Ones, T> {
        BatchValue {
            index: input_index,
            payload: value,
        }
    }
    #[allow(dead_code)]
    fn map<U>(self, f: impl FnOnce(T) -> U) -> BatchValue<Ones, U> {
        BatchValue {
            index: self.index,
            payload: f(self.payload),
        }
    }
    fn with_payload(&self, payload: T) -> BatchValue<Ones, T> {
        BatchValue {
            index: self.index,
            payload,
        }
    }
}

// Enum for simple traversals that always recurse either to the left or right child node.
#[derive(Debug)]
enum Go<T> {
    Left(T),
    Right(T),
}

// Enum for traversals that may recurse into either, both, or neither of the child nodes.
enum GoMulti<T> {
    Left(T),
    Right(T),
    Both(T, T),
    None,
}

// Manages the BFS traversal of the wavelet matrix in a cache-friendly order.
// Each tree level is stored in a bitvector with all left children preceding
// all right children, and this traversal helper ensures that all left children
// are accessed before all right children at each level as you travel down the tree.
#[derive(Debug)]
pub struct Traversal<T> {
    left: Vec<T>,
    right: Vec<T>,
}

// todo: explore an implicit representation for batch indices based on the observation that
// during repeated traversals all of a batch's left children remain contiguous, and so do all
// of its right children. So instead of individually tracking batch indices along with each
// payload, we can simply keep track of how many left/right children each batch has, e.g.
// in an array of length 2 * the number of batches (left children first, then right children).
// And then we can yield the results in an enumerate-style iterator that repeats the batch index
// as many times as it has left (then right) children, based on the "all lefts, then all rights"
// result ordering.
// question: does this have any implications for parallellization where sets of batches are run
// concurrently – how do we ensure that all the batches reflect the global batch indices, rather
// than the subset of batches being run on each independent task?
// Another way to say this is is to observe that we can currently initialize the 'batch index'
// to anything we'd like. With the new idea, we would be limited to strictly ascending indices,
// and maybe there are cases where this becomes an issue. In the specific case of parallel batches
// we can just keep track of the "initial offset" of the batch and add that the computed index.
impl<Ones: BitBlock, T> Traversal<BatchValue<Ones, T>> {
    fn init(&mut self, values: impl IntoIterator<Item = T>) {
        self.left.clear();
        self.left.extend(
            values
                .into_iter()
                .enumerate()
                .map(|(index, value)| BatchValue::new(Ones::from_usize(index), value)),
        );
    }

    pub fn new() -> Traversal<T> {
        Traversal {
            left: Vec::new(),
            right: Vec::new(),
        }
    }

    // Traverses the data and the resulting children at the next level in the order
    // they appear in the wavelet matrix (all left children, then all right children).
    // - has the same 'called once per element left to right' guarantees as retain
    fn traverse(&mut self, f: impl Fn(&T) -> Go<T>) {
        // Retain the elements that went left, then append those that went right.
        self.left.retain_mut(|d| match f(&d.payload) {
            Go::Left(l) => {
                d.payload = l;
                true
            }
            Go::Right(r) => {
                self.right.push(d.with_payload(r));
                false
            }
        });
        self.left.append(&mut self.right);
    }

    fn traverse_multi(&mut self, f: impl Fn(&T) -> GoMulti<T>) {
        // Retain the elements that went left, then append those that went right.
        // Allow a single element to go in both directions (or neither direction).
        self.left.retain_mut(|d| match f(&d.payload) {
            GoMulti::Both(l, r) => {
                d.payload = l;
                self.right.push(d.with_payload(r));
                true
            }
            GoMulti::Left(l) => {
                d.payload = l;
                true
            }
            GoMulti::Right(r) => {
                self.right.push(d.with_payload(r));
                false
            }
            GoMulti::None => false,
        });
        self.left.append(&mut self.right);
    }

    fn result(&self) -> &Vec<BatchValue<Ones, T>> {
        &self.left
    }
}
