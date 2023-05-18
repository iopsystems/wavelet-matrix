#![allow(dead_code)]

use crate::{bitvector::BitVector, rlebitvector::RLEBitVector, simplebitvector::SimpleBitVector};
use std::{
    iter,
    ops::{Range, RangeInclusive},
};

// Implements a wavelet matrix, which is an efficient data structure for
// wavelet tree operations on top of a levelwise bitvector representation
//
// Nice description of wavelet trees:
//   https://www.alexbowe.com/wavelet-trees/
// Overview and uses of the wavelet tree:
//   https://www.sciencedirect.com/science/article/pii/S1570866713000610
// Original wavelet matrix paper:
//   https://users.dcc.uchile.cl/~gnavarro/ps/spire12.4.pdf
// Paper: Practical Wavelet Tree Construction:
//   https://dl.acm.org/doi/fullHtml/10.1145/3457197/
// Paper: New algorithms on wavelet trees and applications to information retrieval:
//   https://www.sciencedirect.com/science/article/pii/S0304397511009625/pdf?md5=32fe86d035e8a0859fd3a4b045e8b36b&pid=1-s2.0-S0304397511009625-main.pdf

// Work in progress.

// The wavelet matrix stores an ordered sequence of unsigned integer symbols, and offers
// functions that are able to operate on an arbitrary slice of the sequence in constant
// time relative to the sequence/slice length.
//
// It is fundamentally a representation of a binary tree where each level is stored as a bit
// array with nodes separated implicitly. At each level, all of the left nodes come first,
// followed by the right nodes. Child nodes are ordered in the same order as their parents
// on the previous level. I.e. if the current level has two nodes, then the children are
// ordered as
// [left child of node 1] [left child of node 2] [right child of node 1] [right child of node 2]
//
// The various functions (index, quantile, counts) are all designed to traverse the data
// in an efficient way with respect to the way that the data is laid out in memory. The
// the data is stored in levelwise bitvectors in the `levels` field, which holds a bit vector
// for each bitplane of the input sequence – e.g. levels[0].bits is a bitvector with the
// highest-order bits of every symbol.
//
// The successive levels of the wavelet matrix incrementally reorder the bits of the input
// sequence in a way that enables efficient queries (e.g. the bits corresponding to the first
// symbol appear with one bit per level, potentially at a different index on each level).
//
// The `Traversal` struct performs levelwise traversal, processing the data node by node and
// allowing recursion into one or both of the child nodes while maintaining its intermediate
// state in the same (left nodes, right nodes) order so that the bit vector rank queries for
// a given level can be done at monotonically increasing indices, leading to a more predictable
// memory access pattern to the bitvector blocks.

// todo: + 'static needed to solve a bincode compilation error that I don't understand
// ("the parameter type `T` may not live long enough; consider adding an explicit lifetime bound...")
#[derive(Debug)]
pub struct WaveletMatrix<T: BitVector + 'static> {
    levels: Vec<Level<T>>,
    max_symbol: u32,
    len: usize,
}

impl<T: BitVector> WaveletMatrix<T> {
    pub fn new(levels: Vec<T>, max_symbol: u32) -> WaveletMatrix<T> {
        let max_level = levels.len() - 1;
        let len = levels.first().map(|level| level.len()).unwrap();

        let levels: Vec<Level<T>> = levels
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

    pub fn index_batch<'a>(
        &'a self,
        indices: impl IntoIterator<Item = usize>,
        ignore_bits: usize,
        traversal: &'a mut Traversal<BatchValue<(usize, u32)>>,
    ) -> impl Iterator<Item = &BatchValue<(usize, u32)>> {
        // stores (index, symbol) batches
        traversal.init(indices.into_iter().map(|index| (index, 0)));
        for level in self.levels(ignore_bits) {
            traversal.traverse(|&(index, symbol)| {
                let (index0, index1) = level.ranks(index);
                match level.bits.access(index) {
                    false => Go::Left((level.to_left_index(index0), level.to_left_symbol(symbol))),
                    true => {
                        Go::Right((level.to_right_index(index1), level.to_right_symbol(symbol)))
                    }
                }
            });
        }
        traversal.result().iter()
    }

    pub fn count_batch<'a>(
        &'a self,
        index_ranges: impl IntoIterator<Item = Range<usize>>,
        ignore_bits: usize,
        traversal: &'a mut Traversal<BatchValue<(Range<usize>, u32)>>,
    ) -> impl Iterator<Item = &BatchValue<(Range<usize>, u32)>> {
        // stores (range, symbol) batches
        traversal.init(index_ranges.into_iter().map(|r| (r, 0)));
        for level in self.levels(ignore_bits) {
            traversal.traverse(|&(ref range, symbol)| {
                let start = level.ranks(range.start);
                let end = level.ranks(range.end);
                match symbol & level.bitmask {
                    0 => Go::Left((
                        level.to_left_range(start.0..end.0),
                        level.to_left_symbol(symbol),
                    )),
                    _ => Go::Right((
                        level.to_right_range(start.1..end.1),
                        level.to_right_symbol(symbol),
                    )),
                }
            });
        }
        traversal.result().iter()
    }

    // todo: decide whether offsets is a good name, or whether we should call them
    // sorted_indices or quantiles.
    // note: the js implementation assumes sorted input offsets and coalesces quantiles that map to the same symbol,
    // which allows it to do less work at each level of the tree and return a more compact result. might not be worth
    // optimizing if we have a small number of offsets, unless we want to compute them over a large number of index ranges.
    // the difference between JS and Rust was pretty significant (>2x), though, so I wonder what's going on...
    pub fn quantile_batch<'a>(
        &'a self,
        index_range: Range<usize>, // todo: impl intoiterator
        offsets: impl IntoIterator<Item = usize>,
        traversal: &'a mut Traversal<BatchValue<(Range<usize>, u32, usize)>>,
    ) -> impl Iterator<Item = &BatchValue<(Range<usize>, u32, usize)>> {
        // stores (range, symbol, offset) batches
        traversal.init(
            offsets
                .into_iter()
                .map(|offset| (index_range.clone(), 0, offset)),
        );
        for level in self.levels.iter() {
            traversal.traverse(|&(ref range, symbol, offset)| {
                let start = level.ranks(range.start);
                let end = level.ranks(range.end);
                let left_count = end.0 - start.0;
                if offset < left_count {
                    Go::Left((
                        level.to_left_range(start.0..end.0),
                        level.to_left_symbol(symbol),
                        offset,
                    ))
                } else {
                    Go::Right((
                        level.to_right_range(start.1..end.1),
                        level.to_right_symbol(symbol),
                        offset - left_count,
                    ))
                }
            });
        }
        traversal.result().iter()
    }

    // returns the symbols found within the given index range, and the number of times they
    // appear within that index range.
    // `symbol_range` can be used to filter the symbols we're interested in returning.
    // `ignore_bits` can be used to limit the recursion by ignoring some number of leaf levels,
    // returning counts for symbol prefixes instead.
    // todo: subcode_separator
    pub fn counts<'a>(
        &'a self,
        index_range: Range<usize>,
        symbol_range: RangeInclusive<u32>,
        ignore_bits: usize,
        traversal: &'a mut Traversal<BatchValue<(Range<usize>, u32)>>,
    ) -> impl Iterator<Item = &BatchValue<(Range<usize>, u32)>> {
        // stores (range, symbol) batches
        traversal.init(iter::once((index_range, 0)));
        for level in self.levels(ignore_bits) {
            traversal.traverse_multi(|&(ref range, symbol)| {
                let start = level.ranks(range.start);
                let end = level.ranks(range.end);

                let left_count = end.0 - start.0;
                let go_left = left_count > 0 && level.overlaps_left_child(&symbol_range, symbol);

                let right_count = end.1 - start.1;
                let go_right = right_count > 0 && level.overlaps_right_child(&symbol_range, symbol);

                use GoMulti::*;
                match (go_left, go_right) {
                    (true, true) => Both(
                        (
                            level.to_left_range(start.0..end.0),
                            level.to_left_symbol(symbol),
                        ),
                        (
                            level.to_right_range(start.1..end.1),
                            level.to_right_symbol(symbol),
                        ),
                    ),
                    (true, false) => Left((
                        level.to_left_range(start.0..end.0),
                        level.to_left_symbol(symbol),
                    )),
                    (false, true) => Right((
                        level.to_right_range(start.1..end.1),
                        level.to_right_symbol(symbol),
                    )),
                    (false, false) => None,
                }
            });
        }
        traversal.result().iter()
    }

    // Returns an iterator over levels from the high bit downwards, ignoring the
    // bottom `ignore_bits` levels.
    fn levels(&self, ignore_bits: usize) -> impl Iterator<Item = &Level<T>> {
        self.levels.iter().take(self.levels.len() - ignore_bits)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn max_symbol(&self) -> u32 {
        self.max_symbol
    }

    #[allow(dead_code)]
    pub fn is_valid_symbol(&self, symbol: u32) -> bool {
        symbol <= self.max_symbol
    }

    pub fn is_valid_index(&self, index: usize) -> bool {
        index < self.len
    }

    pub fn is_valid_index_range(&self, range: Range<usize>) -> bool {
        range.start < self.len && range.end <= self.len && range.start <= range.end
    }

    pub fn is_valid_symbol_range(&self, range: Range<u32>) -> bool {
        range.start <= self.max_symbol && range.end <= self.max_symbol && range.start <= range.end
    }
}

// Compute the number of levels needed for a Waveland matrix that is capable of representing this symbol
pub fn num_levels_for_symbol(symbol: u32) -> usize {
    // Equivalent to max(1, ceil(log2(alphabet_size))), which ensures
    // that we always have at least one level even if all symbols are 0.
    (32 - symbol.leading_zeros() as usize).max(1)
}

#[derive(Debug)]
struct Level<T: BitVector> {
    bits: T,
    num_zeros: usize,
    // unsigned int with a single bit set signifying
    // the magnitude represented at that level.
    // e.g.  levels[0].bitmask == 1 << levels.len() - 1
    bitmask: u32,
}

impl<T: BitVector> Level<T> {
    fn to_left_index(&self, index: usize) -> usize {
        index
    }

    fn to_right_index(&self, index: usize) -> usize {
        self.num_zeros + index
    }

    fn to_left_range(&self, range: Range<usize>) -> Range<usize> {
        range
    }

    fn to_right_range(&self, range: Range<usize>) -> Range<usize> {
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

    // Returns the rank0 and rank1 at `index - 1`:
    // (rank0(index - 1), rank1(index - 1))
    // todo: provide an example of why the - 1 is useful
    pub fn ranks(&self, index: usize) -> (usize, usize) {
        if index == 0 {
            return (0, 0);
        }
        let num_ones = self.bits.rank1(index - 1);
        let num_zeros = index - num_ones;
        (num_zeros, num_ones)
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
impl<T> Traversal<BatchValue<T>> {
    fn init(&mut self, values: impl IntoIterator<Item = T>) {
        self.left.clear();
        self.left.extend(
            values
                .into_iter()
                .enumerate()
                .map(|(index, value)| BatchValue::new(index, value)),
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

    fn result(&self) -> &Vec<BatchValue<T>> {
        &self.left
    }
}

impl WaveletMatrix<SimpleBitVector> {
    pub fn from_data(data: Vec<u32>, max_symbol: u32) -> WaveletMatrix<SimpleBitVector> {
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

        WaveletMatrix::new(levels, max_symbol)
    }

    pub fn from_data_with_multiplicity(
        data: Vec<(u32, usize)>,
        max_symbol: u32,
    ) -> WaveletMatrix<RLEBitVector> {
        // Equivalent to max(1, ceil(log2(alphabet_size))), which ensures
        // that we always have at least one level even if all symbols are 0.
        let num_levels = num_levels_for_symbol(max_symbol);
        let levels = build_levels_large_alphabet_with_multiplicity(data, num_levels);

        WaveletMatrix::new(levels, max_symbol)
    }
}

// Wavelet matrix construction algorithm optimized for the case where we can afford to build a
// dense histogram that counts the number of occurrences of each symbol. Heuristically,
// this is roughly the case where the alphabet size does not exceed the number of data points.
// Implements Algorithm 1 (seq.pc) from the paper "Practical Wavelet Tree Construction".
fn build_levels(data: Vec<u32>, num_levels: usize) -> Vec<SimpleBitVector> {
    let mut levels = vec![SimpleBitVector::builder(data.len()); num_levels];
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
                level.one(i);
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
                level.one(*p);
            }
            *p += 1;
        }
    }

    levels.into_iter().map(|level| level.build()).collect()
}

// Returns an array of level bitvectors built from `data`.
// Handles the sparse case where the alphabet size exceeds the number of data points and
// building a histogram with an entry for each symbol is expensive
fn build_levels_large_alphabet(mut data: Vec<u32>, num_levels: usize) -> Vec<SimpleBitVector> {
    let mut levels = Vec::with_capacity(num_levels);
    let max_level = num_levels - 1;

    // For each level, stably sort the datapoints by their bit value at that level.
    // Elements with a zero bit get sorted left, and elements with a one bits
    // get sorted right, which is effectvely a bucket sort with two buckets.
    let mut right = Vec::new();

    for l in 0..max_level {
        let level_bit = 1 << (max_level - l);
        let mut bits = SimpleBitVector::builder(data.len());
        let mut index = 0;
        // Stably sort all elements with a zero bit at this level to the left, storing
        // the positions of all one bits at this level in `bits`.
        // We retain the elements that went left, then append those that went right.
        data.retain_mut(|d| {
            let value = *d;
            let go_left = value & level_bit == 0;
            if !go_left {
                bits.one(index);
                right.push(value);
            }
            index += 1;
            go_left
        });
        data.append(&mut right);
        levels.push(bits.build());
    }

    // For the last level we don't need to do anything but build the bitvector
    {
        let mut bits = SimpleBitVector::builder(data.len());
        let level_bit = 1 << 0;
        for (index, d) in data.iter().enumerate() {
            if d & level_bit > 0 {
                bits.one(index);
            }
        }
        levels.push(bits.build());
    }

    levels
}

fn build_levels_large_alphabet_with_multiplicity(
    // vec of (value, count)
    mut data: Vec<(u32, usize)>,
    num_levels: usize,
) -> Vec<RLEBitVector> {
    let mut levels = Vec::with_capacity(num_levels);
    let max_level = num_levels - 1;

    // For each level, stably sort the datapoints by their bit value at that level.
    // Elements with a zero bit get sorted left, and elements with a one bits
    // get sorted right, which is effectvely a bucket sort with two buckets.
    let mut right = Vec::new();

    for l in 0..max_level {
        let level_bit = 1 << (max_level - l);
        let mut bits = RLEBitVector::builder();
        // Stably sort all elements with a zero bit at this level to the left, storing
        // the positions of all one bits at this level in `bits`.
        // We retain the elements that went left, then append those that went right.
        data.retain_mut(|d| {
            let (value, count) = *d;
            let go_left = value & level_bit == 0;
            if go_left {
                bits.run(count, 0);
            } else {
                bits.run(0, count);
                right.push((value, count));
            }
            go_left
        });
        data.append(&mut right);
        levels.push(bits.build());
    }

    // For the last level we don't need to do anything but build the bitvector
    {
        let mut bits = RLEBitVector::builder();
        let level_bit = 1 << 0;
        for (value, count) in data {
            let go_left = value & level_bit == 0;
            if go_left {
                bits.run(count, 0);
            } else {
                bits.run(0, count);
            }
        }
        levels.push(bits.build());
    }

    levels
}

// The traversal order means that outputs do not appear in the same order as inputs and
// there may be multiple outputs per input (e.g. symbols found within a given index range)
// so associating each batch with an index allows us to track the association between inputs
// and outputs.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BatchValue<T> {
    pub index: usize,
    pub payload: T,
}

impl<T> BatchValue<T> {
    fn new(input_index: usize, value: T) -> BatchValue<T> {
        BatchValue {
            index: input_index,
            payload: value,
        }
    }
    #[allow(dead_code)]
    fn map<U>(self, f: impl FnOnce(T) -> U) -> BatchValue<U> {
        BatchValue {
            index: self.index,
            payload: f(self.payload),
        }
    }
    fn with_payload(&self, payload: T) -> BatchValue<T> {
        BatchValue {
            index: self.index,
            payload,
        }
    }
}

fn intervals_overlap_inclusive(a_lo: u32, a_hi: u32, b_lo: u32, b_hi: u32) -> bool {
    a_lo <= b_hi && b_lo <= a_hi
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_counts() {
        let symbols = vec![1, 2, 3, 3, 2, 1, 4, 5, 6, 7, 8, 9, 10];
        let max_symbol = *symbols.iter().max().unwrap_or(&0);
        let wm = WaveletMatrix::from_data(symbols.clone(), max_symbol);
        for r in wm.counts(
            0..symbols.len(),
            0..=*symbols.iter().max().unwrap(),
            0,
            &mut Traversal::new(),
        ) {
            let (ref range, symbol) = r.payload;
            println!(
                "index: {:?} symbol: {:?} count: {:?} range: [{:?}, {:?})",
                r.index,
                symbol,
                range.end - range.start,
                range.start,
                range.end,
            );
        }
        // assert!(false)
    }
}
