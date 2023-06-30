use crate::bincode_helpers::{borrow_decode_impl, decode_impl, encode_impl};
use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec};
use num::{One, Zero};
use std::{collections::VecDeque, ops::Range};

// todo
// - ignore_bits
// - batch queries
// - set operations on multiple ranges: union, intersection, ...
// - functions that accept symbol or index ranges should accept .. and x.. and ..x
//   - i think this can be implemented with a trait that has an 'expand' method, or
//     by accepting a RangeBounds and writing a  fn that replaces unbounded with 0 or len/whatever.
type Dense = DenseBitVec<u32, u8>;

struct Traversal<T> {
    cur: VecDeque<KeyValue<T>>,
    next: VecDeque<KeyValue<T>>,
    num_left: usize,
}

impl<T> Traversal<T> {
    fn from_values(values: impl IntoIterator<Item = T>) -> Self {
        let iter = values.into_iter().enumerate().map(KeyValue::from_tuple);
        Self {
            cur: VecDeque::new(),
            next: VecDeque::from_iter(iter),
            num_left: 0,
        }
    }

    fn reset(&mut self, values: impl IntoIterator<Item = T>) {
        self.cur.clear();
        self.next.clear();
        let iter = values.into_iter().enumerate().map(KeyValue::from_tuple);
        self.cur.extend(iter);
        self.num_left = 0;
    }

    fn traverse(&mut self, f: impl Fn(&[KeyValue<T>], &mut Go<KeyValue<T>>)) {
        // precondition: next contains things to traverse.
        // postcondition: next has the next things to traverse, with (reversed)
        // left children followed by (non-reversed) right children, and num_left
        // indicating the number of left elements.

        // swap next into cur, then clear next
        std::mem::swap(&mut self.cur, &mut self.next);
        self.next.clear();

        // note: rather than reversing the left subtree in advance, here's an idea:
        // we could potentially call a callback twice with the left iterator in order,
        // then the right iterator reversed, but the typing gets tricky since the left
        // and right iterators would be of different types.
        // For future ereference, the left slice would be cur[..self.num_left] and the
        // right slice would be cur[self.num_left..]
        let cur = self.cur.make_contiguous();
        cur[..self.num_left].reverse();
        // for lifetime reasons (to avoid having to pass &mut self into f), create
        // an auxiliary structure to let f recurse left and right.
        let mut go = Go {
            next: &mut self.next,
            num_left: 0,
        };
        f(cur, &mut go);
        self.num_left = go.num_left;
    }

    fn results(&mut self) -> &mut [KeyValue<T>] {
        self.next.make_contiguous()
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

#[derive(Debug)] // , bincode::Decode
pub struct WaveletMatrix<V: BitVec> {
    levels: Vec<Level<V>>,
    max_symbol: u32,
    len: V::Ones,
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

    pub fn get_batch(&self, indices: &[V::Ones]) -> Vec<V::Ones> {
        let zero = V::Ones::zero();

        // stores (wm index, symbol) batches, with each batch corresponding to an input index.
        let mut cur = VecDeque::from_iter(
            indices
                .iter()
                .copied()
                .map(|index| (index, zero))
                .enumerate()
                .map(KeyValue::from_tuple),
        );

        let mut next = VecDeque::new();

        for level in self.levels(0) {
            let mut num_left = 0;
            for x in cur.iter() {
                let (index, symbol) = x.value;
                if !level.bv.get(index) {
                    // go left
                    let index = level.bv.rank0(index);
                    // left children are appended to the front of the queue
                    next.push_front(x.value((index, symbol)));
                    num_left += 1;
                } else {
                    // go right
                    let index = level.num_zeros + level.bv.rank1(index);
                    let symbol = symbol + level.bit;
                    // right children are appended to the back of the queue
                    next.push_back(x.value((index, symbol)));
                }
            }
            // reverse the left children so that the elements are in wm left-to-right orde
            next.make_contiguous()[..num_left].reverse();
            cur.clear();
            (next, cur) = (cur, next);
        }
        // for a nicer interface, sort the batches in input order
        // and return a vec of the symbols, analogous to `get`.
        let slice = cur.make_contiguous();
        slice.sort_by_key(|x| x.key);
        slice.iter().map(|x| x.value.1).collect()
    }

    pub fn get_batch_2(&self, indices: &[V::Ones]) -> Vec<V::Ones> {
        // stores (wm index, symbol) batches, with each batch corresponding to an input index.
        let mut traversal =
            Traversal::from_values(indices.iter().copied().map(|index| (index, V::zero())));

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

        // for a nicer interface, sort the batches in input order
        // and return a vec of the symbols, analogous to `get`.
        let slice = traversal.results();
        slice.sort_by_key(|x| x.key);
        slice.iter().map(|x| x.value.1).collect()
    }

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

impl WaveletMatrix<Dense> {
    pub fn from_data(data: Vec<u32>, max_symbol: u32) -> WaveletMatrix<Dense> {
        let num_levels = num_levels_for_symbol(max_symbol);
        // We implement two different wavelet matrix construction algorithms. One of them is more
        // efficient, but that algorithm does not scale well to large alphabets and also cannot
        // cannot handle element multiplicity because it constructs the bitvectors out-of-order.
        // It also requires O(2^num_levels) space. So, we check whether the number of data points
        // is less than 2^num_levels, and if so use the scalable algorithm, and otherise use the
        // the efficient algorithm.
        let levels = if num_levels < (data.len().ilog2() as usize) {
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
    pub fn ranks(&self, index: V::Ones) -> (V::Ones, V::Ones) {
        if index.is_zero() {
            return (V::zero(), V::zero());
        }
        let num_ones = self.bv.rank1(index);
        let num_zeros = index - num_ones;
        (num_zeros, num_ones)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get() {
        let symbols = vec![1, 2, 3, 3, 2, 1, 4, 5, 6, 7, 8, 2, 9, 10];
        let max_symbol = symbols.iter().max().copied().unwrap_or(0);
        let wm = WaveletMatrix::from_data(symbols.clone(), max_symbol);
        for (i, sym) in symbols.iter().copied().enumerate() {
            assert_eq!(sym, wm.get(i as u32));
            // dbg!(sym, wm.rank(sym, 0..wm.len()));
            // dbg!(i, wm.quantile(i as u32, 0..wm.len()));
        }

        // dbg!(wm.select(10, 1, 0..wm.len()));
        let indices = &[0, 1, 2, 1, 2, 0, 13];
        dbg!(wm.get_batch(indices));
        println!("---------------------");
        dbg!(wm.get_batch_2(indices));
        // ;
        // panic!("get");
    }
}
