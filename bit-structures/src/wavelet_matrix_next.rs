use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec};
use num::Zero;
use std::{
    collections::VecDeque,
    ops::{Range, RangeInclusive},
};

// todo
// - simple majority
// - k-majority
//   - note: could we use this with ignore_bits to check if eg. half of the values are in the bottom half/quarter?
//   - ie. doing majority queries on the high bits lets us make some statements about the density of values across
//     *ranges*. so rather than saying "these symbols have frequency >25%" we can say "these symbol ranges have
//     frequency >25%", for power of two frequencies (or actually arbitrary ones, based on the quantiles...right?)
// - quantile, rank w/ frequency
// - select
// - count = ranged rank
// - count symbols less than
// - set operations on multiple ranges: union, intersection, ...
type Dense = DenseBitVec<u32>;

#[derive(Debug)]
pub struct WaveletMatrix<V: BitVec> {
    levels: Vec<Level<V>>,
    max_symbol: u32,
    len: V::Ones,
}

impl<V: BitVec> WaveletMatrix<V> {
    pub fn from_bitvecs(levels: Vec<V>, max_symbol: u32) -> WaveletMatrix<V> {
        let max_level = levels.len() - 1;
        let len = levels.first().map(|level| level.len()).unwrap();
        let levels: Vec<Level<V>> = levels
            .into_iter()
            .enumerate()
            .map(|(index, bits)| Level {
                num_zeros: bits.rank0(bits.len()),
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

    // todo:
    // - prototype batch_get with a VecDeque
    // - try seeing if we can make it one continuous loop somehow
    // - this will still alternate left and right, unless we have a temporary storage to put right-recursers in...
    // - or: treat it as scratch space, push_left left guys and push_right right guys; order gets reversed...
    //   unless we push_left right children and push_right left children, which will reverse the order but
    //   be contiguous. interesting.
    // - so: maybe two vec_deques to toggle between? clear the previous one (keeping the storage), iterate the
    //   current one; can encapsulate them into a "scratch" datastructure
    pub fn get_batch(&self, indices: &[V::Ones]) -> V::Ones {
        // create two vecdeques
        let mut cur = VecDeque::<V::Ones>::with_capacity(indices.len());
        cur.extend(indices.iter().copied());
        let mut next = VecDeque::<V::Ones>::new();

        let mut index = indices.first().copied().unwrap();
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

    pub fn rank(&self, symbol: V::Ones, range: Range<V::Ones>) -> V::Ones {
        let mut range = range;
        for level in self.levels(0) {
            let start = level.ranks(range.start);
            let end = level.ranks(range.end);
            if (symbol & level.bit).is_zero() {
                // go left
                range = start.0..end.0;
            } else {
                // go right
                range = level.num_zeros + start.1..level.num_zeros + end.1;
            }
        }
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
        let frequency = range.end - range.start;
        (symbol, frequency)
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

impl<V: BitVec> Level<V> {
    fn to_left_index(&self, index: V::Ones) -> V::Ones {
        index
    }

    fn to_right_index(&self, index: V::Ones) -> V::Ones {
        self.num_zeros + index
    }

    fn to_left_range(&self, range: Range<V::Ones>) -> Range<V::Ones> {
        range
    }

    fn to_right_range(&self, range: Range<V::Ones>) -> Range<V::Ones> {
        let nz = self.num_zeros;
        nz + range.start..nz + range.end
    }

    fn to_left_symbol(&self, symbol: V::Ones) -> V::Ones {
        symbol
    }

    fn to_right_symbol(&self, symbol: V::Ones) -> V::Ones {
        symbol | self.bit
    }

    fn intervals_overlap_inclusive(
        a_lo: V::Ones,
        a_hi: V::Ones,
        b_lo: V::Ones,
        b_hi: V::Ones,
    ) -> bool {
        a_lo <= b_hi && b_lo <= a_hi
    }

    fn overlaps_left_child(
        &self,
        range: &RangeInclusive<V::Ones>,
        leftmost_symbol: V::Ones,
    ) -> bool {
        let left_start = leftmost_symbol;
        let left_end_inclusive = left_start | (self.bit - V::one());
        Self::intervals_overlap_inclusive(
            left_start,
            left_end_inclusive,
            *range.start(),
            *range.end(),
        )
    }

    fn overlaps_right_child(
        &self,
        range: &RangeInclusive<V::Ones>,
        leftmost_symbol: V::Ones,
    ) -> bool {
        let right_start = leftmost_symbol | self.bit;
        let right_end_inclusive = right_start | (self.bit - V::one());
        Self::intervals_overlap_inclusive(
            right_start,
            right_end_inclusive,
            *range.start(),
            *range.end(),
        )
    }

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
            dbg!(i, wm.quantile(i as u32, 0..wm.len()));
        }
        panic!("get");
    }
}
