use crate::bit_block::BitBlock;
use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec};

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
