use crate::bit_block::BitBlock;
use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec};

#[derive(Debug)]
pub struct WaveletMatrix<Ones: BitBlock, BV: BitVec<Ones>> {
    levels: Vec<Level<Ones, BV>>,
    max_symbol: u32,
    len: Ones,
}

type Dense = DenseBitVec<u32, u8>;

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
        // let levels = if num_levels < (data.len().ilog2() as usize) {
        //     build_levels(data, num_levels)
        // } else {
        //     build_levels_large_alphabet(data, num_levels)
        // };
        let levels = build_levels_large_alphabet(data, num_levels);

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
