#![allow(unused_imports)]
use crate::dense_multi_bit_vec::DenseMultiBitVec;
use crate::slice_bit_vec::SliceBitVec;
use crate::{histogram, histogram::Histogram, sparse_bit_vec::SparseBitVec, wasm_bindgen};

// todo:
// - consider leaving the 32-bit-nese implicit, leaving the suffix off of the struct name (and using 64 for the larger versions).

type Ones = u32;
// type V = SliceBitVec<Ones>;
type V = SparseBitVec<Ones>;
// type V = DenseMultiBitVec<Ones>;

#[wasm_bindgen]
pub struct Histogram32(Histogram<V>);

#[wasm_bindgen]
impl Histogram32 {
    // #[wasm_bindgen(constructor)]
    // pub fn new(a: u32, b: u32, n: u32, values: &[Ones], counts: &[Ones]) -> Histogram32 {
    //     let mut b = Histogram::<V>::builder(a, b, n);
    //     for (&value, &count) in values.iter().zip(counts.iter()) {
    //         b.increment_value(value, count)
    //     }
    //     Histogram32(b.build())
    // }
    #[wasm_bindgen(constructor)]
    pub fn from_bin_counts(
        a: u32,
        b: u32,
        n: u32,
        bin_indices: &[usize],
        counts: &[Ones],
    ) -> Histogram32 {
        // note: bin indices and counts need to be parallel but need not be sorted.
        let mut b = Histogram::<V>::builder(a, b, n);
        for (&bin_index, &count) in bin_indices.iter().zip(counts.iter()) {
            b.increment_index(bin_index, count)
        }
        Histogram32(b.build())
    }
    pub fn cumulative_count(&self, value: Ones) -> Ones {
        self.0.cumulative_count(value)
    }
    pub fn cdf(&self, value: Ones) -> f64 {
        self.0.cdf(value)
    }
    pub fn quantile(&self, q: f64) -> Ones {
        self.0.quantile(q)
    }
    pub fn quantile_to_count(&self, q: f64) -> Ones {
        self.0.quantile_to_count(q)
    }
    pub fn count(&self) -> Ones {
        self.0.count()
    }
    pub fn encode(&self) -> Vec<u8> {
        self.0.encode()
    }
    pub fn decode(data: Vec<u8>) -> Self {
        Self(Histogram::decode(data))
    }
    // convenience
    pub fn num_bins(&self) -> Ones {
        self.0.params().num_bins()
    }
    pub fn bin_index(&self, value: u64) -> u32 {
        self.0.params().bin_index(value)
    }
}

#[wasm_bindgen]
pub struct HistogramParams32(histogram::HistogramParams);

#[wasm_bindgen]
impl HistogramParams32 {
    #[wasm_bindgen(constructor)]
    pub fn from_bin_counts(a: u32, b: u32, n: u32) -> HistogramParams32 {
        let p = crate::histogram::HistogramParams::new(a, b, n);
        HistogramParams32(p)
    }

    pub fn bin_index(&self, value: u64) -> u32 {
        self.0.bin_index(value)
    }

    pub fn low(&self, bin_index: u32) -> u32 {
        self.0
            .low(bin_index)
            .try_into()
            .expect("low cannot be greater than 2^32")
    }

    pub fn high(&self, bin_index: u32) -> u32 {
        self.0
            .high(bin_index)
            .try_into()
            .expect("high cannot be greater than 2^32")
    }

    pub fn max_value(&self) -> u32 {
        self.0
            .max_value()
            .try_into()
            .expect("max_value cannot be greater than 2^32")
    }

    pub fn a(&self) -> u32 {
        self.0.a()
    }

    pub fn b(&self) -> u32 {
        self.0.b()
    }

    pub fn c(&self) -> u32 {
        self.0.c()
    }

    pub fn n(&self) -> u32 {
        self.0.n()
    }

    pub fn num_bins(&self) -> u32 {
        self.0.num_bins()
    }
}
