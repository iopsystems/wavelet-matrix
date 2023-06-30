#![allow(unused_imports)]
use crate::dense_multi_bit_vec::DenseMultiBitVec;
use crate::slice_bit_vec::SliceBitVec;
use crate::{histogram::Histogram, sparse_bit_vec::SparseBitVec, wasm_bindgen};

type Ones = u32;
// type V = SliceBitVec<Ones>;
// type V = SparseBitVec<Ones>;
type V = DenseMultiBitVec<Ones>;

#[wasm_bindgen]
pub struct Histogram32(Histogram<V>);

#[wasm_bindgen]
impl Histogram32 {
    #[wasm_bindgen(constructor)]
    pub fn new(a: u32, b: u32, n: u32, values: &[Ones], counts: &[Ones]) -> Histogram32 {
        let mut b = Histogram::<V>::builder(a, b, n);
        for (&value, &count) in values.iter().zip(counts.iter()) {
            b.increment_value(value, count)
        }
        Histogram32(b.build())
    }
    pub fn from_bin_counts(
        a: u32,
        b: u32,
        n: u32,
        bin_indices: &[usize],
        counts: &[Ones],
    ) -> Histogram32 {
        let mut b = Histogram::<V>::builder(a, b, n);
        for (&bin_index, &count) in bin_indices.iter().zip(counts.iter()) {
            b.increment_index(bin_index, count)
        }
        Histogram32(b.build())
    }
    pub fn cumulative_count(&self, value: Ones) -> Ones {
        self.0.cumulative_count(value)
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
