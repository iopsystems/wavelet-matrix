#![allow(unused_imports)]
use crate::dense_multi_bit_vec::DenseMultiBitVec;
use crate::histogram::HistogramParams;
use crate::slice_bit_vec::SliceBitVec;
use crate::wasm_histogram32;
use crate::{histogram, histogram::Histogram, sparse_bit_vec::SparseBitVec, wasm_bindgen};

// todo:
// - consider leaving the 32-bit-nese implicit, leaving the suffix off of the struct name (and using 64 for the larger versions).
// - rename to all xxxx32 to xxxx64
// - we currently construct sparse histograms inefficiently, since the builder creates the dense cdf.

const MAX_INT_F64: f64 = ((1u64 << 53) - 1) as f64; // maximum representable int in f64

type V = DenseMultiBitVec<u64>;

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
        n: Option<u32>,
        bin_indices: &[usize],
        counts: &[f64],
    ) -> Histogram32 {
        // compute an n value from the maximum bin index if one is not provided.
        // this matters for now because the builder will preallocate an array for the pdf,
        // and the cdf will be larger, too.
        let n = n.unwrap_or_else(|| {
            let p = HistogramParams::new(a, b, 64);
            let max_value = p.high(*bin_indices.iter().max().unwrap_or(&0) as u32);
            (max_value as f64).log2().ceil() as u32
        });
        // log::info!("n: {}", n);
        // note: bin indices and counts need to be parallel but need not be sorted.
        let mut b = Histogram::<V>::builder(a, b, n);
        for (&bin_index, &count) in bin_indices.iter().zip(counts.iter()) {
            assert!(count <= MAX_INT_F64);
            b.increment_index(bin_index, count as u64);
        }
        Histogram32(b.build())
    }
    pub fn cumulative_count(&self, value: f64) -> f64 {
        assert!(value <= MAX_INT_F64);
        self.0.cumulative_count(value as u64) as f64
    }
    pub fn cdf(&self, value: f64) -> f64 {
        assert!(value <= MAX_INT_F64);
        self.0.cdf(value as u64)
    }
    pub fn quantile(&self, q: f64) -> f64 {
        self.0.quantile(q) as f64
    }
    pub fn quantile_to_count(&self, q: f64) -> f64 {
        self.0.quantile_to_count(q) as f64
    }
    pub fn count(&self) -> f64 {
        self.0.count() as f64
    }
    pub fn encode(&self) -> Vec<u8> {
        self.0.encode()
    }
    pub fn decode(data: Vec<u8>) -> Self {
        Self(Histogram::decode(data))
    }
    // convenience
    pub fn num_bins(&self) -> u32 {
        self.0.params().num_bins()
    }
    pub fn bin_index(&self, value: u64) -> u32 {
        self.0.params().bin_index(value)
    }
    pub fn params(&self) -> HistogramParams32 {
        wasm_histogram32::HistogramParams32(self.0.params())
    }
}

#[wasm_bindgen]
pub struct HistogramParams32(histogram::HistogramParams);

#[wasm_bindgen]
impl HistogramParams32 {
    #[wasm_bindgen(constructor)]
    pub fn new(a: u32, b: u32, n: u32) -> HistogramParams32 {
        let p = crate::histogram::HistogramParams::new(a, b, n);
        HistogramParams32(p)
    }

    pub fn bin_index(&self, value: u64) -> u32 {
        self.0.bin_index(value)
    }

    pub fn bin_width(&self, bin_index: u32) -> u32 {
        self.0
            .bin_width(bin_index)
            .try_into()
            .expect("low cannot be greater than 2^32")
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

#[wasm_bindgen]
pub struct HistogramBuilder32(histogram::HistogramBuilder<V>);

#[wasm_bindgen]
impl HistogramBuilder32 {
    #[wasm_bindgen(constructor)]
    pub fn new(a: u32, b: u32, n: u32) -> HistogramBuilder32 {
        let p = histogram::HistogramBuilder::new(a, b, n);
        HistogramBuilder32(p)
    }

    pub fn increment_value(&mut self, value: f64, count: f64) {
        assert!(value <= MAX_INT_F64 && count <= MAX_INT_F64);
        self.0.increment_value(value as u64, count as u64)
    }

    pub fn increment_index(&mut self, bin_index: usize, count: f64) {
        assert!(count <= MAX_INT_F64);
        self.0.increment_index(bin_index, count as u64)
    }

    pub fn build(self) -> Histogram32 {
        Histogram32(self.0.build())
    }
}
