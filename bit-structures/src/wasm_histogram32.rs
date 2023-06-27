use crate::{histogram::Histogram, slice_bit_vec::SliceBitVec, wasm_bindgen};

type Ones = u32;

#[wasm_bindgen]
pub struct Histogram32(Histogram<SliceBitVec<Ones>>);

#[wasm_bindgen]
impl Histogram32 {
    #[wasm_bindgen(constructor)]
    pub fn new(a: u32, b: u32, n: u32, values: &[Ones], counts: &[Ones]) -> Histogram32 {
        let mut b = Histogram::<SliceBitVec<Ones>>::builder(a, b, n);
        for (&value, &count) in values.iter().zip(counts.iter()) {
            b.increment(value, count)
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
}
