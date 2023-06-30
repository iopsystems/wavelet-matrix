use crate::dense_bit_vec::DenseBitVec;
use crate::{wasm_bindgen, wavelet_matrix::WaveletMatrix};
use wasm_bindgen::JsValue;

// note: the Ones type refers to the length of the WM (since that is what determines bitvec size).
// currently the symbol type is always u32.
type Ones = u32;

#[wasm_bindgen]
pub struct SymbolCount {
    pub symbol: Ones,
    pub count: Ones,
}

#[wasm_bindgen]
pub struct WaveletMatrix32(WaveletMatrix<DenseBitVec<Ones>>);

#[wasm_bindgen]
impl WaveletMatrix32 {
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[Ones], max_symbol: Ones) -> WaveletMatrix32 {
        // note: to_vec copies the data. ideally we would just take ownership of the passed-in data.
        // that might involve passing it in as other than a &[Ones].
        // we need a vector for large alphabet construction since it uses retain_mut.
        WaveletMatrix32(WaveletMatrix::from_data(data.to_vec(), max_symbol))
    }

    pub fn simple_majority(&self, range_lo: Ones, range_hi: Ones) -> Option<Ones> {
        self.0.simple_majority(range_lo..range_hi)
    }

    pub fn get(&self, index: Ones) -> Ones {
        self.0.get(index)
    }

    pub fn preceding_count(&self, symbol: Ones, range_lo: Ones, range_hi: Ones) -> Ones {
        self.0.preceding_count(symbol, range_lo..range_hi)
    }

    pub fn count(&self, symbol: Ones, range_lo: Ones, range_hi: Ones) -> Ones {
        self.0.count(symbol, range_lo..range_hi)
    }

    pub fn quantile(&self, k: Ones, range_lo: Ones, range_hi: Ones) -> SymbolCount {
        let (symbol, count) = self.0.quantile(k, range_lo..range_hi);
        SymbolCount { symbol, count }
    }

    pub fn select(&self, symbol: Ones, k: Ones, range_lo: Ones, range_hi: Ones) -> Option<Ones> {
        self.0.select(symbol, k, range_lo..range_hi)
    }
    
    pub fn len(&self) -> Ones {
        self.0.len()
    }

    pub fn count_all(&self, range_lo: Ones, range_hi: Ones) -> Result<JsValue, String> {
        let results = self.0.count_all(range_lo..range_hi);
        let mut symbols = Vec::new();
        let mut counts = Vec::new();
        for (symbol, count) in results {
            symbols.push(symbol);
            counts.push(count)
        }
        let symbols = js_sys::Uint32Array::from(&symbols[..]);
        let counts = js_sys::Uint32Array::from(&counts[..]);
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"symbols".into(), &symbols).expect("could not set symbols");
        js_sys::Reflect::set(&obj, &"counts".into(), &counts).expect("could not set counts");
        Ok(obj.into())
    }

    pub fn encode(&self) -> Vec<u8> {
        self.0.encode()
    }
    pub fn decode(data: Vec<u8>) -> Self {
        Self(WaveletMatrix::decode(data))
    }
}
