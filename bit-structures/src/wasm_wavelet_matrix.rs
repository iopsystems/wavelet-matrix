use crate::dense_bit_vec::DenseBitVec;
use crate::{wasm_bindgen, wavelet_matrix::WaveletMatrix};
use js_sys::Uint32Array;
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
        WaveletMatrix32(WaveletMatrix::new(data.to_vec(), max_symbol))
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

    pub fn count_all(&self, range_lo: Ones, range_hi: Ones) -> Result<JsValue, String> {
        self.count_all_batch(&[range_lo, range_hi])
    }

    // ranges alternates lo, hi, lo, hi, ...
    // because i could not find an efficient way to pass a nested slice or similar
    pub fn count_all_batch(&self, ranges: &[Ones]) -> Result<JsValue, String> {
        assert!(ranges.len() % 2 == 0,);
        let ranges: Vec<_> = ranges.chunks_exact(2).map(|x| x[0]..x[1]).collect();
        let mut traversal = self.0.count_all_batch(&ranges);
        let mut input_index = Vec::new();
        let mut symbol = Vec::new();
        let mut start = Vec::new();
        let mut end = Vec::new();
        for x in traversal.results() {
            input_index.push(Ones::try_from(x.key).unwrap());
            symbol.push(x.value.symbol);
            start.push(x.value.start);
            end.push(x.value.end);
        }
        let obj = js_sys::Object::new();
        let err = "could not set js property";
        js_sys::Reflect::set(
            &obj,
            &"input_index".into(),
            &Uint32Array::from(&input_index[..]),
        )
        .expect(err);
        js_sys::Reflect::set(&obj, &"symbol".into(), &Uint32Array::from(&symbol[..])).expect(err);
        js_sys::Reflect::set(&obj, &"start".into(), &Uint32Array::from(&start[..])).expect(err);
        js_sys::Reflect::set(&obj, &"end".into(), &Uint32Array::from(&end[..])).expect(err);
        js_sys::Reflect::set(&obj, &"length".into(), &symbol.len().into()).expect(err);
        Ok(obj.into())
    }

    pub fn len(&self) -> Ones {
        self.0.len()
    }

    pub fn num_levels(&self) -> usize {
        self.0.num_levels()
    }

    pub fn encode(&self) -> Vec<u8> {
        self.0.encode()
    }
    pub fn decode(data: Vec<u8>) -> Self {
        Self(WaveletMatrix::decode(data))
    }
}
