use crate::dense_bit_vec::DenseBitVec;
use crate::nonempty_extent::Extent;
use crate::wavelet_matrix;
use crate::{wasm_bindgen, wavelet_matrix::WaveletMatrix};
use js_sys::Reflect;
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
    pub fn counts_rs(
        &self,
        symbol_lo: Option<u32>,
        symbol_hi: Option<u32>,
        range_lo: Option<u32>,
        range_hi: Option<u32>,
        masks: Option<Box<[u32]>>,
    ) -> Result<JsValue, String> {
        let range_lo = range_lo.unwrap_or(0);
        let range_hi = range_hi.unwrap_or(self.0.len());
        let symbol_lo = symbol_lo.unwrap_or(0);
        let symbol_hi = symbol_hi.unwrap_or(self.0.max_symbol());
        let masks = masks.as_ref().map(|x| x.as_ref());

        let range = range_lo..range_hi;
        let symbols = Extent::new(symbol_lo, symbol_hi);
        let mut traversal = self.0.count_all_batch(symbols, &[range], masks);

        let mut input_index = Vec::new();
        let mut symbol = Vec::new();
        let mut start = Vec::new();
        let mut end = Vec::new();
        // add this for now, even though it could be computed from start and end.
        let mut count = Vec::new();
        for x in traversal.results() {
            input_index.push(Ones::try_from(x.key).unwrap());
            symbol.push(x.val.symbol);
            start.push(x.val.start);
            end.push(x.val.end);
            count.push(x.val.end - x.val.start);
        }
        let obj = js_sys::Object::new();
        let err = "could not set js property";
        Reflect::set(
            &obj,
            &"input_index".into(),
            &Uint32Array::from(&input_index[..]),
        )
        .expect(err);

        Reflect::set(&obj, &"symbol".into(), &Uint32Array::from(&symbol[..])).expect(err);
        // put count right after symbol for better output in the observable inspector
        Reflect::set(&obj, &"count".into(), &Uint32Array::from(&count[..])).expect(err);
        Reflect::set(&obj, &"start".into(), &Uint32Array::from(&start[..])).expect(err);
        Reflect::set(&obj, &"end".into(), &Uint32Array::from(&end[..])).expect(err);
        Reflect::set(&obj, &"length".into(), &symbol.len().into()).expect(err);

        Ok(obj.into())
    }

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

    pub fn count_all(
        &self,
        symbol_range_lo: Ones,
        symbol_range_hi_inclusve: Ones,
        range_lo: Ones,
        range_hi: Ones,
        masks: &[u32],
    ) -> Result<JsValue, String> {
        self.count_all_batch(
            symbol_range_lo,
            symbol_range_hi_inclusve,
            &[range_lo, range_hi],
            masks,
        )
    }

    // first = 0,
    // last = this.length,
    // lower = 0,
    // upper = this.maxSymbol,
    // ignoreBits = 0,
    // assertPowerOfTwoSymbols = false, // optionally turn on additional error checking for ignorebits
    // subcodeSeparator = 0,
    // sort = false,

    // ranges alternates lo, hi, lo, hi, ...
    // because i could not find an efficient way to pass a nested slice or similar
    pub fn count_all_batch(
        &self,
        symbol_range_lo: Ones,
        symbol_range_hi_inclusve: Ones,
        ranges: &[Ones],
        masks: &[u32],
    ) -> Result<JsValue, String> {
        assert!(ranges.len() % 2 == 0,);
        let ranges: Vec<_> = ranges.chunks_exact(2).map(|x| x[0]..x[1]).collect();
        let mut traversal = self.0.count_all_batch(
            // Extent::new(0, self.0.max_symbol()),
            Extent::new(symbol_range_lo, symbol_range_hi_inclusve),
            &ranges,
            Some(masks),
        );
        let mut input_index = Vec::new();
        let mut symbol = Vec::new();
        let mut start = Vec::new();
        let mut end = Vec::new();
        // add this for now, even though it could be computed from start and end.
        let mut count = Vec::new();
        for x in traversal.results() {
            input_index.push(Ones::try_from(x.key).unwrap());
            symbol.push(x.val.symbol);
            start.push(x.val.start);
            end.push(x.val.end);
            count.push(x.val.end - x.val.start);
        }
        let obj = js_sys::Object::new();
        let err = "could not set js property";
        Reflect::set(
            &obj,
            &"input_index".into(),
            &Uint32Array::from(&input_index[..]),
        )
        .expect(err);

        Reflect::set(&obj, &"symbol".into(), &Uint32Array::from(&symbol[..])).expect(err);
        // put count right after symbol for better output in the observable inspector
        Reflect::set(&obj, &"count".into(), &Uint32Array::from(&count[..])).expect(err);
        Reflect::set(&obj, &"start".into(), &Uint32Array::from(&start[..])).expect(err);
        Reflect::set(&obj, &"end".into(), &Uint32Array::from(&end[..])).expect(err);
        Reflect::set(&obj, &"length".into(), &symbol.len().into()).expect(err);
        // js_sys::Array::get(&self, index)

        Ok(obj.into())
    }

    pub fn count_symbol_range_batch(
        &self,
        symbol_ranges: &[Ones],
        range_lo: u32,
        range_hi: u32,
        dims: u32,
    ) -> Result<JsValue, String> {
        assert!(symbol_ranges.len() % 2 == 0,);
        let symbol_ranges: Vec<_> = symbol_ranges.chunks_exact(2).map(|x| x[0]..x[1]).collect();
        let range = range_lo..range_hi;
        let masks = wavelet_matrix::morton_masks_for_dims(dims, self.0.num_levels());
        let counts = self
            .0
            .count_symbol_range_batch(&symbol_ranges, range, Some(&masks));
        let obj = js_sys::Object::new();
        let err = "could not set js property";
        Reflect::set(&obj, &"counts".into(), &Uint32Array::from(&counts[..])).expect(err);
        Reflect::set(&obj, &"length".into(), &counts.len().into()).expect(err);
        Ok(obj.into())
    }
    pub fn len(&self) -> Ones {
        self.0.len()
    }

    pub fn num_levels(&self) -> usize {
        self.0.num_levels()
    }

    pub fn max_symbol(&self) -> u32 {
        self.0.max_symbol()
    }

    pub fn encode(&self) -> Vec<u8> {
        self.0.encode()
    }
    pub fn decode(data: Vec<u8>) -> Self {
        Self(WaveletMatrix::decode(data))
    }
}
