use crate::dense_bit_vec::DenseBitVec;
use crate::nonempty_extent::Extent;
use crate::{wasm_bindgen, wavelet_matrix::WaveletMatrix};
use js_sys::Reflect;
use js_sys::Uint32Array;
use wasm_bindgen::JsValue;

// todo
// - investigate exposing exclusive symbol range from the js side, which can represent 33-bit integers
//   and accept the argument as a u64..
// - return named fields startIndex/endIndex rather than start/end
type Ones = u32;

#[wasm_bindgen]
pub struct SymbolCount {
    pub symbol: Ones,
    pub count: Ones,
}

#[wasm_bindgen]
pub struct WaveletMatrix32(WaveletMatrix<DenseBitVec<Ones>>);

fn box_to_ref<T: ?Sized>(b: &Option<Box<T>>) -> Option<&T> {
    b.as_ref().map(|x| x.as_ref())
}

#[wasm_bindgen]
impl WaveletMatrix32 {
    pub fn locate_raw(
        &self,
        range_lo: Option<Box<[u32]>>,
        range_hi: Option<Box<[u32]>>,
        symbols: Box<[u32]>,
    ) -> Result<JsValue, String> {
        let ranges = if let (Some(range_lo), Some(range_hi)) = (range_lo, range_hi) {
            assert!(range_lo.len() == range_hi.len());
            let mut ranges = Vec::with_capacity(range_lo.len());
            for (&lo, &hi) in range_lo.iter().zip(range_hi.iter()) {
                ranges.push(lo..hi)
            }
            ranges
        } else {
            vec![0..self.0.len()]
        };

        let mut traversal = self.0.locate_batch(&ranges, &symbols);

        let mut symbol_index = Vec::new();
        let mut range_index = Vec::new();
        let mut symbol = Vec::new();
        let mut count = Vec::new(); // this is derivable from start/end but convenient to include
        let mut preceding_count = Vec::new();
        let mut start = Vec::new();
        let mut end = Vec::new();
        // let num_symbols = symbols.len();
        let num_ranges = ranges.len();
        for x in traversal.results() {
            symbol_index.push((x.key / num_ranges) as u32);
            range_index.push((x.key % num_ranges) as u32);
            symbol.push(x.val.0);
            preceding_count.push(x.val.1);
            start.push(x.val.2);
            end.push(x.val.3);
            count.push(x.val.3 - x.val.2);
        }
        let obj = js_sys::Object::new();
        let err = "could not set js property";
        Reflect::set(
            &obj,
            &"range_index".into(),
            &Uint32Array::from(&range_index[..]),
        )
        .expect(err);

        Reflect::set(
            &obj,
            &"symbol_index".into(),
            &Uint32Array::from(&symbol_index[..]),
        )
        .expect(err);

        Reflect::set(&obj, &"symbol".into(), &Uint32Array::from(&symbol[..])).expect(err);
        Reflect::set(&obj, &"count".into(), &Uint32Array::from(&count[..])).expect(err);
        Reflect::set(
            &obj,
            &"preceding_count".into(),
            &Uint32Array::from(&preceding_count[..]),
        )
        .expect(err);
        Reflect::set(&obj, &"startIndex".into(), &Uint32Array::from(&start[..])).expect(err);
        Reflect::set(&obj, &"endIndex".into(), &Uint32Array::from(&end[..])).expect(err);
        Reflect::set(&obj, &"length".into(), &symbol.len().into()).expect(err);

        Ok(obj.into())
    }

    pub fn count_raw(
        &self,
        range_lo: Option<u32>,
        range_hi: Option<u32>,
        symbol_lo: Option<Box<[u32]>>,
        symbol_hi: Option<Box<[u32]>>,
        masks: Option<Box<[u32]>>,
    ) -> Result<JsValue, String> {
        let range = range_lo.unwrap_or(0)..range_hi.unwrap_or(self.0.len());

        let symbol_ranges = if let (Some(symbol_lo), Some(symbol_hi)) = (symbol_lo, symbol_hi) {
            assert!(symbol_lo.len() == symbol_hi.len());
            let mut symbol_ranges = Vec::with_capacity(symbol_lo.len());
            for (&lo, &hi) in symbol_lo.iter().zip(symbol_hi.iter()) {
                symbol_ranges.push(lo..hi + 1) // for now .count_batch takes exclusive symbol ranges
            }
            symbol_ranges
        } else {
            vec![0..self.0.max_symbol() + 1] // for now .count_batch takes exclusive symbol ranges
        };

        let masks = box_to_ref(&masks);

        let counts = self.0.count_batch(range, &symbol_ranges, masks);
        Ok(Uint32Array::from(&counts[..]).into())
    }

    pub fn counts_raw(
        &self,
        range_lo: Option<Box<[u32]>>,
        range_hi: Option<Box<[u32]>>,
        symbol_lo: Option<u32>,
        symbol_hi: Option<u32>,
        masks: Option<Box<[u32]>>,
    ) -> Result<JsValue, String> {
        let ranges = if let (Some(range_lo), Some(range_hi)) = (range_lo, range_hi) {
            assert!(range_lo.len() == range_hi.len());
            let mut ranges = Vec::with_capacity(range_lo.len());
            for (&lo, &hi) in range_lo.iter().zip(range_hi.iter()) {
                ranges.push(lo..hi)
            }
            ranges
        } else {
            vec![0..self.0.len()]
        };

        let symbols = Extent::new(
            symbol_lo.unwrap_or(0),
            symbol_hi.unwrap_or(self.0.max_symbol()),
        );

        let masks = box_to_ref(&masks);

        let mut traversal = self.0.counts(&ranges, symbols, masks);

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
        Reflect::set(&obj, &"startIndex".into(), &Uint32Array::from(&start[..])).expect(err);
        Reflect::set(&obj, &"endIndex".into(), &Uint32Array::from(&end[..])).expect(err);
        Reflect::set(&obj, &"length".into(), &symbol.len().into()).expect(err);

        Ok(obj.into())
    }

    #[wasm_bindgen(constructor)]
    pub fn new(data: &[Ones], max_symbol: Option<Ones>) -> WaveletMatrix32 {
        // note: to_vec copies the data. ideally we would just take ownership of the passed-in data.
        // that might involve passing it in as other than a &[Ones].
        // we need a vector for large alphabet construction since it uses retain_mut.
        WaveletMatrix32(WaveletMatrix::new(
            data.to_vec(),
            max_symbol.unwrap_or_else(|| data.iter().copied().max().unwrap_or(0)),
        ))
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
        self.0.count(range_lo..range_hi, symbol)
    }

    pub fn quantile(&self, k: Ones, range_lo: Ones, range_hi: Ones) -> SymbolCount {
        let (symbol, count) = self.0.quantile(k, range_lo..range_hi);
        SymbolCount { symbol, count }
    }

    pub fn select(&self, symbol: Ones, k: Ones, range_lo: Ones, range_hi: Ones) -> Option<Ones> {
        self.0.select(symbol, k, range_lo..range_hi)
    }

    pub fn morton_masks_for_dims(&self, dims: u32) -> Result<JsValue, String> {
        let masks = self.0.morton_masks_for_dims(dims);
        Ok(Uint32Array::from(&masks[..]).into())
    }

    // assertPowerOfTwoSymbols = false, // optionally turn on additional error checking for ignorebits

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
