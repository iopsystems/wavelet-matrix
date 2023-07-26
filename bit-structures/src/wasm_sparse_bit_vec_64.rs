use crate::{bit_vec::BitVec, sparse_bit_vec::SparseBitVec, wasm_bindgen};

// We internally represent integers in u64, but the JS interface uses f64
type Ones = u64;

#[wasm_bindgen]
pub struct SparseBitVec64(SparseBitVec<Ones>);

#[wasm_bindgen]
impl SparseBitVec64 {
    #[wasm_bindgen(constructor)]
    pub fn new(ones: &[f64], universe_size: f64) -> SparseBitVec64 {
        let max_int_f64 = ((1u64 << 53) - 1) as f64; // maximum representable int in f64
        assert!(ones.len() <= u32::MAX as usize); // assert that all indices fit a u32
        assert!(universe_size <= max_int_f64); // assert that all indices fit a u32
        let ones: Vec<_> = ones
            .iter()
            .copied()
            .map(|x| {
                assert!(x <= max_int_f64);
                x as u64
            })
            .collect();
        SparseBitVec64(SparseBitVec::<Ones>::new(&ones, universe_size as u64))
    }
    pub fn rank1(&self, index: u32) -> u32 {
        self.0.rank1(index.into()) as u32
    }
    pub fn rank0(&self, index: u32) -> u32 {
        self.0.rank0(index.into()) as u32
    }
    pub fn select1(&self, n: u32) -> Option<f64> {
        self.0.try_select1(n.into()).map(|x| x as f64)
    }
    pub fn select0(&self, n: u32) -> Option<f64> {
        self.0.try_select0(n.into()).map(|x| x as f64)
    }
    pub fn get(&self, index: Ones) -> bool {
        self.0.get(index)
    }
    pub fn num_ones(&self) -> u32 {
        self.0.num_ones() as u32
    }
    pub fn num_zeros(&self) -> u32 {
        self.0.num_zeros() as u32
    }
    pub fn len(&self) -> u32 {
        self.0.universe_size() as u32
    }
    pub fn encode(&self) -> Vec<u8> {
        self.0.encode()
    }
    pub fn decode(data: Vec<u8>) -> Self {
        Self(SparseBitVec::decode(data))
    }
}
