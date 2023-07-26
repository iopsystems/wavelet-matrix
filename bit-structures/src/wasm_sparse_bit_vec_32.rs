use crate::{bit_vec::BitVec, sparse_bit_vec::SparseBitVec, wasm_bindgen};

type Ones = u32;

#[wasm_bindgen]
pub struct SparseBitVec32(SparseBitVec<Ones>);

#[wasm_bindgen]
impl SparseBitVec32 {
    #[wasm_bindgen(constructor)]
    pub fn new(ones: &[Ones], len: Ones) -> SparseBitVec32 {
        SparseBitVec32(SparseBitVec::<Ones>::new(ones, len))
    }
    pub fn rank1(&self, index: Ones) -> Ones {
        self.0.rank1(index)
    }
    pub fn rank0(&self, index: Ones) -> Ones {
        self.0.rank0(index)
    }
    pub fn select1(&self, n: Ones) -> Option<Ones> {
        self.0.try_select1(n)
    }
    pub fn select0(&self, n: Ones) -> Option<Ones> {
        self.0.try_select0(n)
    }
    pub fn get(&self, index: Ones) -> bool {
        self.0.get(index)
    }
    pub fn num_ones(&self) -> Ones {
        self.0.num_ones()
    }
    pub fn num_zeros(&self) -> Ones {
        self.0.num_zeros()
    }
    pub fn len(&self) -> Ones {
        self.0.universe_size()
    }
    pub fn encode(&self) -> Vec<u8> {
        self.0.encode()
    }
    pub fn decode(data: Vec<u8>) -> Self {
        Self(SparseBitVec::decode(data))
    }
}
