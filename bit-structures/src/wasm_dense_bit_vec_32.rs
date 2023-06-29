use crate::{bit_buf::BitBuf, bit_vec::BitVec, dense_bit_vec::DenseBitVec, wasm_bindgen};

type Ones = u32;

#[wasm_bindgen]
pub struct DenseBitVec32(DenseBitVec<Ones, u8>);

#[wasm_bindgen]
impl DenseBitVec32 {
    #[wasm_bindgen(constructor)]
    pub fn new(ones: &[Ones], len: Ones) -> DenseBitVec32 {
        let mut buf = BitBuf::new(len.try_into().unwrap());
        for &one in ones {
            buf.set(one.try_into().unwrap());
        }
        DenseBitVec32(DenseBitVec::<Ones>::new(buf, 10, 10))
    }
    pub fn rank1(&self, index: Ones) -> Ones {
        self.0.rank1(index)
    }
    pub fn rank0(&self, index: Ones) -> Ones {
        self.0.rank0(index)
    }
    pub fn select1(&self, n: Ones) -> Option<Ones> {
        self.0.select1(n)
    }
    pub fn select0(&self, n: Ones) -> Option<Ones> {
        self.0.select0(n)
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
        self.0.len()
    }
}
