// For now.
#![allow(dead_code)]

mod bit_block;
mod bit_buf;
mod bit_vec;
mod compressed_bit_vec;
mod dense_bit_vec;
mod histogram;
mod int_vec;
mod rle_bit_vec;
mod slice_bit_vec;
mod sparse_bit_vec;
pub mod utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_works() {
        assert_eq!(utils::bit_floor(0), 0);
    }
}
