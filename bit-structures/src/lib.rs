// For now.
#![allow(dead_code)]

mod bit_block;
mod bit_vector;
mod compressed_bit_vector;
mod dense_bit_vector;
mod int_vector;
mod naive_bit_vector;
mod new_raw_bit_vector;
mod raw_bit_vector;
mod sparse_bit_vector;
mod utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_works() {
        assert_eq!(utils::bit_floor(0), 0);
    }
}
