// For now.
#![allow(dead_code)]

mod dense_bitvector;
mod fixed_width_intvector;
mod raw_bitvector;
mod sparse_bitvector;
mod utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_works() {
        assert_eq!(utils::bit_floor(0), 0);
    }
}
