// For now.
#![allow(dead_code)]

use log::info;
use wasm_bindgen::prelude::*;

mod bit_block;
mod bit_buf;
mod bit_vec;
mod dense_bit_vec;
mod histogram;
mod int_vec;
mod rle_bit_vec;
mod slice_bit_vec;
mod sparse_bit_vec;
pub mod utils;
mod wavelet_matrix;
// todo: put the wasm-related structs in a submodule?
mod wasm_dense_bit_vec_32;
mod wasm_histogram32;
mod wasm_sparse_bit_vec_32;
mod wasm_utils;

// Called by our JS entry point
#[wasm_bindgen(start)]
fn run() -> Result<(), JsValue> {
    wasm_utils::set_panic_hook();
    wasm_utils::init_log();
    info!("Initializing wasm.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_works() {
        assert_eq!(utils::bit_floor(0), 0);
    }
}
