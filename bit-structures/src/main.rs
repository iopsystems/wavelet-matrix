// For now.
#![allow(dead_code)]

use crate::histogram::Histogram;
use crate::slice_bit_vec::SliceBitVec;
use std::fs;
mod bincode_helpers;
mod bit_block;
mod bit_vec;
mod histogram;
mod slice_bit_vec;

fn main() {
    let mut h = Histogram::<SliceBitVec<u32>>::builder(0, 9, 30);
    h.increment(1, 1);
    h.increment(10_000_000, 1);
    let h = h.build();
    assert_eq!(h.quantile(0.0), 1);
    assert_eq!(h.quantile(0.25), 1);
    assert_eq!(h.quantile(0.75), 10010623);
    assert_eq!(h.quantile(1.0), 10010623);

    let data: Vec<u8> = h.encode();
    // Write the data to a file named "output.bin"
    if let Err(err) = fs::write("output.bin", data) {
        eprintln!("Error writing to file: {}", err);
    } else {
        println!("Data successfully written to file!");
    }
}
