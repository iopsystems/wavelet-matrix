use std::error::Error;

use bio::data_structures::rank_select::RankSelect;
use bv::BitVec;
use bv::BitsMut;

fn main() -> Result<(), Box<dyn Error>> {
    println!("hello there!");

    let mut bits: BitVec<u8> = BitVec::new_fill(false, 64);
    bits.set_bit(5, true);
    bits.set_bit(32, true);
    let rs = RankSelect::new(bits, 1);
    assert!(rs.rank(6).unwrap() == 1);
    assert!(rs.select(1).unwrap() == 5);
    assert!(rs.select(2).unwrap() == 32);

    Ok(())
}
