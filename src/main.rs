use std::error::Error;

use simple_sds::ops::{BitVec, Rank, Select};
use simple_sds::sparse_vector::{SparseBuilder, SparseVector};

fn main() -> Result<(), Box<dyn Error>> {
    println!("hello there!");

    let len = 100;
    let ones = [1, 3];

    let mut b = SparseBuilder::new(len, ones.len()).unwrap();
    b.extend(ones.into_iter());
    let v = SparseVector::try_from(b).unwrap();

    for i in 0..=ones.last().copied().unwrap() {
        dbg!(i, v.rank(i), v.select(i));
    }

    dbg!(v.len());

    Ok(())
}
