pub mod bitvector;
pub mod originalrlebitvector;
pub mod rlebitvector;
pub mod simplebitvector;
pub mod sparsebitvector;
mod utils;
mod waveletmatrix;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
