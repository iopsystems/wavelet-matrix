pub trait BitVector {
    fn rank1(&self, index: usize) -> usize;

    fn rank0(&self, index: usize) -> usize;

    fn access(&self, index: usize) -> bool;

    fn select1(&self, n: usize) -> Option<usize>;

    fn select0(&self, n: usize) -> Option<usize>;

    fn len(&self) -> usize;

    fn num_ones(&self) -> usize;

    fn num_zeros(&self) -> usize;
}
