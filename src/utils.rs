pub fn binary_search_after<T: std::cmp::PartialOrd + Copy>(
    arr: &[T],
    target: T,
    left: usize,
    right: usize,
) -> usize {
    binary_search_after_by(|i| arr[i], target, left, right)
}

pub fn binary_search_after_by<T: std::cmp::PartialOrd + Copy>(
    f: impl Fn(usize) -> T,
    target: T,
    mut left: usize,
    mut right: usize,
) -> usize {
    while left < right {
        let mid = left + ((right - left) >> 1);
        if f(mid) > target {
            right = mid
        } else {
            left = mid + 1
        }
    }
    right
}
