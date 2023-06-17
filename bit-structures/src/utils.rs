// For now.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::ops::Range;

/// Bitwise binary search the range 0..n based on the function `ower_bound_pad`
/// from this article:
///   https://orlp.net/blog/bitwise-binary-search/
///
/// Returns the index of the partition point according to the given predicate
/// (the index of the first element of the second partition).
///
/// The slice is assumed to be partitioned according to the given predicate.
/// This means that all elements for which the predicate returns true are at
/// the start of the slice and all elements for which the predicate returns
/// false are at the end.
///
/// If this slice is not partitioned, the returned result is unspecified
/// and meaningless, as this method performs a kind of binary search.
///
/// See https://doc.rust-lang.org/1.69.0/std/primitive.slice.html#method.partition_point
///
/// See the appendix (bottom of this file for a more elaborate but efficient implementation).
pub fn partition_point(n: usize, pred: impl Fn(usize) -> bool) -> usize {
    let mut b = 0;
    let mut bit = bit_floor(n);
    while bit != 0 {
        let i = (b | bit) - 1;
        if i < n && pred(i) {
            b |= bit
        }
        bit >>= 1;
    }
    b
}

#[derive(Debug)]
pub enum Go<T> {
    Left(T),
    Right(T),
    Both(T, T),
}

/// Multiple binary search, using a similar approach as the one outlined here:
/// https://github.com/juliusmilan/multi_value_binary_search/
/// Also related: "A New Algorithm for Tiered Binary Search":
/// https://www.proquest.com/openview/9bf4d08ffb1c01d0a4854e53b87f9077/1?pq-origsite=gscholar&cbl=1976353
/// Works by searching the haystack for all needles simultaneously. When a partitioning index is sampled
/// from the haystack, we use it to split the needles into those to the left and right of the
/// partition point, recursing into either one (or both) when they are nonempty.
/// This approach is nice when accessing a needle is cheap and accessing a haystack is expensive.
/// It also works well if lots of needles end up pointing to the same place, ie. with count zero between them.
pub fn test_partition_point_multi() {
    // NOTE: just prints, does not yet test/assert.
    let haystack = [10, 25, 30, 30, 50, 100];
    let needles = [5, 6, 7, 11, 75].as_slice(); // results: [0, 0, 0, 1, 5]
    let result = batch_partition_point(
        haystack.len(),
        needles.len(),
        |i, r| {
            let value = haystack[i];
            needles[r].partition_point(|&x| x < value) // todo: < or <=?
        },
        Vec::new(),
    );
    dbg!(result);
}

// fn search(haystack: &[u32], needles: &[u32], results: &mut [usize]) {
//     let mid = haystack.len() / 2;
//     let value = haystack[mid];
//     let (left, right) = needles.split_at(needles.partition_point(|&x| x < value));
//     // (value, left)
//     // todo: < or <=?
//     // haystack.binary_search(x)
// }

/// Like partition_point but can be used to recurse in both directions at once.
/// The return value `Left(...)` is like `false` in partition_point, `Right(...)`
// is like `true`, and `Both(...)` is like both at once, recursing in both directions.
pub fn partition_point_multi<T>(
    n: usize,
    init: T,
    pred: impl Fn(usize, T) -> Go<T>,
) -> Vec<(usize, T)> {
    let mut bit = bit_floor(n);
    // todo: pass this in as scratch space, or use the stack & arguments instead?
    let mut deque = VecDeque::from([(0, init)]);
    while bit != 0 {
        for _ in 0..deque.len() {
            let (index, v) = deque.pop_front().unwrap();
            let i = (index | bit) - 1;
            if i < n {
                match pred(i, v) {
                    Go::Left(value) => deque.push_back((index, value)),
                    Go::Right(value) => deque.push_back((index | bit, value)),
                    Go::Both(left_value, right_value) => {
                        deque.push_back((index, left_value));
                        deque.push_back((index | bit, right_value))
                    }
                }
            }
        }
        bit >>= 1;
    }
    deque.into()
}

// uses the same idea as partition_point_multi, but abstracts it differently.
// given the sizes of the haystack and needlestack, it will iteratively probe
// the haystack using bitwise binary search (https://orlp.net/blog/bitwise-binary-search/)
// and at each probe, will partition the needlestack into two sub-needlestacks by calling
// the predicate with the probed haystack sample and the needle range to search.
// The predicate should return the partition point of the corresponding needlestack slice,
// ie. the number of elements that should "go left" in the recursion. Based on this, we
// recurse into one or both halves of the haystack.
pub fn batch_partition_point(
    n: usize, // size of haystack
    m: usize, // number of needles
    // pred(index, needle_lo..needle_hi) -> needle partition point in lo..hi
    // which tells us how many of the needles should "go left" in the binary search.
    pred: impl Fn(usize, Range<usize>) -> usize,
    // used as the temporary storage for processing
    workspace: Vec<(usize, Range<usize>)>,
) -> Vec<(usize, Range<usize>)> {
    let mut deque = VecDeque::from(workspace);
    deque.clear();
    deque.push_back((0, 0..m));

    let mut bit = bit_floor(n);
    while bit != 0 {
        // For each tree level, iterate through the current contents of the deque
        for _ in 0..deque.len() {
            // an (i, r) entry represents the partial haystack index i covering
            // the needle range r.
            // Based on these, we're going to evaluate the predicate with the next
            // index we'd like to check, then split the needle range at that split point.
            let (i, r) = deque.pop_front().unwrap();
            let index = (i | bit) - 1;
            if index < n {
                let split = pred(index, r.clone());
                if split > 0 {
                    deque.push_back((i, r.start..r.start + split));
                }
                if split < r.len() {
                    deque.push_back((i | bit, r.start + split..r.end));
                }
            }
        }
        bit >>= 1;
    }
    deque.into()
}

pub fn bit_floor(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        let msb = usize::BITS - 1 - n.leading_zeros();
        1 << msb
    }
}

pub fn div_ceil(n: usize, m: usize) -> usize {
    (n + m - 1) / m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foo() {
        test_partition_point_multi();
        panic!("aaah.");
    }

    #[test]
    fn test_bit_floor() {
        assert_eq!(bit_floor(0), 0);
        assert_eq!(bit_floor(1), 1);
        assert_eq!(bit_floor(2), 2);
        assert_eq!(bit_floor(3), 2);
        assert_eq!(bit_floor(4), 4);
        assert_eq!(bit_floor(5), 4);
    }

    #[test]
    fn test_partition_point() {
        let n = 100;
        let target = 60;
        assert_eq!(partition_point(n, |i| i < target), target);
        assert_eq!(partition_point(target - 1, |i| i < target), target - 1);

        assert_eq!(partition_point(0, |_| true), 0);
        assert_eq!(partition_point(1, |_| true), 1);
    }
}

/*

Appendix: Potentially faster but more complex implementation of partition_point

// Prototype implementation of the more complex but faster overlap method
// from the same article. It is the overlap version from the orlp.net article
// with some optimizations described further down in the same article.
fn partition_point_overlap(n: usize, mut pred: impl FnMut(usize) -> bool) -> usize {
    if n == 0 {
        return 0;
    };

    let two_k = bit_floor(n);
    let begin = if pred(n / 2) { n - (two_k - 1) } else { 0 };

    let mut b = 0;
    let mut bit = two_k >> 1;
    while bit != 0 {
        if pred(begin + b + bit) {
            b += bit
        }
        bit >>= 1;
    }
    begin + b + 1
}

*/
