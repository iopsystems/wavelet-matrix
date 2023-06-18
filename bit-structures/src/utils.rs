// For now.
#![allow(dead_code)]

use std::{collections::VecDeque, debug_assert};

/// Bitwise binary search the range 0..n based on the function `lower_bound_pad`
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

/// Multiple binary search, using a similar approach as the one outlined here:
/// https://github.com/juliusmilan/multi_value_binary_search/
/// Also related: "A New Algorithm for Tiered Binary Search":
/// https://www.proquest.com/openview/9bf4d08ffb1c01d0a4854e53b87f9077/1?pq-origsite=gscholar&cbl=1976353
/// Works by searching the haystack for all needles simultaneously. When a partitioning index is sampled
/// from the haystack, we use it to split the needles into those to the left and right of the
/// partition point, recursing into either one (or both) when they are nonempty.
/// This approach is nice when accessing a needle is cheap and accessing a haystack is expensive.
/// It also works well if lots of needles end up pointing to the same place, ie. with count zero between them.

// uses the same idea as partition_point_multi, but abstracts it differently.
// given the sizes of the haystack and needlestack, it will iteratively probe
// the haystack using bitwise binary search (https://orlp.net/blog/bitwise-binary-search/)
// and at each probe, will partition the needlestack into two sub-needlestacks by calling
// the predicate with the probed haystack sample and the needle range to search.
// The predicate should return the partition point of the corresponding needlestack slice,
// ie. the number of elements that should "go left" in the recursion. Based on this, we
// recurse into one or both halves of the haystack.
// Note that bitwise binary search as per `lower_bound_pad` in the article does 0.2 more
// comparisons than is optimal (on average, assuming a uniform query distribution).

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

pub fn batch_partition_point(
    n: usize, // size of haystack
    m: usize, // number of needles
    // pred(index, needle_lo..needle_hi) -> needle partition point in lo..hi
    // which tells us how many of the needles should "go left" in the binary search.
    pred: impl Fn(usize, usize, usize) -> usize,
    // also used as the temporary storage for processing
    results: &mut VecDeque<(usize, usize)>,
) {
    results.clear();
    results.push_back((0, m));

    // Bitwise binary search over the haystack.
    // Since we're searching for multiple needles, the search may
    // descend into both partitions, depending on the result of `pred`.
    let mut bit = bit_floor(n);
    while bit != 0 {
        // The deque entries store the `hi` of each needle range.
        // The corresponding `lo` is always either zero or the `hi`
        // of the previous range.
        let mut lo = 0;
        // Iterate through the current contents of the deque. Since
        // the deque is mutated inside this loop we use an unsightly
        // approach of looping a fixed number of times.
        for _ in 0..results.len() {
            let (i, hi) = results.pop_front().unwrap();
            let index = (i | bit) - 1;
            // if the index is beyond the end of the array, send all needles to the left
            let split = if index < n { pred(index, lo, hi) } else { hi };
            debug_assert!(lo <= split);
            debug_assert!(split <= hi);
            debug_assert!(lo < hi);
            if split > lo {
                results.push_back((i, split));
            }
            if split < hi {
                results.push_back((i | bit, hi));
            }
            lo = hi;
        }
        bit >>= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_partition_point_multi() {
        // NOTE: just prints, does not yet test/assert.
        let haystack = [10, 100];
        let needles = [1, 2, 11, 12, 13, 111, 112].as_slice();
        let mut results = VecDeque::new();
        batch_partition_point(
            haystack.len(),
            needles.len(),
            |i, lo, hi| {
                let value = haystack[i];
                lo + needles[lo..hi].partition_point(|&x| x < value)
            },
            &mut results,
        );
        dbg!(results);
        panic!("nooo");
    }

    #[test]
    fn test_foo() {
        // let haystack = vec![0, 10, 20, len - 1];
        // let needles = vec![0, 10, 20, len - 1];
        // let mut gen = Gen::new();
        // while !gen.done() {
        //     let ones = gen.gen_subset(&input);
        //     test_cases.push(TestCase(ones.copied().collect(), len));
        // }
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
