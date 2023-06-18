use std::collections::VecDeque;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::distributions::Uniform;
use rand::prelude::Distribution;

// todo: explore the parameters
// - size of haystack
// - size of needles
// - clustered-ness of haystack
//   - find a nice parameterization of "clusterness", so we can dial it up and down
// - clustered-ness of needles
//   - this is wrt. the haystack - do many needles land in the same spot, and therefore save work
//     since we can evaluate the predicate one for them all?
// - expense of predicate evaluation (accessing haystack, and work to segment the needles)
//   - with evenly-spaced needles, computing the partition point should be constant-time
fn bench_binary_search(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut haystack: Vec<u64> = Vec::new();
    haystack.resize(5_000_000, 0);
    let mut acc = 0;
    let unif = Uniform::new(0, 1000);
    for x in &mut haystack {
        acc += unif.sample(&mut rng);
        *x = acc;
    }
    assert!(haystack.windows(2).all(|w| w[0] <= w[1])); // assert sorted
    dbg!(haystack.last().unwrap());

    let mut group = c.benchmark_group("group a");
    for num_needles in [10_000, 100_000] {
        let rng = rand::thread_rng();
        // generate needles
        let unif = Uniform::new(0, haystack.last().unwrap() + 1);
        // let unif = Uniform::new(100_000, 1_000_000); // or, generate needles in a narrow range
        let mut needles: Vec<_> = unif.sample_iter(rng).take(num_needles).collect();
        needles.sort();
        let n = haystack.len();
        let m = needles.len();

        // group.bench_function(
        //     BenchmarkId::new("slice.partition_point", needles.len()),
        //     |b| {
        //         b.iter(|| {
        //             let mut ret = 0;
        //             for needle in needles.iter().copied() {
        //                 ret += haystack.partition_point(|&x| x < needle);
        //             }
        //             ret
        //         })
        //     },
        // );

        // group.bench_function(BenchmarkId::new("partition_point", needles.len()), |b| {
        //     b.iter(|| {
        //         let mut ret = 0;
        //         for needle in needles.iter().copied() {
        //             ret += bit_structures::utils::partition_point(n, |i| haystack[i] < needle);
        //         }
        //         ret
        //     })
        // });

        let mut workspace = VecDeque::new();
        // if we don't reserve space, then the initial
        // allocation cost will be experienced by the
        // benchmark runs, which models cold start performance.
        workspace.reserve(needles.len());

        group.bench_function(
            BenchmarkId::new("multi_partition_point", needles.len()),
            |b| {
                b.iter(|| {
                    bit_structures::utils::batch_partition_point(
                        n,
                        m,
                        |i, lo, hi| {
                            let value = haystack[i];
                            // todo: also test with power-of-two spaced bins.
                            // something vaguely like `lo + (hi - value) >> 4`
                            lo + needles[lo..hi].partition_point(|&x| x < value)
                        },
                        &mut workspace,
                    );
                    workspace.len()
                })
            },
        );

        let f = haystack.partition_point(|&x| x < needles[0]);
        let l = haystack.partition_point(|&x| x < needles[needles.len() - 1]);
        dbg!(f, l);
        let f = bit_structures::utils::partition_point(n, |i| haystack[i] < needles[0]);
        let l =
            bit_structures::utils::partition_point(n, |i| haystack[i] < needles[needles.len() - 1]);
        dbg!(f, l);

        let s = workspace.make_contiguous();
        dbg!(s.first().unwrap().0, s.last().unwrap().0);
    }
    group.finish();
}

criterion_group!(benches, bench_binary_search);
criterion_main!(benches);
