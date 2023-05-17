use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use wavelet_matrix::bitvector::BitVector;
use wavelet_matrix::originalrlebitvector::OriginalRLEBitVector;
use wavelet_matrix::rlebitvector::RLEBitVector;

// Number of times to call the core function (eg. rank or select) within the benchmarked function
const N_QUERIES_PER_TEST: usize = 1000;

// returns k runs that sum to n
fn build_runs(k: usize, n: usize) -> Vec<(usize, usize)> {
    // assert we want an even number of runs
    assert!(k % 2 == 0);
    let mut rng = rand::thread_rng();

    // We want to generate k runs, so we first sample k-1 random numbers in 1..n-1.
    // As an example, if k = 3 then we sample k-1 = 2 numbers in 1..n-1, taking the
    // deltas to be the runs:
    // 0 x x n
    //  ^ ^ ^ runs
    //
    // Sample k - 1 numbers in [0..n-2]
    let mut samples = rand::seq::index::sample(&mut rng, n - 1, k - 1).into_vec();
    // Shift them by 1, sort, then prepend 0 and append n, which gives us
    // k+1 numbers from 0 to n, from which we then compute deltas
    for sample in samples.iter_mut() {
        *sample += 1;
    }
    samples.sort();
    samples.insert(0, 0);
    samples.push(n);

    // Compute deltas
    let mut deltas = vec![];
    for (&prev, &cur) in samples.iter().zip(samples.iter().skip(1)) {
        let delta = cur - prev;
        deltas.push(delta);
    }

    // Assert we have an even number of deltas, each corresponding to a run
    assert!(deltas.len() % 2 == 0);

    // Convert deltas to (num_zeros, num_ones) format
    let mut v = vec![];
    for runs in deltas.chunks(2) {
        let num_zeros = runs[0];
        let num_ones = runs[1];
        v.push((num_zeros, num_ones));
    }

    assert!(deltas.iter().sum::<usize>() == n);

    v
}

#[inline]
fn build_orig_bitvector(runs: &[(usize, usize)]) -> OriginalRLEBitVector {
    let mut bb = OriginalRLEBitVector::builder();
    for (num_zeros, num_ones) in runs.iter().copied() {
        bb.run(num_zeros, num_ones);
    }
    bb.build()
}

#[inline]
fn build_new_bitvector(runs: &[(usize, usize)]) -> RLEBitVector {
    let mut bb = RLEBitVector::builder();
    for (num_zeros, num_ones) in runs.iter().copied() {
        bb.run(num_zeros, num_ones);
    }
    bb.build()
}

#[inline]
fn bench_orig_bitvector_rank(bv: &OriginalRLEBitVector) -> usize {
    let mut rng = rand::thread_rng();
    let n = bv.len();
    let mut ret = 0;
    for _ in 0..N_QUERIES_PER_TEST {
        let i = rng.gen_range(1..n);
        let r = bv.rank1(i);
        ret += r;
    }
    ret
}

#[inline]
fn bench_new_bitvector_rank(bv: &RLEBitVector) -> usize {
    let mut rng = rand::thread_rng();
    let n = bv.len();
    let mut ret = 0;
    for _ in 0..N_QUERIES_PER_TEST {
        let i = rng.gen_range(0..n);
        let r = bv.rank1(i);
        ret += r;
    }
    ret
}

#[inline]
fn bench_orig_bitvector_select1(bv: &OriginalRLEBitVector) -> usize {
    let mut rng = rand::thread_rng();
    let n = bv.num_ones();
    let mut ret = 0;
    for _ in 0..N_QUERIES_PER_TEST {
        let i = rng.gen_range(1..=n);
        let r = bv.select1(i).unwrap();
        ret += r;
    }
    ret
}

#[inline]
fn bench_new_bitvector_select1(bv: &RLEBitVector) -> usize {
    let mut rng = rand::thread_rng();
    let n = bv.num_ones();
    let mut ret = 0;
    for _ in 0..N_QUERIES_PER_TEST {
        let i = rng.gen_range(1..=n);
        let r = bv.select1(i).unwrap();
        ret += r;
    }
    ret
}

#[inline]
fn bench_orig_bitvector_select0(bv: &OriginalRLEBitVector) -> usize {
    let mut rng = rand::thread_rng();
    let n = bv.num_zeros();
    let mut ret = 0;
    for _ in 0..N_QUERIES_PER_TEST {
        let i = rng.gen_range(1..=n);
        let r = bv.select0(i).unwrap();
        ret += r;
    }
    ret
}

#[inline]
fn bench_new_bitvector_select0(bv: &RLEBitVector) -> usize {
    let mut rng = rand::thread_rng();
    let n = bv.num_zeros();
    let mut ret = 0;
    for _ in 0..N_QUERIES_PER_TEST {
        let i = rng.gen_range(1..n);
        let r = bv.select0(i).unwrap();
        ret += r;
    }
    ret
}

fn bench_bitvectors(c: &mut Criterion) {
    let num_runs = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]; // k
    let bitvector_length = 1_000_000_000; // n

    let mut group = c.benchmark_group("Rank1");
    for k in num_runs.iter().copied() {
        let runs = build_runs(k, bitvector_length);
        let bv_orig = build_orig_bitvector(&runs);
        let bv_new = build_new_bitvector(&runs);
        let mut ret = 0;
        group.bench_function(BenchmarkId::new("Orig", k), |b| {
            b.iter(|| ret += bench_orig_bitvector_rank(&bv_orig))
        });
        group.bench_function(BenchmarkId::new("New", k), |b| {
            b.iter(|| ret += bench_new_bitvector_rank(&bv_new))
        });
        assert!(ret > 0);
    }
    group.finish();

    let mut group = c.benchmark_group("Select1");
    for k in num_runs.iter().copied() {
        let runs = build_runs(k, bitvector_length);
        let bv_orig = build_orig_bitvector(&runs);
        let bv_new = build_new_bitvector(&runs);
        let mut ret = 0;
        group.bench_function(BenchmarkId::new("Orig", k), |b| {
            b.iter(|| ret += bench_orig_bitvector_select1(&bv_orig))
        });
        group.bench_function(BenchmarkId::new("New", k), |b| {
            b.iter(|| ret += bench_new_bitvector_select1(&bv_new))
        });
        assert!(ret > 0);
    }
    group.finish();

    let mut group = c.benchmark_group("Select0");
    for k in num_runs.iter().copied() {
        let runs = build_runs(k, bitvector_length);
        let bv_orig = build_orig_bitvector(&runs);
        let bv_new = build_new_bitvector(&runs);
        let mut ret = 0;
        group.bench_function(BenchmarkId::new("Orig", k), |b| {
            b.iter(|| ret += bench_orig_bitvector_select0(&bv_orig))
        });
        group.bench_function(BenchmarkId::new("New", k), |b| {
            b.iter(|| ret += bench_new_bitvector_select0(&bv_new))
        });
        assert!(ret > 0);
    }
    group.finish();
}

criterion_group!(benches, bench_bitvectors);
criterion_main!(benches);
