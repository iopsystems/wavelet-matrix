use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use wavelet_matrix::bitvector::BitVector;
use wavelet_matrix::originalrlebitvector::OriginalRLEBitVector;
use wavelet_matrix::rlebitvector::RLEBitVector;

// Number of times to call the core function (eg. rank or select) within the benchmarked function
const N_QUERIES_PER_TEST: usize = 1000;

#[inline]
fn build_orig_bitvector(n_runs: usize) -> OriginalRLEBitVector {
    let mut rng = rand::thread_rng();
    let mut bb = OriginalRLEBitVector::builder();
    for _ in 1..=n_runs {
        let num_zeros = rng.gen_range(1..100);
        let num_ones = rng.gen_range(1..100);
        bb.run(num_zeros, num_ones);
    }
    bb.build()
}

#[inline]
fn build_new_bitvector(n_runs: usize) -> RLEBitVector {
    let mut rng = rand::thread_rng();
    let mut bb = RLEBitVector::builder();
    for _ in 1..=n_runs {
        let num_zeros = rng.gen_range(1..100);
        let num_ones = rng.gen_range(1..100);
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
    let ns = [10_000, 100_000, 1_000_000];

    let mut group = c.benchmark_group("Rank1");
    for n_runs in ns.iter() {
        let bv_orig = build_orig_bitvector(*n_runs as usize);
        let bv_new = build_new_bitvector(*n_runs as usize);
        let mut ret = 0;
        group.bench_with_input(BenchmarkId::new("Orig", n_runs), n_runs, |b, _n_runs| {
            b.iter(|| ret += bench_orig_bitvector_rank(&bv_orig))
        });
        group.bench_with_input(BenchmarkId::new("New", n_runs), n_runs, |b, _n_runs| {
            b.iter(|| ret += bench_new_bitvector_rank(&bv_new))
        });
        assert!(ret > 0);
    }
    group.finish();

    let mut group = c.benchmark_group("Select1");
    for n_runs in ns.iter() {
        let bv_orig = build_orig_bitvector(*n_runs as usize);
        let bv_new = build_new_bitvector(*n_runs as usize);
        let mut ret = 0;
        group.bench_with_input(BenchmarkId::new("Orig", n_runs), n_runs, |b, _n_runs| {
            b.iter(|| ret += bench_orig_bitvector_select1(&bv_orig))
        });
        group.bench_with_input(BenchmarkId::new("New", n_runs), n_runs, |b, _n_runs| {
            b.iter(|| ret += bench_new_bitvector_select1(&bv_new))
        });
        assert!(ret > 0);
    }
    group.finish();

    let mut group = c.benchmark_group("Select0");
    for n_runs in ns.iter() {
        let bv_orig = build_orig_bitvector(*n_runs as usize);
        let bv_new = build_new_bitvector(*n_runs as usize);
        let mut ret = 0;
        group.bench_with_input(BenchmarkId::new("Orig", n_runs), n_runs, |b, _n_runs| {
            b.iter(|| ret += bench_orig_bitvector_select0(&bv_orig))
        });
        group.bench_with_input(BenchmarkId::new("New", n_runs), n_runs, |b, _n_runs| {
            b.iter(|| ret += bench_new_bitvector_select0(&bv_new))
        });
        assert!(ret > 0);
    }
    group.finish();
}

criterion_group!(benches, bench_bitvectors);
criterion_main!(benches);
