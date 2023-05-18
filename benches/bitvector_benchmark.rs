use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::ThreadRng;
use rand::Rng;
use rayon::prelude::*;
use wavelet_matrix::bitvector::BitVector;
use wavelet_matrix::originalrlebitvector::OriginalRLEBitVector;
use wavelet_matrix::rlebitvector::RLEBitVector;

// returns k runs that sum to n
fn build_runs(k: usize, n: usize) -> Vec<(usize, usize)> {
    // assert we want an even number of runs
    assert!(k % 2 == 0);
    let mut rng = rand::thread_rng();

    // We want to generate k runs, so we first sample k-1 random numbers in 1..n-1.
    // As an example, if k = 3 then we sample k-1 = 2 numbers in 1..n-1, taking the

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
fn bench_orig_bitvector_rank(rng: &mut ThreadRng, bv: &OriginalRLEBitVector) -> usize {
    bv.rank1(rng.gen_range(1..bv.len()))
}

#[inline]
fn bench_new_bitvector_rank(rng: &mut ThreadRng, bv: &RLEBitVector) -> usize {
    bv.rank1(rng.gen_range(1..bv.len()))
}

#[inline]
fn bench_orig_bitvector_select1(rng: &mut ThreadRng, bv: &OriginalRLEBitVector) -> usize {
    let n = bv.num_ones();
    let i = rng.gen_range(1..=n);
    bv.select1(i).unwrap()
}

#[inline]
fn bench_new_bitvector_select1(rng: &mut ThreadRng, bv: &RLEBitVector) -> usize {
    let n = bv.num_ones();
    let i = rng.gen_range(1..=n);
    bv.select1(i).unwrap()
}

#[inline]
fn bench_orig_bitvector_select0(rng: &mut ThreadRng, bv: &OriginalRLEBitVector) -> usize {
    let n = bv.num_zeros();
    let i = rng.gen_range(1..=n);
    bv.select0(i).unwrap()
}

#[inline]
fn bench_new_bitvector_select0(rng: &mut ThreadRng, bv: &RLEBitVector) -> usize {
    let n = bv.num_zeros();
    let i = rng.gen_range(1..n);
    bv.select0(i).unwrap()
}

fn bench_bitvectors(c: &mut Criterion) {
    let num_runs = vec![
        1_000_000,     //
        5_000_000,     //
        10_000_000,    //
        50_000_000,    //
        100_000_000,   //
        500_000_000,   //
        1_000_000_000, //
    ]; // k
    let bitvector_length = 4_000_000_000usize; // n

    let runs: Vec<Vec<(usize, usize)>> = num_runs
        .par_iter()
        .map(|&k| {
            return build_runs(k, bitvector_length);
            // use std::fs::File;
            // let fname = format!("runs_{}.bincode", k);
            // let config = bincode::config::standard();
            // let mut f = File::open(fname).expect("Unable to open file");
            // let v: Vec<(usize, usize)> =
            //     bincode::decode_from_std_read(&mut f, config).expect("Unable to decode file");
            // v
            // let ret = build_runs(k, bitvector_length);
            // let f = File::create(fname).expect("Unable to create file");
            // let mut f = BufWriter::new(f);
            // let write_ret =
            // bincode::encode_into_std_write(ret, &mut f, config);
            // dbg!(write_ret);
            // vec![]
        })
        .collect();

    let mut rng = rand::thread_rng();

    let mut group = c.benchmark_group("Orig");
    for (k, runs) in num_runs.iter().copied().zip(runs.iter()) {
        let bv = build_orig_bitvector(runs);
        let mut ret = 0;
        group.bench_function(BenchmarkId::new("Rank1", k), |b| {
            b.iter(|| ret += bench_orig_bitvector_rank(&mut rng, &bv))
        });
        group.bench_function(BenchmarkId::new("Select1", k), |b| {
            b.iter(|| ret += bench_orig_bitvector_select1(&mut rng, &bv))
        });
        group.bench_function(BenchmarkId::new("Select0", k), |b| {
            b.iter(|| ret += bench_orig_bitvector_select0(&mut rng, &bv))
        });
        assert!(ret > 0);
    }
    group.finish();

    let mut group = c.benchmark_group("New");
    for (k, runs) in num_runs.iter().copied().zip(runs.iter()) {
        let bv = build_new_bitvector(runs);
        let mut ret = 0;
        group.bench_function(BenchmarkId::new("Rank1", k), |b| {
            b.iter(|| ret += bench_new_bitvector_rank(&mut rng, &bv))
        });
        group.bench_function(BenchmarkId::new("Select1", k), |b| {
            b.iter(|| ret += bench_new_bitvector_select1(&mut rng, &bv))
        });
        group.bench_function(BenchmarkId::new("Select0", k), |b| {
            b.iter(|| ret += bench_new_bitvector_select0(&mut rng, &bv))
        });
        assert!(ret > 0);
    }
    group.finish();
}

criterion_group!(benches, bench_bitvectors);
criterion_main!(benches);
