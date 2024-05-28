use criterion::{criterion_group, criterion_main, Criterion};

use fastsim_core::utils::interp::*;
use ndarray::prelude::*;
use rand::{self, rngs::StdRng, Rng, SeedableRng};

/// 0-D interpolation (hardcoded)
fn benchmark_0D() {
    let interp_0d = Interpolator::Interp0D(0.5);
    interp_0d.validate().unwrap();
    interp_0d.interpolate(&[], &Strategy::None).unwrap();
}

/// 0-D interpolation (multilinear interpolator)
fn benchmark_0D_multi() {
    let interp_0d_multi = Interpolator::InterpND(InterpND {
        grid: vec![vec![]],
        values: array![0.5].into_dyn(),
    });
    interp_0d_multi.validate().unwrap();
    interp_0d_multi.interpolate(&[], &Strategy::None).unwrap();
}

/// 1-D interpolation (hardcoded)
fn benchmark_1D() {
    let seed = 1234567890;
    let mut rng = StdRng::seed_from_u64(seed);
    let grid_data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    // Generate interpolator data (same as N-D benchmark)
    let values_data: Vec<f64> = (0..100).map(|_| rng.gen::<f64>()).collect();
    // Create a 1-D interpolator with 100 data points
    let interp_1d = Interpolator::Interp1D(Interp1D {
        x: grid_data,
        f_x: values_data,
    });
    interp_1d.validate().unwrap();
    // Sample 1,000 points
    let points: Vec<f64> = (0..1_000).map(|_| rng.gen::<f64>() * 99.).collect();
    for point in points {
        interp_1d.interpolate(&[point], &Strategy::Linear).unwrap();
    }
}

/// 1-D interpolation (multilinear interpolator)
fn benchmark_1D_multi() {
    let seed = 1234567890;
    let mut rng = StdRng::seed_from_u64(seed);
    // Generate interpolator data (same as hardcoded benchmark)
    let grid_data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let values_data: Vec<f64> = (0..100).map(|_| rng.gen::<f64>()).collect();
    // Create an N-D interpolator with 100x100 data (10,000 points)
    let interp_1d_multi = Interpolator::InterpND(InterpND {
        grid: vec![grid_data],
        values: ArrayD::from_shape_vec(IxDyn(&[100]), values_data).unwrap(),
    });
    interp_1d_multi.validate().unwrap();
    // Sample 1,000 points
    let points: Vec<f64> = (0..1_000).map(|_| rng.gen::<f64>() * 99.).collect();
    for point in points {
        interp_1d_multi
            .interpolate(&[point], &Strategy::Linear)
            .unwrap();
    }
}

/// 2-D interpolation (hardcoded)
fn benchmark_2D() {
    let seed = 1234567890;
    let mut rng = StdRng::seed_from_u64(seed);
    let grid_data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    // Generate interpolator data (same as N-D benchmark) and arrange into `Vec<Vec<f64>>`
    let values_data: Vec<f64> = (0..10_000).map(|_| rng.gen::<f64>()).collect();
    let values_data: Vec<Vec<f64>> = (0..100)
        .map(|x| values_data[(100 * x)..(100 + 100 * x)].into())
        .collect();
    // Create a 2-D interpolator with 100x100 data (10,000 points)
    let interp_2d = Interpolator::Interp2D(Interp2D {
        x: grid_data.clone(),
        y: grid_data.clone(),
        f_xy: values_data,
    });
    interp_2d.validate().unwrap();
    // Sample 1,000 points
    let points: Vec<Vec<f64>> = (0..1_000)
        .map(|_| vec![rng.gen::<f64>() * 99., rng.gen::<f64>() * 99.])
        .collect();
    for point in points {
        interp_2d.interpolate(&point, &Strategy::Linear).unwrap();
    }
}

/// 2-D interpolation (multilinear interpolator)
fn benchmark_2D_multi() {
    let seed = 1234567890;
    let mut rng = StdRng::seed_from_u64(seed);
    // Generate interpolator data (same as hardcoded benchmark)
    let grid_data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let values_data: Vec<f64> = (0..10_000).map(|_| rng.gen::<f64>()).collect();
    // Create an N-D interpolator with 100x100 data (10,000 points)
    let interp_2d_multi = Interpolator::InterpND(InterpND {
        grid: vec![grid_data.clone(), grid_data.clone()],
        values: ArrayD::from_shape_vec(IxDyn(&[100, 100]), values_data).unwrap(),
    });
    interp_2d_multi.validate().unwrap();
    // Sample 1,000 points
    let points: Vec<Vec<f64>> = (0..1_000)
        .map(|_| vec![rng.gen::<f64>() * 99., rng.gen::<f64>() * 99.])
        .collect();
    for point in points {
        interp_2d_multi
            .interpolate(&point, &Strategy::Linear)
            .unwrap();
    }
}

/// 3-D interpolation (hardcoded)
fn benchmark_3D() {
    let seed = 1234567890;
    let mut rng = StdRng::seed_from_u64(seed);
    let grid_data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    // Generate interpolator data (same as N-D benchmark) and arrange into `Vec<Vec<Vec<f64>>>`
    let values_data: Vec<f64> = (0..1_000_000).map(|_| rng.gen::<f64>()).collect();
    let values_data: Vec<Vec<Vec<f64>>> = (0..100)
        .map(|x| {
            (0..100)
                .map(|y| values_data[(100 * (y + 100 * x))..(100 + 100 * (y + 100 * x))].into())
                .collect()
        })
        .collect();
    // Create a 3-D interpolator with 100x100x100 data (1,000,000 points)
    let interp_3d = Interpolator::Interp3D(Interp3D {
        x: grid_data.clone(),
        y: grid_data.clone(),
        z: grid_data.clone(),
        f_xyz: values_data,
    });
    interp_3d.validate().unwrap();
    // Sample 1,000 points
    let points: Vec<Vec<f64>> = (0..1_000)
        .map(|_| {
            vec![
                rng.gen::<f64>() * 99.,
                rng.gen::<f64>() * 99.,
                rng.gen::<f64>() * 99.,
            ]
        })
        .collect();
    for point in points {
        interp_3d.interpolate(&point, &Strategy::Linear).unwrap();
    }
}

/// 3-D interpolation (multilinear interpolator)
fn benchmark_3D_multi() {
    let seed = 1234567890;
    let mut rng = StdRng::seed_from_u64(seed);
    // Generate interpolator data (same as hardcoded benchmark)
    let grid_data: Vec<f64> = (0..100).map(|x| x as f64).collect();
    let values_data: Vec<f64> = (0..1_000_000).map(|_| rng.gen::<f64>()).collect();
    // Create an N-D interpolator with 100x100x100 data (1,000,000 points)
    let interp_3d_multi = Interpolator::InterpND(InterpND {
        grid: vec![grid_data.clone(), grid_data.clone(), grid_data.clone()],
        values: ArrayD::from_shape_vec(IxDyn(&[100, 100, 100]), values_data).unwrap(),
    });
    interp_3d_multi.validate().unwrap();
    // Sample 1,000 points
    let points: Vec<Vec<f64>> = (0..1_000)
        .map(|_| {
            vec![
                rng.gen::<f64>() * 99.,
                rng.gen::<f64>() * 99.,
                rng.gen::<f64>() * 99.,
            ]
        })
        .collect();
    for point in points {
        interp_3d_multi
            .interpolate(&point, &Strategy::Linear)
            .unwrap();
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("0-D hardcoded", |b| b.iter(|| benchmark_0D()));
    c.bench_function("0-D multilinear", |b| b.iter(|| benchmark_0D_multi()));
    c.bench_function("1-D hardcoded", |b| b.iter(|| benchmark_1D()));
    c.bench_function("1-D multilinear", |b| b.iter(|| benchmark_1D_multi()));
    c.bench_function("2-D hardcoded", |b| b.iter(|| benchmark_2D()));
    c.bench_function("2-D multilinear", |b| b.iter(|| benchmark_2D_multi()));
    c.bench_function("3-D hardcoded", |b| b.iter(|| benchmark_3D()));
    c.bench_function("3-D multilinear", |b| b.iter(|| benchmark_3D_multi()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
