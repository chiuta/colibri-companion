//! Numeric building blocks. Plain f32, row-major. The inner loops (dot products, norms, softmax,
//! activations, rotary rotation) go through the runtime-dispatched kernels in [`crate::simd`];
//! rows of a matrix–vector product are independent, so large products are split across threads
//! without changing any result.
use crate::simd;
use std::cell::RefCell;

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = simd::sumsq(x) / x.len().max(1) as f32;
    let k = 1.0 / (ms + eps).sqrt();
    simd::mul3(x, w, k, out);
}

/// Numerically stable: subtracts the maximum first.
pub fn softmax(x: &mut [f32]) {
    let m = simd::max(x);
    if !m.is_finite() {
        let n = x.len() as f32;
        x.iter_mut().for_each(|v| *v = 1.0 / n);
        return;
    }
    let s = simd::exp_sub_sum(x, m);
    simd::scale(x, 1.0 / s);
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + simd::rmath::exp(-x))
}

/// `g = silu(g) · u` over the common length, vectorised.
pub fn silu_mul(g: &mut [f32], u: &[f32]) {
    simd::silu_mul(g, u)
}

/// `y += a · x`.
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    simd::axpy(a, x, y)
}

struct RopeScratch {
    d: usize,
    det: bool,
    theta_bits: u64,
    inv_freq: Vec<f64>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}
thread_local! {
    static ROPE: RefCell<RopeScratch> = const { RefCell::new(RopeScratch { d: 0, det: false, theta_bits: 0, inv_freq: Vec::new(), cos: Vec::new(), sin: Vec::new() }) };
}

/// Rotary embedding on one head vector, "rotate-half" pairing (i, i + d/2) as Hugging Face does.
/// Angles are computed in f64 (exact enough for any position); only the rotation is vectorised.
pub fn rope(head: &mut [f32], pos: usize, theta: f64) {
    let d = head.len();
    let half = d / 2;
    ROPE.with(|cell| {
        let mut s = cell.borrow_mut();
        let det = simd::is_reproducible();
        if s.d != d || s.det != det || s.theta_bits != theta.to_bits() {
            s.d = d;
            s.det = det;
            s.theta_bits = theta.to_bits();
            s.inv_freq = (0..half)
                .map(|i| simd::rmath::powf64(theta, -(2.0 * i as f64) / d as f64))
                .collect();
        }
        let RopeScratch {
            inv_freq, cos, sin, ..
        } = &mut *s;
        cos.clear();
        sin.clear();
        for f in inv_freq.iter() {
            let (sn, cs) = simd::rmath::sin_cos64(pos as f64 * f);
            cos.push(cs as f32);
            sin.push(sn as f32);
        }
        simd::rope_rotate(head, cos, sin);
    });
}

fn rows_into(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    for (o, row) in out.iter_mut().zip(w.chunks_exact(cols)) {
        *o = simd::dot_f32(row, x);
    }
}

/// `out = W · x` with `W` of shape `out.len() × cols`.
pub fn matvec(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    // Malformed shapes must not panic (or read out of range) in release builds: the output is zeroed.
    if cols == 0 || x.len() != cols || w.len() != out.len().saturating_mul(cols) {
        out.fill(0.0);
        return;
    }
    if crate::slicegpu::try_matvec(w, cols, x, out) {
        return;
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if w.len() < (1 << 18) || threads == 1 || out.len() < 2 * threads {
        return rows_into(w, cols, x, out);
    }
    let per = out.len().div_ceil(threads);
    std::thread::scope(|s| {
        for (o, wc) in out.chunks_mut(per).zip(w.chunks(per * cols)) {
            s.spawn(move || rows_into(wc, cols, x, o));
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn matvec_with_malformed_shapes_zeroes_instead_of_panicking() {
        let mut o = [7.0f32; 3];
        super::matvec(&[], 0, &[], &mut o);
        assert_eq!(o, [0.0; 3]);
        let mut o = [7.0f32; 2];
        super::matvec(&[1.0; 5], 3, &[1.0; 3], &mut o);
        assert_eq!(o, [0.0; 2]);
        super::matvec(&[1.0; 6], 3, &[1.0; 2], &mut o);
        assert_eq!(o, [0.0; 2]);
        super::matvec(&[1.0; 6], 3, &[1.0; 3], &mut o);
        assert_eq!(o, [3.0; 2]);
    }

    use super::*;

    #[test]
    fn softmax_sums_to_one_and_survives_huge_inputs() {
        let mut v = [1000.0, 1001.0, 999.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[1] > v[0] && v[0] > v[2]);
        let mut e = [f32::NEG_INFINITY; 4];
        softmax(&mut e);
        assert!(e.iter().all(|p| (*p - 0.25).abs() < 1e-6));
    }

    #[test]
    fn softmax_matches_a_reference_on_long_vectors() {
        for n in [1usize, 7, 8, 9, 33, 128, 1000] {
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.21)
                .collect();
            let m = x.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let z: f64 = x.iter().map(|v| (*v as f64 - m).exp()).sum();
            let mut y = x.clone();
            softmax(&mut y);
            for (a, b) in y.iter().zip(&x) {
                assert!(
                    (*a as f64 - (*b as f64 - m).exp() / z).abs() < 2e-6,
                    "n={n}"
                );
            }
        }
    }

    #[test]
    fn rmsnorm_known_value() {
        let mut o = [0.0; 2];
        rmsnorm(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut o);
        let k = 1.0 / (12.5f32).sqrt();
        assert!((o[0] - 3.0 * k).abs() < 1e-6 && (o[1] - 8.0 * k).abs() < 1e-6);
    }

    #[test]
    fn rmsnorm_epsilon_is_added_to_the_mean_square() {
        // eps = 1.5 is large enough that dropping it, or adding it after the root, is unmistakable
        let mut o = [0.0; 2];
        rmsnorm(&[3.0, 4.0], &[1.0, 2.0], 1.5, &mut o);
        let k = 1.0 / (12.5f64 + 1.5).sqrt();
        assert!(
            (o[0] as f64 - 3.0 * k).abs() < 1e-6 && (o[1] as f64 - 8.0 * k).abs() < 1e-6,
            "{o:?}"
        );
    }

    #[test]
    fn silu_mul_agrees_with_the_scalar_silu() {
        let g: Vec<f32> = (0..45).map(|i| (i as f32 - 22.0) * 0.4).collect();
        let u: Vec<f32> = (0..45).map(|i| (i as f32 * 0.3).sin()).collect();
        let mut got = g.clone();
        silu_mul(&mut got, &u);
        for i in 0..45 {
            assert!((got[i] - silu(g[i]) * u[i]).abs() < 1e-5, "i={i}");
        }
    }

    #[test]
    fn rope_is_identity_at_zero_and_preserves_length() {
        let base: Vec<f32> = (0..8).map(|i| i as f32 * 0.3 - 1.0).collect();
        let mut a = base.clone();
        rope(&mut a, 0, 10_000.0);
        assert_eq!(a, base);
        rope(&mut a, 37, 10_000.0);
        let n = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>();
        assert!((n(&a) - n(&base)).abs() < 1e-4);
    }

    #[test]
    fn rope_matches_the_f64_formula_for_several_shapes_and_thetas() {
        for (d, theta, pos) in [
            (8usize, 10_000.0f64, 3usize),
            (64, 500_000.0, 1234),
            (128, 10_000.0, 60_000),
            (6, 1e6, 9),
        ] {
            let base: Vec<f32> = (0..d).map(|i| ((i * 13 % 7) as f32 - 3.0) * 0.4).collect();
            let mut a = base.clone();
            rope(&mut a, pos, theta);
            rope(&mut a.clone(), 1, 10.0); // a different shape in between must not poison the cache
            let half = d / 2;
            for i in 0..half {
                let f = theta.powf(-(2.0 * i as f64) / d as f64);
                let (s, c) = (pos as f64 * f).sin_cos();
                let (x, y) = (base[i] as f64, base[i + half] as f64);
                assert!((a[i] as f64 - (x * c - y * s)).abs() < 2e-6, "d={d} i={i}");
                assert!(
                    (a[i + half] as f64 - (x * s + y * c)).abs() < 2e-6,
                    "d={d} i={i}"
                );
            }
        }
    }

    #[test]
    fn rope_scores_depend_only_on_relative_distance() {
        let q: Vec<f32> = (0..8).map(|i| (i as f32).cos()).collect();
        let k: Vec<f32> = (0..8).map(|i| (i as f32 * 1.7).sin()).collect();
        let score = |m: usize, n: usize| {
            let (mut a, mut b) = (q.clone(), k.clone());
            rope(&mut a, m, 10_000.0);
            rope(&mut b, n, 10_000.0);
            a.iter().zip(&b).map(|(x, y)| x * y).sum::<f32>()
        };
        assert!((score(5, 2) - score(105, 102)).abs() < 1e-3);
        assert!((score(5, 2) - score(6, 2)).abs() > 1e-4);
    }

    #[test]
    fn threaded_matvec_equals_serial() {
        let (rows, cols) = (1024, 512);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.01)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32).sin()).collect();
        let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
        matvec(&w, cols, &x, &mut a);
        rows_into(&w, cols, &x, &mut b);
        assert_eq!(a, b);
    }
}
