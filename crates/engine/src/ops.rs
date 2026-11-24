//! Numeric building blocks. Plain f32, row-major, no unsafe; rows of a matrix–vector product
//! are independent, so large products are split across threads without changing any result.

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32;
    let k = 1.0 / (ms + eps).sqrt();
    for ((o, v), g) in out.iter_mut().zip(x).zip(w) { *o = v * k * g; }
}

/// Numerically stable: subtracts the maximum first.
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() { let n = x.len() as f32; x.iter_mut().for_each(|v| *v = 1.0 / n); return; }
    let mut s = 0.0;
    for v in x.iter_mut() { *v = (*v - m).exp(); s += *v; }
    for v in x.iter_mut() { *v /= s; }
}

pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Rotary embedding on one head vector, "rotate-half" pairing (i, i + d/2) as Hugging Face does.
pub fn rope(head: &mut [f32], pos: usize, theta: f64) {
    let d = head.len();
    let half = d / 2;
    for i in 0..half {
        let freq = theta.powf(-(2.0 * i as f64) / d as f64);
        let (s, c) = (pos as f64 * freq).sin_cos();
        let (a, b) = (head[i] as f64, head[i + half] as f64);
        head[i] = (a * c - b * s) as f32;
        head[i + half] = (a * s + b * c) as f32;
    }
}

fn rows_into(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    for (o, row) in out.iter_mut().zip(w.chunks_exact(cols)) {
        *o = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// `out = W · x` with `W` of shape `out.len() × cols`.
pub fn matvec(w: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), out.len() * cols);
    debug_assert_eq!(x.len(), cols);
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
    fn rmsnorm_known_value() {
        let mut o = [0.0; 2];
        rmsnorm(&[3.0, 4.0], &[1.0, 2.0], 0.0, &mut o);
        let k = 1.0 / (12.5f32).sqrt();
        assert!((o[0] - 3.0 * k).abs() < 1e-6 && (o[1] - 8.0 * k).abs() < 1e-6);
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
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.01).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32).sin()).collect();
        let (mut a, mut b) = (vec![0.0; rows], vec![0.0; rows]);
        matvec(&w, cols, &x, &mut a);
        rows_into(&w, cols, &x, &mut b);
        assert_eq!(a, b);
    }
}
