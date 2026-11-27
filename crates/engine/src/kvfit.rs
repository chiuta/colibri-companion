//! Fitting the values that stay when a cache is shrunk.
//!
//! Dropping rows from a cache changes what attention returns: the weight the dropped rows had is
//! shared out again among the rest, and their values are gone. For a set of *probe queries* (the
//! last tokens of the text, which stand for what will be asked about it) the full cache gives
//! `O = softmax(Q Kᵀ) V`. With only the kept rows the output is `A' V_keep`, where `A'` is the
//! softmax over the kept keys alone. Keys are left as they are (they carry the position) and the
//! kept values are moved the least that makes `A' V'` equal `O` on the probes:
//!
//! ```text
//! minimise  ‖A' V' − O‖² + λ ‖V' − V_keep‖²
//! ```
//!
//! Because `V' = V_keep` is a candidate, the fitted residual is never larger than the plain one on
//! the probes. How far that carries over to queries that were not probed depends on the model and the
//! text, and can only be measured on a real one (`corbul kvfit` does it). The solve is done in the
//! dual form, a `p × p` system for `p` probes, so it does not grow with the number of rows.

/// Outcome of one fit.
#[derive(Clone, Debug, PartialEq)]
pub struct Fit {
    /// The new values of the kept rows, `keep.len() × hd`.
    pub values: Vec<f32>,
    /// `‖A' V_keep − O‖ / ‖O‖` before the fit.
    pub error_before: f32,
    /// The same after the fit.
    pub error_after: f32,
}

fn softcap(v: f32, cap: f64) -> f32 {
    if cap > 0.0 {
        (cap as f32) * (v / cap as f32).tanh()
    } else {
        v
    }
}

/// What [`crate::model::State::refit_values`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitReport {
    /// Layers whose values were refitted (sliding-window layers and layers without probes are not).
    pub layers: usize,
    /// (layer, kv head) pairs fitted.
    pub heads: usize,
    /// Mean `error_before` over the fitted heads.
    pub error_before: f32,
    /// Mean `error_after` over the fitted heads.
    pub error_after: f32,
}

/// Softmax of `logits[..]` over the chosen indices, in `f64`.
fn softmax_over(logits: &[f32], idx: impl Iterator<Item = usize> + Clone) -> Vec<f64> {
    let m = idx
        .clone()
        .map(|i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut w: Vec<f64> = idx.map(|i| ((logits[i] - m) as f64).exp()).collect();
    let s: f64 = w.iter().sum();
    if s > 0.0 {
        w.iter_mut().for_each(|x| *x /= s);
    }
    w
}

/// Solve `M z = r` for symmetric positive definite `M` (`p × p`, row-major) and `r` (`p × cols`).
/// `None` if `M` is not positive definite.
fn solve_spd(m: &mut [f64], r: &mut [f64], p: usize, cols: usize) -> Option<()> {
    // Cholesky, in place (lower triangle)
    for j in 0..p {
        let mut d = m[j * p + j];
        for k in 0..j {
            d -= m[j * p + k] * m[j * p + k];
        }
        if d.is_nan() || d <= 0.0 {
            return None;
        }
        let d = d.sqrt();
        m[j * p + j] = d;
        for i in j + 1..p {
            let mut s = m[i * p + j];
            for k in 0..j {
                s -= m[i * p + k] * m[j * p + k];
            }
            m[i * p + j] = s / d;
        }
    }
    for c in 0..cols {
        for i in 0..p {
            let mut s = r[i * cols + c];
            for k in 0..i {
                s -= m[i * p + k] * r[k * cols + c];
            }
            r[i * cols + c] = s / m[i * p + i];
        }
        for i in (0..p).rev() {
            let mut s = r[i * cols + c];
            for k in i + 1..p {
                s -= m[k * p + i] * r[k * cols + c];
            }
            r[i * cols + c] = s / m[i * p + i];
        }
    }
    Some(())
}

/// Fit the values of the rows `keep` of one head.
///
/// `keys` and `vals` are `n × hd` (one head's rows, keys already rotated), `queries` is `p × hd`
/// (the probes, rotated for their own positions), `scale` and `cap` the attention scale and logit
/// soft cap, `ridge` how strongly to stay near the old values (relative, `0.001` to `0.1`).
/// `keep` is strictly increasing and below `n`. Returns `None` when nothing can be fitted (no
/// probes, no rows, bad sizes, non-finite numbers); the caller then keeps the old values.
#[allow(clippy::too_many_arguments)]
pub fn fit_values(
    keys: &[f32],
    vals: &[f32],
    hd: usize,
    queries: &[f32],
    keep: &[usize],
    scale: f32,
    cap: f64,
    ridge: f32,
) -> Option<Fit> {
    if hd == 0
        || !keys.len().is_multiple_of(hd)
        || vals.len() != keys.len()
        || !queries.len().is_multiple_of(hd)
    {
        return None;
    }
    let (n, p, m) = (keys.len() / hd, queries.len() / hd, keep.len());
    if p == 0 || m == 0 || m > n || keep.windows(2).any(|w| w[0] >= w[1]) || keep[m - 1] >= n {
        return None;
    }
    if ridge.is_nan() || ridge < 0.0 || ridge.is_infinite() {
        return None;
    }
    // full attention output O (p × hd) and the kept-only weights A' (p × m)
    let mut o = vec![0f64; p * hd];
    let mut a = vec![0f64; p * m];
    let mut logits = vec![0f32; n];
    for (qi, q) in queries.chunks(hd).enumerate() {
        for (i, l) in logits.iter_mut().enumerate() {
            let k = &keys[i * hd..(i + 1) * hd];
            *l = softcap(
                q.iter().zip(k).map(|(x, y)| x * y).sum::<f32>() * scale,
                cap,
            );
        }
        if logits.iter().any(|l| !l.is_finite()) {
            return None;
        }
        let w = softmax_over(&logits, 0..n);
        for (i, wi) in w.iter().enumerate() {
            for d in 0..hd {
                o[qi * hd + d] += wi * vals[i * hd + d] as f64;
            }
        }
        let wk = softmax_over(&logits, keep.iter().copied());
        a[qi * m..(qi + 1) * m].copy_from_slice(&wk);
    }
    // residual R = O − A' V_keep
    let mut r = o.clone();
    for qi in 0..p {
        for (j, &row) in keep.iter().enumerate() {
            let w = a[qi * m + j];
            for d in 0..hd {
                r[qi * hd + d] -= w * vals[row * hd + d] as f64;
            }
        }
    }
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let on = norm(&o).max(1e-30);
    let error_before = (norm(&r) / on) as f32;
    // G = A' A'ᵀ + λ I  (p × p), solve G Z = R, then V' = V_keep + A'ᵀ Z
    let mut g = vec![0f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let s: f64 = (0..m).map(|k| a[i * m + k] * a[j * m + k]).sum();
            g[i * p + j] = s;
            g[j * p + i] = s;
        }
    }
    let tr = (0..p).map(|i| g[i * p + i]).sum::<f64>() / p as f64;
    let lam = ridge as f64 * tr + 1e-12;
    for i in 0..p {
        g[i * p + i] += lam;
    }
    let mut z = r.clone();
    solve_spd(&mut g, &mut z, p, hd)?;
    let mut out = vec![0f32; m * hd];
    for (j, &row) in keep.iter().enumerate() {
        for d in 0..hd {
            let mut v = vals[row * hd + d] as f64;
            for qi in 0..p {
                v += a[qi * m + j] * z[qi * hd + d];
            }
            out[j * hd + d] = v as f32;
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // error after
    let mut r2 = o;
    for qi in 0..p {
        for j in 0..m {
            let w = a[qi * m + j];
            for d in 0..hd {
                r2[qi * hd + d] -= w * out[j * hd + d] as f64;
            }
        }
    }
    Some(Fit {
        values: out,
        error_before,
        error_after: (norm(&r2) / on) as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn data(n: usize, hd: usize, p: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut s = seed;
        let k: Vec<f32> = (0..n * hd).map(|_| lcg(&mut s) * 2.0).collect();
        let v: Vec<f32> = (0..n * hd).map(|_| lcg(&mut s)).collect();
        // probes that resemble some keys, so attention is not flat
        let q: Vec<f32> = (0..p * hd)
            .map(|i| k[((i / hd) * 3 % n) * hd + i % hd] + 0.3 * lcg(&mut s))
            .collect();
        (k, v, q)
    }

    #[test]
    fn keeping_everything_changes_nothing() {
        let (k, v, q) = data(20, 8, 6, 1);
        let keep: Vec<usize> = (0..20).collect();
        let f = fit_values(&k, &v, 8, &q, &keep, 0.35, 0.0, 0.01).unwrap();
        assert!(f.error_before < 1e-6 && f.error_after < 1e-6);
        assert!(f.values.iter().zip(&v).all(|(a, b)| (a - b).abs() < 1e-4));
    }

    #[test]
    fn the_fit_is_never_worse_than_the_plain_values_on_the_probes() {
        for seed in 1..12u64 {
            let (n, hd, p) = (60, 8, 10);
            let (k, v, q) = data(n, hd, p, seed);
            let keep: Vec<usize> = (0..n).filter(|i| i % 3 == 0).collect();
            let f = fit_values(&k, &v, hd, &q, &keep, 0.35, 0.0, 0.01).unwrap();
            assert!(f.error_after <= f.error_before + 1e-6, "seed {seed}: {f:?}");
        }
        // and it helps clearly where there is something to repair
        let (k, v, q) = data(80, 8, 12, 5);
        let keep: Vec<usize> = (0..80).filter(|i| i % 4 == 0).collect();
        let f = fit_values(&k, &v, 8, &q, &keep, 0.35, 0.0, 0.001).unwrap();
        assert!(f.error_after < 0.5 * f.error_before, "{f:?}");
    }

    #[test]
    fn a_stronger_ridge_stays_closer_to_the_old_values() {
        let (k, v, q) = data(50, 8, 8, 9);
        let keep: Vec<usize> = (0..50).step_by(2).collect();
        let dist = |ridge: f32| {
            let f = fit_values(&k, &v, 8, &q, &keep, 0.35, 0.0, ridge).unwrap();
            keep.iter()
                .enumerate()
                .map(|(j, &r)| {
                    (0..8)
                        .map(|d| (f.values[j * 8 + d] - v[r * 8 + d]).powi(2))
                        .sum::<f32>()
                })
                .sum::<f32>()
        };
        assert!(dist(10.0) < dist(0.001));
    }

    #[test]
    fn soft_cap_and_hostile_input_are_handled() {
        let (k, v, q) = data(30, 4, 5, 3);
        let keep: Vec<usize> = (0..30).step_by(3).collect();
        assert!(fit_values(&k, &v, 4, &q, &keep, 0.5, 30.0, 0.01).is_some());
        // no probes, no rows, unsorted, out of range, bad shapes, NaN, infinite ridge
        assert!(fit_values(&k, &v, 4, &[], &keep, 0.5, 0.0, 0.01).is_none());
        assert!(fit_values(&k, &v, 4, &q, &[], 0.5, 0.0, 0.01).is_none());
        assert!(fit_values(&k, &v, 4, &q, &[3, 2], 0.5, 0.0, 0.01).is_none());
        assert!(fit_values(&k, &v, 4, &q, &[99], 0.5, 0.0, 0.01).is_none());
        assert!(fit_values(&k, &v[1..], 4, &q, &keep, 0.5, 0.0, 0.01).is_none());
        assert!(fit_values(&k, &v, 0, &q, &keep, 0.5, 0.0, 0.01).is_none());
        assert!(fit_values(&k, &v, 4, &q, &keep, 0.5, 0.0, f32::NAN).is_none());
        assert!(fit_values(&k, &v, 4, &q, &keep, 0.5, 0.0, f32::INFINITY).is_none());
        let mut bad = q.clone();
        bad[2] = f32::NAN;
        assert!(fit_values(&k, &v, 4, &bad, &keep, 0.5, 0.0, 0.01).is_none());
    }
}
