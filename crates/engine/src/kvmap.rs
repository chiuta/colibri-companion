//! Mapping cache rows from one model to another.
//!
//! A context pack ([`crate::ctxpack`]) is the cache a model wrote for a text. It is useless to a
//! different model as it is: other weights, other cache shapes. When two models share a tokenizer
//! their caches for the *same tokens* can still be related. This module learns that relation:
//! a linear map (with a mean, fitted by ridge least squares) from a source layer's rows to the
//! rows the target writes for the same tokens, separately for keys and for values.
//!
//! * Keys are compared **without their position**: they are rotated back to position 0 on the way
//!   out ([`crate::model::Model::neutral_rows`]) and rotated to the new place on the way in
//!   ([`crate::model::Model::append_neutral`]), so the two models may even use different rotary
//!   tables or head sizes.
//! * Layers are paired by relative depth (`dst layer l` ← `src layer l * src_layers / dst_layers`).
//! * A mapped row is only a **guess**. Imported rows are marked approximate and the normal repair
//!   ([`crate::ctxpack::compose_mapped`]) recomputes as many as it takes for the drift to fall under
//!   the tolerance; with tolerance 0 the result equals a full prefill, so a bad map costs time,
//!   never correctness.
//! * How good a map is can only be known by measuring on the two real models: [`fit`] reports the
//!   error on rows it did not fit (against the trivial guess of predicting the mean), and
//!   `corbul map-check` measures what the mapped cache does to the answer.

use crate::model::{Model, ModelError, NeutralRows, State};

/// Failure of a fit, a load or an import.
#[derive(Clone, Debug, PartialEq)]
pub struct KvMapError(pub String);
impl std::fmt::Display for KvMapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for KvMapError {}
impl From<ModelError> for KvMapError {
    fn from(e: ModelError) -> Self {
        KvMapError(e.to_string())
    }
}
fn err<T>(m: impl Into<String>) -> Result<T, KvMapError> {
    Err(KvMapError(m.into()))
}

/// `y = (x − mx) · W + my`.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    /// `kx × ky`, row-major.
    pub w: Vec<f32>,
    pub mx: Vec<f32>,
    pub my: Vec<f32>,
}
impl Linear {
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        let (kx, ky) = (self.mx.len(), self.my.len());
        out.copy_from_slice(&self.my);
        for i in 0..kx {
            let xi = x[i] - self.mx[i];
            if xi != 0.0 {
                for (o, w) in out.iter_mut().zip(&self.w[i * ky..(i + 1) * ky]) {
                    *o += xi * w;
                }
            }
        }
    }
}

/// How well a map fits, measured on rows that were held out of the fit.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MapStats {
    /// Rows (token positions) the fit saw in all.
    pub rows: usize,
    /// Of those, rows kept out of the first fit and used to measure it.
    pub held_out: usize,
    /// Mean over layers of `‖map(x) − y‖ / ‖y‖` for keys / values on the held-out rows.
    pub k_err: f32,
    pub v_err: f32,
    /// The same for the trivial map that always answers the mean of the fitted rows: the map is
    /// worth something only where it is clearly below this.
    pub k_base: f32,
    pub v_base: f32,
}

/// The learnt map between two models.
#[derive(Clone, Debug, PartialEq)]
pub struct KvMap {
    /// [`Model::fingerprint`] of the two models the map was fitted for.
    pub src_fp: [u8; 32],
    pub dst_fp: [u8; 32],
    pub src_layers: usize,
    pub src_kd: usize,
    pub dst_kd: usize,
    /// One map per *target* layer.
    pub k: Vec<Linear>,
    pub v: Vec<Linear>,
    pub stats: MapStats,
}

fn src_layer(l: usize, src: usize, dst: usize) -> usize {
    (l * src / dst).min(src - 1)
}

/// Running sums for one least-squares problem.
struct Gram {
    kx: usize,
    ky: usize,
    n: f64,
    sx: Vec<f64>,
    sy: Vec<f64>,
    xx: Vec<f64>,
    xy: Vec<f64>,
}
impl Gram {
    fn new(kx: usize, ky: usize) -> Gram {
        Gram {
            kx,
            ky,
            n: 0.0,
            sx: vec![0.0; kx],
            sy: vec![0.0; ky],
            xx: vec![0.0; kx * kx],
            xy: vec![0.0; kx * ky],
        }
    }
    fn add(&mut self, x: &[f32], y: &[f32]) {
        let (kx, ky) = (self.kx, self.ky);
        self.n += 1.0;
        for (s, v) in self.sy.iter_mut().zip(y) {
            *s += *v as f64;
        }
        for i in 0..kx {
            let xi = x[i] as f64;
            self.sx[i] += xi;
            if xi == 0.0 {
                continue;
            }
            for j in i..kx {
                self.xx[i * kx + j] += xi * x[j] as f64;
            }
            for (a, v) in self.xy[i * ky..(i + 1) * ky].iter_mut().zip(y) {
                *a += xi * *v as f64;
            }
        }
    }
    /// Ridge solution; the penalty is `ridge` times the mean variance of the inputs.
    fn solve(&self, ridge: f32) -> Option<Linear> {
        let (kx, ky) = (self.kx, self.ky);
        if self.n < 2.0 {
            return None;
        }
        let mx: Vec<f64> = self.sx.iter().map(|s| s / self.n).collect();
        let my: Vec<f64> = self.sy.iter().map(|s| s / self.n).collect();
        let mut a = vec![0f64; kx * kx];
        let mut trace = 0.0;
        for i in 0..kx {
            for j in i..kx {
                let c = self.xx[i * kx + j] - self.n * mx[i] * mx[j];
                a[i * kx + j] = c;
                a[j * kx + i] = c;
            }
            trace += a[i * kx + i];
        }
        let mut b = vec![0f64; kx * ky];
        for i in 0..kx {
            for j in 0..ky {
                b[i * ky + j] = self.xy[i * ky + j] - self.n * mx[i] * my[j];
            }
        }
        let lam = (ridge as f64) * trace / kx as f64 + 1e-12;
        for i in 0..kx {
            a[i * kx + i] += lam;
        }
        if !spd_solve(&mut a, kx, &mut b, ky) {
            return None;
        }
        let w: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        if w.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(Linear {
            w,
            mx: mx.iter().map(|&v| v as f32).collect(),
            my: my.iter().map(|&v| v as f32).collect(),
        })
    }
}

/// Solve `A X = B` for symmetric positive definite `A` (`n × n`, overwritten) and `B` (`n × m`,
/// overwritten with `X`) by Cholesky. `false` if `A` is not positive definite.
fn spd_solve(a: &mut [f64], n: usize, b: &mut [f64], m: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d.is_nan() || d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    for c in 0..m {
        for i in 0..n {
            let mut s = b[i * m + c];
            for k in 0..i {
                s -= a[i * n + k] * b[k * m + c];
            }
            b[i * m + c] = s / a[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = b[i * m + c];
            for k in i + 1..n {
                s -= a[k * n + i] * b[k * m + c];
            }
            b[i * m + c] = s / a[i * n + i];
        }
    }
    true
}

/// `(‖map(x) − y‖², ‖y‖², ‖mean − y‖²)` summed over the rows.
fn measure(lin: &Linear, xs: &[f32], ys: &[f32], mean: &[f32]) -> (f64, f64, f64) {
    let (kx, ky) = (lin.mx.len(), lin.my.len());
    let mut out = vec![0f32; ky];
    let (mut e, mut n, mut b) = (0.0, 0.0, 0.0);
    for (x, y) in xs.chunks(kx).zip(ys.chunks(ky)) {
        lin.apply(x, &mut out);
        for j in 0..ky {
            let (o, t) = (out[j] as f64, y[j] as f64);
            e += (o - t) * (o - t);
            n += t * t;
            let d = mean[j] as f64 - t;
            b += d * d;
        }
    }
    (e, n, b)
}

/// Rows of every layer, gathered from the calibration chunks.
struct Rows {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    kd: usize,
}
fn gather(m: &Model, chunks: &[&[usize]], rows: usize) -> Result<Rows, KvMapError> {
    let mut out = Rows {
        k: Vec::new(),
        v: Vec::new(),
        kd: 0,
    };
    for c in chunks {
        let mut st = m.new_state();
        m.prefill_cached(&mut st, c)?;
        let nr = m.neutral_rows(&st)?;
        if out.k.is_empty() {
            out.k = vec![Vec::with_capacity(rows * nr.kd); nr.k.len()];
            out.v = vec![Vec::with_capacity(rows * nr.kd); nr.v.len()];
            out.kd = nr.kd;
        }
        for (d, s) in out.k.iter_mut().zip(&nr.k) {
            d.extend_from_slice(s);
        }
        for (d, s) in out.v.iter_mut().zip(&nr.v) {
            d.extend_from_slice(s);
        }
    }
    Ok(out)
}

/// Fit the map from `src` to `dst` on `tokens`, which both models must read as the same token ids
/// (the caller checks that the tokenizers agree). Every fifth row is held out of the first fit to
/// measure it; the map that is returned is then refitted on all rows. `ridge` is the penalty
/// relative to the variance of the inputs (`1e-3` is a reasonable start; more when rows are few).
/// At most `max_rows` rows are used, which bounds memory (about `rows × kd × layers × 16` bytes).
pub fn fit(
    src: &Model,
    dst: &Model,
    tokens: &[usize],
    max_rows: usize,
    ridge: f32,
) -> Result<KvMap, KvMapError> {
    if ridge.is_nan() || ridge < 0.0 || ridge.is_infinite() {
        return err("the ridge penalty must be a finite number, zero or more");
    }
    let vocab = src.spec.vocab.min(dst.spec.vocab) as usize;
    if let Some(&t) = tokens.iter().find(|&&t| t >= vocab) {
        return err(format!(
            "token {t} is outside the vocabulary of one of the models"
        ));
    }
    let tokens = &tokens[..tokens.len().min(max_rows.max(1))];
    let step = (src.spec.context.min(dst.spec.context) as usize).clamp(1, 256);
    let chunks: Vec<&[usize]> = tokens.chunks(step).filter(|c| c.len() >= 2).collect();
    let rows: usize = chunks.iter().map(|c| c.len()).sum();
    if rows < 10 {
        return err("too few calibration tokens: give at least ten (hundreds are better)");
    }
    let (rs, rd) = (gather(src, &chunks, rows)?, gather(dst, &chunks, rows)?);
    let (ls, ld) = (rs.k.len(), rd.k.len());
    if ls == 0 || ld == 0 {
        return err("a model has no cache layers");
    }
    let held: Vec<bool> = (0..rows).map(|i| i % 5 == 4).collect();
    let n_held = held.iter().filter(|h| **h).count();
    let (kxs, kyd) = (rs.kd, rd.kd);
    // one layer: (key map, value map, [k err², k norm², k base², v err², v norm², v base²])
    let one = |l: usize| -> Result<(Linear, Linear, [f64; 6]), KvMapError> {
        let s = src_layer(l, ls, ld);
        let mut res: Vec<(Linear, [f64; 3])> = Vec::new();
        for (xs, ys) in [(&rs.k[s], &rd.k[l]), (&rs.v[s], &rd.v[l])] {
            let (mut tr, mut ho) = (Gram::new(kxs, kyd), Gram::new(kxs, kyd));
            let (mut hx, mut hy) = (Vec::new(), Vec::new());
            for i in 0..rows {
                let (x, y) = (&xs[i * kxs..(i + 1) * kxs], &ys[i * kyd..(i + 1) * kyd]);
                if held[i] {
                    ho.add(x, y);
                    hx.extend_from_slice(x);
                    hy.extend_from_slice(y);
                } else {
                    tr.add(x, y);
                }
            }
            let first = tr
                .solve(ridge)
                .ok_or_else(|| KvMapError("the fit failed: not enough varied rows".into()))?;
            let m = measure(&first, &hx, &hy, &first.my);
            // all rows: the training sums plus the held-out ones
            for (a, b) in tr.xx.iter_mut().zip(&ho.xx) {
                *a += b;
            }
            for (a, b) in tr.xy.iter_mut().zip(&ho.xy) {
                *a += b;
            }
            for (a, b) in tr.sx.iter_mut().zip(&ho.sx) {
                *a += b;
            }
            for (a, b) in tr.sy.iter_mut().zip(&ho.sy) {
                *a += b;
            }
            tr.n += ho.n;
            let all = tr
                .solve(ridge)
                .ok_or_else(|| KvMapError("the fit failed: not enough varied rows".into()))?;
            res.push((all, [m.0, m.1, m.2]));
        }
        let (v, mv) = res.pop().unwrap();
        let (k, mk) = res.pop().unwrap();
        Ok((k, v, [mk[0], mk[1], mk[2], mv[0], mv[1], mv[2]]))
    };
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(ld);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new((0..ld).map(|_| None).collect::<Vec<_>>());
    std::thread::scope(|sc| {
        for _ in 0..workers {
            sc.spawn(|| {
                loop {
                    let l = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if l >= ld {
                        break;
                    }
                    let r = one(l);
                    results.lock().unwrap()[l] = Some(r);
                }
            });
        }
    });
    let (mut k, mut v) = (Vec::new(), Vec::new());
    let mut acc = [0f64; 6];
    for r in results.into_inner().unwrap() {
        let (kl, vl, m) = r.unwrap()?;
        k.push(kl);
        v.push(vl);
        let rel = |e: f64, n: f64| if n > 0.0 { (e / n).sqrt() } else { 0.0 };
        acc[0] += rel(m[0], m[1]);
        acc[1] += rel(m[2], m[1]);
        acc[2] += rel(m[3], m[4]);
        acc[3] += rel(m[5], m[4]);
    }
    let d = ld as f64;
    Ok(KvMap {
        src_fp: src.fingerprint(),
        dst_fp: dst.fingerprint(),
        src_layers: ls,
        src_kd: kxs,
        dst_kd: kyd,
        k,
        v,
        stats: MapStats {
            rows,
            held_out: n_held,
            k_err: (acc[0] / d) as f32,
            k_base: (acc[1] / d) as f32,
            v_err: (acc[2] / d) as f32,
            v_base: (acc[3] / d) as f32,
        },
    })
}

impl KvMap {
    /// Map position-free rows of the source model to rows for the target.
    pub fn apply(&self, rows: &NeutralRows) -> Result<NeutralRows, KvMapError> {
        if rows.kd != self.src_kd
            || rows.k.len() != self.src_layers
            || rows.v.len() != self.src_layers
        {
            return err("the rows do not have the shape this map was fitted for");
        }
        let (ld, kd) = (self.k.len(), self.dst_kd);
        let go = |planes: &[Vec<f32>], maps: &[Linear]| -> Vec<Vec<f32>> {
            (0..ld)
                .map(|l| {
                    let src = &planes[src_layer(l, self.src_layers, ld)];
                    let mut out = vec![0f32; rows.n * kd];
                    for (x, o) in src.chunks(self.src_kd).zip(out.chunks_mut(kd)) {
                        maps[l].apply(x, o);
                    }
                    out
                })
                .collect()
        };
        Ok(NeutralRows {
            kd,
            n: rows.n,
            k: go(&rows.k, &self.k),
            v: go(&rows.v, &self.v),
        })
    }

    /// Append the rows of `pack` (a state of `src_model`) to `dst`, a state of `dst_model`, mapped.
    /// `dst` is approximate over them afterwards. The map must belong to this pair of models.
    pub fn import(
        &self,
        src_model: &Model,
        pack: &State,
        dst_model: &Model,
        dst: &mut State,
    ) -> Result<(), KvMapError> {
        let rows = src_model.neutral_rows(pack)?;
        let mapped = self.apply(&rows)?;
        dst_model.append_neutral(dst, pack.tokens(), &mapped)?;
        Ok(())
    }

    /// Does this map belong to these two models? (Compares their fingerprints; reads every weight.)
    pub fn matches(&self, src: &Model, dst: &Model) -> bool {
        self.src_fp == src.fingerprint() && self.dst_fp == dst.fingerprint()
    }
}

const MAGIC: &[u8; 8] = b"CMKVMAP1";

impl KvMap {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Vec::new();
        o.extend_from_slice(MAGIC);
        o.extend_from_slice(&self.src_fp);
        o.extend_from_slice(&self.dst_fp);
        for v in [
            self.src_layers,
            self.k.len(),
            self.src_kd,
            self.dst_kd,
            self.stats.rows,
            self.stats.held_out,
        ] {
            o.extend_from_slice(&(v as u32).to_le_bytes());
        }
        for v in [
            self.stats.k_err,
            self.stats.v_err,
            self.stats.k_base,
            self.stats.v_base,
        ] {
            o.extend_from_slice(&v.to_le_bytes());
        }
        for lin in self.k.iter().chain(&self.v) {
            for v in lin.w.iter().chain(&lin.mx).chain(&lin.my) {
                o.extend_from_slice(&v.to_le_bytes());
            }
        }
        o
    }

    pub fn from_bytes(b: &[u8]) -> Result<KvMap, KvMapError> {
        let head = 8 + 64 + 6 * 4 + 4 * 4;
        if b.len() < head || &b[..8] != MAGIC {
            return err("not a cache map file");
        }
        let u =
            |i: usize| u32::from_le_bytes(b[72 + i * 4..76 + i * 4].try_into().unwrap()) as usize;
        let f = |i: usize| f32::from_le_bytes(b[96 + i * 4..100 + i * 4].try_into().unwrap());
        let (ls, ld, kx, ky) = (u(0), u(1), u(2), u(3));
        let per = kx
            .checked_mul(ky)
            .and_then(|w| w.checked_add(kx + ky))
            .ok_or_else(|| KvMapError("the map header is not valid".into()))?;
        let total = per
            .checked_mul(2)
            .and_then(|p| p.checked_mul(ld))
            .and_then(|p| p.checked_mul(4))
            .and_then(|p| p.checked_add(head))
            .ok_or_else(|| KvMapError("the map header is not valid".into()))?;
        if ls == 0 || ld == 0 || kx == 0 || ky == 0 || total != b.len() {
            return err("the map file is truncated or its header is not valid");
        }
        let mut at = head;
        let mut take = |n: usize| -> Vec<f32> {
            let v = b[at..at + n * 4]
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            at += n * 4;
            v
        };
        let mut lins = Vec::new();
        for _ in 0..ld * 2 {
            lins.push(Linear {
                w: take(kx * ky),
                mx: take(kx),
                my: take(ky),
            });
        }
        if lins
            .iter()
            .any(|l| l.w.iter().chain(&l.mx).chain(&l.my).any(|v| !v.is_finite()))
        {
            return err("the map contains a value that is not finite");
        }
        let v = lins.split_off(ld);
        let mut sfp = [0u8; 32];
        let mut dfp = [0u8; 32];
        sfp.copy_from_slice(&b[8..40]);
        dfp.copy_from_slice(&b[40..72]);
        Ok(KvMap {
            src_fp: sfp,
            dst_fp: dfp,
            src_layers: ls,
            src_kd: kx,
            dst_kd: ky,
            k: lins,
            v,
            stats: MapStats {
                rows: u(4),
                held_out: u(5),
                k_err: f(0),
                v_err: f(1),
                k_base: f(2),
                v_base: f(3),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::tests::tiny;

    fn toks(seed: usize, n: usize, vocab: usize) -> Vec<usize> {
        (0..n)
            .map(|i| (seed * 31 + i * 7 + i * i + i / 3) % vocab)
            .collect()
    }

    #[test]
    fn cholesky_solves_a_known_system() {
        // A = [[4,2],[2,3]], X = [[1],[2]] → B = [[8],[8]]
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        let mut b = vec![8.0, 8.0];
        assert!(spd_solve(&mut a, 2, &mut b, 1));
        assert!((b[0] - 1.0).abs() < 1e-12 && (b[1] - 2.0).abs() < 1e-12);
        let mut bad = vec![1.0, 2.0, 2.0, 1.0];
        assert!(!spd_solve(&mut bad, 2, &mut [0.0, 0.0], 1));
    }

    #[test]
    fn neutral_rows_round_trip_into_an_identical_cache() {
        let m = tiny(4, 2, false);
        let t = toks(3, 14, m.spec.vocab as usize);
        let mut a = m.new_state();
        m.prefill_cached(&mut a, &t).unwrap();
        let rows = m.neutral_rows(&a).unwrap();
        let mut b = m.new_state();
        m.append_neutral(&mut b, &t, &rows).unwrap();
        assert!(!b.is_exact() && b.approx_slots() == 14);
        for l in 0..m.spec.layers as usize {
            for s in 0..14 {
                let (x, y) = (a.key_row(l, s).unwrap(), b.key_row(l, s).unwrap());
                assert!(
                    x.iter().zip(&y).all(|(p, q)| (p - q).abs() < 1e-4),
                    "{l} {s}"
                );
            }
        }
        // shape and token errors change nothing
        let mut c = m.new_state();
        assert!(m.append_neutral(&mut c, &t[..5], &rows).is_err());
        let mut bad = rows.clone();
        bad.k[0][0] = f32::NAN;
        assert!(m.append_neutral(&mut c, &t, &bad).is_err());
        assert!(c.tokens().is_empty() && c.is_exact());
    }

    #[test]
    fn a_model_maps_to_itself_almost_exactly() {
        let m = tiny(4, 2, false);
        let t = toks(5, 120, m.spec.vocab as usize);
        let map = fit(&m, &m, &t[..m.spec.context as usize * 3], 1000, 1e-6).unwrap();
        assert!(
            map.stats.k_err < 0.05 && map.stats.v_err < 0.05,
            "{:?}",
            map.stats
        );
        assert!(map.stats.k_err < map.stats.k_base);
        assert!(map.matches(&m, &m));
    }

    #[test]
    fn different_weights_are_mapped_and_repair_makes_the_answer_exact() {
        use crate::ctxpack::{ComposeOptions, compose_mapped};
        let (a, b) = (tiny(4, 2, false), tiny(2, 2, true));
        let v = a.spec.vocab.min(b.spec.vocab) as usize;
        let cal = toks(9, 600, v);
        let map = fit(&a, &b, &cal, 600, 1e-3).unwrap();
        assert!(!map.k.is_empty());
        let bytes = map.to_bytes();
        assert_eq!(KvMap::from_bytes(&bytes).unwrap(), map);
        let (t1, t2, q) = (toks(1, 9, v), toks(2, 7, v), toks(3, 3, v));
        let (mut p1, mut p2) = (a.new_state(), a.new_state());
        a.prefill_cached(&mut p1, &t1).unwrap();
        a.prefill_cached(&mut p2, &t2).unwrap();
        // tolerance 0: everything recomputed, so the answer equals the plain prefill
        let o = ComposeOptions {
            tolerance: 0.0,
            ..Default::default()
        };
        let (st, logits, rep) = compose_mapped(&map, &a, &[&p1, &p2], &b, &q, &o).unwrap();
        let all: Vec<usize> = [t1, t2, q].concat();
        let want = crate::model::tests::reference_logits(&b, &all);
        assert!(
            logits
                .iter()
                .zip(&want)
                .all(|(x, y)| (x - y).abs() <= 2e-3 * (1.0 + y.abs()))
        );
        assert!(st.is_exact() && rep.refreshed == 16);
    }

    #[test]
    fn files_that_are_damaged_are_refused() {
        let m = tiny(2, 2, true);
        let map = fit(&m, &m, &toks(2, 80, m.spec.vocab as usize), 80, 1e-3).unwrap();
        let good = map.to_bytes();
        assert!(KvMap::from_bytes(&good[..good.len() - 1]).is_err());
        assert!(KvMap::from_bytes(&good[..20]).is_err());
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(KvMap::from_bytes(&bad).is_err());
        let mut nan = good.clone();
        let n = nan.len();
        nan[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(KvMap::from_bytes(&nan).is_err());
        // a map of another shape cannot be applied
        let other = tiny(4, 2, false);
        let mut st = other.new_state();
        other
            .prefill_cached(&mut st, &toks(1, 6, other.spec.vocab as usize))
            .unwrap();
        let rows = other.neutral_rows(&st).unwrap();
        assert!(map.apply(&rows).is_err() || other.spec.kv_heads == m.spec.kv_heads);
        assert!(fit(&m, &m, &[1, 2, 3], 10, 1e-3).is_err());
        assert!(fit(&m, &m, &toks(1, 50, 4), 50, f32::NAN).is_err());
    }
}
