//! Quantised matrix–vector product that never materialises the matrix.
//!
//! * `qmatvec` (fast): the activation vector is quantised once to 8 bits (blocks of 32 for the
//!   legacy formats, 256 for K-quants) and every weight block is unpacked into a fixed stack
//!   buffer, then multiplied with integer arithmetic (i8 × i8 → i32) that LLVM auto-vectorises.
//! * `qmatvec_exact`: weights are decoded block by block to f32 and multiplied with the
//!   unquantised activations (slower, no activation rounding).
//!
//! Rows are independent, so both are split across threads by row chunks with identical results
//! to the serial path. The split runs on the persistent [`crate::pool`] threads, so a product never
//! spawns a thread; `threads` only caps how many chunks the rows are cut into.
use crate::quant::{DType, QuantError, dequantize, f16_to_f32, q3k_scales, scale_min_k4};
use std::borrow::Cow;

/// Weights below this size are not worth the cost of splitting across threads.
const THREAD_MIN_BYTES: usize = 1 << 20;

fn rd16(b: &[u8], o: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]]))
}

fn is_float(t: DType) -> bool {
    matches!(t, DType::F32 | DType::F16 | DType::BF16)
}
/// K-quants (and Q8_K) use 256-element super-blocks; the legacy formats use 32.
fn act_block(t: DType) -> usize {
    if t.geometry().is_some_and(|g| g.0 == 256) {
        256
    } else {
        32
    }
}

/// Check sizes; returns bytes per row.
fn validate(t: DType, w: usize, rows: usize, cols: usize) -> Result<usize, QuantError> {
    if !t.decodable() {
        return Err(QuantError::Unsupported(t));
    }
    let rb = t.bytes_for(cols as u64).ok_or(QuantError::Unsupported(t))? as usize;
    let need = rows.checked_mul(rb).ok_or(QuantError::Unsupported(t))?;
    if w < need {
        return Err(QuantError::BadLength { have: w, need });
    }
    Ok(rb)
}

fn check_io(rows: usize, cols: usize, x: &[f32], out: &[f32]) -> Result<(), QuantError> {
    if out.len() != rows {
        return Err(QuantError::BadLength {
            have: out.len(),
            need: rows,
        });
    }
    if x.len() != cols {
        return Err(QuantError::BadLength {
            have: x.len(),
            need: cols,
        });
    }
    Ok(())
}

// ------------------------------------------------------------ activation quantisation

struct Act {
    /// Elements per block.
    e: usize,
    /// One f32 scale per block.
    d: Vec<f32>,
    q: Vec<i8>,
    /// Sum of the quants over every 16 elements (used for the "minus offset" term).
    bs: Vec<i32>,
}

fn quantise_act(x: &[f32], e: usize) -> Act {
    let nb = x.len() / e;
    let (mut d, mut q, mut bs) = (
        Vec::with_capacity(nb),
        vec![0i8; x.len()],
        vec![0i32; x.len() / 16],
    );
    for (bi, blk) in x.chunks_exact(e).enumerate() {
        let amax = blk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let dd = amax / 127.0;
        let inv = if dd > 0.0 { 1.0 / dd } else { 0.0 };
        for (i, v) in blk.iter().enumerate() {
            q[bi * e + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
        d.push(dd);
    }
    for (g, c) in q.chunks_exact(16).enumerate() {
        bs[g] = c.iter().map(|&v| v as i32).sum();
    }
    Act { e, d, q, bs }
}

// ----------------------------------------------------------------- weight unpacking

/// One weight block as small integers plus per-group multipliers:
/// `w[i] = d * sc[g] * q[i] - dmin * mn[g]`, with `g = i / gs`.
struct Unp {
    q: [i8; 256],
    gs: usize,
    sc: [i32; 16],
    mn: [i32; 16],
    d: f32,
    dmin: f32,
}

impl Unp {
    fn new() -> Unp {
        Unp {
            q: [0; 256],
            gs: 32,
            sc: [0; 16],
            mn: [0; 16],
            d: 0.0,
            dmin: 0.0,
        }
    }

    fn simple(&mut self, d: f32) {
        self.gs = 32;
        self.sc[0] = 1;
        self.mn[0] = 0;
        self.d = d;
        self.dmin = 0.0;
    }

    fn unpack(&mut self, t: DType, b: &[u8]) {
        match t {
            DType::Q8_0 => {
                self.simple(rd16(b, 0));
                for i in 0..32 {
                    self.q[i] = b[2 + i] as i8;
                }
            }
            DType::Q4_0 => {
                self.simple(rd16(b, 0));
                for j in 0..16 {
                    self.q[j] = (b[2 + j] & 0xf) as i8 - 8;
                    self.q[j + 16] = (b[2 + j] >> 4) as i8 - 8;
                }
            }
            DType::IQ4NL => {
                self.simple(rd16(b, 0));
                for j in 0..16 {
                    self.q[j] = crate::quant::KVALUES_IQ4NL[(b[2 + j] & 0xf) as usize];
                    self.q[j + 16] = crate::quant::KVALUES_IQ4NL[(b[2 + j] >> 4) as usize];
                }
            }
            DType::IQ4XS => {
                self.gs = 32;
                self.d = rd16(b, 0);
                self.dmin = 0.0;
                let sh = u16::from_le_bytes([b[2], b[3]]);
                for ib in 0..8 {
                    self.sc[ib] = crate::quant::iq4xs_scale(b, sh, ib);
                    self.mn[ib] = 0;
                    for j in 0..16 {
                        let v = b[8 + 16 * ib + j];
                        self.q[32 * ib + j] = crate::quant::KVALUES_IQ4NL[(v & 0xf) as usize];
                        self.q[32 * ib + j + 16] = crate::quant::KVALUES_IQ4NL[(v >> 4) as usize];
                    }
                }
            }
            DType::Q4_1 => {
                self.simple(rd16(b, 0));
                self.mn[0] = 1;
                self.dmin = -rd16(b, 2);
                for j in 0..16 {
                    self.q[j] = (b[4 + j] & 0xf) as i8;
                    self.q[j + 16] = (b[4 + j] >> 4) as i8;
                }
            }
            DType::Q5_0 => {
                self.simple(rd16(b, 0));
                let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                for j in 0..16 {
                    self.q[j] = ((b[6 + j] & 0xf) | ((((qh >> j) & 1) as u8) << 4)) as i8 - 16;
                    self.q[j + 16] =
                        ((b[6 + j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)) as i8 - 16;
                }
            }
            DType::Q5_1 => {
                self.simple(rd16(b, 0));
                self.mn[0] = 1;
                self.dmin = -rd16(b, 2);
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                for j in 0..16 {
                    self.q[j] = ((b[8 + j] & 0xf) | ((((qh >> j) & 1) as u8) << 4)) as i8;
                    self.q[j + 16] =
                        ((b[8 + j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)) as i8;
                }
            }
            DType::Q8K => {
                self.gs = 256;
                self.sc[0] = 1;
                self.mn[0] = 0;
                self.dmin = 0.0;
                self.d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                for i in 0..256 {
                    self.q[i] = b[4 + i] as i8;
                }
            }
            DType::Q2K => {
                self.gs = 16;
                self.d = rd16(b, 80);
                self.dmin = rd16(b, 82);
                for (g, &v) in b[..16].iter().enumerate() {
                    self.sc[g] = (v & 0xf) as i32;
                    self.mn[g] = (v >> 4) as i32;
                }
                for n in 0..2 {
                    for j in 0..4 {
                        for half in 0..2 {
                            for l in 0..16 {
                                self.q[n * 128 + j * 32 + half * 16 + l] =
                                    ((b[16 + n * 32 + half * 16 + l] >> (2 * j)) & 3) as i8;
                            }
                        }
                    }
                }
            }
            DType::Q3K => {
                self.gs = 16;
                self.d = rd16(b, 108);
                self.dmin = 0.0;
                self.sc = q3k_scales(&b[96..108]);
                self.mn = [0; 16];
                for n in 0..2 {
                    for j in 0..4 {
                        for half in 0..2 {
                            for l in 0..16 {
                                let low = ((b[32 + n * 32 + half * 16 + l] >> (2 * j)) & 3) as i8;
                                let hi = b[half * 16 + l] & (1 << (n * 4 + j)) != 0;
                                self.q[n * 128 + j * 32 + half * 16 + l] =
                                    if hi { low } else { low - 4 };
                            }
                        }
                    }
                }
            }
            DType::Q4K => {
                self.gs = 32;
                self.d = rd16(b, 0);
                self.dmin = rd16(b, 2);
                for g in 0..8 {
                    let (s, m) = scale_min_k4(g, &b[4..16]);
                    self.sc[g] = s as i32;
                    self.mn[g] = m as i32;
                }
                for c in 0..4 {
                    for l in 0..32 {
                        let v = b[16 + c * 32 + l];
                        self.q[c * 64 + l] = (v & 0xf) as i8;
                        self.q[c * 64 + 32 + l] = (v >> 4) as i8;
                    }
                }
            }
            DType::Q5K => {
                self.gs = 32;
                self.d = rd16(b, 0);
                self.dmin = rd16(b, 2);
                for g in 0..8 {
                    let (s, m) = scale_min_k4(g, &b[4..16]);
                    self.sc[g] = s as i32;
                    self.mn[g] = m as i32;
                }
                for c in 0..4 {
                    for l in 0..32 {
                        let (v, h) = (b[48 + c * 32 + l], b[16 + l]);
                        self.q[c * 64 + l] = ((v & 0xf) + (((h >> (2 * c)) & 1) << 4)) as i8;
                        self.q[c * 64 + 32 + l] =
                            ((v >> 4) + (((h >> (2 * c + 1)) & 1) << 4)) as i8;
                    }
                }
            }
            DType::Q6K => {
                self.gs = 16;
                self.d = rd16(b, 208);
                self.dmin = 0.0;
                self.mn = [0; 16];
                for g in 0..16 {
                    self.sc[g] = b[192 + g] as i8 as i32;
                }
                for half in 0..2 {
                    for l in 0..32 {
                        let (ql, ql2, qh) = (
                            b[half * 64 + l],
                            b[half * 64 + l + 32],
                            b[128 + half * 32 + l],
                        );
                        let o = half * 128 + l;
                        self.q[o] = ((ql & 0xf) | ((qh & 3) << 4)) as i8 - 32;
                        self.q[o + 32] = ((ql2 & 0xf) | (((qh >> 2) & 3) << 4)) as i8 - 32;
                        self.q[o + 64] = ((ql >> 4) | (((qh >> 4) & 3) << 4)) as i8 - 32;
                        self.q[o + 96] = ((ql2 >> 4) | (((qh >> 6) & 3) << 4)) as i8 - 32;
                    }
                }
            }
            _ => unreachable!("validated before"),
        }
    }
}

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    crate::simd::dot_i8_i8(a, b)
}

/// Dot of one unpacked weight block (`n` elements) with the matching activation block.
#[inline]
fn block_dot(u: &Unp, n: usize, xq: &[i8], bs: &[i32], dx: f32) -> f32 {
    let (mut main, mut mins) = (0i32, 0i32);
    let per = u.gs / 16;
    for g in 0..n / u.gs {
        main += u.sc[g]
            * dot_i8(
                &u.q[g * u.gs..(g + 1) * u.gs],
                &xq[g * u.gs..(g + 1) * u.gs],
            );
        if u.mn[g] != 0 {
            mins += u.mn[g] * bs[g * per..(g + 1) * per].iter().sum::<i32>();
        }
    }
    dx * (u.d * main as f32 - u.dmin * mins as f32)
}

// ------------------------------------------------------------------------- rows

fn row_float(t: DType, row: &[u8], x: &[f32]) -> f32 {
    match t {
        DType::F32 => row
            .chunks_exact(4)
            .zip(x)
            .map(|(c, v)| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * v)
            .sum(),
        DType::F16 => row
            .chunks_exact(2)
            .zip(x)
            .map(|(c, v)| f16_to_f32(u16::from_le_bytes([c[0], c[1]])) * v)
            .sum(),
        _ => row
            .chunks_exact(2)
            .zip(x)
            .map(|(c, v)| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) * v)
            .sum(),
    }
}

fn row_fast(t: DType, row: &[u8], act: &Act, u: &mut Unp) -> f32 {
    let (e, bsz) = t.geometry().unwrap();
    debug_assert_eq!(e, act.e);
    let mut acc = 0.0;
    for (bi, blk) in row.chunks_exact(bsz).enumerate() {
        u.unpack(t, blk);
        acc += block_dot(
            u,
            e,
            &act.q[bi * e..(bi + 1) * e],
            &act.bs[bi * (e / 16)..(bi + 1) * (e / 16)],
            act.d[bi],
        );
    }
    acc
}

fn row_exact(t: DType, row: &[u8], x: &[f32], buf: &mut Vec<f32>) -> f32 {
    let (e, bsz) = t.geometry().unwrap();
    let mut acc = 0.0;
    for (blk, xs) in row.chunks_exact(bsz).zip(x.chunks_exact(e)) {
        dequantize(t, blk, e, buf).expect("validated");
        acc += buf.iter().zip(xs).map(|(a, b)| a * b).sum::<f32>();
    }
    acc
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Fast,
    Exact,
}

fn auto_threads(rows: usize, bytes: usize) -> usize {
    if bytes < THREAD_MIN_BYTES {
        return 1;
    }
    crate::pool::global().threads().min(rows / 4).max(1)
}

#[allow(clippy::too_many_arguments)]
fn run(
    t: DType,
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
    mode: Mode,
) -> Result<(), QuantError> {
    let rb = validate(t, w.len(), rows, cols)?;
    check_io(rows, cols, x, out)?;
    if rb == 0 || rows == 0 {
        out.fill(0.0);
        return Ok(());
    }
    let w = &w[..rows * rb];
    let act = (mode == Mode::Fast && !is_float(t)).then(|| quantise_act(x, act_block(t)));
    let work = |ws: &[u8], o: &mut [f32]| {
        let (mut u, mut buf) = (Unp::new(), Vec::new());
        for (ob, row) in o.iter_mut().zip(ws.chunks_exact(rb)) {
            *ob = match (&act, is_float(t)) {
                (_, true) => row_float(t, row, x),
                (Some(a), _) => row_fast(t, row, a, &mut u),
                (None, _) => row_exact(t, row, x, &mut buf),
            };
        }
    };
    let n = threads.clamp(1, rows);
    if n == 1 {
        work(w, out);
        return Ok(());
    }
    // Rows are cut into at most `n` pieces (a piece never has fewer than `rows / n` rows); the
    // persistent pool hands the pieces to its workers. Each output row depends only on its own
    // weights, so the result does not depend on how the rows are split.
    let work = &work;
    crate::pool::global().parallel_chunks_mut(out, rows.div_ceil(n), |start, o| {
        work(&w[start * rb..(start + o.len()) * rb], o);
    });
    Ok(())
}

/// `out = W · x`, `W` stored row-major in `dtype` (`rows × cols`). Fast integer path for
/// quantised types; threads are chosen automatically from the size.
pub fn qmatvec(
    t: DType,
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), QuantError> {
    let rb = validate(t, w.len(), rows, cols)?;
    run(
        t,
        w,
        rows,
        cols,
        x,
        out,
        auto_threads(rows, rows.saturating_mul(rb)),
        Mode::Fast,
    )
}
/// Like [`qmatvec`] but with an explicit thread count (`1` = serial).
pub fn qmatvec_threads(
    t: DType,
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
) -> Result<(), QuantError> {
    run(t, w, rows, cols, x, out, threads, Mode::Fast)
}
/// Weights decoded to f32 block by block, activations untouched (no activation rounding).
pub fn qmatvec_exact(
    t: DType,
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    threads: usize,
) -> Result<(), QuantError> {
    run(t, w, rows, cols, x, out, threads, Mode::Exact)
}

/// A weight matrix kept in its stored (quantised) form.
pub struct QMatrix<'a> {
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
    data: Cow<'a, [u8]>,
}

impl<'a> QMatrix<'a> {
    pub fn new(
        dtype: DType,
        rows: usize,
        cols: usize,
        data: impl Into<Cow<'a, [u8]>>,
    ) -> Result<QMatrix<'a>, QuantError> {
        let data = data.into();
        validate(dtype, data.len(), rows, cols)?;
        Ok(QMatrix {
            dtype,
            rows,
            cols,
            data,
        })
    }
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) -> Result<(), QuantError> {
        qmatvec(self.dtype, &self.data, self.rows, self.cols, x, out)
    }
    pub fn matvec_exact(&self, x: &[f32], out: &mut [f32]) -> Result<(), QuantError> {
        let t = auto_threads(self.rows, self.data.len());
        qmatvec_exact(self.dtype, &self.data, self.rows, self.cols, x, out, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::testenc::{encode, signal};

    const ALL: [DType; 16] = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::Q8_0,
        DType::Q4_0,
        DType::Q4_1,
        DType::Q5_0,
        DType::Q5_1,
        DType::Q8K,
        DType::Q2K,
        DType::Q3K,
        DType::Q4K,
        DType::Q5K,
        DType::Q6K,
        DType::IQ4NL,
        DType::IQ4XS,
    ];

    fn matrix(t: DType, rows: usize, cols: usize) -> Vec<u8> {
        (0..rows)
            .flat_map(|r| encode(t, &signal(cols, 100 + r as u64)))
            .collect()
    }
    /// f64 reference built from `dequantize`, plus ‖w_row‖·‖x‖ as the natural error scale.
    fn reference(t: DType, w: &[u8], rows: usize, cols: usize, x: &[f32]) -> (Vec<f64>, Vec<f64>) {
        let rb = t.bytes_for(cols as u64).unwrap() as usize;
        let xn = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let (mut vals, mut scale) = (vec![], vec![]);
        for r in 0..rows {
            let mut row = vec![];
            dequantize(t, &w[r * rb..(r + 1) * rb], cols, &mut row).unwrap();
            vals.push(row.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum());
            scale.push(row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt() * xn);
        }
        (vals, scale)
    }

    #[test]
    fn fast_and_exact_paths_agree_with_dequantise_then_multiply_for_every_type() {
        let (rows, cols) = (6, 512);
        let x = signal(cols, 4242);
        for t in ALL {
            let w = matrix(t, rows, cols);
            let (want, scale) = reference(t, &w, rows, cols, &x);
            let (mut fast, mut exact) = (vec![0.0; rows], vec![0.0; rows]);
            qmatvec(t, &w, rows, cols, &x, &mut fast).unwrap();
            qmatvec_exact(t, &w, rows, cols, &x, &mut exact, 1).unwrap();
            for r in 0..rows {
                let fast_tol = if is_float(t) { 1e-5 } else { 0.02 };
                assert!(
                    (exact[r] as f64 - want[r]).abs() <= 1e-5 * scale[r] + 1e-6,
                    "{t:?} exact row {r}: {} vs {}",
                    exact[r],
                    want[r]
                );
                assert!(
                    (fast[r] as f64 - want[r]).abs() <= fast_tol * scale[r] + 1e-6,
                    "{t:?} fast row {r}: {} vs {}",
                    fast[r],
                    want[r]
                );
            }
        }
    }

    /// The fast path rounds activations to int8 per block with scale `amax/127`. For Q8_0 weights (whose
    /// own decode is exact) the result is therefore predictable to f32 round-off from an independent
    /// f64 emulation of that rounding; a different activation scale (e.g. amax/120) moves several of
    /// the rounded values and breaks agreement far beyond the 1e-5 tolerance.
    #[test]
    fn fast_path_activation_rounding_matches_an_independent_int8_emulation() {
        let (rows, cols) = (4, 256);
        // |x| spread over many magnitudes so that /127 and /120 round differently
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.0173 + if i % 32 == 5 { 1.9 } else { 0.0 })
            .collect();
        let w = matrix(DType::Q8_0, rows, cols);
        let mut got = vec![0.0f32; rows];
        qmatvec(DType::Q8_0, &w, rows, cols, &x, &mut got).unwrap();
        let rb = DType::Q8_0.bytes_for(cols as u64).unwrap() as usize;
        for r in 0..rows {
            let mut wr = vec![];
            dequantize(DType::Q8_0, &w[r * rb..(r + 1) * rb], cols, &mut wr).unwrap();
            let (mut want, mut scale) = (0.0f64, 0.0f64);
            for (bw, bx) in wr.chunks(32).zip(x.chunks(32)) {
                let amax = bx.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let d = amax / 127.0;
                for (wv, xv) in bw.iter().zip(bx) {
                    let q = (*xv as f64 / d as f64).round().clamp(-127.0, 127.0);
                    want += *wv as f64 * q * d as f64;
                    scale += (*wv as f64 * *xv as f64).abs();
                }
            }
            assert!(
                (got[r] as f64 - want).abs() <= 2e-5 * scale + 1e-6,
                "row {r}: {} vs {want}",
                got[r]
            );
        }
    }

    #[test]
    fn threaded_results_are_bit_identical_to_serial() {
        let (rows, cols) = (67, 768);
        let x = signal(cols, 9);
        for t in [DType::Q4K, DType::Q8_0, DType::Q6K, DType::Q2K, DType::F16] {
            let w = matrix(t, rows, cols);
            let (mut a, mut b, mut c, mut d) = (
                vec![0.0; rows],
                vec![0.0; rows],
                vec![0.0; rows],
                vec![0.0; rows],
            );
            qmatvec_threads(t, &w, rows, cols, &x, &mut a, 1).unwrap();
            qmatvec_threads(t, &w, rows, cols, &x, &mut b, 5).unwrap();
            assert_eq!(a, b, "{t:?} fast");
            qmatvec_exact(t, &w, rows, cols, &x, &mut c, 1).unwrap();
            qmatvec_exact(t, &w, rows, cols, &x, &mut d, 4).unwrap();
            assert_eq!(c, d, "{t:?} exact");
        }
        // more threads than rows is fine
        let w = matrix(DType::Q4_0, 2, 64);
        let mut o = vec![0.0; 2];
        qmatvec_threads(DType::Q4_0, &w, 2, 64, &signal(64, 1), &mut o, 64).unwrap();
    }

    #[test]
    fn the_pool_split_is_bit_identical_for_every_shape_and_thread_count() {
        // odd row counts, fewer rows than threads, and counts that do not divide evenly
        for t in [
            DType::Q4_0,
            DType::Q5_1,
            DType::Q8_0,
            DType::Q4K,
            DType::Q6K,
            DType::F16,
            DType::F32,
        ] {
            let cols = if t.geometry().is_some_and(|g| g.0 == 256) {
                512
            } else {
                128
            };
            for rows in [1usize, 2, 3, 7, 31, 64, 97, 255] {
                let w = matrix(t, rows, cols);
                let x = signal(cols, rows as u64);
                let (mut serial, mut exact1) = (vec![0.0; rows], vec![0.0; rows]);
                qmatvec_threads(t, &w, rows, cols, &x, &mut serial, 1).unwrap();
                qmatvec_exact(t, &w, rows, cols, &x, &mut exact1, 1).unwrap();
                for n in [2usize, 3, 4, 8, 17, 1000] {
                    let (mut a, mut b) = (vec![9.0; rows], vec![9.0; rows]);
                    qmatvec_threads(t, &w, rows, cols, &x, &mut a, n).unwrap();
                    qmatvec_exact(t, &w, rows, cols, &x, &mut b, n).unwrap();
                    assert_eq!(a, serial, "{t:?} rows {rows} threads {n} fast");
                    assert_eq!(b, exact1, "{t:?} rows {rows} threads {n} exact");
                }
                let mut auto = vec![9.0; rows];
                qmatvec(t, &w, rows, cols, &x, &mut auto).unwrap();
                assert_eq!(auto, serial, "{t:?} rows {rows} automatic");
            }
        }
    }

    #[test]
    fn many_threads_calling_at_once_all_get_the_serial_answer() {
        let (rows, cols) = (200, 512);
        let w = std::sync::Arc::new(matrix(DType::Q4K, rows, cols));
        let x = std::sync::Arc::new(signal(cols, 5));
        let mut want = vec![0.0; rows];
        qmatvec_threads(DType::Q4K, &w, rows, cols, &x, &mut want, 1).unwrap();
        let hs: Vec<_> = (0..6)
            .map(|k| {
                let (w, x, want) = (w.clone(), x.clone(), want.clone());
                std::thread::spawn(move || {
                    for i in 0..40 {
                        let mut o = vec![0.0; rows];
                        qmatvec_threads(DType::Q4K, &w, rows, cols, &x, &mut o, 2 + (i + k) % 6)
                            .unwrap();
                        assert_eq!(o, want, "caller {k} call {i}");
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
    }

    #[test]
    fn a_product_does_not_spawn_threads_of_its_own() {
        let src = include_str!("qmatmul.rs");
        let code = &src[..src.find("#[cfg(test)]").unwrap()];
        assert!(
            !code.contains("thread::scope") && !code.contains("thread::spawn"),
            "qmatmul must use the persistent pool"
        );
    }

    #[test]
    fn the_integer_path_ignores_blocks_of_zeros_and_handles_a_zero_vector() {
        let w = matrix(DType::Q4K, 3, 256);
        let mut o = vec![9.0; 3];
        qmatvec(DType::Q4K, &w, 3, 256, &vec![0.0; 256], &mut o).unwrap();
        assert_eq!(o, vec![0.0; 3]);
        let mut o = vec![9.0; 3];
        qmatvec(DType::Q4K, &[], 3, 0, &[], &mut o).unwrap();
        assert_eq!(o, vec![0.0; 3]);
    }

    #[test]
    fn hostile_sizes_are_errors_not_panics() {
        let w = matrix(DType::Q4K, 2, 256);
        let x = vec![0.1; 256];
        let mut o = vec![0.0; 2];
        // cols not a multiple of the block
        assert!(qmatvec(DType::Q4K, &w, 2, 100, &x[..100], &mut o).is_err());
        // weights too short
        assert!(matches!(
            qmatvec(DType::Q4K, &w[..w.len() - 1], 2, 256, &x, &mut o),
            Err(QuantError::BadLength { .. })
        ));
        // vector / output length mismatches
        assert!(qmatvec(DType::Q4K, &w, 2, 256, &x[..255], &mut o).is_err());
        assert!(qmatvec(DType::Q4K, &w, 2, 256, &x, &mut o[..1]).is_err());
        // sizes whose product overflows
        let mut big = vec![0.0; 0];
        assert!(qmatvec(DType::Q4K, &w, usize::MAX, 256, &x, &mut big).is_err());
        assert!(
            qmatvec(
                DType::Q4K,
                &w,
                usize::MAX / 2,
                usize::MAX / 2 - 3,
                &x,
                &mut big
            )
            .is_err()
        );
        // types we cannot decode
        assert!(matches!(
            qmatvec(DType::Other(40), &w, 2, 256, &x, &mut o),
            Err(QuantError::Unsupported(_))
        ));
        assert!(qmatvec_exact(DType::Other(40), &w, 2, 256, &x, &mut o, 2).is_err());
        assert!(QMatrix::new(DType::Q4K, 3, 256, w.clone()).is_err());
        assert!(QMatrix::new(DType::Q4_0, 1, 33, vec![0u8; 100]).is_err());
    }

    #[test]
    fn qmatrix_owns_or_borrows() {
        let (rows, cols) = (4, 256);
        let w = matrix(DType::Q6K, rows, cols);
        let x = signal(cols, 17);
        let borrowed = QMatrix::new(DType::Q6K, rows, cols, &w[..]).unwrap();
        let owned = QMatrix::new(DType::Q6K, rows, cols, w.clone()).unwrap();
        let (mut a, mut b, mut c) = (vec![0.0; rows], vec![0.0; rows], vec![0.0; rows]);
        borrowed.matvec(&x, &mut a).unwrap();
        owned.matvec(&x, &mut b).unwrap();
        owned.matvec_exact(&x, &mut c).unwrap();
        assert_eq!(a, b);
        assert_eq!(borrowed.bytes().len(), w.len());
        for (p, q) in a.iter().zip(&c) {
            assert!((p - q).abs() < 0.05 * (1.0 + q.abs()));
        }
    }

    #[test]
    fn unpacked_blocks_reproduce_the_dequantised_values() {
        // The integer path's unpacking is written separately from `dequantize`; they must agree.
        let mut u = Unp::new();
        for t in ALL.into_iter().filter(|t| !is_float(*t)) {
            let (e, _) = t.geometry().unwrap();
            let bytes = encode(t, &signal(e, 31));
            let mut want = vec![];
            dequantize(t, &bytes, e, &mut want).unwrap();
            u.unpack(t, &bytes);
            for (i, w) in want.iter().enumerate() {
                let g = i / u.gs;
                let got = u.d * (u.sc[g] * u.q[i] as i32) as f32 - u.dmin * u.mn[g] as f32;
                assert!(
                    (got - w).abs() <= 1e-5 * (1.0 + w.abs()),
                    "{t:?} element {i}: {got} vs {w}"
                );
            }
        }
    }
}
