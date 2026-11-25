//! Quantisation: encoders for every GGML block type, importance-matrix weighting, per-tensor
//! recipes, Hugging Face → GGUF conversion and GGUF requantisation.
//!
//! The encoders are written from the public block layouts (the decoders are in [`crate::quant`]) and
//! search for the scales that minimise the *weighted* squared error of each block, instead of rounding
//! with a fixed scale:
//! * symmetric types (Q4_0, Q5_0, Q3_K, Q6_K) try nineteen candidate scales around the one that maps
//!   the largest value to the end of the range and keep the best least-squares fit;
//! * affine types (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K) scan fifteen candidate step sizes, solve the least-squares
//!   scale and offset for each and keep the best, with the offset of K-quants constrained to be ≤ 0;
//! * K-quants then quantise the per-sub-block scales themselves to 4–8 bits and re-derive the values from
//!   the scales the decoder will actually see, so the stored block is self-consistent.
//!
//! Weights (`w`) come from an [`Imatrix`] (the mean squared activation of each input column): columns the
//! model is sensitive to get a finer grid. Without one, a magnitude-based weighting is used.
#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::single_match
)]
use crate::gguf_write::MetaValue;
use crate::quant::{DType, f16_to_f32};
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq)]
pub enum QuantizeError {
    /// The type has no encoder (unknown id).
    Unsupported(DType),
    /// Input length is not a whole number of blocks / rows.
    BadShape(String),
    /// NaN or infinity in the input of a quantised type.
    NonFinite,
}
impl std::fmt::Display for QuantizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizeError::Unsupported(t) => write!(f, "cannot quantise to {t:?}"),
            QuantizeError::BadShape(s) => write!(f, "{s}"),
            QuantizeError::NonFinite => write!(f, "the tensor contains NaN or infinity"),
        }
    }
}
impl std::error::Error for QuantizeError {}

// ------------------------------------------------------------------ half floats

/// f32 → IEEE binary16, round to nearest even, with subnormals, infinity and NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 | (m >> 13) as u16 } else { 0 };
    }
    let ne = e - 127 + 15;
    if ne >= 31 {
        return sign | 0x7c00;
    }
    if ne <= 0 {
        if ne < -10 {
            return sign;
        }
        let mm = m | 0x80_0000;
        let shift = (14 - ne) as u32;
        let mut h = (mm >> shift) as u16;
        let rem = mm & ((1 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = (((ne as u32) << 10) | (m >> 13)) as u16;
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    } // a carry into the exponent is correct (→ infinity)
    sign | h
}

/// f32 → bfloat16, round to nearest even (NaN stays NaN).
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b as u64 + 0x7fff + ((b >> 16) & 1) as u64) >> 16) as u16
}

#[cfg(test)]
fn r16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

// ------------------------------------------------------------------ weights and fits

/// Per-element weights for one block. With an importance vector: `imp · sqrt(σ² + x²)`; without:
/// `σ + |x|` (large values matter more). All-zero weights collapse to uniform.
fn block_weights(x: &[f32], imp: Option<&[f32]>, w: &mut [f32]) {
    let n = x.len();
    let sigma2 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    match imp {
        Some(q) => {
            for i in 0..n {
                w[i] = q[i].max(0.0) * (sigma2 + x[i] * x[i]).sqrt();
            }
        }
        None => {
            let s = sigma2.sqrt();
            for i in 0..n {
                w[i] = s + x[i].abs();
            }
        }
    }
    if w[..n].iter().all(|v| !(*v > 0.0)) {
        w[..n].fill(1.0);
    }
}

/// Symmetric fit: `x ≈ s·q` with integer `q ∈ [-nmax, nmax-1]`. Returns `s` (may be negative) and fills `q`.
fn fit_sym(x: &[f32], w: &[f32], nmax: i32, q: &mut [i8]) -> f32 {
    let mut amax = 0f32;
    let mut signed = 0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            signed = v;
        }
    }
    if amax == 0.0 {
        q.iter_mut().for_each(|v| *v = 0);
        return 0.0;
    }
    let mut best = -1f32;
    let mut best_scale = 0f32;
    let mut tmp = [0i8; 256];
    for is in -9i32..=9 {
        let iscale = -(nmax as f32 + 0.1 * is as f32) / signed;
        let (mut sumlx, mut suml2) = (0f32, 0f32);
        for i in 0..x.len() {
            let l = (iscale * x[i])
                .round()
                .clamp(-(nmax as f32), (nmax - 1) as f32);
            tmp[i] = l as i8;
            sumlx += w[i] * x[i] * l;
            suml2 += w[i] * l * l;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            best = sumlx * sumlx / suml2;
            best_scale = sumlx / suml2;
            q[..x.len()].copy_from_slice(&tmp[..x.len()]);
        }
    }
    best_scale
}

/// Affine fit: `x ≈ s·L + off` with integer `L ∈ [0, nmax]`. Returns `(s, off)` and fills `l`.
/// With `off_le_zero` the offset is constrained to ≤ 0 (the K-quant form `s·L − min`).
fn fit_affine(
    x: &[f32],
    w: &[f32],
    nmax: i32,
    nstep: i32,
    rmin: f32,
    rdelta: f32,
    off_le_zero: bool,
    l: &mut [u8],
) -> (f32, f32) {
    let n = x.len();
    let mut mn = x.iter().copied().fold(f32::INFINITY, f32::min);
    let mx = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if off_le_zero {
        mn = mn.min(0.0);
    }
    if mx <= mn {
        l[..n].fill(0);
        return (0.0, mn);
    }
    let (mut sum_w, mut sum_x) = (0f32, 0f32);
    for i in 0..n {
        sum_w += w[i];
        sum_x += w[i] * x[i];
    }
    let nm = nmax as f32;
    let mut best_scale = (mx - mn) / nm;
    let mut best_off = mn;
    let mut best_err = 0f32;
    let inv = 1.0 / best_scale;
    for i in 0..n {
        let li = (inv * (x[i] - mn)).round().clamp(0.0, nm);
        l[i] = li as u8;
        let d = best_scale * li + mn - x[i];
        best_err += w[i] * d * d;
    }
    let mut tmp = [0u8; 256];
    for is in 0..=nstep {
        let iscale = (rmin + rdelta * is as f32 + nm) / (mx - mn);
        let (mut sl, mut sl2, mut sxl) = (0f32, 0f32, 0f32);
        for i in 0..n {
            let li = (iscale * (x[i] - mn)).round().clamp(0.0, nm);
            tmp[i] = li as u8;
            sl += w[i] * li;
            sl2 += w[i] * li * li;
            sxl += w[i] * li * x[i];
        }
        let det = sum_w * sl2 - sl * sl;
        if det <= 0.0 {
            continue;
        }
        let mut sc = (sum_w * sxl - sum_x * sl) / det;
        let mut off = (sl2 * sum_x - sl * sxl) / det;
        if off_le_zero && off > 0.0 {
            off = 0.0;
            sc = if sl2 > 0.0 { sxl / sl2 } else { 0.0 };
        }
        if !(sc.is_finite() && off.is_finite()) {
            continue;
        }
        let mut err = 0f32;
        for i in 0..n {
            let d = sc * tmp[i] as f32 + off - x[i];
            err += w[i] * d * d;
        }
        if err < best_err {
            best_err = err;
            best_scale = sc;
            best_off = off;
            l[..n].copy_from_slice(&tmp[..n]);
        }
    }
    (best_scale, best_off)
}

fn put16(b: &mut [u8], at: usize, v: f32) -> f32 {
    let h = f32_to_f16(v);
    b[at..at + 2].copy_from_slice(&h.to_le_bytes());
    f16_to_f32(h)
}

// ------------------------------------------------------------------ block encoders (write into a zeroed block)

fn enc_q8_0(x: &[f32], _w: &[f32], b: &mut [u8]) {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let d = put16(b, 0, amax / 127.0);
    for i in 0..32 {
        b[2 + i] = if d > 0.0 {
            (x[i] / d).round().clamp(-127.0, 127.0) as i8 as u8
        } else {
            0
        };
    }
}

fn enc_q4_0(x: &[f32], w: &[f32], b: &mut [u8]) {
    let mut q = [0i8; 32];
    let s = fit_sym(x, w, 8, &mut q);
    put16(b, 0, s);
    for j in 0..16 {
        b[2 + j] = (q[j] + 8) as u8 | (((q[j + 16] + 8) as u8) << 4);
    }
}

fn enc_q5_0(x: &[f32], w: &[f32], b: &mut [u8]) {
    let mut q = [0i8; 32];
    let s = fit_sym(x, w, 16, &mut q);
    put16(b, 0, s);
    let mut qh = 0u32;
    for j in 0..16 {
        let (lo, hi) = ((q[j] + 16) as u8, (q[j + 16] + 16) as u8);
        b[6 + j] = (lo & 0xf) | ((hi & 0xf) << 4);
        qh |= ((lo >> 4) as u32) << j;
        qh |= ((hi >> 4) as u32) << (j + 16);
    }
    b[2..6].copy_from_slice(&qh.to_le_bytes());
}

fn enc_q4_1(x: &[f32], w: &[f32], b: &mut [u8]) {
    let mut l = [0u8; 32];
    let (s, off) = fit_affine(x, w, 15, 15, -1.0, 0.1, false, &mut l);
    let (d, m) = (put16(b, 0, s), put16(b, 2, off));
    for i in 0..32 {
        l[i] = if d > 0.0 {
            ((x[i] - m) / d).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
    }
    for j in 0..16 {
        b[4 + j] = l[j] | (l[j + 16] << 4);
    }
}

fn enc_q5_1(x: &[f32], w: &[f32], b: &mut [u8]) {
    let mut l = [0u8; 32];
    let (s, off) = fit_affine(x, w, 31, 15, -1.0, 0.1, false, &mut l);
    let (d, m) = (put16(b, 0, s), put16(b, 2, off));
    for i in 0..32 {
        l[i] = if d > 0.0 {
            ((x[i] - m) / d).round().clamp(0.0, 31.0) as u8
        } else {
            0
        };
    }
    let mut qh = 0u32;
    for j in 0..16 {
        b[8 + j] = (l[j] & 0xf) | ((l[j + 16] & 0xf) << 4);
        qh |= ((l[j] >> 4) as u32) << j;
        qh |= ((l[j + 16] >> 4) as u32) << (j + 16);
    }
    b[4..8].copy_from_slice(&qh.to_le_bytes());
}

fn enc_q8_k(x: &[f32], _w: &[f32], b: &mut [u8]) {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let d = amax / 127.0;
    b[0..4].copy_from_slice(&d.to_le_bytes());
    let mut sums = [0i32; 16];
    for i in 0..256 {
        let q = if d > 0.0 {
            (x[i] / d).round().clamp(-127.0, 127.0) as i32
        } else {
            0
        };
        b[4 + i] = q as i8 as u8;
        sums[i / 16] += q;
    }
    for (g, s) in sums.iter().enumerate() {
        b[260 + 2 * g..262 + 2 * g].copy_from_slice(&(*s as i16).to_le_bytes());
    }
}

/// The 6-bit scale/min packing shared by Q4_K and Q5_K (inverse of `scale_min_k4`).
fn pack_scale_min(ls: &[u8; 8], lm: &[u8; 8], out: &mut [u8]) {
    for j in 0..4 {
        out[j] = ls[j] & 63;
        out[j + 4] = lm[j] & 63;
    }
    for j in 4..8 {
        out[j + 4] = (ls[j] & 0xf) | ((lm[j] & 0xf) << 4);
        out[j - 4] |= (ls[j] >> 4) << 6;
        out[j] |= (lm[j] >> 4) << 6;
    }
}

/// Shared first half of Q4_K / Q5_K: fit eight 32-element sub-blocks, quantise their scales and mins to
/// 6 bits, and re-derive the values from the scales the decoder will see.
fn k_affine8(
    x: &[f32],
    w: &[f32],
    nmax: i32,
    rmin: f32,
    b: &mut [u8],
    q: &mut [u8; 256],
    ls: &mut [u8; 8],
    lm: &mut [u8; 8],
) {
    let (mut sc, mut mn) = ([0f32; 8], [0f32; 8]);
    let mut l = [0u8; 32];
    for j in 0..8 {
        let (s, off) = fit_affine(
            &x[j * 32..j * 32 + 32],
            &w[j * 32..j * 32 + 32],
            nmax,
            15,
            rmin,
            0.1,
            true,
            &mut l,
        );
        sc[j] = s.max(0.0);
        mn[j] = (-off).max(0.0);
    }
    let (ms, mm) = (
        sc.iter().fold(0f32, |a, &v| a.max(v)),
        mn.iter().fold(0f32, |a, &v| a.max(v)),
    );
    let d = put16(b, 0, ms / 63.0);
    let dmin = put16(b, 2, mm / 63.0);
    for j in 0..8 {
        ls[j] = if d > 0.0 {
            (sc[j] / d).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
        lm[j] = if dmin > 0.0 {
            (mn[j] / dmin).round().clamp(0.0, 63.0) as u8
        } else {
            0
        };
        let (dl, ml) = (d * ls[j] as f32, dmin * lm[j] as f32);
        for i in 0..32 {
            q[j * 32 + i] = if dl > 0.0 {
                ((x[j * 32 + i] + ml) / dl).round().clamp(0.0, nmax as f32) as u8
            } else {
                0
            };
        }
    }
    pack_scale_min(ls, lm, &mut b[4..16]);
}

fn enc_q4_k(x: &[f32], w: &[f32], b: &mut [u8]) {
    let (mut q, mut ls, mut lm) = ([0u8; 256], [0u8; 8], [0u8; 8]);
    k_affine8(x, w, 15, -1.0, b, &mut q, &mut ls, &mut lm);
    for c in 0..4 {
        for l in 0..32 {
            b[16 + c * 32 + l] = q[c * 64 + l] | (q[c * 64 + 32 + l] << 4);
        }
    }
}

fn enc_q5_k(x: &[f32], w: &[f32], b: &mut [u8]) {
    let (mut q, mut ls, mut lm) = ([0u8; 256], [0u8; 8], [0u8; 8]);
    k_affine8(x, w, 31, -0.5, b, &mut q, &mut ls, &mut lm);
    for c in 0..4 {
        for l in 0..32 {
            let (lo, hi) = (q[c * 64 + l], q[c * 64 + 32 + l]);
            b[48 + c * 32 + l] = (lo & 0xf) | ((hi & 0xf) << 4);
            b[16 + l] |= ((lo >> 4) << (2 * c)) | ((hi >> 4) << (2 * c + 1));
        }
    }
}

fn enc_q6_k(x: &[f32], w: &[f32], b: &mut [u8]) {
    let mut qs = [0i8; 256];
    let mut sc = [0f32; 16];
    for j in 0..16 {
        sc[j] = fit_sym(
            &x[j * 16..j * 16 + 16],
            &w[j * 16..j * 16 + 16],
            32,
            &mut qs[j * 16..j * 16 + 16],
        );
    }
    let smax = sc
        .iter()
        .copied()
        .fold(0f32, |a, v| if v.abs() > a.abs() { v } else { a });
    let d = put16(b, 208, smax / -128.0);
    let mut u = [0u8; 256];
    for j in 0..16 {
        let ls = if d != 0.0 {
            (sc[j] / d).round().clamp(-128.0, 127.0) as i8
        } else {
            0
        };
        b[192 + j] = ls as u8;
        let dl = d * ls as f32;
        for i in 0..16 {
            let q = if dl != 0.0 {
                (x[j * 16 + i] / dl).round().clamp(-32.0, 31.0) as i32
            } else {
                0
            };
            u[j * 16 + i] = (q + 32) as u8;
        }
    }
    for half in 0..2 {
        for l in 0..32 {
            let o = half * 128;
            let (q1, q2, q3, q4) = (u[o + l], u[o + l + 32], u[o + l + 64], u[o + l + 96]);
            b[half * 64 + l] = (q1 & 0xf) | ((q3 & 0xf) << 4);
            b[half * 64 + l + 32] = (q2 & 0xf) | ((q4 & 0xf) << 4);
            b[128 + half * 32 + l] =
                (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
}

fn enc_q3_k(x: &[f32], w: &[f32], b: &mut [u8]) {
    let mut qs = [0i8; 256];
    let mut sc = [0f32; 16];
    for j in 0..16 {
        sc[j] = fit_sym(
            &x[j * 16..j * 16 + 16],
            &w[j * 16..j * 16 + 16],
            4,
            &mut qs[j * 16..j * 16 + 16],
        );
    }
    let smax = sc
        .iter()
        .copied()
        .fold(0f32, |a, v| if v.abs() > a.abs() { v } else { a });
    let d = put16(b, 108, smax / -32.0);
    let mut q = [0i32; 256];
    let mut ls = [0i32; 16];
    for j in 0..16 {
        ls[j] = if d != 0.0 {
            (sc[j] / d).round().clamp(-32.0, 31.0) as i32
        } else {
            0
        };
        let dl = d * ls[j] as f32;
        for i in 0..16 {
            q[j * 16 + i] = if dl != 0.0 {
                (x[j * 16 + i] / dl).round().clamp(-4.0, 3.0) as i32
            } else {
                0
            };
        }
    }
    // scales: 6 bits each (value + 32), low nibbles in bytes 0..8 (two per byte), high pairs in 8..12
    for (i, &s) in ls.iter().enumerate() {
        let u = (s + 32) as u8;
        let (lo, hi) = (u & 0xf, u >> 4);
        if i < 8 {
            b[96 + i] |= lo
        } else {
            b[96 + i - 8] |= lo << 4
        }
        b[96 + 8 + i % 4] |= hi << (2 * (i / 4));
    }
    for e in 0..256 {
        let (n, r) = (e / 128, e % 128);
        let (j, half, l) = (r / 32, (r % 32) / 16, r % 16);
        let v = q[e];
        let q2 = if v >= 0 {
            b[half * 16 + l] |= 1 << (n * 4 + j);
            v
        } else {
            v + 4
        };
        b[32 + n * 32 + half * 16 + l] |= (q2 as u8) << (2 * j);
    }
}

fn enc_q2_k(x: &[f32], w: &[f32], b: &mut [u8]) {
    let (mut sc, mut mn) = ([0f32; 16], [0f32; 16]);
    let mut l = [0u8; 16];
    for j in 0..16 {
        let (s, off) = fit_affine(
            &x[j * 16..j * 16 + 16],
            &w[j * 16..j * 16 + 16],
            3,
            15,
            -0.5,
            0.1,
            true,
            &mut l,
        );
        sc[j] = s.max(0.0);
        mn[j] = (-off).max(0.0);
    }
    let (ms, mm) = (
        sc.iter().fold(0f32, |a, &v| a.max(v)),
        mn.iter().fold(0f32, |a, &v| a.max(v)),
    );
    let d = put16(b, 80, ms / 15.0);
    let dmin = put16(b, 82, mm / 15.0);
    let mut q = [0u8; 256];
    for j in 0..16 {
        let ls = if d > 0.0 {
            (sc[j] / d).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
        let lm = if dmin > 0.0 {
            (mn[j] / dmin).round().clamp(0.0, 15.0) as u8
        } else {
            0
        };
        b[j] = ls | (lm << 4);
        let (dl, ml) = (d * ls as f32, dmin * lm as f32);
        for i in 0..16 {
            q[j * 16 + i] = if dl > 0.0 {
                ((x[j * 16 + i] + ml) / dl).round().clamp(0.0, 3.0) as u8
            } else {
                0
            };
        }
    }
    for e in 0..256 {
        let (n, r) = (e / 128, e % 128);
        let (j, half, l) = (r / 32, (r % 32) / 16, r % 16);
        b[16 + n * 32 + half * 16 + l] |= q[e] << (2 * j);
    }
}


/// Nearest entry of the IQ4 grid for `v / d`.
fn iq4_nearest(t: f32) -> u8 {
    let mut best = 0usize;
    let mut be = f32::INFINITY;
    for (i, &k) in crate::quant::KVALUES_IQ4NL.iter().enumerate() {
        let e = (t - f32::from(k)).abs();
        if e < be {
            be = e;
            best = i;
        }
    }
    best as u8
}

/// Fit one 32-element IQ4 block: the scale `d` (before f16 rounding) that minimises the weighted squared
/// error over a few candidates, with the codes for it in `idx`.
fn iq4_fit(x: &[f32], w: &[f32], idx: &mut [u8; 32]) -> f32 {
    let amax = x
        .iter()
        .copied()
        .fold(0.0f32, |a, v| if v.abs() > a.abs() { v } else { a });
    if amax == 0.0 || !amax.is_finite() {
        idx.fill(iq4_nearest(0.0));
        return 0.0;
    }
    let mut best = (f64::INFINITY, 0.0f32);
    for step in -8i32..=8 {
        let d0 = amax / (-127.0 * (1.0 + 0.02 * step as f32));
        let mut tmp = [0u8; 32];
        let (mut sxv, mut svv) = (0.0f64, 0.0f64);
        for i in 0..32 {
            tmp[i] = iq4_nearest(x[i] / d0);
            let v = f64::from(crate::quant::KVALUES_IQ4NL[tmp[i] as usize]);
            let wi = f64::from(w.get(i).copied().unwrap_or(1.0).max(0.0));
            sxv += wi * f64::from(x[i]) * v;
            svv += wi * v * v;
        }
        let d = if svv > 0.0 { (sxv / svv) as f32 } else { d0 };
        let mut err = 0.0f64;
        for i in 0..32 {
            let v = f64::from(crate::quant::KVALUES_IQ4NL[iq4_nearest(x[i] / d) as usize]);
            let wi = f64::from(w.get(i).copied().unwrap_or(1.0).max(0.0));
            let e = f64::from(x[i]) - f64::from(d) * v;
            err += wi * e * e;
        }
        if err < best.0 {
            best = (err, d);
        }
    }
    let d = best.1;
    for i in 0..32 {
        idx[i] = iq4_nearest(x[i] / d);
    }
    d
}

fn enc_iq4_nl(x: &[f32], w: &[f32], b: &mut [u8]) {
    let mut idx = [0u8; 32];
    let d = iq4_fit(x, w, &mut idx);
    let d = put16(b, 0, d);
    if d != 0.0 {
        for i in 0..32 {
            idx[i] = iq4_nearest(x[i] / d);
        }
    }
    for j in 0..16 {
        b[2 + j] = idx[j] | (idx[j + 16] << 4);
    }
}

fn enc_iq4_xs(x: &[f32], w: &[f32], b: &mut [u8]) {
    let mut ds = [0.0f32; 8];
    let mut idx = [0u8; 32];
    for ib in 0..8 {
        ds[ib] = iq4_fit(&x[32 * ib..32 * ib + 32], &w[32 * ib..32 * ib + 32], &mut idx);
    }
    let dmax = ds.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let d = put16(b, 0, dmax / 31.0);
    let mut sh = 0u16;
    for ib in 0..8 {
        let ls = if d > 0.0 {
            (ds[ib] / d).round().clamp(-32.0, 31.0) as i32
        } else {
            0
        };
        let u = (ls + 32) as u8;
        b[4 + ib / 2] |= (u & 0xf) << (4 * (ib % 2));
        sh |= u16::from(u >> 4) << (2 * ib);
        let dl = d * ls as f32;
        for i in 0..32 {
            idx[i] = if dl != 0.0 {
                iq4_nearest(x[32 * ib + i] / dl)
            } else {
                iq4_nearest(0.0)
            };
        }
        for j in 0..16 {
            b[8 + 16 * ib + j] = idx[j] | (idx[j + 16] << 4);
        }
    }
    b[2..4].copy_from_slice(&sh.to_le_bytes());
}

type BlockEnc = fn(&[f32], &[f32], &mut [u8]);

fn block_encoder(t: DType) -> Option<BlockEnc> {
    Some(match t {
        DType::Q8_0 => enc_q8_0,
        DType::Q4_0 => enc_q4_0,
        DType::Q5_0 => enc_q5_0,
        DType::Q4_1 => enc_q4_1,
        DType::Q5_1 => enc_q5_1,
        DType::Q8K => enc_q8_k,
        DType::Q4K => enc_q4_k,
        DType::Q5K => enc_q5_k,
        DType::Q6K => enc_q6_k,
        DType::Q3K => enc_q3_k,
        DType::Q2K => enc_q2_k,
        DType::IQ4NL => enc_iq4_nl,
        DType::IQ4XS => enc_iq4_xs,
        _ => return None,
    })
}

/// Encode one row (`x.len()` a whole number of blocks) into `out` (exactly the row's byte size).
/// `imp` is the per-column importance for this row (same length as `x`), if any.
pub fn quantize_row_into(
    t: DType,
    x: &[f32],
    imp: Option<&[f32]>,
    out: &mut [u8],
) -> Result<(), QuantizeError> {
    let (e, bsz) = t.geometry().ok_or(QuantizeError::Unsupported(t))?;
    let need = t.bytes_for(x.len() as u64).ok_or_else(|| {
        QuantizeError::BadShape(format!(
            "{} values are not a whole number of {t:?} blocks",
            x.len()
        ))
    })? as usize;
    if out.len() != need {
        return Err(QuantizeError::BadShape(format!(
            "output buffer is {} bytes, {need} needed",
            out.len()
        )));
    }
    if let Some(q) = imp {
        if q.len() != x.len() {
            return Err(QuantizeError::BadShape(
                "importance vector length differs from the row".into(),
            ));
        }
    }
    match t {
        DType::F32 => {
            for (o, v) in out.chunks_exact_mut(4).zip(x) {
                o.copy_from_slice(&v.to_le_bytes());
            }
            return Ok(());
        }
        DType::F16 => {
            for (o, v) in out.chunks_exact_mut(2).zip(x) {
                o.copy_from_slice(&f32_to_f16(*v).to_le_bytes());
            }
            return Ok(());
        }
        DType::BF16 => {
            for (o, v) in out.chunks_exact_mut(2).zip(x) {
                o.copy_from_slice(&f32_to_bf16(*v).to_le_bytes());
            }
            return Ok(());
        }
        _ => {}
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(QuantizeError::NonFinite);
    }
    let enc = block_encoder(t).ok_or(QuantizeError::Unsupported(t))?;
    let mut w = [0f32; 256];
    out.fill(0);
    for (bi, (xb, ob)) in x.chunks_exact(e).zip(out.chunks_exact_mut(bsz)).enumerate() {
        block_weights(xb, imp.map(|q| &q[bi * e..bi * e + e]), &mut w[..e]);
        enc(xb, &w[..e], ob);
    }
    Ok(())
}

/// Encode one row into a fresh vector.
pub fn quantize_row(t: DType, x: &[f32], imp: Option<&[f32]>) -> Result<Vec<u8>, QuantizeError> {
    let n = t.bytes_for(x.len() as u64).ok_or_else(|| {
        QuantizeError::BadShape(format!("{} values do not fit {t:?} blocks", x.len()))
    })? as usize;
    let mut v = vec![0u8; n];
    quantize_row_into(t, x, imp, &mut v)?;
    Ok(v)
}

/// Encode a `rows × cols` matrix (row-major). Rows are independent, so the work is split across
/// `threads` and the result is bit-identical for any thread count. `imp` has `cols` entries.
pub fn quantize_matrix(
    t: DType,
    data: &[f32],
    cols: usize,
    imp: Option<&[f32]>,
    threads: usize,
) -> Result<Vec<u8>, QuantizeError> {
    if cols == 0 || data.len() % cols != 0 {
        return Err(QuantizeError::BadShape(format!(
            "{} values are not a whole number of rows of {cols}",
            data.len()
        )));
    }
    let rows = data.len() / cols;
    let rb = t.bytes_for(cols as u64).ok_or_else(|| {
        QuantizeError::BadShape(format!(
            "a row of {cols} is not a whole number of {t:?} blocks"
        ))
    })? as usize;
    if let Some(q) = imp {
        if q.len() != cols {
            return Err(QuantizeError::BadShape(
                "importance vector length differs from the row length".into(),
            ));
        }
    }
    let total = rows
        .checked_mul(rb)
        .ok_or_else(|| QuantizeError::BadShape("tensor too large".into()))?;
    let mut out = vec![0u8; total];
    let threads = threads.clamp(1, rows.max(1));
    let per = rows.div_ceil(threads).max(1);
    let result: Vec<Result<(), QuantizeError>> = std::thread::scope(|s| {
        let hs: Vec<_> = out
            .chunks_mut(per * rb)
            .zip(data.chunks(per * cols))
            .map(|(o, d)| {
                s.spawn(move || {
                    for (orow, xrow) in o.chunks_exact_mut(rb).zip(d.chunks_exact(cols)) {
                        quantize_row_into(t, xrow, imp, orow)?;
                    }
                    Ok(())
                })
            })
            .collect();
        hs.into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or(Err(QuantizeError::BadShape("a worker panicked".into())))
            })
            .collect()
    });
    for r in result {
        r?;
    }
    Ok(out)
}

// ------------------------------------------------------------------ errors

#[derive(Debug)]
pub enum ConvertError {
    Io(std::io::Error),
    /// A malformed input file or config.
    Format(String),
    /// A model family, tensor or type this converter does not handle.
    Unsupported(String),
    Quantize(QuantizeError),
    Write(crate::gguf_write::WriteError),
}
impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvertError::Io(e) => write!(f, "{e}"),
            ConvertError::Format(s) | ConvertError::Unsupported(s) => write!(f, "{s}"),
            ConvertError::Quantize(e) => write!(f, "{e}"),
            ConvertError::Write(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for ConvertError {}
impl From<std::io::Error> for ConvertError {
    fn from(e: std::io::Error) -> Self {
        ConvertError::Io(e)
    }
}
impl From<QuantizeError> for ConvertError {
    fn from(e: QuantizeError) -> Self {
        ConvertError::Quantize(e)
    }
}
impl From<crate::gguf_write::WriteError> for ConvertError {
    fn from(e: crate::gguf_write::WriteError) -> Self {
        ConvertError::Write(e)
    }
}
impl From<crate::loader::LoadError> for ConvertError {
    fn from(e: crate::loader::LoadError) -> Self {
        ConvertError::Format(e.to_string())
    }
}
fn fmt_err<T>(s: impl Into<String>) -> Result<T, ConvertError> {
    Err(ConvertError::Format(s.into()))
}

// ------------------------------------------------------------------ importance matrix

/// Per-tensor, per-input-column importance (mean squared activation), keyed by GGUF tensor name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Imatrix {
    pub weights: HashMap<String, Vec<f32>>,
}

const MAX_IMATRIX_VALUES: usize = 1 << 28;

impl Imatrix {
    /// The weights for a tensor with `cols` input columns. Several rows (one per expert) are averaged.
    pub fn for_cols(&self, name: &str, cols: usize) -> Option<Vec<f32>> {
        let w = self.weights.get(name)?;
        if cols == 0 || w.is_empty() || w.len() % cols != 0 {
            return None;
        }
        let rows = w.len() / cols;
        if rows == 1 {
            return Some(w.clone());
        }
        let mut o = vec![0f32; cols];
        for r in w.chunks_exact(cols) {
            for (a, b) in o.iter_mut().zip(r) {
                *a += b / rows as f32;
            }
        }
        Some(o)
    }

    /// `{"blk.0.attn_q.weight": [..], ...}`, optionally nested under `"tensors"`.
    pub fn from_json(s: &str) -> Result<Imatrix, ConvertError> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| ConvertError::Format(format!("imatrix JSON: {e}")))?;
        let obj = v
            .get("tensors")
            .filter(|t| t.is_object())
            .unwrap_or(&v)
            .as_object()
            .ok_or_else(|| ConvertError::Format("imatrix JSON must be an object".into()))?;
        let mut weights = HashMap::new();
        let mut total = 0usize;
        for (k, a) in obj {
            let a = a.as_array().ok_or_else(|| {
                ConvertError::Format(format!("imatrix entry {k} is not an array"))
            })?;
            total = total.saturating_add(a.len());
            if total > MAX_IMATRIX_VALUES {
                return fmt_err("imatrix is implausibly large");
            }
            let mut w = Vec::with_capacity(a.len());
            for x in a {
                let f = x.as_f64().ok_or_else(|| {
                    ConvertError::Format(format!("imatrix entry {k} holds a non-number"))
                })? as f32;
                if !f.is_finite() || f < 0.0 {
                    return fmt_err(format!(
                        "imatrix entry {k} holds a negative or non-finite weight"
                    ));
                }
                w.push(f);
            }
            weights.insert(k.clone(), w);
        }
        Ok(Imatrix { weights })
    }
    pub fn to_json(&self) -> String {
        let mut keys: Vec<&String> = self.weights.keys().collect();
        keys.sort();
        let mut m = serde_json::Map::new();
        for k in keys {
            m.insert(k.clone(), serde_json::json!(self.weights[k]));
        }
        serde_json::Value::Object(m).to_string()
    }

    /// The older llama.cpp `imatrix.dat`: `i32 n`, then per entry `i32 name_len, name, i32 ncall, i32 nval,
    /// f32[nval]` (sums of squared activations, divided here by `ncall`), then an optional trailer that is ignored.
    /// This layout is written from memory of the public format and has not been checked against a real file.
    pub fn from_dat(b: &[u8]) -> Result<Imatrix, ConvertError> {
        let mut p = 0usize;
        let i32_ = |p: &mut usize| -> Result<i32, ConvertError> {
            let s = b
                .get(*p..*p + 4)
                .ok_or_else(|| ConvertError::Format("imatrix.dat is truncated".into()))?;
            *p += 4;
            Ok(i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        };
        let n = i32_(&mut p)?;
        if !(0..=(1 << 20)).contains(&n) {
            return fmt_err("imatrix.dat: implausible entry count");
        }
        let mut weights = HashMap::new();
        let mut total = 0usize;
        for _ in 0..n {
            let len = i32_(&mut p)?;
            if !(1..=4096).contains(&len) {
                return fmt_err("imatrix.dat: implausible name length");
            }
            let name = b
                .get(p..p + len as usize)
                .ok_or_else(|| ConvertError::Format("imatrix.dat is truncated".into()))?;
            p += len as usize;
            let name = String::from_utf8(name.to_vec())
                .map_err(|_| ConvertError::Format("imatrix.dat: name is not UTF-8".into()))?;
            let ncall = i32_(&mut p)?;
            let nval = i32_(&mut p)?;
            if nval < 0 || ncall < 0 {
                return fmt_err("imatrix.dat: negative count");
            }
            let nval = nval as usize;
            total = total.saturating_add(nval);
            if total > MAX_IMATRIX_VALUES {
                return fmt_err("imatrix.dat is implausibly large");
            }
            let raw = b
                .get(p..p + nval * 4)
                .ok_or_else(|| ConvertError::Format("imatrix.dat is truncated".into()))?;
            p += nval * 4;
            let div = ncall.max(1) as f32;
            let mut w: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) / div)
                .collect();
            if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return fmt_err("imatrix.dat holds a negative or non-finite value");
            }
            w.shrink_to_fit();
            weights.insert(name, w);
        }
        Ok(Imatrix { weights })
    }
    pub fn to_dat(&self) -> Vec<u8> {
        let mut keys: Vec<&String> = self.weights.keys().collect();
        keys.sort();
        let mut o = (keys.len() as i32).to_le_bytes().to_vec();
        for k in keys {
            let w = &self.weights[k];
            o.extend((k.len() as i32).to_le_bytes());
            o.extend(k.as_bytes());
            o.extend(1i32.to_le_bytes());
            o.extend((w.len() as i32).to_le_bytes());
            for v in w {
                o.extend(v.to_le_bytes());
            }
        }
        o
    }

    /// The GGUF form: for every tensor `T`, `T.in_sum2` (sums of squares) and `T.counts` (token counts).
    /// Best effort: the exact layout used by recent llama.cpp builds is reconstructed from memory.
    pub fn from_gguf(path: &std::path::Path) -> Result<Imatrix, ConvertError> {
        use crate::loader::TensorSource;
        let f = crate::loader::GgufFile::open(path)?;
        let mut weights = HashMap::new();
        for t in f.infos() {
            let Some(base) = t.name.strip_suffix(".in_sum2") else {
                continue;
            };
            let sums = f.read(&t.name)?;
            let counts = if f.has(&format!("{base}.counts")) {
                f.read(&format!("{base}.counts"))?
            } else {
                vec![1.0]
            };
            // sums are [n_expert, cols] (cols fastest); counts one per expert
            let rows = counts.len().max(1);
            if sums.len() % rows != 0 {
                return fmt_err(format!("{base}: in_sum2 and counts disagree"));
            }
            let cols = sums.len() / rows;
            let mut w = vec![0f32; cols];
            let mut used = 0f32;
            for (r, row) in sums.chunks_exact(cols.max(1)).enumerate() {
                let c = counts.get(r).copied().unwrap_or(1.0);
                if c > 0.0 {
                    used += 1.0;
                    for (a, b) in w.iter_mut().zip(row) {
                        *a += b / c;
                    }
                }
            }
            if used > 0.0 {
                w.iter_mut().for_each(|v| *v /= used);
            }
            if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return fmt_err(format!("{base}: non-finite importance"));
            }
            weights.insert(base.to_string(), w);
        }
        if weights.is_empty() {
            return fmt_err("no *.in_sum2 tensors found: not an imatrix GGUF");
        }
        Ok(Imatrix { weights })
    }

    /// Load by file extension / magic: `.json`, `.dat`, or a GGUF.
    pub fn load(path: &std::path::Path) -> Result<Imatrix, ConvertError> {
        let b = std::fs::read(path)?;
        if b.starts_with(b"GGUF") {
            return Self::from_gguf(path);
        }
        if path.extension().is_some_and(|e| e == "json") || b.first() == Some(&b'{') {
            return Self::from_json(&String::from_utf8_lossy(&b));
        }
        Self::from_dat(&b)
    }
}

/// Accumulates mean squared activations while calibration text is run through the model.
#[derive(Default)]
pub struct ImatrixAccumulator {
    sums: HashMap<String, (Vec<f64>, u64)>,
}
impl ImatrixAccumulator {
    /// `acts` holds whole rows of `cols` inputs of the tensor named `name` (GGUF name).
    pub fn add(&mut self, name: &str, cols: usize, acts: &[f32]) -> Result<(), QuantizeError> {
        if cols == 0 || acts.len() % cols != 0 {
            return Err(QuantizeError::BadShape(format!(
                "{name}: activations are not whole rows of {cols}"
            )));
        }
        let e = self
            .sums
            .entry(name.to_string())
            .or_insert_with(|| (vec![0.0; cols], 0));
        if e.0.len() != cols {
            return Err(QuantizeError::BadShape(format!(
                "{name}: column count changed between calls"
            )));
        }
        for row in acts.chunks_exact(cols) {
            for (s, v) in e.0.iter_mut().zip(row) {
                *s += (*v as f64) * (*v as f64);
            }
            e.1 += 1;
        }
        Ok(())
    }
    pub fn finish(self) -> Imatrix {
        Imatrix {
            weights: self
                .sums
                .into_iter()
                .map(|(k, (s, n))| {
                    (
                        k,
                        s.into_iter()
                            .map(|v| (v / n.max(1) as f64) as f32)
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}

/// Run `run` once per calibration sample (token ids); it feeds the activations of every linear layer into
/// the accumulator. Returns the importance matrix.
pub fn compute_imatrix<E>(
    samples: &[Vec<u32>],
    mut run: impl FnMut(&[u32], &mut ImatrixAccumulator) -> Result<(), E>,
) -> Result<Imatrix, E> {
    let mut acc = ImatrixAccumulator::default();
    for s in samples {
        run(s, &mut acc)?;
    }
    Ok(acc.finish())
}

// ------------------------------------------------------------------ recipes

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerSel {
    All,
    /// llama.cpp's `use_more_bits`: the first eighth, the last eighth, and every third layer between.
    MoreBits,
}
fn use_more_bits(i: usize, n: usize) -> bool {
    i < n / 8 || i >= 7 * n / 8 || (i - n / 8) % 3 == 2
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub pattern: String,
    pub dtype: DType,
    pub layers: LayerSel,
}

/// A per-tensor quantisation plan: user `overrides` first, then the table `rules` (first match wins),
/// then `default`. Norms, biases and routers stay f32 unless an override says otherwise.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub name: String,
    /// llama.cpp `general.file_type`.
    pub file_type: u32,
    pub default: DType,
    pub rules: Vec<Rule>,
    pub overrides: Vec<(String, DType)>,
}

/// `*` matches any run of characters; everything else is literal.
pub fn glob(pat: &str, s: &str) -> bool {
    let (p, s) = (pat.as_bytes(), s.as_bytes());
    let (mut pi, mut si, mut star, mut mark) = (0, 0, usize::MAX, 0);
    while si < s.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = pi;
            mark = si;
            pi += 1;
        } else if pi < p.len() && p[pi] == s[si] {
            pi += 1;
            si += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            mark += 1;
            si = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

fn layer_of(name: &str) -> Option<usize> {
    let r = name.strip_prefix("blk.")?;
    r[..r.find('.')?].parse().ok()
}

pub fn dtype_name(t: DType) -> String {
    match t {
        DType::Q4_0 => "Q4_0",
        DType::Q4_1 => "Q4_1",
        DType::Q5_0 => "Q5_0",
        DType::Q5_1 => "Q5_1",
        DType::Q8_0 => "Q8_0",
        DType::Q2K => "Q2_K",
        DType::Q3K => "Q3_K",
        DType::Q4K => "Q4_K",
        DType::Q5K => "Q5_K",
        DType::Q6K => "Q6_K",
        DType::Q8K => "Q8_K",
        DType::IQ4NL => "IQ4_NL",
        DType::IQ4XS => "IQ4_XS",
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::Other(n) => return format!("type{n}"),
    }
    .to_string()
}
pub fn parse_dtype(s: &str) -> Option<DType> {
    Some(match s.to_ascii_uppercase().replace('-', "_").as_str() {
        "Q4_0" => DType::Q4_0,
        "Q4_1" => DType::Q4_1,
        "Q5_0" => DType::Q5_0,
        "Q5_1" => DType::Q5_1,
        "Q8_0" => DType::Q8_0,
        "Q2_K" => DType::Q2K,
        "Q3_K" => DType::Q3K,
        "Q4_K" => DType::Q4K,
        "Q5_K" => DType::Q5K,
        "Q6_K" => DType::Q6K,
        "Q8_K" => DType::Q8K,
        "IQ4_NL" => DType::IQ4NL,
        "IQ4_XS" => DType::IQ4XS,
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        _ => return None,
    })
}

impl Recipe {
    pub const NAMES: [&'static str; 17] = [
        "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K_M", "Q4_K_S",
        "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "IQ4_NL", "IQ4_XS",
    ];

    pub fn named(name: &str) -> Option<Recipe> {
        let key = name.to_ascii_uppercase().replace('-', "_");
        let r = |pattern: &str, dtype, layers| Rule {
            pattern: pattern.into(),
            dtype,
            layers,
        };
        let (file_type, default, rules): (u32, DType, Vec<Rule>) = match key.as_str() {
            "F32" => (0, DType::F32, vec![]),
            "F16" => (1, DType::F16, vec![]),
            "BF16" => (32, DType::BF16, vec![]),
            "Q8_0" => (7, DType::Q8_0, vec![]),
            "Q4_0" => (2, DType::Q4_0, vec![]),
            "Q4_1" => (3, DType::Q4_1, vec![]),
            "Q5_0" => (8, DType::Q5_0, vec![]),
            "Q5_1" => (9, DType::Q5_1, vec![]),
            "Q2_K" => (
                10,
                DType::Q2K,
                vec![r("output.weight", DType::Q6K, LayerSel::All)],
            ),
            "Q3_K_M" | "Q3_K" => (
                12,
                DType::Q3K,
                vec![
                    r("output.weight", DType::Q6K, LayerSel::All),
                    r("*attn_v.weight", DType::Q5K, LayerSel::MoreBits),
                    r("*ffn_down.weight", DType::Q5K, LayerSel::MoreBits),
                    r("*ffn_down_exps.weight", DType::Q5K, LayerSel::MoreBits),
                ],
            ),
            "Q4_K_S" => (
                14,
                DType::Q4K,
                vec![r("output.weight", DType::Q6K, LayerSel::All)],
            ),
            "Q4_K_M" | "Q4_K" => (
                15,
                DType::Q4K,
                vec![
                    r("output.weight", DType::Q6K, LayerSel::All),
                    r("token_embd.weight", DType::Q6K, LayerSel::All),
                    r("*attn_v.weight", DType::Q6K, LayerSel::MoreBits),
                    r("*ffn_down.weight", DType::Q6K, LayerSel::MoreBits),
                    r("*ffn_down_exps.weight", DType::Q6K, LayerSel::MoreBits),
                ],
            ),
            "Q5_K_S" => (
                16,
                DType::Q5K,
                vec![r("output.weight", DType::Q6K, LayerSel::All)],
            ),
            "Q5_K_M" | "Q5_K" => (
                17,
                DType::Q5K,
                vec![
                    r("output.weight", DType::Q6K, LayerSel::All),
                    r("token_embd.weight", DType::Q6K, LayerSel::All),
                    r("*attn_v.weight", DType::Q6K, LayerSel::MoreBits),
                    r("*ffn_down.weight", DType::Q6K, LayerSel::MoreBits),
                    r("*ffn_down_exps.weight", DType::Q6K, LayerSel::MoreBits),
                ],
            ),
            "Q6_K" => (18, DType::Q6K, vec![]),
            "IQ4_NL" => (
                25,
                DType::IQ4NL,
                vec![r("output.weight", DType::Q6K, LayerSel::All)],
            ),
            "IQ4_XS" => (
                30,
                DType::IQ4XS,
                vec![r("output.weight", DType::Q6K, LayerSel::All)],
            ),
            _ => return None,
        };
        Some(Recipe {
            name: key,
            file_type,
            default,
            rules,
            overrides: vec![],
        })
    }

    /// Force tensors matching `pattern` (`*` wildcards, matched against the GGUF name) to `dtype`.
    pub fn with_override(mut self, pattern: &str, dtype: DType) -> Recipe {
        self.overrides.push((pattern.to_string(), dtype));
        self
    }

    /// The stored type of a tensor. `shape` is row-major; the last dimension is the row length.
    pub fn plan(&self, name: &str, shape: &[u64], n_layers: usize) -> DType {
        let cols = shape.last().copied().unwrap_or(0);
        let wanted = if let Some((_, t)) = self.overrides.iter().find(|(p, _)| glob(p, name)) {
            *t
        } else if shape.len() < 2 || name.contains("ffn_gate_inp") || name.ends_with(".bias") {
            DType::F32
        } else {
            let i = layer_of(name);
            self.rules
                .iter()
                .find(|r| {
                    glob(&r.pattern, name)
                        && match r.layers {
                            LayerSel::All => true,
                            LayerSel::MoreBits => i.is_some_and(|i| {
                                n_layers > 0 && i < n_layers && use_more_bits(i, n_layers)
                            }),
                        }
                })
                .map_or(self.default, |r| r.dtype)
        };
        fit_type(wanted, cols)
    }
}

/// The nearest type whose block size divides `cols` (llama.cpp's fallbacks).
fn fit_type(mut t: DType, cols: u64) -> DType {
    for _ in 0..4 {
        match t.geometry() {
            Some((e, _)) if cols % e as u64 == 0 && cols > 0 => return t,
            _ => {}
        }
        t = match t {
            DType::Q2K | DType::Q3K => DType::Q4_0,
            DType::Q4K => DType::Q5_0,
            DType::Q5K => DType::Q5_1,
            DType::Q6K | DType::Q8K => DType::Q8_0,
            DType::F32 => return DType::F32,
            _ => DType::F16,
        };
    }
    DType::F16
}

// ------------------------------------------------------------------ llama Q/K permutation

/// llama.cpp's interleaved-RoPE row order from the Hugging Face (rotate-half) order: the exact inverse of
/// [`crate::loader::unpermute_rows`]. `rows = n_head × head_dim`.
pub fn permute_rows(w: &[f32], rows: usize, cols: usize, n_head: usize) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    if n_head == 0 || rows % n_head != 0 || w.len() != rows * cols {
        out.copy_from_slice(w);
        return out;
    }
    let hd = rows / n_head;
    let half = hd / 2;
    for h in 0..n_head {
        for t in 0..2 {
            for j in 0..half {
                let (src, dst) = (h * hd + t * half + j, h * hd + j * 2 + t);
                out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
            }
        }
    }
    out
}

// ------------------------------------------------------------------ options and report

pub struct ConvertOptions {
    pub recipe: Recipe,
    pub imatrix: Option<Imatrix>,
    pub threads: usize,
    /// `general.name`; defaults to the folder name.
    pub name: Option<String>,
}
impl ConvertOptions {
    pub fn new(recipe: Recipe) -> ConvertOptions {
        ConvertOptions {
            recipe,
            imatrix: None,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            name: None,
        }
    }
}

#[derive(Debug, Default)]
pub struct ConvertReport {
    pub arch: String,
    pub tensors: usize,
    pub output_bytes: u64,
    pub warnings: Vec<String>,
    /// Every tensor and the type it was stored as.
    pub types: Vec<(String, DType)>,
}

fn encode_f32(
    t: DType,
    data: &[f32],
    cols: usize,
    name: &str,
    o: &ConvertOptions,
) -> Result<Vec<u8>, ConvertError> {
    let imp = if t.geometry().is_some_and(|g| g.0 > 1) {
        o.imatrix.as_ref().and_then(|m| m.for_cols(name, cols))
    } else {
        None
    };
    quantize_matrix(t, data, cols, imp.as_deref(), o.threads.max(1)).map_err(|e| match e {
        QuantizeError::NonFinite => {
            ConvertError::Format(format!("{name}: the tensor contains NaN or infinity"))
        }
        e => e.into(),
    })
}

// ------------------------------------------------------------------ HF → GGUF names

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    Plain,
    Q,
    K,
    Norm,
}
enum Mapped {
    Skip,
    One(String, Role),
    /// One expert of a stacked tensor.
    Expert {
        layer: usize,
        part: &'static str,
        idx: usize,
    },
}

fn gguf_arch(family: &str) -> Option<&'static str> {
    Some(match family {
        "llama" | "mistral" | "mixtral" => "llama",
        "qwen2" => "qwen2",
        "qwen3" => "qwen3",
        "gemma" => "gemma",
        "gemma2" => "gemma2",
        "gemma3" | "gemma3_text" => "gemma3",
        "phi3" => "phi3",
        "qwen2_moe" | "qwen2moe" => "qwen2moe",
        "qwen3_moe" | "qwen3moe" => "qwen3moe",
        "olmoe" => "olmoe",
        _ => return None,
    })
}

fn split_wb(s: &str) -> Option<(&str, &str)> {
    if let Some(b) = s.strip_suffix(".weight") {
        Some((b, "weight"))
    } else {
        s.strip_suffix(".bias").map(|b| (b, "bias"))
    }
}

fn map_hf(name: &str, post_norms: bool, moe: bool) -> Result<Mapped, ConvertError> {
    let mut n = name;
    while let Some(r) = n.strip_prefix("language_model.") {
        n = r;
    }
    let top = match n {
        "model.embed_tokens.weight" => Some("token_embd.weight"),
        "model.norm.weight" | "model.final_layernorm.weight" => Some("output_norm.weight"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(t) = top {
        return Ok(Mapped::One(
            t.into(),
            if t == "output_norm.weight" {
                Role::Norm
            } else {
                Role::Plain
            },
        ));
    }
    if n.starts_with("vision_tower")
        || n.starts_with("multi_modal_projector")
        || n.starts_with("visual.")
        || n.starts_with("vision_model")
    {
        return Ok(Mapped::Skip);
    }
    let unsupported = || {
        ConvertError::Unsupported(format!(
            "tensor {name} has no GGUF mapping in this converter"
        ))
    };
    let rest = n.strip_prefix("model.layers.").ok_or_else(unsupported)?;
    let (idx, rest) = rest.split_once('.').ok_or_else(unsupported)?;
    let i: usize = idx.parse().map_err(|_| unsupported())?;
    if rest.ends_with("rotary_emb.inv_freq") {
        return Ok(Mapped::Skip);
    }
    let one = |s: String, r: Role| Ok(Mapped::One(format!("blk.{i}.{s}"), r));
    // MoE experts
    if let Some(r) = rest
        .strip_prefix("block_sparse_moe.experts.")
        .or_else(|| rest.strip_prefix("mlp.experts."))
    {
        let (e, w) = r.split_once('.').ok_or_else(unsupported)?;
        let e: usize = e.parse().map_err(|_| unsupported())?;
        let part = match w {
            "w1.weight" | "gate_proj.weight" => "gate",
            "w3.weight" | "up_proj.weight" => "up",
            "w2.weight" | "down_proj.weight" => "down",
            _ => return Err(unsupported()),
        };
        return Ok(Mapped::Expert {
            layer: i,
            part,
            idx: e,
        });
    }
    let (base, wb) = split_wb(rest).ok_or_else(unsupported)?;
    let is_w = wb == "weight";
    match base {
        "input_layernorm" if is_w => one("attn_norm.weight".into(), Role::Norm),
        "post_attention_layernorm" if is_w => one(
            if post_norms {
                "post_attention_norm.weight"
            } else {
                "ffn_norm.weight"
            }
            .into(),
            Role::Norm,
        ),
        "pre_feedforward_layernorm" if is_w => one("ffn_norm.weight".into(), Role::Norm),
        "post_feedforward_layernorm" if is_w => one("post_ffw_norm.weight".into(), Role::Norm),
        "self_attn.q_proj" => one(format!("attn_q.{wb}"), Role::Q),
        "self_attn.k_proj" => one(format!("attn_k.{wb}"), Role::K),
        "self_attn.v_proj" => one(format!("attn_v.{wb}"), Role::Plain),
        "self_attn.o_proj" | "self_attn.dense" => one(format!("attn_output.{wb}"), Role::Plain),
        "self_attn.qkv_proj" => one(format!("attn_qkv.{wb}"), Role::Plain),
        "self_attn.q_norm" if is_w => one("attn_q_norm.weight".into(), Role::Norm),
        "self_attn.k_norm" if is_w => one("attn_k_norm.weight".into(), Role::Norm),
        "mlp.gate_proj" => one(format!("ffn_gate.{wb}"), Role::Plain),
        "mlp.up_proj" => one(format!("ffn_up.{wb}"), Role::Plain),
        "mlp.down_proj" => one(format!("ffn_down.{wb}"), Role::Plain),
        // Phi-3: gate and up stacked in one matrix (gate first). llama.cpp stores it as `ffn_up` with twice the
        // rows, and the loader splits it again.
        "mlp.gate_up_proj" => one(format!("ffn_up.{wb}"), Role::Plain),
        "block_sparse_moe.gate" if is_w => one("ffn_gate_inp.weight".into(), Role::Plain),
        "mlp.gate" if is_w && moe => one("ffn_gate_inp.weight".into(), Role::Plain),
        "mlp.shared_expert_gate" | "mlp.shared_experts_gate" if is_w => {
            one("ffn_gate_inp_shexp.weight".into(), Role::Plain)
        }
        b if b.starts_with("mlp.shared_expert.") || b.starts_with("mlp.shared_experts.") => {
            let p = b.rsplit('.').next().unwrap_or("");
            match p {
                "gate_proj" => one("ffn_gate_shexp.weight".into(), Role::Plain),
                "up_proj" => one("ffn_up_shexp.weight".into(), Role::Plain),
                "down_proj" => one("ffn_down_shexp.weight".into(), Role::Plain),
                _ => Err(unsupported()),
            }
        }
        _ => Err(unsupported()),
    }
}

// ------------------------------------------------------------------ metadata

fn u32v(v: u64, what: &str) -> Result<MetaValue, ConvertError> {
    u32::try_from(v)
        .map(MetaValue::U32)
        .map_err(|_| ConvertError::Format(format!("{what} does not fit 32 bits")))
}

fn arch_metadata(
    spec: &crate::arch::Spec,
    cfg: &serde_json::Value,
    a: &str,
    name: &str,
    o: &ConvertOptions,
) -> Result<Vec<(String, MetaValue)>, ConvertError> {
    use crate::arch::{NormKind, RopeScaling};
    let mut m: Vec<(String, MetaValue)> = vec![];
    let mut put = |k: &str, v: MetaValue| m.push((k.to_string(), v));
    put("general.architecture", MetaValue::str(a));
    put("general.name", MetaValue::str(name));
    put("general.quantization_version", MetaValue::U32(2));
    put("general.file_type", MetaValue::U32(o.recipe.file_type));
    let k = |s: &str| format!("{a}.{s}");
    put(&k("context_length"), u32v(spec.context, "context length")?);
    put(&k("embedding_length"), u32v(spec.hidden, "hidden size")?);
    put(&k("block_count"), u32v(spec.layers, "layer count")?);
    put(&k("feed_forward_length"), u32v(spec.ffn, "ffn size")?);
    put(&k("attention.head_count"), u32v(spec.heads, "head count")?);
    put(
        &k("attention.head_count_kv"),
        u32v(spec.kv_heads, "kv head count")?,
    );
    put(
        &k("attention.key_length"),
        u32v(spec.head_dim, "head size")?,
    );
    put(
        &k("attention.value_length"),
        u32v(spec.head_dim, "head size")?,
    );
    let eps_key = if spec.norm == NormKind::Layer {
        "attention.layer_norm_epsilon"
    } else {
        "attention.layer_norm_rms_epsilon"
    };
    put(&k(eps_key), MetaValue::F32(spec.norm_eps as f32));
    put(&k("rope.freq_base"), MetaValue::F32(spec.rope_theta as f32));
    if spec.rotary_dim != spec.head_dim {
        put(
            &k("rope.dimension_count"),
            u32v(spec.rotary_dim, "rotary dim")?,
        );
    }
    if spec.rope_theta_local > 0.0 {
        put(
            &k("rope.freq_base_swa"),
            MetaValue::F32(spec.rope_theta_local as f32),
        );
    }
    match &spec.rope_scaling {
        RopeScaling::Linear { factor } => {
            put(&k("rope.scaling.type"), MetaValue::str("linear"));
            put(&k("rope.scaling.factor"), MetaValue::F32(*factor as f32));
        }
        RopeScaling::Yarn {
            factor,
            original_max,
            ..
        } => {
            put(&k("rope.scaling.type"), MetaValue::str("yarn"));
            put(&k("rope.scaling.factor"), MetaValue::F32(*factor as f32));
            put(
                &k("rope.scaling.original_context_length"),
                MetaValue::U32(*original_max as u32),
            );
        }
        _ => {}
    }
    if spec.experts > 0 {
        put(&k("expert_count"), u32v(spec.experts, "expert count")?);
        put(
            &k("expert_used_count"),
            u32v(spec.experts_used, "experts used")?,
        );
        if spec.moe_ffn > 0 {
            put(
                &k("expert_feed_forward_length"),
                u32v(spec.moe_ffn, "expert ffn size")?,
            );
        }
        if spec.shared_experts > 0 {
            put(
                &k("expert_shared_count"),
                u32v(spec.shared_experts, "shared experts")?,
            );
        }
        put(
            &k("expert_weights_norm"),
            MetaValue::Bool(spec.norm_topk_prob),
        );
        if spec.routed_scale != 1.0 {
            put(
                &k("expert_weights_scale"),
                MetaValue::F32(spec.routed_scale as f32),
            );
        }
    }
    if spec.attn_softcap != 0.0 {
        put(
            &k("attn_logit_softcapping"),
            MetaValue::F32(spec.attn_softcap as f32),
        );
    }
    if spec.final_softcap != 0.0 {
        put(
            &k("final_logit_softcapping"),
            MetaValue::F32(spec.final_softcap as f32),
        );
    }
    if spec.attn_scale > 0.0 {
        put(
            &k("attention.scale"),
            MetaValue::F32(spec.attn_scale as f32),
        );
    }
    if spec.sliding_window > 0 {
        put(
            &k("attention.sliding_window"),
            u32v(spec.sliding_window, "sliding window")?,
        );
        // smallest period p with `layer i is local ⇔ (i+1) % p != 0` that reproduces the per-layer list
        let n = spec.layers;
        let pat = (2..=n + 1).chain([1, 0]).find(|&p| {
            (0..n).all(|i| {
                spec.sliding_layers
                    .get(i as usize)
                    .copied()
                    .unwrap_or(false)
                    == (p == 0 || (i + 1) % p != 0)
            })
        });
        match pat {
            Some(p) => put(&k("attention.sliding_window_pattern"), u32v(p, "pattern")?),
            None => {}
        }
    }
    let _ = cfg;
    Ok(m)
}

// ------------------------------------------------------------------ tokenizer metadata

fn tok_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}
fn has_byte_level(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Object(o) => {
            o.get("type").and_then(|t| t.as_str()) == Some("ByteLevel")
                || o.values().any(has_byte_level)
        }
        serde_json::Value::Array(a) => a.iter().any(has_byte_level),
        _ => false,
    }
}
fn split_regexes(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(o) => {
            if o.get("type").and_then(|t| t.as_str()) == Some("Split") {
                if let Some(r) = o
                    .get("pattern")
                    .and_then(|p| p.get("Regex"))
                    .and_then(|r| r.as_str())
                {
                    out.push(r.to_string());
                }
            }
            for x in o.values() {
                split_regexes(x, out);
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| split_regexes(x, out)),
        _ => {}
    }
}
fn pre_name(rx: &[String]) -> &'static str {
    let all = rx.join("|");
    if all.contains("\\p{N}{1,3}") {
        "llama-bpe"
    } else if all.contains("\\p{N}") && !all.contains("\\p{N}+") && all.contains("(?i:") {
        "qwen2"
    } else {
        "gpt-2"
    }
}

/// `tokenizer.ggml.*` entries from `tokenizer.json` (+ `tokenizer_config.json`, `config.json`). A model folder
/// without a tokenizer yields a warning and no entries.
fn tokenizer_metadata(
    dir: &std::path::Path,
    cfg: &serde_json::Value,
    warnings: &mut Vec<String>,
) -> Result<Vec<(String, MetaValue)>, ConvertError> {
    let tj = dir.join("tokenizer.json");
    if !tj.exists() {
        warnings.push("no tokenizer.json: the GGUF has no tokenizer metadata".into());
        return Ok(vec![]);
    }
    let t: serde_json::Value = serde_json::from_slice(&std::fs::read(&tj)?)
        .map_err(|e| ConvertError::Format(format!("tokenizer.json: {e}")))?;
    let tc: serde_json::Value = match std::fs::read(dir.join("tokenizer_config.json")) {
        Ok(b) => serde_json::from_slice(&b)
            .map_err(|e| ConvertError::Format(format!("tokenizer_config.json: {e}")))?,
        Err(_) => serde_json::Value::Null,
    };
    let model = t
        .get("model")
        .ok_or_else(|| ConvertError::Format("tokenizer.json has no model".into()))?;
    let kind = model.get("type").and_then(|v| v.as_str()).unwrap_or("BPE");
    let byte_level = has_byte_level(t.get("pre_tokenizer").unwrap_or(&serde_json::Value::Null))
        || has_byte_level(t.get("decoder").unwrap_or(&serde_json::Value::Null));
    const MAX_VOCAB: usize = 1 << 22;
    let mut entries: Vec<(usize, String, f32)> = vec![];
    match (model.get("vocab"), kind) {
        (Some(serde_json::Value::Object(v)), _) => {
            for (tok, id) in v {
                let id = id.as_u64().ok_or_else(|| {
                    ConvertError::Format("tokenizer vocab id is not a number".into())
                })? as usize;
                if id >= MAX_VOCAB {
                    return fmt_err("tokenizer vocab id is implausibly large");
                }
                entries.push((id, tok.clone(), -(id as f32)));
            }
        }
        (Some(serde_json::Value::Array(v)), _) => {
            for (id, e) in v.iter().enumerate() {
                let a = e.as_array().ok_or_else(|| {
                    ConvertError::Format("unigram vocab entry is not [token, score]".into())
                })?;
                entries.push((
                    id,
                    a.first().and_then(|x| x.as_str()).unwrap_or("").to_string(),
                    a.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
                ));
                if id >= MAX_VOCAB {
                    return fmt_err("tokenizer vocab is implausibly large");
                }
            }
        }
        _ => return fmt_err("tokenizer.json has no vocab"),
    }
    let mut added: HashMap<usize, (String, bool)> = HashMap::new();
    if let Some(a) = t.get("added_tokens").and_then(|a| a.as_array()) {
        for e in a {
            let (Some(id), Some(c)) = (
                e.get("id").and_then(|i| i.as_u64()),
                e.get("content").and_then(|c| c.as_str()),
            ) else {
                continue;
            };
            if id as usize >= MAX_VOCAB {
                return fmt_err("added token id is implausibly large");
            }
            added.insert(
                id as usize,
                (
                    c.to_string(),
                    e.get("special").and_then(|s| s.as_bool()).unwrap_or(false),
                ),
            );
        }
    }
    let n = entries
        .iter()
        .map(|e| e.0 + 1)
        .chain(added.keys().map(|k| k + 1))
        .max()
        .unwrap_or(0);
    if n == 0 {
        return fmt_err("tokenizer has an empty vocabulary");
    }
    let mut tokens: Vec<String> = (0..n).map(|i| format!("[PAD{i}]")).collect();
    let mut types = vec![5i32; n];
    let mut scores = vec![0f32; n];
    let unk_text = tc.get("unk_token").and_then(tok_text).or_else(|| {
        model
            .get("unk_token")
            .and_then(|u| u.as_str())
            .map(String::from)
    });
    for (id, tok, sc) in entries {
        types[id] = if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') && !byte_level
        {
            6
        } else {
            1
        };
        if unk_text.as_deref() == Some(tok.as_str()) {
            types[id] = 2;
        }
        tokens[id] = tok;
        scores[id] = sc;
    }
    for (id, (c, special)) in &added {
        if types[*id] == 5 || tokens[*id] != *c {
            tokens[*id] = c.clone();
        }
        types[*id] = if unk_text.as_deref() == Some(c.as_str()) {
            2
        } else if *special {
            3
        } else {
            4
        };
    }
    let mut m: Vec<(String, MetaValue)> = vec![];
    let ggml_model = if byte_level { "gpt2" } else { "llama" };
    m.push(("tokenizer.ggml.model".into(), MetaValue::str(ggml_model)));
    if byte_level {
        let mut rx = vec![];
        split_regexes(
            t.get("pre_tokenizer").unwrap_or(&serde_json::Value::Null),
            &mut rx,
        );
        m.push(("tokenizer.ggml.pre".into(), MetaValue::str(pre_name(&rx))));
    } else if kind == "BPE" {
        warnings.push("SentencePiece-style BPE tokenizer.json has no scores: they were set to -token id; convert from tokenizer.model for exact scores".into());
    }
    let ids: HashMap<&str, usize> = tokens
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    m.push((
        "tokenizer.ggml.tokens".into(),
        MetaValue::strings(tokens.iter().cloned()),
    ));
    m.push((
        "tokenizer.ggml.scores".into(),
        MetaValue::Array(scores.iter().map(|s| MetaValue::F32(*s)).collect()),
    ));
    m.push((
        "tokenizer.ggml.token_type".into(),
        MetaValue::Array(types.iter().map(|s| MetaValue::I32(*s)).collect()),
    ));
    if byte_level || kind == "BPE" {
        if let Some(mg) = model.get("merges").and_then(|v| v.as_array()) {
            if mg.len() >= (1 << 20) {
                return fmt_err(
                    "more than 2^20 merges: this build's GGUF reader would truncate them",
                );
            }
            let mut v = Vec::with_capacity(mg.len());
            for e in mg {
                v.push(match e {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Array(a) if a.len() == 2 => format!(
                        "{} {}",
                        a[0].as_str().unwrap_or(""),
                        a[1].as_str().unwrap_or("")
                    ),
                    _ => return fmt_err("a merge is neither a string nor a pair"),
                });
            }
            m.push(("tokenizer.ggml.merges".into(), MetaValue::strings(v)));
        }
    }
    let id_of = |k: &str, cfgk: &str| -> Option<u32> {
        tc.get(k)
            .and_then(tok_text)
            .and_then(|s| ids.get(s.as_str()).copied())
            .or_else(|| {
                cfg.get(cfgk)
                    .and_then(|v| {
                        v.as_u64().or_else(|| {
                            v.as_array()
                                .and_then(|a| a.first())
                                .and_then(|x| x.as_u64())
                        })
                    })
                    .map(|x| x as usize)
            })
            .filter(|i| *i < n)
            .map(|i| i as u32)
    };
    for (k, ck, key) in [
        ("bos_token", "bos_token_id", "bos_token_id"),
        ("eos_token", "eos_token_id", "eos_token_id"),
        ("unk_token", "unk_token_id", "unknown_token_id"),
        ("pad_token", "pad_token_id", "padding_token_id"),
    ] {
        if let Some(i) = id_of(k, ck) {
            m.push((format!("tokenizer.ggml.{key}"), MetaValue::U32(i)));
        }
    }
    for (jk, key) in [
        ("add_bos_token", "add_bos_token"),
        ("add_eos_token", "add_eos_token"),
    ] {
        if let Some(b) = tc.get(jk).and_then(|v| v.as_bool()) {
            m.push((format!("tokenizer.ggml.{key}"), MetaValue::Bool(b)));
        }
    }
    let tpl = match tc.get("chat_template") {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
            .or_else(|| a.first())
            .and_then(|e| e.get("template"))
            .and_then(|t| t.as_str())
            .map(String::from),
        _ => None,
    };
    if let Some(s) = tpl {
        m.push(("tokenizer.chat_template".into(), MetaValue::Str(s)));
    }
    Ok(m)
}

// ------------------------------------------------------------------ convert

struct Out {
    name: String,
    shape: Vec<u64>,
    src: Src,
    permute: Option<usize>,
    plus_one: bool,
    key: (u8, usize, String),
}
enum Src {
    One(String),
    Stack(Vec<String>),
}

fn out_key(name: &str) -> (u8, usize, String) {
    match layer_of(name) {
        Some(l) => (1, l, name.to_string()),
        None => (
            match name {
                "token_embd.weight" => 0,
                "output_norm.weight" => 2,
                _ => 3,
            },
            0,
            name.to_string(),
        ),
    }
}

/// Convert a Hugging Face model folder (`config.json`, `*.safetensors`, `tokenizer.json`…) into one GGUF file.
/// Tensors are read, converted and written one at a time, so peak memory is one tensor (one layer's
/// stacked experts for mixture-of-experts models), never the whole model.
pub fn convert_hf_dir(
    dir: &std::path::Path,
    out: &std::path::Path,
    opts: &ConvertOptions,
) -> Result<ConvertReport, ConvertError> {
    use crate::loader::{SafeTensors, TensorSource};
    let cfg_text = std::fs::read_to_string(dir.join("config.json"))?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_text)
        .map_err(|e| ConvertError::Format(format!("config.json: {e}")))?;
    let spec = crate::arch::from_hf_config(&cfg_text)
        .map_err(|e| ConvertError::Format(format!("config.json: {e}")))?;
    let a = gguf_arch(&spec.family).ok_or_else(|| {
        ConvertError::Unsupported(format!(
            "model family {:?} has no GGUF architecture mapping here",
            spec.family
        ))
    })?;
    let mut rep = ConvertReport {
        arch: a.to_string(),
        ..Default::default()
    };
    if let crate::arch::Support::Planned(why) = &spec.support {
        rep.warnings.push(format!(
            "the engine cannot run {} yet ({why}), the file is still written",
            spec.family
        ));
    }
    let st = SafeTensors::open(dir)?;
    let infos = st.infos();
    let gemma = a.starts_with("gemma");
    let permute_qk = a == "llama";
    let moe = spec.experts > 0;
    let shape_of: HashMap<&str, &[u64]> = infos
        .iter()
        .map(|t| (t.name.as_str(), t.dims.as_slice()))
        .collect();

    // ---- plan the output tensors
    let mut outs: Vec<Out> = vec![];
    let mut stacks: HashMap<(usize, &'static str), Vec<Option<String>>> = HashMap::new();
    let mut skipped = 0usize;
    let mut sorted: Vec<&crate::gguf::TensorInfo> = infos.iter().collect();
    sorted.sort_by(|x, y| x.name.cmp(&y.name));
    for t in sorted {
        match map_hf(&t.name, spec.post_norms, moe)? {
            Mapped::Skip => skipped += 1,
            Mapped::One(name, role) => {
                let permute = match role {
                    Role::Q if permute_qk && name.ends_with(".weight") => Some(spec.heads as usize),
                    Role::K if permute_qk && name.ends_with(".weight") => {
                        Some(spec.kv_heads as usize)
                    }
                    _ => None,
                };
                if permute_qk && matches!(role, Role::Q | Role::K) && name.ends_with(".bias") {
                    rep.warnings.push(format!(
                        "{name}: llama.cpp permutes this bias too, this converter leaves it as is"
                    ));
                }
                if let Some(h) = permute {
                    let ok = t.dims.len() == 2 && h > 0 && t.dims[0] % (2 * h as u64) == 0;
                    if !ok {
                        return fmt_err(format!(
                            "{}: shape {:?} cannot be split into {h} heads",
                            t.name, t.dims
                        ));
                    }
                }
                if outs.iter().any(|o| o.name == name) {
                    return fmt_err(format!("two tensors map to {name}"));
                }
                outs.push(Out {
                    key: out_key(&name),
                    name,
                    shape: t.dims.clone(),
                    src: Src::One(t.name.clone()),
                    permute,
                    plus_one: gemma && role == Role::Norm,
                });
            }
            Mapped::Expert { layer, part, idx } => {
                if spec.experts == 0 {
                    return fmt_err(format!(
                        "{}: expert tensor in a model with no experts",
                        t.name
                    ));
                }
                if idx as u64 >= spec.experts {
                    return fmt_err(format!(
                        "{}: expert index {idx} is outside 0..{}",
                        t.name, spec.experts
                    ));
                }
                let slot = stacks
                    .entry((layer, part))
                    .or_insert_with(|| vec![None; spec.experts as usize]);
                if slot[idx].replace(t.name.clone()).is_some() {
                    return fmt_err(format!("{}: duplicate expert", t.name));
                }
            }
        }
    }
    let mut stack_keys: Vec<_> = stacks.keys().copied().collect();
    stack_keys.sort();
    for (layer, part) in stack_keys {
        let names: Vec<String> = stacks[&(layer, part)]
            .iter()
            .enumerate()
            .map(|(e, n)| {
                n.clone().ok_or_else(|| {
                    ConvertError::Format(format!(
                        "layer {layer}: expert {e} of the {part} projection is missing"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        let first = shape_of[names[0].as_str()];
        if first.len() != 2 {
            return fmt_err(format!("{}: expert weights must be 2-D", names[0]));
        }
        if names.iter().any(|n| shape_of[n.as_str()] != first) {
            return fmt_err(format!("layer {layer}: the {part} experts differ in shape"));
        }
        let mut shape = vec![names.len() as u64];
        shape.extend_from_slice(first);
        let name = format!("blk.{layer}.ffn_{part}_exps.weight");
        outs.push(Out {
            key: out_key(&name),
            name,
            shape,
            src: Src::Stack(names),
            permute: None,
            plus_one: false,
        });
    }
    if outs.is_empty() {
        return fmt_err("the folder holds no tensors this converter knows");
    }
    if skipped > 0 {
        rep.warnings.push(format!(
            "{skipped} non-text tensors (vision tower, rotary tables) were skipped"
        ));
    }
    for must in ["token_embd.weight"] {
        if !outs.iter().any(|o| o.name == must) {
            return fmt_err(format!("the model has no {must}"));
        }
    }
    outs.sort_by(|x, y| x.key.cmp(&y.key));

    // ---- metadata and tensor table
    let folder = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model")
        .to_string();
    let name = opts.name.clone().unwrap_or(folder);
    let mut meta = arch_metadata(&spec, &cfg, a, &name, opts)?;
    meta.extend(tokenizer_metadata(dir, &cfg, &mut rep.warnings)?);
    let n_layers = spec.layers as usize;
    let mut descs = vec![];
    for o in &outs {
        let t = opts.recipe.plan(&o.name, &o.shape, n_layers);
        let t = if o.name.len() > crate::gguf_write::MAX_NAME {
            return fmt_err(format!("{}: name longer than 64 bytes", o.name));
        } else {
            t
        };
        descs.push(crate::gguf_write::TensorDesc::new(&o.name, &o.shape, t));
        rep.types.push((o.name.clone(), t));
    }

    // ---- stream the data
    let file = std::io::BufWriter::new(std::fs::File::create(out)?);
    let mut failure: Option<ConvertError> = None;
    let result = crate::gguf_write::GgufWriter::write_all(file, &meta, descs, |i, d| {
        let o = &outs[i];
        let r = (|| -> Result<Vec<u8>, ConvertError> {
            let cols = *o.shape.last().unwrap_or(&1) as usize;
            let one = |hf: &str| -> Result<Vec<f32>, ConvertError> {
                let mut v = st.read(hf)?;
                if let Some(h) = o.permute {
                    v = permute_rows(&v, o.shape[0] as usize, cols, h);
                }
                if o.plus_one {
                    v.iter_mut().for_each(|x| *x += 1.0);
                }
                Ok(v)
            };
            match &o.src {
                Src::One(hf) => {
                    let v = one(hf)?;
                    encode_f32(d.dtype, &v, cols.max(1), &o.name, opts)
                }
                Src::Stack(names) => {
                    let mut bytes = Vec::new();
                    for n in names {
                        let v = one(n)?;
                        bytes.extend(encode_f32(d.dtype, &v, cols, &o.name, opts)?);
                    }
                    Ok(bytes)
                }
            }
        })();
        r.map_err(|e| {
            let msg = e.to_string();
            failure = Some(e);
            crate::gguf_write::WriteError::Invalid(msg)
        })
    });
    if let Some(e) = failure {
        let _ = std::fs::remove_file(out);
        return Err(e);
    }
    let w = match result {
        Ok(w) => w,
        Err(e) => {
            let _ = std::fs::remove_file(out);
            return Err(e.into());
        }
    };
    w.into_inner()
        .map_err(|e| ConvertError::Io(e.into_error()))?
        .sync_all()
        .ok();
    rep.tensors = outs.len();
    rep.output_bytes = std::fs::metadata(out)?.len();
    Ok(rep)
}

// ------------------------------------------------------------------ requantise a GGUF

/// Requantise `input` into `output` following `opts.recipe`. Tensors whose planned type equals their stored
/// type are copied byte for byte; others are decoded to f32 and re-encoded (quality only drops further than the
/// input's, so start from F16/BF16/Q8_0 when you can). Metadata is carried over, with `general.file_type` and
/// the quantisation version updated.
pub fn quantize(
    input: &std::path::Path,
    output: &std::path::Path,
    opts: &ConvertOptions,
) -> Result<ConvertReport, ConvertError> {
    use std::io::{Read, Seek, SeekFrom};
    if input == output {
        return fmt_err("input and output are the same file");
    }
    let g = {
        let fh = std::fs::File::open(input)?;
        let n = fh.metadata()?.len();
        crate::gguf::Gguf::parse_with_len(std::io::BufReader::new(fh), n)
    }
    .map_err(|e| ConvertError::Format(format!("{}: {e}", input.display())))?;
    let mut rep = ConvertReport {
        arch: g.architecture().unwrap_or("").to_string(),
        ..Default::default()
    };
    let mut keys: Vec<&String> = g.meta.keys().collect();
    keys.sort();
    let mut meta: Vec<(String, MetaValue)> = vec![];
    for k in keys {
        if k == "general.file_type" || k == "general.quantization_version" {
            continue;
        }
        let v = &g.meta[k];
        if let crate::gguf::Value::Array(a) = v {
            if a.len() >= (1 << 20) {
                return fmt_err(format!(
                    "{k}: array reached the reader's 2^20 limit and may be truncated"
                ));
            }
        }
        meta.push((k.clone(), MetaValue::from_read(v)));
    }
    meta.push((
        "general.file_type".into(),
        MetaValue::U32(opts.recipe.file_type),
    ));
    meta.push(("general.quantization_version".into(), MetaValue::U32(2)));
    let n_layers = g.get_u64("block_count").unwrap_or(0) as usize;
    let len = std::fs::metadata(input)?.len();
    let mut descs = vec![];
    let mut srcs = vec![];
    for t in &g.tensors {
        let shape: Vec<u64> = t.dims.iter().rev().copied().collect();
        let want = if t.dtype.decodable() {
            opts.recipe.plan(&t.name, &shape, n_layers)
        } else {
            t.dtype
        };
        let bytes = t
            .bytes()
            .ok_or_else(|| ConvertError::Format(format!("{}: bad size", t.name)))?;
        let end = g
            .data_start
            .checked_add(t.offset)
            .and_then(|s| s.checked_add(bytes));
        if end.is_none_or(|e| e > len) {
            return fmt_err(format!("{}: data lies past the end of the file", t.name));
        }
        if want != t.dtype && !t.dtype.decodable() {
            return Err(ConvertError::Unsupported(format!(
                "{}: cannot decode {:?}",
                t.name, t.dtype
            )));
        }
        descs.push(crate::gguf_write::TensorDesc::new(&t.name, &shape, want));
        rep.types.push((t.name.clone(), want));
        srcs.push((shape, t.dtype, g.data_start + t.offset, bytes));
    }
    let mut f = std::fs::File::open(input)?;
    let file = std::io::BufWriter::new(std::fs::File::create(output)?);
    let mut failure: Option<ConvertError> = None;
    let tensors = &g.tensors;
    let result = crate::gguf_write::GgufWriter::write_all(file, &meta, descs, |i, d| {
        let (shape, from, off, bytes) = &srcs[i];
        let r = (|| -> Result<Vec<u8>, ConvertError> {
            let mut raw = vec![0u8; *bytes as usize];
            f.seek(SeekFrom::Start(*off))?;
            f.read_exact(&mut raw)?;
            if *from == d.dtype {
                return Ok(raw);
            }
            let n = shape.iter().product::<u64>() as usize;
            let mut x = Vec::new();
            crate::quant::dequantize(*from, &raw, n, &mut x)
                .map_err(|e| ConvertError::Format(format!("{}: {e}", tensors[i].name)))?;
            let cols = *shape.last().unwrap_or(&1) as usize;
            encode_f32(d.dtype, &x, cols.max(1), &d.name, opts)
        })();
        r.map_err(|e| {
            let msg = e.to_string();
            failure = Some(e);
            crate::gguf_write::WriteError::Invalid(msg)
        })
    });
    if let Some(e) = failure {
        let _ = std::fs::remove_file(output);
        return Err(e);
    }
    let w = match result {
        Ok(w) => w,
        Err(e) => {
            let _ = std::fs::remove_file(output);
            return Err(e.into());
        }
    };
    w.into_inner()
        .map_err(|e| ConvertError::Io(e.into_error()))?
        .sync_all()
        .ok();
    rep.tensors = g.tensors.len();
    rep.output_bytes = std::fs::metadata(output)?.len();
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::dequantize;

    fn rng(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn gauss(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| (0..4).map(|_| rng(&mut s)).sum::<f32>() * 0.5)
            .collect()
    }
    fn rmse(a: &[f32], b: &[f32]) -> f32 {
        (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32).sqrt()
    }
    fn roundtrip(t: DType, x: &[f32]) -> Vec<f32> {
        let q = quantize_row(t, x, None).unwrap();
        let mut o = Vec::new();
        dequantize(t, &q, x.len(), &mut o).unwrap();
        o
    }
    const ALL: [(DType, f32); 13] = [
        (DType::Q8_0, 0.01),
        (DType::Q4_0, 0.2),
        (DType::Q4_1, 0.2),
        (DType::Q5_0, 0.1),
        (DType::Q5_1, 0.1),
        (DType::Q2K, 0.9),
        (DType::Q3K, 0.45),
        (DType::Q4K, 0.2),
        (DType::Q5K, 0.1),
        (DType::Q6K, 0.05),
        (DType::Q8K, 0.01),
        (DType::IQ4NL, 0.15),
        (DType::IQ4XS, 0.15),
    ];

    #[test]
    fn f16_bf16_conversions() {
        for &v in &[
            0.0f32,
            -0.0,
            1.0,
            -2.5,
            65504.0,
            6.1e-5,
            5.9e-8,
            1e-9,
            70000.0,
            f32::INFINITY,
            0.1,
            3.3333,
        ] {
            let h = r16(v);
            if v.abs() < 65520.0 && v.abs() > 1e-4 {
                assert!(((h - v) / v).abs() < 1e-3, "{v} {h}");
            }
        }
        assert_eq!(f32_to_f16(70000.0), 0x7c00);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert!(f32_to_bf16(f32::NAN) & 0x7f80 == 0x7f80);
        // exhaustive: f16 → f32 → f16 is the identity for finite values
        for h in 0u16..=0xffff {
            let f = f16_to_f32(h);
            if f.is_finite() {
                assert_eq!(f32_to_f16(f), h, "{h:#x}");
            }
        }
    }

    #[test]
    fn error_bounds_every_type() {
        let x = gauss(256 * 8, 7);
        let sd = rmse(&x, &vec![0.0; x.len()]);
        println!("{:<6} {:>10} {:>10}", "type", "rmse", "rel");
        for (t, bound) in ALL {
            let y = roundtrip(t, &x);
            let e = rmse(&x, &y);
            println!("{:<6} {:>10.5} {:>10.4}", format!("{t:?}"), e, e / sd);
            assert!(
                e / sd < bound,
                "{t:?}: relative error {} over {bound}",
                e / sd
            );
        }
        for t in [DType::F32, DType::F16, DType::BF16] {
            let y = roundtrip(t, &x);
            assert!(rmse(&x, &y) / sd < 0.01);
        }
        assert_eq!(roundtrip(DType::F32, &x), x);
    }

    #[test]
    fn zeros_and_constants() {
        for (t, _) in ALL {
            assert!(
                roundtrip(t, &vec![0.0; 256]).iter().all(|v| *v == 0.0),
                "{t:?} zeros"
            );
            let y = roundtrip(t, &vec![0.5; 256]);
            assert!(
                y.iter().all(|v| (v - 0.5).abs() < 0.1),
                "{t:?} constant: {:?}",
                &y[..4]
            );
        }
    }

    /// Naive baselines written independently: fixed scale from the extremum.
    fn naive_q4_0(x: &[f32]) -> Vec<f32> {
        x.chunks(32)
            .flat_map(|c| {
                let m = c
                    .iter()
                    .copied()
                    .fold(0f32, |a, v| if v.abs() > a.abs() { v } else { a });
                let d = r16(m / -8.0);
                c.iter()
                    .map(move |v| {
                        if d == 0.0 {
                            0.0
                        } else {
                            ((v / d + 8.5).floor().clamp(0.0, 15.0) - 8.0) * d
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    fn naive_affine(x: &[f32], blk: usize, nmax: f32) -> Vec<f32> {
        x.chunks(blk)
            .flat_map(|c| {
                let mn = c.iter().copied().fold(f32::INFINITY, f32::min).min(0.0);
                let mx = c.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let d = (mx - mn) / nmax;
                c.iter()
                    .map(move |v| {
                        if d == 0.0 {
                            0.0
                        } else {
                            (((v - mn) / d).round().clamp(0.0, nmax)) * d + mn
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn search_beats_naive() {
        let x = gauss(256 * 32, 11);
        let q4 = rmse(&x, &roundtrip(DType::Q4_0, &x));
        assert!(q4 <= rmse(&x, &naive_q4_0(&x)) * 1.001, "q4_0 {q4}");
        let k4 = rmse(&x, &roundtrip(DType::Q4K, &x));
        assert!(k4 < rmse(&x, &naive_affine(&x, 32, 15.0)), "q4_k {k4}");
        let k2 = rmse(&x, &roundtrip(DType::Q2K, &x));
        assert!(k2 < rmse(&x, &naive_affine(&x, 16, 3.0)), "q2_k {k2}");
        let k5 = rmse(&x, &roundtrip(DType::Q5K, &x));
        assert!(k5 < rmse(&x, &naive_affine(&x, 32, 31.0)), "q5_k {k5}");
    }

    #[test]
    fn imatrix_lowers_weighted_error() {
        let cols = 256;
        let x = gauss(cols * 16, 5);
        let imp: Vec<f32> = (0..cols)
            .map(|c| if c % 8 == 0 { 100.0 } else { 0.01 })
            .collect();
        for t in [DType::Q4_0, DType::Q4K, DType::Q2K, DType::Q6K] {
            let plain = quantize_matrix(t, &x, cols, None, 1).unwrap();
            let weighted = quantize_matrix(t, &x, cols, Some(&imp), 1).unwrap();
            let werr = |q: &[u8]| {
                let mut o = Vec::new();
                dequantize(t, q, x.len(), &mut o).unwrap();
                x.iter()
                    .zip(&o)
                    .enumerate()
                    .map(|(i, (a, b))| imp[i % cols] * (a - b) * (a - b))
                    .sum::<f32>()
            };
            assert!(
                werr(&weighted) < werr(&plain),
                "{t:?}: {} !< {}",
                werr(&weighted),
                werr(&plain)
            );
        }
    }

    #[test]
    fn threads_are_reproducible() {
        let x = gauss(256 * 37, 3);
        let a = quantize_matrix(DType::Q4K, &x, 256, None, 1).unwrap();
        for th in [2, 3, 8, 64] {
            assert_eq!(a, quantize_matrix(DType::Q4K, &x, 256, None, th).unwrap());
        }
    }

    #[test]
    fn hostile_inputs() {
        let mut x = gauss(256, 1);
        assert!(quantize_row(DType::Q4_0, &x[..33], None).is_err());
        assert!(quantize_matrix(DType::Q4K, &x, 100, None, 1).is_err());
        assert!(quantize_matrix(DType::Q4K, &x, 0, None, 1).is_err());
        assert!(quantize_row(DType::Q4_0, &x, Some(&[1.0; 5])).is_err());
        assert!(quantize_row(DType::Other(999), &x, None).is_err());
        x[3] = f32::NAN;
        assert_eq!(
            quantize_row(DType::Q4K, &x, None),
            Err(QuantizeError::NonFinite)
        );
        x[3] = f32::INFINITY;
        assert_eq!(
            quantize_row(DType::Q8_0, &x, None),
            Err(QuantizeError::NonFinite)
        );
        let mut out = vec![0u8; 3];
        assert!(quantize_row_into(DType::Q8_0, &x[..32], None, &mut out).is_err());
    }

    // ------------------------------------------------------------------ recipes, imatrix, permutation

    #[test]
    fn glob_matching() {
        assert!(glob("*attn_v.weight", "blk.3.attn_v.weight"));
        assert!(glob("blk.*.ffn_down.weight", "blk.12.ffn_down.weight"));
        assert!(!glob("*attn_v.weight", "blk.3.attn_v.bias"));
        assert!(glob("*", ""));
        assert!(glob("a*b*c", "aXXbYYc") && !glob("a*b*c", "aXXbYY"));
        assert!(!glob("", "x") && glob("", ""));
    }

    #[test]
    fn q4_k_m_assigns_the_expected_types() {
        let r = Recipe::named("Q4_K_M").unwrap();
        assert_eq!(r.file_type, 15);
        let n = 32;
        let sh = [4096u64, 4096];
        assert_eq!(r.plan("output.weight", &sh, n), DType::Q6K);
        assert_eq!(r.plan("token_embd.weight", &sh, n), DType::Q6K);
        assert_eq!(r.plan("blk.5.attn_q.weight", &sh, n), DType::Q4K);
        assert_eq!(r.plan("blk.5.ffn_up.weight", &sh, n), DType::Q4K);
        // use_more_bits: first eighth, last eighth, every third in between
        let more: Vec<usize> = (0..n)
            .filter(|&i| r.plan(&format!("blk.{i}.attn_v.weight"), &sh, n) == DType::Q6K)
            .collect();
        let want: Vec<usize> = (0..n)
            .filter(|&i| !(4..28).contains(&i) || (i - 4) % 3 == 2)
            .collect();
        assert_eq!(more, want);
        assert_eq!(r.plan("blk.0.ffn_down.weight", &sh, n), DType::Q6K);
        assert_eq!(r.plan("blk.10.ffn_down.weight", &sh, n), DType::Q4K);
        assert_eq!(
            r.plan("blk.6.ffn_down_exps.weight", &[8, 4096, 4096], n),
            DType::Q6K
        );
        // norms, biases and routers stay f32
        assert_eq!(r.plan("blk.1.attn_norm.weight", &[4096], n), DType::F32);
        assert_eq!(r.plan("blk.1.attn_q.bias", &[4096], n), DType::F32);
        assert_eq!(
            r.plan("blk.1.ffn_gate_inp.weight", &[8, 4096], n),
            DType::F32
        );
        // rows that are not a block multiple fall back
        assert_eq!(r.plan("blk.5.attn_q.weight", &[64, 100], n), DType::F16);
    }

    #[test]
    fn fallbacks_and_overrides() {
        let r = Recipe::named("q6_k").unwrap();
        assert_eq!(r.plan("blk.0.attn_q.weight", &[8, 288], 4), DType::Q8_0); // 288 = 9×32, not 256-aligned
        assert_eq!(
            Recipe::named("Q2_K")
                .unwrap()
                .plan("blk.0.attn_q.weight", &[8, 96], 4),
            DType::Q4_0
        );
        assert_eq!(
            Recipe::named("Q4_K_M")
                .unwrap()
                .plan("blk.0.attn_q.weight", &[8, 50], 4),
            DType::F16
        );
        let r = Recipe::named("Q4_K_M")
            .unwrap()
            .with_override("*attn_k.weight", DType::Q8_0)
            .with_override("blk.0.*", DType::F16);
        assert_eq!(r.plan("blk.3.attn_k.weight", &[256, 256], 8), DType::Q8_0);
        assert_eq!(
            r.plan("blk.0.attn_k.weight", &[256, 256], 8),
            DType::Q8_0,
            "first override wins"
        );
        assert_eq!(r.plan("blk.0.ffn_up.weight", &[256, 256], 8), DType::F16);
        for n in Recipe::NAMES {
            assert!(Recipe::named(n).is_some(), "{n}");
        }
        assert!(Recipe::named("Q9_Z").is_none());
        for t in [
            DType::Q2K,
            DType::Q3K,
            DType::Q4K,
            DType::Q5K,
            DType::Q6K,
            DType::Q8_0,
            DType::F16,
            DType::BF16,
        ] {
            assert_eq!(parse_dtype(&dtype_name(t)), Some(t));
        }
    }

    #[test]
    fn permute_is_the_inverse_of_unpermute() {
        let (heads, hd, cols) = (4usize, 16usize, 5usize);
        let rows = heads * hd;
        let mut s = 9;
        let w: Vec<f32> = (0..rows * cols).map(|_| rng(&mut s)).collect();
        let p = permute_rows(&w, rows, cols, heads);
        assert_ne!(p, w);
        assert_eq!(crate::loader::unpermute_rows(&p, rows, cols, heads), w);
        assert_eq!(
            permute_rows(
                &crate::loader::unpermute_rows(&w, rows, cols, heads),
                rows,
                cols,
                heads
            ),
            w
        );
        // not a head multiple: returned unchanged instead of panicking
        assert_eq!(
            permute_rows(&w[..7 * cols], 7, cols, 4),
            w[..7 * cols].to_vec()
        );
    }

    #[test]
    fn imatrix_json_dat_and_accumulator() {
        let m = Imatrix::from_json(
            r#"{"tensors":{"a.weight":[1.0,2.0,3.0,4.0],"b.weight":[0.5,0.5]}}"#,
        )
        .unwrap();
        assert_eq!(Imatrix::from_json(&m.to_json()).unwrap(), m);
        assert_eq!(Imatrix::from_dat(&m.to_dat()).unwrap(), m);
        // two expert rows are averaged
        assert_eq!(m.for_cols("a.weight", 2).unwrap(), vec![2.0, 3.0]);
        assert_eq!(m.for_cols("a.weight", 4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(m.for_cols("a.weight", 3).is_none() && m.for_cols("zzz", 2).is_none());
        // counts divide the sums in the .dat form
        let mut d = 1i32.to_le_bytes().to_vec();
        d.extend(3i32.to_le_bytes());
        d.extend(b"x.w");
        d.extend(4i32.to_le_bytes());
        d.extend(2i32.to_le_bytes());
        d.extend(8.0f32.to_le_bytes());
        d.extend(4.0f32.to_le_bytes());
        assert_eq!(
            Imatrix::from_dat(&d).unwrap().weights["x.w"],
            vec![2.0, 1.0]
        );
        let im = compute_imatrix(&[vec![1, 2], vec![3]], |s, acc| {
            acc.add("t", 2, &[s[0] as f32, 0.0, 0.0, s[0] as f32])
        })
        .unwrap();
        // rows: (1,0),(0,1),(3,0),(0,3) twice per sample pair → mean of squares per column
        assert!(
            (im.weights["t"][0] - (1.0 + 9.0) / 4.0).abs() < 1e-6
                && (im.weights["t"][1] - (1.0 + 9.0) / 4.0).abs() < 1e-6,
            "{:?}",
            im.weights["t"]
        );
    }

    #[test]
    fn imatrix_hostile_inputs() {
        for bad in [
            "",
            "[]",
            "{\"a\":1}",
            "{\"a\":[-1.0]}",
            "{\"a\":[\"x\"]}",
            "{\"a\":[1e999]}",
        ] {
            assert!(Imatrix::from_json(bad).is_err(), "{bad}");
        }
        let m = Imatrix::from_json(r#"{"a":[1.0,2.0]}"#).unwrap();
        let d = m.to_dat();
        for cut in 0..d.len() {
            assert!(Imatrix::from_dat(&d[..cut]).is_err(), "cut {cut}");
        }
        assert!(Imatrix::from_dat(&i32::MAX.to_le_bytes()).is_err());
        assert!(Imatrix::from_dat(&(-1i32).to_le_bytes()).is_err());
        let mut acc = ImatrixAccumulator::default();
        assert!(acc.add("t", 0, &[1.0]).is_err() && acc.add("t", 3, &[1.0, 2.0]).is_err());
        acc.add("t", 2, &[1.0, 2.0]).unwrap();
        assert!(acc.add("t", 4, &[1.0; 4]).is_err());
    }

    // ------------------------------------------------------------------ conversion end to end

    use crate::loader::{GgufFile, Naming, SafeTensors, load_dense};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};

    fn tmpdir() -> PathBuf {
        static N: AtomicU32 = AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!(
            "engine-quantize-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn write_st(path: &Path, ts: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut hdr = serde_json::Map::new();
        let (mut off, mut data) = (0usize, vec![]);
        for (n, shape, v) in ts {
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            hdr.insert(n.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [off, off + b.len()]}));
            off += b.len();
            data.extend(b);
        }
        let h = serde_json::to_vec(&hdr).unwrap();
        let mut f = (h.len() as u64).to_le_bytes().to_vec();
        f.extend(h);
        f.extend(data);
        std::fs::write(path, f).unwrap();
    }
    fn mat(shape: &[usize], seed: &mut u64, scale: f32) -> (Vec<usize>, Vec<f32>) {
        (
            shape.to_vec(),
            (0..shape.iter().product::<usize>())
                .map(|_| rng(seed) * scale)
                .collect(),
        )
    }

    const H: usize = 256;
    const V: usize = 32;
    struct Tiny {
        dir: PathBuf,
        tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
    }

    /// A llama-style model small enough to convert in milliseconds, wide enough for K-quant blocks.
    fn tiny_llama(model_type: &str, experts: usize) -> Tiny {
        let dir = tmpdir();
        let (heads, kv, hd, ffn, layers) = (4usize, 2usize, 64usize, 256usize, 2usize);
        let mut cfg = serde_json::json!({"model_type": model_type, "hidden_size": H, "num_hidden_layers": layers, "num_attention_heads": heads,
            "num_key_value_heads": kv, "intermediate_size": ffn, "vocab_size": V, "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
            "max_position_embeddings": 128, "tie_word_embeddings": false});
        if experts > 0 {
            cfg["num_local_experts"] = experts.into();
            cfg["num_experts_per_tok"] = 2.into();
        }
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let mut s = 42u64;
        let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = vec![];
        let mut add = |n: String, m: (Vec<usize>, Vec<f32>)| t.push((n, m.0, m.1));
        add(
            "model.embed_tokens.weight".into(),
            mat(&[V, H], &mut s, 0.5),
        );
        add(
            "model.norm.weight".into(),
            (
                vec![H],
                (0..H).map(|i| 1.0 + 0.01 * (i % 7) as f32).collect(),
            ),
        );
        add("lm_head.weight".into(), mat(&[V, H], &mut s, 0.5));
        for l in 0..layers {
            let p = format!("model.layers.{l}");
            for n in ["input_layernorm", "post_attention_layernorm"] {
                add(format!("{p}.{n}.weight"), (vec![H], vec![1.0; H]));
            }
            add(
                format!("{p}.self_attn.q_proj.weight"),
                mat(&[heads * hd, H], &mut s, 0.1),
            );
            add(
                format!("{p}.self_attn.k_proj.weight"),
                mat(&[kv * hd, H], &mut s, 0.1),
            );
            add(
                format!("{p}.self_attn.v_proj.weight"),
                mat(&[kv * hd, H], &mut s, 0.1),
            );
            add(
                format!("{p}.self_attn.o_proj.weight"),
                mat(&[H, heads * hd], &mut s, 0.1),
            );
            if experts == 0 {
                add(
                    format!("{p}.mlp.gate_proj.weight"),
                    mat(&[ffn, H], &mut s, 0.1),
                );
                add(
                    format!("{p}.mlp.up_proj.weight"),
                    mat(&[ffn, H], &mut s, 0.1),
                );
                add(
                    format!("{p}.mlp.down_proj.weight"),
                    mat(&[H, ffn], &mut s, 0.1),
                );
            } else {
                add(
                    format!("{p}.block_sparse_moe.gate.weight"),
                    mat(&[experts, H], &mut s, 0.3),
                );
                for e in 0..experts {
                    for w in ["w1", "w3"] {
                        add(
                            format!("{p}.block_sparse_moe.experts.{e}.{w}.weight"),
                            mat(&[ffn, H], &mut s, 0.1),
                        );
                    }
                    add(
                        format!("{p}.block_sparse_moe.experts.{e}.w2.weight"),
                        mat(&[H, ffn], &mut s, 0.1),
                    );
                }
            }
        }
        write_st(&dir.join("model.safetensors"), &t);
        Tiny { dir, tensors: t }
    }

    fn logits(m: &crate::model::Model) -> Vec<f32> {
        let mut st = m.new_state();
        m.forward(&mut st, 3).unwrap();
        m.forward(&mut st, 11).unwrap()
    }
    fn load_gguf(p: &Path, hf: &crate::arch::Spec) -> crate::model::Model {
        let f = GgufFile::open(p).unwrap();
        let mut sp = crate::arch::from_gguf(&f.gguf).unwrap();
        sp.vocab = hf.vocab; // there is no tokenizer in these tiny files
        for (a, b) in [
            (sp.layers, hf.layers),
            (sp.hidden, hf.hidden),
            (sp.heads, hf.heads),
            (sp.kv_heads, hf.kv_heads),
            (sp.head_dim, hf.head_dim),
            (sp.ffn, hf.ffn),
            (sp.context, hf.context),
        ] {
            assert_eq!(a, b);
        }
        assert_eq!(sp.rope_theta, hf.rope_theta);
        assert!((sp.norm_eps - hf.norm_eps).abs() < 1e-9);
        load_dense(sp, &f, Naming::Gguf, true).unwrap()
    }

    #[test]
    fn llama_converts_and_reloads_with_the_same_logits() {
        let t = tiny_llama("llama", 0);
        let cfg = std::fs::read_to_string(t.dir.join("config.json")).unwrap();
        let hf_spec = crate::arch::from_hf_config(&cfg).unwrap();
        let reference = load_dense(
            hf_spec.clone(),
            &SafeTensors::open(&t.dir).unwrap(),
            Naming::Hf,
            false,
        )
        .unwrap();
        let want = logits(&reference);
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        println!("{:<8} {:>9} {:>10}", "recipe", "bytes", "rel rmse");
        for (name, bound) in [
            ("F32", 0.0f32),
            ("F16", 0.01),
            ("BF16", 0.02),
            ("Q8_0", 0.03),
            ("Q6_K", 0.06),
            ("Q5_K_M", 0.1),
            ("Q4_K_M", 0.2),
            ("Q4_0", 0.25),
            ("Q3_K_M", 0.45),
            ("Q2_K", 0.9),
        ] {
            let out = t.dir.join(format!("{name}.gguf"));
            let mut o = ConvertOptions::new(Recipe::named(name).unwrap());
            o.threads = 2;
            let rep = convert_hf_dir(&t.dir, &out, &o).unwrap();
            assert_eq!(rep.arch, "llama");
            let got = logits(&load_gguf(&out, &hf_spec));
            let err = (want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / want.len() as f32)
                .sqrt()
                / rms;
            println!("{:<8} {:>9} {:>10.4}", name, rep.output_bytes, err);
            assert!(
                err <= bound,
                "{name}: relative logit error {err} over {bound}"
            );
            if name == "F32" {
                assert_eq!(
                    want, got,
                    "F32 conversion must be exact (permute/unpermute included)"
                );
            }
        }
        std::fs::remove_dir_all(&t.dir).ok();
    }

    #[test]
    fn q4_k_m_file_has_the_planned_types() {
        let t = tiny_llama("llama", 0);
        let out = t.dir.join("m.gguf");
        let rep = convert_hf_dir(
            &t.dir,
            &out,
            &ConvertOptions::new(Recipe::named("Q4_K_M").unwrap()),
        )
        .unwrap();
        let g = GgufFile::open(&out).unwrap().gguf;
        let ty = |n: &str| {
            g.tensors
                .iter()
                .find(|x| x.name == n)
                .unwrap_or_else(|| panic!("{n}"))
                .dtype
        };
        assert_eq!(ty("output.weight"), DType::Q6K);
        assert_eq!(ty("token_embd.weight"), DType::Q6K);
        assert_eq!(ty("blk.0.attn_q.weight"), DType::Q4K);
        assert_eq!(ty("blk.0.attn_v.weight"), DType::Q4K); // use_more_bits(0, 2) is false …
        assert_eq!(ty("blk.1.attn_v.weight"), DType::Q6K); // … and the last layer is always in the final eighth
        assert_eq!(ty("blk.1.attn_norm.weight"), DType::F32);
        assert_eq!(ty("output_norm.weight"), DType::F32);
        assert_eq!(g.get_u64("block_count"), Some(2));
        assert_eq!(g.meta["general.file_type"].as_u64(), Some(15));
        assert_eq!(g.meta["general.quantization_version"].as_u64(), Some(2));
        assert_eq!(rep.types.len(), g.tensors.len());
        // dims are fastest-first: k_proj is [kv*hd, H] row-major → [H, kv*hd]
        assert_eq!(
            g.tensors
                .iter()
                .find(|x| x.name == "blk.0.attn_k.weight")
                .unwrap()
                .dims,
            vec![H as u64, 128]
        );
        std::fs::remove_dir_all(&t.dir).ok();
    }

    #[test]
    fn requantize_gguf_and_copy_matching_tensors_verbatim() {
        let t = tiny_llama("llama", 0);
        let hf_spec = crate::arch::from_hf_config(
            &std::fs::read_to_string(t.dir.join("config.json")).unwrap(),
        )
        .unwrap();
        let (f16, q8, q4) = (
            t.dir.join("a.gguf"),
            t.dir.join("b.gguf"),
            t.dir.join("c.gguf"),
        );
        convert_hf_dir(
            &t.dir,
            &f16,
            &ConvertOptions::new(Recipe::named("F16").unwrap()),
        )
        .unwrap();
        quantize(
            &f16,
            &q8,
            &ConvertOptions::new(Recipe::named("Q8_0").unwrap()),
        )
        .unwrap();
        let rep = quantize(
            &q8,
            &q4,
            &ConvertOptions::new(Recipe::named("Q8_0").unwrap()),
        )
        .unwrap();
        assert!(
            rep.types
                .iter()
                .all(|(n, d)| if n.ends_with("norm.weight") {
                    *d == DType::F32
                } else {
                    *d == DType::Q8_0
                })
        );
        // identical type → bytes are copied, not re-encoded
        let (a, b) = (std::fs::read(&q8).unwrap(), std::fs::read(&q4).unwrap());
        assert_eq!(a, b);
        let ref_logits = logits(&load_gguf(&f16, &hf_spec));
        let q_logits = logits(&load_gguf(&q8, &hf_spec));
        let rms = (ref_logits.iter().map(|v| v * v).sum::<f32>() / ref_logits.len() as f32).sqrt();
        let err = (ref_logits
            .iter()
            .zip(&q_logits)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / ref_logits.len() as f32)
            .sqrt()
            / rms;
        assert!(err < 0.03, "{err}");
        // metadata carried over, file type updated
        let g = GgufFile::open(&q8).unwrap().gguf;
        assert_eq!(g.meta["general.file_type"].as_u64(), Some(7));
        assert_eq!(g.architecture(), Some("llama"));
        // refuses to overwrite its own input
        assert!(
            quantize(
                &q8,
                &q8,
                &ConvertOptions::new(Recipe::named("Q4_K_M").unwrap())
            )
            .is_err()
        );
        assert!(
            quantize(
                &t.dir.join("nope.gguf"),
                &q4,
                &ConvertOptions::new(Recipe::named("Q4_K_M").unwrap())
            )
            .is_err()
        );
        std::fs::remove_dir_all(&t.dir).ok();
    }

    #[test]
    fn mixtral_experts_are_stacked_in_order() {
        let t = tiny_llama("mixtral", 4);
        let out = t.dir.join("moe.gguf");
        let rep = convert_hf_dir(
            &t.dir,
            &out,
            &ConvertOptions::new(Recipe::named("F32").unwrap()),
        )
        .unwrap();
        assert_eq!(rep.arch, "llama");
        let f = GgufFile::open(&out).unwrap();
        use crate::loader::TensorSource;
        let find = |n: &str| t.tensors.iter().find(|x| x.0 == n).unwrap().2.clone();
        for (part, w) in [("gate", "w1"), ("down", "w2"), ("up", "w3")] {
            let stacked = f.read(&format!("blk.1.ffn_{part}_exps.weight")).unwrap();
            let want: Vec<f32> = (0..4)
                .flat_map(|e| {
                    find(&format!(
                        "model.layers.1.block_sparse_moe.experts.{e}.{w}.weight"
                    ))
                })
                .collect();
            assert_eq!(stacked, want, "{part}");
        }
        let d = &f
            .gguf
            .tensors
            .iter()
            .find(|x| x.name == "blk.0.ffn_gate_exps.weight")
            .unwrap()
            .dims;
        assert_eq!(d, &vec![H as u64, 256, 4]);
        assert_eq!(
            f.read("blk.0.ffn_gate_inp.weight").unwrap(),
            find("model.layers.0.block_sparse_moe.gate.weight")
        );
        assert_eq!(f.gguf.get_u64("expert_count"), Some(4));
        assert_eq!(f.gguf.get_u64("expert_used_count"), Some(2));
        // quantised stacks are encoded expert by expert: each expert's bytes equal a standalone encode
        let q = t.dir.join("moeq.gguf");
        convert_hf_dir(
            &t.dir,
            &q,
            &ConvertOptions::new(Recipe::named("Q4_K_M").unwrap()),
        )
        .unwrap();
        let g = GgufFile::open(&q).unwrap().gguf;
        let ti = g
            .tensors
            .iter()
            .find(|x| x.name == "blk.0.ffn_gate_exps.weight")
            .unwrap();
        assert_eq!(ti.dtype, DType::Q4K);
        let bytes = std::fs::read(&q).unwrap();
        let per = DType::Q4K.bytes_for(256 * 256).unwrap() as usize;
        let start = (g.data_start + ti.offset) as usize;
        for e in 1..4 {
            let x = find(&format!(
                "model.layers.0.block_sparse_moe.experts.{e}.w1.weight"
            ));
            let solo = quantize_matrix(DType::Q4K, &x, 256, None, 1).unwrap();
            assert_eq!(
                &bytes[start + e * per..start + (e + 1) * per],
                &solo[..],
                "expert {e}"
            );
        }
        std::fs::remove_dir_all(&t.dir).ok();
    }

    #[test]
    fn gemma_norms_get_plus_one_and_sandwich_names() {
        let dir = tmpdir();
        let cfg = serde_json::json!({"model_type": "gemma2", "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 32, "intermediate_size": 64, "vocab_size": 8, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "max_position_embeddings": 64,
            "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0, "sliding_window": 16});
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let mut s = 1;
        let mut t = vec![
            (
                "model.embed_tokens.weight".to_string(),
                vec![8, 64],
                mat(&[8, 64], &mut s, 0.5).1,
            ),
            ("model.norm.weight".into(), vec![64], vec![0.25; 64]),
        ];
        for n in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            t.push((
                format!("model.layers.0.{n}.weight"),
                vec![64],
                vec![0.5; 64],
            ));
        }
        for (n, sh) in [
            ("self_attn.q_proj", [64, 64]),
            ("self_attn.k_proj", [32, 64]),
            ("self_attn.v_proj", [32, 64]),
            ("self_attn.o_proj", [64, 64]),
            ("mlp.gate_proj", [64, 64]),
            ("mlp.up_proj", [64, 64]),
            ("mlp.down_proj", [64, 64]),
        ] {
            t.push((
                format!("model.layers.0.{n}.weight"),
                sh.to_vec(),
                mat(&sh, &mut s, 0.1).1,
            ));
        }
        write_st(&dir.join("model.safetensors"), &t);
        let out = dir.join("g.gguf");
        let rep = convert_hf_dir(
            &dir,
            &out,
            &ConvertOptions::new(Recipe::named("F32").unwrap()),
        )
        .unwrap();
        assert_eq!(rep.arch, "gemma2");
        let f = GgufFile::open(&out).unwrap();
        use crate::loader::TensorSource;
        assert!(
            f.read("output_norm.weight")
                .unwrap()
                .iter()
                .all(|v| *v == 1.25)
        );
        for n in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            assert!(
                f.read(&format!("blk.0.{n}.weight"))
                    .unwrap()
                    .iter()
                    .all(|v| *v == 1.5),
                "{n}"
            );
        }
        assert!(!f.has("output.weight"));
        assert_eq!(
            f.gguf.meta["gemma2.attn_logit_softcapping"].as_f64(),
            Some(50.0)
        );
        assert_eq!(f.gguf.get_u64("attention.sliding_window"), Some(16));
        assert_eq!(f.gguf.get_u64("attention.sliding_window_pattern"), Some(2));
        let sp = crate::arch::from_gguf(&f.gguf).unwrap();
        assert!(sp.post_norms && !sp.norm_offset && sp.final_softcap == 30.0);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tokenizer_metadata_byte_level_and_sentencepiece() {
        let t = tiny_llama("llama", 0);
        let tok = serde_json::json!({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "Ġ": 3, "<|eot|>": 4}, "merges": ["a b", ["Ġ", "a"]]},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "(?i:'s|'t)|\\p{N}{1,3}| ?\\p{L}+"}, "behavior": "Isolated"}, {"type": "ByteLevel"}]},
            "added_tokens": [{"id": 4, "content": "<|eot|>", "special": true}, {"id": 6, "content": "<x>", "special": false}]});
        std::fs::write(t.dir.join("tokenizer.json"), tok.to_string()).unwrap();
        std::fs::write(t.dir.join("tokenizer_config.json"), serde_json::json!({"eos_token": "<|eot|>", "add_bos_token": false,
            "chat_template": [{"name": "other", "template": "X"}, {"name": "default", "template": "Y {{ m }}"}]}).to_string()).unwrap();
        let out = t.dir.join("t.gguf");
        let rep = convert_hf_dir(
            &t.dir,
            &out,
            &ConvertOptions::new(Recipe::named("F16").unwrap()),
        )
        .unwrap();
        assert!(rep.warnings.iter().all(|w| !w.contains("tokenizer")));
        let g = GgufFile::open(&out).unwrap().gguf;
        let s = |k: &str| g.meta[k].as_str().unwrap().to_string();
        assert_eq!(s("tokenizer.ggml.model"), "gpt2");
        assert_eq!(s("tokenizer.ggml.pre"), "llama-bpe");
        assert_eq!(s("tokenizer.chat_template"), "Y {{ m }}");
        let arr = |k: &str| {
            if let crate::gguf::Value::Array(a) = &g.meta[k] {
                a.clone()
            } else {
                panic!("{k}")
            }
        };
        assert_eq!(arr("tokenizer.ggml.tokens").len(), 7);
        assert_eq!(arr("tokenizer.ggml.tokens")[5].as_str(), Some("[PAD5]"));
        let types: Vec<u64> = arr("tokenizer.ggml.token_type")
            .iter()
            .map(|v| match v {
                crate::gguf::Value::I64(i) => *i as u64,
                v => v.as_u64().unwrap(),
            })
            .collect();
        assert_eq!(types, [1, 1, 1, 1, 3, 5, 4]);
        let merges: Vec<String> = arr("tokenizer.ggml.merges")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(merges, ["a b", "Ġ a"]);
        assert_eq!(g.meta["tokenizer.ggml.eos_token_id"].as_u64(), Some(4));
        assert_eq!(
            g.meta["tokenizer.ggml.add_bos_token"],
            crate::gguf::Value::Bool(false)
        );

        // SentencePiece-style: no ByteLevel → "llama" with scores and byte tokens
        let tok = serde_json::json!({"model": {"type": "Unigram", "vocab": [["<unk>", 0.0], ["<s>", 0.0], ["</s>", 0.0], ["<0x0A>", 0.0], ["▁a", -1.5]], "unk_id": 0},
            "added_tokens": [{"id": 1, "content": "<s>", "special": true}, {"id": 2, "content": "</s>", "special": true}]});
        std::fs::write(t.dir.join("tokenizer.json"), tok.to_string()).unwrap();
        std::fs::write(t.dir.join("tokenizer_config.json"), serde_json::json!({"bos_token": {"content": "<s>"}, "eos_token": "</s>", "unk_token": "<unk>"}).to_string()).unwrap();
        convert_hf_dir(
            &t.dir,
            &out,
            &ConvertOptions::new(Recipe::named("F16").unwrap()),
        )
        .unwrap();
        let g = GgufFile::open(&out).unwrap().gguf;
        assert_eq!(g.meta["tokenizer.ggml.model"].as_str(), Some("llama"));
        assert!(
            !g.meta.contains_key("tokenizer.ggml.pre")
                && !g.meta.contains_key("tokenizer.ggml.merges")
        );
        let tk = crate::tokenizer::Tokenizer::from_gguf(&g).unwrap();
        assert_eq!((tk.bos, tk.eos), (Some(1), Some(2)));
        std::fs::remove_dir_all(&t.dir).ok();
    }

    #[test]
    fn hostile_models_are_rejected_not_panicked_on() {
        let base = tiny_llama("llama", 0);
        let o = ConvertOptions::new(Recipe::named("Q4_K_M").unwrap());
        let out = base.dir.join("x.gguf");
        let try_cfg = |cfg: &str| {
            std::fs::write(base.dir.join("config.json"), cfg).unwrap();
            convert_hf_dir(&base.dir, &out, &o)
        };
        for bad in [
            "",
            "[]",
            "{}",
            "{\"model_type\":\"llama\"}",
            "{\"model_type\":\"mamba\",\"hidden_size\":8,\"num_hidden_layers\":1,\"num_attention_heads\":1}",
            "{\"model_type\":\"llama\",\"hidden_size\":256,\"num_hidden_layers\":2,\"num_attention_heads\":3}",
        ] {
            assert!(try_cfg(bad).is_err(), "{bad}");
            assert!(
                !out.exists(),
                "a failed conversion must not leave a file behind"
            );
        }
        // a tensor nobody knows
        let good = serde_json::json!({"model_type": "llama", "hidden_size": H, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "intermediate_size": 256, "vocab_size": V}).to_string();
        let mut t = base.tensors.clone();
        t.push((
            "model.layers.0.mystery.weight".into(),
            vec![4],
            vec![0.0; 4],
        ));
        write_st(&base.dir.join("model.safetensors"), &t);
        let e = try_cfg(&good).unwrap_err();
        assert!(e.to_string().contains("mystery"), "{e}");
        // NaN in a weight → error naming the tensor, no output
        let mut t = base.tensors.clone();
        t.iter_mut()
            .find(|x| x.0 == "model.layers.1.self_attn.v_proj.weight")
            .unwrap()
            .2[5] = f32::NAN;
        write_st(&base.dir.join("model.safetensors"), &t);
        let e = try_cfg(&good).unwrap_err();
        assert!(
            e.to_string().contains("v_proj") || e.to_string().contains("attn_v"),
            "{e}"
        );
        assert!(!out.exists());
        // missing expert
        let moe = tiny_llama("mixtral", 4);
        let mut t = moe.tensors.clone();
        t.retain(|x| !x.0.contains("experts.2.w1"));
        write_st(&moe.dir.join("model.safetensors"), &t);
        let e = convert_hf_dir(&moe.dir, &moe.dir.join("y.gguf"), &o).unwrap_err();
        assert!(e.to_string().contains("expert 2"), "{e}");
        // a truncated / garbage safetensors, and a folder with nothing in it
        std::fs::write(
            base.dir.join("model.safetensors"),
            b"\xff\xff\xff\xff\xff\xff\xff\x7fjunk",
        )
        .unwrap();
        assert!(try_cfg(&good).is_err());
        let empty = tmpdir();
        assert!(convert_hf_dir(&empty, &empty.join("z.gguf"), &o).is_err());
        // not a GGUF
        std::fs::write(empty.join("n.gguf"), b"not a gguf at all").unwrap();
        assert!(quantize(&empty.join("n.gguf"), &empty.join("m.gguf"), &o).is_err());
        for d in [&base.dir, &moe.dir, &empty] {
            std::fs::remove_dir_all(d).ok();
        }
    }
}
