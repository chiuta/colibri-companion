//! Pure-Rust twins of the WGSL kernels. Same data layout, same work split (32 lanes per matvec
//! row, 256 threads per norm/attention workgroup), same accumulation and tree-reduction order, the
//! same bit tricks — so a mistake in a kernel's maths shows up here, on any machine, and is
//! compared against the engine's CPU kernels by the tests below.
use crate::layout::Format;
use crate::shaders::ROW_LANES;

fn ubyte(w: u32, i: u32) -> u32 {
    (w >> (i * 8)) & 255
}
fn sbyte(w: u32, i: u32) -> i32 {
    ((w << (24 - i * 8)) as i32) >> 24
}
fn wbyte(row: &[u32], base: usize, off: usize) -> u32 {
    ubyte(row[base + (off >> 2)], (off & 3) as u32)
}
/// Bit-for-bit mirror of the shader's `h2f` (the optional `unpack2x16float` capability is avoided).
pub fn h2f(h: u32) -> f32 {
    let s = (h & 0x8000) << 16;
    let (e, m) = ((h >> 10) & 31, h & 1023);
    if e == 0 {
        let v = m as f32 * 5.960_464_5e-8;
        return if s != 0 { -v } else { v };
    }
    if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    }
    f32::from_bits(s | ((e + 112) << 23) | (m << 13))
}
fn half_lo(w: u32) -> f32 {
    h2f(w & 0xffff)
}
fn half_hi(w: u32) -> f32 {
    h2f(w >> 16)
}

fn sc_min(row: &[u32], sb: usize, j: usize) -> (u32, u32) {
    if j < 4 {
        return (wbyte(row, sb, j) & 63, wbyte(row, sb, j + 4) & 63);
    }
    let a = wbyte(row, sb, j + 4);
    (
        (a & 15) | ((wbyte(row, sb, j - 4) >> 6) << 4),
        (a >> 4) | ((wbyte(row, sb, j) >> 6) << 4),
    )
}

/// What one lane computes for work item `it` of a row (`row` starts at the row's first word).
pub fn item_dot(f: Format, row: &[u32], it: usize, x: &[f32], cols: usize) -> f32 {
    match f {
        Format::F32 => {
            let mut s = 0.0f32;
            for j in 0..32 {
                let e = it * 32 + j;
                if e < cols {
                    s += f32::from_bits(row[e]) * x[e];
                }
            }
            s
        }
        Format::F16 => {
            let mut s = 0.0f32;
            for j in 0..32 {
                let e = it * 32 + j;
                if e < cols {
                    let w = row[e >> 1];
                    s += (if e & 1 == 1 { half_hi(w) } else { half_lo(w) }) * x[e];
                }
            }
            s
        }
        Format::Q8_0 => {
            let b = it * 9;
            let d = f32::from_bits(row[b]);
            let mut s = 0.0f32;
            for j in 0..32 {
                s += sbyte(row[b + 1 + (j >> 2)], (j & 3) as u32) as f32 * x[it * 32 + j];
            }
            d * s
        }
        Format::Q4_0 => {
            let b = it * 5;
            let d = f32::from_bits(row[b]);
            let mut s = 0.0f32;
            for j in 0..16 {
                let q = ubyte(row[b + 1 + (j >> 2)], (j & 3) as u32);
                let (lo, hi) = ((q & 15) as i32 - 8, (q >> 4) as i32 - 8);
                s = s + lo as f32 * x[it * 32 + j] + hi as f32 * x[it * 32 + 16 + j];
            }
            d * s
        }
        Format::Q4K => {
            let (blk, g) = (it >> 3, it & 7);
            let b = blk * 36;
            let (d, dmin) = (half_lo(row[b]), half_hi(row[b]));
            let (sc, mn) = sc_min(row, b + 1, g);
            let (c, high) = (g >> 1, g & 1 == 1);
            let (mut sq, mut sx) = (0.0f32, 0.0f32);
            for l in 0..32 {
                let byte = wbyte(row, b, 16 + c * 32 + l);
                let q = if high { byte >> 4 } else { byte & 15 };
                let xv = x[blk * 256 + g * 32 + l];
                sq += q as f32 * xv;
                sx += xv;
            }
            d * sc as f32 * sq - dmin * mn as f32 * sx
        }
        Format::Q6K => {
            let (blk, g) = (it >> 4, it & 15);
            let b = blk * 53;
            let (n, is, quarter) = (g >> 3, g & 1, (g & 7) >> 1);
            let d = half_lo(row[b + 52]);
            let sc = sbyte(row[b + ((192 + g) >> 2)], ((192 + g) & 3) as u32) as f32;
            let mut s = 0.0f32;
            for l in is * 16..is * 16 + 16 {
                let qlb = wbyte(row, b, n * 64 + (quarter & 1) * 32 + l);
                let nib = if quarter >= 2 { qlb >> 4 } else { qlb & 15 };
                let h = (wbyte(row, b, 128 + n * 32 + l) >> (2 * quarter)) & 3;
                let q = (nib | (h << 4)) as i32 - 32;
                s += q as f32 * x[blk * 256 + g * 16 + (l - is * 16)];
            }
            d * sc * s
        }
    }
}

/// One output row: 32 lanes stride over the items, then a tree reduction.
pub fn matvec_row(f: Format, row: &[u32], x: &[f32], cols: usize) -> f32 {
    let items = f.row_items(cols);
    let mut part = [0.0f32; ROW_LANES];
    for (lane, p) in part.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let mut it = lane;
        while it < items {
            acc += item_dot(f, row, it, x, cols);
            it += ROW_LANES;
        }
        *p = acc;
    }
    let mut s = ROW_LANES / 2;
    while s > 0 {
        for l in 0..s {
            part[l] += part[l + s];
        }
        s >>= 1;
    }
    part[0]
}

pub fn matvec(f: Format, words: &[u32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let rw = f.row_words(cols);
    (0..rows)
        .map(|r| matvec_row(f, &words[r * rw..(r + 1) * rw], x, cols))
        .collect()
}

fn tree(part: &mut [f32; 256], f: impl Fn(f32, f32) -> f32) -> f32 {
    let mut s = 128;
    while s > 0 {
        for t in 0..s {
            part[t] = f(part[t], part[t + s]);
        }
        s >>= 1;
    }
    part[0]
}

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut part = [0.0f32; 256];
    for (t, p) in part.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let mut i = t;
        while i < n {
            acc += x[i] * x[i];
            i += 256;
        }
        *p = acc;
    }
    let k = 1.0 / (tree(&mut part, |a, b| a + b) / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * k * w[i]).collect()
}

pub fn rope(x: &mut [f32], head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for i in 0..x.len() / head_dim * half {
        let (h, k) = (i / half, i % half);
        let o = h * head_dim + k;
        let (a, b) = (x[o], x[o + half]);
        x[o] = a * cos[k] - b * sin[k];
        x[o + half] = a * sin[k] + b * cos[k];
    }
}

pub fn silu_mul(g: &mut [f32], u: &[f32]) {
    for (v, m) in g.iter_mut().zip(u) {
        *v = *v / (1.0 + (-*v).exp()) * m;
    }
}
pub fn add(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Attention for all heads of one query token over cache positions `start..n_pos`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    n_heads: usize,
    kv_heads: usize,
    hd: usize,
    n_pos: usize,
    start: usize,
    scale: f32,
) -> Vec<f32> {
    let row = kv_heads * hd;
    let mut out = vec![0.0f32; n_heads * hd];
    let mut sc = vec![0.0f32; n_pos];
    for h in 0..n_heads {
        let kvh = h / (n_heads / kv_heads);
        let qb = h * hd;
        let mut part = [-3.0e38f32; 256];
        for (t, p) in part.iter_mut().enumerate() {
            let mut s = start + t;
            let mut m = -3.0e38f32;
            while s < n_pos {
                let mut d = 0.0f32;
                for i in 0..hd {
                    d += q[qb + i] * kc[s * row + kvh * hd + i];
                }
                d *= scale;
                sc[s] = d;
                m = m.max(d);
                s += 256;
            }
            *p = m;
        }
        let mx = tree(&mut part, f32::max);
        let mut part = [0.0f32; 256];
        for (t, p) in part.iter_mut().enumerate() {
            let mut s = start + t;
            let mut z = 0.0f32;
            while s < n_pos {
                let e = (sc[s] - mx).exp();
                sc[s] = e;
                z += e;
                s += 256;
            }
            *p = z;
        }
        let zt = tree(&mut part, |a, b| a + b);
        for d in 0..hd {
            let mut acc = 0.0f32;
            for s in start..n_pos {
                acc += sc[s] * vc[s * row + kvh * hd + d];
            }
            out[qb + d] = acc / zt;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::repack;
    use crate::plan::rope_tables;
    use companion_engine::backend::{AttnArgs, Backend, CpuBackend};
    use companion_engine::model::f32_to_f16;
    use companion_engine::quant::{DType, dequantize};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        }
    }

    fn blocks(t: DType, rows: usize, cols: usize, r: &mut Rng) -> Vec<u8> {
        let rb = t.bytes_for(cols as u64).unwrap() as usize;
        let mut b: Vec<u8> = (0..rb * rows).map(|_| r.next() as u8).collect();
        let (bs, d_at): (usize, &[(usize, f32)]) = match t {
            DType::Q8_0 => (34, &[(0, 0.02)]),
            DType::Q4_0 => (18, &[(0, 0.07)]),
            DType::Q4K => (144, &[(0, 0.011), (2, 0.017)]),
            DType::Q6K => (210, &[(208, 0.009)]),
            _ => return b,
        };
        for blk in b.chunks_mut(bs) {
            for &(o, v) in d_at {
                blk[o..o + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
            }
        }
        b
    }

    fn reference(
        t: DType,
        bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let rb = bytes.len() / rows;
        let mut out = vec![];
        let mut mag = vec![];
        for row in bytes.chunks_exact(rb) {
            let mut w = vec![];
            dequantize(t, row, cols, &mut w).unwrap();
            out.push(w.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum());
            mag.push(
                w.iter()
                    .zip(x)
                    .map(|(a, b)| (*a as f64 * *b as f64).abs())
                    .sum(),
            );
        }
        (out, mag)
    }

    #[test]
    fn the_shader_half_decoder_is_exact_for_all_65536_values() {
        for h in 0..=0xffffu32 {
            let (a, b) = (h2f(h), companion_engine::quant::f16_to_f32(h as u16));
            assert!(
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                "{h:#06x}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn every_matvec_kernel_matches_dequantise_then_multiply() {
        let mut r = Rng(0x1234_5678_9abc_def1);
        let cases: [(DType, &[usize]); 6] = [
            (DType::F32, &[33, 100, 2048]),
            (DType::F16, &[33, 100, 2048]),
            (DType::Q8_0, &[32, 96, 2048]),
            (DType::Q4_0, &[32, 96, 2048]),
            (DType::Q4K, &[256, 768, 4096]),
            (DType::Q6K, &[256, 768, 4096]),
        ];
        for (t, colss) in cases {
            for &cols in colss {
                let rows = 5;
                let bytes = if t == DType::F16 {
                    (0..rows * cols)
                        .flat_map(|_| f32_to_f16(r.f() * 2.0).to_le_bytes())
                        .collect()
                } else if t == DType::F32 {
                    (0..rows * cols)
                        .flat_map(|_| (r.f() * 2.0).to_le_bytes())
                        .collect()
                } else {
                    blocks(t, rows, cols, &mut r)
                };
                let x: Vec<f32> = (0..cols).map(|_| r.f()).collect();
                let fmt = Format::from_dtype(t).unwrap();
                let words = repack(t, rows, cols, &bytes).unwrap();
                assert_eq!(words.len(), rows * fmt.row_words(cols));
                let got = matvec(fmt, &words, rows, cols, &x);
                let (want, mag) = reference(t, &bytes, rows, cols, &x);
                for i in 0..rows {
                    let tol = 2e-4 * mag[i] + 1e-6;
                    assert!(
                        (got[i] as f64 - want[i]).abs() <= tol,
                        "{t:?} cols {cols} row {i}: {} vs {} (tol {tol})",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn q6k_and_q4k_cover_extreme_bit_patterns() {
        // all-ones and all-zeros payloads stress the high-bit extraction and the sign handling
        for fill in [0x00u8, 0xff, 0x55, 0xaa] {
            for t in [DType::Q4K, DType::Q6K, DType::Q8_0, DType::Q4_0] {
                let mut r = Rng(7);
                let mut bytes = blocks(t, 2, 512, &mut r);
                let (bs, protect): (usize, std::ops::Range<usize>) = match t {
                    DType::Q8_0 => (34, 0..2),
                    DType::Q4_0 => (18, 0..2),
                    DType::Q4K => (144, 0..4),
                    _ => (210, 208..210),
                };
                for blk in bytes.chunks_mut(bs) {
                    for (i, b) in blk.iter_mut().enumerate() {
                        if !protect.contains(&i) {
                            *b = fill;
                        }
                    }
                }
                let x: Vec<f32> = (0..512).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
                let fmt = Format::from_dtype(t).unwrap();
                let got = matvec(fmt, &repack(t, 2, 512, &bytes).unwrap(), 2, 512, &x);
                let (want, mag) = reference(t, &bytes, 2, 512, &x);
                for i in 0..2 {
                    assert!(
                        (got[i] as f64 - want[i]).abs() <= 2e-4 * mag[i] + 1e-6,
                        "{t:?} fill {fill:#x}: {} vs {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn rmsnorm_rope_silu_add_match_the_cpu_kernels() {
        let mut r = Rng(99);
        for n in [8usize, 100, 256, 1000, 4096] {
            let (x, w): (Vec<f32>, Vec<f32>) = (
                (0..n).map(|_| r.f()).collect(),
                (0..n).map(|_| 1.0 + r.f() * 0.1).collect(),
            );
            let mut want = vec![0.0; n];
            companion_engine::ops::rmsnorm(&x, &w, 1e-5, &mut want);
            for (a, b) in rmsnorm(&x, &w, 1e-5).iter().zip(&want) {
                assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "n {n}: {a} vs {b}");
            }
        }
        for pos in [0usize, 5, 4093] {
            let base: Vec<f32> = (0..4 * 16).map(|_| r.f()).collect();
            let mut want = base.clone();
            for h in want.chunks_mut(16) {
                companion_engine::ops::rope(h, pos, 10_000.0);
            }
            let (c, s) = rope_tables(pos, 16, 10_000.0);
            let mut got = base.clone();
            rope(&mut got, 16, &c, &s);
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-5, "pos {pos}: {a} vs {b}");
            }
        }
        let (mut g, u): (Vec<f32>, Vec<f32>) = (
            (0..300).map(|_| r.f() * 8.0).collect(),
            (0..300).map(|_| r.f()).collect(),
        );
        let mut want = g.clone();
        companion_engine::ops::silu_mul(&mut want, &u);
        silu_mul(&mut g, &u);
        for (a, b) in g.iter().zip(&want) {
            assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} vs {b}");
        }
        let mut a = vec![1.0f32, 2.0];
        add(&mut a, &[0.5, -2.0]);
        assert_eq!(a, vec![1.5, 0.0]);
    }

    #[test]
    fn attention_matches_the_cpu_backend_for_mha_gqa_mqa_windows_and_long_contexts() {
        let be = CpuBackend::new();
        let mut r = Rng(2024);
        for (heads, kvh, hd, n_pos, win) in [
            (4usize, 4usize, 8usize, 9usize, None),
            (8, 2, 16, 40, None),
            (4, 1, 8, 300, None),
            (6, 3, 8, 700, Some(100usize)),
            (2, 2, 8, 1, None),
        ] {
            let row = kvh * hd;
            let kv = be.kv_create(kvh, hd, n_pos).unwrap();
            let (mut kc, mut vc) = (vec![], vec![]);
            for pos in 0..n_pos {
                let (k, v): (Vec<f32>, Vec<f32>) = (
                    (0..row).map(|_| r.f()).collect(),
                    (0..row).map(|_| r.f()).collect(),
                );
                be.kv_append(
                    &kv,
                    pos,
                    &be.alloc_from(&k).unwrap(),
                    &be.alloc_from(&v).unwrap(),
                )
                .unwrap();
                kc.extend(k);
                vc.extend(v);
            }
            let q: Vec<f32> = (0..heads * hd).map(|_| r.f()).collect();
            let (qb, ob) = (be.alloc_from(&q).unwrap(), be.alloc(heads * hd).unwrap());
            let scale = 1.0 / (hd as f32).sqrt();
            be.attention(
                &kv,
                &qb,
                &ob,
                &AttnArgs {
                    n_heads: heads,
                    n_pos,
                    scale,
                    window: win,
                },
            )
            .unwrap();
            let mut want = vec![0.0; heads * hd];
            be.read(&ob, &mut want).unwrap();
            let start = win.map_or(0, |w| n_pos.saturating_sub(w));
            let got = attention(&q, &kc, &vc, heads, kvh, hd, n_pos, start, scale);
            for (a, b) in got.iter().zip(&want) {
                assert!(
                    (a - b).abs() <= 2e-5 * (1.0 + b.abs()),
                    "{heads}/{kvh}/{hd}/{n_pos}: {a} vs {b}"
                );
            }
        }
    }
}
