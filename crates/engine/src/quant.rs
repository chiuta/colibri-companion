//! Block-quantisation formats: geometry, dequantisation and fused dot products.

/// Tensor element types found in GGUF files (numbering follows the public format table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
    /// A type this build does not know; carries the raw id.
    Other(u32),
}

impl DType {
    pub fn from_id(id: u32) -> DType {
        match id {
            0 => DType::F32, 1 => DType::F16, 2 => DType::Q4_0, 3 => DType::Q4_1,
            6 => DType::Q5_0, 7 => DType::Q5_1, 8 => DType::Q8_0,
            10 => DType::Q2K, 11 => DType::Q3K, 12 => DType::Q4K, 13 => DType::Q5K,
            14 => DType::Q6K, 15 => DType::Q8K, 30 => DType::BF16,
            n => DType::Other(n),
        }
    }
    /// (elements per block, bytes per block). `None` for types we cannot size.
    pub fn geometry(self) -> Option<(usize, usize)> {
        Some(match self {
            DType::F32 => (1, 4), DType::F16 | DType::BF16 => (1, 2),
            DType::Q4_0 => (32, 18), DType::Q4_1 => (32, 20),
            DType::Q5_0 => (32, 22), DType::Q5_1 => (32, 24), DType::Q8_0 => (32, 34),
            DType::Q2K => (256, 84), DType::Q3K => (256, 110), DType::Q4K => (256, 144),
            DType::Q5K => (256, 176), DType::Q6K => (256, 210), DType::Q8K => (256, 292),
            DType::Other(_) => return None,
        })
    }
    /// Bytes needed for `n` elements, or `None` if the type is unknown or `n` is not block-aligned.
    pub fn bytes_for(self, n: u64) -> Option<u64> {
        let (e, b) = self.geometry()?;
        if n % e as u64 != 0 { return None; }
        (n / e as u64).checked_mul(b as u64)
    }
    /// Can [`dequantize`] decode this type?
    pub fn decodable(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16 | DType::Q4_0 | DType::Q4_1
            | DType::Q5_0 | DType::Q5_1 | DType::Q8_0 | DType::Q2K | DType::Q3K | DType::Q4K
            | DType::Q5K | DType::Q6K | DType::Q8K)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum QuantError { Unsupported(DType), BadLength { have: usize, need: usize } }

impl std::fmt::Display for QuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantError::Unsupported(t) => write!(f, "tensor type {t:?} is not decodable yet"),
            QuantError::BadLength { have, need } => write!(f, "{have} bytes given, {need} needed"),
        }
    }
}
impl std::error::Error for QuantError {}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 { s << 31 } else {
            // subnormal: normalise
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 { (s << 31) | 0x7f80_0000 | (m << 13) }
    else { (s << 31) | ((e + 127 - 15) << 23) | (m << 13) };
    f32::from_bits(bits)
}

fn rd16(b: &[u8], o: usize) -> f32 { f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]])) }

fn rd32(b: &[u8], o: usize) -> f32 { f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) }

/// The 6-bit scale/min pair `j` (0..8) of the 12-byte packed array used by Q4_K and Q5_K.
pub(crate) fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) }
    else { ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}

/// The sixteen 6-bit scales of a Q3_K block (12 packed bytes), already re-centred by -32.
pub(crate) fn q3k_scales(s: &[u8]) -> [i32; 16] {
    let mut o = [0i32; 16];
    for (i, v) in o.iter_mut().enumerate() {
        let lo = if i < 8 { s[i] & 0xf } else { s[i - 8] >> 4 };
        let hi = (s[8 + (i % 4)] >> (2 * (i / 4))) & 3;
        *v = (lo | (hi << 4)) as i32 - 32;
    }
    o
}

/// Decode `n` elements of type `t` from `src` into `out` (cleared first).
pub fn dequantize(t: DType, src: &[u8], n: usize, out: &mut Vec<f32>) -> Result<(), QuantError> {
    if !t.decodable() { return Err(QuantError::Unsupported(t)); }
    let need = t.bytes_for(n as u64).ok_or(QuantError::Unsupported(t))? as usize;
    if src.len() < need { return Err(QuantError::BadLength { have: src.len(), need }); }
    out.clear();
    out.reserve(n);
    match t {
        DType::F32 => for c in src[..need].chunks_exact(4) { out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]])) },
        DType::F16 => for c in src[..need].chunks_exact(2) { out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]]))) },
        DType::BF16 => for c in src[..need].chunks_exact(2) { out.push(f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)) },
        DType::Q8_0 => for b in src[..need].chunks_exact(34) {
            let d = rd16(b, 0);
            for &q in &b[2..34] { out.push(d * (q as i8) as f32); }
        },
        DType::Q4_0 => for b in src[..need].chunks_exact(18) {
            let d = rd16(b, 0);
            for j in 0..16 { out.push(d * ((b[2 + j] & 0xf) as i32 - 8) as f32); }
            for j in 0..16 { out.push(d * ((b[2 + j] >> 4) as i32 - 8) as f32); }
        },
        DType::Q4_1 => for b in src[..need].chunks_exact(20) {
            let (d, m) = (rd16(b, 0), rd16(b, 2));
            for j in 0..16 { out.push(d * (b[4 + j] & 0xf) as f32 + m); }
            for j in 0..16 { out.push(d * (b[4 + j] >> 4) as f32 + m); }
        },
        DType::Q4K => for b in src[..need].chunks_exact(144) {
            let (d, dmin) = (rd16(b, 0), rd16(b, 2));
            let sc = &b[4..16];
            let qs = &b[16..144];
            for (chunk, is) in (0..4).zip((0..8).step_by(2)) {
                let (s1, m1) = scale_min_k4(is, sc);
                let (s2, m2) = scale_min_k4(is + 1, sc);
                let q = &qs[chunk * 32..chunk * 32 + 32];
                for &v in q { out.push(d * s1 as f32 * (v & 0xf) as f32 - dmin * m1 as f32); }
                for &v in q { out.push(d * s2 as f32 * (v >> 4) as f32 - dmin * m2 as f32); }
            }
        },
        DType::Q6K => for b in src[..need].chunks_exact(210) {
            let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
            let d = rd16(b, 208);
            let mut y = [0f32; 256];
            for half in 0..2 {
                let (ql, qh, sc) = (&ql[half * 64..], &qh[half * 32..], &sc[half * 8..]);
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xf) | ((qh[l] & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0xf) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                    let o = half * 128;
                    y[o + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    y[o + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    y[o + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    y[o + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
            out.extend_from_slice(&y);
        },
        DType::Q5_0 => for b in src[..need].chunks_exact(22) {
            let d = rd16(b, 0);
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            let qs = &b[6..22];
            for (j, &q) in qs.iter().enumerate() { out.push(d * ((q & 0xf) as u32 | (((qh >> j) & 1) << 4)) as i32 as f32 - 16.0 * d); }
            for (j, &q) in qs.iter().enumerate() { out.push(d * ((q >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 as f32 - 16.0 * d); }
        },
        DType::Q5_1 => for b in src[..need].chunks_exact(24) {
            let (d, m) = (rd16(b, 0), rd16(b, 2));
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let qs = &b[8..24];
            for (j, &q) in qs.iter().enumerate() { out.push(d * ((q & 0xf) as u32 | (((qh >> j) & 1) << 4)) as f32 + m); }
            for (j, &q) in qs.iter().enumerate() { out.push(d * ((q >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as f32 + m); }
        },
        DType::Q8K => for b in src[..need].chunks_exact(292) {
            let d = rd32(b, 0);
            for &q in &b[4..260] { out.push(d * (q as i8) as f32); }
        },
        DType::Q2K => for b in src[..need].chunks_exact(84) {
            let (sc, qs) = (&b[0..16], &b[16..80]);
            let (d, dmin) = (rd16(b, 80), rd16(b, 82));
            let mut y = [0f32; 256];
            for n in 0..2 { for j in 0..4 { for half in 0..2 {
                let is = n * 8 + j * 2 + half;
                let (dl, ml) = (d * (sc[is] & 0xf) as f32, dmin * (sc[is] >> 4) as f32);
                for l in 0..16 {
                    let q = (qs[n * 32 + half * 16 + l] >> (2 * j)) & 3;
                    y[n * 128 + j * 32 + half * 16 + l] = dl * q as f32 - ml;
                }
            } } }
            out.extend_from_slice(&y);
        },
        DType::Q3K => for b in src[..need].chunks_exact(110) {
            let (hm, qs) = (&b[0..32], &b[32..96]);
            let sc = q3k_scales(&b[96..108]);
            let d = rd16(b, 108);
            let mut y = [0f32; 256];
            for n in 0..2 { for j in 0..4 { for half in 0..2 {
                let is = n * 8 + j * 2 + half;
                let dl = d * sc[is] as f32;
                let bit = 1u8 << (n * 4 + j);
                for l in 0..16 {
                    let q2 = ((qs[n * 32 + half * 16 + l] >> (2 * j)) & 3) as i32;
                    let v = q2 - if hm[half * 16 + l] & bit != 0 { 0 } else { 4 };
                    y[n * 128 + j * 32 + half * 16 + l] = dl * v as f32;
                }
            } } }
            out.extend_from_slice(&y);
        },
        DType::Q5K => for b in src[..need].chunks_exact(176) {
            let (d, dmin) = (rd16(b, 0), rd16(b, 2));
            let (sc, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
            let mut y = [0f32; 256];
            for c in 0..4 {
                let (s1, m1) = scale_min_k4(2 * c, sc);
                let (s2, m2) = scale_min_k4(2 * c + 1, sc);
                for l in 0..32 {
                    let lo = (qs[c * 32 + l] & 0xf) as u32 + (((qh[l] >> (2 * c)) & 1) as u32) * 16;
                    let hi = (qs[c * 32 + l] >> 4) as u32 + (((qh[l] >> (2 * c + 1)) & 1) as u32) * 16;
                    y[c * 64 + l] = d * s1 as f32 * lo as f32 - dmin * m1 as f32;
                    y[c * 64 + 32 + l] = d * s2 as f32 * hi as f32 - dmin * m2 as f32;
                }
            }
            out.extend_from_slice(&y);
        },
        _ => return Err(QuantError::Unsupported(t)),
    }
    Ok(())
}

/// Dot product of one quantised row with an f32 vector, decoded block by block so the row is
/// never materialised in full (this is what keeps huge models inside a small RAM budget).
pub fn dot_row(t: DType, row: &[u8], x: &[f32]) -> Result<f32, QuantError> {
    let (e, b) = t.geometry().ok_or(QuantError::Unsupported(t))?;
    if !t.decodable() { return Err(QuantError::Unsupported(t)); }
    let need = t.bytes_for(x.len() as u64).ok_or(QuantError::Unsupported(t))? as usize;
    if row.len() < need { return Err(QuantError::BadLength { have: row.len(), need }); }
    let mut acc = 0f32;
    let mut buf = Vec::with_capacity(e.max(256));
    for (blk, xs) in row[..need].chunks_exact(b).zip(x.chunks(e)) {
        dequantize(t, blk, xs.len(), &mut buf)?;
        acc += buf.iter().zip(xs).map(|(a, b)| a * b).sum::<f32>();
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_round_trip_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12); // smallest subnormal
    }

    #[test]
    fn sizes_match_the_block_tables() {
        assert_eq!(DType::Q4K.bytes_for(512), Some(288));
        assert_eq!(DType::Q6K.bytes_for(256), Some(210));
        assert_eq!(DType::Q8_0.bytes_for(64), Some(68));
        assert_eq!(DType::Q4_0.bytes_for(33), None);
        assert_eq!(DType::Other(99).bytes_for(32), None);
    }

    #[test]
    fn q8_0_and_q4_0_decode() {
        let mut b = vec![0u8; 34];
        b[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        for i in 0..32 { b[2 + i] = (i as i8 - 16) as u8; }
        let mut o = vec![];
        dequantize(DType::Q8_0, &b, 32, &mut o).unwrap();
        assert_eq!(o[0], -16.0);
        assert_eq!(o[31], 15.0);

        let mut b = vec![0u8; 18];
        b[..2].copy_from_slice(&0x4000u16.to_le_bytes()); // d = 2
        b[2] = 0xf1; // low nibble 1, high nibble 15
        dequantize(DType::Q4_0, &b, 32, &mut o).unwrap();
        assert_eq!(o[0], 2.0 * (1 - 8) as f32);
        assert_eq!(o[16], 2.0 * (15 - 8) as f32);
    }

    #[test]
    fn q4k_scales_unpack_for_all_eight_groups() {
        // scales 1..=8, mins 0; every nibble 1; d = 1  -> output of group g is (g+1)
        let mut b = vec![0u8; 144];
        b[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        for j in 0..4 { b[4 + j] = (j + 1) as u8; }            // scales 0..3
        for j in 4..8 { b[4 + j + 4] = ((j + 1) as u8) & 0xf; } // low 4 bits of scales 4..7 in bytes 8..11
        for q in &mut b[16..] { *q = 0x11; }
        let mut o = vec![];
        dequantize(DType::Q4K, &b, 256, &mut o).unwrap();
        for g in 0..8 { assert_eq!(o[g * 32], (g + 1) as f32, "group {g}"); }
    }

    #[test]
    fn q6k_decodes_a_constant_block() {
        let mut b = vec![0u8; 210];
        for v in &mut b[..128] { *v = 0x22; }    // low and high nibbles = 2
        for v in &mut b[192..208] { *v = 1; }    // scales 1
        b[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        let mut o = vec![];
        dequantize(DType::Q6K, &b, 256, &mut o).unwrap();
        assert!(o.iter().all(|&v| v == (2 - 32) as f32));
    }

    #[test]
    fn dot_matches_dequantise_then_multiply() {
        let mut row = vec![0u8; 34 * 3];
        for blk in row.chunks_mut(34) {
            blk[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // 0.5
            for (i, v) in blk[2..].iter_mut().enumerate() { *v = (i as i8 - 7) as u8; }
        }
        let x: Vec<f32> = (0..96).map(|i| (i as f32).sin()).collect();
        let mut full = vec![];
        dequantize(DType::Q8_0, &row, 96, &mut full).unwrap();
        let want: f32 = full.iter().zip(&x).map(|(a, b)| a * b).sum();
        let got = dot_row(DType::Q8_0, &row, &x).unwrap();
        assert!((want - got).abs() < 1e-4);
    }

    #[test]
    fn short_or_unknown_input_is_an_error_not_a_panic() {
        let mut o = vec![];
        assert!(matches!(dequantize(DType::Q4K, &[0; 10], 256, &mut o), Err(QuantError::BadLength { .. })));
        assert!(matches!(dequantize(DType::Other(5), &[0; 84], 256, &mut o), Err(QuantError::Unsupported(_))));
        assert!(dot_row(DType::Other(77), &[], &[]).is_err());
    }
}

/// Test-only reference encoders, written from the quantiser's point of view (value -> bits).
/// They are deliberately independent of the decoders so a round trip checks the bit layouts.
#[cfg(test)]
pub(crate) mod testenc {
    use super::*;

    pub fn f32_to_f16(x: f32) -> u16 {
        let b = x.to_bits();
        let s = ((b >> 16) & 0x8000) as u16;
        let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
        if x == 0.0 || e <= 0 { return s; }
        if e >= 31 { return s | 0x7bff; }
        let m = ((b & 0x7f_ffff) + 0x1000) >> 13; // round to nearest; a carry bumps the exponent
        s | (((e as u32) << 10) + m) as u16
    }
    pub fn r16(x: f32) -> f32 { f16_to_f32(f32_to_f16(x)) }
    fn put16(v: &mut [u8], o: usize, x: f32) { v[o..o + 2].copy_from_slice(&f32_to_f16(x).to_le_bytes()); }
    fn amax_signed(x: &[f32]) -> f32 { x.iter().copied().fold(0.0, |a: f32, v| if v.abs() > a.abs() { v } else { a }) }
    fn rnd(x: f32) -> i32 { x.round() as i32 }

    fn enc_q8_0(x: &[f32]) -> Vec<u8> {
        let d = r16(amax_signed(x).abs() / 127.0);
        let mut b = vec![0u8; 34];
        put16(&mut b, 0, d);
        for (i, v) in x.iter().enumerate() { b[2 + i] = if d == 0.0 { 0 } else { rnd(v / d).clamp(-127, 127) as i8 as u8 }; }
        b
    }
    fn enc_sym4(x: &[f32]) -> Vec<u8> { // Q4_0
        let d = r16(amax_signed(x) / -8.0);
        let mut b = vec![0u8; 18];
        put16(&mut b, 0, d);
        let q = |v: f32| if d == 0.0 { 8 } else { (rnd(v / d) + 8).clamp(0, 15) as u8 };
        for j in 0..16 { b[2 + j] = q(x[j]) | (q(x[j + 16]) << 4); }
        b
    }
    fn enc_q4_1(x: &[f32]) -> Vec<u8> {
        let (mn, mx) = x.iter().fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        let (d, m) = (r16((mx - mn) / 15.0), r16(mn));
        let mut b = vec![0u8; 20];
        put16(&mut b, 0, d); put16(&mut b, 2, m);
        let q = |v: f32| if d == 0.0 { 0 } else { rnd((v - m) / d).clamp(0, 15) as u8 };
        for j in 0..16 { b[4 + j] = q(x[j]) | (q(x[j + 16]) << 4); }
        b
    }
    fn enc_q5_0(x: &[f32]) -> Vec<u8> {
        let d = r16(amax_signed(x) / -16.0);
        let mut b = vec![0u8; 22];
        put16(&mut b, 0, d);
        let q = |v: f32| if d == 0.0 { 16 } else { (rnd(v / d) + 16).clamp(0, 31) as u32 };
        let mut qh = 0u32;
        for j in 0..16 {
            let (a, c) = (q(x[j]), q(x[j + 16]));
            b[6 + j] = ((a & 0xf) | ((c & 0xf) << 4)) as u8;
            qh |= ((a >> 4) & 1) << j;
            qh |= ((c >> 4) & 1) << (j + 16);
        }
        b[2..6].copy_from_slice(&qh.to_le_bytes());
        b
    }
    fn enc_q5_1(x: &[f32]) -> Vec<u8> {
        let (mn, mx) = x.iter().fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
        let (d, m) = (r16((mx - mn) / 31.0), r16(mn));
        let mut b = vec![0u8; 24];
        put16(&mut b, 0, d); put16(&mut b, 2, m);
        let q = |v: f32| if d == 0.0 { 0 } else { rnd((v - m) / d).clamp(0, 31) as u32 };
        let mut qh = 0u32;
        for j in 0..16 {
            let (a, c) = (q(x[j]), q(x[j + 16]));
            b[8 + j] = ((a & 0xf) | ((c & 0xf) << 4)) as u8;
            qh |= ((a >> 4) & 1) << j;
            qh |= ((c >> 4) & 1) << (j + 16);
        }
        b[4..8].copy_from_slice(&qh.to_le_bytes());
        b
    }
    fn enc_q8k(x: &[f32]) -> Vec<u8> {
        let d = amax_signed(x).abs() / 127.0;
        let mut b = vec![0u8; 292];
        b[0..4].copy_from_slice(&d.to_le_bytes());
        let mut sums = [0i16; 16];
        for (i, v) in x.iter().enumerate() {
            let q = if d == 0.0 { 0 } else { rnd(v / d).clamp(-127, 127) };
            b[4 + i] = q as i8 as u8;
            sums[i / 16] += q as i16;
        }
        for (g, s) in sums.iter().enumerate() { b[260 + 2 * g..262 + 2 * g].copy_from_slice(&s.to_le_bytes()); }
        b
    }

    /// Affine super-block quantiser shared by Q2_K / Q4_K / Q5_K:
    /// returns (d, dmin, scale codes, min codes, quants).
    fn affine(x: &[f32], gsize: usize, maxq: u32, sbits: u32) -> (f32, f32, Vec<u32>, Vec<u32>, Vec<u32>) {
        let ng = x.len() / gsize;
        let (mut scs, mut mns) = (vec![0f32; ng], vec![0f32; ng]);
        for g in 0..ng {
            let xs = &x[g * gsize..(g + 1) * gsize];
            let (mn, mx) = xs.iter().fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
            mns[g] = (-mn).max(0.0);
            scs[g] = (mx + mns[g]) / maxq as f32;
        }
        let lim = ((1u32 << sbits) - 1) as f32;
        let (smax, mmax) = (scs.iter().cloned().fold(0.0, f32::max), mns.iter().cloned().fold(0.0, f32::max));
        let d = r16(if smax > 0.0 { smax / lim } else { 1.0 });
        let dmin = r16(if mmax > 0.0 { mmax / lim } else { 1.0 });
        let sci: Vec<u32> = scs.iter().map(|s| rnd(s / d).clamp(0, lim as i32) as u32).collect();
        let mi: Vec<u32> = mns.iter().map(|m| rnd(m / dmin).clamp(0, lim as i32) as u32).collect();
        let q = x.iter().enumerate().map(|(i, &v)| {
            let g = i / gsize;
            let es = d * sci[g] as f32;
            if es == 0.0 { 0 } else { rnd((v + dmin * mi[g] as f32) / es).clamp(0, maxq as i32) as u32 }
        }).collect();
        (d, dmin, sci, mi, q)
    }
    /// Inverse of `scale_min_k4` for 8 six-bit scales and 8 six-bit mins.
    fn pack_k4(sc: &[u32], mn: &[u32]) -> [u8; 12] {
        let mut o = [0u8; 12];
        for j in 0..4 {
            o[j] = (sc[j] | ((sc[j + 4] >> 4) << 6)) as u8;
            o[j + 4] = (mn[j] | ((mn[j + 4] >> 4) << 6)) as u8;
            o[j + 8] = ((sc[j + 4] & 0xf) | ((mn[j + 4] & 0xf) << 4)) as u8;
        }
        o
    }
    fn enc_q2k(x: &[f32]) -> Vec<u8> {
        let (d, dmin, sc, mn, q) = affine(x, 16, 3, 4);
        let mut b = vec![0u8; 84];
        for g in 0..16 { b[g] = (sc[g] | (mn[g] << 4)) as u8; }
        for n in 0..2 { for j in 0..4 { for half in 0..2 { for l in 0..16 {
            b[16 + n * 32 + half * 16 + l] |= (q[n * 128 + j * 32 + half * 16 + l] as u8) << (2 * j);
        } } } }
        put16(&mut b, 80, d); put16(&mut b, 82, dmin);
        b
    }
    fn enc_q4k(x: &[f32]) -> Vec<u8> {
        let (d, dmin, sc, mn, q) = affine(x, 32, 15, 6);
        let mut b = vec![0u8; 144];
        put16(&mut b, 0, d); put16(&mut b, 2, dmin);
        b[4..16].copy_from_slice(&pack_k4(&sc, &mn));
        for c in 0..4 { for l in 0..32 { b[16 + c * 32 + l] = (q[c * 64 + l] | (q[c * 64 + 32 + l] << 4)) as u8; } }
        b
    }
    fn enc_q5k(x: &[f32]) -> Vec<u8> {
        let (d, dmin, sc, mn, q) = affine(x, 32, 31, 6);
        let mut b = vec![0u8; 176];
        put16(&mut b, 0, d); put16(&mut b, 2, dmin);
        b[4..16].copy_from_slice(&pack_k4(&sc, &mn));
        for c in 0..4 { for l in 0..32 {
            let (a, h) = (q[c * 64 + l], q[c * 64 + 32 + l]);
            b[48 + c * 32 + l] = ((a & 0xf) | ((h & 0xf) << 4)) as u8;
            b[16 + l] |= (((a >> 4) & 1) << (2 * c) | ((h >> 4) & 1) << (2 * c + 1)) as u8;
        } }
        b
    }
    /// Symmetric 16-groups with a signed super-scale: shared by Q3_K (range ±4) and Q6_K (±32).
    fn symmetric(x: &[f32], half_range: i32, slim: i32) -> (f32, Vec<i32>, Vec<i32>) {
        let sc: Vec<f32> = x.chunks(16).map(|g| amax_signed(g) / -(half_range as f32)).collect();
        let smax = sc.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = r16(if smax > 0.0 { smax / slim as f32 } else { 1.0 });
        let sci: Vec<i32> = sc.iter().map(|s| rnd(s / d).clamp(-slim - 1, slim)).collect();
        let q = x.iter().enumerate().map(|(i, &v)| {
            let es = d * sci[i / 16] as f32;
            if es == 0.0 { 0 } else { rnd(v / es).clamp(-half_range, half_range - 1) }
        }).collect();
        (d, sci, q)
    }
    fn enc_q3k(x: &[f32]) -> Vec<u8> {
        let (d, sci, q) = symmetric(x, 4, 31);
        let mut b = vec![0u8; 110];
        let u: Vec<u32> = sci.iter().map(|s| (s + 32) as u32).collect();
        for i in 0..16 {
            if i < 8 { b[96 + i] |= (u[i] & 0xf) as u8; } else { b[96 + i - 8] |= ((u[i] & 0xf) << 4) as u8; }
            b[96 + 8 + (i % 4)] |= (((u[i] >> 4) & 3) << (2 * (i / 4))) as u8;
        }
        for n in 0..2 { for j in 0..4 { for half in 0..2 { for l in 0..16 {
            let v = q[n * 128 + j * 32 + half * 16 + l];
            let (low, hb) = if v >= 0 { (v, 1u8) } else { (v + 4, 0u8) };
            b[32 + n * 32 + half * 16 + l] |= (low as u8) << (2 * j);
            b[half * 16 + l] |= hb << (n * 4 + j);
        } } } }
        put16(&mut b, 108, d);
        b
    }
    fn enc_q6k(x: &[f32]) -> Vec<u8> {
        let (d, sci, q) = symmetric(x, 32, 127);
        let mut b = vec![0u8; 210];
        for (g, s) in sci.iter().enumerate() { b[192 + g] = *s as i8 as u8; }
        for half in 0..2 { for l in 0..32 {
            let v = |k: usize| (q[half * 128 + l + 32 * k] + 32) as u8;
            b[half * 64 + l] = (v(0) & 0xf) | ((v(2) & 0xf) << 4);
            b[half * 64 + l + 32] = (v(1) & 0xf) | ((v(3) & 0xf) << 4);
            b[128 + half * 32 + l] = (v(0) >> 4) | ((v(1) >> 4) << 2) | ((v(2) >> 4) << 4) | ((v(3) >> 4) << 6);
        } }
        put16(&mut b, 208, d);
        b
    }

    /// Encode `x` (a whole number of blocks) as type `t`.
    pub fn encode(t: DType, x: &[f32]) -> Vec<u8> {
        let (e, _) = t.geometry().unwrap();
        assert_eq!(x.len() % e, 0);
        match t {
            DType::F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DType::F16 => x.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect(),
            DType::BF16 => x.iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect(),
            _ => {
                let f: fn(&[f32]) -> Vec<u8> = match t {
                    DType::Q8_0 => enc_q8_0, DType::Q4_0 => enc_sym4, DType::Q4_1 => enc_q4_1, DType::Q5_0 => enc_q5_0,
                    DType::Q5_1 => enc_q5_1, DType::Q8K => enc_q8k, DType::Q2K => enc_q2k, DType::Q3K => enc_q3k,
                    DType::Q4K => enc_q4k, DType::Q5K => enc_q5k, DType::Q6K => enc_q6k, _ => panic!("no encoder for {t:?}"),
                };
                x.chunks(e).flat_map(f).collect()
            }
        }
    }

    /// Deterministic test signal: smooth + noisy, roughly in [-1, 1].
    pub fn signal(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n).map(|i| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let noise = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.8;
            0.6 * (i as f32 * 0.07).sin() + noise
        }).collect()
    }
}

#[cfg(test)]
mod more_tests {
    use super::testenc::*;
    use super::*;

    fn rel_rmse(a: &[f32], b: &[f32]) -> f32 {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = a.iter().map(|x| x * x).sum();
        (num / den).sqrt()
    }

    #[test]
    fn f16_encoder_matches_the_decoder() {
        for v in [1.0f32, -2.5, 0.333, 1000.0, 6.2e-5, -0.0078125, 65000.0] {
            assert!(((r16(v) - v) / v).abs() < 1e-3, "{v}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
    }

    #[test]
    fn every_type_round_trips_within_its_precision() {
        // (type, max relative RMS error for a smooth+noisy signal)
        for (t, bound) in [(DType::F32, 1e-7), (DType::F16, 1e-3), (DType::BF16, 1e-2), (DType::Q8_0, 0.01), (DType::Q8K, 0.01),
            (DType::Q6K, 0.03), (DType::Q5_0, 0.05), (DType::Q5_1, 0.05), (DType::Q5K, 0.06), (DType::Q4_0, 0.10),
            (DType::Q4_1, 0.10), (DType::Q4K, 0.12), (DType::Q3K, 0.25), (DType::Q2K, 0.55)] {
            assert!(t.decodable(), "{t:?}");
            let x = signal(1024, 7);
            let bytes = encode(t, &x);
            assert_eq!(bytes.len() as u64, t.bytes_for(1024).unwrap(), "{t:?} size");
            let mut y = vec![];
            dequantize(t, &bytes, 1024, &mut y).unwrap();
            let e = rel_rmse(&x, &y);
            assert!(e <= bound, "{t:?}: relative error {e} > {bound}");
            assert!(e > 0.0 || matches!(t, DType::F32), "{t:?} should not be lossless");
        }
    }

    #[test]
    fn higher_precision_types_really_are_more_precise() {
        let x = signal(1024, 99);
        let err = |t| { let mut y = vec![]; dequantize(t, &encode(t, &x), 1024, &mut y).unwrap(); rel_rmse(&x, &y) };
        assert!(err(DType::Q8_0) < err(DType::Q5_0) && err(DType::Q5_0) < err(DType::Q4_0));
        assert!(err(DType::Q6K) < err(DType::Q5K) && err(DType::Q5K) < err(DType::Q4K));
        assert!(err(DType::Q4K) < err(DType::Q3K) && err(DType::Q3K) < err(DType::Q2K));
    }

    // Position probes: one non-zero quant at a spot taken from the reference layout, d = 1.

    #[test]
    fn q5_0_fifth_bits_land_on_the_right_elements() {
        let mut b = vec![0u8; 22];
        b[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        // all quants 16 (=> 0) except: element 3 gets bit 4 via qh bit 3, element 20 via qh bit 20
        for j in 0..16 { b[6 + j] = 0; }
        let qh: u32 = (1 << 3) | (1 << 20);
        b[2..6].copy_from_slice(&qh.to_le_bytes());
        let mut o = vec![];
        dequantize(DType::Q5_0, &b, 32, &mut o).unwrap();
        for (i, v) in o.iter().enumerate() {
            let want = if i == 3 || i == 20 { 0.0 } else { -16.0 };
            assert_eq!(*v, want, "element {i}");
        }
        // high nibble of byte 5 is element 21
        b[2..6].copy_from_slice(&0u32.to_le_bytes());
        b[6 + 5] = 0x70;
        dequantize(DType::Q5_0, &b, 32, &mut o).unwrap();
        assert_eq!(o[21], 7.0 - 16.0);
        assert_eq!(o[5], -16.0);
    }

    #[test]
    fn q5_1_uses_offset_and_fifth_bit() {
        let mut b = vec![0u8; 24];
        b[0..2].copy_from_slice(&0x4000u16.to_le_bytes()); // d = 2
        b[2..4].copy_from_slice(&0x3c00u16.to_le_bytes()); // m = 1
        b[4..8].copy_from_slice(&(1u32 << 17).to_le_bytes());
        b[8 + 1] = 0x03; // element 1 = 3
        let mut o = vec![];
        dequantize(DType::Q5_1, &b, 32, &mut o).unwrap();
        assert_eq!(o[1], 2.0 * 3.0 + 1.0);
        assert_eq!(o[17], 2.0 * 16.0 + 1.0);
        assert_eq!(o[0], 1.0);
    }

    #[test]
    fn q8_k_scales_by_its_f32_d() {
        let mut b = vec![0u8; 292];
        b[0..4].copy_from_slice(&0.5f32.to_le_bytes());
        b[4] = 10; b[4 + 255] = (-4i8) as u8;
        let mut o = vec![];
        dequantize(DType::Q8K, &b, 256, &mut o).unwrap();
        assert_eq!((o[0], o[255], o[100]), (5.0, -2.0, 0.0));
    }

    #[test]
    fn q2_k_quant_position_scale_and_min() {
        let mut b = vec![0u8; 84];
        b[80..82].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        b[82..84].copy_from_slice(&0x4000u16.to_le_bytes()); // dmin = 2
        // group index 2*4+... : n=1, j=2, half=0  => is = 8 + 4 + 0 = 12 ; scale 5, min 3
        b[12] = 5 | (3 << 4);
        // quant 3 at byte n*32 + half*16 + l = 40 (l = 8), shift 2*j = 4
        b[16 + 40] = 3 << 4;
        let mut o = vec![];
        dequantize(DType::Q2K, &b, 256, &mut o).unwrap();
        let idx = 128 + 2 * 32 + 8; // 200
        assert_eq!(o[idx], 5.0 * 3.0 - 2.0 * 3.0);
        assert_eq!(o[idx + 1], 5.0 * 0.0 - 6.0); // same group, quant 0: only the min shows
        assert_eq!(o[idx + 16], 0.0); // next group, scale 0 min 0
    }

    #[test]
    fn q3_k_high_mask_scale_unpacking_and_position() {
        let mut b = vec![0u8; 110];
        b[108..110].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        // scales: all raw 32 (=> 0) except group 13 = raw 32 + 7. Raw 39 = 0b100111:
        // low nibble 7 sits in the HIGH nibble of byte 5 (i-8 = 5), bits 5..4 = 10 in byte 8+1 at shift 2*3
        let mut sc = [0u8; 12];
        for i in 0..16usize {
            let raw: u32 = if i == 13 { 39 } else { 32 };
            if i < 8 { sc[i] |= (raw & 0xf) as u8 } else { sc[i - 8] |= ((raw & 0xf) << 4) as u8 }
            sc[8 + i % 4] |= (((raw >> 4) & 3) << (2 * (i / 4))) as u8;
        }
        b[96..108].copy_from_slice(&sc);
        assert_eq!(q3k_scales(&sc)[13], 7);
        assert_eq!(q3k_scales(&sc)[12], 0);
        // group 13 = n 1, j 2, half 1 (13 = 8 + 4 + 1): elements 128+64+16 .. +16
        // hmask bit (n*4+j)=6 set for l=2 (mask index half*16+l = 18); low2 = 3 at byte 32 + 1*32 + 16 + 2, shift 4
        b[18] |= 1 << 6;
        b[32 + 32 + 16 + 2] |= 3 << 4;
        let mut o = vec![];
        dequantize(DType::Q3K, &b, 256, &mut o).unwrap();
        let idx = 128 + 64 + 16 + 2;
        assert_eq!(o[idx], 7.0 * 3.0); // hbit set: value 3
        assert_eq!(o[idx + 1], 7.0 * -4.0); // hbit clear, low2 0: value -4
    }

    #[test]
    fn q5_k_fifth_bit_and_scale_min() {
        let mut b = vec![0u8; 176];
        b[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        b[2..4].copy_from_slice(&0x3c00u16.to_le_bytes());
        // chunk c=2: low-nibble half uses scale index 4, high-nibble half index 5 (indices >= 4: packed split form)
        let sc = pack_scales(&[1, 1, 1, 1, 9, 6, 1, 1], &[0, 0, 0, 0, 2, 4, 0, 0]);
        b[4..16].copy_from_slice(&sc);
        // low half element l=7 of chunk 2 = (qs&0xf)=5 + fifth bit (qh[7] bit 4) ; high half l=7: (qs>>4)=1 + fifth bit (qh[7] bit 5)
        b[48 + 2 * 32 + 7] = 0x15;
        b[16 + 7] = (1 << 4) | (1 << 5);
        let mut o = vec![];
        dequantize(DType::Q5K, &b, 256, &mut o).unwrap();
        assert_eq!(o[2 * 64 + 7], 9.0 * (5.0 + 16.0) - 2.0);
        assert_eq!(o[2 * 64 + 32 + 7], 6.0 * (1.0 + 16.0) - 4.0);
        assert_eq!(o[2 * 64 + 6], -2.0);
    }

    fn pack_scales(sc: &[u32; 8], mn: &[u32; 8]) -> [u8; 12] {
        let mut o = [0u8; 12];
        for j in 0..4 {
            o[j] = (sc[j] | ((sc[j + 4] >> 4) << 6)) as u8;
            o[j + 4] = (mn[j] | ((mn[j + 4] >> 4) << 6)) as u8;
            o[j + 8] = ((sc[j + 4] & 0xf) | ((mn[j + 4] & 0xf) << 4)) as u8;
        }
        o
    }

    #[test]
    fn dot_row_works_for_every_decodable_type() {
        for t in [DType::Q5_0, DType::Q5_1, DType::Q2K, DType::Q3K, DType::Q5K, DType::Q8K, DType::Q6K] {
            let x = signal(512, 3);
            let w = encode(t, &x);
            let v: Vec<f32> = signal(512, 5);
            let mut full = vec![];
            dequantize(t, &w, 512, &mut full).unwrap();
            let want: f32 = full.iter().zip(&v).map(|(a, b)| a * b).sum();
            assert!((dot_row(t, &w, &v).unwrap() - want).abs() < 1e-3, "{t:?}");
        }
    }
}
