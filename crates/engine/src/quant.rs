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
            | DType::Q8_0 | DType::Q4K | DType::Q6K)
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

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) }
    else { ((q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
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
        assert!(matches!(dequantize(DType::Q2K, &[0; 84], 256, &mut o), Err(QuantError::Unsupported(_))));
        assert!(dot_row(DType::Other(77), &[], &[]).is_err());
    }
}
