//! GPU weight layouts. Every format is stored as little-endian `u32` words; scales that the CPU
//! keeps as f16 are kept as f16 only where the block is already word-aligned (Q4_K, Q6_K) and are
//! widened to f32 otherwise (Q8_0, Q4_0), so a shader never needs unaligned reads.
use companion_engine::quant::DType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format { F32, F16, Q8_0, Q4_0, Q4K, Q6K }

pub const ALL: [Format; 6] = [Format::F32, Format::F16, Format::Q8_0, Format::Q4_0, Format::Q4K, Format::Q6K];

#[derive(Debug, PartialEq, Eq)]
pub enum LayoutError { Unsupported(DType), Size { have: usize, need: usize } }
impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { LayoutError::Unsupported(t) => write!(f, "{t:?} has no GPU kernel"), LayoutError::Size { have, need } => write!(f, "{have} bytes given, {need} needed") }
    }
}
impl std::error::Error for LayoutError {}

impl Format {
    pub fn from_dtype(t: DType) -> Option<Format> {
        Some(match t { DType::F32 => Format::F32, DType::F16 => Format::F16, DType::Q8_0 => Format::Q8_0, DType::Q4_0 => Format::Q4_0, DType::Q4K => Format::Q4K, DType::Q6K => Format::Q6K, _ => return None })
    }
    /// Elements covered by one stored block (1 for the float formats).
    pub fn block_elems(self) -> usize { match self { Format::F32 | Format::F16 => 1, Format::Q8_0 | Format::Q4_0 => 32, Format::Q4K | Format::Q6K => 256 } }
    pub fn block_words(self) -> usize { match self { Format::F32 => 1, Format::F16 => 1, Format::Q8_0 => 9, Format::Q4_0 => 5, Format::Q4K => 36, Format::Q6K => 53 } }
    /// Independent work items per block: what one shader lane processes at a time.
    pub fn items_per_block(self) -> usize { match self { Format::Q8_0 | Format::Q4_0 => 1, Format::Q4K => 8, Format::Q6K => 16, _ => 1 } }
    pub fn name(self) -> &'static str { match self { Format::F32 => "f32", Format::F16 => "f16", Format::Q8_0 => "q8_0", Format::Q4_0 => "q4_0", Format::Q4K => "q4_k", Format::Q6K => "q6_k" } }

    /// 32-bit words per matrix row.
    pub fn row_words(self, cols: usize) -> usize {
        match self { Format::F32 => cols, Format::F16 => cols.div_ceil(2), f => cols / f.block_elems() * f.block_words() }
    }
    /// Work items per row (`ceil(cols/32)` chunks for the float formats).
    pub fn row_items(self, cols: usize) -> usize {
        match self { Format::F32 | Format::F16 => cols.div_ceil(32), f => cols / f.block_elems() * f.items_per_block() }
    }
    /// Is `cols` a legal row length for this format?
    pub fn accepts_cols(self, cols: usize) -> bool { cols > 0 && (matches!(self, Format::F32 | Format::F16) || cols.is_multiple_of(self.block_elems())) }
}

fn words_le(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| { let mut w = [0u8; 4]; w[..c.len()].copy_from_slice(c); u32::from_le_bytes(w) }).collect()
}
fn f16(b: &[u8]) -> f32 { companion_engine::quant::f16_to_f32(u16::from_le_bytes([b[0], b[1]])) }

/// Repack `rows × cols` elements of the CPU byte layout `t` into GPU words.
pub fn repack(t: DType, rows: usize, cols: usize, bytes: &[u8]) -> Result<Vec<u32>, LayoutError> {
    let fmt = Format::from_dtype(t).ok_or(LayoutError::Unsupported(t))?;
    if !fmt.accepts_cols(cols) { return Err(LayoutError::Unsupported(t)); }
    let row_bytes = t.bytes_for(cols as u64).ok_or(LayoutError::Unsupported(t))? as usize;
    let need = row_bytes.checked_mul(rows).ok_or(LayoutError::Size { have: bytes.len(), need: usize::MAX })?;
    if bytes.len() != need { return Err(LayoutError::Size { have: bytes.len(), need }); }
    let mut out = Vec::with_capacity(rows * fmt.row_words(cols));
    for row in bytes.chunks_exact(row_bytes) {
        match fmt {
            Format::F32 | Format::F16 => out.extend(words_le(row)),
            Format::Q8_0 => for b in row.chunks_exact(34) {
                out.push(f16(b).to_bits());
                out.extend(words_le(&b[2..34]));
            },
            Format::Q4_0 => for b in row.chunks_exact(18) {
                out.push(f16(b).to_bits());
                out.extend(words_le(&b[2..18]));
            },
            Format::Q4K => out.extend(words_le(row)),
            Format::Q6K => for b in row.chunks_exact(210) {
                let mut padded = [0u8; 212];
                padded[..210].copy_from_slice(b);
                out.extend(words_le(&padded));
            },
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_matches_the_cpu_formats() {
        for (t, f) in [(DType::Q8_0, Format::Q8_0), (DType::Q4_0, Format::Q4_0), (DType::Q4K, Format::Q4K), (DType::Q6K, Format::Q6K)] {
            let (e, b) = t.geometry().unwrap();
            assert_eq!(f.block_elems(), e);
            assert!(f.block_words() * 4 >= b, "{f:?}: GPU block must hold the CPU block");
        }
        assert_eq!(Format::Q4K.block_words() * 4, 144, "Q4_K is already word aligned and kept raw");
        assert_eq!(Format::Q6K.block_words() * 4, 212);
        assert_eq!(Format::F16.row_words(5), 3);
        assert_eq!(Format::Q4K.row_items(512), 16);
        assert_eq!(Format::F32.row_items(33), 2);
    }

    #[test]
    fn repack_sizes_and_errors() {
        let w = repack(DType::Q8_0, 2, 64, &[0u8; 2 * 2 * 34]).unwrap();
        assert_eq!(w.len(), 2 * Format::Q8_0.row_words(64));
        let w = repack(DType::Q6K, 3, 256, &[0u8; 3 * 210]).unwrap();
        assert_eq!(w.len(), 3 * 53);
        assert!(matches!(repack(DType::Q4K, 1, 256, &[0; 10]), Err(LayoutError::Size { .. })));
        assert!(matches!(repack(DType::Q5K, 1, 256, &[0; 176]), Err(LayoutError::Unsupported(_))));
        assert!(repack(DType::Q4_0, 1, 33, &[0; 100]).is_err());
        assert!(repack(DType::F32, usize::MAX, 4, &[]).is_err());
    }

    #[test]
    fn widened_scales_are_exact() {
        let mut b = vec![0u8; 34];
        b[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // 1.0
        b[2] = 0xfe; // -2 as i8
        let w = repack(DType::Q8_0, 1, 32, &b).unwrap();
        assert_eq!(f32::from_bits(w[0]), 1.0);
        assert_eq!(w[1] & 0xff, 0xfe);
    }
}
