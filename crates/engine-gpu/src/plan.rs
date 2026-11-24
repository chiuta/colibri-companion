//! Dispatch and buffer arithmetic shared by the runner and the tests.

pub const MAX_DIM: u64 = 65_535;

/// Split `n` workgroups over x and y so neither exceeds the per-dimension limit. Kernels rebuild
/// the linear index as `x + y * nwgx` and ignore indices past their data, so the last row of
/// workgroups may overshoot harmlessly. Returns `(x, y, nwgx)`.
pub fn dispatch_dims(n: u64) -> Option<(u32, u32, u32)> {
    if n == 0 { return Some((1, 1, 1)); }
    let x = n.min(MAX_DIM);
    let y = n.div_ceil(x);
    if y > MAX_DIM { return None; }
    Some((x as u32, y as u32, x as u32))
}

/// Split a matrix into row ranges whose bytes each fit one storage-buffer binding.
pub fn chunk_rows(rows: usize, row_bytes: u64, max_binding: u64) -> Option<Vec<(usize, usize)>> {
    if rows == 0 { return Some(vec![]); }
    if row_bytes == 0 || row_bytes > max_binding { return None; }
    let per = ((max_binding / row_bytes) as usize).max(1);
    Some((0..rows).step_by(per).map(|s| (s, per.min(rows - s))).collect())
}

/// `cos`/`sin` for `pos` over `head_dim / 2` frequencies, angles in f64 (exact for any position)
/// and rounded once to f32, exactly as the CPU kernel does.
pub fn rope_tables(pos: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    (0..half).map(|i| {
        let freq = theta.powf(-(2.0 * i as f64) / head_dim as f64);
        let (s, c) = (pos as f64 * freq).sin_cos();
        (c as f32, s as f32)
    }).unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_covers_every_workgroup_without_exceeding_limits() {
        for n in [0u64, 1, 65_535, 65_536, 200_000, 4_000_000_000] {
            let (x, y, nwgx) = dispatch_dims(n).unwrap();
            assert!(x as u64 <= MAX_DIM && y as u64 <= MAX_DIM);
            assert_eq!(nwgx, x);
            assert!(x as u64 * y as u64 >= n);
            assert!(n == 0 || (x as u64 * y as u64) < n + x as u64, "at most one row of overshoot");
        }
        assert_eq!(dispatch_dims(MAX_DIM * MAX_DIM + 1), None);
    }

    #[test]
    fn chunks_tile_the_matrix_and_respect_the_binding_limit() {
        let c = chunk_rows(1000, 100, 25_050).unwrap();
        assert_eq!(c.iter().map(|c| c.1).sum::<usize>(), 1000);
        assert!(c.iter().all(|&(_, n)| n as u64 * 100 <= 25_050));
        assert!(c.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0));
        assert_eq!(chunk_rows(5, 100, 99), None, "a single row larger than a binding cannot be uploaded");
        assert_eq!(chunk_rows(0, 100, 10).unwrap(), vec![]);
    }

    #[test]
    fn rope_tables_agree_with_the_cpu_rotation() {
        use companion_engine::ops;
        for pos in [0usize, 1, 37, 100_000] {
            let base: Vec<f32> = (0..16).map(|i| (i as f32 * 0.3).sin()).collect();
            let mut want = base.clone();
            ops::rope(&mut want, pos, 10_000.0);
            let (c, s) = rope_tables(pos, 16, 10_000.0);
            let mut got = base.clone();
            for i in 0..8 { got[i] = base[i] * c[i] - base[i + 8] * s[i]; got[i + 8] = base[i] * s[i] + base[i + 8] * c[i]; }
            for (a, b) in got.iter().zip(&want) { assert!((a - b).abs() < 1e-5, "pos {pos}: {a} vs {b}"); }
        }
    }
}
