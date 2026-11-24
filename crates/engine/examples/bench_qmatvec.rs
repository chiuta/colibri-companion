//! `cargo run --release -p companion-engine --example bench_qmatvec [rows] [cols]`
//! Prints weight-streaming throughput (GB/s of weights consumed) for each format.
use companion_engine::qmatmul::{qmatvec, qmatvec_exact, QMatrix};
use companion_engine::quant::DType;
use std::time::Instant;

/// Offsets of the f16 scale fields that must hold sane values in a random block (f32 for Q8_K).
fn patch(t: DType, blk: &mut [u8]) {
    let h = 0x2c00u16.to_le_bytes(); // 0.0625
    let set = |b: &mut [u8], o: usize| b[o..o + 2].copy_from_slice(&h);
    match t {
        DType::Q4_0 | DType::Q5_0 | DType::Q8_0 => set(blk, 0),
        DType::Q4_1 | DType::Q5_1 | DType::Q4K | DType::Q5K => { set(blk, 0); set(blk, 2) }
        DType::Q2K => { set(blk, 80); set(blk, 82) }
        DType::Q3K => set(blk, 108),
        DType::Q6K => set(blk, 208),
        DType::Q8K => blk[0..4].copy_from_slice(&0.01f32.to_le_bytes()),
        _ => {}
    }
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let (rows, cols) = (a.first().copied().unwrap_or(4096), a.get(1).copied().unwrap_or(4096));
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut seed = 0x1234_5678_9abc_def1u64;
    println!("{rows} x {cols}, {} threads available", std::thread::available_parallelism().map_or(1, |n| n.get()));
    println!("{:<6} {:>9} {:>12} {:>12} {:>12}", "type", "MB", "fast GB/s", "exact GB/s", "serial GB/s");
    for t in [DType::Q8_0, DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::Q5_1, DType::Q8K, DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K, DType::Q6K, DType::F16] {
        let (e, bsz) = t.geometry().unwrap();
        if cols % e != 0 { continue; }
        let mut w = vec![0u8; rows * (cols / e) * bsz];
        for b in w.iter_mut() { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; *b = (seed >> 24) as u8; }
        for blk in w.chunks_exact_mut(bsz) { patch(t, blk); }
        let m = QMatrix::new(t, rows, cols, &w[..]).unwrap();
        let mut out = vec![0f32; rows];
        let time = |f: &mut dyn FnMut()| {
            f();
            let (start, mut n) = (Instant::now(), 0);
            while start.elapsed().as_secs_f64() < 0.3 || n < 3 { f(); n += 1; }
            w.len() as f64 * n as f64 / start.elapsed().as_secs_f64() / 1e9
        };
        let fast = time(&mut || { m.matvec(&x, &mut out).unwrap(); });
        let exact = time(&mut || { m.matvec_exact(&x, &mut out).unwrap(); });
        let serial = time(&mut || { qmatvec_exact(t, &w, rows, cols, &x, &mut out, 1).unwrap(); });
        let _ = qmatvec; // public fast entry point, exercised through QMatrix above
        println!("{:<6} {:>9.1} {:>12.2} {:>12.2} {:>12.2}", format!("{t:?}"), w.len() as f64 / 1e6, fast, exact, serial);
        if t != DType::F16 { assert!(out.iter().all(|v| v.is_finite())); }
    }
}
