//! Throughput of every SIMD variant available on this CPU, with speed-up over the scalar one.
//! Run: `cargo run --release -p companion-engine --example bench_simd`
use companion_engine::simd::{self, Kernels};
use std::hint::black_box;
use std::time::Instant;

fn rate<F: FnMut()>(mut f: F, work: f64, min_secs: f64) -> f64 {
    for _ in 0..3 {
        f();
    }
    let (mut iters, t0) = (0u64, Instant::now());
    while t0.elapsed().as_secs_f64() < min_secs {
        for _ in 0..8 {
            f();
        }
        iters += 8;
    }
    work * iters as f64 / t0.elapsed().as_secs_f64()
}

fn main() {
    let ks = simd::available();
    println!(
        "active: {}   variants: {}",
        simd::active(),
        ks.iter().map(|k| k.name).collect::<Vec<_>>().join(", ")
    );
    let secs = 0.25;
    let n = 1 << 14; // 64 KiB of f32: stays in cache, measures the core
    let big = 1 << 23; // 32 MiB: memory-bound
    let a: Vec<f32> = (0..big)
        .map(|i| ((i * 7 % 101) as f32 - 50.0) * 0.01)
        .collect();
    let b: Vec<f32> = (0..big)
        .map(|i| ((i * 13 % 89) as f32 - 44.0) * 0.01)
        .collect();
    let ia: Vec<i8> = (0..big).map(|i| (i * 7 % 251) as i8).collect();
    let ib: Vec<i8> = (0..big).map(|i| (i * 11 % 241) as i8).collect();
    let h: Vec<u16> = (0..n).map(|i| (0x3000 + (i % 0x0800)) as u16).collect();
    let nb = 4096;
    let mut qw = vec![0u8; nb * 18];
    let mut qa = vec![0u8; nb * 34];
    for b in 0..nb {
        qw[b * 18..b * 18 + 2].copy_from_slice(&0x2c00u16.to_le_bytes());
        qa[b * 34..b * 34 + 2].copy_from_slice(&0x2800u16.to_le_bytes());
        for j in 0..16 {
            qw[b * 18 + 2 + j] = (b * 31 + j * 7) as u8;
        }
        for j in 0..32 {
            qa[b * 34 + 2 + j] = (b * 17 + j * 5) as u8;
        }
    }
    let (cs, sn): (Vec<f32>, Vec<f32>) = (0..64)
        .map(|i| ((i as f32 * 0.1).cos(), (i as f32 * 0.1).sin()))
        .unzip();

    type Row<'a> = (
        &'static str,
        &'static str,
        Box<dyn Fn(&Kernels) -> f64 + 'a>,
    );
    let rows: Vec<Row<'_>> = vec![
        (
            "dot f32 (64 KiB)",
            "GFLOP/s",
            Box::new(|k| {
                rate(
                    || {
                        black_box((k.dot_f32)(black_box(&a[..n]), black_box(&b[..n])));
                    },
                    2.0 * n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "dot f32 (32 MiB)",
            "GB/s",
            Box::new(|k| {
                rate(
                    || {
                        black_box((k.dot_f32)(black_box(&a), black_box(&b)));
                    },
                    8.0 * big as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "sumsq",
            "GFLOP/s",
            Box::new(|k| {
                rate(
                    || {
                        black_box((k.sumsq)(black_box(&a[..n])));
                    },
                    2.0 * n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "max",
            "Gelem/s",
            Box::new(|k| {
                rate(
                    || {
                        black_box((k.max)(black_box(&a[..n])));
                    },
                    n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "axpy",
            "GFLOP/s",
            Box::new(|k| {
                let mut y = b[..n].to_vec();
                rate(
                    || (k.axpy)(0.5, black_box(&a[..n]), black_box(&mut y)),
                    2.0 * n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "exp(x-m)+sum",
            "Gelem/s",
            Box::new(|k| {
                let mut t = vec![0f32; n];
                rate(
                    || {
                        for (d, s) in t.iter_mut().zip(&a) {
                            *d = *s * 40.0 - 20.0;
                        }
                        black_box((k.exp_sub_sum)(black_box(&mut t), 0.0));
                    },
                    n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "silu*mul",
            "Gelem/s",
            Box::new(|k| {
                let mut g = a[..n].to_vec();
                rate(
                    || {
                        g.copy_from_slice(&a[..n]);
                        (k.silu_mul)(black_box(&mut g), black_box(&b[..n]));
                    },
                    n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "rope (head 128)",
            "Mhead/s",
            Box::new(|k| {
                let mut hd = vec![0.5f32; 128];
                rate(|| (k.rope)(black_box(&mut hd), &cs, &sn), 1.0, secs) / 1e6
            }),
        ),
        (
            "f16 -> f32",
            "Gelem/s",
            Box::new(|k| {
                let mut d = vec![0f32; n];
                rate(
                    || (k.f16_to_f32)(black_box(&h), black_box(&mut d)),
                    n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "bf16 -> f32",
            "Gelem/s",
            Box::new(|k| {
                let mut d = vec![0f32; n];
                rate(
                    || (k.bf16_to_f32)(black_box(&h), black_box(&mut d)),
                    n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "dot i8 (16 KiB)",
            "GOP/s",
            Box::new(|k| {
                rate(
                    || {
                        black_box((k.dot_i8)(black_box(&ia[..n]), black_box(&ib[..n])));
                    },
                    2.0 * n as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "dot i8 (16 MiB)",
            "GB/s",
            Box::new(|k| {
                rate(
                    || {
                        black_box((k.dot_i8)(black_box(&ia), black_box(&ib)));
                    },
                    2.0 * big as f64,
                    secs,
                ) / 1e9
            }),
        ),
        (
            "q4_0 x q8_0 row",
            "GB/s weights",
            Box::new(|k| {
                rate(
                    || {
                        black_box((k.q4_0_q8_0_row)(black_box(&qw), black_box(&qa), nb));
                    },
                    (nb * 18) as f64,
                    secs,
                ) / 1e9
            }),
        ),
    ];

    print!("{:<20}{:<14}", "kernel", "unit");
    for k in &ks {
        print!("{:>20}", k.name);
    }
    println!();
    for (name, unit, f) in &rows {
        let vals: Vec<f64> = ks.iter().map(f).collect();
        print!("{name:<20}{unit:<14}");
        for (i, v) in vals.iter().enumerate() {
            if i == 0 {
                print!("{v:>20.2}");
            } else {
                print!("{:>20}", format!("{v:.2} ({:.1}x)", v / vals[0]));
            }
        }
        println!();
    }
}
