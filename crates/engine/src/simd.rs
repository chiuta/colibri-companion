//! Runtime-dispatched numeric kernels.
//!
//! One table of function pointers ([`Kernels`]) is chosen once, on first use, from what the CPU
//! reports (`is_x86_feature_detected!` / `is_aarch64_feature_detected!`). A scalar variant is always
//! present: it is the fallback on unknown CPUs and the oracle the tests compare every other
//! variant against. The public free functions are safe; `unsafe` is confined to the
//! `#[target_feature]` implementations, which are only ever reached through a table that was
//! built after the matching detection succeeded.
//!
//! Variants (best available wins): `avx512+vnni`, `avx512`, `avx2+fma+avxvnni`, `avx2+fma`
//! (x86-64); `neon+dotprod`, `neon` (aarch64); `scalar` everywhere. Set the environment variable
//! `ENGINE_SIMD` to a variant name to force it (for benchmarks and bug reports).
//!
//! Numeric contract:
//! * integer kernels are exact (modulo 2^32 for overflow) in every variant, including `-128` inputs;
//! * float reductions may differ from the scalar order by normal rounding (reassociation);
//! * `exp`-based kernels use a degree-7 polynomial, relative error about 2e-7, unless
//!   [`set_precise`] is on, which routes them through libm.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Table of kernels for one instruction-set level.
#[derive(Clone, Copy)]
pub struct Kernels {
    pub name: &'static str,
    /// Σ a·b over the common length.
    pub dot_f32: fn(&[f32], &[f32]) -> f32,
    /// Σ a².
    pub sumsq: fn(&[f32]) -> f32,
    /// Maximum, NaN ignored (−∞ for an empty or all-NaN slice).
    pub max: fn(&[f32]) -> f32,
    /// `y += a·x`.
    pub axpy: fn(f32, &[f32], &mut [f32]),
    /// `x *= k`.
    pub scale: fn(&mut [f32], f32),
    /// `out = x·k·w`.
    pub mul3: fn(&[f32], &[f32], f32, &mut [f32]),
    /// `x = exp(x − m)` in place; returns the sum of the results.
    pub exp_sub_sum: fn(&mut [f32], f32) -> f32,
    /// `g = silu(g)·u` in place.
    pub silu_mul: fn(&mut [f32], &[f32]),
    /// Rotate-half rotation of one head vector by precomputed `cos`/`sin` tables.
    pub rope: fn(&mut [f32], &[f32], &[f32]),
    pub f16_to_f32: fn(&[u16], &mut [f32]),
    pub bf16_to_f32: fn(&[u16], &mut [f32]),
    /// Exact signed 8-bit dot product.
    pub dot_i8: fn(&[i8], &[i8]) -> i32,
    /// Σ (nibble − 8)·y for one Q4_0 block (low nibbles are elements 0..16, high nibbles 16..32).
    pub q4_0_block: fn(&[u8; 16], &[i8; 32]) -> i32,
    /// Dot of `n` Q4_0 weight blocks (18 bytes each) with `n` Q8_0 activation blocks (34 bytes each).
    pub q4_0_q8_0_row: fn(&[u8], &[u8], usize) -> f32,
}

static PRECISE: AtomicBool = AtomicBool::new(false);

/// When on, `exp`-based kernels use libm instead of the polynomial (slower, correctly rounded-ish).
pub fn set_precise(on: bool) { PRECISE.store(on, Ordering::Relaxed); }
pub fn is_precise() -> bool { PRECISE.load(Ordering::Relaxed) }

fn table() -> &'static Kernels {
    static K: OnceLock<Kernels> = OnceLock::new();
    K.get_or_init(|| {
        let all = available();
        if let Ok(want) = std::env::var("ENGINE_SIMD") {
            if let Some(k) = all.iter().find(|k| k.name == want) { return *k; }
        }
        *all.last().expect("the scalar variant always exists")
    })
}

/// Name of the variant in use, e.g. `"avx2+fma"`.
pub fn active() -> &'static str { table().name }

/// Every variant this CPU can run, scalar first, best last.
pub fn available() -> Vec<Kernels> {
    let mut v = vec![scalar::kernels()];
    #[cfg(target_arch = "x86_64")]
    v.extend(x86::variants());
    #[cfg(target_arch = "aarch64")]
    v.extend(arm::variants());
    v
}

// ------------------------------------------------------------------ safe front door

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 { (table().dot_f32)(a, b) }
pub fn sumsq(a: &[f32]) -> f32 { (table().sumsq)(a) }
pub fn max(a: &[f32]) -> f32 { (table().max)(a) }
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) { (table().axpy)(a, x, y) }
pub fn scale(x: &mut [f32], k: f32) { (table().scale)(x, k) }
/// `out[i] = x[i]·k·w[i]` over the shortest of the three slices.
pub fn mul3(x: &[f32], w: &[f32], k: f32, out: &mut [f32]) { (table().mul3)(x, w, k, out) }
pub fn exp_sub_sum(x: &mut [f32], m: f32) -> f32 {
    if is_precise() { scalar::exp_sub_sum(x, m) } else { (table().exp_sub_sum)(x, m) }
}
pub fn silu_mul(g: &mut [f32], u: &[f32]) {
    if is_precise() { scalar::silu_mul(g, u) } else { (table().silu_mul)(g, u) }
}
/// Rotate `head` (pairs `i`, `i + half`, `half = cos.len()`) by the angles given as `cos`/`sin`.
pub fn rope_rotate(head: &mut [f32], cos: &[f32], sin: &[f32]) { (table().rope)(head, cos, sin) }
pub fn f16_to_f32(src: &[u16], dst: &mut [f32]) { (table().f16_to_f32)(src, dst) }
pub fn bf16_to_f32(src: &[u16], dst: &mut [f32]) { (table().bf16_to_f32)(src, dst) }
pub fn dot_i8_i8(a: &[i8], b: &[i8]) -> i32 { (table().dot_i8)(a, b) }
pub fn dot_q4_0_block(qs: &[u8; 16], y: &[i8; 32]) -> i32 { (table().q4_0_block)(qs, y) }
/// Σ over blocks of `d_w·d_a·Σ (q−8)·y`; `n` is clamped to what both slices hold.
pub fn q4_0_q8_0_row(w: &[u8], a: &[u8], n: usize) -> f32 {
    let n = n.min(w.len() / 18).min(a.len() / 34);
    (table().q4_0_q8_0_row)(w, a, n)
}
/// Same for Q8_0 weights (34-byte blocks) against Q8_0 activations.
pub fn q8_0_q8_0_row(w: &[u8], a: &[u8], n: usize) -> f32 {
    let n = n.min(w.len() / 34).min(a.len() / 34);
    q8_0_q8_0_with(table().dot_i8, w, a, n)
}
fn q8_0_q8_0_with(dot: fn(&[i8], &[i8]) -> i32, w: &[u8], a: &[u8], n: usize) -> f32 {
    let mut acc = 0f32;
    for (wb, ab) in w.chunks_exact(34).zip(a.chunks_exact(34)).take(n) {
        let (dw, da) = (scalar::f16_to_f32(u16::from_le_bytes([wb[0], wb[1]])), scalar::f16_to_f32(u16::from_le_bytes([ab[0], ab[1]])));
        let (mut x, mut y) = ([0i8; 32], [0i8; 32]);
        for i in 0..32 { x[i] = wb[2 + i] as i8; y[i] = ab[2 + i] as i8; }
        acc += dw * da * dot(&x, &y) as f32;
    }
    acc
}

// ------------------------------------------------------------------------ shared math

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_145_75;
const LN2_LO: f32 = 1.428_606_8e-6;
const EXP_HI: f32 = 88.0;
const EXP_LO: f32 = -87.336_55;
const C2: f32 = 0.5;
const C3: f32 = 1.0 / 6.0;
const C4: f32 = 1.0 / 24.0;
const C5: f32 = 1.0 / 120.0;
const C6: f32 = 1.0 / 720.0;
const C7: f32 = 1.0 / 5040.0;

/// Generic element-wise loops. They are `inline(always)` so each `#[target_feature]` wrapper that
/// calls them gets its own auto-vectorised copy compiled for that instruction set.
mod gen_loops {
    #[inline(always)]
    pub fn axpy<const FMA: bool>(a: f32, x: &[f32], y: &mut [f32]) {
        for (y, x) in y.iter_mut().zip(x) { *y = if FMA { a.mul_add(*x, *y) } else { a * x + *y }; }
    }
    #[inline(always)]
    pub fn scale(x: &mut [f32], k: f32) { for v in x { *v *= k; } }
    #[inline(always)]
    pub fn mul3(x: &[f32], w: &[f32], k: f32, out: &mut [f32]) {
        for ((o, x), w) in out.iter_mut().zip(x).zip(w) { *o = x * k * w; }
    }
    #[inline(always)]
    pub fn rope<const FMA: bool>(h: &mut [f32], c: &[f32], s: &[f32]) {
        let half = c.len().min(s.len()).min(h.len() / 2);
        let (lo, rest) = h.split_at_mut(half);
        let hi = &mut rest[..half];
        for i in 0..half {
            let (a, b) = (lo[i], hi[i]);
            if FMA {
                lo[i] = a.mul_add(c[i], -(b * s[i]));
                hi[i] = a.mul_add(s[i], b * c[i]);
            } else {
                lo[i] = a * c[i] - b * s[i];
                hi[i] = a * s[i] + b * c[i];
            }
        }
    }
    #[inline(always)]
    pub fn bf16(src: &[u16], dst: &mut [f32]) {
        for (d, s) in dst.iter_mut().zip(src) { *d = f32::from_bits((*s as u32) << 16); }
    }
}

// ----------------------------------------------------------------------------- scalar

pub(crate) mod scalar {
    use super::*;

    pub fn kernels() -> Kernels {
        Kernels {
            name: "scalar", dot_f32, sumsq, max, axpy: axpy_k, scale: scale_k, mul3: mul3_k,
            exp_sub_sum, silu_mul, rope: rope_k, f16_to_f32: f16_slice, bf16_to_f32: bf16_k,
            dot_i8, q4_0_block, q4_0_q8_0_row,
        }
    }
    pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
    pub fn sumsq(a: &[f32]) -> f32 { a.iter().map(|x| x * x).sum() }
    pub fn max(a: &[f32]) -> f32 { a.iter().copied().fold(f32::NEG_INFINITY, f32::max) }
    fn axpy_k(a: f32, x: &[f32], y: &mut [f32]) { gen_loops::axpy::<false>(a, x, y) }
    fn scale_k(x: &mut [f32], k: f32) { gen_loops::scale(x, k) }
    fn mul3_k(x: &[f32], w: &[f32], k: f32, o: &mut [f32]) { gen_loops::mul3(x, w, k, o) }
    fn rope_k(h: &mut [f32], c: &[f32], s: &[f32]) { gen_loops::rope::<false>(h, c, s) }
    fn bf16_k(s: &[u16], d: &mut [f32]) { gen_loops::bf16(s, d) }
    pub fn exp_sub_sum(x: &mut [f32], m: f32) -> f32 {
        let mut s = 0.0;
        for v in x.iter_mut() { *v = (*v - m).exp(); s += *v; }
        s
    }
    pub fn silu_mul(g: &mut [f32], u: &[f32]) {
        for (g, u) in g.iter_mut().zip(u) { *g = *g / (1.0 + (-*g).exp()) * u; }
    }
    /// The polynomial every vector variant implements, used for their scalar tails.
    pub fn exp_poly(x: f32) -> f32 {
        if x.is_nan() { return x; }
        if x < EXP_LO { return 0.0; }
        let xc = x.min(EXP_HI);
        let n = (xc * LOG2E).round_ties_even();
        let r = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, xc));
        let mut p = C7;
        for c in [C6, C5, C4, C3, C2, 1.0, 1.0] { p = p.mul_add(r, c); }
        p * f32::from_bits(((n as i32 + 127) as u32) << 23)
    }
    pub fn silu_poly(g: f32, u: f32) -> f32 { g / (1.0 + exp_poly(-g)) * u }
    /// IEEE binary16 → f32, all values exact.
    pub fn f16_to_f32(h: u16) -> f32 {
        let (s, e, m) = (((h >> 15) & 1) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
        let bits = if e == 0 {
            if m == 0 { s << 31 } else {
                let (mut e2, mut m2) = (127 - 15 + 1u32, m);
                while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
                (s << 31) | (e2 << 23) | ((m2 & 0x3ff) << 13)
            }
        } else if e == 31 { (s << 31) | 0x7f80_0000 | (m << 13) } else { (s << 31) | ((e + 112) << 23) | (m << 13) };
        f32::from_bits(bits)
    }
    fn f16_slice(src: &[u16], dst: &mut [f32]) { for (d, s) in dst.iter_mut().zip(src) { *d = f16_to_f32(*s); } }
    pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        a.iter().zip(b).fold(0i32, |s, (x, y)| s.wrapping_add(*x as i32 * *y as i32))
    }
    pub fn q4_0_block(qs: &[u8; 16], y: &[i8; 32]) -> i32 {
        let mut s = 0i32;
        for j in 0..16 {
            s = s.wrapping_add(((qs[j] & 15) as i32 - 8) * y[j] as i32);
            s = s.wrapping_add(((qs[j] >> 4) as i32 - 8) * y[16 + j] as i32);
        }
        s
    }
    pub fn q4_0_q8_0_row(w: &[u8], a: &[u8], n: usize) -> f32 {
        let mut acc = 0f32;
        for (wb, ab) in w.chunks_exact(18).zip(a.chunks_exact(34)).take(n) {
            let dw = f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
            let da = f16_to_f32(u16::from_le_bytes([ab[0], ab[1]]));
            let qs: &[u8; 16] = wb[2..18].try_into().unwrap();
            let mut y = [0i8; 32];
            for (d, s) in y.iter_mut().zip(&ab[2..34]) { *d = *s as i8; }
            acc += dw * da * q4_0_block(qs, &y) as f32;
        }
        acc
    }
}

// ------------------------------------------------------------------- shim generators

macro_rules! shim_dot {
    ($shim:ident, $imp:ident) => {
        fn $shim(a: &[f32], b: &[f32]) -> f32 {
            let n = a.len().min(b.len());
            // SAFETY: only reachable through a table built after the CPU feature was detected.
            unsafe { $imp(&a[..n], &b[..n]) }
        }
    };
}
macro_rules! shim_unary {
    ($shim:ident, $imp:ident) => {
        fn $shim(a: &[f32]) -> f32 { unsafe { $imp(a) } }
    };
}
macro_rules! shim_axpy {
    ($shim:ident, $imp:ident) => {
        fn $shim(a: f32, x: &[f32], y: &mut [f32]) {
            let n = x.len().min(y.len());
            unsafe { $imp(a, &x[..n], &mut y[..n]) }
        }
    };
}
macro_rules! shim_scale {
    ($shim:ident, $imp:ident) => { fn $shim(x: &mut [f32], k: f32) { unsafe { $imp(x, k) } } };
}
macro_rules! shim_mul3 {
    ($shim:ident, $imp:ident) => {
        fn $shim(x: &[f32], w: &[f32], k: f32, o: &mut [f32]) {
            let n = x.len().min(w.len()).min(o.len());
            unsafe { $imp(&x[..n], &w[..n], k, &mut o[..n]) }
        }
    };
}
macro_rules! shim_exp {
    ($shim:ident, $imp:ident) => { fn $shim(x: &mut [f32], m: f32) -> f32 { unsafe { $imp(x, m) } } };
}
macro_rules! shim_silu {
    ($shim:ident, $imp:ident) => {
        fn $shim(g: &mut [f32], u: &[f32]) {
            let n = g.len().min(u.len());
            unsafe { $imp(&mut g[..n], &u[..n]) }
        }
    };
}
macro_rules! shim_rope {
    ($shim:ident, $imp:ident) => {
        fn $shim(h: &mut [f32], c: &[f32], s: &[f32]) { unsafe { $imp(h, c, s) } }
    };
}
macro_rules! shim_conv {
    ($shim:ident, $imp:ident) => {
        fn $shim(s: &[u16], d: &mut [f32]) {
            let n = s.len().min(d.len());
            unsafe { $imp(&s[..n], &mut d[..n]) }
        }
    };
}
macro_rules! shim_i8 {
    ($shim:ident, $imp:ident) => {
        fn $shim(a: &[i8], b: &[i8]) -> i32 {
            let n = a.len().min(b.len());
            unsafe { $imp(&a[..n], &b[..n]) }
        }
    };
}
macro_rules! shim_q4blk {
    ($shim:ident, $imp:ident) => {
        fn $shim(q: &[u8; 16], y: &[i8; 32]) -> i32 { unsafe { $imp(q.as_ptr(), y.as_ptr()) } }
    };
}
macro_rules! shim_q4row {
    ($shim:ident, $imp:ident) => {
        fn $shim(w: &[u8], a: &[u8], n: usize) -> f32 {
            let n = n.min(w.len() / 18).min(a.len() / 34);
            unsafe { $imp(w.as_ptr(), a.as_ptr(), n) }
        }
    };
}

// ------------------------------------------------------------------------------ x86-64

// Newer compilers make pure-register intrinsics safe inside a matching `target_feature` fn; the
// `unsafe` blocks stay for older ones. The AVX-512 and AVX-VNNI intrinsics are stable only since
// Rust 1.89, newer than the workspace `rust-version` (1.85): bump it to build this module.
#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe, clippy::incompatible_msrv)]
mod x86 {
    use super::*;
    use std::arch::x86_64::*;

    const RND: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
    }
    fn has_512() -> bool {
        has_avx2() && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vl")
    }

    pub fn variants() -> Vec<Kernels> {
        let mut v = vec![];
        if !has_avx2() { return v; }
        let avx2 = Kernels {
            name: "avx2+fma", dot_f32: s_dot_avx2, sumsq: s_sumsq_avx2, max: s_max_avx2, axpy: s_axpy_avx2,
            scale: s_scale_avx2, mul3: s_mul3_avx2, exp_sub_sum: s_exp_avx2, silu_mul: s_silu_avx2,
            rope: s_rope_avx2, f16_to_f32: s_f16_avx2, bf16_to_f32: s_bf16_avx2, dot_i8: s_i8_avx2,
            q4_0_block: s_q4b_avx2, q4_0_q8_0_row: s_q4r_avx2,
        };
        v.push(avx2);
        if is_x86_feature_detected!("avxvnni") {
            v.push(Kernels { name: "avx2+fma+avxvnni", dot_i8: s_i8_avxvnni, q4_0_block: s_q4b_avxvnni, q4_0_q8_0_row: s_q4r_avxvnni, ..avx2 });
        }
        if has_512() {
            let avx512 = Kernels {
                name: "avx512", dot_f32: s_dot_512, sumsq: s_sumsq_512, max: s_max_512, axpy: s_axpy_512,
                scale: s_scale_512, mul3: s_mul3_512, exp_sub_sum: s_exp_512, silu_mul: s_silu_512,
                rope: s_rope_512, f16_to_f32: s_f16_512, bf16_to_f32: s_bf16_512, ..avx2
            };
            v.push(avx512);
            if is_x86_feature_detected!("avx512vnni") {
                v.push(Kernels { name: "avx512+vnni", dot_i8: s_i8_vnni512, q4_0_block: s_q4b_vnni512, q4_0_q8_0_row: s_q4r_vnni512, ..avx512 });
            }
        }
        v
    }

    // ------------------------------------------------------------------ AVX2 + FMA

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum256(v: __m256) -> f32 { unsafe {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        _mm_cvtss_f32(s)
    } }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn hsum_epi32(v: __m256i) -> i32 { unsafe {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
        _mm_cvtsi128_si32(s)
    } }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 { unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), a3);
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), a0);
            i += 8;
        }
        let mut s = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        while i < n { s += *pa.add(i) * *pb.add(i); i += 1; }
        s
    } }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn sumsq_avx2(a: &[f32]) -> f32 { unsafe { dot_avx2(a, a) } }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn max_avx2(a: &[f32]) -> f32 { unsafe {
        let n = a.len();
        let p = a.as_ptr();
        let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 8 <= n { acc = _mm256_max_ps(_mm256_loadu_ps(p.add(i)), acc); i += 8; } // NaN lane keeps `acc`
        let mut t = [0f32; 8];
        _mm256_storeu_ps(t.as_mut_ptr(), acc);
        let mut m = t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        while i < n { m = m.max(*p.add(i)); i += 1; }
        m
    } }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn axpy_avx2(a: f32, x: &[f32], y: &mut [f32]) { gen_loops::axpy::<true>(a, x, y) }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn scale_avx2(x: &mut [f32], k: f32) { gen_loops::scale(x, k) }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mul3_avx2(x: &[f32], w: &[f32], k: f32, o: &mut [f32]) { gen_loops::mul3(x, w, k, o) }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn rope_avx2(h: &mut [f32], c: &[f32], s: &[f32]) { gen_loops::rope::<true>(h, c, s) }
    #[target_feature(enable = "avx2,fma")]
    unsafe fn bf16_avx2(s: &[u16], d: &mut [f32]) { gen_loops::bf16(s, d) }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp256(x: __m256) -> __m256 { unsafe {
        let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(EXP_LO));
        // operand order makes a NaN lane come out as NaN instead of being clamped
        let xc = _mm256_max_ps(_mm256_set1_ps(EXP_LO), _mm256_min_ps(_mm256_set1_ps(EXP_HI), x));
        let n = _mm256_round_ps::<RND>(_mm256_mul_ps(xc, _mm256_set1_ps(LOG2E)));
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), xc);
        let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), r);
        let mut p = _mm256_set1_ps(C7);
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C6));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C5));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C4));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C3));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(C2));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
        p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0));
        let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)));
        _mm256_andnot_ps(lt, _mm256_mul_ps(p, _mm256_castsi256_ps(e)))
    } }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn exp_avx2(x: &mut [f32], m: f32) -> f32 { unsafe {
        let n = x.len();
        let p = x.as_mut_ptr();
        let mv = _mm256_set1_ps(m);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let e = exp256(_mm256_sub_ps(_mm256_loadu_ps(p.add(i)), mv));
            _mm256_storeu_ps(p.add(i), e);
            acc = _mm256_add_ps(acc, e);
            i += 8;
        }
        let mut s = hsum256(acc);
        while i < n { let e = scalar::exp_poly(*p.add(i) - m); *p.add(i) = e; s += e; i += 1; }
        s
    } }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn silu_avx2(g: &mut [f32], u: &[f32]) { unsafe {
        let n = g.len();
        let (pg, pu) = (g.as_mut_ptr(), u.as_ptr());
        let (zero, one) = (_mm256_setzero_ps(), _mm256_set1_ps(1.0));
        let mut i = 0;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(pg.add(i));
            let d = _mm256_add_ps(one, exp256(_mm256_sub_ps(zero, v)));
            _mm256_storeu_ps(pg.add(i), _mm256_mul_ps(_mm256_div_ps(v, d), _mm256_loadu_ps(pu.add(i))));
            i += 8;
        }
        while i < n { *pg.add(i) = scalar::silu_poly(*pg.add(i), *pu.add(i)); i += 1; }
    } }

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn f16_avx2(s: &[u16], d: &mut [f32]) { unsafe {
        let n = s.len();
        let (ps, pd) = (s.as_ptr(), d.as_mut_ptr());
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(pd.add(i), _mm256_cvtph_ps(_mm_loadu_si128(ps.add(i) as *const __m128i)));
            i += 8;
        }
        while i < n { *pd.add(i) = scalar::f16_to_f32(*ps.add(i)); i += 1; }
    } }

    /// Exact for every input: widen to i16, multiply-add pairs into i32.
    #[target_feature(enable = "avx2")]
    unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 { unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i + 16 <= n {
            let x = _mm256_cvtepi8_epi16(_mm_loadu_si128(pa.add(i) as *const __m128i));
            let y = _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.add(i) as *const __m128i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
            i += 16;
        }
        let mut s = hsum_epi32(acc);
        while i < n { s = s.wrapping_add(*pa.add(i) as i32 * *pb.add(i) as i32); i += 1; }
        s
    } }

    /// Nibbles as u8 in element order (low half first) — shared by the Q4_0 kernels.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn nibbles(qs: *const u8) -> __m256i { unsafe {
        let v = _mm_loadu_si128(qs as *const __m128i);
        let m = _mm_set1_epi8(0x0f);
        let lo = _mm_and_si128(v, m);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(v), m);
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    } }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn q4blk_avx2(qs: *const u8, y: *const i8) -> i32 { unsafe {
        let yv = _mm256_loadu_si256(y as *const __m256i);
        let one16 = _mm256_set1_epi16(1);
        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(nibbles(qs), yv), one16);
        let s = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), yv), one16);
        hsum_epi32(_mm256_sub_epi32(p, _mm256_slli_epi32::<3>(s)))
    } }

    #[inline]
    #[target_feature(enable = "f16c")]
    unsafe fn half(p: *const u8) -> f32 { unsafe {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(u16::from_le_bytes([*p, *p.add(1)]) as i32)))
    } }

    macro_rules! q4_row {
        ($name:ident, $feat:literal, $blk:ident) => {
            #[target_feature(enable = $feat)]
            unsafe fn $name(w: *const u8, a: *const u8, n: usize) -> f32 { unsafe {
                let mut acc = 0f32;
                for b in 0..n {
                    let (wp, ap) = (w.add(b * 18), a.add(b * 34));
                    let d = half(wp) * half(ap);
                    acc = d.mul_add($blk(wp.add(2), ap.add(2) as *const i8) as f32, acc);
                }
                acc
            } }
        };
    }
    q4_row!(q4row_avx2, "avx2,fma,f16c", q4blk_avx2);

    // AVX-VNNI (ymm): u8×s8 dot-accumulate; signed×signed through the +128 bias trick.
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn dot_i8_avxvnni(a: &[i8], b: &[i8]) -> i32 { unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (ones, bias) = (_mm256_set1_epi8(1), _mm256_set1_epi8(-128));
        let (mut acc, mut sb) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        let mut i = 0;
        while i + 32 <= n {
            let x = _mm256_xor_si256(_mm256_loadu_si256(pa.add(i) as *const __m256i), bias);
            let y = _mm256_loadu_si256(pb.add(i) as *const __m256i);
            acc = _mm256_dpbusd_avx_epi32(acc, x, y);
            sb = _mm256_dpbusd_avx_epi32(sb, ones, y);
            i += 32;
        }
        let mut s = hsum_epi32(acc).wrapping_sub(hsum_epi32(sb).wrapping_shl(7));
        while i < n { s = s.wrapping_add(*pa.add(i) as i32 * *pb.add(i) as i32); i += 1; }
        s
    } }

    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn q4blk_avxvnni(qs: *const u8, y: *const i8) -> i32 { unsafe {
        let yv = _mm256_loadu_si256(y as *const __m256i);
        let z = _mm256_setzero_si256();
        let p = _mm256_dpbusd_avx_epi32(z, nibbles(qs), yv);
        let s = _mm256_dpbusd_avx_epi32(z, _mm256_set1_epi8(1), yv);
        hsum_epi32(_mm256_sub_epi32(p, _mm256_slli_epi32::<3>(s)))
    } }
    q4_row!(q4row_avxvnni, "avx2,fma,f16c,avxvnni", q4blk_avxvnni);

    // ------------------------------------------------------------------ AVX-512

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn exp512(x: __m512) -> __m512 { unsafe {
        let lt = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_set1_ps(EXP_LO));
        let xc = _mm512_max_ps(_mm512_set1_ps(EXP_LO), _mm512_min_ps(_mm512_set1_ps(EXP_HI), x));
        let n = _mm512_roundscale_ps::<RND>(_mm512_mul_ps(xc, _mm512_set1_ps(LOG2E)));
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_HI), xc);
        let r = _mm512_fnmadd_ps(n, _mm512_set1_ps(LN2_LO), r);
        let mut p = _mm512_set1_ps(C7);
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C6));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C5));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C4));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C3));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(C2));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
        p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0));
        let e = _mm512_slli_epi32::<23>(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)));
        _mm512_mask_blend_ps(lt, _mm512_mul_ps(p, _mm512_castsi512_ps(e)), _mm512_setzero_ps())
    } }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn dot_512(a: &[f32], b: &[f32]) -> f32 { unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (mut a0, mut a1, mut a2, mut a3) = (_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps());
        let mut i = 0;
        while i + 64 <= n {
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i + 16)), _mm512_loadu_ps(pb.add(i + 16)), a1);
            a2 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i + 32)), _mm512_loadu_ps(pb.add(i + 32)), a2);
            a3 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i + 48)), _mm512_loadu_ps(pb.add(i + 48)), a3);
            i += 64;
        }
        while i + 16 <= n {
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(pa.add(i)), _mm512_loadu_ps(pb.add(i)), a0);
            i += 16;
        }
        let mut s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
        while i < n { s += *pa.add(i) * *pb.add(i); i += 1; }
        s
    } }
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn sumsq_512(a: &[f32]) -> f32 { unsafe { dot_512(a, a) } }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn max_512(a: &[f32]) -> f32 { unsafe {
        let n = a.len();
        let p = a.as_ptr();
        let mut acc = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 16 <= n { acc = _mm512_max_ps(_mm512_loadu_ps(p.add(i)), acc); i += 16; }
        let mut t = [0f32; 16];
        _mm512_storeu_ps(t.as_mut_ptr(), acc);
        let mut m = t.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        while i < n { m = m.max(*p.add(i)); i += 1; }
        m
    } }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn axpy_512(a: f32, x: &[f32], y: &mut [f32]) { gen_loops::axpy::<true>(a, x, y) }
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn scale_512(x: &mut [f32], k: f32) { gen_loops::scale(x, k) }
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn mul3_512(x: &[f32], w: &[f32], k: f32, o: &mut [f32]) { gen_loops::mul3(x, w, k, o) }
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn rope_512(h: &mut [f32], c: &[f32], s: &[f32]) { gen_loops::rope::<true>(h, c, s) }
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn bf16_512(s: &[u16], d: &mut [f32]) { gen_loops::bf16(s, d) }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn exp_512(x: &mut [f32], m: f32) -> f32 { unsafe {
        let n = x.len();
        let p = x.as_mut_ptr();
        let mv = _mm512_set1_ps(m);
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let e = exp512(_mm512_sub_ps(_mm512_loadu_ps(p.add(i)), mv));
            _mm512_storeu_ps(p.add(i), e);
            acc = _mm512_add_ps(acc, e);
            i += 16;
        }
        let mut s = _mm512_reduce_add_ps(acc);
        while i < n { let e = scalar::exp_poly(*p.add(i) - m); *p.add(i) = e; s += e; i += 1; }
        s
    } }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
    unsafe fn silu_512(g: &mut [f32], u: &[f32]) { unsafe {
        let n = g.len();
        let (pg, pu) = (g.as_mut_ptr(), u.as_ptr());
        let (zero, one) = (_mm512_setzero_ps(), _mm512_set1_ps(1.0));
        let mut i = 0;
        while i + 16 <= n {
            let v = _mm512_loadu_ps(pg.add(i));
            let d = _mm512_add_ps(one, exp512(_mm512_sub_ps(zero, v)));
            _mm512_storeu_ps(pg.add(i), _mm512_mul_ps(_mm512_div_ps(v, d), _mm512_loadu_ps(pu.add(i))));
            i += 16;
        }
        while i < n { *pg.add(i) = scalar::silu_poly(*pg.add(i), *pu.add(i)); i += 1; }
    } }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,f16c")]
    unsafe fn f16_512(s: &[u16], d: &mut [f32]) { unsafe {
        let n = s.len();
        let (ps, pd) = (s.as_ptr(), d.as_mut_ptr());
        let mut i = 0;
        while i + 16 <= n {
            _mm512_storeu_ps(pd.add(i), _mm512_cvtph_ps(_mm256_loadu_si256(ps.add(i) as *const __m256i)));
            i += 16;
        }
        while i < n { *pd.add(i) = scalar::f16_to_f32(*ps.add(i)); i += 1; }
    } }

    // AVX512-VNNI: 64 bytes per step, then one 32-byte step, then a scalar tail.
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2")]
    unsafe fn dot_i8_vnni512(a: &[i8], b: &[i8]) -> i32 { unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (ones, bias) = (_mm512_set1_epi8(1), _mm512_set1_epi8(-128));
        let (mut acc, mut sb) = (_mm512_setzero_si512(), _mm512_setzero_si512());
        let mut i = 0;
        while i + 64 <= n {
            let x = _mm512_xor_si512(_mm512_loadu_si512(pa.add(i) as *const _), bias);
            let y = _mm512_loadu_si512(pb.add(i) as *const _);
            acc = _mm512_dpbusd_epi32(acc, x, y);
            sb = _mm512_dpbusd_epi32(sb, ones, y);
            i += 64;
        }
        let mut s = _mm512_reduce_add_epi32(acc).wrapping_sub(_mm512_reduce_add_epi32(sb).wrapping_shl(7));
        if i + 32 <= n {
            let x = _mm256_xor_si256(_mm256_loadu_si256(pa.add(i) as *const __m256i), _mm256_set1_epi8(-128));
            let y = _mm256_loadu_si256(pb.add(i) as *const __m256i);
            let p = _mm256_dpbusd_epi32(_mm256_setzero_si256(), x, y);
            let q = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_set1_epi8(1), y);
            s = s.wrapping_add(hsum_epi32(p).wrapping_sub(hsum_epi32(q).wrapping_shl(7)));
            i += 32;
        }
        while i < n { s = s.wrapping_add(*pa.add(i) as i32 * *pb.add(i) as i32); i += 1; }
        s
    } }

    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni,avx2")]
    unsafe fn q4blk_vnni512(qs: *const u8, y: *const i8) -> i32 { unsafe {
        let yv = _mm256_loadu_si256(y as *const __m256i);
        let z = _mm256_setzero_si256();
        let p = _mm256_dpbusd_epi32(z, nibbles(qs), yv);
        let s = _mm256_dpbusd_epi32(z, _mm256_set1_epi8(1), yv);
        hsum_epi32(_mm256_sub_epi32(p, _mm256_slli_epi32::<3>(s)))
    } }
    q4_row!(q4row_vnni512, "avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma,f16c", q4blk_vnni512);

    // ------------------------------------------------------------------ shims

    shim_dot!(s_dot_avx2, dot_avx2);
    shim_unary!(s_sumsq_avx2, sumsq_avx2);
    shim_unary!(s_max_avx2, max_avx2);
    shim_axpy!(s_axpy_avx2, axpy_avx2);
    shim_scale!(s_scale_avx2, scale_avx2);
    shim_mul3!(s_mul3_avx2, mul3_avx2);
    shim_exp!(s_exp_avx2, exp_avx2);
    shim_silu!(s_silu_avx2, silu_avx2);
    shim_rope!(s_rope_avx2, rope_avx2);
    shim_conv!(s_f16_avx2, f16_avx2);
    shim_conv!(s_bf16_avx2, bf16_avx2);
    shim_i8!(s_i8_avx2, dot_i8_avx2);
    shim_q4blk!(s_q4b_avx2, q4blk_avx2);
    shim_q4row!(s_q4r_avx2, q4row_avx2);
    shim_i8!(s_i8_avxvnni, dot_i8_avxvnni);
    shim_q4blk!(s_q4b_avxvnni, q4blk_avxvnni);
    shim_q4row!(s_q4r_avxvnni, q4row_avxvnni);
    shim_dot!(s_dot_512, dot_512);
    shim_unary!(s_sumsq_512, sumsq_512);
    shim_unary!(s_max_512, max_512);
    shim_axpy!(s_axpy_512, axpy_512);
    shim_scale!(s_scale_512, scale_512);
    shim_mul3!(s_mul3_512, mul3_512);
    shim_exp!(s_exp_512, exp_512);
    shim_silu!(s_silu_512, silu_512);
    shim_rope!(s_rope_512, rope_512);
    shim_conv!(s_f16_512, f16_512);
    shim_conv!(s_bf16_512, bf16_512);
    shim_i8!(s_i8_vnni512, dot_i8_vnni512);
    shim_q4blk!(s_q4b_vnni512, q4blk_vnni512);
    shim_q4row!(s_q4r_vnni512, q4row_vnni512);
}

// ----------------------------------------------------------------------------- aarch64
//
// NOT compiled in the sandbox that wrote it (no aarch64 std available there): reviewed by hand
// and restricted to long-stable NEON intrinsics; the two `asm!` uses are `fcvtl` (half → single,
// base ARMv8) and `sdot` (dotprod). Run `cargo test -p companion-engine` on an ARM machine.

#[cfg(target_arch = "aarch64")]
#[allow(unused_unsafe)]
mod arm {
    use super::*;
    use std::arch::aarch64::*;

    pub fn variants() -> Vec<Kernels> {
        // NEON is mandatory on aarch64.
        let neon = Kernels {
            name: "neon", dot_f32: s_dot_neon, sumsq: s_sumsq_neon, max: s_max_neon, axpy: s_axpy_neon,
            scale: s_scale_neon, mul3: s_mul3_neon, exp_sub_sum: s_exp_neon, silu_mul: s_silu_neon,
            rope: s_rope_neon, f16_to_f32: s_f16_neon, bf16_to_f32: s_bf16_neon, dot_i8: s_i8_neon,
            q4_0_block: s_q4b_neon, q4_0_q8_0_row: s_q4r_neon,
        };
        let mut v = vec![neon];
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            v.push(Kernels { name: "neon+dotprod", dot_i8: s_i8_dot, q4_0_block: s_q4b_dot, q4_0_q8_0_row: s_q4r_dot, ..neon });
        }
        v
    }

    unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 { unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (mut a0, mut a1, mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let mut i = 0;
        while i + 16 <= n {
            a0 = vfmaq_f32(a0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            a1 = vfmaq_f32(a1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            a2 = vfmaq_f32(a2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            a3 = vfmaq_f32(a3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
            i += 16;
        }
        while i + 4 <= n { a0 = vfmaq_f32(a0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i))); i += 4; }
        let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        while i < n { s += *pa.add(i) * *pb.add(i); i += 1; }
        s
    } }
    unsafe fn sumsq_neon(a: &[f32]) -> f32 { unsafe { dot_neon(a, a) } }

    unsafe fn max_neon(a: &[f32]) -> f32 { unsafe {
        let n = a.len();
        let p = a.as_ptr();
        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 4 <= n { acc = vmaxnmq_f32(acc, vld1q_f32(p.add(i))); i += 4; } // maxNum: NaN lanes ignored
        let mut m = vmaxnmvq_f32(acc);
        while i < n { m = m.max(*p.add(i)); i += 1; }
        m
    } }

    unsafe fn axpy_neon(a: f32, x: &[f32], y: &mut [f32]) { gen_loops::axpy::<true>(a, x, y) }
    unsafe fn scale_neon(x: &mut [f32], k: f32) { gen_loops::scale(x, k) }
    unsafe fn mul3_neon(x: &[f32], w: &[f32], k: f32, o: &mut [f32]) { gen_loops::mul3(x, w, k, o) }
    unsafe fn rope_neon(h: &mut [f32], c: &[f32], s: &[f32]) { gen_loops::rope::<true>(h, c, s) }
    unsafe fn bf16_neon(s: &[u16], d: &mut [f32]) { gen_loops::bf16(s, d) }

    #[inline]
    unsafe fn exp_q(x: float32x4_t) -> float32x4_t { unsafe {
        let lt = vcltq_f32(x, vdupq_n_f32(EXP_LO));
        // vminq/vmaxq propagate NaN (unlike vminnm/vmaxnm)
        let xc = vmaxq_f32(vdupq_n_f32(EXP_LO), vminq_f32(vdupq_n_f32(EXP_HI), x));
        let n = vrndnq_f32(vmulq_f32(xc, vdupq_n_f32(LOG2E)));
        let r = vfmsq_f32(xc, n, vdupq_n_f32(LN2_HI));
        let r = vfmsq_f32(r, n, vdupq_n_f32(LN2_LO));
        let mut p = vdupq_n_f32(C7);
        p = vfmaq_f32(vdupq_n_f32(C6), p, r);
        p = vfmaq_f32(vdupq_n_f32(C5), p, r);
        p = vfmaq_f32(vdupq_n_f32(C4), p, r);
        p = vfmaq_f32(vdupq_n_f32(C3), p, r);
        p = vfmaq_f32(vdupq_n_f32(C2), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
        let v = vmulq_f32(p, vreinterpretq_f32_s32(e));
        vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(v), lt))
    } }

    unsafe fn exp_neon(x: &mut [f32], m: f32) -> f32 { unsafe {
        let n = x.len();
        let p = x.as_mut_ptr();
        let mv = vdupq_n_f32(m);
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + 4 <= n {
            let e = exp_q(vsubq_f32(vld1q_f32(p.add(i)), mv));
            vst1q_f32(p.add(i), e);
            acc = vaddq_f32(acc, e);
            i += 4;
        }
        let mut s = vaddvq_f32(acc);
        while i < n { let e = scalar::exp_poly(*p.add(i) - m); *p.add(i) = e; s += e; i += 1; }
        s
    } }

    unsafe fn silu_neon(g: &mut [f32], u: &[f32]) { unsafe {
        let n = g.len();
        let (pg, pu) = (g.as_mut_ptr(), u.as_ptr());
        let one = vdupq_n_f32(1.0);
        let mut i = 0;
        while i + 4 <= n {
            let v = vld1q_f32(pg.add(i));
            let d = vaddq_f32(one, exp_q(vnegq_f32(v)));
            vst1q_f32(pg.add(i), vmulq_f32(vdivq_f32(v, d), vld1q_f32(pu.add(i))));
            i += 4;
        }
        while i < n { *pg.add(i) = scalar::silu_poly(*pg.add(i), *pu.add(i)); i += 1; }
    } }

    /// `fcvtl` widens four binary16 lanes exactly; it is part of base ARMv8 (no fp16 extension needed).
    unsafe fn f16_neon(s: &[u16], d: &mut [f32]) { unsafe {
        let n = s.len();
        let (ps, pd) = (s.as_ptr(), d.as_mut_ptr());
        let mut i = 0;
        while i + 4 <= n {
            let h: uint16x4_t = vld1_u16(ps.add(i));
            let r: float32x4_t;
            std::arch::asm!("fcvtl {r:v}.4s, {h:v}.4h", r = lateout(vreg) r, h = in(vreg) h, options(pure, nomem, nostack));
            vst1q_f32(pd.add(i), r);
            i += 4;
        }
        while i < n { *pd.add(i) = scalar::f16_to_f32(*ps.add(i)); i += 1; }
    } }

    /// Baseline NEON, exact: widening 8×8→16 multiplies, pairwise-accumulated into 32 bits.
    unsafe fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 { unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = vdupq_n_s32(0);
        let mut i = 0;
        while i + 16 <= n {
            let (x, y) = (vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
            acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
            acc = vpadalq_s16(acc, vmull_high_s8(x, y));
            i += 16;
        }
        let mut s = vaddvq_s32(acc);
        while i < n { s = s.wrapping_add(*pa.add(i) as i32 * *pb.add(i) as i32); i += 1; }
        s
    } }

    /// `sdot`: four signed 8-bit products summed into each 32-bit lane.
    #[inline]
    #[target_feature(enable = "dotprod")]
    unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t { unsafe {
        let mut acc = acc;
        std::arch::asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc, a = in(vreg) a, b = in(vreg) b,
            options(pure, nomem, nostack)
        );
        acc
    } }

    #[target_feature(enable = "dotprod")]
    unsafe fn dot_i8_dot(a: &[i8], b: &[i8]) -> i32 { unsafe {
        let n = a.len();
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let (mut a0, mut a1) = (vdupq_n_s32(0), vdupq_n_s32(0));
        let mut i = 0;
        while i + 32 <= n {
            a0 = sdot(a0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i)));
            a1 = sdot(a1, vld1q_s8(pa.add(i + 16)), vld1q_s8(pb.add(i + 16)));
            i += 32;
        }
        while i + 16 <= n { a0 = sdot(a0, vld1q_s8(pa.add(i)), vld1q_s8(pb.add(i))); i += 16; }
        let mut s = vaddvq_s32(vaddq_s32(a0, a1));
        while i < n { s = s.wrapping_add(*pa.add(i) as i32 * *pb.add(i) as i32); i += 1; }
        s
    } }

    /// (nibble − 8) as signed bytes in element order.
    #[inline]
    unsafe fn q4_signed(qs: *const u8) -> (int8x16_t, int8x16_t) { unsafe {
        let v = vld1q_u8(qs);
        let eight = vdupq_n_s8(8);
        let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, vdupq_n_u8(0x0f))), eight);
        let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), eight);
        (lo, hi)
    } }

    unsafe fn q4blk_neon(qs: *const u8, y: *const i8) -> i32 { unsafe {
        let (lo, hi) = q4_signed(qs);
        let (y0, y1) = (vld1q_s8(y), vld1q_s8(y.add(16)));
        let mut acc = vdupq_n_s32(0);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(lo), vget_low_s8(y0)));
        acc = vpadalq_s16(acc, vmull_high_s8(lo, y0));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(hi), vget_low_s8(y1)));
        acc = vpadalq_s16(acc, vmull_high_s8(hi, y1));
        vaddvq_s32(acc)
    } }

    #[target_feature(enable = "dotprod")]
    unsafe fn q4blk_dot(qs: *const u8, y: *const i8) -> i32 { unsafe {
        let (lo, hi) = q4_signed(qs);
        let z = vdupq_n_s32(0);
        vaddvq_s32(sdot(sdot(z, lo, vld1q_s8(y)), hi, vld1q_s8(y.add(16))))
    } }

    unsafe fn q4row_neon(w: *const u8, a: *const u8, n: usize) -> f32 { unsafe {
        let mut acc = 0f32;
        for b in 0..n {
            let (wp, ap) = (w.add(b * 18), a.add(b * 34));
            let d = scalar::f16_to_f32(u16::from_le_bytes([*wp, *wp.add(1)])) * scalar::f16_to_f32(u16::from_le_bytes([*ap, *ap.add(1)]));
            acc = d.mul_add(q4blk_neon(wp.add(2), ap.add(2) as *const i8) as f32, acc);
        }
        acc
    } }
    #[target_feature(enable = "dotprod")]
    unsafe fn q4row_dot(w: *const u8, a: *const u8, n: usize) -> f32 { unsafe {
        let mut acc = 0f32;
        for b in 0..n {
            let (wp, ap) = (w.add(b * 18), a.add(b * 34));
            let d = scalar::f16_to_f32(u16::from_le_bytes([*wp, *wp.add(1)])) * scalar::f16_to_f32(u16::from_le_bytes([*ap, *ap.add(1)]));
            acc = d.mul_add(q4blk_dot(wp.add(2), ap.add(2) as *const i8) as f32, acc);
        }
        acc
    } }

    shim_dot!(s_dot_neon, dot_neon);
    shim_unary!(s_sumsq_neon, sumsq_neon);
    shim_unary!(s_max_neon, max_neon);
    shim_axpy!(s_axpy_neon, axpy_neon);
    shim_scale!(s_scale_neon, scale_neon);
    shim_mul3!(s_mul3_neon, mul3_neon);
    shim_exp!(s_exp_neon, exp_neon);
    shim_silu!(s_silu_neon, silu_neon);
    shim_rope!(s_rope_neon, rope_neon);
    shim_conv!(s_f16_neon, f16_neon);
    shim_conv!(s_bf16_neon, bf16_neon);
    shim_i8!(s_i8_neon, dot_i8_neon);
    shim_q4blk!(s_q4b_neon, q4blk_neon);
    shim_q4row!(s_q4r_neon, q4row_neon);
    shim_i8!(s_i8_dot, dot_i8_dot);
    shim_q4blk!(s_q4b_dot, q4blk_dot);
    shim_q4row!(s_q4r_dot, q4row_dot);
}

// ------------------------------------------------------------------------------- tests

/// Test helper: f32 → little-endian binary16 bytes (round to nearest even, normal range only).
#[cfg(test)]
trait ToHalf { fn to_le_bytes_f16(self) -> [u8; 2]; }
#[cfg(test)]
impl ToHalf for f32 {
    fn to_le_bytes_f16(self) -> [u8; 2] {
        let b = self.to_bits();
        let (s, e, m) = ((b >> 16) & 0x8000, ((b >> 23) & 0xff) as i32 - 127 + 15, b & 0x7f_ffff);
        let h = if e <= 0 { s } else if e >= 31 { s | 0x7c00 } else {
            let mut h = s | ((e as u32) << 10) | (m >> 13);
            if m & 0x1000 != 0 && (m & 0x2fff) != 0 { h += 1; }
            h
        };
        (h as u16).to_le_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static PRECISE_LOCK: Mutex<()> = Mutex::new(());

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn unit(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0 }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> { (0..n).map(|_| self.unit() * scale).collect() }
        fn i8s(&mut self, n: usize) -> Vec<i8> { (0..n).map(|_| (self.next() >> 56) as u8 as i8).collect() }
    }

    const LENS: [usize; 9] = [131, 160, 255, 256, 257, 511, 512, 1000, 4097];
    fn lens() -> impl Iterator<Item = usize> { (0..=130).chain(LENS) }

    #[test]
    fn variants_are_listed_and_the_active_one_is_among_them() {
        let all = available();
        assert_eq!(all[0].name, "scalar");
        assert!(all.iter().any(|k| k.name == active()), "{} not in {:?}", active(), all.iter().map(|k| k.name).collect::<Vec<_>>());
        let mut names: Vec<_> = all.iter().map(|k| k.name).collect();
        names.dedup();
        assert_eq!(names.len(), all.len());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn every_detected_x86_feature_gives_a_variant() {
        let names: Vec<_> = available().iter().map(|k| k.name).collect();
        let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
        assert_eq!(names.contains(&"avx2+fma"), avx2);
        assert_eq!(names.contains(&"avx2+fma+avxvnni"), avx2 && is_x86_feature_detected!("avxvnni"));
        let a512 = avx2 && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vl");
        assert_eq!(names.contains(&"avx512"), a512);
        assert_eq!(names.contains(&"avx512+vnni"), a512 && is_x86_feature_detected!("avx512vnni"));
    }

    #[test]
    fn float_reductions_match_an_f64_reference() {
        for k in available() {
            let mut r = Rng(0x1234_5678_9abc_def1);
            for n in lens() {
                let a = r.vec(n + 3, 4.0);
                let b = r.vec(n + 3, 4.0);
                for off in 0..3 {
                    let (a, b) = (&a[off..off + n], &b[off..off + n]);
                    let want: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
                    let mag: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 * *y as f64).abs()).sum();
                    let got = (k.dot_f32)(a, b) as f64;
                    assert!((got - want).abs() <= 2e-6 * mag + 1e-30, "{} dot n={n} off={off}: {got} vs {want}", k.name);
                    let want: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
                    let got = (k.sumsq)(a) as f64;
                    assert!((got - want).abs() <= 2e-6 * want + 1e-30, "{} sumsq n={n}", k.name);
                }
            }
        }
    }

    #[test]
    fn max_matches_the_scalar_fold_with_nan_and_infinities() {
        for k in available() {
            let mut r = Rng(77);
            for n in lens() {
                let mut v = r.vec(n, 100.0);
                if n > 3 {
                    v[n / 2] = f32::NAN;
                    v[n - 1] = f32::INFINITY * if r.next() & 1 == 0 { 1.0 } else { -1.0 };
                    v[1] = f32::NAN;
                }
                let want = scalar::max(&v);
                let got = (k.max)(&v);
                assert!(got == want || (got.is_nan() && want.is_nan()), "{} max n={n}: {got} vs {want}", k.name);
            }
            assert_eq!((k.max)(&[]), f32::NEG_INFINITY);
            assert_eq!((k.max)(&[f32::NAN; 40]), f32::NEG_INFINITY);
        }
    }

    #[test]
    fn elementwise_kernels_match_the_scalar_ones() {
        for k in available() {
            let mut r = Rng(4242);
            for n in lens() {
                let x = r.vec(n, 3.0);
                let w = r.vec(n, 3.0);
                let y0 = r.vec(n, 3.0);
                let mut y = y0.clone();
                (k.axpy)(0.37, &x, &mut y);
                for i in 0..n { assert!((y[i] - (y0[i] + 0.37 * x[i])).abs() <= 1e-6 * (y0[i].abs() + x[i].abs() + 1.0), "{} axpy", k.name); }
                let mut s = x.clone();
                (k.scale)(&mut s, -1.25);
                for i in 0..n { assert_eq!(s[i], x[i] * -1.25, "{} scale", k.name); }
                let mut o = vec![9.0; n];
                (k.mul3)(&x, &w, 0.5, &mut o);
                for i in 0..n { assert_eq!(o[i], x[i] * 0.5 * w[i], "{} mul3", k.name); }
            }
            // unequal lengths: only the common prefix is touched
            let mut y = vec![1.0f32; 10];
            (k.axpy)(2.0, &[1.0; 4], &mut y);
            assert_eq!(&y[..4], &[3.0; 4]);
            assert_eq!(&y[4..], &[1.0; 6]);
        }
    }

    #[test]
    fn the_polynomial_exp_is_accurate_over_its_whole_range() {
        let mut worst = 0f64;
        let mut x = -87.3f32;
        while x < 88.0 {
            let want = (x as f64).exp();
            let got = scalar::exp_poly(x) as f64;
            worst = worst.max(((got - want) / want).abs());
            x += 0.0137;
        }
        assert!(worst < 4e-7, "worst relative error {worst}");
        assert_eq!(scalar::exp_poly(-200.0), 0.0);
        assert_eq!(scalar::exp_poly(f32::NEG_INFINITY), 0.0);
        assert!(scalar::exp_poly(f32::NAN).is_nan());
        assert!((scalar::exp_poly(0.0) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn exp_sub_sum_matches_libm_in_every_variant() {
        for k in available() {
            let mut r = Rng(9);
            for n in lens() {
                let mut x = r.vec(n, 20.0);
                if n > 6 { x[2] = f32::NEG_INFINITY; x[4] = -1000.0; }
                let m = scalar::max(&x);
                let want: Vec<f64> = x.iter().map(|v| ((*v - m) as f64).exp()).collect();
                let want_sum: f64 = want.iter().sum();
                let mut got = x.clone();
                let sum = (k.exp_sub_sum)(&mut got, m) as f64;
                for i in 0..n { assert!((got[i] as f64 - want[i]).abs() <= 1e-6 * want[i] + 1e-37, "{} exp n={n} i={i}: {} vs {}", k.name, got[i], want[i]); }
                assert!((sum - want_sum).abs() <= 2e-6 * want_sum + 1e-30, "{} sum n={n}", k.name);
                if n > 6 { assert_eq!(got[2], 0.0); assert_eq!(got[4], 0.0); }
            }
            let mut v = [f32::NAN, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
            (k.exp_sub_sum)(&mut v, 0.0);
            assert!(v[0].is_nan(), "{} lost a NaN", k.name);
        }
    }

    #[test]
    fn silu_mul_matches_the_f64_reference() {
        for k in available() {
            let mut r = Rng(31);
            for n in lens() {
                let mut g = r.vec(n, 12.0);
                if n > 8 { g[1] = -100.0; g[3] = 100.0; g[5] = 0.0; }
                let u = r.vec(n, 2.0);
                let want: Vec<f64> = g.iter().zip(&u).map(|(g, u)| *g as f64 / (1.0 + (-(*g as f64)).exp()) * *u as f64).collect();
                let mut got = g.clone();
                (k.silu_mul)(&mut got, &u);
                for i in 0..n {
                    assert!((got[i] as f64 - want[i]).abs() <= 3e-6 * want[i].abs() + 1e-30, "{} silu n={n} i={i}: {} vs {}", k.name, got[i], want[i]);
                }
            }
        }
    }

    #[test]
    fn rope_rotation_matches_the_f64_reference_and_leaves_extra_lanes_alone() {
        for k in available() {
            let mut r = Rng(5);
            for half in (0..=70).chain([128, 129]) {
                let c: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).cos()).collect();
                let s: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).sin()).collect();
                for extra in 0..2 {
                    let h0 = r.vec(2 * half + extra, 2.0);
                    let mut h = h0.clone();
                    (k.rope)(&mut h, &c, &s);
                    for i in 0..half {
                        let (a, b) = (h0[i] as f64, h0[i + half] as f64);
                        let (cc, ss) = (c[i] as f64, s[i] as f64);
                        assert!((h[i] as f64 - (a * cc - b * ss)).abs() < 1e-6 * (a.abs() + b.abs() + 1.0), "{} rope", k.name);
                        assert!((h[i + half] as f64 - (a * ss + b * cc)).abs() < 1e-6 * (a.abs() + b.abs() + 1.0), "{} rope", k.name);
                    }
                    if extra == 1 { assert_eq!(h[2 * half], h0[2 * half]); }
                }
            }
            // tables longer than the head: only what fits is rotated, nothing panics
            let mut h = vec![1.0f32; 6];
            (k.rope)(&mut h, &[1.0; 10], &[0.0; 10]);
            assert_eq!(h, vec![1.0; 6]);
        }
    }

    #[test]
    fn half_precision_conversions_are_exact() {
        for k in available() {
            let all: Vec<u16> = (0..=u16::MAX).collect();
            for (off, len) in [(0usize, 65536usize), (1, 65535), (3, 1000), (0, 7), (5, 0)] {
                let src = &all[off..off + len];
                let mut got = vec![0f32; len];
                (k.f16_to_f32)(src, &mut got);
                for (i, (&h, &g)) in src.iter().zip(&got).enumerate() {
                    let want = scalar::f16_to_f32(h);
                    assert!(g.to_bits() == want.to_bits() || (g.is_nan() && want.is_nan()), "{} f16 {h:#06x} at {i}: {g} vs {want}", k.name);
                }
                (k.bf16_to_f32)(src, &mut got);
                for (&h, &g) in src.iter().zip(&got) { assert_eq!(g.to_bits(), (h as u32) << 16, "{} bf16", k.name); }
            }
            // shorter destination: only that much is written
            let mut d = [7.0f32; 3];
            (k.f16_to_f32)(&[0x3c00; 9], &mut d);
            assert_eq!(d, [1.0; 3]);
        }
    }

    #[test]
    fn the_scalar_f16_decoder_agrees_with_the_quant_module() {
        for h in 0..=u16::MAX {
            let (a, b) = (scalar::f16_to_f32(h), crate::quant::f16_to_f32(h));
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{h:#06x}");
        }
    }

    fn dot_ref(a: &[i8], b: &[i8]) -> i32 { a.iter().zip(b).map(|(x, y)| *x as i64 * *y as i64).sum::<i64>() as i32 }

    #[test]
    fn integer_dot_is_exact_including_the_extremes() {
        for k in available() {
            let mut r = Rng(2024);
            for n in lens() {
                let a = r.i8s(n + 2);
                let b = r.i8s(n + 2);
                for off in 0..2 {
                    assert_eq!((k.dot_i8)(&a[off..off + n], &b[off..off + n]), dot_ref(&a[off..off + n], &b[off..off + n]), "{} i8 n={n}", k.name);
                }
                for (x, y) in [(-128i8, -128i8), (-128, 127), (127, 127), (0, -128), (127, -128)] {
                    let (a, b) = (vec![x; n], vec![y; n]);
                    assert_eq!((k.dot_i8)(&a, &b), dot_ref(&a, &b), "{} i8 {x}*{y} n={n}", k.name);
                }
            }
            assert_eq!((k.dot_i8)(&[1, 2, 3], &[4, 5]), 14); // common prefix
        }
    }

    #[test]
    fn q4_0_block_and_row_match_the_scalar_oracle() {
        for k in available() {
            let mut r = Rng(88);
            for _ in 0..300 {
                let mut qs = [0u8; 16];
                for q in &mut qs { *q = (r.next() >> 56) as u8; }
                let mut y = [0i8; 32];
                for v in &mut y { *v = (r.next() >> 56) as u8 as i8; }
                assert_eq!((k.q4_0_block)(&qs, &y), scalar::q4_0_block(&qs, &y), "{} q4 block", k.name);
            }
            for (qs, y) in [([0x00u8; 16], [-128i8; 32]), ([0xffu8; 16], [-128i8; 32]), ([0xffu8; 16], [127i8; 32]), ([0x0fu8; 16], [127i8; 32])] {
                assert_eq!((k.q4_0_block)(&qs, &y), scalar::q4_0_block(&qs, &y), "{} q4 extremes", k.name);
            }
            for nb in [0usize, 1, 2, 3, 7, 33] {
                let (mut w, mut a) = (vec![], vec![]);
                for _ in 0..nb {
                    w.extend((0.02 + r.unit().abs() * 0.1).to_le_bytes_f16());
                    for _ in 0..16 { w.push((r.next() >> 56) as u8); }
                    a.extend((0.01 + r.unit().abs() * 0.1).to_le_bytes_f16());
                    for _ in 0..32 { a.push((r.next() >> 56) as u8); }
                }
                let want = scalar::q4_0_q8_0_row(&w, &a, nb) as f64;
                let got = (k.q4_0_q8_0_row)(&w, &a, nb) as f64;
                let mag: f64 = (0..nb).map(|b| {
                    let dw = scalar::f16_to_f32(u16::from_le_bytes([w[b * 18], w[b * 18 + 1]])) as f64;
                    let da = scalar::f16_to_f32(u16::from_le_bytes([a[b * 34], a[b * 34 + 1]])) as f64;
                    dw * da * 32.0 * 128.0 * 8.0
                }).sum();
                assert!((got - want).abs() <= 1e-5 * mag + 1e-9, "{} q4 row nb={nb}: {got} vs {want}", k.name);
            }
        }
    }

    #[test]
    fn public_row_helpers_clamp_to_the_data_they_have() {
        let w = vec![0u8; 18 * 2];
        let a = vec![0u8; 34 * 5];
        assert_eq!(q4_0_q8_0_row(&w, &a, 99), 0.0);
        assert_eq!(q8_0_q8_0_row(&[], &a, 3), 0.0);
        // q8_0 · q8_0 against a hand computation: d = 1.0 (0x3c00)
        let mut blk = vec![0x00, 0x3c];
        blk.extend((0..32).map(|i| i as u8));
        let got = q8_0_q8_0_row(&blk, &blk, 1);
        let want: i32 = (0..32).map(|i| i * i).sum();
        assert_eq!(got, want as f32);
    }

    #[test]
    fn precise_mode_is_libm_and_the_flag_round_trips() {
        let _g = PRECISE_LOCK.lock().unwrap();
        set_precise(true);
        assert!(is_precise());
        let mut v: Vec<f32> = (0..37).map(|i| -(i as f32) * 0.7).collect();
        let want: Vec<f32> = v.iter().map(|x| x.exp()).collect();
        exp_sub_sum(&mut v, 0.0);
        set_precise(false);
        assert_eq!(v, want);
        assert!(!is_precise());
    }

    #[test]
    fn hostile_lengths_never_panic() {
        for k in available() {
            let big = vec![1.0f32; 5];
            assert_eq!((k.dot_f32)(&big, &[]), 0.0);
            assert_eq!((k.sumsq)(&[]), 0.0);
            let mut e: [f32; 0] = [];
            assert_eq!((k.exp_sub_sum)(&mut e, 0.0), 0.0);
            (k.silu_mul)(&mut [], &[1.0]);
            (k.rope)(&mut [], &[], &[]);
            assert_eq!((k.dot_i8)(&[], &[]), 0);
        }
    }
}
