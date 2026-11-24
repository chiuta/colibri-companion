//! WGSL compute kernels, written from scratch. Layouts are those of [`crate::layout`]; the
//! algorithms (work split, accumulation and reduction order) are mirrored one-to-one in
//! [`crate::emu`], which is what the tests check against the CPU kernels.
use crate::layout::Format;

pub const MATVEC_WG: u32 = 64;
/// Output rows computed by one matvec workgroup (32 lanes per row).
pub const MATVEC_ROWS_PER_WG: u32 = 2;
pub const ROW_LANES: usize = 32;
pub const NORM_WG: u32 = 256;
pub const ATTN_WG: u32 = 256;
pub const ELEM_WG: u32 = 64;

/// Uniform block of the matvec kernels (8 × u32).
pub const MATVEC_PARAMS: &str = "rows, cols, row_words, n_items, row_off, out_stride, x_stride, nwgx";

const MATVEC_HEAD: &str = r#"
struct P { rows: u32, cols: u32, row_words: u32, n_items: u32, row_off: u32, out_stride: u32, x_stride: u32, nwgx: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, 64>;

fn ubyte(word: u32, i: u32) -> u32 { return (word >> (i * 8u)) & 255u; }
fn sbyte(word: u32, i: u32) -> i32 { return bitcast<i32>(word << (24u - i * 8u)) >> 24u; }
fn wbyte(base: u32, off: u32) -> u32 { return ubyte(w[base + (off >> 2u)], off & 3u); }

fn h2f(h: u32) -> f32 {
    let s = (h & 0x8000u) << 16u;
    let e = (h >> 10u) & 31u;
    let m = h & 1023u;
    if (e == 0u) {
        let v = f32(m) * 5.960464477539063e-8;
        return select(v, -v, s != 0u);
    }
    if (e == 31u) { return bitcast<f32>(s | 0x7f800000u | (m << 13u)); }
    return bitcast<f32>(s | ((e + 112u) << 23u) | (m << 13u));
}
"#;

const MATVEC_MAIN: &str = r#"
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let wg = wid.x + wid.y * p.nwgx;
    let lane = t & 31u;
    let row = wg * 2u + (t >> 5u);
    let valid = row < p.rows;
    let rr = select(0u, row, valid);
    let base = rr * p.row_words;
    let xo = wid.z * p.x_stride;
    var acc = 0.0;
    for (var it = lane; it < p.n_items; it = it + 32u) {
        acc = acc + item_dot(base, it, xo);
    }
    part[t] = acc;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s = s >> 1u) {
        if (lane < s) { part[t] = part[t] + part[t + s]; }
        workgroupBarrier();
    }
    if (valid && lane == 0u) { y[wid.z * p.out_stride + p.row_off + row] = part[t]; }
}
"#;

fn item_dot(f: Format) -> &'static str {
    match f {
        Format::F32 => r#"
fn item_dot(base: u32, it: u32, xo: u32) -> f32 {
    var s = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) {
        let e = it * 32u + j;
        if (e < p.cols) { s = s + bitcast<f32>(w[base + e]) * x[xo + e]; }
    }
    return s;
}"#,
        Format::F16 => r#"
fn item_dot(base: u32, it: u32, xo: u32) -> f32 {
    var s = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) {
        let e = it * 32u + j;
        if (e < p.cols) {
            let hw = w[base + (e >> 1u)];
            s = s + h2f(select(hw & 65535u, hw >> 16u, (e & 1u) == 1u)) * x[xo + e];
        }
    }
    return s;
}"#,
        Format::Q8_0 => r#"
fn item_dot(base: u32, it: u32, xo: u32) -> f32 {
    let b = base + it * 9u;
    let d = bitcast<f32>(w[b]);
    var s = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) {
        s = s + f32(sbyte(w[b + 1u + (j >> 2u)], j & 3u)) * x[xo + it * 32u + j];
    }
    return d * s;
}"#,
        Format::Q4_0 => r#"
fn item_dot(base: u32, it: u32, xo: u32) -> f32 {
    let b = base + it * 5u;
    let d = bitcast<f32>(w[b]);
    var s = 0.0;
    for (var j = 0u; j < 16u; j = j + 1u) {
        let q = ubyte(w[b + 1u + (j >> 2u)], j & 3u);
        let lo = f32(i32(q & 15u) - 8);
        let hi = f32(i32(q >> 4u) - 8);
        s = s + lo * x[xo + it * 32u + j] + hi * x[xo + it * 32u + 16u + j];
    }
    return d * s;
}"#,
        Format::Q4K => r#"
fn sc_min(sb: u32, j: u32) -> vec2<u32> {
    if (j < 4u) { return vec2<u32>(wbyte(sb, j) & 63u, wbyte(sb, j + 4u) & 63u); }
    let a = wbyte(sb, j + 4u);
    return vec2<u32>((a & 15u) | ((wbyte(sb, j - 4u) >> 6u) << 4u), (a >> 4u) | ((wbyte(sb, j) >> 6u) << 4u));
}
fn item_dot(base: u32, it: u32, xo: u32) -> f32 {
    let blk = it >> 3u;
    let g = it & 7u;
    let b = base + blk * 36u;
    let dd = vec2<f32>(h2f(w[b] & 65535u), h2f(w[b] >> 16u));
    let sm = sc_min(b + 1u, g);
    let c = g >> 1u;
    let high = (g & 1u) == 1u;
    var sq = 0.0;
    var sx = 0.0;
    for (var l = 0u; l < 32u; l = l + 1u) {
        let byte = wbyte(b, 16u + c * 32u + l);
        let q = select(byte & 15u, byte >> 4u, high);
        let xv = x[xo + blk * 256u + g * 32u + l];
        sq = sq + f32(q) * xv;
        sx = sx + xv;
    }
    return dd.x * f32(sm.x) * sq - dd.y * f32(sm.y) * sx;
}"#,
        Format::Q6K => r#"
fn item_dot(base: u32, it: u32, xo: u32) -> f32 {
    let blk = it >> 4u;
    let g = it & 15u;
    let b = base + blk * 53u;
    let n = g >> 3u;
    let is = g & 1u;
    let quarter = (g & 7u) >> 1u;
    let d = h2f(w[b + 52u] & 65535u);
    let sc = f32(sbyte(w[b + ((192u + g) >> 2u)], (192u + g) & 3u));
    var s = 0.0;
    for (var l = is * 16u; l < is * 16u + 16u; l = l + 1u) {
        let qlb = wbyte(b, n * 64u + (quarter & 1u) * 32u + l);
        let nib = select(qlb & 15u, qlb >> 4u, quarter >= 2u);
        let h = (wbyte(b, 128u + n * 32u + l) >> (2u * quarter)) & 3u;
        let q = i32(nib | (h << 4u)) - 32;
        s = s + f32(q) * x[xo + blk * 256u + g * 16u + (l - is * 16u)];
    }
    return d * sc * s;
}"#,
    }
}

/// Matrix–vector kernel for one weight format. Dispatch: `ceil(rows / 2)` workgroups over x/y,
/// `batch` over z.
pub fn matvec(f: Format) -> String { format!("{MATVEC_HEAD}{}{MATVEC_MAIN}", item_dot(f)) }

/// RMS norm over `n` values per workgroup (one workgroup per batch row).
pub const RMSNORM: &str = r#"
struct P { n: u32, eps: f32, x_stride: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let o = wid.x * p.x_stride;
    var acc = 0.0;
    for (var i = t; i < p.n; i = i + 256u) { let v = x[o + i]; acc = acc + v * v; }
    part[t] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { part[t] = part[t] + part[t + s]; }
        workgroupBarrier();
    }
    let k = 1.0 / sqrt(part[0] / f32(p.n) + p.eps);
    for (var i = t; i < p.n; i = i + 256u) { y[o + i] = x[o + i] * k * wt[i]; }
}
"#;

/// Rotate-half RoPE in place; `cs` holds `(cos, sin)` per frequency for the current position.
pub const ROPE: &str = r#"
struct P { n_pairs: u32, head_dim: u32, half_dim: u32, nwgx: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> cs: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = (wid.x + wid.y * p.nwgx) * 64u + t;
    if (i < p.n_pairs) {
        let h = i / p.half_dim;
        let k = i % p.half_dim;
        let o = h * p.head_dim + k;
        let a = x[o];
        let b = x[o + p.half_dim];
        let c = cs[k].x;
        let s = cs[k].y;
        x[o] = a * c - b * s;
        x[o + p.half_dim] = a * s + b * c;
    }
}
"#;

/// `g = silu(g) * u`.
pub const SILU_MUL: &str = r#"
struct P { n: u32, nwgx: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> g: array<f32>;
@group(0) @binding(2) var<storage, read> u: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = (wid.x + wid.y * p.nwgx) * 64u + t;
    if (i < p.n) { let v = g[i]; g[i] = v / (1.0 + exp(-v)) * u[i]; }
}
"#;

/// `a += b`.
pub const ADD: &str = r#"
struct P { n: u32, nwgx: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = (wid.x + wid.y * p.nwgx) * 64u + t;
    if (i < p.n) { a[i] = a[i] + b[i]; }
}
"#;

/// Softmax attention, one workgroup per query head. Scores go through a scratch buffer of
/// `n_heads × cap` floats; keys/values are laid out `[pos][kv_head][head_dim]`.
pub const ATTENTION: &str = r#"
struct P { n_heads: u32, kv_heads: u32, head_dim: u32, n_pos: u32, start: u32, cap: u32, scale: f32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> kc: array<f32>;
@group(0) @binding(3) var<storage, read> vc: array<f32>;
@group(0) @binding(4) var<storage, read_write> sc: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.x;
    let kvh = h / (p.n_heads / p.kv_heads);
    let row = p.kv_heads * p.head_dim;
    let qb = h * p.head_dim;

    var m = -3.0e38;
    for (var s = p.start + t; s < p.n_pos; s = s + 256u) {
        var d = 0.0;
        for (var i = 0u; i < p.head_dim; i = i + 1u) { d = d + q[qb + i] * kc[s * row + kvh * p.head_dim + i]; }
        d = d * p.scale;
        sc[h * p.cap + s] = d;
        m = max(m, d);
    }
    part[t] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { part[t] = max(part[t], part[t + s]); }
        workgroupBarrier();
    }
    let mx = part[0];
    workgroupBarrier();

    var z = 0.0;
    for (var s = p.start + t; s < p.n_pos; s = s + 256u) {
        let e = exp(sc[h * p.cap + s] - mx);
        sc[h * p.cap + s] = e;
        z = z + e;
    }
    part[t] = z;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { part[t] = part[t] + part[t + s]; }
        workgroupBarrier();
    }
    let zt = part[0];
    storageBarrier();
    workgroupBarrier();

    for (var d = t; d < p.head_dim; d = d + 256u) {
        var acc = 0.0;
        for (var s = p.start; s < p.n_pos; s = s + 1u) { acc = acc + sc[h * p.cap + s] * vc[s * row + kvh * p.head_dim + d]; }
        o[qb + d] = acc / zt;
    }
}
"#;

/// Every kernel with its name, for validation and for the runner's pipeline table.
pub fn all() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = crate::layout::ALL.iter().map(|f| (format!("matvec_{}", f.name()), matvec(*f))).collect();
    for (n, s) in [("rmsnorm", RMSNORM), ("rope", ROPE), ("silu_mul", SILU_MUL), ("add", ADD), ("attention", ATTENTION)] { v.push((n.to_string(), s.to_string())); }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert!(module.entry_points.iter().any(|e| e.name == "main" && e.stage == naga::ShaderStage::Compute), "{name}: no compute entry point");
    }

    #[test]
    fn every_kernel_parses_and_validates() {
        let all = all();
        assert_eq!(all.len(), 11);
        for (n, s) in &all { validate(n, s); }
    }

    #[test]
    fn workgroup_sizes_in_the_source_match_the_constants() {
        for f in crate::layout::ALL { assert!(matvec(f).contains(&format!("@workgroup_size({MATVEC_WG})"))); }
        assert!(RMSNORM.contains(&format!("@workgroup_size({NORM_WG})")));
        assert!(ATTENTION.contains(&format!("@workgroup_size({ATTN_WG})")));
        for s in [ROPE, SILU_MUL, ADD] { assert!(s.contains(&format!("@workgroup_size({ELEM_WG})"))); }
        assert_eq!(MATVEC_WG as usize, ROW_LANES * MATVEC_ROWS_PER_WG as usize);
    }

    #[test]
    fn the_validator_really_rejects_a_broken_kernel() {
        // An ill-typed module must be refused, proving the check is live.
        let bad = "@compute @workgroup_size(1)\nfn main() { let z = 1.0 + 1u; }";
        let refused = match naga::front::wgsl::parse_str(bad) {
            Err(_) => true,
            Ok(m) => naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty()).validate(&m).is_err(),
        };
        assert!(refused);
        assert!(naga::front::wgsl::parse_str("fn main( {").is_err());
    }
}
