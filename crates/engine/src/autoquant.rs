//! Automatic mixed-precision quantisation: measure how much each tensor group *hurts the output* when it
//! is stored in each candidate format, then choose the formats that minimise the total damage for a given
//! size (or RAM) budget.
//!
//! The pipeline has three independent steps, so a caller can use any of them alone:
//!
//! 1. [`analyse`] fills a [`Sensitivity`] table (`damage[group][candidate]`) from a [`Probe`]. The damage
//!    is whatever the probe measures — normally the mean KL divergence of the logits against the
//!    full-precision reference on a small calibration set ([`kl_damage`]). Three [`Method`]s trade cost
//!    for accuracy: `Measured` runs the model for every (group, format) pair, `Proxy` never runs it and
//!    uses the (imatrix-weighted) relative weight error, and `Hybrid` measures each group once at an
//!    anchor format and extrapolates to the other formats with the proxy ratio.
//! 2. [`solve`] picks one format per group under a [`Budget`]. [`Solver::Exact`] is a multiple-choice
//!    knapsack by dynamic programming (exact on the byte grid it uses, which is the true byte count
//!    whenever the grid is fine enough, and always on the safe side of the budget); [`Solver::Greedy`]
//!    walks the convex hull of every group's size/damage trade-off by marginal benefit per byte. An
//!    optional [`SpeedModel`] adds a decode-speed penalty so that formats that are slow on this hardware
//!    are avoided.
//! 3. [`Plan::to_recipe`] turns the choice into a [`Recipe`] that `quantize.rs` consumes, and [`Report`]
//!    prints the per-layer sensitivity table, the choice and predicted-versus-measured damage.
//!
//! [`auto_quant`] runs all of it, including a verification pass that measures the *joint* damage of the
//! chosen plan and of every uniform format for comparison.
//!
//! Assumption, stated plainly: the solver treats damages as additive across groups. That holds well when
//! each group's error is small and does not hit the same weights twice; the verification step exists to
//! show how far it is off for a given model. The module does not touch a real checkpoint: a [`Probe`]
//! for [`crate::model::Model`] still has to be written. [`synthetic::SyntheticNet`] is a small residual
//! network used by the tests and the demo.
#![allow(clippy::too_many_arguments)]
use crate::eval;
use crate::hw::HardwareProfile;
use crate::quant::{DType, dequantize};
use crate::quantize::{Recipe, dtype_name, quantize_matrix};
use serde_json::{Value as Json, json};
use std::collections::HashMap;

// ------------------------------------------------------------------ errors

#[derive(Debug, PartialEq)]
pub enum AutoQuantError {
    /// The probe has no tensor groups, or the candidate list is empty.
    Empty(String),
    /// The probe failed or returned something unusable (NaN, infinity).
    Probe(String),
    /// Quantising or dequantising a tensor failed.
    Quant(String),
    /// A group has no candidate whose block size divides its row length.
    NoCandidate(String),
    /// Even the smallest format everywhere does not fit.
    Infeasible { needed: u64, budget: u64 },
    /// `Method::Proxy` / `Hybrid` needs weights the probe did not provide.
    MissingProxy(String),
    /// The cancel hook asked to stop.
    Cancelled,
    /// A base recipe name that `quantize.rs` does not know.
    BadRecipe(String),
}
impl std::fmt::Display for AutoQuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutoQuantError::Empty(s)
            | AutoQuantError::Probe(s)
            | AutoQuantError::Quant(s)
            | AutoQuantError::NoCandidate(s)
            | AutoQuantError::MissingProxy(s)
            | AutoQuantError::BadRecipe(s) => write!(f, "{s}"),
            AutoQuantError::Infeasible { needed, budget } => write!(
                f,
                "the smallest allowed formats need {needed} bytes but the budget is {budget}"
            ),
            AutoQuantError::Cancelled => write!(f, "cancelled"),
        }
    }
}
impl std::error::Error for AutoQuantError {}

// ------------------------------------------------------------------ progress and cancel

#[derive(Clone, Debug)]
pub struct Progress {
    pub phase: &'static str,
    pub done: usize,
    pub total: usize,
    pub detail: String,
}

/// Optional progress callback and cancel check, polled between probe runs.
#[derive(Default)]
pub struct Control<'a> {
    pub progress: Option<&'a mut dyn FnMut(&Progress)>,
    pub cancel: Option<&'a dyn Fn() -> bool>,
}
impl<'a> Control<'a> {
    pub fn none() -> Control<'static> {
        Control {
            progress: None,
            cancel: None,
        }
    }
    fn tick(
        &mut self,
        phase: &'static str,
        done: usize,
        total: usize,
        detail: String,
    ) -> Result<(), AutoQuantError> {
        if self.cancel.is_some_and(|c| c()) {
            return Err(AutoQuantError::Cancelled);
        }
        if let Some(p) = self.progress.as_mut() {
            p(&Progress {
                phase,
                done,
                total,
                detail,
            });
        }
        Ok(())
    }
}

// ------------------------------------------------------------------ groups and probes

/// Tensors that are quantised together and share one format. `tensors` are the exact GGUF names that
/// end up as recipe overrides; all must have the same row length `cols`.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub name: String,
    pub tensors: Vec<String>,
    pub layer: Option<usize>,
    pub params: u64,
    pub cols: usize,
}

/// Weights of a group, for the cheap proxies. `imp` is the per-column importance (imatrix) if known;
/// `scale` multiplies the proxy (an estimate of how strongly the output depends on this group, 1.0 if
/// unknown).
#[derive(Clone, Debug)]
pub struct ProxyData {
    pub data: Vec<f32>,
    pub cols: usize,
    pub imp: Option<Vec<f32>>,
    pub scale: f64,
}

/// Something that can say how much the model's output suffers when tensors are quantised.
pub trait Probe {
    fn groups(&self) -> Vec<Group>;
    /// Bytes of everything that is not in a group (norms, router, anything kept at full precision).
    fn fixed_bytes(&self) -> u64 {
        0
    }
    /// Damage (≥ 0, e.g. mean KL) when only `group` is stored as `dtype` and everything else is exact.
    fn damage(&mut self, group: usize, dtype: DType) -> Result<f64, String>;
    /// Damage when every `(group, dtype)` pair in `assignment` is applied at once.
    fn joint_damage(&mut self, assignment: &[(usize, DType)]) -> Result<f64, String>;
    fn proxy_data(&self, _group: usize) -> Option<ProxyData> {
        None
    }
}

/// Mean KL divergence of `test` logits from `reference` logits, the usual damage measure.
pub fn kl_damage(reference: &[Vec<f32>], test: &[Vec<f32>]) -> Result<f64, String> {
    let s = eval::kl_divergence(reference, test, None, None).map_err(|e| e.to_string())?;
    if s.mean.is_finite() {
        Ok(s.mean.max(0.0))
    } else {
        Err("non-finite KL divergence".into())
    }
}

// ------------------------------------------------------------------ bits, speed, proxies

pub fn bits_per_weight(t: DType) -> f64 {
    t.geometry()
        .map_or(f64::INFINITY, |(e, b)| b as f64 * 8.0 / e as f64)
}

/// Relative decode cost per weight of each format, in "bit equivalents": the bits that must be streamed
/// from memory plus an unpacking overhead. The overheads are **heuristics** (small for formats with
/// simple unpacking, larger for 2- and 3-bit K-quants and when the CPU has no wide integer SIMD); they
/// are not measurements.
#[derive(Clone, Debug)]
pub struct SpeedModel {
    overhead: HashMap<DType, f64>,
}
impl SpeedModel {
    /// Memory traffic only.
    pub fn flat() -> SpeedModel {
        SpeedModel {
            overhead: HashMap::new(),
        }
    }
    pub fn from_isa(isa: &[&str]) -> SpeedModel {
        let has = |f: &str| isa.contains(&f);
        let fast = (has("avx2") && has("fma")) || has("avx512f") || (has("neon") && has("dotprod"));
        let k = if fast { 1.0 } else { 2.5 };
        let l = if fast { 1.0 } else { 3.0 };
        let table = [
            (DType::Q2K, 0.5 * k),
            (DType::Q3K, 0.7 * k),
            (DType::Q4K, 0.2 * k),
            (DType::Q5K, 0.3 * k),
            (DType::Q6K, 0.3 * k),
            (DType::Q4_0, 0.0),
            (DType::Q4_1, 0.1 * l),
            (DType::Q5_0, 0.2 * l),
            (DType::Q5_1, 0.2 * l),
            (DType::Q8_0, 0.0),
            (DType::F16, 0.5 * l),
            (DType::BF16, 0.5 * l),
        ];
        SpeedModel {
            overhead: table.into_iter().collect(),
        }
    }
    pub fn from_profile(p: &HardwareProfile) -> SpeedModel {
        SpeedModel::from_isa(&p.cpu.isa)
    }
    pub fn with_overhead(mut self, t: DType, bits: f64) -> SpeedModel {
        self.overhead.insert(t, bits.max(0.0));
        self
    }
    pub fn cost_per_param(&self, t: DType) -> f64 {
        bits_per_weight(t) + self.overhead.get(&t).copied().unwrap_or(0.0)
    }
}

/// Quantise `data` (row-major, rows of `cols`) to `dtype` and decode it again.
pub fn roundtrip(
    dtype: DType,
    data: &[f32],
    cols: usize,
    imp: Option<&[f32]>,
    threads: usize,
) -> Result<Vec<f32>, AutoQuantError> {
    if dtype == DType::F32 {
        return Ok(data.to_vec());
    }
    let bytes = quantize_matrix(dtype, data, cols, imp, threads)
        .map_err(|e| AutoQuantError::Quant(format!("{}: {e}", dtype_name(dtype))))?;
    let mut out = Vec::with_capacity(data.len());
    dequantize(dtype, &bytes, data.len(), &mut out)
        .map_err(|e| AutoQuantError::Quant(format!("{}: {e}", dtype_name(dtype))))?;
    Ok(out)
}

/// Relative output-weighted quantisation error `Σ w_c (x-x̂)² / Σ w_c x²` of a matrix, where `w_c` is the
/// column importance (all ones without an imatrix). It needs no model run, so it scales to models that
/// are too big to evaluate repeatedly. It cannot see how strongly later layers amplify a group's error;
/// pair it with [`Method::Hybrid`] (or `ProxyData::scale`) when that matters.
pub fn proxy_damage(
    dtype: DType,
    data: &[f32],
    cols: usize,
    imp: Option<&[f32]>,
    threads: usize,
) -> Result<f64, AutoQuantError> {
    if data.iter().any(|v| !v.is_finite()) {
        return Err(AutoQuantError::Probe(
            "weights contain NaN or infinity".into(),
        ));
    }
    if imp.is_some_and(|i| i.len() != cols || i.iter().any(|v| !v.is_finite() || *v < 0.0)) {
        return Err(AutoQuantError::Probe(
            "importance vector must have one non-negative finite value per column".into(),
        ));
    }
    let rt = roundtrip(dtype, data, cols, imp, threads)?;
    let (mut num, mut den) = (0f64, 0f64);
    for (row, rrow) in data.chunks_exact(cols).zip(rt.chunks_exact(cols)) {
        for c in 0..cols {
            let w = imp.map_or(1.0, |i| i[c] as f64);
            let (x, y) = (row[c] as f64, rrow[c] as f64);
            num += w * (x - y) * (x - y);
            den += w * x * x;
        }
    }
    Ok(if den > 0.0 { num / den } else { 0.0 })
}

// ------------------------------------------------------------------ sensitivity analysis

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Run the probe for every (group, candidate): most accurate, `groups × candidates` runs.
    Measured,
    /// Never run the model: relative weight error only (needs `Probe::proxy_data`).
    Proxy,
    /// Run the probe once per group at the anchor format (default: the candidate closest to 4.5 bits per
    /// weight) and scale the proxy of the other formats by `measured/proxy` at the anchor.
    Hybrid { anchor: Option<DType> },
}

pub const DEFAULT_CANDIDATES: [DType; 7] = [
    DType::Q2K,
    DType::Q3K,
    DType::Q4_0,
    DType::Q4K,
    DType::Q5K,
    DType::Q6K,
    DType::Q8_0,
];

#[derive(Clone, Debug)]
pub struct Sensitivity {
    pub groups: Vec<Group>,
    /// Ascending by bits per weight.
    pub candidates: Vec<DType>,
    /// `damage[group][candidate]`; `None` where the format's block size does not divide the row length.
    pub damage: Vec<Vec<Option<f64>>>,
    pub method: Method,
    /// How many times the probe's `damage` was called.
    pub probes: usize,
    pub fixed_bytes: u64,
    pub threads: usize,
}

fn valid_for(g: &Group, t: DType) -> bool {
    g.cols > 0
        && t.bytes_for(g.cols as u64).is_some()
        && t.bytes_for(g.params).is_some()
        && t.decodable()
}

fn clean(v: f64, what: &str) -> Result<f64, AutoQuantError> {
    if v.is_finite() {
        Ok(v.max(0.0))
    } else {
        Err(AutoQuantError::Probe(format!(
            "{what}: damage is not a finite number"
        )))
    }
}

pub fn analyse(
    probe: &mut dyn Probe,
    candidates: &[DType],
    method: Method,
    threads: usize,
    ctl: &mut Control,
) -> Result<Sensitivity, AutoQuantError> {
    let groups = probe.groups();
    if groups.is_empty() {
        return Err(AutoQuantError::Empty(
            "the probe has no tensor groups".into(),
        ));
    }
    let mut cands: Vec<DType> = vec![];
    for &c in candidates {
        if !cands.contains(&c) && c.decodable() && c.geometry().is_some() {
            cands.push(c);
        }
    }
    if cands.is_empty() {
        return Err(AutoQuantError::Empty("no usable candidate formats".into()));
    }
    cands.sort_by(|a, b| bits_per_weight(*a).total_cmp(&bits_per_weight(*b)));
    for g in &groups {
        if g.params == 0 || !cands.iter().any(|&c| valid_for(g, c)) {
            return Err(AutoQuantError::NoCandidate(format!(
                "{}: no candidate format fits rows of {}",
                g.name, g.cols
            )));
        }
    }
    let threads = threads.max(1);
    let total = match method {
        Method::Measured => groups
            .iter()
            .map(|g| cands.iter().filter(|&&c| valid_for(g, c)).count())
            .sum(),
        Method::Proxy => 0,
        Method::Hybrid { .. } => groups.len(),
    };
    let (mut damage, mut probes) = (vec![vec![None; cands.len()]; groups.len()], 0usize);
    for (gi, g) in groups.iter().enumerate() {
        let valid: Vec<usize> = (0..cands.len())
            .filter(|&ci| valid_for(g, cands[ci]))
            .collect();
        match method {
            Method::Measured => {
                for &ci in &valid {
                    ctl.tick(
                        "sensitivity",
                        probes,
                        total,
                        format!("{} as {}", g.name, dtype_name(cands[ci])),
                    )?;
                    let d = probe.damage(gi, cands[ci]).map_err(AutoQuantError::Probe)?;
                    probes += 1;
                    damage[gi][ci] = Some(clean(d, &g.name)?);
                }
            }
            Method::Proxy | Method::Hybrid { .. } => {
                let pd = probe.proxy_data(gi).ok_or_else(|| {
                    AutoQuantError::MissingProxy(format!(
                        "{}: the probe gave no weights for the proxy",
                        g.name
                    ))
                })?;
                if pd.cols != g.cols || pd.data.is_empty() || pd.data.len() % pd.cols.max(1) != 0 {
                    return Err(AutoQuantError::MissingProxy(format!(
                        "{}: proxy weights do not match the group shape",
                        g.name
                    )));
                }
                let scale = if pd.scale.is_finite() && pd.scale >= 0.0 {
                    pd.scale
                } else {
                    1.0
                };
                let mut px = vec![0f64; cands.len()];
                for &ci in &valid {
                    px[ci] =
                        proxy_damage(cands[ci], &pd.data, pd.cols, pd.imp.as_deref(), threads)?
                            * scale;
                }
                if let Method::Hybrid { anchor } = method {
                    let anchor_ci = match anchor
                        .and_then(|a| valid.iter().copied().find(|&ci| cands[ci] == a))
                    {
                        Some(ci) => ci,
                        None => *valid
                            .iter()
                            .min_by(|&&a, &&b| {
                                (bits_per_weight(cands[a]) - 4.5)
                                    .abs()
                                    .total_cmp(&(bits_per_weight(cands[b]) - 4.5).abs())
                            })
                            .expect("valid is non-empty"),
                    };
                    ctl.tick(
                        "sensitivity",
                        probes,
                        total,
                        format!("{} as {} (anchor)", g.name, dtype_name(cands[anchor_ci])),
                    )?;
                    let real = clean(
                        probe
                            .damage(gi, cands[anchor_ci])
                            .map_err(AutoQuantError::Probe)?,
                        &g.name,
                    )?;
                    probes += 1;
                    let ratio = if px[anchor_ci] > 0.0 {
                        real / px[anchor_ci]
                    } else {
                        0.0
                    };
                    for &ci in &valid {
                        damage[gi][ci] = Some(if ci == anchor_ci {
                            real
                        } else {
                            px[ci] * ratio
                        });
                    }
                } else {
                    ctl.tick(
                        "sensitivity",
                        gi,
                        groups.len(),
                        format!("{} (proxy)", g.name),
                    )?;
                    for &ci in &valid {
                        damage[gi][ci] = Some(px[ci]);
                    }
                }
            }
        }
    }
    Ok(Sensitivity {
        fixed_bytes: probe.fixed_bytes(),
        groups,
        candidates: cands,
        damage,
        method,
        probes,
        threads,
    })
}

// ------------------------------------------------------------------ budgets and solving

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    /// Total bytes, including [`Sensitivity::fixed_bytes`]. Use this for a file size or a RAM budget.
    Bytes(u64),
    /// Average bits per weight over the grouped tensors only.
    Bpw(f64),
    /// The size of storing every group in this format (groups it does not fit get the closest format).
    Uniform(DType),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Solver {
    Greedy,
    Exact,
}

#[derive(Clone, Debug)]
pub struct SolveOpts {
    pub solver: Solver,
    pub speed: Option<SpeedModel>,
    /// 0 ignores speed. 1 means going from the fastest to the slowest configuration costs as much as the
    /// worst-case total damage.
    pub speed_weight: f64,
    /// Upper bound on the dynamic-programming table (groups × byte cells).
    pub max_cells: usize,
}
impl Default for SolveOpts {
    fn default() -> SolveOpts {
        SolveOpts {
            solver: Solver::Exact,
            speed: None,
            speed_weight: 0.0,
            max_cells: 48_000_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub choice: Vec<DType>,
    /// Bytes of all groups plus the fixed tensors.
    pub bytes: u64,
    pub group_bytes: u64,
    pub budget_bytes: u64,
    pub bpw: f64,
    /// Sum of the per-group damages from the sensitivity table (the additive prediction).
    pub predicted_damage: f64,
    /// Mean decode cost per weight under the speed model (bit equivalents); `None` without one.
    pub decode_cost: Option<f64>,
    pub solver: Solver,
}

#[derive(Clone, Debug)]
struct Opt {
    dtype: DType,
    bytes: u64,
    cost: f64,
    damage: f64,
}

fn budget_bytes(sens: &Sensitivity, b: Budget) -> u64 {
    let params: u64 = sens.groups.iter().map(|g| g.params).sum();
    match b {
        Budget::Bytes(n) => n,
        Budget::Bpw(x) => {
            if x.is_finite() && x > 0.0 {
                sens.fixed_bytes
                    .saturating_add((x * params as f64 / 8.0).floor() as u64)
            } else {
                0
            }
        }
        Budget::Uniform(t) => sens.fixed_bytes.saturating_add(
            uniform_assignment(sens, t)
                .iter()
                .zip(&sens.groups)
                .map(|(c, g)| c.bytes_for(g.params).unwrap_or(0))
                .sum::<u64>(),
        ),
    }
}

/// Every group stored as `t`, or as the valid candidate closest in bits per weight where `t` does not fit.
pub fn uniform_assignment(sens: &Sensitivity, t: DType) -> Vec<DType> {
    sens.groups
        .iter()
        .enumerate()
        .map(|(gi, g)| {
            if valid_for(g, t) {
                return t;
            }
            let want = bits_per_weight(t);
            (0..sens.candidates.len())
                .filter(|&ci| sens.damage[gi][ci].is_some())
                .min_by(|&a, &b| {
                    (bits_per_weight(sens.candidates[a]) - want)
                        .abs()
                        .total_cmp(&(bits_per_weight(sens.candidates[b]) - want).abs())
                })
                .map_or(t, |ci| sens.candidates[ci])
        })
        .collect()
}

fn frontier(sens: &Sensitivity, opts: &SolveOpts) -> Vec<Vec<Opt>> {
    let scale: f64 = sens
        .groups
        .iter()
        .enumerate()
        .map(|(gi, _)| {
            sens.damage[gi]
                .iter()
                .flatten()
                .fold(0f64, |a, &b| a.max(b))
        })
        .sum::<f64>();
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let speed = opts.speed.as_ref().filter(|_| opts.speed_weight > 0.0);
    let tmax: f64 = speed.map_or(1.0, |s| {
        sens.groups
            .iter()
            .enumerate()
            .map(|(gi, g)| {
                g.params as f64
                    * (0..sens.candidates.len())
                        .filter(|&ci| sens.damage[gi][ci].is_some())
                        .map(|ci| s.cost_per_param(sens.candidates[ci]))
                        .fold(0f64, f64::max)
            })
            .sum::<f64>()
            .max(1e-12)
    });
    sens.groups
        .iter()
        .enumerate()
        .map(|(gi, g)| {
            let mut v: Vec<Opt> = (0..sens.candidates.len())
                .filter_map(|ci| {
                    let d = sens.damage[gi][ci]?;
                    let t = sens.candidates[ci];
                    let pen = speed.map_or(0.0, |s| {
                        opts.speed_weight * scale * g.params as f64 * s.cost_per_param(t) / tmax
                    });
                    Some(Opt {
                        dtype: t,
                        bytes: t.bytes_for(g.params)?,
                        cost: d + pen,
                        damage: d,
                    })
                })
                .collect();
            v.sort_by(|a, b| a.bytes.cmp(&b.bytes).then(a.cost.total_cmp(&b.cost)));
            let mut keep: Vec<Opt> = vec![];
            for o in v {
                if keep.last().is_none_or(|l| o.cost < l.cost) {
                    keep.push(o);
                }
            }
            keep
        })
        .collect()
}

/// Indices of the lower convex hull of a frontier (marginal benefit per byte non-increasing).
fn hull(f: &[Opt]) -> Vec<usize> {
    let slope = |a: usize, b: usize| (f[a].cost - f[b].cost) / (f[b].bytes - f[a].bytes) as f64;
    let mut h: Vec<usize> = vec![];
    for i in 0..f.len() {
        while h.len() >= 2 && slope(h[h.len() - 2], h[h.len() - 1]) < slope(h[h.len() - 1], i) {
            h.pop();
        }
        h.push(i);
    }
    h
}

fn solve_greedy(fr: &[Vec<Opt>], budget: u64) -> Vec<usize> {
    let mut cur = vec![0usize; fr.len()];
    let mut total: u64 = fr.iter().map(|f| f[0].bytes).sum();
    struct Step {
        g: usize,
        from: usize,
        to: usize,
        ratio: f64,
        db: u64,
    }
    let mut steps: Vec<Step> = vec![];
    for (g, f) in fr.iter().enumerate() {
        let h = hull(f);
        for w in h.windows(2) {
            let db = f[w[1]].bytes - f[w[0]].bytes;
            steps.push(Step {
                g,
                from: w[0],
                to: w[1],
                ratio: (f[w[0]].cost - f[w[1]].cost) / db as f64,
                db,
            });
        }
    }
    steps.sort_by(|a, b| b.ratio.total_cmp(&a.ratio).then(a.g.cmp(&b.g)));
    for s in &steps {
        if cur[s.g] == s.from && total + s.db <= budget {
            cur[s.g] = s.to;
            total += s.db;
        }
    }
    // Refinement: any single upgrade (also off the hull) that still fits and lowers the cost.
    loop {
        let mut best: Option<(usize, usize, f64, u64)> = None;
        for (g, f) in fr.iter().enumerate() {
            for (j, o) in f.iter().enumerate() {
                if j <= cur[g] {
                    continue;
                }
                let db = o.bytes - f[cur[g]].bytes;
                if total + db > budget {
                    continue;
                }
                let gain = f[cur[g]].cost - o.cost;
                if gain <= 0.0 {
                    continue;
                }
                let r = gain / db.max(1) as f64;
                if best.is_none_or(|b| r > b.2) {
                    best = Some((g, j, r, db));
                }
            }
        }
        match best {
            Some((g, j, _, db)) => {
                cur[g] = j;
                total += db;
            }
            None => break,
        }
    }
    cur
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn solve_exact(fr: &[Vec<Opt>], budget: u64, max_cells: usize) -> Vec<usize> {
    let g = fr.len();
    let min_total: u64 = fr.iter().map(|f| f[0].bytes).sum();
    let max_total: u64 = fr.iter().map(|f| f[f.len() - 1].bytes).sum();
    let gc = fr
        .iter()
        .flatten()
        .fold(0u64, |a, o| gcd(a, o.bytes))
        .max(1);
    let cap_limit = (max_cells / g.max(1)).max(2) as u64;
    let mut unit = gc;
    if max_total / unit > cap_limit {
        unit = max_total.div_ceil(cap_limit).div_ceil(gc) * gc;
    }
    let cap = (budget.min(max_total) / unit) as usize;
    let units = |o: &Opt| o.bytes.div_ceil(unit) as usize;
    let min_units: usize = fr.iter().map(|f| units(&f[0])).sum();
    if min_total > budget || min_units > cap {
        return vec![0; g];
    }
    let cells = cap + 1;
    let mut prev = vec![0f64; cells];
    let mut cur = vec![0f64; cells];
    let mut choice = vec![0u8; g * cells];
    for (gi, f) in fr.iter().enumerate() {
        for b in 0..cells {
            let mut best = f64::INFINITY;
            let mut bi = 0u8;
            for (oi, o) in f.iter().enumerate() {
                let u = units(o);
                if u <= b {
                    let v = prev[b - u] + o.cost;
                    if v < best {
                        best = v;
                        bi = oi as u8;
                    }
                }
            }
            cur[b] = best;
            choice[gi * cells + b] = bi;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let (mut sel, mut b) = (vec![0usize; g], cap);
    for gi in (0..g).rev() {
        let oi = choice[gi * cells + b] as usize;
        sel[gi] = oi;
        b -= units(&fr[gi][oi]);
    }
    sel
}

/// Choose one format per group under `budget`.
pub fn solve(sens: &Sensitivity, budget: Budget, opts: &SolveOpts) -> Result<Plan, AutoQuantError> {
    let total_budget = budget_bytes(sens, budget);
    let fr = frontier(sens, opts);
    if fr.iter().any(|f| f.len() > 250) {
        return Err(AutoQuantError::Empty("too many candidate formats".into()));
    }
    let min_bytes: u64 = fr
        .iter()
        .map(|f| f[0].bytes)
        .sum::<u64>()
        .saturating_add(sens.fixed_bytes);
    if min_bytes > total_budget {
        return Err(AutoQuantError::Infeasible {
            needed: min_bytes,
            budget: total_budget,
        });
    }
    let room = total_budget - sens.fixed_bytes;
    let sel = match opts.solver {
        Solver::Greedy => solve_greedy(&fr, room),
        Solver::Exact => solve_exact(&fr, room, opts.max_cells.max(1024)),
    };
    let picked: Vec<&Opt> = sel.iter().zip(&fr).map(|(&i, f)| &f[i]).collect();
    let group_bytes: u64 = picked.iter().map(|o| o.bytes).sum();
    let params: u64 = sens.groups.iter().map(|g| g.params).sum();
    let decode_cost = opts.speed.as_ref().map(|s| {
        picked
            .iter()
            .zip(&sens.groups)
            .map(|(o, g)| s.cost_per_param(o.dtype) * g.params as f64)
            .sum::<f64>()
            / params.max(1) as f64
    });
    Ok(Plan {
        choice: picked.iter().map(|o| o.dtype).collect(),
        bytes: group_bytes + sens.fixed_bytes,
        group_bytes,
        budget_bytes: total_budget,
        bpw: group_bytes as f64 * 8.0 / params.max(1) as f64,
        predicted_damage: picked.iter().map(|o| o.damage).sum(),
        decode_cost,
        solver: opts.solver,
    })
}

impl Plan {
    pub fn assignment(&self) -> Vec<(usize, DType)> {
        self.choice.iter().copied().enumerate().collect()
    }

    /// A recipe that starts from the named base recipe (for example `"Q4_K_M"`, which decides the tensors
    /// outside the groups) and overrides every grouped tensor with the chosen format.
    pub fn to_recipe(&self, base: &str, groups: &[Group]) -> Result<Recipe, AutoQuantError> {
        let mut r = Recipe::named(base)
            .ok_or_else(|| AutoQuantError::BadRecipe(format!("unknown base recipe {base}")))?;
        for (g, &t) in groups.iter().zip(&self.choice) {
            for name in &g.tensors {
                r = r.with_override(name, t);
            }
        }
        r.name = format!("auto-{}", r.name);
        Ok(r)
    }
}

// ------------------------------------------------------------------ the whole pipeline and the report

#[derive(Clone, Debug)]
pub struct Baseline {
    pub dtype: DType,
    pub bytes: u64,
    pub fits: bool,
    pub predicted: f64,
    pub measured: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub sens: Sensitivity,
    pub plan: Plan,
    pub measured: Option<f64>,
    pub baselines: Vec<Baseline>,
}

#[derive(Clone, Debug)]
pub struct AutoOpts {
    pub threads: usize,
    pub solve: SolveOpts,
    pub verify: bool,
}
impl Default for AutoOpts {
    fn default() -> AutoOpts {
        AutoOpts {
            threads: std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .min(4),
            solve: SolveOpts::default(),
            verify: true,
        }
    }
}

fn predicted_for(sens: &Sensitivity, choice: &[DType]) -> f64 {
    choice
        .iter()
        .enumerate()
        .map(|(gi, t)| {
            sens.candidates
                .iter()
                .position(|c| c == t)
                .and_then(|ci| sens.damage[gi][ci])
                .unwrap_or(0.0)
        })
        .sum()
}

/// Analyse, solve and (optionally) verify with joint measurements of the plan and of every uniform format.
pub fn auto_quant(
    probe: &mut dyn Probe,
    candidates: &[DType],
    method: Method,
    budget: Budget,
    opts: &AutoOpts,
    ctl: &mut Control,
) -> Result<Outcome, AutoQuantError> {
    let sens = analyse(probe, candidates, method, opts.threads, ctl)?;
    let plan = solve(&sens, budget, &opts.solve)?;
    let (mut measured, mut baselines) = (None, vec![]);
    if opts.verify {
        let n = 1 + sens.candidates.len();
        ctl.tick("verify", 0, n, "the chosen plan".into())?;
        measured = Some(clean(
            probe
                .joint_damage(&plan.assignment())
                .map_err(AutoQuantError::Probe)?,
            "the plan",
        )?);
        for (i, &t) in sens.candidates.iter().enumerate() {
            let choice = uniform_assignment(&sens, t);
            let bytes = sens.fixed_bytes
                + choice
                    .iter()
                    .zip(&sens.groups)
                    .map(|(c, g)| c.bytes_for(g.params).unwrap_or(0))
                    .sum::<u64>();
            let assignment: Vec<(usize, DType)> = choice.iter().copied().enumerate().collect();
            ctl.tick("verify", i + 1, n, format!("uniform {}", dtype_name(t)))?;
            let m = clean(
                probe
                    .joint_damage(&assignment)
                    .map_err(AutoQuantError::Probe)?,
                "a uniform baseline",
            )?;
            baselines.push(Baseline {
                dtype: t,
                bytes,
                fits: bytes <= plan.budget_bytes,
                predicted: predicted_for(&sens, &choice),
                measured: Some(m),
            });
        }
    } else {
        for &t in &sens.candidates {
            let choice = uniform_assignment(&sens, t);
            let bytes = sens.fixed_bytes
                + choice
                    .iter()
                    .zip(&sens.groups)
                    .map(|(c, g)| c.bytes_for(g.params).unwrap_or(0))
                    .sum::<u64>();
            baselines.push(Baseline {
                dtype: t,
                bytes,
                fits: bytes <= plan.budget_bytes,
                predicted: predicted_for(&sens, &choice),
                measured: None,
            });
        }
    }
    Ok(Outcome {
        sens,
        plan,
        measured,
        baselines,
    })
}

fn sci(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else {
        format!("{v:.2e}")
    }
}
fn mib(b: u64) -> String {
    format!("{:.2} MiB", b as f64 / 1048576.0)
}

impl Outcome {
    /// Human-readable report: sensitivity table, the choice, and predicted versus measured damage.
    pub fn to_markdown(&self) -> String {
        let (s, p) = (&self.sens, &self.plan);
        let mut o = String::from("# Mixed-precision quantisation report\n\n");
        o += &format!(
            "Method: {:?}, {} probe runs. Solver: {:?}. Budget {} → plan {} ({:.3} bits per weight over the grouped tensors).\n\n",
            s.method,
            s.probes,
            p.solver,
            mib(p.budget_bytes),
            mib(p.bytes),
            p.bpw
        );
        o += "## Sensitivity (damage when only this group uses the format)\n\n| group | layer | params |";
        for c in &s.candidates {
            o += &format!(" {} |", dtype_name(*c));
        }
        o += " chosen |\n|---|---|---|";
        for _ in &s.candidates {
            o += "---|";
        }
        o += "---|\n";
        for (gi, g) in s.groups.iter().enumerate() {
            o += &format!(
                "| {} | {} | {} |",
                g.name,
                g.layer.map_or("-".into(), |l| l.to_string()),
                g.params
            );
            for ci in 0..s.candidates.len() {
                o += &format!(" {} |", s.damage[gi][ci].map_or("—".into(), sci));
            }
            o += &format!(" **{}** |\n", dtype_name(p.choice[gi]));
        }
        o += &format!(
            "\n## Result\n\n* Predicted damage (sum over groups): {}\n",
            sci(p.predicted_damage)
        );
        match self.measured {
            Some(m) => {
                let ratio = if p.predicted_damage > 0.0 {
                    format!(" (measured/predicted = {:.2})", m / p.predicted_damage)
                } else {
                    String::new()
                };
                o += &format!("* Measured joint damage: {}{}\n", sci(m), ratio);
            }
            None => o += "* Measured joint damage: not verified\n",
        }
        if let Some(c) = p.decode_cost {
            o += &format!("* Mean decode cost: {c:.2} bit-equivalents per weight (heuristic)\n");
        }
        o += "\n## Uniform formats for comparison\n\n| format | size | fits | predicted | measured |\n|---|---|---|---|---|\n";
        for b in &self.baselines {
            o += &format!(
                "| {} | {} | {} | {} | {} |\n",
                dtype_name(b.dtype),
                mib(b.bytes),
                if b.fits { "yes" } else { "no" },
                sci(b.predicted),
                b.measured.map_or("—".into(), sci)
            );
        }
        o += &format!(
            "| **auto** | {} | yes | {} | {} |\n",
            mib(p.bytes),
            sci(p.predicted_damage),
            self.measured.map_or("—".into(), sci)
        );
        o
    }

    pub fn to_json(&self) -> Json {
        let (s, p) = (&self.sens, &self.plan);
        json!({
            "method": format!("{:?}", s.method), "probes": s.probes, "solver": format!("{:?}", p.solver),
            "budget_bytes": p.budget_bytes, "bytes": p.bytes, "bpw": p.bpw,
            "predicted_damage": p.predicted_damage, "measured_damage": self.measured, "decode_cost": p.decode_cost,
            "candidates": s.candidates.iter().map(|c| dtype_name(*c)).collect::<Vec<_>>(),
            "groups": s.groups.iter().enumerate().map(|(gi, g)| json!({
                "name": g.name, "layer": g.layer, "params": g.params, "tensors": g.tensors,
                "damage": s.damage[gi], "chosen": dtype_name(p.choice[gi]),
            })).collect::<Vec<_>>(),
            "uniform": self.baselines.iter().map(|b| json!({"dtype": dtype_name(b.dtype), "bytes": b.bytes, "fits": b.fits, "predicted": b.predicted, "measured": b.measured})).collect::<Vec<_>>(),
        })
    }
}

// ------------------------------------------------------------------ a synthetic network for tests and the demo

pub mod synthetic {
    //! A small residual network with engineered sensitivity: layer `l` adds `gain[l] · W_l h` to the hidden
    //! state, so a layer with a large gain moves the logits more when its weights are rounded. Calibration
    //! inputs are random hidden vectors. Good for tests and demos; it says nothing about a real model.
    use super::*;

    pub struct Rng(u64);
    impl Rng {
        pub fn new(seed: u64) -> Rng {
            Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
        }
        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        pub fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        pub fn normal(&mut self) -> f32 {
            let (a, b) = (self.uniform().max(1e-12), self.uniform());
            ((-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()) as f32
        }
    }

    /// Dequantised replacements: one optional matrix per layer, and the optional head.
    type Replaced = (Vec<Option<Vec<f32>>>, Option<Vec<f32>>);

    pub struct SyntheticNet {
        pub d: usize,
        pub vocab: usize,
        pub gains: Vec<f32>,
        pub layers: Vec<Vec<f32>>,
        pub head: Vec<f32>,
        pub inputs: Vec<Vec<f32>>,
        pub reference: Vec<Vec<f32>>,
        pub threads: usize,
    }

    impl SyntheticNet {
        /// `d` must be a multiple of 256 for the K-quants; layers listed in `sensitive` get a large gain.
        pub fn new(
            seed: u64,
            n_layers: usize,
            d: usize,
            vocab: usize,
            samples: usize,
            sensitive: &[usize],
        ) -> SyntheticNet {
            let mut rng = Rng::new(seed);
            let layers: Vec<Vec<f32>> = (0..n_layers)
                .map(|_| {
                    (0..d * d)
                        .map(|_| rng.normal() / (d as f32).sqrt())
                        .collect()
                })
                .collect();
            let head: Vec<f32> = (0..vocab * d)
                .map(|_| rng.normal() * 2.5 / (d as f32).sqrt())
                .collect();
            let gains = (0..n_layers)
                .map(|l| if sensitive.contains(&l) { 0.9 } else { 0.12 })
                .collect();
            let inputs = (0..samples)
                .map(|_| (0..d).map(|_| rng.normal()).collect())
                .collect();
            let mut n = SyntheticNet {
                d,
                vocab,
                gains,
                layers,
                head,
                inputs,
                reference: vec![],
                threads: 1,
            };
            n.reference = n.forward(&|l| &n.layers[l], &n.head);
            n
        }

        fn forward<'a>(
            &'a self,
            layer: &dyn Fn(usize) -> &'a [f32],
            head: &[f32],
        ) -> Vec<Vec<f32>> {
            let d = self.d;
            self.inputs
                .iter()
                .map(|x| {
                    let mut h = x.clone();
                    for l in 0..self.layers.len() {
                        let w = layer(l);
                        let upd: Vec<f32> = (0..d)
                            .map(|r| {
                                w[r * d..(r + 1) * d]
                                    .iter()
                                    .zip(&h)
                                    .map(|(a, b)| a * b)
                                    .sum::<f32>()
                            })
                            .collect();
                        for (hv, u) in h.iter_mut().zip(upd) {
                            *hv += self.gains[l] * u;
                        }
                    }
                    (0..self.vocab)
                        .map(|v| {
                            head[v * d..(v + 1) * d]
                                .iter()
                                .zip(&h)
                                .map(|(a, b)| a * b)
                                .sum::<f32>()
                        })
                        .collect()
                })
                .collect()
        }

        fn quantised(&self, assignment: &[(usize, DType)]) -> Result<Replaced, String> {
            let mut layers: Vec<Option<Vec<f32>>> = vec![None; self.layers.len()];
            let mut head = None;
            for &(g, t) in assignment {
                if g < self.layers.len() {
                    layers[g] = Some(
                        roundtrip(t, &self.layers[g], self.d, None, self.threads)
                            .map_err(|e| e.to_string())?,
                    );
                } else if g == self.layers.len() {
                    head = Some(
                        roundtrip(t, &self.head, self.d, None, self.threads)
                            .map_err(|e| e.to_string())?,
                    );
                } else {
                    return Err(format!("no group {g}"));
                }
            }
            Ok((layers, head))
        }
    }

    impl Probe for SyntheticNet {
        fn groups(&self) -> Vec<Group> {
            let mut v: Vec<Group> = (0..self.layers.len())
                .map(|l| Group {
                    name: format!("blk.{l}.ffn_down.weight"),
                    tensors: vec![format!("blk.{l}.ffn_down.weight")],
                    layer: Some(l),
                    params: (self.d * self.d) as u64,
                    cols: self.d,
                })
                .collect();
            v.push(Group {
                name: "output.weight".into(),
                tensors: vec!["output.weight".into()],
                layer: None,
                params: (self.vocab * self.d) as u64,
                cols: self.d,
            });
            v
        }
        fn damage(&mut self, group: usize, dtype: DType) -> Result<f64, String> {
            self.joint_damage(&[(group, dtype)])
        }
        fn joint_damage(&mut self, assignment: &[(usize, DType)]) -> Result<f64, String> {
            let (ql, qh) = self.quantised(assignment)?;
            let out = self.forward(
                &|l| ql[l].as_deref().unwrap_or(&self.layers[l]),
                qh.as_deref().unwrap_or(&self.head),
            );
            kl_damage(&self.reference, &out)
        }
        fn proxy_data(&self, group: usize) -> Option<ProxyData> {
            let data = if group < self.layers.len() {
                self.layers[group].clone()
            } else if group == self.layers.len() {
                self.head.clone()
            } else {
                return None;
            };
            Some(ProxyData {
                data,
                cols: self.d,
                imp: None,
                scale: 1.0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::synthetic::*;
    use super::*;

    fn net() -> SyntheticNet {
        SyntheticNet::new(7, 6, 256, 48, 16, &[1, 4])
    }
    fn quick() -> AutoOpts {
        AutoOpts {
            threads: 1,
            ..Default::default()
        }
    }

    /// One analysed sensitivity table shared by the solver tests (building it is the slow part).
    fn table() -> Sensitivity {
        static T: std::sync::OnceLock<Sensitivity> = std::sync::OnceLock::new();
        T.get_or_init(|| {
            analyse(
                &mut net(),
                &DEFAULT_CANDIDATES,
                Method::Measured,
                1,
                &mut Control::none(),
            )
            .unwrap()
        })
        .clone()
    }

    fn uniform_bytes(s: &Sensitivity, t: DType) -> u64 {
        budget_bytes(s, Budget::Uniform(t))
    }

    #[test]
    fn sensitivity_table_has_the_expected_shape_and_order() {
        let s = table();
        assert_eq!(s.groups.len(), 7);
        assert_eq!(s.probes, 7 * DEFAULT_CANDIDATES.len());
        let bpw: Vec<f64> = s.candidates.iter().map(|c| bits_per_weight(*c)).collect();
        assert!(bpw.windows(2).all(|w| w[0] <= w[1]));
        // More bits never hurts more, in the coarse sense: Q2_K is worse than Q8_0 everywhere.
        for gi in 0..s.groups.len() {
            let q2 =
                s.damage[gi][s.candidates.iter().position(|c| *c == DType::Q2K).unwrap()].unwrap();
            let q8 =
                s.damage[gi][s.candidates.iter().position(|c| *c == DType::Q8_0).unwrap()].unwrap();
            assert!(q2 > q8 * 10.0, "group {gi}: Q2_K {q2} vs Q8_0 {q8}");
        }
        // The engineered layers are the sensitive ones.
        let q3 = s.candidates.iter().position(|c| *c == DType::Q3K).unwrap();
        let d = |g: usize| s.damage[g][q3].unwrap();
        for hot in [1, 4] {
            for cold in [0, 2, 3, 5] {
                assert!(
                    d(hot) > d(cold) * 5.0,
                    "layer {hot} should be far more sensitive than {cold}"
                );
            }
        }
    }

    #[test]
    fn plans_never_exceed_the_budget_for_either_solver() {
        let s = table();
        let min = uniform_bytes(&s, DType::Q2K);
        let max = uniform_bytes(&s, DType::Q8_0);
        for solver in [Solver::Greedy, Solver::Exact] {
            for k in 0..=24u64 {
                let b = min + (max - min) * k / 24 + k % 3 * 17;
                let p = solve(
                    &s,
                    Budget::Bytes(b),
                    &SolveOpts {
                        solver,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert!(p.bytes <= b, "{solver:?}: {} > {b}", p.bytes);
                assert!(p.bytes <= p.budget_bytes);
            }
        }
        // A budget in bits per weight maps to bytes without overshooting.
        let params: u64 = s.groups.iter().map(|g| g.params).sum();
        let p = solve(&s, Budget::Bpw(4.0), &SolveOpts::default()).unwrap();
        assert!(p.group_bytes as f64 * 8.0 <= 4.0 * params as f64 + 1e-6);
    }

    #[test]
    fn a_budget_below_the_smallest_formats_is_refused() {
        let s = table();
        let min = uniform_bytes(&s, DType::Q2K);
        match solve(&s, Budget::Bytes(min - 1), &SolveOpts::default()) {
            Err(AutoQuantError::Infeasible { needed, budget }) => {
                assert!(needed > budget);
            }
            other => panic!("{other:?}"),
        }
        assert!(solve(&s, Budget::Bytes(0), &SolveOpts::default()).is_err());
        assert!(solve(&s, Budget::Bpw(f64::NAN), &SolveOpts::default()).is_err());
        // The smallest format is what the minimum budget gets.
        let p = solve(&s, Budget::Bytes(min), &SolveOpts::default()).unwrap();
        assert!(p.choice.iter().all(|c| *c == DType::Q2K));
    }

    #[test]
    fn exact_damage_is_monotone_in_the_budget_and_never_worse_than_greedy() {
        let s = table();
        let (min, max) = (
            uniform_bytes(&s, DType::Q2K),
            uniform_bytes(&s, DType::Q8_0),
        );
        let mut last = f64::INFINITY;
        for k in 0..=40u64 {
            let b = min + (max - min) * k / 40;
            let e = solve(
                &s,
                Budget::Bytes(b),
                &SolveOpts {
                    solver: Solver::Exact,
                    ..Default::default()
                },
            )
            .unwrap();
            let g = solve(
                &s,
                Budget::Bytes(b),
                &SolveOpts {
                    solver: Solver::Greedy,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(
                e.predicted_damage <= last + 1e-15,
                "budget {b}: {} after {last}",
                e.predicted_damage
            );
            assert!(
                e.predicted_damage <= g.predicted_damage + 1e-15,
                "exact {} vs greedy {}",
                e.predicted_damage,
                g.predicted_damage
            );
            last = e.predicted_damage;
        }
    }

    #[test]
    fn a_coarse_grid_still_respects_the_budget_and_stays_monotone() {
        let s = table();
        let (min, max) = (
            uniform_bytes(&s, DType::Q2K),
            uniform_bytes(&s, DType::Q8_0),
        );
        let mut last = f64::INFINITY;
        for k in 0..=30u64 {
            let b = min + (max - min) * k / 30 + 11;
            let p = solve(
                &s,
                Budget::Bytes(b),
                &SolveOpts {
                    max_cells: 2048,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(p.bytes <= b);
            assert!(p.predicted_damage <= last + 1e-15);
            last = p.predicted_damage;
        }
    }

    #[test]
    fn exact_matches_brute_force_on_a_small_instance() {
        let mut n = SyntheticNet::new(3, 3, 256, 32, 8, &[1]);
        let s = analyse(
            &mut n,
            &[DType::Q2K, DType::Q4K, DType::Q6K],
            Method::Measured,
            1,
            &mut Control::none(),
        )
        .unwrap();
        let budget = uniform_bytes(&s, DType::Q4K);
        let fr = frontier(&s, &SolveOpts::default());
        let mut best = f64::INFINITY;
        let g = s.groups.len();
        let idx = |mut n: usize, f: &Vec<Opt>| {
            let r = n % f.len();
            n /= f.len();
            (r, n)
        };
        let combos: usize = fr.iter().map(|f| f.len()).product();
        for c in 0..combos {
            let (mut rest, mut bytes, mut cost) = (c, 0u64, 0f64);
            for f in fr.iter().take(g) {
                let (i, r) = idx(rest, f);
                rest = r;
                bytes += f[i].bytes;
                cost += f[i].cost;
            }
            if bytes <= budget && cost < best {
                best = cost;
            }
        }
        let p = solve(&s, Budget::Bytes(budget), &SolveOpts::default()).unwrap();
        assert!(
            (p.predicted_damage - best).abs() <= 1e-12 * best.max(1.0),
            "{} vs brute force {best}",
            p.predicted_damage
        );
    }

    #[test]
    fn the_mixed_plan_beats_uniform_quantisation_at_the_same_size_when_measured() {
        let mut n = net();
        for target in [DType::Q4K, DType::Q3K] {
            let s = table();
            let budget = Budget::Bytes(uniform_bytes(&s, target));
            let out = auto_quant(
                &mut n,
                &DEFAULT_CANDIDATES,
                Method::Measured,
                budget,
                &quick(),
                &mut Control::none(),
            )
            .unwrap();
            let uni = out.baselines.iter().find(|b| b.dtype == target).unwrap();
            assert!(
                out.plan.bytes <= uni.bytes,
                "the plan must not be larger than the uniform format"
            );
            let (mixed, flat) = (out.measured.unwrap(), uni.measured.unwrap());
            assert!(
                mixed < flat,
                "{}: mixed {mixed:e} should beat uniform {flat:e}",
                dtype_name(target)
            );
            assert!(out.plan.predicted_damage <= uni.predicted + 1e-15);
            // The sensitive layers got at least as many bits as the insensitive ones.
            let bits = |g: usize| bits_per_weight(out.plan.choice[g]);
            assert!(
                bits(1).min(bits(4)) >= bits(0).max(bits(2)).max(bits(3)).max(bits(5)) - 1e-9,
                "choice {:?}",
                out.plan.choice
            );
        }
    }

    #[test]
    fn the_additive_prediction_is_close_to_the_measured_joint_damage() {
        let mut n = net();
        let out = auto_quant(
            &mut n,
            &DEFAULT_CANDIDATES,
            Method::Measured,
            Budget::Bpw(4.5),
            &quick(),
            &mut Control::none(),
        )
        .unwrap();
        let r = out.measured.unwrap() / out.plan.predicted_damage;
        assert!(r > 0.3 && r < 3.0, "measured/predicted = {r}");
    }

    #[test]
    fn hybrid_needs_one_probe_per_group_and_still_beats_uniform() {
        let mut n = net();
        let opts = quick();
        let full = table();
        let budget = Budget::Bytes(uniform_bytes(&full, DType::Q4K));
        let hy = auto_quant(
            &mut n,
            &DEFAULT_CANDIDATES,
            Method::Hybrid { anchor: None },
            budget,
            &opts,
            &mut Control::none(),
        )
        .unwrap();
        assert_eq!(hy.sens.probes, hy.sens.groups.len());
        assert!(hy.sens.probes < full.probes / 3);
        let uni = hy
            .baselines
            .iter()
            .find(|b| b.dtype == DType::Q4K)
            .unwrap()
            .measured
            .unwrap();
        assert!(
            hy.measured.unwrap() < uni,
            "hybrid {:e} vs uniform {uni:e}",
            hy.measured.unwrap()
        );
        assert!(hy.plan.bytes <= hy.plan.budget_bytes);
    }

    #[test]
    fn proxy_only_runs_the_model_zero_times_and_orders_formats_sensibly() {
        let mut n = net();
        let s = analyse(
            &mut n,
            &DEFAULT_CANDIDATES,
            Method::Proxy,
            1,
            &mut Control::none(),
        )
        .unwrap();
        assert_eq!(s.probes, 0);
        for gi in 0..s.groups.len() {
            let get = |t: DType| {
                s.damage[gi][s.candidates.iter().position(|c| *c == t).unwrap()].unwrap()
            };
            assert!(
                get(DType::Q2K) > get(DType::Q3K)
                    && get(DType::Q3K) > get(DType::Q4K)
                    && get(DType::Q4K) > get(DType::Q6K)
                    && get(DType::Q6K) > get(DType::Q8_0)
            );
        }
        // It cannot tell the hot layers apart (relative weight error is the same for all of them).
        let q4 = s.candidates.iter().position(|c| *c == DType::Q4K).unwrap();
        let (a, b) = (s.damage[1][q4].unwrap(), s.damage[0][q4].unwrap());
        assert!((a / b - 1.0).abs() < 0.2);
    }

    #[test]
    fn the_importance_weighted_proxy_follows_the_important_columns() {
        let mut rng = Rng::new(5);
        let (rows, cols) = (8, 256);
        let data: Vec<f32> = (0..rows * cols).map(|_| rng.normal()).collect();
        let plain = proxy_damage(DType::Q3K, &data, cols, None, 1).unwrap();
        assert!(plain > 0.0 && plain < 1.0);
        let imp: Vec<f32> = (0..cols)
            .map(|c| if c < 16 { 100.0 } else { 0.01 })
            .collect();
        let weighted = proxy_damage(DType::Q3K, &data, cols, Some(&imp), 1).unwrap();
        assert!(weighted.is_finite() && weighted > 0.0 && (weighted - plain).abs() > 1e-6);
        assert!(proxy_damage(DType::Q3K, &data, cols, Some(&imp[..10]), 1).is_err());
        let mut bad = data.clone();
        bad[3] = f32::NAN;
        assert!(proxy_damage(DType::Q3K, &bad, cols, None, 1).is_err());
        assert_eq!(
            proxy_damage(DType::Q4K, &vec![0.0; rows * cols], cols, None, 1).unwrap(),
            0.0
        );
    }

    #[test]
    fn the_hook_reports_progress_and_can_cancel() {
        let mut n = net();
        let mut seen: Vec<(usize, usize)> = vec![];
        {
            let mut cb = |p: &Progress| seen.push((p.done, p.total));
            let mut ctl = Control {
                progress: Some(&mut cb),
                cancel: None,
            };
            analyse(
                &mut n,
                &[DType::Q3K, DType::Q6K],
                Method::Measured,
                1,
                &mut ctl,
            )
            .unwrap();
        }
        assert_eq!(seen.len(), 14);
        assert!(seen.windows(2).all(|w| w[0].0 < w[1].0) && seen.iter().all(|s| s.1 == 14));
        let calls = std::cell::Cell::new(0);
        let cancel = || {
            calls.set(calls.get() + 1);
            calls.get() > 5
        };
        let mut ctl = Control {
            progress: None,
            cancel: Some(&cancel),
        };
        assert_eq!(
            analyse(&mut n, &DEFAULT_CANDIDATES, Method::Measured, 1, &mut ctl).unwrap_err(),
            AutoQuantError::Cancelled
        );
    }

    #[test]
    fn the_recipe_applies_the_choice_and_keeps_norms_in_full_precision() {
        let s = table();
        let p = solve(&s, Budget::Bpw(4.2), &SolveOpts::default()).unwrap();
        let r = p.to_recipe("Q4_K_M", &s.groups).unwrap();
        for (g, t) in s.groups.iter().zip(&p.choice) {
            let shape = [g.params / g.cols as u64, g.cols as u64];
            assert_eq!(r.plan(&g.tensors[0], &shape, 6), *t, "{}", g.name);
        }
        assert_eq!(r.plan("blk.0.attn_norm.weight", &[256], 6), DType::F32);
        assert!(r.name.starts_with("auto-"));
        assert!(matches!(
            p.to_recipe("NOPE", &s.groups),
            Err(AutoQuantError::BadRecipe(_))
        ));
    }

    struct Narrow;
    impl Probe for Narrow {
        fn groups(&self) -> Vec<Group> {
            vec![
                Group {
                    name: "a".into(),
                    tensors: vec!["a".into()],
                    layer: Some(0),
                    params: 96 * 4,
                    cols: 96,
                },
                Group {
                    name: "b".into(),
                    tensors: vec!["b".into()],
                    layer: Some(1),
                    params: 256 * 4,
                    cols: 256,
                },
            ]
        }
        fn damage(&mut self, g: usize, t: DType) -> Result<f64, String> {
            Ok((g as f64 + 1.0) / bits_per_weight(t).powi(2))
        }
        fn joint_damage(&mut self, a: &[(usize, DType)]) -> Result<f64, String> {
            a.iter().map(|&(g, t)| self.damage(g, t)).sum()
        }
    }

    #[test]
    fn formats_that_do_not_divide_the_row_length_are_never_chosen() {
        let s = analyse(
            &mut Narrow,
            &DEFAULT_CANDIDATES,
            Method::Measured,
            1,
            &mut Control::none(),
        )
        .unwrap();
        let ci = |t: DType| s.candidates.iter().position(|c| *c == t).unwrap();
        assert!(s.damage[0][ci(DType::Q4K)].is_none() && s.damage[0][ci(DType::Q2K)].is_none());
        assert!(s.damage[0][ci(DType::Q4_0)].is_some() && s.damage[1][ci(DType::Q4K)].is_some());
        for solver in [Solver::Exact, Solver::Greedy] {
            let p = solve(
                &s,
                Budget::Bytes(1 << 20),
                &SolveOpts {
                    solver,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(valid_for(&s.groups[0], p.choice[0]));
            assert_eq!(
                p.choice[0],
                DType::Q8_0,
                "an unlimited budget gets the best format that fits"
            );
        }
        assert_eq!(uniform_assignment(&s, DType::Q4K)[0], DType::Q4_0);
    }

    #[test]
    fn hostile_probes_are_rejected_without_panicking() {
        struct Bad(f64, usize);
        impl Probe for Bad {
            fn groups(&self) -> Vec<Group> {
                (0..self.1)
                    .map(|i| Group {
                        name: format!("g{i}"),
                        tensors: vec![],
                        layer: None,
                        params: 256,
                        cols: 256,
                    })
                    .collect()
            }
            fn damage(&mut self, _: usize, _: DType) -> Result<f64, String> {
                Ok(self.0)
            }
            fn joint_damage(&mut self, _: &[(usize, DType)]) -> Result<f64, String> {
                Err("boom".into())
            }
        }
        let mut ctl = Control::none();
        assert!(matches!(
            analyse(
                &mut Bad(f64::NAN, 2),
                &DEFAULT_CANDIDATES,
                Method::Measured,
                1,
                &mut ctl
            ),
            Err(AutoQuantError::Probe(_))
        ));
        assert!(matches!(
            analyse(
                &mut Bad(f64::INFINITY, 2),
                &DEFAULT_CANDIDATES,
                Method::Measured,
                1,
                &mut ctl
            ),
            Err(AutoQuantError::Probe(_))
        ));
        assert!(matches!(
            analyse(
                &mut Bad(1.0, 0),
                &DEFAULT_CANDIDATES,
                Method::Measured,
                1,
                &mut ctl
            ),
            Err(AutoQuantError::Empty(_))
        ));
        assert!(matches!(
            analyse(&mut Bad(1.0, 1), &[], Method::Measured, 1, &mut ctl),
            Err(AutoQuantError::Empty(_))
        ));
        assert!(matches!(
            analyse(
                &mut Bad(1.0, 1),
                &[DType::Other(99)],
                Method::Measured,
                1,
                &mut ctl
            ),
            Err(AutoQuantError::Empty(_))
        ));
        assert!(matches!(
            analyse(
                &mut Bad(1.0, 1),
                &DEFAULT_CANDIDATES,
                Method::Proxy,
                1,
                &mut ctl
            ),
            Err(AutoQuantError::MissingProxy(_))
        ));
        // Negative damage from a noisy probe is clamped, not trusted.
        let s = analyse(
            &mut Bad(-1.0, 1),
            &DEFAULT_CANDIDATES,
            Method::Measured,
            1,
            &mut ctl,
        )
        .unwrap();
        assert!(s.damage[0].iter().flatten().all(|d| *d == 0.0));
        // A failing joint measurement surfaces as an error.
        assert!(matches!(
            auto_quant(
                &mut Bad(1.0, 1),
                &DEFAULT_CANDIDATES,
                Method::Measured,
                Budget::Bpw(8.5),
                &quick(),
                &mut ctl
            ),
            Err(AutoQuantError::Probe(_))
        ));
    }

    #[test]
    fn a_speed_penalty_steers_away_from_formats_that_are_slow_on_this_cpu() {
        let s = table();
        let budget = Budget::Bytes(uniform_bytes(&s, DType::Q4K));
        let base = solve(&s, budget, &SolveOpts::default()).unwrap();
        let slow_k = SpeedModel::flat()
            .with_overhead(DType::Q4K, 40.0)
            .with_overhead(DType::Q3K, 40.0)
            .with_overhead(DType::Q5K, 40.0)
            .with_overhead(DType::Q6K, 40.0)
            .with_overhead(DType::Q2K, 40.0);
        let opts = SolveOpts {
            speed: Some(slow_k),
            speed_weight: 5.0,
            ..Default::default()
        };
        let fast = solve(&s, budget, &opts).unwrap();
        assert!(fast.bytes <= fast.budget_bytes);
        let k = |p: &Plan| {
            p.choice
                .iter()
                .filter(|c| {
                    matches!(
                        c,
                        DType::Q2K | DType::Q3K | DType::Q4K | DType::Q5K | DType::Q6K
                    )
                })
                .count()
        };
        assert!(
            k(&fast) < k(&base) || k(&base) == 0,
            "K-quant groups: {} with the penalty, {} without",
            k(&fast),
            k(&base)
        );
        assert!(fast.decode_cost.is_some() && base.decode_cost.is_none());
        // A hardware profile without SIMD makes the K-quants dearer than one with it.
        let slow = SpeedModel::from_isa(&["sse4.2"]);
        let quick_cpu = SpeedModel::from_isa(&["avx2", "fma"]);
        assert!(slow.cost_per_param(DType::Q3K) > quick_cpu.cost_per_param(DType::Q3K));
        assert_eq!(
            slow.cost_per_param(DType::Q8_0),
            quick_cpu.cost_per_param(DType::Q8_0)
        );
    }

    #[test]
    fn the_report_names_every_group_and_the_json_parses_back() {
        let mut n = net();
        let out = auto_quant(
            &mut n,
            &DEFAULT_CANDIDATES,
            Method::Measured,
            Budget::Bpw(4.6),
            &quick(),
            &mut Control::none(),
        )
        .unwrap();
        let md = out.to_markdown();
        for g in &out.sens.groups {
            assert!(md.contains(&g.name), "{} missing", g.name);
        }
        for (g, t) in out.sens.groups.iter().zip(&out.plan.choice) {
            assert!(
                md.lines()
                    .any(|l| l.contains(&g.name) && l.contains(&format!("**{}**", dtype_name(*t))))
            );
        }
        assert!(
            md.contains("Measured joint damage")
                && md.contains("measured/predicted")
                && md.contains("**auto**")
        );
        let j: Json = serde_json::from_str(&out.to_json().to_string()).unwrap();
        assert_eq!(j["groups"].as_array().unwrap().len(), 7);
        assert!(
            j["measured_damage"].is_number()
                && j["bytes"].as_u64().unwrap() <= j["budget_bytes"].as_u64().unwrap()
        );
        let unverified = auto_quant(
            &mut n,
            &DEFAULT_CANDIDATES,
            Method::Proxy,
            Budget::Bpw(4.6),
            &AutoOpts {
                verify: false,
                ..quick()
            },
            &mut Control::none(),
        )
        .unwrap();
        assert!(unverified.measured.is_none() && unverified.to_markdown().contains("not verified"));
    }

    #[test]
    fn solving_is_deterministic() {
        let s = table();
        let a = solve(&s, Budget::Bpw(4.3), &SolveOpts::default()).unwrap();
        let b = solve(&s, Budget::Bpw(4.3), &SolveOpts::default()).unwrap();
        assert_eq!(a.choice, b.choice);
        assert_eq!(a.bytes, b.bytes);
    }
}
