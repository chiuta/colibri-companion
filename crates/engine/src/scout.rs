//! Looks at the model files on a disk and says, per file, what they are, whether corbul can run
//! them, whether they fit this computer, and roughly how fast they would decode on the processor
//! and on the graphics card.
//!
//! Everything about a file is **read from its header** (architecture, sizes, tensors, template),
//! so the numbers below are exact for the file. Everything about speed is an **estimate** from a
//! memory-bandwidth model, and is labelled as one: decoding one token reads the weights that
//! token uses once, so `tokens/s ≈ usable bandwidth / bytes read per token`. The processor's
//! bandwidth is *measured* on this machine ([`measure_ram_gbs`]); a card's bandwidth comes from a
//! table of approximate figures by name, or a range for unknown cards. Nothing here has been
//! compared with a real run on a real card.
use crate::arch::{self, Support};
use crate::gguf::{Gguf, TensorInfo};
use crate::hw::{DiskKind, GpuKind, HardwareProfile};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const GB: f64 = 1e9;

/// What the header of a model file says.
#[derive(Clone, Debug)]
pub struct Facts {
    pub path: PathBuf,
    pub name: String,
    pub arch: String,
    pub parts: usize,
    pub parts_missing: usize,
    pub bytes: u64,
    pub params: u64,
    /// The type holding most of the weight, e.g. `Q4K`.
    pub quant: String,
    pub bits_per_weight: f64,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub context: usize,
    pub experts: usize,
    pub used: usize,
    pub vocab: usize,
    pub chat_template: bool,
    pub tools: bool,
    /// A vision projector (`mmproj`) sits beside the file.
    pub vision: bool,
    pub support: Support,
    /// Bytes of weights one decoded token reads (routed experts counted at the used share, the
    /// embedding table not at all unless it doubles as the output head).
    pub active_bytes: u64,
    pub expert_bytes: u64,
    /// Matrix products one token needs (a device round trip each when offloaded).
    pub matrices_per_token: usize,
    /// Conversation memory per token of context, bytes (zero for models without attention).
    pub kv_bytes_per_token: u64,
}

impl Facts {
    pub fn is_moe(&self) -> bool {
        self.experts > 1
    }
    /// Billions of parameters.
    pub fn params_b(&self) -> f64 {
        self.params as f64 / GB
    }
    /// Parameters a token uses, billions (a mixture of experts uses a fraction).
    pub fn active_params_b(&self) -> f64 {
        if self.bytes == 0 {
            return 0.0;
        }
        self.params_b() * self.active_bytes as f64 / self.bytes as f64
    }
}

/// `…-00001-of-00004.gguf` → (`…-`, 1, 4, width). `None` for a file that is not split.
fn split_name(file: &str) -> Option<(String, usize, usize, usize)> {
    let stem = file
        .strip_suffix(".gguf")
        .or_else(|| file.strip_suffix(".GGUF"))?;
    let of = stem.rfind("-of-")?;
    let dash = stem[..of].rfind('-')?;
    let (a, b) = (&stem[dash + 1..of], &stem[of + 4..]);
    if a.is_empty() || b.is_empty() || !a.bytes().chain(b.bytes()).all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((
        stem[..=dash].to_string(),
        a.parse().ok()?,
        b.parse().ok()?,
        a.len(),
    ))
}

/// The parts of a split model that exist beside `first`, and how many are missing.
pub fn part_paths(first: &Path) -> (Vec<PathBuf>, usize) {
    let file = first.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let Some((prefix, _, total, width)) = split_name(file) else {
        return (vec![first.to_path_buf()], 0);
    };
    let dir = first.parent().unwrap_or(Path::new("."));
    let mut found = vec![];
    for i in 1..=total.min(4096) {
        let p = dir.join(format!("{prefix}{i:0width$}-of-{total:0width$}.gguf"));
        if p.is_file() {
            found.push(p);
        }
    }
    let missing = total.saturating_sub(found.len());
    (found, missing)
}

fn read_header(p: &Path) -> Result<Gguf, String> {
    let f = std::fs::File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    Gguf::parse_with_len(std::io::BufReader::new(f), len)
        .map_err(|e| format!("{}: {e}", p.display()))
}

fn dtype_name(t: &TensorInfo) -> String {
    format!("{:?}", t.dtype)
}

/// Reads the header of `first` (and of the other parts of a split model) and works out the facts.
pub fn inspect(first: &Path) -> Result<Facts, String> {
    let (paths, missing) = part_paths(first);
    let Some(head) = paths.first() else {
        return Err(format!("{}: the first part is not there", first.display()));
    };
    let g = read_header(head)?;
    let arch_name = g
        .architecture()
        .ok_or_else(|| format!("{}: no architecture in the header", first.display()))?
        .to_string();
    let mut tensors: Vec<TensorInfo> = g.tensors.clone();
    let mut bytes: u64 = paths
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    for p in paths.iter().skip(1) {
        tensors.extend(read_header(p)?.tensors);
    }
    if bytes == 0 {
        bytes = g.total_tensor_bytes();
    }
    let u = |k: &str| g.get_u64(k).unwrap_or(0) as usize;
    let layers = u("block_count");
    let hidden = u("embedding_length");
    let heads = u("attention.head_count");
    let kv_heads = match u("attention.head_count_kv") {
        0 => heads,
        n => n,
    };
    let head_dim = match u("attention.key_length") {
        0 if heads > 0 => hidden / heads,
        n => n,
    };
    let experts = u("expert_count");
    let used = match u("expert_used_count") {
        0 => experts.min(1),
        n => n.min(experts.max(1)),
    };
    let vocab = g
        .meta
        .get("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            crate::gguf::Value::Array(a) => Some(a.len()),
            _ => None,
        })
        .unwrap_or(0);
    let template = g
        .meta
        .get("tokenizer.chat_template")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let support = arch::classify(&arch_name);

    let tied = !tensors.iter().any(|t| t.name == "output.weight");
    let (mut params, mut active, mut expert_bytes, mut mats) = (0u64, 0f64, 0u64, 0f64);
    let mut by_type: HashMap<String, u64> = HashMap::new();
    for t in &tensors {
        let (n, b) = (t.elements().unwrap_or(0), t.bytes().unwrap_or(0));
        params += n;
        if t.dims.len() >= 2 {
            *by_type.entry(dtype_name(t)).or_default() += b;
        }
        let embedding = t.name.contains("token_embd");
        if t.name.contains("_exps.") {
            expert_bytes += b;
            let share = used as f64 / experts.max(1) as f64;
            active += b as f64 * share;
            mats += used as f64;
        } else if embedding && !tied {
            // a lookup of one row, not a pass over the table
        } else {
            active += b as f64;
            if t.dims.len() >= 2 && !embedding {
                mats += 1.0;
            }
        }
    }
    let quant = by_type
        .into_iter()
        .max_by_key(|(_, b)| *b)
        .map(|(n, _)| n)
        .unwrap_or_else(|| "?".into());
    let kv_bytes_per_token = if matches!(support, Support::Recurrent) {
        0
    } else {
        (2 * layers * kv_heads * head_dim * 4) as u64
    };
    let dir = first.parent().unwrap_or(Path::new("."));
    let vision = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                let n = e.file_name().to_string_lossy().to_ascii_lowercase();
                n.contains("mmproj") && n.ends_with(".gguf")
            })
        })
        .unwrap_or(false);
    let file = first.file_name().and_then(|f| f.to_str()).unwrap_or("");
    let name = g
        .meta
        .get("general.name")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            let stem = file.trim_end_matches(".gguf").trim_end_matches(".GGUF");
            split_name(file).map_or(stem.to_string(), |(p, ..)| {
                p.trim_end_matches('-').to_string()
            })
        });
    Ok(Facts {
        path: first.to_path_buf(),
        name,
        arch: arch_name,
        parts: paths.len() + missing,
        parts_missing: missing,
        bytes,
        params,
        quant,
        bits_per_weight: if params > 0 {
            bytes as f64 * 8.0 / params as f64
        } else {
            0.0
        },
        layers,
        hidden,
        heads,
        kv_heads,
        head_dim,
        context: u("context_length"),
        experts,
        used,
        vocab,
        chat_template: !template.is_empty(),
        tools: template.contains("tools") || template.contains("tool_call"),
        vision,
        support,
        active_bytes: active as u64,
        expert_bytes,
        matrices_per_token: mats as usize,
        kv_bytes_per_token,
    })
}

// ------------------------------------------------------------------------------- the machine

/// Approximate memory bandwidth of common cards, GB/s, by a part of the name (most specific first).
/// Figures are the vendors' published peaks, rounded; they are only used to rank and to bound.
const CARD_GBS: &[(&str, f64)] = &[
    ("5090", 1792.0),
    ("5080", 960.0),
    ("5070 ti", 896.0),
    ("5070", 672.0),
    ("5060 ti", 448.0),
    ("5060", 448.0),
    ("4090", 1008.0),
    ("4080", 717.0),
    ("4070 ti super", 672.0),
    ("4070 ti", 504.0),
    ("4070 super", 504.0),
    ("4070", 504.0),
    ("4060 ti", 288.0),
    ("4060", 272.0),
    ("3090", 936.0),
    ("3080 ti", 912.0),
    ("3080", 760.0),
    ("3070", 448.0),
    ("3060 ti", 448.0),
    ("3060", 360.0),
    ("3050", 224.0),
    ("2080", 448.0),
    ("2070", 448.0),
    ("2060", 336.0),
    ("1660", 288.0),
    ("1650", 128.0),
    ("1080 ti", 484.0),
    ("1080", 320.0),
    ("1070", 256.0),
    ("1060", 192.0),
    ("7900 xtx", 960.0),
    ("7900 xt", 800.0),
    ("7900 gre", 576.0),
    ("7800 xt", 624.0),
    ("7700 xt", 432.0),
    ("7600", 288.0),
    ("6950", 576.0),
    ("6900", 512.0),
    ("6800", 512.0),
    ("6700", 384.0),
    ("6600", 224.0),
    ("a770", 560.0),
    ("a750", 512.0),
    ("a380", 186.0),
];

#[derive(Clone, Debug)]
pub struct GpuSpec {
    pub name: String,
    /// Memory the card has for weights; `None` when the system does not say.
    pub vram: Option<u64>,
    /// Bandwidth range, GB/s, and whether it came from the table (`true`) or is a class guess.
    pub gbs: (f64, f64),
    pub known: bool,
    /// Shares main memory with the processor.
    pub shared: bool,
}

#[derive(Clone, Debug)]
pub struct Machine {
    /// Measured sustained read bandwidth of main memory, GB/s.
    pub cpu_gbs: f64,
    pub threads: usize,
    pub ram_total: Option<u64>,
    pub ram_free: Option<u64>,
    pub disk: DiskKind,
    pub gpu: Option<GpuSpec>,
}

fn gpu_spec(hw: &HardwareProfile, cpu_gbs: f64) -> Option<GpuSpec> {
    // the card with the most memory, discrete ones first
    let g = hw
        .gpus
        .iter()
        .filter(|g| g.kind != GpuKind::Other || g.vram.is_some())
        .max_by_key(|g| (g.vram.is_some() as u8, g.vram.unwrap_or(0)))?;
    let shared =
        hw.unified_memory || g.kind == GpuKind::Apple || g.vram.is_none_or(|v| v < 1 << 30);
    let lower = g.name.to_ascii_lowercase();
    let table = CARD_GBS
        .iter()
        .find(|(n, _)| lower.contains(n))
        .map(|(_, b)| *b);
    let (gbs, known) = match (shared, table) {
        (true, _) => ((cpu_gbs * 0.9, cpu_gbs * 1.6), false),
        (false, Some(b)) => ((b * 0.9, b), true),
        (false, None) => ((150.0, 500.0), false),
    };
    let ram = hw.total_ram.unwrap_or(0);
    Some(GpuSpec {
        name: g.name.clone(),
        vram: if shared {
            Some(ram / 2).filter(|v| *v > 0)
        } else {
            g.vram
        },
        gbs,
        known,
        shared,
    })
}

/// Describes this computer from a detected profile and a measured memory bandwidth.
pub fn machine(hw: &HardwareProfile, cpu_gbs: f64) -> Machine {
    Machine {
        cpu_gbs: if cpu_gbs.is_finite() && cpu_gbs > 0.1 {
            cpu_gbs
        } else {
            8.0
        },
        threads: hw.usable_threads.max(1),
        ram_total: hw.total_ram,
        ram_free: hw.available_ram,
        disk: hw.disk,
        gpu: gpu_spec(hw, cpu_gbs),
    }
}

/// Sustained read bandwidth of main memory, GB/s, measured by summing a buffer too large for the
/// caches on every core. Takes about a third of a second. Returns the best of three passes.
pub fn measure_ram_gbs() -> f64 {
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(32);
    let per = 32usize << 20; // 32 MiB of f32 per thread: well past any cache
    let mut best = 0f64;
    let bufs: Vec<Vec<f32>> = (0..threads)
        .map(|t| (0..per / 4).map(|i| ((i + t) % 97) as f32).collect())
        .collect();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let sum: f32 = std::thread::scope(|s| {
            let hs: Vec<_> = bufs
                .iter()
                .map(|b| s.spawn(move || b.chunks(8).map(|c| c.iter().sum::<f32>()).sum::<f32>()))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(0.0)).sum()
        });
        let dt = t0.elapsed().as_secs_f64().max(1e-6);
        std::hint::black_box(sum);
        best = best.max((threads * per) as f64 / dt / GB);
    }
    best
}

// ------------------------------------------------------------------------------- the verdict

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Weights and conversation fit in the memory of the card.
    Gpu,
    /// Fit in main memory.
    Ram,
    /// Larger than the free memory: the disk is read while answering.
    Disk,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    pub low: f64,
    pub high: f64,
}
impl Range {
    pub fn mid(&self) -> f64 {
        (self.low.max(1e-9) * self.high.max(1e-9)).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct Assessment {
    /// Memory the model needs: weights, conversation at `ctx` tokens, a margin.
    pub need_bytes: u64,
    pub fit: Fit,
    /// Decoding speed on the processor, tokens per second.
    pub cpu: Range,
    /// Decoding speed with the card, when there is one.
    pub gpu: Option<Range>,
    /// Share of the weights the card can hold.
    pub gpu_share: f64,
    pub bytes_per_token: u64,
    pub kv_bytes: u64,
    pub ctx: usize,
    /// The card is expected to beat the processor.
    pub gpu_wins: bool,
}

/// Share of the bandwidth the engine reaches on the processor (low, high). The engine's own
/// README measures well under half of the peak on small machines; a streaming read is the ceiling.
const CPU_EFF: (f64, f64) = (0.35, 0.75);
/// Same for a card driven by this engine (one write, one product and one read per matrix).
const GPU_EFF: (f64, f64) = (0.25, 0.6);
/// Round trip per matrix on the card, seconds: slow (high) and fast (low).
const GPU_LAT: (f64, f64) = (150e-6, 40e-6);
const MAX_TOK_S: f64 = 2000.0;

fn disk_gbs(d: DiskKind) -> f64 {
    match d {
        DiskKind::Hdd => 0.1,
        DiskKind::Ssd => 1.5,
        DiskKind::Unknown => 0.8,
    }
}

fn rate(secs: f64) -> f64 {
    (1.0 / secs.max(1.0 / MAX_TOK_S)).min(MAX_TOK_S)
}

/// Estimates fit and decoding speed of `f` on `m` with a conversation of `ctx` tokens.
pub fn assess(f: &Facts, m: &Machine, ctx: usize) -> Assessment {
    let ctx = ctx.max(1).min(f.context.max(1));
    let kv = f.kv_bytes_per_token * ctx as u64;
    let need = ((f.bytes + kv) as f64 * 1.08) as u64;
    // reading the conversation back costs bandwidth too (about half of it on average)
    let per_tok = f.active_bytes as f64 + kv as f64 * 0.5;
    let per_tok_gb = per_tok / GB;

    let free = m.ram_free.or(m.ram_total).map(|r| r as f64);
    let in_ram = free.map_or(1.0, |r| (r * 0.85 / need.max(1) as f64).min(1.0));
    let dgbs = disk_gbs(m.disk);
    let cpu_time = |eff: f64| per_tok_gb * (in_ram / (m.cpu_gbs * eff) + (1.0 - in_ram) / dgbs);
    let cpu = Range {
        low: rate(cpu_time(CPU_EFF.0)),
        high: rate(cpu_time(CPU_EFF.1)),
    };

    let mut gpu = None;
    let mut share = 0.0;
    let mut fit = if in_ram >= 1.0 { Fit::Ram } else { Fit::Disk };
    if let Some(g) = &m.gpu {
        let budget = g.vram.map_or(0.0, |v| v as f64 * 0.9);
        share = (budget / need.max(1) as f64).min(1.0);
        if share >= 1.0 && !g.shared {
            fit = Fit::Gpu;
        }
        if share > 0.0 {
            let mats = f.matrices_per_token as f64;
            let t = |gbs: f64, eff: f64, lat: f64, cpu_eff: f64| {
                let on_gpu = share * per_tok_gb / (gbs * eff) + share * mats * lat;
                let rest = (1.0 - share)
                    * per_tok_gb
                    * (in_ram / (m.cpu_gbs * cpu_eff) + (1.0 - in_ram) / dgbs);
                on_gpu + rest
            };
            gpu = Some(Range {
                low: rate(t(g.gbs.0, GPU_EFF.0, GPU_LAT.0, CPU_EFF.0)),
                high: rate(t(g.gbs.1, GPU_EFF.1, GPU_LAT.1, CPU_EFF.1)),
            });
        }
    }
    let gpu_wins = gpu.is_some_and(|g| g.mid() > cpu.mid() * 1.15);
    Assessment {
        need_bytes: need,
        fit,
        cpu,
        gpu,
        gpu_share: share,
        bytes_per_token: per_tok as u64,
        kv_bytes: kv,
        ctx,
        gpu_wins,
    }
}

/// One model file with its facts and its estimate.
#[derive(Clone, Debug)]
pub struct Entry {
    pub facts: Facts,
    pub assessment: Assessment,
}

/// Reads and assesses every file in `paths` on `m`; a file that cannot be read comes back as an
/// error string with its path instead of stopping the rest.
pub fn survey(paths: &[PathBuf], m: &Machine, ctx: usize) -> Vec<Result<Entry, (PathBuf, String)>> {
    paths
        .iter()
        .map(|p| {
            inspect(p).map_err(|e| (p.clone(), e)).map(|facts| {
                let assessment = assess(&facts, m, ctx);
                Entry { facts, assessment }
            })
        })
        .collect()
}

/// Every `.gguf` model (first part only) under `roots`, to `depth` levels; vision projectors and
/// files under 20 MB are skipped, links are not followed.
pub fn find(roots: &[PathBuf], depth: usize) -> Vec<PathBuf> {
    let mut out = vec![];
    let mut seen = 0usize;
    for root in roots {
        let mut stack = vec![(root.clone(), 0usize)];
        while let Some((d, level)) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&d) else {
                continue;
            };
            for e in rd.flatten() {
                seen += 1;
                if seen > 200_000 {
                    return out;
                }
                let Ok(ft) = e.file_type() else { continue };
                if ft.is_symlink() {
                    continue;
                }
                let p = e.path();
                if ft.is_dir() {
                    if level < depth {
                        stack.push((p, level + 1));
                    }
                    continue;
                }
                let file = e.file_name().to_string_lossy().to_ascii_lowercase();
                if !file.ends_with(".gguf") || file.contains("mmproj") {
                    continue;
                }
                if let Some((_, n, ..)) = split_name(&file)
                    && n != 1
                {
                    continue;
                }
                if e.metadata().map(|m| m.len()).unwrap_or(0) < 20 << 20 || out.contains(&p) {
                    continue;
                }
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_write::{GgufWriter, MetaValue as M, TensorDesc};
    use crate::quant::DType;

    fn write(path: &Path, arch: &str, experts: u32, used: u32, split_name_ok: bool) {
        let (h, v, ff, layers) = (64usize, 100usize, 128usize, 3usize);
        let mut t: Vec<(String, Vec<u64>, usize)> = vec![
            ("token_embd.weight".into(), vec![h as u64, v as u64], h * v),
            ("output_norm.weight".into(), vec![h as u64], h),
            ("output.weight".into(), vec![h as u64, v as u64], h * v),
        ];
        for i in 0..layers {
            for (n, d) in [
                ("attn_q.weight", vec![h, h]),
                ("attn_k.weight", vec![h, 32]),
                ("attn_v.weight", vec![h, 32]),
                ("attn_output.weight", vec![h, h]),
            ] {
                t.push((
                    format!("blk.{i}.{n}"),
                    d.iter().map(|x| *x as u64).collect(),
                    d.iter().product(),
                ));
            }
            if experts > 0 {
                for n in ["ffn_gate_exps", "ffn_up_exps"] {
                    t.push((
                        format!("blk.{i}.{n}.weight"),
                        vec![h as u64, ff as u64, experts as u64],
                        h * ff * experts as usize,
                    ));
                }
                t.push((
                    format!("blk.{i}.ffn_down_exps.weight"),
                    vec![ff as u64, h as u64, experts as u64],
                    h * ff * experts as usize,
                ));
            } else {
                for n in ["ffn_gate", "ffn_up", "ffn_down"] {
                    t.push((
                        format!("blk.{i}.{n}.weight"),
                        vec![h as u64, ff as u64],
                        h * ff,
                    ));
                }
            }
        }
        let mut meta: Vec<(String, M)> = vec![
            ("general.architecture".into(), M::str(arch)),
            ("general.name".into(), M::str("Tiny Test")),
            (format!("{arch}.block_count"), M::U32(layers as u32)),
            (format!("{arch}.context_length"), M::U32(4096)),
            (format!("{arch}.embedding_length"), M::U32(h as u32)),
            (format!("{arch}.attention.head_count"), M::U32(4)),
            (format!("{arch}.attention.head_count_kv"), M::U32(2)),
            (
                "tokenizer.chat_template".into(),
                M::str("{% if tools %}x{% endif %}"),
            ),
        ];
        if experts > 0 {
            meta.push((format!("{arch}.expert_count"), M::U32(experts)));
            meta.push((format!("{arch}.expert_used_count"), M::U32(used)));
        }
        let _ = split_name_ok;
        let descs: Vec<TensorDesc> = t
            .iter()
            .map(|(n, s, _)| TensorDesc::new(n.clone(), s, DType::F32))
            .collect();
        let file = std::fs::File::create(path).unwrap();
        GgufWriter::write_all(file, &meta, descs, |i, _| Ok(vec![0u8; t[i].2 * 4])).unwrap();
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("scout-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn big(f: &Facts, bytes: u64) -> Facts {
        let k = bytes as f64 / f.bytes as f64;
        Facts {
            bytes,
            active_bytes: (f.active_bytes as f64 * k) as u64,
            expert_bytes: (f.expert_bytes as f64 * k) as u64,
            ..f.clone()
        }
    }

    fn machine_with(gpu: Option<GpuSpec>, ram_gb: u64) -> Machine {
        Machine {
            cpu_gbs: 40.0,
            threads: 8,
            ram_total: Some(ram_gb << 30),
            ram_free: Some(ram_gb << 30),
            disk: DiskKind::Ssd,
            gpu,
        }
    }
    fn card(vram_gb: u64) -> GpuSpec {
        GpuSpec {
            name: "RTX 4070".into(),
            vram: Some(vram_gb << 30),
            gbs: (454.0, 504.0),
            known: true,
            shared: false,
        }
    }

    #[test]
    fn a_dense_file_is_read_exactly() {
        let d = tmp("dense");
        let p = d.join("tiny.gguf");
        write(&p, "llama", 0, 0, false);
        let f = inspect(&p).unwrap();
        assert_eq!(
            (f.arch.as_str(), f.layers, f.hidden, f.heads, f.kv_heads),
            ("llama", 3, 64, 4, 2)
        );
        assert_eq!(f.name, "Tiny Test");
        assert!(f.chat_template && f.tools && !f.vision && !f.is_moe());
        assert_eq!(f.quant, "F32");
        assert!((f.bits_per_weight - 32.0).abs() < 1.0);
        assert!(f.support.runnable());
        // the output head is separate, so the embedding table is a lookup, not a pass
        let table = 64 * 100 * 4;
        assert!(f.active_bytes + table as u64 <= f.bytes);
        assert_eq!(f.kv_bytes_per_token, (2 * 3 * 2 * 16 * 4) as u64);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_mixture_reads_only_the_used_share_of_its_experts() {
        let d = tmp("moe");
        let p = d.join("moe.gguf");
        write(&p, "qwen3moe", 8, 2, false);
        let f = inspect(&p).unwrap();
        assert!(f.is_moe() && f.experts == 8 && f.used == 2);
        assert!(
            f.active_bytes < f.bytes / 2,
            "{} of {}",
            f.active_bytes,
            f.bytes
        );
        assert!(f.active_params_b() < f.params_b());
        assert!(f.matrices_per_token >= 3 * 2 * 3);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn unsupported_and_broken_files_are_reported_not_panicked_on() {
        let d = tmp("bad");
        let p = d.join("x.gguf");
        std::fs::write(&p, b"not a model").unwrap();
        assert!(inspect(&p).is_err());
        assert!(inspect(&d.join("missing.gguf")).is_err());
        write(&p, "opt", 0, 0, false);
        assert!(!inspect(&p).unwrap().support.runnable());
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn split_models_are_found_once_and_missing_parts_counted() {
        assert_eq!(
            split_name("m-00001-of-00003.gguf").map(|s| (s.1, s.2)),
            Some((1, 3))
        );
        assert!(split_name("plain.gguf").is_none());
        let d = tmp("split");
        write(&d.join("m-00001-of-00003.gguf"), "llama", 0, 0, true);
        write(&d.join("m-00002-of-00003.gguf"), "llama", 0, 0, true);
        let (parts, missing) = part_paths(&d.join("m-00001-of-00003.gguf"));
        assert_eq!((parts.len(), missing), (2, 1));
        let f = inspect(&d.join("m-00001-of-00003.gguf")).unwrap();
        assert_eq!((f.parts, f.parts_missing), (3, 1));
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn speed_follows_size_bandwidth_and_the_card() {
        let d = tmp("speed");
        let p = d.join("tiny.gguf");
        write(&p, "llama", 0, 0, false);
        let f = big(&inspect(&p).unwrap(), 4_000_000_000);
        let cpu_only = assess(&f, &machine_with(None, 32), 2048);
        assert!(cpu_only.gpu.is_none() && !cpu_only.gpu_wins && cpu_only.fit == Fit::Ram);
        assert!(cpu_only.cpu.low <= cpu_only.cpu.high && cpu_only.cpu.low > 0.5);
        // a bigger model is slower, a faster memory is faster
        let bigger = assess(&big(&f, 16_000_000_000), &machine_with(None, 32), 2048);
        assert!(bigger.cpu.mid() < cpu_only.cpu.mid());
        let mut fast = machine_with(None, 32);
        fast.cpu_gbs = 80.0;
        assert!(assess(&f, &fast, 2048).cpu.mid() > cpu_only.cpu.mid());
        // a card that holds the model beats the processor; one that holds half is in between
        let full = assess(&f, &machine_with(Some(card(12)), 32), 2048);
        assert!(full.fit == Fit::Gpu && full.gpu_share >= 1.0 && full.gpu_wins);
        let half = assess(&f, &machine_with(Some(card(3)), 32), 2048);
        assert!(half.gpu_share < 1.0 && half.gpu.unwrap().mid() < full.gpu.unwrap().mid());
        assert!(half.gpu.unwrap().mid() > half.cpu.mid() * 0.9);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_model_bigger_than_memory_streams_from_the_disk() {
        let d = tmp("disk");
        let p = d.join("tiny.gguf");
        write(&p, "llama", 0, 0, false);
        let f = big(&inspect(&p).unwrap(), 40_000_000_000);
        let a = assess(&f, &machine_with(None, 16), 2048);
        assert_eq!(a.fit, Fit::Disk);
        assert!(a.cpu.high < 2.0, "{a:?}");
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn card_names_map_to_bandwidth_most_specific_first() {
        let hw = |name: &str| HardwareProfile {
            gpus: vec![crate::hw::Gpu {
                name: name.into(),
                vram: Some(12 << 30),
                kind: GpuKind::Nvidia,
            }],
            total_ram: Some(32 << 30),
            usable_threads: 8,
            ..Default::default()
        };
        let a = machine(&hw("NVIDIA GeForce RTX 4070 Ti SUPER"), 40.0)
            .gpu
            .unwrap();
        let b = machine(&hw("NVIDIA GeForce RTX 4070"), 40.0).gpu.unwrap();
        assert!(a.known && b.known && a.gbs.1 > b.gbs.1);
        let u = machine(&hw("NVIDIA Mystery 9000"), 40.0).gpu.unwrap();
        assert!(!u.known && u.gbs.0 < u.gbs.1);
        assert!(machine(&HardwareProfile::default(), 40.0).gpu.is_none());
    }

    #[test]
    fn the_memory_probe_gives_a_sane_number() {
        let g = measure_ram_gbs();
        assert!(g.is_finite() && g > 0.1 && g < 5000.0, "{g}");
    }

    #[test]
    fn find_lists_first_parts_only_and_skips_projectors() {
        let d = tmp("find");
        let big = |n: &str| {
            let f = std::fs::File::create(d.join(n)).unwrap();
            f.set_len(21 << 20).unwrap();
        };
        big("a.gguf");
        big("b-00001-of-00002.gguf");
        big("b-00002-of-00002.gguf");
        big("mmproj-x.gguf");
        std::fs::write(d.join("small.gguf"), b"x").unwrap();
        let got: Vec<String> = find(std::slice::from_ref(&d), 2)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(got, ["a.gguf", "b-00001-of-00002.gguf"]);
        std::fs::remove_dir_all(d).unwrap();
    }
}
