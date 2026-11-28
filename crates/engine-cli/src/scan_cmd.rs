//! `corbul scan [FOLDER…] [--ctx N] [--depth N] [--json]`: find the `.gguf` models in the folders
//! (or on the drives) given, say what each one is, whether corbul runs it, whether it fits this
//! computer, and estimate how fast it decodes on the processor and with the graphics card.
//!
//! The file facts are read from the headers and are exact. The speeds are estimates from a
//! memory-bandwidth model (see `companion_engine::scout`); the processor's bandwidth is measured
//! here and now, a card's comes from a table by name. No model is loaded or run.
use crate::args::Args;
use companion_engine::arch::Support;
use companion_engine::hw::HardwareProfile;
use companion_engine::scout::{self, Assessment, Fit, Machine, Range};
use std::path::PathBuf;

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}
fn tps(r: Range) -> String {
    if r.high < 10.0 {
        format!("{:.1}–{:.1}", r.low, r.high)
    } else {
        format!("{:.0}–{:.0}", r.low, r.high)
    }
}

pub fn support_text(s: &Support) -> String {
    match s {
        Support::Standard | Support::Moe => "runs (corbul)".into(),
        Support::Recurrent | Support::Hybrid => "runs (corbul, state-space)".into(),
        Support::Native(w) => format!("runs (corbul, {w})"),
        Support::Partial(w) => format!("partial: {w}"),
        Support::Planned(w) => format!("not yet: {w}"),
    }
}

pub fn fit_text(a: &Assessment) -> &'static str {
    match a.fit {
        Fit::Gpu => "fits the card",
        Fit::Ram => "fits in memory",
        Fit::Disk => "larger than free memory (reads the disk)",
    }
}

/// The machine line printed above the table.
pub fn machine_line(m: &Machine) -> String {
    let gpu = match &m.gpu {
        Some(g) => format!(
            " · card {} ({}, {}–{} GB/s{})",
            g.name,
            g.vram.map_or("memory unknown".to_string(), gb),
            g.gbs.0.round(),
            g.gbs.1.round(),
            if g.known { "" } else { ", guessed" }
        ),
        None => " · no usable card found".into(),
    };
    format!(
        "this computer: {} threads · memory {} (measured read {:.0} GB/s){gpu}",
        m.threads,
        m.ram_total.map_or("?".to_string(), gb),
        m.cpu_gbs
    )
}

pub fn cmd(args: &Args) -> Result<(), String> {
    let ctx: usize = args.num("ctx")?.unwrap_or(2048);
    let depth: usize = args.num("depth")?.unwrap_or(4);
    let roots: Vec<PathBuf> = if args.pos.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        args.pos.iter().map(PathBuf::from).collect()
    };
    let files = scout::find(&roots, depth);
    let hw = HardwareProfile::detect(roots.first().map(|p| p.as_path()));
    let m = scout::machine(&hw, scout::measure_ram_gbs());
    let rows = scout::survey(&files, &m, ctx);
    if args.flag("json") {
        let v: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| match r {
                Ok(e) => {
                    let (f, a) = (&e.facts, &e.assessment);
                    serde_json::json!({
                        "path": f.path, "name": f.name, "arch": f.arch, "bytes": f.bytes,
                        "params_b": f.params_b(), "active_params_b": f.active_params_b(),
                        "quant": f.quant, "bits_per_weight": f.bits_per_weight,
                        "context": f.context, "experts": f.experts, "used": f.used,
                        "chat_template": f.chat_template, "tools": f.tools, "vision": f.vision,
                        "parts": f.parts, "parts_missing": f.parts_missing,
                        "runs": f.support.runnable(), "support": support_text(&f.support),
                        "need_bytes": a.need_bytes, "fit": fit_text(a),
                        "bytes_per_token": a.bytes_per_token, "kv_bytes": a.kv_bytes, "ctx": a.ctx,
                        "cpu_tok_s": [a.cpu.low, a.cpu.high],
                        "gpu_tok_s": a.gpu.map(|g| [g.low, g.high]),
                        "gpu_share": a.gpu_share, "gpu_wins": a.gpu_wins,
                    })
                }
                Err((p, e)) => serde_json::json!({ "path": p, "error": e }),
            })
            .collect();
        outln!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "machine": machine_line(&m), "ctx": ctx, "models": v
            }))
            .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    outln!("{}", machine_line(&m));
    outln!(
        "speeds are ESTIMATES of decoding (tokens/s, low–high) at {ctx} tokens of context; file facts are exact"
    );
    if files.is_empty() {
        outln!(
            "no .gguf models found under {}",
            roots
                .iter()
                .map(|r| r.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    for r in rows {
        match r {
            Ok(e) => {
                let (f, a) = (&e.facts, &e.assessment);
                outln!();
                outln!("{}  [{}]", f.name, f.path.display());
                outln!(
                    "  {} · {} · {:.1}B params{} · {} ({:.1} bits/weight) · context {}",
                    f.arch,
                    gb(f.bytes),
                    f.params_b(),
                    if f.is_moe() {
                        format!(
                            " ({:.1}B active, {} of {} experts)",
                            f.active_params_b(),
                            f.used,
                            f.experts
                        )
                    } else {
                        String::new()
                    },
                    f.quant,
                    f.bits_per_weight,
                    f.context
                );
                let mut caps = vec![];
                if f.chat_template {
                    caps.push("chat template");
                }
                if f.tools {
                    caps.push("tool calls");
                }
                if f.vision {
                    caps.push("vision projector beside it");
                }
                if f.is_moe() {
                    caps.push("mixture of experts");
                }
                outln!(
                    "  can: {}",
                    if caps.is_empty() {
                        "plain text completion".into()
                    } else {
                        caps.join(", ")
                    }
                );
                outln!("  engine: {}", support_text(&f.support));
                if f.parts_missing > 0 {
                    outln!(
                        "  WARNING: {} of {} parts are missing beside the first file",
                        f.parts_missing,
                        f.parts
                    );
                }
                outln!(
                    "  memory: needs {} at {} tokens ({} of it conversation) · {}",
                    gb(a.need_bytes),
                    a.ctx,
                    gb(a.kv_bytes),
                    fit_text(a)
                );
                outln!("  reads {} per token", gb(a.bytes_per_token));
                outln!("  CPU : {} tok/s", tps(a.cpu));
                match a.gpu {
                    Some(g) => outln!(
                        "  GPU : {} tok/s ({:.0}% of the weights on the card){}",
                        tps(g),
                        a.gpu_share * 100.0,
                        if a.gpu_wins {
                            " — faster than the CPU"
                        } else {
                            " — not clearly faster than the CPU"
                        }
                    ),
                    None => outln!("  GPU : no usable card"),
                }
            }
            Err((p, e)) => {
                outln!();
                outln!("{}\n  unreadable: {e}", p.display());
            }
        }
    }
    Ok(())
}
