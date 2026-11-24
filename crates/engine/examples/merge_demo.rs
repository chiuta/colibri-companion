//! Model merging demo.
//!
//! Without arguments it builds two tiny synthetic "fine-tunes" of a common base in a temp folder
//! and merges them with every method, printing the plan, the accounted peak memory and how far
//! each result is from the inputs. With arguments it merges real files:
//!
//! `merge_demo <config.json> <out.safetensors> <input> <input> ...`
//!
//! (inputs: `.safetensors` files or folders, or `.gguf` files; see `merge::MergeConfig::from_json`).
use companion_engine::loader::{SafeTensors, TensorSource};
use companion_engine::merge::{merge, plan, write_safetensors, Input, MergeConfig, OutDType};
use std::collections::BTreeMap;
use std::path::Path;

fn run(inputs: &[Input], cfg: &MergeConfig, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let p = plan(inputs, cfg)?;
    println!("plan: {} tensors, {} values, output {} bytes, accounted peak {} KiB (largest tensor {} values){}",
        p.tensors.len(), p.total_elems, p.out_bytes, p.peak_f32 * 4 / 1024, p.max_tensor_elems,
        p.out_layers.map(|l| format!(", {l} layers")).unwrap_or_default());
    let st = merge(inputs, cfg, out, &mut |pr| {
        if pr.done % 50 == 0 || pr.done == pr.total { println!("  {}/{} {}", pr.done, pr.total, pr.tensor); }
    })?;
    println!("wrote {} ({} bytes)", out.display(), st.bytes_written);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() >= 3 {
        let cfg = MergeConfig::from_json(&std::fs::read_to_string(&args[0])?)?;
        let inputs = args[2..].iter().map(|p| Input::open(Path::new(p))).collect::<Result<Vec<_>, _>>()?;
        return run(&inputs, &cfg, Path::new(&args[1]));
    }
    let dir = std::env::temp_dir().join(format!("merge-demo-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let mut s = 1u64;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> { (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / 16_777_216.0 - 0.5) * scale }).collect() };
    let shapes: Vec<(String, Vec<u64>)> = std::iter::once(("embed.weight".to_string(), vec![64u64, 32]))
        .chain((0..4).flat_map(|l| [(format!("layers.{l}.attn.weight"), vec![32u64, 32]), (format!("layers.{l}.mlp.weight"), vec![64, 32]), (format!("layers.{l}.norm.weight"), vec![32])])).collect();
    let base: Vec<(String, Vec<u64>, Vec<f32>)> = shapes.iter().map(|(n, sh)| (n.clone(), sh.clone(), rnd(sh.iter().product::<u64>() as usize, 1.0))).collect();
    let tune = |scale: f32, rnd: &mut dyn FnMut(usize, f32) -> Vec<f32>| -> Vec<(String, Vec<u64>, Vec<f32>)> {
        base.iter().map(|(n, sh, w)| { let d = rnd(w.len(), scale); (n.clone(), sh.clone(), w.iter().zip(d).map(|(a, b)| a + b).collect()) }).collect()
    };
    let (a, b) = (tune(0.2, &mut rnd), tune(0.2, &mut rnd));
    let meta = BTreeMap::new();
    for (name, t) in [("base", &base), ("a", &a), ("b", &b)] { write_safetensors(&dir.join(format!("{name}.safetensors")), t, OutDType::F32, &meta)?; }
    let open = |names: &[&str]| names.iter().map(|n| Input::open(&dir.join(format!("{n}.safetensors")))).collect::<Result<Vec<_>, _>>();
    let configs = [
        ("linear", r#"{"method":"linear","weights":[0.5,0.5],"dtype":"f16"}"#, vec!["a", "b"]),
        ("slerp", r#"{"method":"slerp","t":0.5,"dtype":"bf16"}"#, vec!["a", "b"]),
        ("task", r#"{"method":"task","base":0,"scale":1.0}"#, vec!["base", "a", "b"]),
        ("ties", r#"{"method":"ties","base":0,"density":0.5}"#, vec!["base", "a", "b"]),
        ("dare", r#"{"method":"dare","base":0,"density":0.3,"seed":7,"sign_election":true}"#, vec!["base", "a", "b"]),
        ("frankenmerge", r#"{"slices":[{"input":0,"start":0,"end":3},{"input":1,"start":1,"end":4}],"dtype":"f16"}"#, vec!["a", "b"]),
        ("per-tensor rules", r#"{"method":"linear","rules":[{"match":"*norm*","method":"passthrough","input":0}],"dtype":"f32"}"#, vec!["a", "b"]),
    ];
    for (title, json, names) in configs {
        println!("\n== {title}");
        let out = dir.join(format!("out-{}.safetensors", title.replace(' ', "-")));
        run(&open(&names)?, &MergeConfig::from_json(json)?, &out)?;
        let m = SafeTensors::open(&out)?.read("embed.weight")?;
        let (ra, rb) = (SafeTensors::open(&dir.join("a.safetensors"))?.read("embed.weight")?, SafeTensors::open(&dir.join("b.safetensors"))?.read("embed.weight")?);
        let dist = |x: &[f32], y: &[f32]| (x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum::<f32>() / x.len() as f32).sqrt();
        println!("  rms distance of embed.weight: to a {:.4}, to b {:.4}", dist(&m, &ra), dist(&m, &rb));
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}
