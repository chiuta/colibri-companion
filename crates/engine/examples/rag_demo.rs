//! Retrieval demo: chunk two small documents, index them with the model-free hash embedder, run a
//! hybrid (vector + BM25) search, build quote-lock sources from the hits, and time the HNSW index
//! against brute force on random vectors.
//!
//! `cargo run --release -p companion-engine --example rag_demo -- [n_vectors] [dim]`
use companion_engine::quotelock::Options as QuoteOptions;
use companion_engine::rag::{ChunkOptions, HashEmbedder, Hnsw, HnswParams, KnowledgeBase, SearchOptions, Storage};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);

    let docs = [
        ("sarmale", "Sarmalele se fac din carne tocată și varză murată. Se fierb la foc mic câteva ore. Se servesc cu smântână și mămăligă."),
        ("cosmos", "Soarele este o stea din galaxia noastră. Planetele se rotesc în jurul Soarelui. Jupiter este cea mai mare planetă."),
    ];
    let mut kb = KnowledgeBase::new(Box::new(HashEmbedder::new(1024, 7)), HnswParams::default()).unwrap();
    for (name, text) in docs {
        kb.add_document(name, text, &ChunkOptions { max_chars: 90, overlap_chars: 20 }).unwrap();
    }
    println!("{} documents, {} chunks", kb.doc_count(), kb.chunk_count());
    let ctx = kb.retrieve_cited("cum se prepară sarmalele?", &SearchOptions { k: 2, ..Default::default() }).unwrap();
    print!("{}", ctx.prompt_block());
    let idx = ctx.quote_index(QuoteOptions::default()).unwrap();
    println!("quote index over {} sources ready for the quote lock", idx.source_count());
    if let Some(c) = idx.find("varză murată") {
        let span = ctx.locate(&c).unwrap();
        println!("\"varză murată\" is bytes {}..{} of {:?}", span.byte_start, span.byte_end, span.doc_name);
    }

    // synthetic scale test
    let mut seed = 1u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    for storage in [Storage::F32, Storage::Int8] {
        let mut h = Hnsw::new(dim, HnswParams { storage, ..Default::default() }).unwrap();
        let t = Instant::now();
        for i in 0..n { let v: Vec<f32> = (0..dim).map(|_| rnd()).collect(); h.insert(i as u64, &v).unwrap(); }
        let build = t.elapsed();
        let queries: Vec<Vec<f32>> = (0..200).map(|_| (0..dim).map(|_| rnd()).collect()).collect();
        let (mut hits, mut t_h, mut t_b) = (0usize, 0.0f64, 0.0f64);
        for q in &queries {
            let a = Instant::now();
            let approx = h.search(q, 10, None).unwrap();
            t_h += a.elapsed().as_secs_f64();
            let b = Instant::now();
            let exact = h.brute_force(q, 10).unwrap();
            t_b += b.elapsed().as_secs_f64();
            hits += approx.iter().filter(|x| exact.iter().any(|e| e.0 == x.0)).count();
        }
        println!(
            "{storage:?}: {n}×{dim} built in {:.2}s, recall@10 {:.3}, {:.0} µs/query vs {:.0} µs brute force, file {} KiB",
            build.as_secs_f64(), hits as f64 / (queries.len() * 10) as f64, t_h / 200.0 * 1e6, t_b / 200.0 * 1e6, h.to_bytes().len() / 1024
        );
    }
}
