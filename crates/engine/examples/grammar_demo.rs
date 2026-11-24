//! Structured-output demo: compile a JSON schema, build a token trie over a synthetic
//! byte-level vocabulary and time the logit masks while "generating" a document.
//!
//! `cargo run --release -p companion-engine --example grammar_demo`
use companion_engine::grammar::*;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let schema: serde_json::Value = serde_json::from_str(r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 12},
            "age": {"type": "integer", "minimum": 0},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "kind": {"enum": ["cat", "dog", null]}
        },
        "required": ["name", "age"]
    }"#).unwrap();
    let opts = SchemaOptions { whitespace: Whitespace::Compact, ..Default::default() };
    let t = Instant::now();
    let gram = Arc::new(Grammar::from_json_schema(&schema, &opts).unwrap());
    println!("grammar: {} rules, {} elements, compiled in {:?}", gram.rule_count(), gram.element_count(), t.elapsed());

    // synthetic 64k vocabulary: every byte, then random 2..6 byte "words" over printable ASCII
    let mut toks: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
    let mut x = 0x2545F4914F6CDD1Du64;
    let mut next = || { x ^= x << 13; x ^= x >> 7; x ^= x << 17; x };
    while toks.len() < 64_000 {
        let n = 2 + (next() % 5) as usize;
        toks.push((0..n).map(|_| 32 + (next() % 95) as u8).collect());
    }
    for w in ["{\"", "\":", "\",\"", "null", "true", "\"name\"", "\"age\"", "\"tags\"", "\"kind\"", "\"cat\"", "\"dog\"", "\"]}"] { toks.push(w.as_bytes().to_vec()); }
    let eos = toks.len() as u32;
    toks.push(vec![]);
    let t = Instant::now();
    let trie = Arc::new(TokenTrie::from_token_bytes(&toks, Some(eos)));
    println!("trie: {} tokens, {} nodes, built in {:?}", trie.vocab_size(), trie.node_count(), t.elapsed());

    let mut c = Constraint::new(gram, trie.clone()).unwrap();
    let mut out = Vec::new();
    let mut rng = 12345u64;
    let (mut total, mut steps, mut worst) = (std::time::Duration::ZERO, 0u32, std::time::Duration::ZERO);
    while !c.is_finished() && steps < 400 {
        let t = Instant::now();
        let m = c.mask();
        let d = t.elapsed();
        let allowed: Vec<u32> = m.iter().collect();
        total += d;
        worst = worst.max(d);
        steps += 1;
        if allowed.is_empty() { println!("dead end"); break; }
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // prefer ending when allowed
        let pick = if allowed.contains(&eos) { eos } else { allowed[(rng >> 33) as usize % allowed.len()] };
        if steps <= 3 { println!("step {steps}: {} of {} tokens allowed, mask in {:?}", allowed.len(), trie.vocab_size(), d); }
        assert!(c.accept(pick).unwrap());
        out.extend_from_slice(trie.token_bytes(pick));
    }
    println!("{} steps, mean mask {:?}, worst {:?}", steps, total / steps.max(1), worst);
    println!("output: {}", String::from_utf8_lossy(&out));
    match serde_json::from_slice::<serde_json::Value>(&out) { Ok(_) => println!("valid JSON"), Err(e) => println!("invalid JSON: {e}") }
}
