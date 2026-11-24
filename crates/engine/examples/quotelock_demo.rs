//! Quote-locked decoding demo: build an index, drive the lock with a toy
//! byte vocabulary, verify text, and time the 10 MB / 150k-vocab limits.
use companion_engine::quotelock::{Options, QuoteIndex, QuoteLock, QuoteVocab};
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let s1 = "The committee found that rainfall in the northern basin fell by nearly a third over the decade. Researchers cautioned against drawing firm conclusions.";
    let s2 = "Șoarecele și pisica trăiesc în „casa” veche — ţara noastră e frumoasă.";
    let idx = Arc::new(QuoteIndex::new(&[s1, s2], Options::default()).unwrap());

    // toy vocabulary: every byte, a few words, EOS
    let mut toks: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    toks[0] = vec![];
    for w in [
        " the",
        " rainfall",
        " northern",
        " basin",
        " over",
        " decade",
    ] {
        toks.push(w.as_bytes().to_vec());
    }
    let vocab = Arc::new(QuoteVocab::from_token_bytes(&toks));
    let mut lock = QuoteLock::new(idx.clone(), vocab.clone());
    for b in "As reported, [S1]«rainfall in the northern".bytes() {
        lock.accept(b as u32).unwrap();
    }
    println!(
        "inside quote: {}, allowed next tokens: {}",
        lock.is_inside_quote(),
        lock.mask().count()
    );
    println!(
        "'x' allowed: {}, ' ' allowed: {}",
        lock.mask().get(b'x' as usize),
        lock.mask().get(b' ' as usize)
    );

    let text = "Result: [S1]«fell by nearly a third» and «fell by nearly half» and «rainfall in the southern basin».";
    let rep = idx.verify(text);
    for q in &rep.quotes {
        println!(
            "{:<28} {:?} sim={:.2} support={:.2} cite={:?}",
            q.text,
            q.status,
            q.similarity,
            q.support,
            q.citation.as_ref().map(|c| (c.source_id, c.start, c.end))
        );
    }

    // performance: ~10 MB of text, 150k-token vocabulary
    let mut big = String::new();
    let mut x = 12345u64;
    while big.len() < 10_000_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        for _ in 0..(2 + x % 8) {
            big.push((b'a' + ((x >> 8) % 26) as u8) as char);
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
        }
        big.push(' ');
    }
    let t = Instant::now();
    let bigidx = Arc::new(QuoteIndex::new(&[&big], Options::default()).unwrap());
    println!("index 10 MB: {:?}", t.elapsed());
    let mut v: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    v[0] = vec![];
    while v.len() < 150_000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let n = 2 + (x % 7) as usize;
        let mut w = vec![b' '];
        for k in 0..n {
            w.push(b'a' + ((x >> (8 + k)) % 26) as u8);
        }
        v.push(w);
    }
    let vv = Arc::new(QuoteVocab::from_token_bytes(&v));
    let mut l = QuoteLock::new(bigidx, vv);
    let t = Instant::now();
    for b in "«".bytes() {
        l.accept(b as u32).unwrap();
    }
    println!(
        "first mask (full interval): {:?}, allowed {}",
        t.elapsed(),
        l.mask().count()
    );
    for b in big[5000..5040].bytes().take(12) {
        let t = Instant::now();
        l.accept(b as u32).unwrap();
        println!(
            "  byte {:?}: mask in {:?}, allowed {}",
            b as char,
            t.elapsed(),
            l.mask().count()
        );
    }
}
