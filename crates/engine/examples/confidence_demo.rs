//! ANSI heat-map of token confidence, risk report and self-consistency on synthetic data.
//! Run: cargo run -p companion-engine --example confidence_demo
use companion_engine::confidence::*;

fn main() {
    let ctx = "The Eiffel Tower in Paris was completed in 1889 by Gustave Eiffel.";
    let text = "The Eiffel Tower was completed in 1887 by Zorblax Quimby.";
    let unsure = ["1887", "Zorblax", "Quimby."];
    let (mut spans, mut stats, mut at) = (vec![], vec![], 0usize);
    for w in text.split_inclusive(' ') {
        spans.push(TokenSpan {
            start: at,
            end: at + w.len(),
        });
        let logits: Vec<f32> = if unsure.contains(&w.trim()) {
            vec![1.0, 0.8, 0.7, 0.6, 0.0]
        } else {
            vec![9.0, 0.0, 0.0, 0.0, 0.0]
        };
        stats.push(step_stats(&logits, Some(0)).unwrap());
        at += w.len();
    }
    let tr = UncertaintyTrace::build(text, &spans, &stats).unwrap();
    for t in &tr.tokens {
        let g = (255.0 * (1.0 - t.heat)) as u8; // green fades to red
        let r = (255.0 * t.heat) as u8;
        print!(
            "\x1b[48;2;{r};{g};60m\x1b[30m{}\x1b[0m",
            &text[t.start..t.end]
        );
    }
    println!("\n");
    let s = &tr.summary;
    println!(
        "min {:.2}  mean {:.2}  p10 {:.2}  weakest sentence: {:?}",
        s.min_conf, s.mean_conf, s.p10_conf, s.weakest_sentence
    );
    let risk = assess(&tr, Some(ctx), &RiskOptions::default());
    println!("heuristic risk {:.2} (not ground truth)", risk.score);
    for r in &risk.reasons {
        println!("  - [{}] {} (weight {:.2})", r.code, r.message, r.weight);
    }
    let c = consistency(&[
        "The answer is 1889.",
        "answer: 1889",
        "I think 1887",
        "The answer is 1,889",
    ])
    .unwrap();
    let m = c.majority.unwrap();
    println!(
        "\nself-consistency: {} with {}/{} votes, agreement {:.2}",
        m.answer,
        m.votes,
        m.k,
        c.agreement.unwrap()
    );
    let scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
    let labels = [true, true, false, true, false, false];
    println!(
        "ECE {:.3}  Brier {:.3}  AUROC {:.3}",
        ece(&scores, &labels, 5).unwrap(),
        brier(&scores, &labels).unwrap(),
        auroc(&scores, &labels).unwrap()
    );
}
