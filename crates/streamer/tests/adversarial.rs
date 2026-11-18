mod hostile_gen;
use expert_streamer::*;
use hostile_gen::{Rng, hammer, iterations};

fn header(json: &str) -> Vec<u8> {
    let mut v = (json.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(json.as_bytes());
    v
}

#[test]
fn trace_text() {
    hammer(
        "trace::parse_trace",
        &["0 1 2 3\n4 5 6 7\n", "step 1: [1,2,3]", "1,2;3,4"],
        |s| {
            for n in [0, 1, 8, 256, u32::MAX] {
                let _ = trace::parse_trace(s, n);
            }
        },
    );
}

#[test]
fn safetensors_headers_text_and_bytes() {
    let good =
        r#"{"__metadata__":{"a":"b"},"w":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
    hammer(
        "safetensors::parse_header",
        &[
            good,
            "{}",
            r#"{"w":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,1]}}"#,
        ],
        |s| {
            let b = header(s);
            for len in [0u64, 8, b.len() as u64, 1 << 40, u64::MAX] {
                let _ = safetensors::parse_header(&b, len);
            }
            let _ = safetensors::duplicate_key(s.as_bytes());
        },
    );
    let mut r = Rng::new(11);
    for _ in 0..iterations() {
        let n = r.below(600);
        let b: Vec<u8> = (0..n).map(|_| r.next_u64() as u8).collect();
        let _ = safetensors::parse_header(&b, n as u64);
        let _ = safetensors::parse_header(&b, u64::MAX);
        let mut c = (r.below(900) as u64).to_le_bytes().to_vec();
        c.extend_from_slice(&b);
        let _ = safetensors::parse_header(&c, c.len() as u64);
    }
}
