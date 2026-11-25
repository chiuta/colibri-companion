//! A small client for an OpenAI-style `/v1/embeddings` endpoint (corbul `serve --encoder`, llama.cpp
//! `--embeddings`, Ollama's compatible route, ...). It sends text only to the address you give.

use crate::http::{self, CancelToken, RequestOptions};
use serde_json::{Value, json};
use std::time::Duration;

/// Passages sent in one request.
pub const BATCH: usize = 16;
/// The most characters of one passage that are sent.
const MAX_CHARS: usize = 6000;

/// Embeds `inputs` in order. Every vector must be finite and all must have the same length.
pub fn embed_batch(
    base: &str,
    model: &str,
    bearer: Option<&str>,
    inputs: &[String],
    cancel: Option<&CancelToken>,
) -> Result<Vec<Vec<f32>>, String> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    let clipped: Vec<String> = inputs
        .iter()
        .map(|t| t.chars().take(MAX_CHARS).collect())
        .collect();
    let body = json!({"model": model, "input": clipped}).to_string();
    let opts = RequestOptions {
        connect_timeout: Duration::from_secs(3),
        read_timeout: Some(Duration::from_secs(120)),
        bearer,
        cancel,
    };
    let resp = http::send(base, "POST", "/v1/embeddings", Some(&body), &opts)
        .map_err(|e| format!("could not reach the embeddings server: {e}"))?;
    let status = resp.status;
    let text = resp.text();
    if status != 200 {
        let short: String = text.chars().take(200).collect();
        return Err(format!("the embeddings server answered {status}: {short}"));
    }
    let v: Value = serde_json::from_str(&text).map_err(|_| "the answer is not JSON".to_string())?;
    let data = v
        .get("data")
        .and_then(Value::as_array)
        .ok_or("the answer has no `data` list")?;
    if data.len() != inputs.len() {
        return Err(format!(
            "asked for {} embeddings and got {}",
            inputs.len(),
            data.len()
        ));
    }
    // `index` says where each one belongs; fall back to the order given
    let mut out: Vec<Option<Vec<f32>>> = vec![None; inputs.len()];
    for (pos, item) in data.iter().enumerate() {
        let at = item
            .get("index")
            .and_then(Value::as_u64)
            .map_or(pos, |i| i as usize);
        let emb = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or("an item has no `embedding`")?;
        let vec: Vec<f32> = emb
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
            .collect();
        if vec.is_empty() || vec.iter().any(|x| !x.is_finite()) {
            return Err("an embedding is empty or not a finite number".into());
        }
        *out.get_mut(at).ok_or("an `index` is out of range")? = Some(vec);
    }
    let out: Vec<Vec<f32>> = out
        .into_iter()
        .collect::<Option<_>>()
        .ok_or("an embedding is missing")?;
    let dim = out[0].len();
    if out.iter().any(|v| v.len() != dim) {
        return Err("the embeddings have different sizes".into());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    fn serve_once(reply: &'static str) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = l.accept() {
                let mut buf = [0u8; 8192];
                let _ = s.read(&mut buf);
                let _ = s.write_all(reply.as_bytes());
            }
        });
        format!("http://{addr}")
    }

    #[test]
    fn reads_embeddings_in_index_order() {
        let r = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"data\":[{\"index\":1,\"embedding\":[0.0,1.0]},{\"index\":0,\"embedding\":[1.0,0.0]}]}";
        let base = serve_once(r);
        let v = embed_batch(&base, "m", None, &["a".into(), "b".into()], None).unwrap();
        assert_eq!(v, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    }

    #[test]
    fn errors_are_plain_words() {
        let base = serve_once("HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\nno");
        let e = embed_batch(&base, "m", None, &["a".into()], None).unwrap_err();
        assert!(e.contains("404"), "{e}");
        let base = serve_once("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"data\":[]}");
        assert!(embed_batch(&base, "m", None, &["a".into()], None).is_err());
        assert!(embed_batch("http://127.0.0.1:1", "m", None, &["a".into()], None).is_err());
        assert_eq!(
            embed_batch("http://127.0.0.1:1", "m", None, &[], None)
                .unwrap()
                .len(),
            0
        );
    }
}
