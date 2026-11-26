//! A small client for an OpenAI-style `/v1/embeddings` endpoint (corbul `serve --encoder`, llama.cpp
//! `--embeddings`, Ollama's compatible route, ...). It sends text only to the address you give.

use crate::http::{self, CancelToken, RequestOptions};
use serde_json::{Value, json};
use std::time::Duration;

/// Passages sent in one request.
pub const BATCH: usize = 16;
/// The most characters of one passage that are sent.
const MAX_CHARS: usize = 6000;

/// The longest vector accepted (the largest embedding models stay below 16,384 numbers).
const MAX_DIM: usize = 32_768;

/// True when `base` points at this computer or a private network, where plain `http://` does not
/// cross the open internet. A key is never sent anywhere else.
fn is_private_host(base: &str) -> bool {
    let rest = base.split("://").nth(1).unwrap_or(base);
    let auth = rest.split(['/', '?', '#']).next().unwrap_or("");
    let auth = auth.rsplit('@').next().unwrap_or(auth);
    let host = if let Some(v6) = auth.strip_prefix('[') {
        v6.split(']').next().unwrap_or("")
    } else {
        auth.split(':').next().unwrap_or("")
    }
    .to_ascii_lowercase();
    if host.is_empty() {
        return false;
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return match ip {
            std::net::IpAddr::V4(v) => v.is_loopback() || v.is_private() || v.is_link_local(),
            std::net::IpAddr::V6(v) => {
                v.is_loopback()
                    || (v.segments()[0] & 0xfe00) == 0xfc00
                    || (v.segments()[0] & 0xffc0) == 0xfe80
            }
        };
    }
    host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".lan")
        || !host.contains('.')
}

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
    if bearer.is_some_and(|k| !k.trim().is_empty()) && !is_private_host(base) {
        return Err(
            "refusing to send the key in clear text (http://) to a computer outside your network"
                .into(),
        );
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
        let at = match item.get("index") {
            None => pos,
            Some(i) => i
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .ok_or("an `index` is not a position")?,
        };
        let emb = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or("an item has no `embedding`")?;
        let vec: Vec<f32> = emb
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
            .collect();
        if vec.len() > MAX_DIM {
            return Err("an embedding is far larger than any real model produces".into());
        }
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

    #[test]
    fn a_key_is_only_sent_inside_your_own_network() {
        for ok in [
            "http://127.0.0.1:8080",
            "localhost:8080",
            "http://[::1]:80/x",
            "http://192.168.1.5:9",
            "http://10.0.0.2",
            "http://172.16.4.4",
            "http://nas:8080",
            "http://box.local/v1",
            "http://user:pw@127.0.0.1:1",
        ] {
            assert!(is_private_host(ok), "{ok}");
        }
        for bad in [
            "http://example.com",
            "http://8.8.8.8:80",
            "http://172.32.0.1",
            "http://127.0.0.1.evil.com",
            "http://localhost.evil.com",
            "http://evil.com/@127.0.0.1",
            "http://[2001:db8::1]:80",
            "",
            "http://",
        ] {
            assert!(!is_private_host(bad), "{bad}");
        }
        let r = embed_batch(
            "http://example.com",
            "m",
            Some("secret"),
            &["x".into()],
            None,
        );
        assert!(
            r.unwrap_err().contains("refusing"),
            "the key must not leave"
        );
    }

    #[test]
    fn hostile_answers_are_refused_not_trusted() {
        let big = format!(
            r#"{{"data":[{{"index":0,"embedding":[{}]}}]}}"#,
            vec!["0.5"; MAX_DIM + 1].join(",")
        );
        let big: &'static str = Box::leak(big.into_boxed_str());
        let reply = |body: &str| -> &'static str {
            Box::leak(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .into_boxed_str(),
            )
        };
        for body in [
            big,
            r#"{"data":[{"index":5,"embedding":[1.0]}]}"#,
            r#"{"data":[{"index":0,"embedding":[1e999]}]}"#,
            r#"{"data":[{"index":0,"embedding":["a"]}]}"#,
            r#"{"data":[{"index":-1,"embedding":[1.0]}]}"#,
            r#"{"data":[{"index":0,"embedding":null}]}"#,
            r#"{"data":[{"index":0,"embedding":[]}]}"#,
            r#"{"data":"x"}"#,
            r#"{"data":[]}"#,
            r#"[1,2,3]"#,
            "",
            "<html>",
            r#"{"data":[{"index":18446744073709551615,"embedding":[1.0]}]}"#,
        ] {
            let url = serve_once(reply(body));
            let r = embed_batch(&url, "m", None, &["x".into()], None);
            assert!(r.is_err(), "accepted: {}", &body[..body.len().min(60)]);
        }
    }

    fn serve_once(reply: &'static str) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = l.accept() {
                // read the whole request first: closing with unread data makes Windows reset the
                // connection and drop the reply
                let _ = s.set_read_timeout(Some(std::time::Duration::from_millis(500)));
                let mut got = Vec::new();
                let mut buf = [0u8; 8192];
                while let Ok(n) = s.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    got.extend_from_slice(&buf[..n]);
                    if let Some(h) = got.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&got[..h]).to_lowercase();
                        let len = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if got.len() >= h + 4 + len {
                            break;
                        }
                    }
                }
                let _ = s.write_all(reply.as_bytes());
                let _ = s.flush();
                let _ = s.shutdown(std::net::Shutdown::Write);
                while s.read(&mut buf).is_ok_and(|n| n > 0) {}
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
