//! The token loop behind every endpoint. It is our own (not `Model::generate`) because the server needs what
//! that does not expose: the logits, for `logprobs`, `logit_bias`, penalties and grammar constraints.
use crate::registry::Slot;
use crate::tools::{self, Delta, ToolStream};
use companion_engine::model::{ModelError, Sampler, State};
use serde_json::{Value, json};
use std::time::Instant;

/// Restricts what the model may write next (structured output). `apply` masks the logits for the next token;
/// `accept` is told which token was chosen.
pub trait Constraint {
    fn apply(&mut self, logits: &mut [f32]);
    fn accept(&mut self, token: usize);
}

/// Holds back text that might be the start of a stop sequence, and cuts at the first full match.
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
    /// The sequence that ended the text, once one did.
    pub hit: Option<String>,
}
impl StopFilter {
    pub fn new(stops: &[String]) -> StopFilter {
        StopFilter {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            held: String::new(),
            hit: None,
        }
    }
    /// Returns (text that is safe to send, whether a stop sequence was found).
    pub fn push(&mut self, s: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (s.to_string(), false);
        }
        self.held.push_str(s);
        // earliest match; on a tie the longest sequence
        let best = self
            .stops
            .iter()
            .filter_map(|st| self.held.find(st.as_str()).map(|i| (i, st)))
            .min_by(|a, b| a.0.cmp(&b.0).then(b.1.len().cmp(&a.1.len())));
        if let Some((i, st)) = best {
            let out = self.held[..i].to_string();
            self.hit = Some(st.clone());
            self.held.clear();
            return (out, true);
        }
        let mut keep = 0;
        for st in &self.stops {
            let max = (st.len() - 1).min(self.held.len());
            for k in (1..=max).rev() {
                if st.is_char_boundary(k)
                    && self.held.is_char_boundary(self.held.len() - k)
                    && self.held.ends_with(&st[..k])
                {
                    keep = keep.max(k);
                    break;
                }
            }
        }
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held.drain(..cut);
        (out, false)
    }
    /// What is still held back when generation ends without a match.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub rep: f32,
    pub presence: f32,
    pub frequency: f32,
    pub stops: Vec<String>,
    pub logit_bias: Vec<(usize, f32)>,
    /// `Some(n)`: report the log-probability of every chosen token and its `n` best alternatives.
    pub logprobs: Option<usize>,
}

/// Where the text, tool calls and log-probabilities go as they are produced.
pub trait Sink {
    /// Called once, after the queue and validation, right before the first token. `false` = give up.
    fn start(&mut self) -> bool;
    fn text(&mut self, choice: usize, piece: &str, logprobs: &[Value]) -> bool;
    fn call(&mut self, choice: usize, index: usize, id: &str, name: &str, arguments: &str) -> bool;
}

#[derive(Debug, Default, Clone)]
pub struct ChoiceOut {
    pub text: String,
    pub calls: Vec<(String, tools::ToolCall)>,
    /// OpenAI names: stop, length, tool_calls, cancelled.
    pub finish: &'static str,
    pub stop_sequence: Option<String>,
    pub tokens: usize,
    pub logprobs: Vec<Value>,
    pub ttft: f64,
}

fn log_softmax_entry(
    tok: &companion_engine::tokenizer::Tokenizer,
    logits: &[f32],
    chosen: usize,
    top: usize,
) -> Value {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f32>().ln();
    let describe = |id: usize| -> (String, Vec<u8>) {
        let s = tok.decode(&[id as u32]);
        let b = s.as_bytes().to_vec();
        (s, b)
    };
    let mk = |id: usize| {
        let (t, b) = describe(id);
        json!({"token": t, "logprob": logits.get(id).map_or(f32::NEG_INFINITY, |l| l - lse), "bytes": b})
    };
    let mut e = mk(chosen);
    let mut alts = vec![];
    if top > 0 {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        let k = top.min(20).min(idx.len());
        if k > 0 {
            idx.select_nth_unstable_by(k - 1, |a, b| logits[*b].total_cmp(&logits[*a]));
            idx.truncate(k);
            idx.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
            alts = idx.into_iter().map(mk).collect();
        }
    }
    e["top_logprobs"] = Value::Array(alts);
    e
}

/// Generate one reply. `st` must already hold the prompt (`logits` = what the model said after it).
#[allow(clippy::too_many_arguments)]
pub fn run_choice(
    slot: &Slot,
    st: &mut State,
    mut logits: Vec<f32>,
    prompt: &[usize],
    p: &GenParams,
    tool_names: Option<Vec<String>>,
    mut cons: Option<Box<dyn Constraint>>,
    alive: &dyn Fn() -> bool,
    sink: &mut dyn Sink,
    choice: usize,
    id_seed: u64,
    t_req: Instant,
) -> Result<ChoiceOut, ModelError> {
    let (model, tok) = (&slot.loaded.model, &slot.loaded.tok);
    let ctx = model.spec.context as usize;
    let mut sampler = Sampler::new(
        p.temperature,
        p.top_k,
        p.seed.wrapping_add(choice as u64 * 0x9e37_79b9),
    )
    .with_top_p(p.top_p)
    .with_repetition_penalty(p.rep, 64);
    sampler.observe_all(prompt);
    let mut dec = tok.decoder();
    // tool-call markers such as `<|python_tag|>` are special tokens
    dec.skip_special = tool_names.is_none();
    let mut filt = StopFilter::new(&p.stops);
    let mut ts = tool_names.map(ToolStream::new);
    let mut out = ChoiceOut::default();
    let (mut all, mut pending_lp) = (String::new(), Vec::<Value>::new());
    let mut counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let penalised = p.presence != 0.0 || p.frequency != 0.0;
    let (mut stopped, mut gone, mut hit_stop_token) = (false, false, false);
    let mut length = false;

    // Pass visible text on; with tools, through the tool-call scanner.
    let emit = |sink: &mut dyn Sink,
                out: &mut ChoiceOut,
                ts: &mut Option<ToolStream>,
                all: &mut String,
                add: &str,
                fin: bool,
                pending: &mut Vec<Value>|
     -> bool {
        all.push_str(add);
        match ts {
            None => {
                if (add.is_empty() && pending.is_empty()) || (add.is_empty() && !fin) {
                    return true;
                }
                out.text.push_str(add);
                let ok = sink.text(choice, add, pending);
                pending.clear();
                ok
            }
            Some(t) => {
                let (deltas, _) = t.push(all, fin);
                for d in deltas {
                    match d {
                        Delta::Text(x) => {
                            out.text.push_str(&x);
                            if !sink.text(choice, &x, pending) {
                                return false;
                            }
                            pending.clear();
                        }
                        Delta::Call(i, c) => {
                            let id = tools::call_id(id_seed, choice * 64 + i);
                            let ok = sink.call(choice, i, &id, &c.name, &c.arguments);
                            out.calls.push((id, c));
                            if !ok {
                                return false;
                            }
                        }
                    }
                }
                true
            }
        }
    };

    loop {
        if out.tokens >= p.max_tokens {
            length = true;
            break;
        }
        if !alive() {
            gone = true;
            break;
        }
        let raw = p.logprobs.map(|_| logits.clone());
        for (id, b) in &p.logit_bias {
            if let Some(l) = logits.get_mut(*id) {
                *l += *b;
            }
        }
        if penalised {
            for (id, c) in &counts {
                if let Some(l) = logits.get_mut(*id) {
                    *l -= p.frequency * *c as f32 + p.presence;
                }
            }
        }
        if let Some(c) = cons.as_mut() {
            c.apply(&mut logits);
        }
        let t = sampler.sample(&logits);
        sampler.observe(t);
        if slot.loaded.stop_ids.contains(&t) {
            hit_stop_token = true;
            break;
        }
        if out.tokens == 0 {
            out.ttft = t_req.elapsed().as_secs_f64();
        }
        out.tokens += 1;
        if penalised {
            *counts.entry(t).or_insert(0) += 1;
        }
        if let Some(c) = cons.as_mut() {
            c.accept(t);
        }
        if let (Some(n), Some(raw)) = (p.logprobs, raw.as_ref()) {
            let e = log_softmax_entry(tok, raw, t, n);
            out.logprobs.push(e.clone());
            pending_lp.push(e);
        }
        let piece = dec.push(t as u32);
        if !piece.is_empty() {
            let (vis, hit) = filt.push(&piece);
            if !emit(
                sink,
                &mut out,
                &mut ts,
                &mut all,
                &vis,
                false,
                &mut pending_lp,
            ) {
                gone = true;
                break;
            }
            if hit {
                stopped = true;
                break;
            }
        }
        if st.pos >= ctx {
            length = true;
            break;
        }
        logits = model.forward(st, t)?;
    }
    if !stopped && !gone {
        let tail = dec.finish();
        let (mut vis, hit) = filt.push(&tail);
        if hit {
            stopped = true;
        } else {
            vis.push_str(&filt.flush());
        }
        if !emit(
            sink,
            &mut out,
            &mut ts,
            &mut all,
            &vis,
            false,
            &mut pending_lp,
        ) {
            gone = true;
        }
    }
    if !gone {
        // final scan: an unclosed call at the end of generation counts; leftovers flush as text
        if !emit(sink, &mut out, &mut ts, &mut all, "", true, &mut pending_lp) {
            gone = true;
        }
    }
    out.stop_sequence = filt.hit.clone();
    out.finish = if gone {
        "cancelled"
    } else if !out.calls.is_empty() {
        "tool_calls"
    } else if stopped || hit_stop_token {
        "stop"
    } else if length {
        "length"
    } else {
        "stop"
    };
    Ok(out)
}

/// OpenAI's legacy `/v1/completions` log-probability block from chat-style entries.
pub fn legacy_logprobs(entries: &[Value], text_len_hint: usize) -> Value {
    let mut offset = 0usize;
    let (mut tokens, mut lps, mut tops, mut offs) = (vec![], vec![], vec![], vec![]);
    for e in entries {
        let t = e["token"].as_str().unwrap_or("").to_string();
        offs.push(offset.min(text_len_hint.max(offset)));
        offset += t.chars().count();
        lps.push(e["logprob"].clone());
        let mut m = serde_json::Map::new();
        for a in e["top_logprobs"].as_array().into_iter().flatten() {
            m.insert(
                a["token"].as_str().unwrap_or("").to_string(),
                a["logprob"].clone(),
            );
        }
        tops.push(Value::Object(m));
        tokens.push(Value::String(t));
    }
    json!({"tokens": tokens, "token_logprobs": lps, "top_logprobs": tops, "text_offset": offs})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(f: &mut StopFilter, parts: &[&str]) -> (String, bool) {
        let mut out = String::new();
        for p in parts {
            let (o, hit) = f.push(p);
            out.push_str(&o);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&f.flush());
        (out, false)
    }

    #[test]
    fn cuts_at_the_stop_and_remembers_which() {
        let mut f = StopFilter::new(&["END".into(), "STOP".into()]);
        assert_eq!(
            feed(&mut f, &["hello E", "ND tail"]),
            ("hello ".into(), true)
        );
        assert_eq!(f.hit.as_deref(), Some("END"));
        let mut f = StopFilter::new(&["ab".into(), "abc".into()]);
        assert_eq!(feed(&mut f, &["xabc"]), ("x".into(), true));
        assert_eq!(
            f.hit.as_deref(),
            Some("abc"),
            "longest at the same position"
        );
    }

    /// Property: however the text is cut into pieces, the output equals cutting the whole text at the first stop,
    /// nothing is lost, and no emitted prefix ever contains a stop sequence.
    #[test]
    fn hold_back_is_independent_of_chunking() {
        let stops = vec![
            "</s>".to_string(),
            "\n\nUser:".to_string(),
            "é!".to_string(),
        ];
        let texts = [
            "plain text without any stop",
            "before </s> after",
            "a\n\nUser: b",
            "ends with partial </",
            "é! é é!",
            "<</s>",
            "\n\n\nUser:x",
            "aaaa\n\nUs",
        ];
        let mut seed = 0xdead_beef_u64;
        let mut rnd = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for t in texts {
            let expect_cut = stops.iter().filter_map(|s| t.find(s.as_str())).min();
            let expected = match expect_cut {
                Some(i) => t[..i].to_string(),
                None => t.to_string(),
            };
            for _ in 0..200 {
                let mut f = StopFilter::new(&stops);
                let (mut out, mut cur, mut hit) = (String::new(), 0usize, false);
                while cur < t.len() && !hit {
                    let mut end = (cur + 1 + rnd(5) as usize).min(t.len());
                    while !t.is_char_boundary(end) {
                        end += 1;
                    }
                    let (o, h) = f.push(&t[cur..end]);
                    assert!(!stops.iter().any(|s| o.contains(s.as_str())));
                    out.push_str(&o);
                    hit = h;
                    cur = end;
                }
                if !hit {
                    out.push_str(&f.flush());
                }
                assert_eq!(out, expected, "text {t:?}");
                assert_eq!(hit, expect_cut.is_some(), "text {t:?}");
            }
        }
    }

    #[test]
    fn no_stops_is_a_passthrough_and_empty_stops_are_ignored() {
        let mut f = StopFilter::new(&[String::new()]);
        assert_eq!(f.push("abc"), ("abc".into(), false));
        assert_eq!(f.flush(), "");
    }

    #[test]
    fn legacy_logprobs_shape() {
        let e = vec![
            json!({"token": "a", "logprob": -0.5, "bytes": [97], "top_logprobs": [{"token": "a", "logprob": -0.5, "bytes": [97]}]}),
            json!({"token": "bc", "logprob": -1.0, "bytes": [], "top_logprobs": []}),
        ];
        let v = legacy_logprobs(&e, 3);
        assert_eq!(v["tokens"], json!(["a", "bc"]));
        assert_eq!(v["text_offset"], json!([0, 1]));
        assert_eq!(v["top_logprobs"][0]["a"], json!(-0.5));
    }
}
