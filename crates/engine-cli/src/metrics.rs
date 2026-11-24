//! Prometheus text-format metrics (`GET /metrics`): requests, tokens, latency histograms, queue depth,
//! generation speed and memory. No dependencies; everything is atomics or one small mutex.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

const BUCKETS: [f64; 13] = [
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

#[derive(Default)]
pub struct Histogram {
    counts: [AtomicU64; 13],
    count: AtomicU64,
    sum_us: AtomicU64,
}
impl Histogram {
    pub fn observe(&self, secs: f64) {
        let s = if secs.is_finite() { secs.max(0.0) } else { 0.0 };
        for (i, b) in BUCKETS.iter().enumerate() {
            if s <= *b {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_us.fetch_add((s * 1e6) as u64, Ordering::Relaxed);
    }
    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut cum = 0;
        for (i, b) in BUCKETS.iter().enumerate() {
            cum += self.counts[i].load(Ordering::Relaxed);
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {cum}");
        }
        let n = self.count.load(Ordering::Relaxed);
        let _ = writeln!(
            out,
            "{name}_bucket{{le=\"+Inf\"}} {n}\n{name}_sum {}\n{name}_count {n}",
            self.sum_us.load(Ordering::Relaxed) as f64 / 1e6
        );
    }
}

pub struct Metrics {
    started: Instant,
    requests: Mutex<BTreeMap<(String, u16), u64>>,
    tokens: Mutex<BTreeMap<String, (u64, u64)>>,
    pub in_flight: AtomicI64,
    pub queued: AtomicI64,
    pub latency: Histogram,
    pub ttft: Histogram,
    pub queue_wait: Histogram,
    last_tps: AtomicU64,
    gen_tokens: AtomicU64,
    gen_micros: AtomicU64,
}
impl Default for Metrics {
    fn default() -> Self {
        Metrics::new()
    }
}

/// Label values: backslash, quote and newline are escaped.
pub fn esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

impl Metrics {
    pub fn new() -> Metrics {
        Metrics {
            started: Instant::now(),
            requests: Mutex::default(),
            tokens: Mutex::default(),
            in_flight: AtomicI64::new(0),
            queued: AtomicI64::new(0),
            latency: Histogram::default(),
            ttft: Histogram::default(),
            queue_wait: Histogram::default(),
            last_tps: AtomicU64::new(0),
            gen_tokens: AtomicU64::new(0),
            gen_micros: AtomicU64::new(0),
        }
    }
    pub fn request(&self, endpoint: &str, status: u16) {
        let mut m = self.requests.lock().unwrap_or_else(|e| e.into_inner());
        if m.len() < 512 || m.contains_key(&(endpoint.to_string(), status)) {
            *m.entry((endpoint.to_string(), status)).or_insert(0) += 1;
        }
    }
    pub fn tokens(&self, model: &str, prompt: usize, completion: usize) {
        let mut m = self.tokens.lock().unwrap_or_else(|e| e.into_inner());
        if m.len() < 256 || m.contains_key(model) {
            let e = m.entry(model.to_string()).or_insert((0, 0));
            e.0 += prompt as u64;
            e.1 += completion as u64;
        }
    }
    /// Record one finished generation (its speed feeds the gauge and the running average).
    pub fn generated(&self, completion: usize, secs: f64) {
        if completion > 0 && secs > 0.0 {
            self.last_tps
                .store((completion as f64 / secs).to_bits(), Ordering::Relaxed);
            self.gen_tokens
                .fetch_add(completion as u64, Ordering::Relaxed);
            self.gen_micros
                .fetch_add((secs * 1e6) as u64, Ordering::Relaxed);
        }
    }
    pub fn request_total(&self) -> u64 {
        self.requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .sum()
    }

    /// `extra`: gauges the caller knows (name, help, value, optional label set).
    pub fn render(&self, extra: &[(&str, &str, f64, String)]) -> String {
        let mut o = String::new();
        let _ = writeln!(
            o,
            "# HELP corbul_requests_total Requests served, by endpoint and status.\n# TYPE corbul_requests_total counter"
        );
        for ((e, s), n) in self
            .requests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            let _ = writeln!(
                o,
                "corbul_requests_total{{endpoint=\"{}\",status=\"{s}\"}} {n}",
                esc(e)
            );
        }
        let _ = writeln!(
            o,
            "# HELP corbul_prompt_tokens_total Prompt tokens processed.\n# TYPE corbul_prompt_tokens_total counter"
        );
        let t = self.tokens.lock().unwrap_or_else(|e| e.into_inner());
        for (m, (p, _)) in t.iter() {
            let _ = writeln!(o, "corbul_prompt_tokens_total{{model=\"{}\"}} {p}", esc(m));
        }
        let _ = writeln!(
            o,
            "# HELP corbul_completion_tokens_total Tokens generated.\n# TYPE corbul_completion_tokens_total counter"
        );
        for (m, (_, c)) in t.iter() {
            let _ = writeln!(
                o,
                "corbul_completion_tokens_total{{model=\"{}\"}} {c}",
                esc(m)
            );
        }
        drop(t);
        let _ = writeln!(
            o,
            "# HELP corbul_requests_in_flight Requests being handled.\n# TYPE corbul_requests_in_flight gauge\ncorbul_requests_in_flight {}",
            self.in_flight.load(Ordering::Relaxed).max(0)
        );
        let _ = writeln!(
            o,
            "# HELP corbul_queue_depth Requests waiting for a model.\n# TYPE corbul_queue_depth gauge\ncorbul_queue_depth {}",
            self.queued.load(Ordering::Relaxed).max(0)
        );
        let _ = writeln!(
            o,
            "# HELP corbul_tokens_per_second Speed of the last generation.\n# TYPE corbul_tokens_per_second gauge\ncorbul_tokens_per_second {}",
            f64::from_bits(self.last_tps.load(Ordering::Relaxed))
        );
        let (gt, gm) = (
            self.gen_tokens.load(Ordering::Relaxed),
            self.gen_micros.load(Ordering::Relaxed),
        );
        let _ = writeln!(
            o,
            "# HELP corbul_tokens_per_second_avg Average generation speed since start.\n# TYPE corbul_tokens_per_second_avg gauge\ncorbul_tokens_per_second_avg {}",
            if gm > 0 {
                gt as f64 / (gm as f64 / 1e6)
            } else {
                0.0
            }
        );
        let _ = writeln!(
            o,
            "# HELP corbul_uptime_seconds Time since the server started.\n# TYPE corbul_uptime_seconds gauge\ncorbul_uptime_seconds {}",
            self.started.elapsed().as_secs_f64()
        );
        if let Some(rss) = process_rss() {
            let _ = writeln!(
                o,
                "# HELP corbul_process_resident_memory_bytes Resident memory of the process.\n# TYPE corbul_process_resident_memory_bytes gauge\ncorbul_process_resident_memory_bytes {rss}"
            );
        }
        self.latency.render(
            &mut o,
            "corbul_request_duration_seconds",
            "Time from request to last byte.",
        );
        self.ttft.render(
            &mut o,
            "corbul_time_to_first_token_seconds",
            "Time from request to the first generated token.",
        );
        self.queue_wait.render(
            &mut o,
            "corbul_queue_wait_seconds",
            "Time spent waiting for the model.",
        );
        let mut seen: Vec<&str> = vec![];
        for (name, help, value, labels) in extra {
            if !seen.contains(name) {
                let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge");
                seen.push(name);
            }
            if labels.is_empty() {
                let _ = writeln!(o, "{name} {value}");
            } else {
                let _ = writeln!(o, "{name}{{{labels}}} {value}");
            }
        }
        o
    }
}

/// Resident set size in bytes (Linux; `None` elsewhere).
pub fn process_rss() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/statm").ok()?;
    s.split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()
        .map(|p| p * 4096)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_valid_exposition() {
        let m = Metrics::new();
        m.request("/v1/chat/completions", 200);
        m.request("/v1/chat/completions", 200);
        m.request("/v1/chat/completions", 429);
        m.tokens("a\"b", 10, 5);
        m.generated(50, 2.0);
        m.latency.observe(0.3);
        m.latency.observe(100.0);
        m.latency.observe(f64::NAN);
        let t = m.render(&[
            ("corbul_models_loaded", "Loaded models.", 2.0, String::new()),
            ("corbul_model_bytes", "Weights.", 5.0, "model=\"x\"".into()),
            ("corbul_model_bytes", "Weights.", 6.0, "model=\"y\"".into()),
        ]);
        assert!(
            t.contains("corbul_requests_total{endpoint=\"/v1/chat/completions\",status=\"200\"} 2")
        );
        assert!(
            t.contains("corbul_requests_total{endpoint=\"/v1/chat/completions\",status=\"429\"} 1")
        );
        assert!(t.contains("corbul_prompt_tokens_total{model=\"a\\\"b\"} 10"));
        assert!(t.contains("corbul_completion_tokens_total{model=\"a\\\"b\"} 5"));
        assert!(t.contains("corbul_tokens_per_second 25"));
        assert!(t.contains("corbul_request_duration_seconds_bucket{le=\"0.5\"} 2")); // 0.3 and the NaN (as 0)
        assert!(t.contains("corbul_request_duration_seconds_bucket{le=\"+Inf\"} 3"));
        assert!(t.contains("corbul_request_duration_seconds_count 3"));
        assert_eq!(t.matches("# TYPE corbul_model_bytes").count(), 1);
        assert!(t.contains("corbul_model_bytes{model=\"y\"} 6"));
        // every sample line is `name[{labels}] value`
        for l in t.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
            let (name, v) = l.rsplit_once(' ').unwrap();
            assert!(v.parse::<f64>().is_ok(), "{l}");
            assert!(name.chars().next().unwrap().is_ascii_alphabetic());
        }
        assert_eq!(m.request_total(), 3);
    }

    #[test]
    fn histogram_buckets_are_cumulative() {
        let h = Histogram::default();
        for s in [0.005, 0.02, 0.02, 7.0] {
            h.observe(s);
        }
        let mut o = String::new();
        h.render(&mut o, "x", "h");
        assert!(
            o.contains("x_bucket{le=\"0.01\"} 1")
                && o.contains("x_bucket{le=\"0.05\"} 3")
                && o.contains("x_bucket{le=\"10\"} 4")
                && o.contains("x_count 4")
        );
    }

    #[test]
    fn label_cardinality_is_bounded() {
        let m = Metrics::new();
        for i in 0..2000 {
            m.request(&format!("/x{i}"), 404);
        }
        assert!(m.render(&[]).matches("corbul_requests_total{").count() <= 512);
    }
}
