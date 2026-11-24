//! The web interface served by `corbul serve` at `/` and `/ui`.
//!
//! One self-contained HTML page (`webui/index.html`, no external files, no network requests except to
//! the same server) in eight languages: chat with streaming, model and sampling panel, sources with
//! quote-lock citations, receipt download, and a hardware/plan panel. The page itself is public so a
//! browser can open it and ask for the token; every API call it makes still needs the token. It is
//! served with a strict Content-Security-Policy: scripts and styles only from the page itself, requests
//! only to the same origin, no framing.
pub const INDEX: &str = include_str!("../webui/index.html");

pub const CSP: &str = "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

/// Security headers for the page (the HTTP layer adds `Cache-Control: no-store` and `nosniff`).
pub const HEADERS: &[(&str, &str)] = &[
    ("Content-Security-Policy", CSP),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "no-referrer"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    (
        "Permissions-Policy",
        "camera=(), microphone=(), geolocation=()",
    ),
];

/// Paths that serve the page.
pub fn is_page(path: &str) -> bool {
    path == "/" || path == "/ui" || path == "/ui/"
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn the_page_is_self_contained() {
        assert!(INDEX.len() > 1000 && INDEX.contains("<html"));
        // no external scripts, styles or fonts
        for bad in [
            "<script src",
            "<link rel=\"stylesheet\"",
            "@import",
            "src=\"http",
        ] {
            assert!(!INDEX.contains(bad), "{bad}");
        }
        assert!(is_page("/") && is_page("/ui") && !is_page("/v1/models"));
    }
}
