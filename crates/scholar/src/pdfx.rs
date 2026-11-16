// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! PDF text layer and table-aware passage splitting.
//!
//! Reading is pure Rust (`lopdf`), the text layer only: a scanned PDF without a text layer yields
//! empty pages (no OCR, no external programs). Table and equation reconstruction from `pdftotext
//! -layout` output was left out of this port because it needs poppler. Every limit below exists so
//! a hostile file costs bounded memory and time.

use crate::index::{Error, Result};

/// Largest PDF accepted, in bytes.
pub const MAX_PDF_BYTES: usize = 256 << 20;
/// Most pages read from one PDF.
pub const MAX_PDF_PAGES: usize = 20_000;
/// Largest decompressed content stream of one page.
pub const MAX_PAGE_STREAM: usize = 32 << 20;
/// Most text kept from one PDF (all pages together).
pub const MAX_PDF_TEXT: usize = 128 << 20;

/// Text of every page, in order (index 0 = page 1). Pages whose text cannot be extracted are empty
/// strings, so page numbers stay aligned with the printed document. Encrypted or unparseable files
/// are an error. Runs on a large-stack thread so deeply nested objects cannot overflow the caller.
pub fn text_pages(bytes: &[u8]) -> Result<Vec<String>> {
    if bytes.len() > MAX_PDF_BYTES {
        return Err(Error::Limit(format!("PDF is {} bytes", bytes.len())));
    }
    let owned = bytes.to_vec();
    let h = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || pages_inner(&owned))
        .map_err(|e| Error::Io(e.to_string()))?;
    h.join()
        .map_err(|_| Error::Format("PDF parser crashed".into()))?
}

fn pages_inner(bytes: &[u8]) -> Result<Vec<String>> {
    if !bytes.windows(5).take(1024 + 5).any(|w| w == b"%PDF-") {
        return Err(Error::Format("not a PDF".into()));
    }
    let doc = lopdf::Document::load_mem(bytes).map_err(|e| Error::Format(format!("PDF: {e}")))?;
    if doc.is_encrypted() {
        return Err(Error::Format("PDF is encrypted".into()));
    }
    let nums: Vec<u32> = doc.get_pages().keys().copied().collect();
    if nums.len() > MAX_PDF_PAGES {
        return Err(Error::Limit(format!("PDF has {} pages", nums.len())));
    }
    let mut total = 0usize;
    let mut out = Vec::with_capacity(nums.len());
    for n in nums {
        let t = if total >= MAX_PDF_TEXT {
            String::new()
        } else {
            doc.extract_text_with_limit(&[n], MAX_PAGE_STREAM)
                .unwrap_or_default()
        };
        total += t.len();
        out.push(t);
    }
    Ok(out)
}

/// True when the extracted pages hold almost no text (a scan with no text layer).
pub fn looks_scanned(pages: &[String]) -> bool {
    let chars: usize = pages
        .iter()
        .map(|p| p.chars().filter(|c| !c.is_whitespace()).count())
        .sum();
    chars < 25 * pages.len().max(1)
}

fn is_caption(l: &str) -> bool {
    let t = l.trim().to_lowercase();
    ["table ", "tabel ", "tabelul ", "tabela ", "tab. "]
        .iter()
        .any(|p| {
            t.strip_prefix(p)
                .map(|r| {
                    r.chars()
                        .next()
                        .map(|c| c.is_ascii_digit())
                        .unwrap_or(false)
                })
                .unwrap_or(false)
        })
}

fn words(s: &str) -> usize {
    s.split_whitespace().count()
}

// ------------------------------------------------------------ chunk support

fn is_table_line(l: &str) -> bool {
    l.trim_start().starts_with('|')
}

/// Splits a page's text into passages, keeping Markdown tables whole (and repeating the header when a long table must be split)
/// so that a row is never separated from its column names. Prose goes through `chunk_words` as usual.
pub fn chunk_page(
    text: &str,
    size: usize,
    overlap: usize,
    prose: &dyn Fn(&str, usize, usize) -> Vec<String>,
) -> Vec<String> {
    if !text.lines().any(is_table_line) {
        return prose(text, size, overlap);
    }
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut i = 0;
    while i < lines.len() {
        let caption_next =
            is_caption(lines[i]) && i + 1 < lines.len() && is_table_line(lines[i + 1]);
        if caption_next || is_table_line(lines[i]) {
            out.extend(prose(&buf, size, overlap));
            buf.clear();
            let cap = if caption_next {
                i += 1;
                Some(lines[i - 1].trim().to_string())
            } else {
                None
            };
            let mut j = i;
            while j < lines.len() && is_table_line(lines[j]) {
                j += 1;
            }
            let tab = &lines[i..j];
            let head: Vec<&str> = tab.iter().take(2).cloned().collect();
            let body = &tab[head.len()..];
            let prefix =
                |c: &Option<String>| c.as_ref().map(|c| format!("{c}\n")).unwrap_or_default();
            let mut cur: Vec<&str> = Vec::new();
            let mut w = words(&head.join(" "));
            for r in body {
                if !cur.is_empty() && w + words(r) > size * 3 / 2 {
                    out.push(format!(
                        "{}{}\n{}",
                        prefix(&cap),
                        head.join("\n"),
                        cur.join("\n")
                    ));
                    cur.clear();
                    w = words(&head.join(" "));
                }
                cur.push(r);
                w += words(r);
            }
            out.push(format!(
                "{}{}{}",
                prefix(&cap),
                head.join("\n"),
                if cur.is_empty() {
                    String::new()
                } else {
                    format!("\n{}", cur.join("\n"))
                }
            ));
            i = j;
        } else {
            buf.push_str(lines[i]);
            buf.push('\n');
            i += 1;
        }
    }
    out.extend(prose(&buf, size, overlap));
    out.into_iter().filter(|c| !c.trim().is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prose(t: &str, s: usize, o: usize) -> Vec<String> {
        crate::ingest::chunk_words(t, s, o)
    }

    #[test]
    fn chunker_keeps_tables_whole_and_repeats_header() {
        let md = "Some intro words here.\nTable 1: Mean processing time by country (days).\n| Country | Applications |\n| --- | --- |\n| Romania | 1240 |\n| France | 3310 |\n| Germany | 5120 |\nAfter text.\n";
        let ch = chunk_page(md, 200, 40, &prose);
        let tab = ch.iter().find(|c| c.contains("Romania")).unwrap();
        assert!(tab.starts_with("Table 1:") && tab.contains("| Germany | 5120 |"));
        let mut big = String::from("Table 9: Big.\n| a | b |\n| --- | --- |\n");
        for i in 0..400 {
            big.push_str(&format!("| row{i} | value{i} |\n"));
        }
        let parts = chunk_page(&big, 100, 10, &prose);
        assert!(
            parts.len() > 2
                && parts
                    .iter()
                    .all(|p| p.contains("| a | b |") && p.starts_with("Table 9"))
        );
    }

    #[test]
    fn prose_without_tables_passes_through() {
        assert_eq!(
            chunk_page("one two three", 200, 40, &prose),
            vec!["one two three"]
        );
        assert!(chunk_page("", 200, 40, &prose).is_empty());
    }

    /// A tiny hand-built one-page PDF with a text layer.
    fn mini_pdf(text: &str) -> Vec<u8> {
        let content = format!("BT /F1 12 Tf 72 700 Td ({text}) Tj ET");
        let objs = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
            format!("<< /Length {} >>\nstream\n{}\nendstream", content.len(), content),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut out = b"%PDF-1.4\n".to_vec();
        let mut offs = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            offs.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", i + 1, o).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes(),
        );
        for o in offs {
            out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
                objs.len() + 1,
                xref
            )
            .as_bytes(),
        );
        out
    }

    #[test]
    fn reads_text_layer() {
        let p = text_pages(&mini_pdf("Hello burden of proof")).unwrap();
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("burden of proof"), "{:?}", p[0]);
    }

    #[test]
    fn hostile_pdfs_error_cleanly() {
        assert!(text_pages(b"").is_err());
        assert!(text_pages(b"not a pdf at all").is_err());
        let good = mini_pdf("abc");
        for cut in [10, 40, good.len() / 2, good.len() - 5] {
            let _ = text_pages(&good[..cut]); // error or partial, never a crash
        }
        let mut x = 0x1234_5678_9abc_def1u64;
        for _ in 0..200 {
            let mut b = good.clone();
            for _ in 0..4 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let i = (x as usize) % b.len();
                b[i] = (x >> 24) as u8;
            }
            let _ = text_pages(&b);
        }
        let deep = format!(
            "%PDF-1.4\n1 0 obj\n{}\nendobj\ntrailer\n<< /Root 1 0 R >>\n",
            "[".repeat(200_000)
        );
        let _ = text_pages(deep.as_bytes());
    }

    #[test]
    fn scan_detection() {
        assert!(looks_scanned(&["".into(), " ".into()]));
        assert!(!looks_scanned(&["x".repeat(100)]));
    }
}
