mod hostile_gen;
use companion_scholar::*;
use hostile_gen::hammer;

const BIB: &str = "@article{key1, author={Ionescu, Ana and O'Brien, Jim}, title={A {B}ig \\\"u Study}, year=2020, journal={J}, doi={10.1/x}, pages={1--9}}\n@book{k2, author=\"Doe, J.\", title=\"T\", year={n.d.}} @comment{x} @string{a=\"b\"}";
const RIS: &str =
    "TY  - JOUR\nAU  - Doe, John\nTI  - Title\nPY  - 2020///\nSP  - 1\nEP  - 9\nER  - \n";
const CSL: &str = r#"[{"id":"a","type":"book","title":"T","author":[{"family":"Doe","given":"J"}],"issued":{"date-parts":[[2020,1]]}}]"#;
const XML: &str =
    "<?xml version=\"1.0\"?><a x=\"1\"><b>t&amp;&#x41;</b><![CDATA[x]]><!-- c --></a>";
const TEXT: &str = "The study found 12.5% growth [@report, p. 3] and \"a quote\" [1, 2]. See doi:10.1000/182 and arXiv:2101.00001v2, ISBN 978-3-16-148410-0.";

#[test]
fn bibliography_readers() {
    hammer("bib::parse_any", &[BIB, RIS, CSL], |s| {
        let _ = bib::parse_any("", s);
        let _ = bib::parse_any("x.ris", s);
        let _ = bib::parse_any("x.json", s);
    });
    hammer("bib::parse_with_warnings", &[BIB], |s| {
        let _ = bib::parse_with_warnings(s);
    });
    hammer("bibtex::read", &[BIB], |s| {
        let _ = bibtex::read(s);
    });
    hammer("ris::parse", &[RIS], |s| {
        let _ = ris::parse(s);
    });
    hammer("csljson::parse", &[CSL], |s| {
        let _ = csljson::parse(s);
    });
    hammer(
        "jcs::parse",
        &[CSL, "{\"a\":[1,2,{\"b\":null}]}", "[1e400, -0, 1.5e-320]"],
        |s| {
            let _ = jcs::parse(s);
        },
    );
}

#[test]
fn round_trip_of_whatever_was_read_never_panics() {
    hammer("export", &[BIB, RIS, CSL], |s| {
        if let Ok((e, _)) = bib::parse_any("", s) {
            for x in &e {
                let _ = x.to_bibtex()
                    + &x.to_ris()
                    + &x.to_csl().to_string()
                    + &x.cite()
                    + &x.reference();
            }
            let _ = csljson::to_json(&e);
            for st in [
                export::Style::Apa,
                export::Style::AuthorYear,
                export::Style::Numeric,
            ] {
                let _ = export::bibliography(&e, st);
            }
            let mut k = keys::KeySet::new();
            for x in &e {
                let _ = k.unique(x, &x.key);
            }
            let r = dedup::analyze(&e);
            let _ = r.render(e.len());
            let _ = r.to_json();
            let _ = dedup::unify(&e);
        }
    });
}

#[test]
fn text_and_markup_readers() {
    hammer("xmlmini", &[XML], |s| {
        let _ = xmlmini::parse(s);
    });
    hammer(
        "names",
        &[
            "Doe, John and van der Berg, A. B. and {Big Corp}",
            "A B C D E",
        ],
        |s| {
            let _ = names::parse_names(s);
        },
    );
    hammer(
        "locator",
        &["p. 3", "pp. 12-15; ch. 2, § 4", "fig. 1"],
        |s| {
            let _ = locator::parse(s);
        },
    );
    hammer(
        "phrase",
        &["\"exact phrase\" and -not term", "a OR b"],
        |s| {
            let _ = phrase::parse(s);
        },
    );
    hammer("ids", &[TEXT], |s| {
        let _ = ids::extract(s);
    });
    hammer(
        "strip_tex",
        &[
            "\\section{A}\\textbf{b} $x^2$ % c\n\\begin{eq}\\end{eq}",
            "\\\\\\{",
        ],
        |s| {
            let _ = ingest::strip_tex(s);
        },
    );
    hammer(
        "verify parsers",
        &["1: 5\n2: 9", "supported", "{\"scores\":[1,2]}"],
        |s| {
            let _ = verify::parse_scores(s, 3);
            let _ = verify::parse_verdict(s);
        },
    );
    hammer("evalset", &["{\"q\":\"x\",\"expect\":[\"a\"]}\n"], |s| {
        let _ = evalset::parse_cases(s);
    });
}

#[test]
fn audit_with_hostile_drafts_and_sources() {
    let opts = audit::AuditOptions::default();
    hammer("audit_answer", &[TEXT], |s| {
        let ex = [
            audit::Excerpt {
                label: "a",
                page: 1,
                text: "growth of 12.5% was seen in the study",
            },
            audit::Excerpt {
                label: "",
                page: 0,
                text: s,
            },
        ];
        let r = audit::audit_answer(&ex, s, &opts);
        let _ = r.markdown();
        let _ = audit::extract_cites(s);
        let _ = audit::parse_sem(s);
    });
}

#[test]
fn file_formats_from_bytes() {
    let seeds: &[&str] = &[
        "%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\nstream\nBT (hi) Tj ET\nendstream\ntrailer<<>>%%EOF",
        "PK\x03\x04",
        "<html><body><p>x</p></body></html>",
        "plain text",
    ];
    hammer("read_pages_bytes", seeds, |s| {
        for ext in [
            "pdf", "epub", "docx", "odt", "html", "htm", "txt", "md", "tex", "rtf", "xml", "zip",
            "bin", "",
        ] {
            let _ = ingest::read_pages_bytes(ext, s.as_bytes());
        }
        let _ = epubpages::read(s.as_bytes());
    });
    // raw non-UTF-8 bytes too
    let mut r = hostile_gen::Rng::new(7);
    for _ in 0..hostile_gen::iterations() {
        let n = r.below(3000);
        let b: Vec<u8> = (0..n).map(|_| r.next_u64() as u8).collect();
        for ext in ["pdf", "epub", "docx", "txt", "html"] {
            let _ = std::panic::catch_unwind(|| ingest::read_pages_bytes(ext, &b))
                .unwrap_or_else(|_| panic!("panic: {ext} {n} bytes"));
        }
    }
}
