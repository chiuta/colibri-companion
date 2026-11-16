use super::*;

fn lib() -> MemPassages {
    let mut m = MemPassages::default();
    let info = |k: &str, y: &str, pages: u32| SourceInfo {
        key: k.into(),
        title: k.into(),
        year: y.into(),
        pages,
        ..SourceInfo::default()
    };
    m.add(
        info("smith2019", "2019", 10),
        vec![
            (3, "The burden of proof lies with the competent authority when it refuses recognition of the diploma.".into()),
            (5, "Compensation measures include an aptitude test or an adaptation period for the migrant professional.".into()),
        ],
    );
    m.add(
        info("doe2020", "2020", 4),
        vec![(
            2,
            "Mutual trust between member states underpins automatic recognition of qualifications."
                .into(),
        )],
    );
    m
}

fn verdicts(r: &AuditReport) -> Vec<Verdict> {
    r.findings.iter().map(|f| f.verdict).collect()
}

fn run(text: &str) -> AuditReport {
    audit(&lib(), text, &AuditOptions::default())
}

const CLAIM: &str =
    "The burden of proof lies with the competent authority when it refuses recognition";

#[test]
fn parses_pandoc_cites() {
    let (m, g) =
        extract_cites("A claim [@smith2019, p. 3; @doe2020]. Not a cite [1] or [see here].");
    assert_eq!(
        m.chars().filter(|c| text::mark_index(*c).is_some()).count(),
        1
    );
    let got: Vec<(&str, Option<usize>)> = g[0].iter().map(|c| (c.key.as_str(), c.page)).collect();
    assert_eq!(got, vec![("smith2019", Some(3)), ("doe2020", None)]);
    assert_eq!(g[0][0].locators.len(), 1);
}

#[test]
fn an_email_address_is_not_a_citation() {
    let (_, g) = extract_cites("Mail [john@example.com] or cite [see @a, p. 2] or [-@b].");
    let keys: Vec<&str> = g.iter().flatten().map(|c| c.key.as_str()).collect();
    assert_eq!(keys, vec!["a", "b"]);
}

#[test]
fn locators_decide_what_is_a_page() {
    let v = |loc: &str| verdicts(&run(&format!("{CLAIM} [@smith2019{loc}].")));
    assert_eq!(v(", cap. 12"), vec![Verdict::Supported]);
    assert!(
        run(&format!("{CLAIM} [@smith2019, cap. 12].")).findings[0]
            .note
            .contains("chapter 12")
    );
    assert_eq!(v(", pp. 2\u{2013}4"), vec![Verdict::Supported]);
    assert_eq!(v(", pp. 3-5"), vec![Verdict::Supported]);
    assert_eq!(v(", p. xii"), vec![Verdict::Supported]);
    assert_eq!(v(", 2019"), vec![Verdict::Supported]); // the work's own year is not a page
    assert_eq!(v(", p. 1, 3"), vec![Verdict::Supported]);
    assert_eq!(v(", pp. 9\u{2013}12"), vec![Verdict::PageOutOfRange]);
    assert_eq!(v(", pp. 5\u{2013}3"), vec![Verdict::BadRange]);
    assert_eq!(v(", p. 5"), vec![Verdict::WrongPage]);
    let r = run(&format!("{CLAIM} [@smith2019, pp. 5\u{2013}3]."));
    assert!(audit_markdown(&r.findings).contains("bad page interval 1"));
}

#[test]
fn a_page_on_an_unpaged_source_is_reported_as_unchecked() {
    let mut m = MemPassages::default();
    m.add(
        SourceInfo {
            key: "txt".into(),
            title: "txt".into(),
            ..SourceInfo::default()
        },
        vec![(0, CLAIM.to_string())],
    );
    let r = audit(
        &m,
        &format!("{CLAIM} [@txt, p. 400]."),
        &AuditOptions::default(),
    );
    assert_eq!(verdicts(&r), vec![Verdict::Supported]);
    assert!(
        r.findings[0].note.contains("no page numbers"),
        "{:?}",
        r.findings[0].note
    );
}

#[test]
fn splits_sentences_keeping_late_markers() {
    let (m, _) =
        extract_cites("First claim here. [@a] Second one, see Smith et al. for more [@b]. Third.");
    let s = sentences(&m);
    assert_eq!(s.len(), 3, "{s:?}");
    let has = |x: &str| x.chars().any(|c| text::mark_index(c).is_some());
    assert!(has(&s[0]) && has(&s[1]) && !has(&s[2]));
}

#[test]
fn latex_and_lyx_input_become_pandoc_cites() {
    let tex = "\\documentclass{article}\n\\begin{document}\n\\section{Intro}\nThe \\emph{burden of proof} lies with the competent authority when it refuses recognition \\parencite[p.~3]{smith2019}. % comment \\cite{x}\nSee \\citep{doe2020,smith2019}.\n\\end{document}";
    let m = normalize_markup(tex);
    assert!(m.contains("# Intro"), "{m}");
    assert!(m.contains("[@smith2019, p. 3]"), "{m}");
    assert!(m.contains("[@doe2020; @smith2019]"), "{m}");
    assert!(!m.contains("comment") && !m.contains("\\emph"), "{m}");
    let r = run(tex);
    assert!(
        r.findings
            .iter()
            .any(|x| matches!(x.verdict, Verdict::Supported | Verdict::Weak)),
        "{:?}",
        r.findings
    );
}

#[test]
fn swapped_figures_are_caught() {
    let r = run(&format!("{CLAIM} within ten days [@smith2019, p. 3]."));
    assert!(
        verdicts(&r).contains(&Verdict::NumberMismatch),
        "{:?}",
        r.findings
    );
    let r = run(&format!("{CLAIM} [@smith2019, p. 3]."));
    assert!(verdicts(&r).iter().all(|v| *v == Verdict::Supported));
}

#[test]
fn audit_verdicts() {
    let txt = "The burden of proof lies with the competent authority when it refuses recognition [@smith2019, p. 3]. \
               The burden of proof lies with the competent authority when it refuses recognition [@smith2019, p. 5]. \
               Penguins migrate across the frozen southern ocean every winter [@smith2019, p. 3]. \
               Mutual trust underpins recognition [@nobody]. Anything [@doe2020, p. 99]. \
               He wrote \"an aptitude test or an adaptation period\" [@smith2019, p. 5]. \
               He wrote \"this sentence never appears in any book here\" [@smith2019]. \
               He wrote \"mutual trust between member states underpins\" [@smith2019].";
    let r = run(txt);
    let v = verdicts(&r);
    for want in [
        Verdict::Supported,
        Verdict::WrongPage,
        Verdict::Unsupported,
        Verdict::UnknownSource,
        Verdict::PageOutOfRange,
        Verdict::QuoteVerbatim,
        Verdict::QuoteNotFound,
        Verdict::QuoteElsewhere,
    ] {
        assert!(v.contains(&want), "{want} missing in {v:?}");
    }
    assert!(!r.truncated());
}

#[test]
fn a_sentence_that_is_only_a_citation_does_not_shift_later_groups() {
    // the original attributed [@doe2020] to the wrong sentence after a marker-only paragraph
    let r = run(&format!("[@doe2020]\n\n{CLAIM} [@smith2019, p. 3]."));
    assert_eq!(verdicts(&r), vec![Verdict::Supported]);
    assert_eq!(r.findings[0].cite.as_deref(), Some("smith2019"));
}

#[test]
fn very_short_sentences_cannot_be_called_supported() {
    let r = run("Recognition [@doe2020, p. 2].");
    assert_eq!(verdicts(&r), vec![Verdict::Weak]);
    assert!(r.findings[0].note.contains("too few"));
}

#[test]
fn numbered_answers_are_checked_against_their_excerpts() {
    let ex = [
        Excerpt {
            label: "Smith 2019",
            page: 3,
            text: "The burden of proof lies with the competent authority when it refuses recognition of the diploma.",
        },
        Excerpt {
            label: "Doe 2020",
            page: 2,
            text: "Mutual trust between member states underpins automatic recognition of qualifications.",
        },
    ];
    let a = "The burden of proof lies with the competent authority when it refuses recognition [1]. \
             Mutual trust between member states underpins automatic recognition [2, 3]. \
             Ghosts walk the halls of the competent authority every single night of the year without any exception whatsoever. \
             Nothing at all [0]. Huge [99999999999999999999999].";
    let r = audit_answer(&ex, a, &AuditOptions::default());
    let by = |c: &str| -> Vec<Verdict> {
        r.findings
            .iter()
            .filter(|f| f.cite.as_deref() == Some(c))
            .map(|f| f.verdict)
            .collect()
    };
    assert_eq!(by("[1]"), vec![Verdict::Supported]);
    assert_eq!(by("[2]"), vec![Verdict::Supported]);
    assert_eq!(by("[3]"), vec![Verdict::UnknownSource]);
    assert_eq!(by("[0]"), vec![Verdict::UnknownSource]);
    assert!(r.findings.iter().any(|f| f.verdict == Verdict::Uncited));
    assert!(
        r.findings
            .iter()
            .any(|f| f.verdict == Verdict::UnknownSource && f.note.contains("far beyond")),
        "{:?}",
        r.findings
    );
}

#[test]
fn quotes_are_matched_on_word_boundaries_and_cannot_be_trivial() {
    // "the competent author" is a prefix of words in the text, not a quotation of it
    let r = run("He wrote \"the competent author when it\" [@smith2019].");
    assert!(
        verdicts(&r).contains(&Verdict::QuoteNotFound),
        "{:?}",
        r.findings
    );
    // punctuation-only "quotation" is ignored instead of matching everywhere
    let r = run("He wrote \"- - - -\" [@smith2019].");
    assert!(
        !verdicts(&r)
            .iter()
            .any(|v| matches!(v, Verdict::QuoteVerbatim))
    );
}

#[test]
fn look_alike_letters_do_not_forge_a_verbatim_quote() {
    // Cyrillic 'а' (U+0430) and 'о' (U+043E) in place of Latin letters
    let forged = "He wrote \"an aptitude test or \u{430}n \u{430}d\u{430}ptati\u{43e}n period\" [@smith2019].";
    let r = run(forged);
    assert!(
        !verdicts(&r).contains(&Verdict::QuoteVerbatim),
        "{:?}",
        r.findings
    );
}

#[test]
fn semantic_pass_rescues_and_catches() {
    let mk = |v: Verdict, sup: f32| Finding {
        sentence: "S".into(),
        cite: Some("k".into()),
        page: None,
        verdict: v,
        support: sup,
        note: String::new(),
        passage: Some("P".into()),
    };
    let mut fs = vec![
        mk(Verdict::Weak, 0.4),
        mk(Verdict::Supported, 0.9),
        mk(Verdict::Supported, 0.8),
        mk(Verdict::Unsupported, 0.1),
        mk(Verdict::UnknownSource, 0.0),
    ];
    let answers = [Sem::Supports, Sem::Contradicts, Sem::Partial, Sem::Unclear];
    let mut i = 0;
    let (checked, changed) = semantic_pass(&mut fs, 100, &mut |_, _| {
        i += 1;
        Some(answers[i - 1])
    });
    assert_eq!((checked, changed), (4, 3));
    assert_eq!(fs[0].verdict, Verdict::Supported);
    assert_eq!(fs[1].verdict, Verdict::Contradicted);
    assert_eq!(fs[2].verdict, Verdict::Weak);
    assert_eq!(fs[3].verdict, Verdict::Unsupported);
    assert_eq!(fs[4].verdict, Verdict::UnknownSource);
    assert!(fs[0].note.contains("[model]"));
    assert_eq!(parse_sem("Contradicts."), Sem::Contradicts);
    assert_eq!(parse_sem("SUPPORTS because"), Sem::Supports);
    assert_eq!(parse_sem("hmm"), Sem::Unclear);
}

#[test]
fn a_model_cannot_lift_a_number_mismatch() {
    let mut fs = vec![Finding {
        sentence: "S".into(),
        cite: None,
        page: None,
        verdict: Verdict::NumberMismatch,
        support: 0.9,
        note: String::new(),
        passage: Some("P".into()),
    }];
    let (_, changed) = semantic_pass(&mut fs, 5, &mut |_, _| Some(Sem::Supports));
    assert_eq!((changed, fs[0].verdict), (0, Verdict::NumberMismatch));
    let (_, changed) = semantic_pass(&mut fs, 5, &mut |_, _| Some(Sem::Contradicts));
    assert_eq!((changed, fs[0].verdict), (1, Verdict::Contradicted));
}

#[test]
fn the_second_opinion_respects_its_call_limit() {
    let mk = || Finding {
        sentence: "S".into(),
        cite: None,
        page: None,
        verdict: Verdict::Weak,
        support: 0.4,
        note: String::new(),
        passage: Some("P".into()),
    };
    let mut fs: Vec<Finding> = (0..10).map(|_| mk()).collect();
    let mut calls = 0;
    semantic_pass(&mut fs, 3, &mut |_, _| {
        calls += 1;
        None
    });
    assert_eq!(calls, 3);
}

#[test]
fn verdict_names_round_trip() {
    for v in Verdict::ALL {
        assert_eq!(Verdict::parse(v.as_str()), Some(v));
    }
    assert_eq!(Verdict::ALL.len(), 13);
    assert_eq!(Verdict::parse("nope"), None);
}

// ------------------------------------------------------------ hostile input

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

#[test]
fn random_text_never_panics_and_respects_caps() {
    let mut next = xorshift(0xdead_beef_cafe_f00d);
    let pool: Vec<&str> = vec![
        "[",
        "]",
        "@smith2019",
        "@doe2020",
        "@x",
        ", p. ",
        "3",
        "12",
        "\"",
        "«",
        "»",
        "“",
        "”",
        ". ",
        "? ",
        "\n\n",
        " ",
        "the",
        "burden",
        "of",
        "proof",
        "authority",
        "ten",
        "10",
        "\u{1}",
        "\u{f0000}",
        "\u{202e}",
        "é",
        "😀",
        "#",
        "\\cite{",
        "}",
        "\\begin{document}",
        "[1]",
        "[2, 3]",
        "[99999999999999999999]",
    ];
    for round in 0..400 {
        let n = (next() % 80) as usize;
        let t: String = (0..n)
            .map(|_| pool[(next() as usize) % pool.len()])
            .collect();
        let r = run(&t);
        assert!(r.findings.len() <= AuditOptions::default().max_findings);
        let ex = [Excerpt {
            label: "x\u{202e}y",
            page: 1,
            text: "the burden of proof",
        }];
        let _ = audit_answer(&ex, &t, &AuditOptions::default());
        let _ = (round, r.markdown());
    }
}

#[test]
fn a_million_citations_are_capped_and_reported() {
    let t0 = std::time::Instant::now();
    let text = "claim [@smith2019, p. 3]. ".repeat(1_000_000 / 26 * 26 / 26 * 10);
    let opts = AuditOptions {
        max_input_bytes: 1 << 20,
        ..AuditOptions::default()
    };
    let r = audit(&lib(), &text, &opts);
    assert!(r.truncated());
    assert!(r.findings.len() <= opts.max_citations + opts.max_quotes + 1);
    assert!(t0.elapsed().as_secs() < 30, "{:?}", t0.elapsed());
    let single = format!("{}.", "[@smith2019]".repeat(300_000));
    let r = audit(&lib(), &single, &AuditOptions::default());
    assert!(
        r.truncated(),
        "a million groups in one sentence must be reported"
    );
}

#[test]
fn huge_and_pathological_inputs_finish() {
    let t0 = std::time::Instant::now();
    let big = "word ".repeat(3_000_000);
    let r = run(&big);
    assert!(r.notes.iter().any(|n| n.contains("longer than")));
    for t in [
        "[".repeat(500_000),
        "«".repeat(300_000),
        format!("\"{}", "a b c d ".repeat(100_000)),
        "a.".repeat(300_000),
        "9".repeat(500_000),
        "one hundred ".repeat(100_000),
        "[@a".repeat(200_000),
        "\\cite{".repeat(100_000),
    ] {
        let _ = run(&t);
    }
    assert!(t0.elapsed().as_secs() < 60, "{:?}", t0.elapsed());
}

#[test]
fn a_huge_library_stops_at_the_work_limit_and_says_so() {
    let mut m = MemPassages::default();
    for i in 0..300 {
        m.add(
            SourceInfo {
                key: format!("k{i}"),
                title: format!("k{i}"),
                ..SourceInfo::default()
            },
            vec![(0, "lorem ipsum dolor sit amet consectetur ".repeat(2_000))],
        );
    }
    let opts = AuditOptions {
        work_budget: 2_000_000,
        ..AuditOptions::default()
    };
    let r = audit(
        &m,
        "He wrote \"a quotation that is nowhere in any of these sources\" [@k1].",
        &opts,
    );
    assert_eq!(verdicts(&r), vec![Verdict::QuoteNotFound]);
    assert!(
        r.findings[0].note.contains("limit"),
        "{}",
        r.findings[0].note
    );
    assert!(r.truncated());
}

#[test]
fn control_characters_in_the_draft_do_not_reach_findings() {
    let r = run(&format!("{CLAIM}\u{202e}\u{0} [@smith2019\u{1b}, p. 3]."));
    for f in &r.findings {
        assert!(!f.sentence.contains('\u{202e}') && !f.sentence.contains('\u{0}'));
        assert!(f.cite.as_deref().is_none_or(|c| !c.contains('\u{1b}')));
    }
}
