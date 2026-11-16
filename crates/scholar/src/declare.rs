// Derived from TEZAUR (MIT, (c) Alexio, https://trom.tf)
//! A plain, first-person declaration of AI use, assembled from receipts, in eight languages (Romanian, English,
//! French, Italian, Spanish, European Portuguese, German, Hungarian).
//!
//! The declaration says what the receipts record (tool, models, period, number of queries and documents) and nothing
//! it cannot know. It does **not** claim that answers are accurate, that every citation was checked, or who wrote the
//! text: the author has to confirm those, and the text ends with a marked line asking for exactly that. It does not
//! say that no text left the computer unless the caller states so ([`Locality::Local`]); the original said it
//! unconditionally.
//!
//! The texts are data: one table per language indexed by [`Key`], so a missing translation is a compile error and a
//! test checks that every language has every key with the same placeholders. They were written with care but have
//! not been read by native speakers of every language; the Portuguese and Hungarian ones especially deserve a look.

use crate::sanitize;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lang {
    Ro,
    En,
    Fr,
    It,
    Es,
    Pt,
    De,
    Hu,
}

impl Lang {
    pub const ALL: [Lang; 8] = [
        Lang::Ro,
        Lang::En,
        Lang::Fr,
        Lang::It,
        Lang::Es,
        Lang::Pt,
        Lang::De,
        Lang::Hu,
    ];

    /// Language code as used elsewhere in the app (`pt` is European Portuguese).
    pub fn code(self) -> &'static str {
        match self {
            Lang::Ro => "ro",
            Lang::En => "en",
            Lang::Fr => "fr",
            Lang::It => "it",
            Lang::Es => "es",
            Lang::Pt => "pt",
            Lang::De => "de",
            Lang::Hu => "hu",
        }
    }

    /// `ro`, `en`, `pt`, `pt-PT`, `PT_pt`... (case-insensitive; the region after `-` or `_` is ignored).
    pub fn from_code(s: &str) -> Option<Lang> {
        let base = s
            .trim()
            .split(['-', '_'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        Lang::ALL.into_iter().find(|l| l.code() == base)
    }

    fn idx(self) -> usize {
        Lang::ALL.iter().position(|l| *l == self).unwrap_or(1)
    }
}

/// The pieces of text a declaration is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Title,
    /// Placeholders: `{tool}`, `{models}`, `{period}`, `{queries}`, `{docs}`.
    Intro,
    LocalYes,
    LocalNo,
    Grounding,
    Confirm,
    ReceiptsHeading,
    NoModels,
    NoPeriod,
    NoReceipts,
    /// Placeholder: `{n}`.
    More,
}

impl Key {
    pub const ALL: [Key; 11] = [
        Key::Title,
        Key::Intro,
        Key::LocalYes,
        Key::LocalNo,
        Key::Grounding,
        Key::Confirm,
        Key::ReceiptsHeading,
        Key::NoModels,
        Key::NoPeriod,
        Key::NoReceipts,
        Key::More,
    ];
}

const N: usize = Key::ALL.len();

/// Row order: ro, en, fr, it, es, pt, de, hu. Column order: `Key::ALL`.
static TEXTS: [[&str; N]; 8] = [
    // ro
    [
        "Declarație privind utilizarea instrumentelor de inteligență artificială",
        "La elaborarea acestei lucrări am folosit {tool}, un instrument care caută în sursele mele proprii și redactează răspunsuri cu un model de limbaj ({models}). Perioada: {period}. Interogări înregistrate: {queries}. Documente din biblioteca mea citate ca surse: {docs}.",
        "Modelul a rulat pe calculatorul meu și niciun text nu a fost trimis către un serviciu extern.",
        "Modelul nu a rulat doar pe calculatorul meu: este posibil ca texte să fi fost trimise către un serviciu extern.",
        "Instrumentul arată pasaje din aceste documente lângă fiecare răspuns. Asta nu face un răspuns corect și nu susțin că l-ar face.",
        "[AUTOR: înainte de semnare, confirmă că ai verificat fiecare citare în sursa ei și că argumentația și textul final îți aparțin. Această declarație consemnează folosirea unui instrument; nu dovedește cine a scris textul și nici că răspunsurile sunt exacte.]",
        "Chitanțe (rezumate SHA-256; fiecare arată că o înregistrare nu s-a schimbat de la emitere, nu că ce conține este corect):",
        "nu s-a înregistrat",
        "nu s-a înregistrat",
        "Nu au fost furnizate chitanțe.",
        "… și încă {n}",
    ],
    // en
    [
        "AI-use declaration",
        "While preparing this work I used {tool}, a tool that searches my own sources and drafts answers with a language model ({models}). Period: {period}. Logged queries: {queries}. Documents from my library cited as sources: {docs}.",
        "The model ran on my own computer and no text was sent to an external service.",
        "The model did not run only on my own computer: text may have been sent to an external service.",
        "The tool shows passages from those documents next to each answer. That does not make an answer correct, and I do not claim that it does.",
        "[AUTHOR: before you sign, confirm that you checked every citation against its source and that the argument and the final text are your own. This declaration records the use of a tool; it does not prove who wrote the text or that the answers are accurate.]",
        "Receipts (SHA-256 digests; each shows that a record has not changed since it was issued, not that its content is correct):",
        "not recorded",
        "not recorded",
        "No receipts were provided.",
        "… and {n} more",
    ],
    // fr
    [
        "Déclaration d'utilisation de l'intelligence artificielle",
        "Pour préparer ce travail, j'ai utilisé {tool}, un outil qui cherche dans mes propres sources et rédige des réponses avec un modèle de langage ({models}). Période : {period}. Requêtes enregistrées : {queries}. Documents de ma bibliothèque cités comme sources : {docs}.",
        "Le modèle s'est exécuté sur mon propre ordinateur et aucun texte n'a été envoyé à un service externe.",
        "Le modèle ne s'est pas exécuté uniquement sur mon propre ordinateur : des textes ont pu être envoyés à un service externe.",
        "L'outil affiche des passages de ces documents à côté de chaque réponse. Cela ne rend pas une réponse exacte, et je ne prétends pas le contraire.",
        "[AUTEUR : avant de signer, confirme que tu as vérifié chaque citation dans sa source et que l'argumentation et le texte final sont de toi. Cette déclaration consigne l'usage d'un outil ; elle ne prouve ni qui a écrit le texte ni que les réponses sont exactes.]",
        "Reçus (empreintes SHA-256 ; chacune montre qu'un enregistrement n'a pas changé depuis son émission, pas que son contenu est exact) :",
        "non enregistré",
        "non enregistrée",
        "Aucun reçu n'a été fourni.",
        "… et {n} de plus",
    ],
    // it
    [
        "Dichiarazione sull'uso dell'intelligenza artificiale",
        "Nella preparazione di questo lavoro ho usato {tool}, uno strumento che cerca nelle mie fonti e redige risposte con un modello linguistico ({models}). Periodo: {period}. Richieste registrate: {queries}. Documenti della mia biblioteca citati come fonti: {docs}.",
        "Il modello è stato eseguito sul mio computer e nessun testo è stato inviato a un servizio esterno.",
        "Il modello non è stato eseguito soltanto sul mio computer: dei testi possono essere stati inviati a un servizio esterno.",
        "Lo strumento mostra passaggi di questi documenti accanto a ogni risposta. Questo non rende corretta una risposta e non sostengo che lo faccia.",
        "[AUTORE: prima di firmare, conferma di aver verificato ogni citazione nella sua fonte e che l'argomentazione e il testo finale sono tuoi. Questa dichiarazione registra l'uso di uno strumento; non prova chi ha scritto il testo né che le risposte siano esatte.]",
        "Ricevute (impronte SHA-256; ciascuna mostra che un record non è cambiato dall'emissione, non che il suo contenuto sia corretto):",
        "non registrato",
        "non registrato",
        "Non sono state fornite ricevute.",
        "… e altre {n}",
    ],
    // es
    [
        "Declaración de uso de inteligencia artificial",
        "Para preparar este trabajo utilicé {tool}, una herramienta que busca en mis propias fuentes y redacta respuestas con un modelo de lenguaje ({models}). Período: {period}. Consultas registradas: {queries}. Documentos de mi biblioteca citados como fuentes: {docs}.",
        "El modelo se ejecutó en mi propio ordenador y no se envió ningún texto a un servicio externo.",
        "El modelo no se ejecutó solo en mi propio ordenador: es posible que se hayan enviado textos a un servicio externo.",
        "La herramienta muestra pasajes de esos documentos junto a cada respuesta. Eso no hace que una respuesta sea correcta, y no afirmo que lo haga.",
        "[AUTOR: antes de firmar, confirma que has comprobado cada cita en su fuente y que la argumentación y el texto final son tuyos. Esta declaración registra el uso de una herramienta; no prueba quién escribió el texto ni que las respuestas sean exactas.]",
        "Recibos (huellas SHA-256; cada una muestra que un registro no ha cambiado desde su emisión, no que su contenido sea correcto):",
        "no registrado",
        "no registrado",
        "No se proporcionó ningún recibo.",
        "… y {n} más",
    ],
    // pt (European)
    [
        "Declaração de utilização de inteligência artificial",
        "Na elaboração deste trabalho utilizei {tool}, uma ferramenta que pesquisa nas minhas próprias fontes e redige respostas com um modelo de linguagem ({models}). Período: {period}. Consultas registadas: {queries}. Documentos da minha biblioteca citados como fontes: {docs}.",
        "O modelo foi executado no meu próprio computador e nenhum texto foi enviado para um serviço externo.",
        "O modelo não foi executado apenas no meu próprio computador: é possível que textos tenham sido enviados para um serviço externo.",
        "A ferramenta mostra excertos desses documentos junto de cada resposta. Isso não torna uma resposta correta, e não afirmo que o torne.",
        "[AUTOR: antes de assinar, confirma que verificaste cada citação na respetiva fonte e que a argumentação e o texto final são teus. Esta declaração regista a utilização de uma ferramenta; não prova quem escreveu o texto nem que as respostas estejam corretas.]",
        "Recibos (resumos SHA-256; cada um mostra que um registo não mudou desde a sua emissão, não que o seu conteúdo esteja correto):",
        "não registado",
        "não registado",
        "Não foi fornecido nenhum recibo.",
        "… e mais {n}",
    ],
    // de
    [
        "Erklärung zur Nutzung von künstlicher Intelligenz",
        "Bei der Erstellung dieser Arbeit habe ich {tool} verwendet, ein Werkzeug, das meine eigenen Quellen durchsucht und mit einem Sprachmodell ({models}) Antworten entwirft. Zeitraum: {period}. Protokollierte Anfragen: {queries}. Als Quellen zitierte Dokumente aus meiner Bibliothek: {docs}.",
        "Das Modell lief auf meinem eigenen Computer, und es wurde kein Text an einen externen Dienst gesendet.",
        "Das Modell lief nicht nur auf meinem eigenen Computer: Texte können an einen externen Dienst gesendet worden sein.",
        "Das Werkzeug zeigt zu jeder Antwort Passagen aus diesen Dokumenten. Das macht eine Antwort nicht richtig, und ich behaupte das auch nicht.",
        "[AUTOR: Bestätige vor der Unterschrift, dass du jede Quellenangabe an ihrer Quelle geprüft hast und dass die Argumentation und der endgültige Text von dir stammen. Diese Erklärung hält die Nutzung eines Werkzeugs fest; sie belegt weder, wer den Text geschrieben hat, noch, dass die Antworten zutreffen.]",
        "Belege (SHA-256-Prüfsummen; jede zeigt, dass sich ein Eintrag seit seiner Ausstellung nicht verändert hat, nicht dass sein Inhalt stimmt):",
        "nicht erfasst",
        "nicht erfasst",
        "Es wurden keine Belege angegeben.",
        "… und {n} weitere",
    ],
    // hu
    [
        "Nyilatkozat a mesterséges intelligencia használatáról",
        "A munka elkészítéséhez a(z) {tool} eszközt használtam, amely a saját forrásaimban keres, és egy nyelvi modellel ({models}) fogalmaz meg válaszokat. Időszak: {period}. Naplózott lekérdezések: {queries}. A könyvtáramból forrásként idézett dokumentumok: {docs}.",
        "A modell a saját számítógépemen futott, és semmilyen szöveg nem került külső szolgáltatáshoz.",
        "A modell nem kizárólag a saját számítógépemen futott: szövegek külső szolgáltatáshoz kerülhettek.",
        "Az eszköz minden válasz mellett megmutatja e dokumentumok részleteit. Ettől egy válasz még nem lesz helyes, és nem is állítom, hogy az lenne.",
        "[SZERZŐ: aláírás előtt erősítsd meg, hogy minden hivatkozást ellenőriztél a forrásában, és hogy az érvelés és a végleges szöveg a tiéd. Ez a nyilatkozat egy eszköz használatát rögzíti; nem bizonyítja, ki írta a szöveget, és azt sem, hogy a válaszok pontosak.]",
        "Igazolások (SHA-256 lenyomatok; mindegyik azt mutatja, hogy egy bejegyzés a kiállítása óta nem változott, nem azt, hogy a tartalma helyes):",
        "nincs rögzítve",
        "nincs rögzítve",
        "Nem adtak meg igazolást.",
        "… és még {n}",
    ],
];

/// The text for `key` in `lang` (placeholders still in it).
pub fn text(lang: Lang, key: Key) -> &'static str {
    let k = Key::ALL.iter().position(|x| *x == key).unwrap_or(0);
    TEXTS[lang.idx()][k]
}

/// Where the model ran. The declaration states this only when the caller knows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Locality {
    /// The model ran on the author's own computer and nothing was sent out.
    Local,
    /// Some text may have gone to an external service.
    External,
    /// Not known: the declaration says nothing about it.
    Unstated,
}

/// What a declaration is built from.
#[derive(Clone, Debug)]
pub struct Use {
    pub tool: String,
    pub models: Vec<String>,
    pub first: Option<u64>,
    pub last: Option<u64>,
    pub queries: usize,
    pub documents: usize,
    /// (seconds since the epoch, digest) of each receipt.
    pub receipts: Vec<(u64, String)>,
    pub locality: Locality,
}

/// Most receipts listed in full; the rest are counted.
pub const MAX_LISTED: usize = 200;
const MAX_MODELS: usize = 20;
const MAX_RECEIPTS_READ: usize = 100_000;
const MAX_DOC_PATHS: usize = 1_000_000;

impl Use {
    /// Summarises receipt texts (format 1 or 2). Returns the summary and how many texts could not be read.
    /// Nothing is verified here: run [`crate::receipt::verify`] first on receipts you do not trust.
    pub fn from_receipts(texts: &[&str], tool: &str, locality: Locality) -> (Use, usize) {
        let mut models = BTreeSet::new();
        let mut docs = BTreeSet::new();
        let mut receipts = Vec::new();
        let (mut first, mut last) = (None::<u64>, None::<u64>);
        let mut skipped = 0;
        for t in texts.iter().take(MAX_RECEIPTS_READ) {
            let Some(v) = (t.len() <= crate::receipt::MAX_RECEIPT_BYTES)
                .then(|| serde_json::from_str::<serde_json::Value>(t).ok())
                .flatten()
                .filter(|v| v.is_object())
            else {
                skipped += 1;
                continue;
            };
            if let Some(m) = v.get("model").and_then(|m| m.as_str()) {
                let m = sanitize::short(m, 120);
                if !m.is_empty() {
                    models.insert(m);
                }
            }
            if let Some(a) = v.get("excerpts").and_then(|a| a.as_array()) {
                for e in a {
                    if docs.len() < MAX_DOC_PATHS
                        && let Some(p) = e.get("path").and_then(|p| p.as_str())
                    {
                        docs.insert(p.to_string());
                    }
                }
            }
            let created = v.get("created").and_then(|c| c.as_u64()).unwrap_or(0);
            first = Some(first.map_or(created, |f| f.min(created)));
            last = Some(last.map_or(created, |l| l.max(created)));
            let digest = v.get("digest").and_then(|d| d.as_str()).unwrap_or("");
            receipts.push((created, digest.to_string()));
        }
        skipped += texts.len().saturating_sub(MAX_RECEIPTS_READ);
        (
            Use {
                tool: tool.to_string(),
                models: models.into_iter().collect(),
                first,
                last,
                queries: receipts.len(),
                documents: docs.len(),
                receipts,
                locality,
            },
            skipped,
        )
    }
}

/// `yyyy-mm-dd` for seconds since the epoch (UTC); `?` beyond the year 9999.
pub fn civil(secs: u64) -> String {
    if secs > 253_402_300_799 {
        return "?".into();
    }
    let z = (secs / 86400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{:02}-{:02}", if m <= 2 { y + 1 } else { y }, m, d)
}

fn fill(template: &str, pairs: &[(&str, &str)]) -> String {
    let mut s = template.to_string();
    for (k, v) in pairs {
        s = s.replace(&format!("{{{k}}}"), v);
    }
    s
}

fn is_hex_digest(d: &str) -> bool {
    d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The declaration as Markdown, in `lang`.
pub fn declaration(u: &Use, lang: Lang) -> String {
    let t = |k| text(lang, k);
    let mut models: Vec<String> = u
        .models
        .iter()
        .take(MAX_MODELS)
        .map(|m| sanitize::short(m, 120))
        .filter(|m| !m.is_empty())
        .collect();
    if u.models.len() > MAX_MODELS {
        models.push(fill(
            t(Key::More),
            &[("n", &(u.models.len() - MAX_MODELS).to_string())],
        ));
    }
    let models = if models.is_empty() {
        t(Key::NoModels).to_string()
    } else {
        models.join(", ")
    };
    let period = match (u.first, u.last) {
        (Some(a), Some(b)) if civil(a) == civil(b) => civil(a),
        (Some(a), Some(b)) => format!("{} – {}", civil(a), civil(b)),
        _ => t(Key::NoPeriod).to_string(),
    };
    let tool = sanitize::short(&u.tool, 120);
    let mut s = format!("# {}\n\n", t(Key::Title));
    s.push_str(&fill(
        t(Key::Intro),
        &[
            ("tool", &tool),
            ("models", &models),
            ("period", &period),
            ("queries", &u.queries.to_string()),
            ("docs", &u.documents.to_string()),
        ],
    ));
    s.push_str("\n\n");
    match u.locality {
        Locality::Local => s.push_str(&format!("{}\n\n", t(Key::LocalYes))),
        Locality::External => s.push_str(&format!("{}\n\n", t(Key::LocalNo))),
        Locality::Unstated => {}
    }
    s.push_str(&format!("{}\n\n{}\n\n", t(Key::Grounding), t(Key::Confirm)));
    if u.receipts.is_empty() {
        s.push_str(t(Key::NoReceipts));
        s.push('\n');
        return s;
    }
    s.push_str(t(Key::ReceiptsHeading));
    s.push('\n');
    for (created, digest) in u.receipts.iter().take(MAX_LISTED) {
        let d = if is_hex_digest(digest) {
            digest.to_ascii_lowercase()
        } else {
            "—".to_string()
        };
        s.push_str(&format!("- {} — {}\n", civil(*created), d));
    }
    if u.receipts.len() > MAX_LISTED {
        s.push_str(&format!(
            "- {}\n",
            fill(
                t(Key::More),
                &[("n", &(u.receipts.len() - MAX_LISTED).to_string())]
            )
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placeholders(s: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        let mut rest = s;
        while let Some(a) = rest.find('{') {
            let Some(b) = rest[a..].find('}') else { break };
            out.insert(rest[a + 1..a + b].to_string());
            rest = &rest[a + b + 1..];
        }
        out
    }

    #[test]
    fn every_language_has_every_key_with_the_same_placeholders() {
        for key in Key::ALL {
            let want = placeholders(text(Lang::En, key));
            for lang in Lang::ALL {
                let t = text(lang, key);
                assert!(!t.trim().is_empty(), "{lang:?} {key:?} is empty");
                assert_eq!(placeholders(t), want, "{lang:?} {key:?}");
            }
        }
        let intro = placeholders(text(Lang::En, Key::Intro));
        for p in ["tool", "models", "period", "queries", "docs"] {
            assert!(intro.contains(p), "{p}");
        }
        // distinct texts: a copy of the English row would pass the checks above
        for key in [Key::Title, Key::Intro, Key::Confirm, Key::Grounding] {
            let set: BTreeSet<&str> = Lang::ALL.iter().map(|l| text(*l, key)).collect();
            assert_eq!(
                set.len(),
                Lang::ALL.len(),
                "{key:?} has duplicate translations"
            );
        }
        assert_eq!(TEXTS.len(), Lang::ALL.len());
    }

    #[test]
    fn the_texts_claim_neither_accuracy_nor_identity() {
        // the author-confirmation line is present in every language and marked, so it cannot be mistaken for text
        for lang in Lang::ALL {
            let c = text(lang, Key::Confirm);
            assert!(c.starts_with('[') && c.ends_with(']'), "{lang:?}");
        }
    }

    #[test]
    fn language_codes() {
        assert_eq!(Lang::from_code("pt-PT"), Some(Lang::Pt));
        assert_eq!(Lang::from_code(" RO "), Some(Lang::Ro));
        assert_eq!(Lang::from_code("hu_HU"), Some(Lang::Hu));
        assert_eq!(Lang::from_code("xx"), None);
        assert_eq!(Lang::from_code(""), None);
        for l in Lang::ALL {
            assert_eq!(Lang::from_code(l.code()), Some(l));
        }
    }

    #[test]
    fn civil_dates() {
        assert_eq!(civil(0), "1970-01-01");
        assert_eq!(civil(951_782_400), "2000-02-29");
        assert_eq!(civil(1_800_000_000), "2027-01-15");
        assert_eq!(civil(253_402_300_799), "9999-12-31");
        assert_eq!(civil(253_402_300_800), "?");
        assert_eq!(civil(u64::MAX), "?");
    }

    fn receipts() -> Vec<String> {
        let d = "a".repeat(64);
        vec![
            format!(
                r#"{{"format":"tezaur-receipt/2","created":1800000000,"model":"llama-3-8b","excerpts":[{{"path":"a.pdf"}},{{"path":"b.pdf"}}],"digest":"{d}"}}"#
            ),
            format!(
                r#"{{"format":"tezaur-receipt/1","created":1800086400,"model":"qwen","excerpts":[{{"path":"a.pdf"}}],"digest":"{d}"}}"#
            ),
            "not json".to_string(),
        ]
    }

    #[test]
    fn built_from_receipts() {
        let r = receipts();
        let refs: Vec<&str> = r.iter().map(String::as_str).collect();
        let (u, skipped) = Use::from_receipts(&refs, "Colibri Companion 0.1", Locality::Local);
        assert_eq!((u.queries, u.documents, skipped), (2, 2, 1));
        assert_eq!(u.models, vec!["llama-3-8b", "qwen"]);
        for lang in Lang::ALL {
            let d = declaration(&u, lang);
            assert!(
                d.contains("Colibri Companion 0.1") && d.contains("llama-3-8b, qwen"),
                "{lang:?}\n{d}"
            );
            assert!(d.contains("2027-01-15 – 2027-01-16"), "{lang:?}");
            assert!(d.contains(text(lang, Key::LocalYes)));
            assert!(
                !d.contains('{') && !d.contains('}'),
                "unfilled placeholder in {lang:?}: {d}"
            );
            assert_eq!(d.matches(&"a".repeat(64)).count(), 2);
        }
    }

    #[test]
    fn locality_is_stated_only_when_known() {
        let (mut u, _) = Use::from_receipts(&[], "T", Locality::Unstated);
        let d = declaration(&u, Lang::En);
        assert!(!d.contains("ran on my own computer") && !d.contains("did not run only"));
        assert!(d.contains(text(Lang::En, Key::NoReceipts)));
        assert!(d.contains(text(Lang::En, Key::NoModels)));
        u.locality = Locality::External;
        assert!(declaration(&u, Lang::En).contains("may have been sent"));
        u.locality = Locality::Local;
        assert!(declaration(&u, Lang::En).contains("no text was sent"));
    }

    #[test]
    fn hostile_receipts_cannot_inject_text_or_blow_up() {
        let evil_model = "gpt\u{202e}\n# Injected heading\u{0}".to_string();
        let r = format!(
            r#"{{"created":18446744073709551615,"model":{},"digest":"<script>"}}"#,
            serde_json::to_string(&evil_model).unwrap()
        );
        let (u, _) = Use::from_receipts(&[&r], "tool\nname\u{1b}", Locality::Unstated);
        let d = declaration(&u, Lang::De);
        assert!(!d.contains("\n# Injected"), "{d}");
        assert!(!d.contains('\u{202e}') && !d.contains('\u{1b}') && !d.contains('\u{0}'));
        assert!(d.contains("— ?") || d.contains("?"), "{d}"); // absurd timestamp
        assert!(!d.contains("<script>"));
        // many receipts: the list is cut and says so
        let many: Vec<String> = (0..MAX_LISTED + 25)
            .map(|i| format!(r#"{{"created":{i},"model":"m{i}","digest":"x"}}"#))
            .collect();
        let refs: Vec<&str> = many.iter().map(String::as_str).collect();
        let (u, _) = Use::from_receipts(&refs, "T", Locality::Unstated);
        let d = declaration(&u, Lang::En);
        assert!(d.contains("… and 25 more"), "{d}");
        assert!(d.lines().count() < 300);
        assert!(d.contains("… and 205 more")); // models beyond the first 20 are counted too
    }
}
