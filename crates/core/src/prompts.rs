//! A small library of ready-made prompts with fill-in places like `[TOPIC]`, favourites, the
//! person's own prompts and a few answer styles. Everything is local; nothing is sent anywhere.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prompt {
    pub id: String,
    pub title: String,
    pub category: String,
    pub text: String,
    #[serde(default)]
    pub own: bool,
}

/// (id, category en, category ro, title en, title ro, text en, text ro)
type Row = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
);

const BUILTIN: &[Row] = &[
    (
        "w-email",
        "Writing",
        "Scriere",
        "Polite e-mail",
        "E-mail politicos",
        "Write a short, polite e-mail to [RECIPIENT] about [SUBJECT]. Tone: [TONE]. End with a clear next step.",
        "Scrie un e-mail scurt și politicos către [DESTINATAR] despre [SUBIECT]. Ton: [TON]. Încheie cu un pas următor clar.",
    ),
    (
        "w-improve",
        "Writing",
        "Scriere",
        "Improve my text",
        "Îmbunătățește textul meu",
        "Improve the following text: fix mistakes, make it clearer and shorter, keep my meaning and voice. Show the result, then list what you changed.\n\n[TEXT]",
        "Îmbunătățește textul de mai jos: corectează greșelile, fă-l mai clar și mai scurt, păstrează sensul și vocea mea. Arată rezultatul, apoi enumeră ce ai schimbat.\n\n[TEXT]",
    ),
    (
        "w-summary",
        "Writing",
        "Scriere",
        "Summarise",
        "Rezumă",
        "Summarise the text below in [NUMBER] bullet points for someone who has two minutes. Then give one sentence on what to do next.\n\n[TEXT]",
        "Rezumă textul de mai jos în [NUMĂR] puncte pentru cineva care are două minute. Apoi dă o propoziție despre ce e de făcut în continuare.\n\n[TEXT]",
    ),
    (
        "w-story",
        "Writing",
        "Scriere",
        "Short story",
        "Povestire scurtă",
        "Write a story of about [LENGTH] words. Genre: [GENRE]. Main character: [CHARACTER]. It must end with a surprise that fits the beginning.",
        "Scrie o povestire de aproximativ [LUNGIME] cuvinte. Gen: [GEN]. Personaj principal: [PERSONAJ]. Trebuie să se încheie cu o surpriză care se potrivește începutului.",
    ),
    (
        "w-titles",
        "Writing",
        "Scriere",
        "Ten titles",
        "Zece titluri",
        "Give me ten different titles for [WORK]. Mix serious, playful and mysterious ones, and mark your favourite.",
        "Dă-mi zece titluri diferite pentru [LUCRARE]. Amestecă unele serioase, jucăușe și misterioase, și marchează-l pe cel preferat.",
    ),
    (
        "c-explain",
        "Code",
        "Cod",
        "Explain this code",
        "Explică acest cod",
        "Explain what this [LANGUAGE] code does, step by step, for someone new to it. Point out bugs and risky parts.\n\n[CODE]",
        "Explică ce face acest cod [LIMBAJ], pas cu pas, pentru cineva nou în domeniu. Arată erorile și părțile riscante.\n\n[COD]",
    ),
    (
        "c-bug",
        "Code",
        "Cod",
        "Find the bug",
        "Găsește eroarea",
        "This [LANGUAGE] code should [EXPECTED] but instead [ACTUAL]. Find the cause, show the smallest fix and explain why it works.\n\n[CODE]",
        "Acest cod [LIMBAJ] ar trebui să [AȘTEPTAT] dar în schimb [OBSERVAT]. Găsește cauza, arată cea mai mică corecție și explică de ce funcționează.\n\n[COD]",
    ),
    (
        "c-tests",
        "Code",
        "Cod",
        "Write tests",
        "Scrie teste",
        "Write tests for the code below using [FRAMEWORK]. Cover normal use, edge cases and errors. Explain what each test protects.\n\n[CODE]",
        "Scrie teste pentru codul de mai jos folosind [FRAMEWORK]. Acoperă folosirea normală, cazurile limită și erorile. Explică ce protejează fiecare test.\n\n[COD]",
    ),
    (
        "c-regex",
        "Code",
        "Cod",
        "Regular expression",
        "Expresie regulată",
        "Write a regular expression that matches [WHAT], with three examples that match and three that do not. Explain each part.",
        "Scrie o expresie regulată care potrivește [CE], cu trei exemple care se potrivesc și trei care nu. Explică fiecare parte.",
    ),
    (
        "c-sql",
        "Code",
        "Cod",
        "SQL query",
        "Interogare SQL",
        "Tables: [TABLES]. Write a SQL query that [GOAL]. Explain it and say how to make it fast.",
        "Tabele: [TABELE]. Scrie o interogare SQL care [SCOP]. Explic-o și spune cum se poate face rapidă.",
    ),
    (
        "s-eli5",
        "Study",
        "Studiu",
        "Explain simply",
        "Explică simplu",
        "Explain [TOPIC] as if I were [AGE] years old, with one everyday example and one question to check I understood.",
        "Explică [TEMA] ca și cum aș avea [VÂRSTA] ani, cu un exemplu din viața de zi cu zi și o întrebare prin care să verifici dacă am înțeles.",
    ),
    (
        "s-quiz",
        "Study",
        "Studiu",
        "Quiz me",
        "Întreabă-mă",
        "Quiz me on [TOPIC] with [NUMBER] questions, one at a time. Wait for my answer, say if it is right, explain, then ask the next.",
        "Întreabă-mă din [TEMA] cu [NUMĂR] întrebări, una câte una. Așteaptă răspunsul meu, spune dacă e corect, explică, apoi pune următoarea.",
    ),
    (
        "s-plan",
        "Study",
        "Studiu",
        "Study plan",
        "Plan de studiu",
        "Make a study plan to learn [SUBJECT] in [TIME] with [HOURS] hours a week. Give weekly goals and a way to check progress.",
        "Fă un plan de studiu pentru a învăța [SUBIECT] în [TIMP] cu [ORE] ore pe săptămână. Dă obiective săptămânale și o metodă de a verifica progresul.",
    ),
    (
        "s-cards",
        "Study",
        "Studiu",
        "Flashcards",
        "Fișe de memorare",
        "Make [NUMBER] flashcards from the text below as a two-column table: question | answer.\n\n[TEXT]",
        "Fă [NUMĂR] fișe de memorare din textul de mai jos, ca tabel cu două coloane: întrebare | răspuns.\n\n[TEXT]",
    ),
    (
        "a-compare",
        "Analysis",
        "Analiză",
        "Compare options",
        "Compară variante",
        "Compare [OPTION A] and [OPTION B] for [GOAL]. Use a table with the criteria that matter, then give a recommendation and say what would change it.",
        "Compară [VARIANTA A] și [VARIANTA B] pentru [SCOP]. Folosește un tabel cu criteriile care contează, apoi dă o recomandare și spune ce ar putea să o schimbe.",
    ),
    (
        "a-pros",
        "Analysis",
        "Analiză",
        "Pros, cons, risks",
        "Pro, contra, riscuri",
        "I am thinking about [DECISION]. List the pros, the cons and the risks, ask me the three questions that matter most, and wait for my answers.",
        "Mă gândesc la [DECIZIE]. Enumeră avantajele, dezavantajele și riscurile, pune-mi cele trei întrebări care contează cel mai mult și așteaptă răspunsurile mele.",
    ),
    (
        "a-critic",
        "Analysis",
        "Analiză",
        "Critique my idea",
        "Critică ideea mea",
        "Play a fair but demanding critic of this idea: [IDEA]. Name the weakest assumptions, what could go wrong and how to test it cheaply.",
        "Fii un critic corect dar exigent al acestei idei: [IDEEA]. Numește cele mai slabe presupuneri, ce poate merge prost și cum se poate testa ieftin.",
    ),
    (
        "a-data",
        "Analysis",
        "Analiză",
        "Read this data",
        "Citește aceste date",
        "Here is data: [DATA]. Describe what stands out, what looks wrong, and which three questions it can and cannot answer.",
        "Iată niște date: [DATE]. Descrie ce iese în evidență, ce pare greșit și la ce trei întrebări pot și la ce nu pot răspunde.",
    ),
    (
        "t-translate",
        "Translation",
        "Traducere",
        "Translate",
        "Traduce",
        "Translate the text into [LANGUAGE]. Keep the tone and formatting. Where a phrase has no direct equivalent, add a short note.\n\n[TEXT]",
        "Tradu textul în [LIMBA]. Păstrează tonul și formatarea. Unde o expresie nu are echivalent direct, adaugă o scurtă notă.\n\n[TEXT]",
    ),
    (
        "t-idioms",
        "Translation",
        "Traducere",
        "Idioms",
        "Expresii idiomatice",
        "Give me [NUMBER] idioms about [THEME] in [LANGUAGE], with a literal translation, the real meaning and an example sentence.",
        "Dă-mi [NUMĂR] expresii idiomatice despre [TEMA] în [LIMBA], cu traducere literală, sensul real și o propoziție-exemplu.",
    ),
    (
        "r-grammar",
        "Romanian",
        "Limba română",
        "Check my Romanian",
        "Verifică-mi româna",
        "Check this Romanian text for diacritics, agreement, punctuation and the usual traps (s-au/sau, i-a/ia, într-o/Intr-o). List each correction with the rule.\n\n[TEXT]",
        "Verifică acest text românesc: diacritice, acord, punctuație și capcanele obișnuite (s-au/sau, i-a/ia, într-o). Enumeră fiecare corectură cu regula.\n\n[TEXT]",
    ),
    (
        "r-formal",
        "Romanian",
        "Limba română",
        "Formal letter",
        "Scrisoare oficială",
        "Write a formal letter in Romanian to [INSTITUTION] requesting [REQUEST]. Include the usual opening and closing formulas and a place for the registration number.",
        "Scrie o scrisoare oficială către [INSTITUȚIA] prin care solicit [SOLICITARE]. Include formulele obișnuite de început și încheiere și un loc pentru numărul de înregistrare.",
    ),
    (
        "b-meeting",
        "Work",
        "Muncă",
        "Meeting notes",
        "Notițe de ședință",
        "Turn these rough notes into minutes: decisions, tasks with owner and date, open questions.\n\n[NOTES]",
        "Transformă aceste notițe brute în proces-verbal: decizii, sarcini cu responsabil și dată, întrebări deschise.\n\n[NOTIȚE]",
    ),
    (
        "b-cv",
        "Work",
        "Muncă",
        "Tailor my CV",
        "Adaptează-mi CV-ul",
        "Here is my CV: [CV]. And the job: [JOB]. Suggest what to emphasise, what to cut and three honest sentences for the cover letter. Do not invent anything.",
        "Iată CV-ul meu: [CV]. Și postul: [POST]. Sugerează ce să subliniez, ce să tai și trei propoziții sincere pentru scrisoarea de intenție. Nu inventa nimic.",
    ),
    (
        "b-plan",
        "Work",
        "Muncă",
        "Break into tasks",
        "Împarte în sarcini",
        "Break [PROJECT] into tasks of at most half a day each, in order, with what blocks what, and the first three things I can do today.",
        "Împarte [PROIECTUL] în sarcini de cel mult o jumătate de zi fiecare, în ordine, cu ce blochează ce, și primele trei lucruri pe care le pot face azi.",
    ),
    (
        "e-recipe",
        "Everyday",
        "Zilnic",
        "Cook with what I have",
        "Gătesc cu ce am",
        "I have [INGREDIENTS], [TIME] minutes and [EQUIPMENT]. Suggest three meals, then give the full recipe of the one I pick.",
        "Am [INGREDIENTE], [TIMP] minute și [ECHIPAMENT]. Sugerează trei mâncăruri, apoi dă rețeta completă pentru cea aleasă de mine.",
    ),
    (
        "e-trip",
        "Everyday",
        "Zilnic",
        "Plan a trip",
        "Planifică o excursie",
        "Plan [DAYS] days in [PLACE] for [PEOPLE], budget [BUDGET]. Per day: morning, afternoon, evening, travel times. Say what to book early.",
        "Planifică [ZILE] zile în [LOC] pentru [PERSOANE], buget [BUGET]. Pe zile: dimineață, după-amiază, seară, timpi de deplasare. Spune ce trebuie rezervat din timp.",
    ),
    (
        "e-gift",
        "Everyday",
        "Zilnic",
        "Gift ideas",
        "Idei de cadouri",
        "Give me ten gift ideas for [PERSON] who likes [INTERESTS], budget [BUDGET]. Group them: practical, sentimental, surprising.",
        "Dă-mi zece idei de cadouri pentru [PERSOANA] căreia îi place [PASIUNI], buget [BUGET]. Grupează-le: practice, sentimentale, surprinzătoare.",
    ),
    (
        "e-decide",
        "Everyday",
        "Zilnic",
        "Help me decide",
        "Ajută-mă să decid",
        "Ask me up to five questions, one at a time, to help me decide [DECISION]. Then give your honest recommendation and the main reason.",
        "Pune-mi cel mult cinci întrebări, una câte una, ca să mă ajuți să decid [DECIZIA]. Apoi dă recomandarea ta sinceră și motivul principal.",
    ),
    (
        "v-roleplay",
        "Creative",
        "Creativ",
        "Role play",
        "Joc de rol",
        "Play [CHARACTER] in [SCENE]. Stay in character, keep replies short, and let me lead. Start now.",
        "Joacă rolul [PERSONAJ] în [SCENA]. Rămâi în rol, răspunde scurt și lasă-mă pe mine să conduc. Începe acum.",
    ),
    (
        "v-brainstorm",
        "Creative",
        "Creativ",
        "Brainstorm",
        "Furtună de idei",
        "Brainstorm [NUMBER] ideas for [GOAL]. First the obvious ones, then the odd ones, then three that combine two ideas.",
        "Fă o furtună de idei: [NUMĂR] idei pentru [SCOP]. Mai întâi cele evidente, apoi cele ciudate, apoi trei care combină două idei.",
    ),
];

pub fn builtin(romanian: bool) -> Vec<Prompt> {
    BUILTIN
        .iter()
        .map(|r| Prompt {
            id: r.0.to_string(),
            category: (if romanian { r.2 } else { r.1 }).to_string(),
            title: (if romanian { r.4 } else { r.3 }).to_string(),
            text: (if romanian { r.6 } else { r.5 }).to_string(),
            own: false,
        })
        .collect()
}

/// The fill-in places of a prompt, in order of first appearance: `[TOPIC]` gives `TOPIC`.
pub fn placeholders(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest = text;
    while let Some(a) = rest.find('[') {
        let after = &rest[a + 1..];
        let Some(b) = after.find(']') else {
            break;
        };
        let name = &after[..b];
        let ok = !name.is_empty()
            && name.chars().count() <= 40
            && !name.contains('\n')
            && !name.contains('[')
            && name.chars().any(|c| c.is_alphabetic());
        if ok && !out.iter().any(|x| x == name) {
            out.push(name.to_string());
        }
        rest = if ok { &after[b + 1..] } else { after };
    }
    out
}

/// Puts the typed values in the places; places left empty stay as they are.
pub fn fill(text: &str, values: &[(String, String)]) -> String {
    let mut out = text.to_string();
    for (name, v) in values {
        if !v.trim().is_empty() {
            out = out.replace(&format!("[{name}]"), v.trim());
        }
    }
    out
}

fn fold(s: &str) -> String {
    s.chars()
        .flat_map(|c| c.to_lowercase())
        .map(|c| match c {
            'ă' | 'â' | 'á' | 'à' | 'ä' => 'a',
            'î' | 'í' | 'ì' | 'ï' => 'i',
            'ș' | 'ş' | 'ś' => 's',
            'ț' | 'ţ' => 't',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'ó' | 'ò' | 'ô' | 'ö' | 'ő' => 'o',
            'ú' | 'ù' | 'û' | 'ü' | 'ű' => 'u',
            'ç' => 'c',
            other => other,
        })
        .collect()
}

/// Case- and diacritic-blind match of every word of `query` against title, category and text.
pub fn matches(p: &Prompt, query: &str) -> bool {
    let hay = fold(&format!("{} {} {}", p.title, p.category, p.text));
    fold(query).split_whitespace().all(|w| hay.contains(w))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Library {
    #[serde(default)]
    pub own: Vec<Prompt>,
    #[serde(default)]
    pub favorites: Vec<String>,
}

impl Library {
    pub fn load(path: &Path) -> Library {
        fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(d) = path.parent() {
            fs::create_dir_all(d)?;
        }
        let tmp = path.with_extension("tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(self).unwrap_or_default())?;
        fs::rename(tmp, path)
    }

    pub fn toggle_favorite(&mut self, id: &str) {
        if let Some(i) = self.favorites.iter().position(|f| f == id) {
            self.favorites.remove(i);
        } else {
            self.favorites.push(id.to_string());
        }
    }

    pub fn is_favorite(&self, id: &str) -> bool {
        self.favorites.iter().any(|f| f == id)
    }

    /// Adds or replaces a prompt of the person; the id is made here for a new one.
    pub fn upsert(&mut self, mut p: Prompt) {
        p.own = true;
        p.title = p.title.trim().to_string();
        if p.id.is_empty() {
            let mut n = self.own.len() + 1;
            while self.own.iter().any(|o| o.id == format!("own-{n}")) {
                n += 1;
            }
            p.id = format!("own-{n}");
        }
        if p.category.trim().is_empty() {
            p.category = "Mine".into();
        }
        if let Some(o) = self.own.iter_mut().find(|o| o.id == p.id) {
            *o = p;
        } else {
            self.own.push(p);
        }
    }

    pub fn remove(&mut self, id: &str) {
        self.own.retain(|o| o.id != id);
        self.favorites.retain(|f| f != id);
    }

    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.own).unwrap_or_default()
    }

    /// Adds prompts from a JSON list (what `export_json` writes). Returns how many were new.
    /// Anything that is not a list of prompts with a title and a text is refused.
    pub fn import_json(&mut self, json: &str) -> Result<usize, String> {
        let list: Vec<Prompt> = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let mut added = 0;
        for mut p in list {
            if p.title.trim().is_empty() || p.text.trim().is_empty() || p.text.len() > 100_000 {
                continue;
            }
            if self
                .own
                .iter()
                .any(|o| o.title == p.title && o.text == p.text)
            {
                continue;
            }
            p.id.clear();
            self.upsert(p);
            added += 1;
        }
        Ok(added)
    }
}

/// How the answer should be written. Added to the system prompt for one conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Style {
    #[default]
    Normal,
    Exact,
    Didactic,
    Socratic,
    Contrarian,
    Brainstorm,
    Telegraphic,
    Coach,
    Simple,
}

impl Style {
    pub const ALL: [Style; 9] = [
        Style::Normal,
        Style::Exact,
        Style::Didactic,
        Style::Socratic,
        Style::Contrarian,
        Style::Brainstorm,
        Style::Telegraphic,
        Style::Coach,
        Style::Simple,
    ];

    pub fn instruction(self) -> Option<&'static str> {
        Some(match self {
            Style::Normal => return None,
            Style::Exact => {
                "Answer precisely and only with what you are sure of. Say clearly when you are unsure or do not know. Do not guess."
            }
            Style::Didactic => {
                "Teach: start from the simple idea, build step by step, give one example, and end with a short question to check understanding."
            }
            Style::Socratic => {
                "Do not give the answer at once. Guide with short questions, one at a time, and let the person reach the answer. Give it only if they ask."
            }
            Style::Contrarian => {
                "Politely argue the strongest opposite view of what the person says, so they can test their idea. Be fair and say when they are right."
            }
            Style::Brainstorm => {
                "Give many varied ideas, including odd ones, without judging them. Group them and mark the three most promising."
            }
            Style::Telegraphic => {
                "Answer in the fewest words that stay clear: short lines, no filler."
            }
            Style::Simple => {
                "Write in Easy Read: short sentences of under fifteen words, common everyday words, one idea per sentence, short lists instead of long paragraphs. Explain any hard word the first time. No metaphors, irony or abbreviations."
            }
            Style::Coach => {
                "Act as a warm, direct coach: ask what the person wants, point out what they are already doing well, and agree on one small next step."
            }
        })
    }
}

/// The system prompt plus the style, as sent to the model.
pub fn with_style(system: &str, style: Style) -> String {
    match (system.trim(), style.instruction()) {
        ("", None) => String::new(),
        ("", Some(i)) => i.to_string(),
        (s, None) => s.to_string(),
        (s, Some(i)) => format!("{s}\n\n{i}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_prompts_are_complete_in_both_languages() {
        let (en, ro) = (builtin(false), builtin(true));
        assert!(en.len() >= 30 && en.len() == ro.len());
        let mut ids: Vec<&str> = en.iter().map(|p| p.id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), en.len(), "ids are unique");
        for (a, b) in en.iter().zip(&ro) {
            assert!(!a.title.is_empty() && !a.text.is_empty());
            assert_eq!(
                placeholders(&a.text).len(),
                placeholders(&b.text).len(),
                "{} has the same places in both languages",
                a.id
            );
        }
    }

    #[test]
    fn placeholders_are_found_once_and_filled() {
        let t = "Write to [NAME] about [TOPIC]. Tell [NAME] [ 1 ] more [x\ny].";
        assert_eq!(placeholders(t), ["NAME", "TOPIC"]);
        let out = fill(
            t,
            &[("NAME".into(), " Ana ".into()), ("TOPIC".into(), "".into())],
        );
        assert!(out.starts_with("Write to Ana about [TOPIC]. Tell Ana"));
    }

    #[test]
    fn search_ignores_case_and_diacritics() {
        let p = Prompt {
            id: "x".into(),
            title: "Scrisoare oficială".into(),
            category: "Limba română".into(),
            text: "…".into(),
            own: false,
        };
        assert!(matches(&p, "OFICIALA scrisoare"));
        assert!(matches(&p, "romana"));
        assert!(!matches(&p, "sudoku"));
        assert!(matches(&p, ""));
    }

    #[test]
    fn own_prompts_survive_save_load_and_import_skips_duplicates_and_junk() {
        let dir = std::env::temp_dir().join(format!("cc-prompts-{}", std::process::id()));
        let path = dir.join("prompts.json");
        let mut lib = Library::default();
        lib.upsert(Prompt {
            id: String::new(),
            title: "Mine".into(),
            category: String::new(),
            text: "Hello [WHO]".into(),
            own: false,
        });
        lib.toggle_favorite("own-1");
        lib.save(&path).unwrap();
        let back = Library::load(&path);
        assert_eq!(back.own.len(), 1);
        assert!(back.is_favorite("own-1") && back.own[0].category == "Mine");
        let json = back.export_json();
        let mut other = Library::default();
        assert_eq!(other.import_json(&json).unwrap(), 1);
        assert_eq!(other.import_json(&json).unwrap(), 0);
        assert!(other.import_json("not json").is_err());
        assert_eq!(
            other
                .import_json(r#"[{"id":"a","title":"","category":"","text":"x"}]"#)
                .unwrap(),
            0
        );
        other.remove("own-1");
        assert!(other.own.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn style_is_added_to_the_system_prompt() {
        assert_eq!(with_style("", Style::Normal), "");
        assert_eq!(with_style("Be kind.", Style::Normal), "Be kind.");
        assert!(with_style("Be kind.", Style::Socratic).starts_with("Be kind.\n\n"));
        assert!(with_style("", Style::Exact).contains("precisely"));
        assert!(Style::ALL.iter().skip(1).all(|s| s.instruction().is_some()));
    }
}
