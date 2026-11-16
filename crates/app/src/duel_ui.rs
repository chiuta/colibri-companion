//! Model duels: the same question goes to two anonymous models, the person votes for the better
//! answer and only then sees who was who. Ratings (Elo) are kept on this computer.

use crate::app::{CompanionApp, Event};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::chat::{ChatClient, ChatMessage, ChatRequest, StreamEvent, Thinking};
use companion_core::duel::{self, Book, Combatant, Record, Vote};
use companion_core::http::CancelToken;
use egui::{Align2, Key, RichText, vec2};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Arena,
    Board,
    Models,
}

pub struct Round {
    pub id: u64,
    /// Indices into `book.combatants`; side A and side B.
    pub who: [usize; 2],
    pub text: [String; 2],
    pub done: [bool; 2],
    pub err: [Option<String>; 2],
    pub prompt: String,
    pub category: String,
    pub voted: Option<Vote>,
    pub cancel: CancelToken,
}

pub struct DuelState {
    pub open: bool,
    pub loaded: bool,
    pub book: Book,
    pub tab: Tab,
    pub question: String,
    pub category: String,
    pub set: Vec<(String, String)>,
    pub set_idx: usize,
    pub set_text: String,
    pub round: Option<Round>,
    pub next_id: u64,
    pub board_category: String,
    pub name: String,
    pub base: String,
    pub model: String,
    pub status: String,
}

impl Default for DuelState {
    fn default() -> Self {
        DuelState {
            open: false,
            loaded: false,
            book: Book::default(),
            tab: Tab::Arena,
            question: String::new(),
            category: String::new(),
            set: Vec::new(),
            set_idx: 0,
            set_text: String::new(),
            round: None,
            next_id: 1,
            board_category: String::new(),
            name: String::new(),
            base: String::new(),
            model: String::new(),
            status: String::new(),
        }
    }
}

fn book_path() -> std::path::PathBuf {
    companion_core::paths::data_dir().join("duels.json")
}

/// Only this computer or this network: questions go out in plain http, never to the internet.
fn local_or_lan(url: &str) -> bool {
    let base = companion_core::chat::normalize_base(url);
    companion_core::net::is_local_base(&base)
        || companion_core::connect::check_url(&base).is_ok()
            && companion_core::http::parse_authority(&base).is_ok_and(|(h, _)| {
                h.parse::<std::net::Ipv4Addr>()
                    .is_ok_and(|ip| ip.is_private())
            })
}

fn save_unique(name: &str, ext: &str, text: &str) -> Result<std::path::PathBuf, String> {
    let dir = companion_core::paths::downloads_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut path = dir.join(format!("{name}.{ext}"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{name}-{n}.{ext}"));
        n += 1;
    }
    std::fs::write(&path, text).map_err(|e| e.to_string())?;
    Ok(path)
}

impl CompanionApp {
    pub fn open_duel(&mut self) {
        if !self.duel.loaded {
            self.duel.book = Book::load(&book_path());
            self.duel.loaded = true;
        }
        self.duel.open = true;
    }

    fn duel_save(&mut self) {
        let _ = self.duel.book.save(&book_path());
    }

    fn active_models(&self) -> Vec<usize> {
        self.duel
            .book
            .combatants
            .iter()
            .enumerate()
            .filter(|(_, c)| c.active)
            .map(|(i, _)| i)
            .collect()
    }

    fn load_set(&mut self, text: &str) {
        self.duel.set = duel::parse_prompt_set(text);
        self.duel.set_idx = 0;
        self.next_from_set();
    }

    fn next_from_set(&mut self) {
        if self.duel.set.is_empty() {
            return;
        }
        let (c, q) = self.duel.set[self.duel.set_idx % self.duel.set.len()].clone();
        self.duel.category = c;
        self.duel.question = q;
        self.duel.set_idx += 1;
    }

    fn start_round(&mut self) {
        let q = self.duel.question.trim().to_string();
        let active = self.active_models();
        if q.is_empty() || active.len() < 2 {
            return;
        }
        if let Some(r) = &self.duel.round {
            r.cancel.cancel();
        }
        let id = self.duel.next_id;
        self.duel.next_id += 1;
        let (i, j) = duel::pick_two(active.len(), id);
        let who = [active[i], active[j]];
        let cancel = CancelToken::new();
        self.duel.round = Some(Round {
            id,
            who,
            text: [String::new(), String::new()],
            done: [false, false],
            err: [None, None],
            prompt: q.clone(),
            category: self.duel.category.trim().to_string(),
            voted: None,
            cancel: cancel.clone(),
        });
        self.duel.status.clear();
        for (side, &idx) in who.iter().enumerate() {
            let c = self.duel.book.combatants[idx].clone();
            let (q, cancel, ctx) = (q.clone(), cancel.clone(), self.ctx.clone());
            self.spawn(move |tx| {
                let req = ChatRequest {
                    model: c.model.clone(),
                    messages: vec![ChatMessage::new("user", q)],
                    temperature: 0.7,
                    top_p: 0.95,
                    max_tokens: Some(600),
                    thinking: Thinking::Off,
                };
                let res = ChatClient::new(&c.base, None).stream_chat(&req, &cancel, |ev| {
                    if let StreamEvent::Content(piece) = ev {
                        let _ = tx.send(Event::DuelPiece {
                            round: id,
                            side,
                            piece,
                        });
                        ctx.request_repaint();
                    }
                });
                let err = res.err().map(|e| e.message);
                let _ = tx.send(Event::DuelEnd {
                    round: id,
                    side,
                    err,
                });
            });
        }
    }

    pub fn duel_piece(&mut self, round: u64, side: usize, piece: String) {
        if let Some(r) = self
            .duel
            .round
            .as_mut()
            .filter(|r| r.id == round && side < 2)
        {
            r.text[side].push_str(&piece);
        }
    }

    pub fn duel_end(&mut self, round: u64, side: usize, err: Option<String>) {
        if let Some(r) = self
            .duel
            .round
            .as_mut()
            .filter(|r| r.id == round && side < 2)
        {
            r.done[side] = true;
            r.err[side] = err;
        }
    }

    fn vote(&mut self, v: Vote) {
        let Some(r) = self.duel.round.as_mut() else {
            return;
        };
        if r.voted.is_some() || !(r.done[0] && r.done[1]) {
            return;
        }
        r.voted = Some(v);
        let names = [
            self.duel.book.combatants[r.who[0]].name.clone(),
            self.duel.book.combatants[r.who[1]].name.clone(),
        ];
        let at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let rec = Record {
            at,
            category: r.category.clone(),
            a: names[0].clone(),
            b: names[1].clone(),
            vote: v,
            prompt: r.prompt.clone(),
        };
        self.duel.book.record(rec);
        self.duel_save();
    }

    fn export(&mut self, kind: u8) {
        let cat = self.duel.board_category.clone();
        let (name, ext, text) = match kind {
            0 => (
                "colibri-duels-leaderboard",
                "csv",
                self.duel.book.leaderboard_csv(&cat),
            ),
            1 => (
                "colibri-duels-leaderboard",
                "md",
                self.duel.book.leaderboard_markdown(&cat),
            ),
            _ => ("colibri-duels-votes", "jsonl", self.duel.book.votes_jsonl()),
        };
        self.duel.status = match save_unique(name, ext, &text) {
            Ok(p) => format!("{} {}", self.t("Salvat în", "Saved to"), p.display()),
            Err(e) => format!("{}: {e}", self.t("Nu a mers", "It failed")),
        };
    }

    pub fn duel_window(&mut self, ctx: &egui::Context) {
        if !self.duel.open {
            return;
        }
        let p = self.p;
        let mut open = true;
        // Keys 1-4 vote when no text box has the keyboard.
        if !ctx.egui_wants_keyboard_input() {
            for (k, v) in [
                (Key::Num1, Vote::A),
                (Key::Num2, Vote::B),
                (Key::Num3, Vote::Tie),
                (Key::Num4, Vote::BothBad),
            ] {
                if ctx.input(|i| i.key_pressed(k)) {
                    self.vote(v);
                }
            }
        }
        let tabs = [
            (Tab::Arena, self.t("Arenă", "Arena")),
            (Tab::Board, self.t("Clasament", "Leaderboard")),
            (Tab::Models, self.t("Modele", "Models")),
        ];
        egui::Window::new(self.t("Dueluri între modele", "Model duels"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(fit(ctx, 860.0, 680.0))
            .collapsible(false)
            .show(ctx, |ui| {
                muted(
                    ui,
                    &p,
                    self.t(
                        "Aceeași întrebare merge la două modele anonime. Votezi răspunsul mai bun și abia apoi vezi cine a fost cine. Totul rămâne pe acest calculator; modelele pot fi doar pe acest calculator sau din rețeaua ta.",
                        "The same question goes to two anonymous models. You vote for the better answer and only then see who was who. Everything stays on this computer; the models can only be on this computer or on your network.",
                    ),
                );
                ui.horizontal(|ui| {
                    for (t, label) in tabs {
                        if ui.selectable_label(self.duel.tab == t, label).clicked() {
                            self.duel.tab = t;
                        }
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| match self.duel.tab {
                    Tab::Arena => self.arena_tab(ui),
                    Tab::Board => self.board_tab(ui),
                    Tab::Models => self.models_tab(ui),
                });
            });
        self.duel.open = open;
        if !open {
            if let Some(r) = &self.duel.round {
                r.cancel.cancel();
            }
        }
    }

    fn arena_tab(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let active = self.active_models().len();
        if active < 2 {
            muted(
                ui,
                &p,
                self.t(
                    "Ai nevoie de cel puțin două modele active: adaugă-le în fila «Modele».",
                    "You need at least two active models: add them in the “Models” tab.",
                ),
            );
            if secondary_button(ui, self.t("Mergi la Modele", "Go to Models")).clicked() {
                self.duel.tab = Tab::Models;
            }
            return;
        }
        let h_q = self.t("Scrie întrebarea…", "Write the question…");
        let l_set = self.t("Încarcă setul de întrebări", "Load the sample questions");
        let l_next = self.t("Următoarea întrebare din set", "Next question from the set");
        let l_go = self.t("Începe duelul", "Start the duel");
        ui.add(
            egui::TextEdit::multiline(&mut self.duel.question)
                .hint_text(h_q)
                .desired_rows(3)
                .desired_width(f32::INFINITY),
        );
        ui.horizontal_wrapped(|ui| {
            if secondary_button(ui, l_set).clicked() {
                let set = crate::i18n::demo_set(self.s.lang);
                self.load_set(&set);
            }
            if !self.duel.set.is_empty() && secondary_button(ui, l_next).clicked() {
                self.next_from_set();
            }
            if !self.duel.set.is_empty() {
                muted(
                    ui,
                    &p,
                    &format!(
                        "{}/{}  {}",
                        self.duel.set_idx.min(self.duel.set.len()),
                        self.duel.set.len(),
                        self.duel.category
                    ),
                );
            }
        });
        let l_own = self.t("Setul meu de întrebări", "My own set of questions");
        let h_own = self.t(
            "Câte o întrebare pe linie, sau «categorie | întrebare»",
            "One question per line, or “category | question”",
        );
        let l_use = self.t("Folosește aceste întrebări", "Use these questions");
        egui::CollapsingHeader::new(l_own).show(ui, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut self.duel.set_text)
                    .hint_text(h_own)
                    .desired_rows(4)
                    .desired_width(f32::INFINITY),
            );
            if secondary_button(ui, l_use).clicked() {
                let text = self.duel.set_text.clone();
                self.load_set(&text);
            }
        });
        let busy = self
            .duel
            .round
            .as_ref()
            .is_some_and(|r| !(r.done[0] && r.done[1]));
        ui.add_enabled_ui(!busy && !self.duel.question.trim().is_empty(), |ui| {
            if primary_button(ui, &p, l_go).clicked() {
                self.start_round();
            }
        });
        ui.add_space(8.0);
        let Some(r) = &self.duel.round else {
            return;
        };
        let (texts, done, errs, voted) = (r.text.clone(), r.done, r.err.clone(), r.voted);
        let who = r.who;
        let l_wait = self.t("Se gândește…", "Thinking…");
        ui.columns(2, |cols| {
            for (side, ui) in cols.iter_mut().enumerate() {
                card(ui, &p, |ui| {
                    let title = if side == 0 { "A" } else { "B" };
                    ui.label(RichText::new(title).heading().semibold());
                    if let Some(e) = &errs[side] {
                        ui.colored_label(p.bad, e);
                    }
                    if texts[side].is_empty() && !done[side] {
                        muted(ui, &p, l_wait);
                    }
                    egui::ScrollArea::vertical()
                        .id_salt(("duel-answer", side))
                        .max_height(260.0)
                        .show(ui, |ui| {
                            ui.add(egui::Label::new(&texts[side]).wrap());
                        });
                });
            }
        });
        ui.add_space(8.0);
        if done[0] && done[1] && voted.is_none() {
            let labels = [
                (Vote::A, self.t("1 · A e mai bun", "1 · A is better")),
                (Vote::B, self.t("2 · B e mai bun", "2 · B is better")),
                (Vote::Tie, self.t("3 · La egalitate", "3 · A tie")),
                (
                    Vote::BothBad,
                    self.t("4 · Amândouă slabe", "4 · Both are bad"),
                ),
            ];
            ui.horizontal_wrapped(|ui| {
                for (v, l) in labels {
                    if ui.button(l).clicked() {
                        self.vote(v);
                    }
                }
            });
        }
        if let Some(v) = voted {
            let (na, nb) = (
                self.duel.book.combatants[who[0]].name.clone(),
                self.duel.book.combatants[who[1]].name.clone(),
            );
            let l_a = self.t("Ai ales", "You chose");
            let verdict = match v {
                Vote::A => format!("{l_a}: A"),
                Vote::B => format!("{l_a}: B"),
                Vote::Tie => self.t("Egalitate", "A tie").to_string(),
                Vote::BothBad => self.t("Amândouă slabe", "Both are bad").to_string(),
            };
            card(ui, &p, |ui| {
                ui.label(RichText::new(verdict).semibold());
                ui.label(format!("A = {na}   ·   B = {nb}"));
            });
        }
    }

    fn board_tab(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let cats = self.duel.book.categories();
        let l_all = self.t("Toate categoriile", "All categories");
        let l_csv = self.t("Exportă CSV", "Export CSV");
        let l_md = self.t("Exportă Markdown", "Export Markdown");
        let l_votes = self.t("Exportă voturile (JSONL)", "Export the votes (JSONL)");
        let l_keep = self.t(
            "Păstrează textul întrebărilor în istoric",
            "Keep the question text in the history",
        );
        let l_none = self.t(
            "Încă nu există niciun vot. Fă un duel în fila «Arenă».",
            "There are no votes yet. Run a duel in the “Arena” tab.",
        );
        let cur = if self.duel.board_category.is_empty() {
            l_all.to_string()
        } else {
            self.duel.board_category.clone()
        };
        egui::ComboBox::from_id_salt("duel-cat")
            .selected_text(cur)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.duel.board_category, String::new(), l_all);
                for c in &cats {
                    if !c.is_empty() {
                        ui.selectable_value(&mut self.duel.board_category, c.clone(), c);
                    }
                }
            });
        let rows = self.duel.book.leaderboard(&self.duel.board_category);
        if rows.is_empty() {
            muted(ui, &p, l_none);
        } else {
            let heads = [
                self.t("Loc", "Rank"),
                self.t("Model", "Model"),
                self.t("Punctaj", "Rating"),
                self.t("Dueluri", "Duels"),
                self.t("Victorii", "Wins"),
                self.t("Egalități", "Draws"),
                self.t("Înfrângeri", "Losses"),
            ];
            egui::Grid::new("duel-board")
                .striped(true)
                .spacing([14.0, 6.0])
                .show(ui, |ui| {
                    for h in heads {
                        ui.label(RichText::new(h).small().semibold());
                    }
                    ui.end_row();
                    for (i, r) in rows.iter().enumerate() {
                        ui.label((i + 1).to_string());
                        ui.label(RichText::new(&r.name).semibold());
                        ui.label(format!("{:.0}", r.rating));
                        ui.label(r.duels.to_string());
                        ui.label(r.wins.to_string());
                        ui.label(r.draws.to_string());
                        ui.label(r.losses.to_string());
                        ui.end_row();
                    }
                });
            muted(
                ui,
                &p,
                self.t(
                    "Punctajul pornește de la 1000 și se recalculează din istoric (Elo, K = 32). Cu puține dueluri, diferențele mici nu înseamnă nimic.",
                    "Ratings start at 1000 and are recomputed from the history (Elo, K = 32). With few duels, small differences mean nothing.",
                ),
            );
        }
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            if ui.button(l_csv).clicked() {
                self.export(0);
            }
            if ui.button(l_md).clicked() {
                self.export(1);
            }
            if ui.button(l_votes).clicked() {
                self.export(2);
            }
        });
        let mut keep = self.duel.book.keep_prompts;
        if ui.checkbox(&mut keep, l_keep).changed() {
            self.duel.book.keep_prompts = keep;
            self.duel_save();
        }
        if !self.duel.status.is_empty() {
            muted(ui, &p, &self.duel.status.clone());
        }
    }

    fn models_tab(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        muted(
            ui,
            &p,
            self.t(
                "Orice server care vorbește protocolul OpenAI de chat, pe acest calculator sau în rețeaua ta (colibri, Ollama, LM Studio, llama.cpp…). Adresa începe cu http://.",
                "Any server that speaks the OpenAI chat protocol, on this computer or on your network (colibri, Ollama, LM Studio, llama.cpp…). The address starts with http://.",
            ),
        );
        let mut remove: Option<usize> = None;
        let mut changed = false;
        let l_rm = self.t("Scoate", "Remove");
        for (i, c) in self.duel.book.combatants.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                if ui.checkbox(&mut c.active, "").changed() {
                    changed = true;
                }
                ui.label(RichText::new(&c.name).semibold());
                ui.label(
                    RichText::new(format!("{} · {}", c.model, c.base))
                        .small()
                        .color(p.muted),
                );
                if ui.small_button(l_rm).clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.duel.book.combatants.remove(i);
            changed = true;
        }
        ui.add_space(6.0);
        ui.label(RichText::new(self.t("Adaugă un model", "Add a model")).semibold());
        let h_name = self.t("Nume afișat", "Display name");
        let h_base = self.t(
            "Adresa serverului, de ex. http://127.0.0.1:8000",
            "Server address, e.g. http://127.0.0.1:8000",
        );
        let h_model = self.t("Numele modelului", "Model name");
        ui.add(
            egui::TextEdit::singleline(&mut self.duel.name)
                .hint_text(h_name)
                .desired_width(f32::INFINITY),
        );
        ui.add(
            egui::TextEdit::singleline(&mut self.duel.base)
                .hint_text(h_base)
                .desired_width(f32::INFINITY),
        );
        ui.add(
            egui::TextEdit::singleline(&mut self.duel.model)
                .hint_text(h_model)
                .desired_width(f32::INFINITY),
        );
        let l_add = self.t("Adaugă", "Add");
        let l_cur = self.t(
            "Adaugă modelul care rulează acum",
            "Add the model running now",
        );
        ui.horizontal_wrapped(|ui| {
            let ready = !self.duel.name.trim().is_empty()
                && !self.duel.model.trim().is_empty()
                && !self.duel.base.trim().is_empty();
            if ui.add_enabled(ready, egui::Button::new(l_add)).clicked() {
                if local_or_lan(&self.duel.base) {
                    self.duel.book.combatants.push(Combatant {
                        name: self.duel.name.trim().to_string(),
                        base: companion_core::chat::normalize_base(&self.duel.base),
                        model: self.duel.model.trim().to_string(),
                        active: true,
                    });
                    self.duel.name.clear();
                    self.duel.model.clear();
                    self.duel.status.clear();
                    changed = true;
                } else {
                    self.duel.status = self
                        .t(
                            "Adresa nu e bună: doar servere http:// de pe acest calculator sau din rețeaua ta.",
                            "The address is not valid: only http:// servers on this computer or your own network.",
                        )
                        .into();
                }
            }
            if self.server_ready() && ui.button(l_cur).clicked() {
                let model = self.model_id.clone().unwrap_or_else(|| "colibri".into());
                self.duel.book.combatants.push(Combatant {
                    name: model.clone(),
                    base: self.base(),
                    model,
                    active: true,
                });
                changed = true;
            }
        });
        if !self.duel.status.is_empty() {
            muted(ui, &p, &self.duel.status.clone());
        }
        if changed {
            self.duel_save();
        }
    }
}
