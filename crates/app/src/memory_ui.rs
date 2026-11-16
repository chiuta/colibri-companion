//! Memory window (constellation of conversations, pinned notes, search) and the glue that puts
//! memory in front of the model: pinned notes, fragments of similar conversations, and a running
//! summary for conversations too long for the model's window.

use crate::app::{CompanionApp, Event};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::chat::{ChatClient, ChatMessage, ChatRequest, StreamEvent, Thinking};
use companion_core::http::CancelToken;
use companion_core::memory::{self, Graph, Index, Msg, Pins};
use companion_core::store::date_label;
use companion_core::{ConversationStore, paths};
use egui::{Align2, Color32, Pos2, RichText, Sense, Stroke, vec2};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    #[default]
    Map,
    Notes,
    Search,
}

pub struct MemoryState {
    pub open: bool,
    pub tab: Tab,
    pub index: Option<Arc<Index>>,
    pub graph: Option<Arc<Graph>>,
    pub pins: Pins,
    pub pins_loaded: bool,
    pub building: bool,
    pub dirty: bool,
    pub last_build: Option<Instant>,
    pub sel: Option<usize>,
    pub zoom: f32,
    pub pan: egui::Vec2,
    pub query: String,
    pub new_note: String,
    /// What the last request took from memory: (conversation title, text).
    pub used: Vec<(String, String)>,
    pub summarising: bool,
}

impl Default for MemoryState {
    fn default() -> Self {
        Self {
            open: false,
            tab: Tab::Map,
            index: None,
            graph: None,
            pins: Pins::default(),
            pins_loaded: false,
            building: false,
            dirty: true,
            last_build: None,
            sel: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            query: String::new(),
            new_note: String::new(),
            used: Vec::new(),
            summarising: false,
        }
    }
}

fn pins_path() -> std::path::PathBuf {
    paths::data_dir().join("memory.json")
}

const PALETTE: [Color32; 8] = [
    Color32::from_rgb(167, 139, 250),
    Color32::from_rgb(240, 160, 70),
    Color32::from_rgb(120, 200, 110),
    Color32::from_rgb(90, 170, 250),
    Color32::from_rgb(235, 110, 140),
    Color32::from_rgb(210, 190, 60),
    Color32::from_rgb(110, 205, 190),
    Color32::from_rgb(180, 185, 200),
];

impl CompanionApp {
    pub fn open_memory(&mut self) {
        self.load_pins();
        self.memory.open = true;
    }

    fn load_pins(&mut self) {
        if !self.memory.pins_loaded {
            self.memory.pins = Pins::load(&pins_path());
            self.memory.pins_loaded = true;
        }
    }

    fn save_pins(&mut self) {
        let _ = self.memory.pins.save(&pins_path());
    }

    pub fn pin_text(&mut self, text: &str) {
        self.load_pins();
        let conv = self.chat.conv.id.clone();
        if self
            .memory
            .pins
            .add(text, &conv, companion_core::store::now_secs())
        {
            self.save_pins();
            self.toast(self.t(
                "Am reținut. O voi folosi în conversațiile următoare.",
                "Remembered. I will use it in later conversations.",
            ));
        } else {
            self.toast(self.t("Era deja reținut.", "Already remembered."));
        }
    }

    /// Called every frame: rebuilds the index and the constellation in the background when
    /// conversations have changed.
    pub fn memory_tick(&mut self) {
        if self.memory.building {
            return;
        }
        let stale = self.memory.index.is_none()
            || (self.memory.dirty
                && self
                    .memory
                    .last_build
                    .is_none_or(|t| t.elapsed() > Duration::from_secs(20)));
        if !stale {
            return;
        }
        self.memory.building = true;
        self.memory.dirty = false;
        self.memory.last_build = Some(Instant::now());
        self.spawn(move |tx| {
            let convs = ConversationStore::new().list();
            let index = Index::build(&convs);
            let graph = Graph::build(&convs, &index);
            let _ = tx.send(Event::Memory(Box::new((index, graph))));
        });
    }

    pub fn memory_built(&mut self, b: (Index, Graph)) {
        let (i, g) = b;
        self.memory.index = Some(Arc::new(i));
        self.memory.graph = Some(Arc::new(g));
        self.memory.building = false;
        self.memory.sel = None;
    }

    /// How many characters of conversation fit in a request.
    pub fn context_budget(&self) -> usize {
        self.facts()
            .and_then(|f| f.context_tokens)
            .map_or(24_000, |c| (c as usize).saturating_mul(3) / 2)
            .clamp(6_000, 3_000_000)
    }

    /// The messages to send, with memory and a summary in place of what no longer fits.
    /// Returns the text to add to the system prompt, the request messages after it, and how many old messages were left out.
    pub fn memory_messages(&mut self, system_chars: usize) -> (String, Vec<ChatMessage>, usize) {
        self.load_pins();
        let n = self.chat.conv.messages.len().saturating_sub(1); // all but the empty placeholder
        let msgs = &self.chat.conv.messages[..n];
        let history: Vec<Msg> = msgs
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        let budget = self.context_budget().saturating_sub(system_chars);
        let summary = self.chat.conv.summary.clone();
        let on = self.s.memory_on;
        let pins: Vec<String> = if on {
            self.memory
                .pins
                .notes
                .iter()
                .rev()
                .take(20)
                .map(|p| p.text.chars().take(300).collect())
                .collect()
        } else {
            Vec::new()
        };
        // First pass without fragments: which messages stay in front of the model?
        let first = memory::plan(&history, budget, &summary, &pins, &[]);
        let mut fragments: Vec<String> = Vec::new();
        let mut used: Vec<(String, String)> = Vec::new();
        if on {
            if let (Some(ix), Some(g)) = (self.memory.index.clone(), self.memory.graph.clone()) {
                let query: String = history
                    .iter()
                    .rev()
                    .find(|(r, _)| r == "user")
                    .map(|(_, t)| t.chars().take(600).collect())
                    .unwrap_or_default();
                let (cur, keep_from) = (self.chat.conv.id.clone(), first.dropped);
                let skip = move |d: &memory::Doc| d.conv == cur && d.msg >= keep_from;
                for h in ix.search(&query, 5, &skip) {
                    let d = &ix.docs[h.doc];
                    let title = g
                        .nodes
                        .iter()
                        .find(|x| x.conv == d.conv)
                        .map_or(String::new(), |x| x.title.clone());
                    let when = date_label(d.ts);
                    fragments.push(format!("({title}, {when}) {}", d.text));
                    used.push((title, d.text.clone()));
                }
            }
        }
        self.memory.used = used;
        let planned = memory::plan(&history, budget, &summary, &pins, &fragments);
        let extras = planned.messages.len() - (history.len() - planned.dropped);
        // Memory and the summary join the system prompt: many chat templates accept a system
        // message only at the very start.
        let extra_text = planned.messages[..extras]
            .iter()
            .map(|(_, t)| t.clone())
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut out: Vec<ChatMessage> = Vec::new();
        out.extend(msgs[planned.dropped..].iter().map(|m| {
            let mut c = ChatMessage::new(&m.role, m.content.clone());
            c.images = m.images.clone();
            c
        }));
        (extra_text, out, planned.dropped)
    }

    /// After an answer: if old messages were dropped that the summary does not yet cover, ask
    /// the model to fold them into the summary (in the background).
    pub fn maybe_summarise(&mut self) {
        if self.memory.summarising || !self.server_ready() {
            return;
        }
        let msgs = &self.chat.conv.messages;
        let history: Vec<Msg> = msgs
            .iter()
            .map(|m| (m.role.clone(), m.content.clone()))
            .collect();
        let budget = self.context_budget();
        let dropped = memory::plan(&history, budget, &self.chat.conv.summary, &[], &[]).dropped;
        let covered = self.chat.conv.summary_upto;
        // Wait until a handful of new messages have fallen out, so each summary is worth its cost.
        if dropped < covered + 4 {
            return;
        }
        let prompt = memory::summary_prompt(&self.chat.conv.summary, &history, covered, dropped);
        let (id, client, model) = (
            self.chat.conv.id.clone(),
            ChatClient::new(&self.base(), Some(self.s.api_key.clone())),
            self.model_id.clone().unwrap_or_else(|| "colibri".into()),
        );
        self.memory.summarising = true;
        self.spawn(move |tx| {
            let req = ChatRequest {
                model,
                messages: vec![ChatMessage::new("user", prompt)],
                temperature: 0.2,
                top_p: 0.9,
                max_tokens: Some(700),
                thinking: Thinking::Off,
            };
            let mut text = String::new();
            let r = client
                .stream_chat(&req, &CancelToken::new(), |ev| {
                    if let StreamEvent::Content(c) = ev {
                        text.push_str(&c);
                    }
                })
                .map(|_| text)
                .map_err(|e| e.message);
            let _ = tx.send(Event::Summary {
                conv: id,
                upto: dropped,
                result: r,
            });
        });
    }

    pub fn summary_done(&mut self, conv: String, upto: usize, result: Result<String, String>) {
        self.memory.summarising = false;
        if let Ok(text) = result {
            let text = text.trim().to_string();
            if self.chat.conv.id == conv && !text.is_empty() && upto > self.chat.conv.summary_upto {
                self.chat.conv.summary = text.chars().take(4_000).collect();
                self.chat.conv.summary_upto = upto;
                self.persist_conversation();
            }
        }
    }

    pub fn memory_window(&mut self, ctx: &egui::Context) {
        if !self.memory.open {
            return;
        }
        let p = self.p;
        let mut open = true;
        let mut open_conv: Option<String> = None;
        let mut pins_changed = false;
        egui::Window::new(self.t("Memorie și constelație", "Memory and constellation"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(fit(ctx, 860.0, 700.0))
            .collapsible(false)
            .vscroll(true)
            .show(ctx, |ui| {
                muted(ui, &p, self.t(
                    "Tot ce ai vorbit rămâne pe acest calculator. La fiecare întrebare caut în toate conversațiile fragmentele cele mai apropiate și le dau modelului ca fundal, nu ca instrucțiuni. În conversațiile foarte lungi păstrez mesajele recente întregi și un rezumat al celor vechi. Constelația arată cum se leagă conversațiile între ele.",
                    "Everything you have said stays on this computer. With each question I look through all conversations for the closest fragments and give them to the model as background, not as instructions. In very long conversations I keep the recent messages whole and a summary of the old ones. The constellation shows how conversations relate.",
                ));
                let l = self.t("Folosește memoria (note fixate și fragmente din alte conversații)", "Use memory (pinned notes and fragments from other conversations)");
                if ui.checkbox(&mut self.s.memory_on, l).changed() {
                    let _ = self.s.save();
                }
                ui.horizontal(|ui| {
                    let (a, b, c) = (self.t("Constelație", "Constellation"), self.t("Note fixate", "Pinned notes"), self.t("Căutare", "Search"));
                    ui.selectable_value(&mut self.memory.tab, Tab::Map, a);
                    ui.selectable_value(&mut self.memory.tab, Tab::Notes, b);
                    ui.selectable_value(&mut self.memory.tab, Tab::Search, c);
                    if self.memory.building {
                        crate::theme::busy(ui);
                    }
                });
                ui.separator();
                match self.memory.tab {
                    Tab::Map => self.map_tab(ui, &mut open_conv),
                    Tab::Notes => pins_changed = self.notes_tab(ui),
                    Tab::Search => self.search_tab(ui, &mut open_conv),
                }
            });
        self.memory.open = open;
        if pins_changed {
            self.save_pins();
        }
        if let Some(id) = open_conv {
            self.open_conversation(&id);
        }
    }

    fn map_tab(&mut self, ui: &mut egui::Ui, open_conv: &mut Option<String>) {
        let p = self.p;
        let ro = self.is_ro();
        let Some(g) = self.memory.graph.clone() else {
            muted(
                ui,
                &p,
                self.t("Construiesc constelația…", "Building the constellation…"),
            );
            return;
        };
        if g.nodes.is_empty() {
            muted(
                ui,
                &p,
                self.t(
                    "Nu ai încă conversații. După prima, apare aici.",
                    "No conversations yet. After the first one it appears here.",
                ),
            );
            return;
        }
        muted(ui, &p, self.t(
            "Fiecare stea e o conversație, fiecare linie o asemănare între conținuturi, culoarea e grupul din care face parte. Apasă o stea, sau alege o conversație din listă, ca să vezi detaliile.",
            "Each star is a conversation, each line a resemblance in content, the colour is the group it belongs to. Press a star, or pick a conversation from the list, to see its details.",
        ));
        let (t_pick, t_in, t_out, t_reset) = (
            self.t("— alege —", "— pick —"),
            self.t("Mărește", "Zoom in"),
            self.t("Micșorează", "Zoom out"),
            self.t("Resetează zoomul", "Reset zoom"),
        );
        ui.horizontal_wrapped(|ui| {
            let cur = self
                .memory
                .sel
                .and_then(|i| g.nodes.get(i))
                .map_or(t_pick.to_string(), |n| n.title.clone());
            egui::ComboBox::from_id_salt("mem-pick")
                .selected_text(cur)
                .width(300.0)
                .show_ui(ui, |ui| {
                    for (i, n) in g.nodes.iter().enumerate() {
                        let t = if n.title.is_empty() { "…" } else { &n.title };
                        ui.selectable_value(&mut self.memory.sel, Some(i), t);
                    }
                });
            if ui.button(t_in).clicked() {
                self.memory.zoom = (self.memory.zoom * 1.3).min(12.0);
            }
            if ui.button(t_out).clicked() {
                self.memory.zoom = (self.memory.zoom / 1.3).max(0.4);
            }
            if ui.button(t_reset).clicked() {
                self.memory.zoom = 1.0;
                self.memory.pan = egui::Vec2::ZERO;
            }
        });
        let w = ui.available_width();
        let h = (ui.ctx().content_rect().height() - 400.0).clamp(240.0, 520.0);
        let (resp, painter) = ui.allocate_painter(vec2(w, h), Sense::click_and_drag());
        let rect = resp.rect;
        painter.rect_filled(rect, 8.0, p.panel);
        if resp.dragged() {
            self.memory.pan += resp.drag_delta();
        }
        if resp.hovered() {
            let s = ui.input(|i| i.smooth_scroll_delta.y);
            if s != 0.0 {
                self.memory.zoom = (self.memory.zoom * (1.0 + s * 0.002)).clamp(0.4, 12.0);
            }
        }
        let base = rect.width().min(rect.height()) * 0.46;
        let (zoom, pan, c) = (self.memory.zoom, self.memory.pan, rect.center());
        let to_screen = |x: f32, y: f32| {
            Pos2::new(c.x + pan.x + x * base * zoom, c.y + pan.y + y * base * zoom)
        };
        let sel = self.memory.sel;
        let near: Vec<usize> = sel
            .map(|i| g.neighbours(i).into_iter().map(|(j, _)| j).collect())
            .unwrap_or_default();
        let painter = painter.with_clip_rect(rect);
        for &(a, b, s) in &g.edges {
            let hot = sel.is_some_and(|i| i == a || i == b);
            let alpha = if hot {
                200
            } else {
                (40.0 + s * 160.0).min(120.0) as u8
            };
            let col = Color32::from_rgba_unmultiplied(p.muted.r(), p.muted.g(), p.muted.b(), alpha);
            painter.line_segment(
                [
                    to_screen(g.nodes[a].x, g.nodes[a].y),
                    to_screen(g.nodes[b].x, g.nodes[b].y),
                ],
                Stroke::new(if hot { 2.0 } else { 1.0 }, col),
            );
        }
        // Group names at the centre of each group of two or more.
        for (ci, words) in g.clusters.iter().enumerate() {
            let members: Vec<&memory::Node> = g.nodes.iter().filter(|n| n.cluster == ci).collect();
            if members.len() < 2 {
                continue;
            }
            let (sx, sy) = members
                .iter()
                .fold((0.0, 0.0), |a, n| (a.0 + n.x, a.1 + n.y));
            let pos = to_screen(sx / members.len() as f32, sy / members.len() as f32)
                - egui::vec2(0.0, 30.0);
            let pos = Pos2::new(pos.x, pos.y.max(rect.top() + 10.0));
            painter.text(
                pos,
                Align2::CENTER_CENTER,
                words.join(" · ").to_uppercase(),
                egui::FontId::proportional((11.0 + zoom.min(3.0) * 2.0).min(20.0)),
                Color32::from_rgba_unmultiplied(p.muted.r(), p.muted.g(), p.muted.b(), 150),
            );
        }
        let hover = resp.hover_pos().and_then(|hp| {
            g.nodes
                .iter()
                .enumerate()
                .map(|(i, n)| (i, to_screen(n.x, n.y).distance(hp)))
                .filter(|(_, d)| *d <= 14.0)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(i, _)| i)
        });
        for (i, n) in g.nodes.iter().enumerate() {
            let pos = to_screen(n.x, n.y);
            let r = 4.0 + (n.messages as f32).sqrt().min(6.0);
            let col = PALETTE[n.cluster % PALETTE.len()];
            let dim = sel.is_some_and(|s| s != i && !near.contains(&i));
            let fill = if dim {
                Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 90)
            } else {
                col
            };
            painter.circle_filled(pos, r, fill);
            painter.circle_stroke(pos, r, Stroke::new(1.0, p.bg));
            if sel == Some(i) {
                painter.circle_stroke(pos, r + 4.0, Stroke::new(2.5, p.text));
            }
            if hover == Some(i) {
                painter.circle_stroke(pos, r + 3.0, Stroke::new(1.5, p.text));
            }
            // A short name beside each star, so the map reads without hovering.
            if !n.title.is_empty() && !dim {
                let short: String = n.title.chars().take(24).collect();
                let short = if n.title.chars().count() > 24 {
                    format!("{short}…")
                } else {
                    short
                };
                painter.text(
                    pos + egui::vec2(r + 5.0, 0.0),
                    Align2::LEFT_CENTER,
                    short,
                    egui::FontId::proportional(12.0),
                    p.text,
                );
            }
        }
        if let Some(h) = hover {
            let n = &g.nodes[h];
            egui::Tooltip::always_open(
                ui.ctx().clone(),
                ui.layer_id(),
                resp.id.with("tip"),
                egui::PopupAnchor::Pointer,
            )
            .show(|ui| {
                ui.label(RichText::new(if n.title.is_empty() { "…" } else { &n.title }).semibold());
                ui.label(n.topics.join(", "));
            });
            if resp.clicked() {
                self.memory.sel = Some(h);
            }
        } else if resp.clicked() {
            self.memory.sel = None;
        }
        // Details of the chosen conversation.
        if let Some(i) = self.memory.sel.filter(|i| *i < g.nodes.len()) {
            let n = &g.nodes[i];
            ui.add_space(6.0);
            card(ui, &p, |ui| {
                ui.label(RichText::new(if n.title.is_empty() { "…" } else { &n.title }).semibold());
                muted(
                    ui,
                    &p,
                    &format!(
                        "{} · {} {} · {}",
                        date_label(n.updated),
                        n.messages,
                        self.t("mesaje", "messages"),
                        n.topics.join(", ")
                    ),
                );
                if let Some(words) = g.clusters.get(n.cluster).filter(|w| !w.is_empty()) {
                    muted(
                        ui,
                        &p,
                        &format!("{}: {}", self.t("Grup", "Group"), words.join(" · ")),
                    );
                }
                let nb = g.neighbours(i);
                if !nb.is_empty() {
                    ui.label(self.t("Conversații asemănătoare:", "Similar conversations:"));
                    ui.horizontal_wrapped(|ui| {
                        for (j, s) in nb {
                            let t = &g.nodes[j].title;
                            if ui
                                .small_button(format!(
                                    "{} ({:.0}%)",
                                    if t.is_empty() { "…" } else { t },
                                    s * 100.0
                                ))
                                .clicked()
                            {
                                self.memory.sel = Some(j);
                            }
                        }
                    });
                }
                if ui
                    .button(self.t("Deschide conversația", "Open the conversation"))
                    .clicked()
                {
                    *open_conv = Some(n.conv.clone());
                }
            });
        }
        let _ = ro;
    }

    fn notes_tab(&mut self, ui: &mut egui::Ui) -> bool {
        let p = self.p;
        self.load_pins();
        let mut changed = false;
        muted(ui, &p, self.t(
            "Notele fixate se dau modelului în fiecare conversație. Fixează un mesaj cu butonul «Ține minte» de sub el, sau scrie aici ce vrei să știe mereu (preferințe, nume, reguli ale tale).",
            "Pinned notes are given to the model in every conversation. Pin a message with the “Remember” button under it, or write here what you want it to always know (preferences, names, your own rules).",
        ));
        let h_note = self.t("O notă nouă…", "A new note…");
        ui.add(
            egui::TextEdit::multiline(&mut self.memory.new_note)
                .desired_rows(2)
                .desired_width(f32::INFINITY)
                .hint_text(h_note),
        );
        if ui
            .add_enabled(
                !self.memory.new_note.trim().is_empty(),
                egui::Button::new(self.t("Adaugă nota", "Add the note")),
            )
            .clicked()
        {
            let t = std::mem::take(&mut self.memory.new_note);
            self.pin_text(&t);
        }
        ui.add_space(6.0);
        let mut remove: Option<usize> = None;
        for (i, n) in self.memory.pins.notes.iter_mut().enumerate() {
            ui.horizontal_top(|ui| {
                if ui
                    .add(
                        egui::TextEdit::multiline(&mut n.text)
                            .desired_rows(1)
                            .desired_width(ui.available_width() - 90.0),
                    )
                    .changed()
                {
                    changed = true;
                }
                if ui.button("✕").on_hover_text("Delete").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.memory.pins.notes.remove(i);
            changed = true;
        }
        ui.separator();
        ui.label(
            RichText::new(self.t(
                "Rezumatul conversației curente",
                "Summary of the current conversation",
            ))
            .semibold(),
        );
        if self.chat.conv.summary.is_empty() {
            muted(
                ui,
                &p,
                self.t(
                    "Încă nu e nevoie: conversația încape întreagă în fereastra modelului.",
                    "Not needed yet: the conversation fits whole in the model's window.",
                ),
            );
        } else {
            muted(
                ui,
                &p,
                &format!(
                    "{} {} {}",
                    self.t("Acoperă primele", "Covers the first"),
                    self.chat.conv.summary_upto,
                    self.t("mesaje.", "messages.")
                ),
            );
            ui.label(self.chat.conv.summary.clone());
            if ui
                .small_button(self.t("Șterge rezumatul", "Delete the summary"))
                .clicked()
            {
                self.chat.conv.summary.clear();
                self.chat.conv.summary_upto = 0;
                self.persist_conversation();
            }
        }
        ui.separator();
        ui.label(
            RichText::new(self.t(
                "Ce a folosit ultimul răspuns din memorie",
                "What the last answer used from memory",
            ))
            .semibold(),
        );
        if self.memory.used.is_empty() {
            muted(ui, &p, self.t("Nimic încă.", "Nothing yet."));
        }
        for (t, x) in &self.memory.used {
            let short: String = x.chars().take(160).collect();
            muted(ui, &p, &format!("• {t}: {short}"));
        }
        changed
    }

    fn search_tab(&mut self, ui: &mut egui::Ui, open_conv: &mut Option<String>) {
        let p = self.p;
        let h_q = self.t("Caută în toate conversațiile…", "Search all conversations…");
        ui.add(
            egui::TextEdit::singleline(&mut self.memory.query)
                .desired_width(f32::INFINITY)
                .hint_text(h_q),
        );
        let (Some(ix), Some(g)) = (self.memory.index.clone(), self.memory.graph.clone()) else {
            muted(
                ui,
                &p,
                self.t("Construiesc indexul…", "Building the index…"),
            );
            return;
        };
        if self.memory.query.trim().is_empty() {
            muted(
                ui,
                &p,
                &format!(
                    "{} {} {}",
                    self.t("Indexate:", "Indexed:"),
                    ix.docs.len(),
                    self.t("fragmente.", "fragments.")
                ),
            );
            return;
        }
        let hits = ix.search(&self.memory.query, 25, &|_| false);
        if hits.is_empty() {
            muted(ui, &p, self.t("Nimic găsit.", "Nothing found."));
        }
        for h in hits {
            let d = &ix.docs[h.doc];
            let title = g
                .nodes
                .iter()
                .find(|n| n.conv == d.conv)
                .map_or("…".to_string(), |n| n.title.clone());
            card(ui, &p, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(title).semibold());
                    muted(ui, &p, &date_label(d.ts));
                    if ui.small_button(self.t("Deschide", "Open")).clicked() {
                        *open_conv = Some(d.conv.clone());
                    }
                });
                ui.label(d.text.chars().take(260).collect::<String>());
            });
        }
    }
}
