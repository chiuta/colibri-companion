//! The conversation screen: saved chats on the left, messages in the middle,
//! one comfortable box to type in at the bottom.

use crate::app::{CompanionApp, HeightKey, SearchCache};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::chat::{ChatError, ErrorKind, Health, Thinking};
use companion_core::store::date_label;
use companion_core::{StoredMessage, Style};
use egui::{Align, CornerRadius, Frame, Key, Layout, Margin, Modifiers, Panel, RichText, Ui, vec2};
use std::time::{Duration, Instant};

enum MsgAction {
    Copy(String),
    Pin(String),
    Speak(String),
    StopSpeaking,
    Regenerate,
    Edit,
    Branch(usize),
    SaveBlock(companion_core::blocks::Block),
    Preview(companion_core::blocks::Block),
    Ask(usize, usize),
    CancelAsk,
    Usage(Box<companion_core::meter::AnswerStats>),
}

/// Width of the conversation list for a window `w` wide, and whether the usage panel still leaves
/// the conversation at least 420 px.
pub fn chat_panel_widths(w: f32) -> (f32, bool) {
    let side = (w * 0.24).clamp(200.0, 270.0);
    (side, w - side - 220.0 >= 420.0)
}

impl CompanionApp {
    pub fn show_chat(&mut self, ui: &mut Ui) {
        let p = self.p;
        // The side panels share the window with the conversation: they get narrower with it, and
        // the usage panel steps aside when the conversation would be left with under ~420 px.
        let w = ui.available_width();
        let (side, room_for_meter) = chat_panel_widths(w);
        let show_meter = self.show_meter() && room_for_meter;
        Panel::left("sidebar")
            .default_size(side)
            .min_size(180.0)
            .max_size((w * 0.4).max(200.0))
            .frame(Frame::new().fill(p.panel).inner_margin(Margin::same(10)))
            .show(ui, |ui| self.sidebar(ui));
        if show_meter {
            Panel::right("meter")
                .default_size((w * 0.26).clamp(220.0, 290.0))
                .min_size(200.0)
                .max_size((w * 0.4).max(220.0))
                .frame(Frame::new().fill(p.panel).inner_margin(Margin::same(10)))
                .show(ui, |ui| self.meter_panel(ui));
        }
        egui::CentralPanel::default()
            .frame(Frame::new().fill(p.bg))
            .show(ui, |ui| self.chat_main(ui));
    }

    fn sidebar(&mut self, ui: &mut Ui) {
        let p = self.p;
        Panel::bottom("sidebar_foot")
            .frame(Frame::new())
            .show(ui, |ui| {
                ui.add_space(6.0);
                if !self.chat.conv.messages.is_empty() {
                    ui.horizontal(|ui| {
                        let w = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
                        if ui
                            .add_sized([w, 32.0], egui::Button::new(self.t("Copiază", "Copy")))
                            .on_hover_text(self.t(
                                "Copiază conversația (Markdown)",
                                "Copy conversation (Markdown)",
                            ))
                            .clicked()
                        {
                            let md = self.chat.conv.to_markdown(self.t("Eu", "Me"), "Colibri");
                            ui.ctx().copy_text(md);
                            self.toast(self.t("Conversația a fost copiată", "Conversation copied"));
                        }
                        if ui
                            .add_sized(
                                [w, 32.0],
                                egui::Button::new(self.t("Salvează", "Save as file")),
                            )
                            .on_hover_text(self.t(
                                "Salvează conversația ca fișier .md în Descărcări",
                                "Save the conversation as a .md file in Downloads",
                            ))
                            .clicked()
                        {
                            self.export_conversation();
                        }
                    });
                }
            });
        if primary_button(ui, &p, self.t("+ Conversație nouă", "+ New conversation")).clicked() {
            self.new_conversation();
        }
        ui.add_space(4.0);
        ui.label(
            RichText::new(self.t("Conversațiile tale", "Your conversations"))
                .semibold()
                .color(p.muted),
        );
        let search_id = egui::Id::unique("sidebar-search");
        ui.horizontal(|ui| {
            let spacing = ui.spacing().item_spacing.x;
            let clear_w = 38.0 + spacing; // always reserved: the box must not change size while typing
            let hint = self.t("Caută… (Ctrl+K)", "Search… (Ctrl+K)");
            let r = ui.add_sized(
                [(ui.available_width() - clear_w - 2.0).max(60.0), 30.0],
                egui::TextEdit::singleline(&mut self.chat.search)
                    .id(search_id)
                    .hint_text(hint),
            );
            if self.chat.focus_search {
                r.request_focus();
                self.chat.focus_search = false;
            }
            if r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.chat.search.clear();
            }
            if self.chat.search.is_empty() {
                ui.allocate_space(vec2(clear_w - spacing, 30.0));
            } else if ui
                .add_sized([clear_w - spacing, 30.0], egui::Button::new("×"))
                .clicked()
            {
                self.chat.search.clear();
            }
        });
        let current = self.chat.conv.id.clone();
        let query = self.chat.search.trim().to_lowercase();
        let stamp = (
            self.chat.list.len(),
            self.chat
                .list
                .iter()
                .fold(0u64, |a, c| a.wrapping_add(c.updated)),
        );
        // Searching reads every saved conversation, so it runs once per search text (and again when
        // the list changes or after a few seconds), not on every redraw.
        let fresh = search_cache_is_fresh(self.chat.search_cache.as_ref(), &query, stamp);
        if query.is_empty() {
            self.chat.search_cache = None;
        } else if !fresh {
            self.chat.search_cache = Some(SearchCache {
                hits: self.store.search(&query).into_iter().collect(),
                query: query.clone(),
                stamp,
                at: Instant::now(),
            });
        }
        let searching = !query.is_empty();
        // indices only: the rows that are on screen are copied below, not the whole history
        let shown: Vec<usize> = self
            .chat
            .list
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                self.chat
                    .search_cache
                    .as_ref()
                    .is_none_or(|s| s.hits.contains(&c.id))
            })
            .map(|(i, _)| i)
            .collect();
        let mut open = None;
        let mut delete = None;
        let mut rename: Option<(String, String)> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_rows(ui, 40.0, shown.len(), |ui, range| {
                if shown.is_empty() {
                    let msg = if searching {
                        self.t(
                            "Nicio conversație nu se potrivește.",
                            "No conversations match.",
                        )
                    } else {
                        self.t(
                            "Aici vor apărea conversațiile salvate.",
                            "Saved conversations will show up here.",
                        )
                    };
                    ui.label(RichText::new(msg).color(p.muted));
                }
                let visible: Vec<(String, String, u64)> = range
                    .filter_map(|r| self.chat.list.get(*shown.get(r)?))
                    .map(|c| (c.id.clone(), c.title.clone(), c.updated))
                    .collect();
                for (id, title, updated) in &visible {
                    let new_name_hint = self.t("Nume nou", "New name");
                    if let Some((rid, name)) =
                        self.chat.renaming.as_mut().filter(|(rid, _)| rid == id)
                    {
                        let edit_id = egui::Id::unique(("rename", rid.as_str()));
                        let first = !ui.ctx().memory(|m| m.has_focus(edit_id));
                        let chars = name.chars().count();
                        let r = ui.add_sized(
                            [ui.available_width(), 40.0],
                            egui::TextEdit::singleline(name)
                                .id(edit_id)
                                .hint_text(new_name_hint),
                        );
                        if first {
                            // the old name starts selected, so typing replaces it
                            r.request_focus();
                            if let Some(mut st) = egui::TextEdit::load_state(ui.ctx(), edit_id) {
                                st.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                                    egui::text::CCursor::new(0),
                                    egui::text::CCursor::new(chars),
                                )));
                                egui::TextEdit::store_state(ui.ctx(), edit_id, st);
                            }
                        }
                        let enter = (r.has_focus() || r.lost_focus())
                            && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
                        if enter {
                            rename = Some((rid.clone(), name.clone()));
                        } else if esc || (r.lost_focus() && !enter) {
                            rename = Some((String::new(), String::new()));
                        }
                        continue;
                    }
                    ui.horizontal(|ui| {
                        let confirming = self.chat.confirm_delete.as_deref() == Some(id.as_str());
                        let shown = if title.is_empty() {
                            self.t("(fără titlu)", "(untitled)")
                        } else {
                            title.as_str()
                        };
                        let del_w = if confirming { 64.0 } else { 38.0 };
                        let w =
                            (ui.available_width() - del_w - ui.spacing().item_spacing.x).max(60.0);
                        let resp = ui
                            .add_sized(
                                [w, 40.0],
                                egui::Button::selectable(
                                    *id == current,
                                    RichText::new(shown).size(14.5),
                                )
                                .truncate()
                                .right_text(""),
                            )
                            .on_hover_text(format!("{} · {}", shown, date_label(*updated)));
                        if resp.clicked() {
                            open = Some(id.clone());
                        }
                        resp.context_menu(|ui| {
                            if ui.button(self.t("Redenumește", "Rename")).clicked() {
                                self.chat.renaming = Some((id.clone(), title.clone()));
                                ui.close();
                            }
                        });
                        let (icon, tip) = if confirming {
                            (
                                self.t("Sigur?", "Sure?"),
                                self.t(
                                    "Apasă încă o dată pentru a șterge",
                                    "Press again to delete",
                                ),
                            )
                        } else {
                            ("×", self.t("Șterge conversația", "Delete conversation"))
                        };
                        if ui
                            .add_sized(
                                [del_w, 40.0],
                                egui::Button::new(RichText::new(icon).color(p.bad)),
                            )
                            .on_hover_text(tip)
                            .clicked()
                        {
                            if confirming {
                                delete = Some(id.clone());
                            } else {
                                self.chat.confirm_delete = Some(id.clone());
                            }
                        }
                    });
                }
            });
        if let Some((id, name)) = rename {
            if !id.is_empty() {
                self.rename_conversation(&id, &name);
            }
            self.chat.renaming = None;
        }
        if let Some(id) = open {
            self.chat.confirm_delete = None;
            self.open_conversation(&id);
        }
        if let Some(id) = delete {
            self.chat.confirm_delete = None;
            self.delete_conversation(&id);
        }
    }

    fn chat_main(&mut self, ui: &mut Ui) {
        let p = self.p;
        Panel::bottom("composer")
            .frame(
                Frame::new()
                    .fill(p.bg)
                    .inner_margin(Margin::symmetric(14, 10)),
            )
            .show(ui, |ui| column(ui, 880.0, |ui| self.composer(ui)));
        egui::CentralPanel::default()
            .frame(
                Frame::new()
                    .fill(p.bg)
                    .inner_margin(Margin::symmetric(14, 0)),
            )
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        column(ui, 880.0, |ui| {
                            ui.add_space(12.0);
                            if self.chat.conv.messages.is_empty() {
                                self.empty_state(ui);
                            } else {
                                self.messages(ui);
                            }
                            ui.add_space(16.0);
                        });
                    });
            });
    }

    fn empty_state(&mut self, ui: &mut Ui) {
        let p = self.p;
        ui.add_space(50.0);
        ui.vertical_centered(|ui| {
            ui.label(RichText::new(self.t("Cu ce te pot ajuta astăzi?", "How can I help today?")).size(30.0).semibold());
            ui.label(RichText::new(self.t("Scrie mai jos sau alege o idee. Totul rămâne pe calculatorul tău.", "Type below or pick an idea. Everything stays on your computer.")).size(16.0).color(p.muted));
            ui.add_space(16.0);
            let ideas: [(&str, &str); 4] = [
                (self.t("Explică-mi ceva simplu", "Explain something simply"), self.t("Explică-mi, pe înțelesul unui copil de 10 ani, cum funcționează internetul.", "Explain to me, like I'm 10, how the internet works.")),
                (self.t("Ajută-mă cu un e-mail", "Help me with an email"), self.t("Ajută-mă să scriu un e-mail politicos prin care cer o amânare a unui termen.", "Help me write a polite email asking for a deadline extension.")),
                (self.t("Rezumă și clarifică", "Summarise and clarify"), self.t("Voi lipi un text; rezumă-l în 5 puncte și spune-mi ce e neclar.", "I'll paste a text; summarise it in 5 points and tell me what's unclear.")),
                (self.t("Idei pentru o seară", "Ideas for an evening"), self.t("Dă-mi 5 idei de seri liniștite acasă, fără ecrane.", "Give me 5 ideas for quiet evenings at home, without screens.")),
            ];
            for (label, prompt) in ideas {
                if ui.add(egui::Button::new(RichText::new(label).size(16.0)).min_size(vec2(340.0, 44.0))).clicked() {
                    self.chat.input = prompt.to_string();
                    self.send_message();
                }
            }
        });
    }

    fn messages(&mut self, ui: &mut Ui) {
        let p = self.p;
        let tokens_word = self.t("tokeni", "tokens");
        let (w_thinking, w_thought) = (
            self.t("Se gândește…", "Thinking…"),
            self.t("Cum s-a gândit", "How it thought"),
        );
        let w_reading = self.t("Citesc întrebarea… {} s", "Reading your message… {} s");
        let w_block = self.t("Cod", "Code");
        let w_save = self.t("Salvează ca fișier", "Save as file");
        let w_preview = self.t("Previzualizare", "Preview");
        let w_ask = self.t(
            "Pagina se deschide în browser și poate rula scripturi scrise de model. Deschide-o doar dacă ai încredere în conținutul afișat mai sus.",
            "The page opens in your browser and may run scripts the model wrote. Open it only if you trust what you see above.",
        );
        let w_open_anyway = self.t("Deschide oricum", "Open anyway");
        let w_cancel = self.t("Renunță", "Cancel");
        let (w_stopped, w_copy, w_regen) = (
            self.t("(oprit înainte de final)", "(stopped before the end)"),
            self.t("Copiază", "Copy"),
            self.t("Regenerează", "Regenerate"),
        );
        let w_sources = self.t("Surse din documente", "Sources from your documents");
        let w_edit = self.t("Editează", "Edit");
        let w_branch = self.t("Ramifică de aici", "Branch from here");
        let w_usage = self.t("Consum", "Usage");
        let w_usage_help = self.t(
            "Arată tokenii, viteza și încărcarea calculatorului pentru acest răspuns",
            "Show the tokens, speed and machine load of this answer",
        );
        let w_speak = self.t("Citește cu voce tare", "Read aloud");
        let w_pin = self.t("Reține", "Remember");
        let w_unspeak = self.t("Oprește citirea", "Stop reading");
        let speaking = self.is_speaking();
        let last_user = self
            .chat
            .conv
            .messages
            .iter()
            .rposition(|m| m.role == "user");
        let generating = self
            .chat
            .generating
            .as_ref()
            .map(|g| g.started.elapsed().as_secs());
        let count = self.chat.conv.messages.len();
        let show_thinking = self.s.show_thinking;
        let mut cache = std::mem::take(&mut self.chat.cache);
        let mut action: Option<MsgAction> = None;
        if self.chat.heights_for != self.chat.conv.id {
            self.chat.heights.clear();
            self.chat.heights_for = self.chat.conv.id.clone();
        }
        self.chat.heights.resize(count, None);
        let width = ui.available_width() as u32;
        let view = ui.clip_rect();
        let margin = 800.0; // keep a screenful drawn above and below, so scrolling never shows blanks
        let mut heights = std::mem::take(&mut self.chat.heights);
        for (i, m) in self.chat.conv.messages.iter().enumerate() {
            let last = i + 1 == count;
            let streaming = last && generating.is_some();
            let key = HeightKey {
                len: m.content.len(),
                reasoning: m.reasoning.len(),
                width,
                flags: u8::from(streaming)
                    | u8::from(show_thinking) << 1
                    | u8::from(m.stopped) << 2
                    | u8::from(last) << 3
                    | u8::from(last_user == Some(i) && generating.is_none()) << 4,
            };
            let top = ui.cursor().min.y;
            if let Some((k, h)) = heights[i] {
                if k == key && (top + h < view.top() - margin || top > view.bottom() + margin) {
                    ui.add_space(h);
                    continue;
                }
            }
            if m.role == "user" {
                ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                    Frame::new()
                        .fill(p.bubble_user)
                        .corner_radius(CornerRadius::same(14))
                        .inner_margin(Margin::symmetric(14, 10))
                        .show(ui, |ui| {
                            ui.set_max_width(ui.available_width() * 0.82);
                            if !m.images.is_empty() {
                                ui.label(
                                    RichText::new(format!("🖼 × {}", m.images.len())).color(p.muted),
                                );
                            }
                            ui.add(
                                egui::Label::new(RichText::new(&m.content).size(16.0))
                                    .selectable(true),
                            );
                        });
                });
                if last_user == Some(i) && generating.is_none() {
                    ui.with_layout(Layout::right_to_left(Align::TOP), |ui| {
                        if ui.small_button(w_edit).clicked() {
                            action = Some(MsgAction::Edit);
                        }
                    });
                }
                ui.add_space(8.0);
            } else {
                ui.horizontal_top(|ui| {
                    avatar(ui, &p);
                    ui.vertical(|ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new("Colibri").semibold().color(p.accent));
                        if show_thinking && !m.reasoning.is_empty() {
                            let title = if streaming && m.content.is_empty() {
                                w_thinking
                            } else {
                                w_thought
                            };
                            egui::CollapsingHeader::new(RichText::new(title).color(p.muted))
                                .id_salt(("think", &m.role, i))
                                .default_open(streaming && m.content.is_empty())
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new(m.reasoning.trim())
                                            .italics()
                                            .color(p.muted)
                                            .size(14.5),
                                    );
                                });
                        }
                        if !m.content.is_empty() {
                            if too_nested(&m.content) {
                                // the markdown renderer needs time that grows steeply with nesting
                                ui.add(egui::Label::new(&m.content).selectable(true));
                            } else {
                                egui_commonmark::CommonMarkViewer::new()
                                    .show(ui, &mut cache, &m.content);
                            }
                        } else if streaming && m.reasoning.is_empty() {
                            ui.horizontal(|ui| {
                                crate::widgets::spinner(ui);
                                let secs = generating.unwrap_or(0);
                                let msg = w_reading.replacen("{}", &secs.to_string(), 1);
                                ui.label(RichText::new(msg).color(p.muted));
                            });
                        }
                        if m.stopped && !streaming {
                            ui.label(RichText::new(w_stopped).small().color(p.muted));
                        }
                        if !m.sources.is_empty() && !streaming {
                            egui::CollapsingHeader::new(
                                RichText::new(w_sources).small().color(p.muted),
                            )
                            .id_salt(("sources", i))
                            .show(ui, |ui| {
                                for (n, s) in m.sources.iter().enumerate() {
                                    ui.label(
                                        RichText::new(format!("[{}] {s}", n + 1))
                                            .small()
                                            .color(p.muted),
                                    );
                                }
                            });
                        }
                        if !streaming && (m.content.contains("```") || m.content.contains("~~~")) {
                            let blocks = companion_core::blocks::extract(&m.content);
                            for (bi, b) in blocks.iter().enumerate() {
                                ui.horizontal_wrapped(|ui| {
                                    let name = if b.lang.is_empty() { "text" } else { &b.lang };
                                    ui.label(
                                        RichText::new(format!("{} {}", w_block, bi + 1))
                                            .color(p.muted),
                                    );
                                    ui.label(RichText::new(name).monospace().color(p.muted));
                                    if ui.small_button(w_copy).clicked() {
                                        action = Some(MsgAction::Copy(b.code.clone()));
                                    }
                                    if ui.small_button(w_save).clicked() {
                                        action = Some(MsgAction::SaveBlock(b.clone()));
                                    }
                                    if b.is_previewable() && ui.small_button(w_preview).clicked() {
                                        action = Some(if b.is_active() {
                                            MsgAction::Ask(i, bi)
                                        } else {
                                            MsgAction::Preview(b.clone())
                                        });
                                    }
                                });
                                if self.chat.preview_ask == Some((i, bi)) {
                                    tinted(ui, p.bad_soft, p.bad, |ui| {
                                        ui.label(RichText::new(w_ask).color(p.bad));
                                        ui.horizontal(|ui| {
                                            if ui.button(w_open_anyway).clicked() {
                                                action = Some(MsgAction::Preview(b.clone()));
                                            }
                                            if ui.button(w_cancel).clicked() {
                                                action = Some(MsgAction::CancelAsk);
                                            }
                                        });
                                    });
                                }
                            }
                        }
                        if !streaming && !m.content.is_empty() {
                            ui.horizontal_wrapped(|ui| {
                                if ui.small_button(w_copy).clicked() {
                                    action = Some(MsgAction::Copy(m.content.clone()));
                                }
                                if ui.small_button(w_pin).clicked() {
                                    action = Some(MsgAction::Pin(m.content.clone()));
                                }
                                if speaking && last {
                                    if ui.small_button(w_unspeak).clicked() {
                                        action = Some(MsgAction::StopSpeaking);
                                    }
                                } else if ui.small_button(w_speak).clicked() {
                                    action = Some(MsgAction::Speak(m.content.clone()));
                                }
                                if !last && ui.small_button(w_branch).clicked() {
                                    action = Some(MsgAction::Branch(i));
                                }
                                if last && ui.small_button(w_regen).clicked() {
                                    action = Some(MsgAction::Regenerate);
                                }
                                if let Some(stats) = stats_text(m, tokens_word) {
                                    ui.label(RichText::new(stats).small().color(p.muted));
                                }
                                if let Some(st) = &m.stats
                                    && ui
                                        .small_button(w_usage)
                                        .on_hover_text(w_usage_help)
                                        .clicked()
                                {
                                    action = Some(MsgAction::Usage(Box::new(st.clone())));
                                }
                            });
                        }
                    });
                });
                ui.add_space(12.0);
            }
            heights[i] = Some((key, ui.cursor().min.y - top));
        }
        self.chat.heights = heights;
        self.chat.cache = cache;
        match action {
            Some(MsgAction::Copy(text)) => {
                ui.ctx().copy_text(text);
                self.toast(self.t("Copiat în clipboard", "Copied to the clipboard"));
            }
            Some(MsgAction::Usage(st)) => {
                self.meter.shown = Some(*st);
                self.meter.shown_history.clear();
                self.meter.open = true;
                self.meter.dismissed = false;
            }
            Some(MsgAction::Pin(text)) => self.pin_text(&text),
            Some(MsgAction::Speak(text)) => self.speak(&text),
            Some(MsgAction::StopSpeaking) => self.stop_speaking(),
            Some(MsgAction::Regenerate) => self.regenerate(),
            Some(MsgAction::Edit) => self.edit_last_message(),
            Some(MsgAction::Branch(i)) => self.branch_conversation(i),
            Some(MsgAction::SaveBlock(b)) => self.save_block(&b),
            Some(MsgAction::Preview(b)) => {
                self.chat.preview_ask = None;
                self.preview_block(&b);
            }
            Some(MsgAction::Ask(i, bi)) => self.chat.preview_ask = Some((i, bi)),
            Some(MsgAction::CancelAsk) => self.chat.preview_ask = None,
            None => {}
        }
    }

    fn composer(&mut self, ui: &mut Ui) {
        let p = self.p;
        if let Some(err) = self.chat.error.clone() {
            tinted(ui, p.bad_soft, p.bad, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(self.friendly_error(&err))
                            .color(p.bad)
                            .semibold(),
                    );
                    if err.kind == ErrorKind::Busy
                        && ui.button(self.t("Încearcă din nou", "Try again")).clicked()
                    {
                        self.regenerate_or_resend();
                    }
                    if ui.button(self.t("Închide", "Dismiss")).clicked() {
                        self.chat.error = None;
                    }
                });
                if !err.message.is_empty()
                    && !matches!(err.kind, ErrorKind::Unreachable | ErrorKind::ContextTooLong)
                {
                    ui.label(RichText::new(&err.message).small().color(p.muted));
                }
            });
            ui.add_space(6.0);
        }
        if !self.server_ready() {
            tinted(ui, p.warn_soft, p.warn, |ui| {
                ui.horizontal_wrapped(|ui| {
                    let loading =
                        matches!(self.health, Health::Loading(_)) || self.starting_since.is_some();
                    ui.label(
                        RichText::new(if loading {
                            self.t("Modelul se încarcă…", "The model is loading…")
                        } else {
                            self.t(
                                "Modelul nu rulează în prezent, deci poți doar citi conversațiile.",
                                "The model isn't running, so you can only read your conversations.",
                            )
                        })
                        .color(p.warn)
                        .semibold(),
                    );
                    if !loading
                        && self.coli.is_some()
                        && ui
                            .button(self.t("Pornește modelul", "Start the model"))
                            .clicked()
                    {
                        self.start_server();
                    }
                    if ui.button(self.t("Înapoi", "Back")).clicked() {
                        self.browse_history = false;
                    }
                });
            });
            ui.add_space(6.0);
        }

        if !self.chat.pending_images.is_empty() {
            let mut remove = None;
            ui.horizontal_wrapped(|ui| {
                for (i, (name, _, size)) in self.chat.pending_images.iter().enumerate() {
                    ui.label(
                        RichText::new(format!("🖼 {name} ({} KB)", size / 1024 + 1)).color(p.accent),
                    );
                    if ui.small_button("✕").clicked() {
                        remove = Some(i);
                    }
                }
            });
            if let Some(i) = remove {
                self.chat.pending_images.remove(i);
            }
        }
        if !self.chat.pending_docs.is_empty() {
            let mut remove = None;
            let total: usize = self
                .chat
                .pending_docs
                .iter()
                .map(|(_, t)| t.chars().count())
                .sum();
            ui.horizontal_wrapped(|ui| {
                for (i, (name, text)) in self.chat.pending_docs.iter().enumerate() {
                    ui.label(
                        RichText::new(format!("📄 {name} ({} KB)", text.len() / 1024 + 1))
                            .color(p.accent),
                    );
                    if ui.small_button("✕").clicked() {
                        remove = Some(i);
                    }
                }
                if self.chat.pending_docs.len() > 1
                    && ui
                        .small_button(self.t("Șterge toate", "Remove all"))
                        .clicked()
                {
                    remove = Some(usize::MAX);
                }
            });
            ui.label(
                RichText::new(format!(
                    "{} {} {} {}",
                    self.chat.pending_docs.len(),
                    self.t("fișiere,", "files,"),
                    total / 1000,
                    self.t("mii de caractere", "thousand characters")
                ))
                .small()
                .color(p.muted),
            );
            match remove {
                Some(usize::MAX) => self.chat.pending_docs.clear(),
                Some(i) => {
                    self.chat.pending_docs.remove(i);
                }
                None => {}
            }
        }
        let id = egui::Id::unique("composer-text");
        let generating = self.chat.generating.is_some();
        let can_type = self.server_ready();
        let focused = ui.ctx().memory(|m| m.has_focus(id));
        let mut send = false;
        if focused
            && can_type
            && ui.input(|i| {
                i.key_pressed(Key::Enter)
                    && !i.modifiers.shift
                    && !i.modifiers.command
                    && !i.modifiers.alt
            })
        {
            ui.input_mut(|i| i.consume_key(Modifiers::NONE, Key::Enter));
            send = true;
        }
        if can_type {
            self.style_bar(ui);
            self.format_toolbar(ui, id);
        }
        ui.horizontal(|ui| {
            let w = ui.available_width() - 120.0;
            let hint = if can_type {
                self.t(
                    "Scrie aici…  (Enter trimite · Shift+Enter rând nou)",
                    "Type here…  (Enter sends · Shift+Enter new line)",
                )
            } else {
                ""
            };
            let resp = ui.add_enabled(
                can_type,
                egui::TextEdit::multiline(&mut self.chat.input)
                    .id(id)
                    .desired_rows(2)
                    .desired_width(w)
                    .hint_text(hint)
                    .font(egui::TextStyle::Body)
                    .lock_focus(true),
            );
            if self.chat.focus_input && can_type {
                resp.request_focus();
                self.chat.focus_input = false;
            }
            if generating {
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new(self.t("Oprește", "Stop"))
                                .color(p.on_accent)
                                .semibold(),
                        )
                        .fill(p.bad)
                        .min_size(vec2(104.0, 52.0)),
                    )
                    .clicked()
                {
                    self.stop_generation();
                }
            } else {
                let enabled = can_type
                    && (!self.chat.input.trim().is_empty() || !self.chat.pending_docs.is_empty());
                ui.add_enabled_ui(enabled, |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(self.t("Trimite", "Send"))
                                    .color(p.on_accent)
                                    .semibold(),
                            )
                            .fill(p.accent)
                            .min_size(vec2(104.0, 52.0)),
                        )
                        .clicked()
                    {
                        send = true;
                    }
                });
            }
        });
        if send && (!self.chat.input.trim().is_empty() || !self.chat.pending_docs.is_empty()) {
            self.send_message();
            self.chat.focus_input = true;
        }
        ui.add_space(2.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(self.t("Stil:", "Style:")).color(p.muted));
            for (style, ro, en) in [
                (Style::Precise, "Precis", "Precise"),
                (Style::Balanced, "Echilibrat", "Balanced"),
                (Style::Creative, "Creativ", "Creative"),
            ] {
                let label = self.t(ro, en);
                if ui
                    .selectable_value(&mut self.s.style, style, label)
                    .clicked()
                {
                    let _ = self.s.save();
                }
            }
            ui.separator();
            ui.label(RichText::new(self.t("Gândire:", "Thinking:")).color(p.muted));
            for (mode, ro, en) in [
                (Thinking::Auto, "Automat", "Auto"),
                (Thinking::On, "Pornită", "On"),
                (Thinking::Off, "Oprită", "Off"),
            ] {
                let label = self.t(ro, en);
                if ui
                    .selectable_value(&mut self.s.thinking, mode, label)
                    .clicked()
                {
                    let _ = self.s.save();
                }
            }
        });
    }

    fn regenerate_or_resend(&mut self) {
        if self
            .chat
            .conv
            .messages
            .last()
            .is_some_and(|m| m.role == "user")
        {
            self.chat.conv.messages.push(StoredMessage {
                role: "assistant".into(),
                ..Default::default()
            });
            self.chat.error = None;
            self.start_generation();
        } else {
            self.regenerate();
        }
    }

    pub fn friendly_error(&self, e: &ChatError) -> String {
        match e.kind {
            ErrorKind::Unreachable => self.t("Nu se poate ajunge la server. Pornește modelul și încearcă din nou.", "I can't reach the server. Start the model and try again.").into(),
            ErrorKind::Busy => self.t("Serverul este ocupat sau încă se încarcă. Încearcă din nou peste câteva secunde.", "The server is busy or still loading. Try again in a few seconds.").into(),
            ErrorKind::Unauthorized => self.t("Serverul cere o cheie API. Adaug-o în Setări → Server.", "The server wants an API key. Add it in Settings → Server.").into(),
            ErrorKind::ContextTooLong => self.t("Conversația a devenit prea lungă pentru acest model. Începe una nouă (stânga sus).", "This conversation got too long for the model. Start a new one (top left).").into(),
            ErrorKind::BadRequest | ErrorKind::Server => self.t("Serverul a răspuns cu o eroare.", "The server answered with an error.").into(),
        }
    }
}

fn stats_text(m: &StoredMessage, tokens_word: &str) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(t) = m.tokens_per_second {
        parts.push(loc(format!("{t:.1} tok/s")));
    }
    if let Some(n) = m.completion_tokens {
        parts.push(format!("{n} {tokens_word}"));
    }
    (!parts.is_empty()).then(|| parts.join(" · "))
}

/// A small round badge with a "C", drawn with shapes (no image files, no emoji fonts).
fn avatar(ui: &mut Ui, p: &crate::theme::Palette) {
    let (rect, _) = ui.allocate_exact_size(vec2(30.0, 30.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 15.0, p.accent);
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        "C",
        egui::FontId::proportional(17.0),
        p.on_accent,
    );
}

/// Frame-cost measurements (`cargo test --release -p colibri-companion perf_ -- --ignored --nocapture`).
/// They only print timings; nothing here asserts a speed.
/// Whether the last search can be shown again: same text, same list, and not older than a few seconds.
fn search_cache_is_fresh(cache: Option<&SearchCache>, query: &str, stamp: (usize, u64)) -> bool {
    cache.is_some_and(|c| {
        c.query == query && c.stamp == stamp && c.at.elapsed() < Duration::from_secs(5)
    })
}

#[cfg(test)]
mod search_cache_tests {
    use super::*;

    fn cache(query: &str, stamp: (usize, u64), age: Duration) -> SearchCache {
        SearchCache {
            query: query.into(),
            stamp,
            at: Instant::now().checked_sub(age).unwrap_or_else(Instant::now),
            hits: Default::default(),
        }
    }

    #[test]
    fn the_search_is_reused_only_for_the_same_text_and_list() {
        let c = cache("rust", (3, 10), Duration::ZERO);
        assert!(search_cache_is_fresh(Some(&c), "rust", (3, 10)));
        assert!(!search_cache_is_fresh(Some(&c), "rus", (3, 10)));
        assert!(!search_cache_is_fresh(Some(&c), "rust", (4, 10)));
        assert!(!search_cache_is_fresh(Some(&c), "rust", (3, 11)));
        assert!(!search_cache_is_fresh(None, "rust", (3, 10)));
        let old = cache("rust", (3, 10), Duration::from_secs(6));
        assert!(!search_cache_is_fresh(Some(&old), "rust", (3, 10)));
    }
}

#[cfg(test)]
mod perf {
    use super::*;
    use companion_core::store::{Conversation, Settings};
    use std::time::Instant;

    fn app() -> CompanionApp {
        let ctx = egui::Context::default();
        CompanionApp::new(
            ctx,
            Settings::default(),
            Some("http://127.0.0.1:1".into()),
            false,
        )
    }

    fn reply(i: usize) -> String {
        format!(
            "## Answer {i}\n\nSome **markdown** with `code`, a list:\n\n- one\n- two\n- three\n\n```rust\nfn main() {{\n    println!(\"{i}\");\n}}\n```\n\n{}",
            "A sentence that makes the paragraph wrap over several lines in the window. ".repeat(6)
        )
    }

    fn conversation(n: usize) -> Conversation {
        let mut c = Conversation::new();
        c.id = "perf-bench".into();
        for i in 0..n {
            let (role, content) = if i % 2 == 0 {
                ("user", format!("Question number {i}: how does this work?"))
            } else {
                ("assistant", reply(i))
            };
            c.messages.push(StoredMessage {
                role: role.into(),
                content,
                ..Default::default()
            });
        }
        c
    }

    #[test]
    fn the_usage_panel_draws_while_writing_and_afterwards() {
        use companion_core::meter::{Kind, Meter, Sample};
        let mut a = app();
        a.chat.conv = conversation(3);
        let mut m = Meter::new(Some(4096));
        for i in 0..40 {
            m.delta(100 + i * 60, Kind::Answer, "word ");
        }
        m.sample(Sample {
            cpu_percent: 55.0,
            engine_cores: Some(6.5),
            engine_rss_gb: Some(9.0),
            ram_used_gb: 12.0,
            ram_total_gb: 16.0,
            gpu_util_percent: Some(30.0),
            vram_used_gb: Some(2.0),
            vram_total_gb: Some(8.0),
            ..Sample::default()
        });
        a.meter.run = Some((1, m.clone(), Instant::now()));
        a.meter.open = true;
        assert!(a.show_meter());
        frame(&mut a, 1400.0, 800.0);
        // afterwards: the finished figures, with and without server usage
        a.meter.run = None;
        a.meter.shown = Some(m.finish(2600, None));
        frame(&mut a, 1400.0, 800.0);
        // a narrow window and an empty stats record must not panic either
        a.meter.shown = Some(Default::default());
        frame(&mut a, 700.0, 500.0);
    }

    fn frame(app: &mut CompanionApp, w: f32, h: f32) -> f64 {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, vec2(w, h))),
            ..Default::default()
        };
        let ctx = app.ctx.clone();
        let t = Instant::now();
        let mut out = ctx.run_ui(input, |ui| app.chat_main(ui));
        out.textures_delta.clear();
        t.elapsed().as_secs_f64() * 1000.0
    }

    #[test]
    #[ignore]
    fn perf_chat_frame_cost() {
        for n in [100usize, 1000, 4000] {
            let mut a = app();
            a.chat.conv = conversation(n);
            let cold = frame(&mut a, 1000.0, 800.0);
            let mut warm = Vec::new();
            for _ in 0..20 {
                warm.push(frame(&mut a, 1000.0, 800.0));
            }
            warm.sort_by(|x, y| x.total_cmp(y));
            println!(
                "perf_chat n={n:>5}: first frame {cold:>9.1} ms, steady median {:>7.2} ms, p90 {:>7.2} ms",
                warm[warm.len() / 2],
                warm[warm.len() * 9 / 10]
            );
        }
    }

    #[test]
    #[ignore]
    fn perf_streaming_long_answer() {
        // the last message grows by a few words per frame, like a streamed answer
        let mut a = app();
        a.chat.conv = conversation(20);
        a.chat.generating = Some(crate::app::Generation {
            id: 1,
            cancel: companion_core::http::CancelToken::new(),
            started: Instant::now(),
        });
        let mut times = Vec::new();
        for step in 0..300 {
            if let Some(m) = a.chat.conv.messages.last_mut() {
                m.role = "assistant".into();
                m.content.push_str(
                    "another few words of the streamed answer, with `code` and **bold**. ",
                );
                if step % 40 == 0 {
                    m.content.push_str("\n\n- item\n- item\n\n");
                }
            }
            times.push(frame(&mut a, 1000.0, 800.0));
        }
        let at = |r: std::ops::Range<usize>| times[r.clone()].iter().sum::<f64>() / r.len() as f64;
        println!(
            "perf_stream: chars {} | mean frame ms: first 30 {:.2}, middle 30 {:.2}, last 30 {:.2}",
            a.chat.conv.messages.last().map_or(0, |m| m.content.len()),
            at(0..30),
            at(135..165),
            at(270..300)
        );
    }

    #[test]
    #[ignore]
    fn perf_sidebar_with_many_conversations() {
        let mut a = app();
        for i in 0..2000 {
            let mut c = Conversation::new();
            c.id = format!("c{i}");
            c.title = format!("Conversation title number {i}");
            c.updated = i as u64;
            a.chat.list.push(c);
        }
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                vec2(1000.0, 800.0),
            )),
            ..Default::default()
        };
        let mut v = Vec::new();
        for _ in 0..30 {
            let ctx = a.ctx.clone();
            let t = Instant::now();
            let mut out = ctx.run_ui(input.clone(), |ui| a.sidebar(ui));
            out.textures_delta.clear();
            v.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        v.sort_by(|x, y| x.total_cmp(y));
        println!(
            "perf_sidebar 2000 chats: median {:.2} ms, p90 {:.2} ms",
            v[15], v[27]
        );
    }

    #[test]
    fn side_panels_leave_the_conversation_room_at_every_width() {
        for w in [760.0_f32, 820.0, 900.0, 1120.0, 1600.0, 2560.0] {
            let (side, meter) = chat_panel_widths(w);
            let used = side
                + if meter {
                    (w * 0.26).clamp(220.0, 290.0)
                } else {
                    0.0
                };
            assert!(w - used >= 420.0, "{w}: only {} px left", w - used);
        }
        assert!(
            !chat_panel_widths(760.0).1,
            "the usage panel steps aside on a small window"
        );
        assert!(
            chat_panel_widths(1120.0).1,
            "and is there on the default size"
        );
    }
}

/// How deep a markdown list or quote may nest before the answer is shown as plain text: the
/// renderer's time grows steeply with depth (3000 nested items froze the window for half a
/// minute), and a real answer never nests more than a handful of levels.
const MAX_MARKDOWN_DEPTH: usize = 24;

/// True when some line nests lists or quotes deeper than [`MAX_MARKDOWN_DEPTH`], by markers
/// (`> > - 1. 1.`) or by indentation in front of a marker.
pub(crate) fn too_nested(md: &str) -> bool {
    md.lines().any(|line| {
        let indent = line.chars().take_while(|c| *c == ' ').count() / 4;
        let mut rest = line.trim_start();
        let mut depth = indent;
        while depth <= MAX_MARKDOWN_DEPTH {
            let next = if let Some(r) = rest.strip_prefix('>') {
                r
            } else if let Some(r) = ["- ", "* ", "+ "].iter().find_map(|m| rest.strip_prefix(m)) {
                r
            } else {
                let digits = rest.chars().take_while(char::is_ascii_digit).count();
                match rest.get(digits..) {
                    Some(r)
                        if (1..=9).contains(&digits)
                            && (r.starts_with(". ") || r.starts_with(") ")) =>
                    {
                        &r[2..]
                    }
                    _ => break,
                }
            };
            depth += 1;
            rest = next.trim_start();
        }
        depth > MAX_MARKDOWN_DEPTH
    })
}

#[cfg(test)]
mod nesting_tests {
    use super::too_nested;

    #[test]
    fn ordinary_answers_are_not_flagged() {
        assert!(!too_nested(
            "# T\n\n- a\n  - b\n    - c\n\n1. x\n2. y\n\n> quote\n> > inner\n\n```\n- - - - -\n```"
        ));
        assert!(!too_nested(&format!("{}x", "    ".repeat(5) + "- ")));
        assert!(!too_nested("3.14 is pi. 1.5 liters - ok"));
    }

    #[test]
    fn deep_nesting_is_flagged_however_it_is_built() {
        assert!(too_nested(&("1. ".repeat(30) + "x")));
        assert!(too_nested(&("> ".repeat(30) + "x")));
        assert!(too_nested(&("- ".repeat(30) + "x")));
        assert!(too_nested(&("> >".repeat(30) + "x")));
        assert!(too_nested(&("    ".repeat(40) + "- x")));
        assert!(too_nested(&("1. ".repeat(3_000) + "n")));
    }
}
