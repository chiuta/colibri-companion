//! The conversation screen: saved chats on the left, messages in the middle,
//! one comfortable box to type in at the bottom.

use crate::app::{CompanionApp, HeightKey};
use crate::theme::Semibold;
use crate::widgets::*;
use companion_core::chat::{ChatError, ErrorKind, Health, Thinking};
use companion_core::store::date_label;
use companion_core::{StoredMessage, Style};
use egui::{
    Align, CornerRadius, Frame, Key, Layout, Margin, Modifiers, Panel, RichText, Spinner, Ui, vec2,
};

enum MsgAction {
    Copy(String),
    Regenerate,
    Edit,
}

impl CompanionApp {
    pub fn show_chat(&mut self, ui: &mut Ui) {
        let p = self.p;
        Panel::left("sidebar")
            .default_size(270.0)
            .min_size(200.0)
            .frame(Frame::new().fill(p.panel).inner_margin(Margin::same(10)))
            .show(ui, |ui| self.sidebar(ui));
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
        let hits: Option<std::collections::HashSet<String>> = if query.is_empty() {
            None
        } else {
            Some(self.store.search(&query).into_iter().collect())
        };
        let searching = hits.is_some();
        let list: Vec<(String, String, u64)> = self
            .chat
            .list
            .iter()
            .filter(|c| hits.as_ref().is_none_or(|h| h.contains(&c.id)))
            .map(|c| (c.id.clone(), c.title.clone(), c.updated))
            .collect();
        let mut open = None;
        let mut delete = None;
        let mut rename: Option<(String, String)> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if list.is_empty() {
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
                for (id, title, updated) in &list {
                    let new_name_hint = self.t("Nume nou", "New name");
                    if let Some((rid, name)) =
                        self.chat.renaming.as_mut().filter(|(rid, _)| rid == id)
                    {
                        let edit_id = egui::Id::unique(("rename", rid.as_str()));
                        let first = !ui.ctx().memory(|m| m.has_focus(edit_id));
                        let chars = name.chars().count();
                        let r = ui.add_sized(
                            [ui.available_width(), 36.0],
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
                                self.t("Apasă încă o dată ca să ștergi", "Press again to delete"),
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
            ui.label(RichText::new(self.t("Cu ce te pot ajuta azi?", "How can I help today?")).size(30.0).semibold());
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
        let (w_stopped, w_copy, w_regen) = (
            self.t("(oprit înainte de final)", "(stopped before the end)"),
            self.t("Copiază", "Copy"),
            self.t("Regenerează", "Regenerate"),
        );
        let w_edit = self.t("Editează", "Edit");
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
                            egui_commonmark::CommonMarkViewer::new()
                                .show(ui, &mut cache, &m.content);
                        } else if streaming && m.reasoning.is_empty() {
                            ui.horizontal(|ui| {
                                ui.add(Spinner::new());
                                let secs = generating.unwrap_or(0);
                                let msg = w_reading.replacen("{}", &secs.to_string(), 1);
                                ui.label(RichText::new(msg).color(p.muted));
                            });
                        }
                        if m.stopped && !streaming {
                            ui.label(RichText::new(w_stopped).small().color(p.muted));
                        }
                        if !streaming && !m.content.is_empty() {
                            ui.horizontal(|ui| {
                                if ui.small_button(w_copy).clicked() {
                                    action = Some(MsgAction::Copy(m.content.clone()));
                                }
                                if last && ui.small_button(w_regen).clicked() {
                                    action = Some(MsgAction::Regenerate);
                                }
                                if let Some(stats) = stats_text(m, tokens_word) {
                                    ui.label(RichText::new(stats).small().color(p.muted));
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
            Some(MsgAction::Regenerate) => self.regenerate(),
            Some(MsgAction::Edit) => self.edit_last_message(),
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
                                "Modelul nu rulează acum, deci poți doar citi conversațiile.",
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
                let enabled = can_type && !self.chat.input.trim().is_empty();
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
        if send && !self.chat.input.trim().is_empty() {
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
            ErrorKind::Unreachable => self.t("Nu pot ajunge la server. Pornește modelul și încearcă din nou.", "I can't reach the server. Start the model and try again.").into(),
            ErrorKind::Busy => self.t("Serverul e ocupat sau încă se încarcă. Încearcă din nou în câteva secunde.", "The server is busy or still loading. Try again in a few seconds.").into(),
            ErrorKind::Unauthorized => self.t("Serverul cere o cheie API. Adaug-o în Setări → Server.", "The server wants an API key. Add it in Settings → Server.").into(),
            ErrorKind::ContextTooLong => self.t("Conversația a devenit prea lungă pentru acest model. Începe una nouă (stânga sus).", "This conversation got too long for the model. Start a new one (top left).").into(),
            ErrorKind::BadRequest | ErrorKind::Server => self.t("Serverul a răspuns cu o eroare.", "The server answered with an error.").into(),
        }
    }
}

fn stats_text(m: &StoredMessage, tokens_word: &str) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(t) = m.tokens_per_second {
        parts.push(format!("{t:.1} tok/s"));
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
