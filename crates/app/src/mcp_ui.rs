//! MCP tools: a window to add and start tool servers, the approval card shown before every tool
//! call, and the glue that lets a model ask for a tool and see its answer.

use crate::app::{CompanionApp, Event};
use crate::widgets::*;
use companion_core::mcp::{self, Client, Registry, Tool, ToolRequest};
use companion_core::paths;
use egui::{Align2, RichText, vec2};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Default)]
pub struct McpState {
    pub open: bool,
    pub reg: Registry,
    pub loaded: bool,
    pub clients: HashMap<String, Arc<Client>>,
    pub tools: Vec<(String, Tool)>,
    pub starting: HashSet<String>,
    pub new_name: String,
    pub new_cmd: String,
    pub message: String,
    /// A call the model asked for, waiting for the person's answer.
    pub approval: Option<ToolRequest>,
    /// Tools the person allowed for the rest of this conversation: (server, tool).
    pub trusted: HashSet<(String, String)>,
    /// Tool calls made since the person's last message.
    pub rounds: usize,
    pub calling: bool,
    pub manual_server: String,
    pub manual_tool: String,
    pub manual_args: String,
    pub manual_out: String,
}

fn pretty(v: &serde_json::Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_default()
}

impl CompanionApp {
    pub fn open_mcp(&mut self) {
        self.load_mcp();
        self.mcp.open = true;
    }

    fn load_mcp(&mut self) {
        if !self.mcp.loaded {
            self.mcp.reg = Registry::load(&paths::mcp_file());
            self.mcp.loaded = true;
        }
    }

    fn save_mcp(&mut self) {
        if let Err(e) = self.mcp.reg.save(&paths::mcp_file()) {
            self.toast(format!(
                "{}: {e}",
                self.t("Nu am putut salva serverele", "Could not save the servers")
            ));
        }
    }

    pub fn mcp_start(&mut self, name: &str) {
        let Some(cfg) = self
            .mcp
            .reg
            .servers
            .iter()
            .find(|s| s.name == name)
            .cloned()
        else {
            return;
        };
        if self.mcp.clients.contains_key(name) || !self.mcp.starting.insert(name.to_string()) {
            return;
        }
        self.mcp.message.clear();
        let n = name.to_string();
        self.spawn(move |tx| {
            let result = Client::start(&cfg).map(Box::new);
            let _ = tx.send(Event::McpStarted { name: n, result });
        });
    }

    pub fn mcp_stop(&mut self, name: &str) {
        self.mcp.clients.remove(name); // dropping the client ends the program
        self.mcp.tools.retain(|(s, _)| s != name);
    }

    pub fn mcp_started(&mut self, name: String, result: Result<Box<(Client, Vec<Tool>)>, String>) {
        self.mcp.starting.remove(&name);
        match result {
            Ok(b) => {
                let (client, tools) = *b;
                self.mcp.tools.retain(|(s, _)| s != &name);
                self.mcp
                    .tools
                    .extend(tools.into_iter().map(|t| (name.clone(), t)));
                self.mcp.clients.insert(name, Arc::new(client));
            }
            Err(e) => {
                self.mcp.message = format!("«{name}»: {e}");
            }
        }
    }

    /// The part of the system prompt that tells the model about the running tools.
    pub fn mcp_system_note(&self) -> String {
        mcp::system_note(&self.mcp.tools)
    }

    /// After an answer: if it ends with a tool request, ask the person (or run it when they have
    /// already allowed that tool for this conversation).
    pub fn mcp_after_answer(&mut self) {
        let Some(last) = self.chat.conv.messages.last() else {
            return;
        };
        if last.role != "assistant" || last.stopped || self.mcp.clients.is_empty() {
            return;
        }
        let Some(req) = mcp::parse_tool_request(&last.content) else {
            return;
        };
        if !self.mcp.clients.contains_key(&req.server) {
            return;
        }
        if self.mcp.rounds >= mcp::MAX_TOOL_ROUNDS {
            self.toast(self.t(
                "Prea multe apeluri de unelte la rând; m-am oprit.",
                "Too many tool calls in a row; I stopped.",
            ));
            return;
        }
        if self
            .mcp
            .trusted
            .contains(&(req.server.clone(), req.tool.clone()))
        {
            self.mcp_run(req);
        } else {
            self.mcp.approval = Some(req);
        }
    }

    fn mcp_run(&mut self, req: ToolRequest) {
        let Some(client) = self.mcp.clients.get(&req.server).cloned() else {
            return;
        };
        self.mcp.rounds += 1;
        self.mcp.calling = true;
        self.spawn(move |tx| {
            let result = client.call(&req.tool, req.arguments.clone());
            let _ = tx.send(Event::McpResult { req, result });
        });
    }

    pub fn mcp_result(&mut self, req: ToolRequest, result: Result<String, String>) {
        self.mcp.calling = false;
        if self.chat.generating.is_some() {
            return;
        }
        let text = mcp::result_message(&req, &result);
        self.push_user_turn(text, Vec::new());
        let n = self.chat.conv.messages.len();
        if n >= 2 {
            let shown = match &result {
                Ok(_) => self.t("Rezultat de la unealta {}", "Result from tool {}"),
                Err(_) => self.t("Eroare de la unealta {}", "Error from tool {}"),
            }
            .replacen("{}", &format!("{}/{}", req.server, req.tool), 1);
            self.chat.conv.messages[n - 2].display = shown;
        }
    }

    pub fn mcp_manual_result(&mut self, result: Result<String, String>) {
        self.mcp.calling = false;
        self.mcp.manual_out = match result {
            Ok(t) | Err(t) => t,
        };
    }

    pub fn mcp_windows(&mut self, ctx: &egui::Context) {
        self.mcp_approval_window(ctx);
        if !self.mcp.open {
            return;
        }
        let p = self.p;
        let mut open = true;
        let hint_name = self.t(
            "Nume scurt (litere și cifre)",
            "Short name (letters and digits)",
        );
        let hint_cmd = self.t(
            "Comanda, de ex. npx -y @modelcontextprotocol/server-filesystem /calea/ta",
            "The command, e.g. npx -y @modelcontextprotocol/server-filesystem /your/path",
        );
        let mut add = false;
        let mut start: Option<String> = None;
        let mut stop: Option<String> = None;
        let mut remove: Option<String> = None;
        let mut run_manual = false;
        egui::Window::new(self.t("Unelte MCP", "MCP tools"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .default_size(fit(ctx, 760.0, 640.0))
            .collapsible(false)
            .show(ctx, |ui| {
                tinted(ui, p.warn_soft, p.warn, |ui| {
                    ui.label(
                        self.t(
                            "Un server MCP este un program care rulează pe calculatorul tău, cu drepturile tale. Adaugă doar programe în care ai încredere. Nu pornește singur: îl pornești tău, iar fiecare apel de unealtă îți solicită acordul.",
                            "An MCP server is a program that runs on your computer with your rights. Add only programs you trust. It never starts by itself: you start it, and every tool call asks for your approval.",
                        ),
                    );
                });
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.mcp.new_name)
                            .hint_text(hint_name)
                            .desired_width(200.0),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut self.mcp.new_cmd)
                            .hint_text(hint_cmd)
                            .desired_width(ui.available_width() - 90.0),
                    );
                    if secondary_button(ui, self.t("Adaugă", "Add")).clicked() {
                        add = true;
                    }
                });
                if !self.mcp.message.is_empty() {
                    ui.label(RichText::new(&self.mcp.message).color(p.bad));
                }
                ui.separator();
                if self.mcp.reg.servers.is_empty() {
                    muted(ui, &p, self.t("Niciun server adăugat încă.", "No server added yet."));
                }
                for s in &self.mcp.reg.servers {
                    let running = self.mcp.clients.contains_key(&s.name);
                    let starting = self.mcp.starting.contains(&s.name);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(&s.name).strong());
                        ui.label(
                            RichText::new(format!("{} {}", s.command, s.args.join(" ")))
                                .monospace()
                                .small()
                                .color(p.muted),
                        );
                    });
                    ui.horizontal(|ui| {
                        if starting {
                            crate::widgets::spinner(ui);
                            ui.label(self.t("Pornește…", "Starting…"));
                        } else if running {
                            pill(ui, self.t("pornit", "running"), p.ok, p.accent_soft);
                            let n = self.mcp.tools.iter().filter(|(sv, _)| sv == &s.name).count();
                            ui.label(
                                self.t("{} unelte", "{} tools").replacen("{}", &n.to_string(), 1),
                            );
                            if ui.small_button(self.t("Oprește", "Stop")).clicked() {
                                stop = Some(s.name.clone());
                            }
                        } else if ui.button(self.t("Pornește", "Start")).clicked() {
                            start = Some(s.name.clone());
                        }
                        if !running && !starting && ui.small_button(self.t("Scoate", "Remove")).clicked()
                        {
                            remove = Some(s.name.clone());
                        }
                    });
                    for (sv, t) in &self.mcp.tools {
                        if sv == &s.name {
                            ui.horizontal(|ui| {
                                ui.add_space(16.0);
                                if ui
                                    .small_button(t.name.as_str())
                                    .on_hover_text(&t.description)
                                    .clicked()
                                {
                                    self.mcp.manual_server = sv.clone();
                                    self.mcp.manual_tool = t.name.clone();
                                    self.mcp.manual_args = pretty(&serde_json::json!({}));
                                }
                                ui.label(
                                    RichText::new(&t.description).small().color(p.muted),
                                );
                            });
                        }
                    }
                    ui.add_space(4.0);
                }
                if !self.mcp.manual_tool.is_empty() {
                    ui.separator();
                    ui.label(
                        RichText::new(format!(
                            "{} {}/{}",
                            self.t("Rulează o unealtă:", "Run a tool:"),
                            self.mcp.manual_server,
                            self.mcp.manual_tool
                        ))
                        .strong(),
                    );
                    ui.add(
                        egui::TextEdit::multiline(&mut self.mcp.manual_args)
                            .code_editor()
                            .desired_rows(3)
                            .desired_width(f32::INFINITY),
                    );
                    ui.horizontal(|ui| {
                        if primary_button(ui, &p, self.t("Rulează", "Run")).clicked()
                            && !self.mcp.calling
                        {
                            run_manual = true;
                        }
                        if self.mcp.calling {
                            crate::widgets::spinner(ui);
                        }
                        if !self.mcp.manual_out.is_empty()
                            && ui
                                .button(self.t("Plasează rezultatul în mesaj", "Put the result in the message"))
                                .clicked()
                        {
                            let out = self.mcp.manual_out.clone();
                            let tool = format!("{}/{}", self.mcp.manual_server, self.mcp.manual_tool);
                            self.chat.input.push_str(&format!(
                                "{}\n{}\n",
                                companion_core::attach::fenced(&tool, &out),
                                ""
                            ));
                        }
                    });
                    if !self.mcp.manual_out.is_empty() {
                        egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                            ui.label(RichText::new(&self.mcp.manual_out).monospace().small());
                        });
                    }
                }
            });
        if add {
            let (n, c) = (self.mcp.new_name.clone(), self.mcp.new_cmd.clone());
            match self.mcp.reg.add(&n, &c) {
                Ok(()) => {
                    self.mcp.new_name.clear();
                    self.mcp.new_cmd.clear();
                    self.mcp.message.clear();
                    self.save_mcp();
                }
                Err(_) => {
                    self.mcp.message = self
                        .t(
                            "Nume nevalid sau deja folosit, sau comandă goală.",
                            "Invalid or already used name, or an empty command.",
                        )
                        .to_string();
                }
            }
        }
        if let Some(n) = start {
            self.mcp_start(&n);
        }
        if let Some(n) = stop {
            self.mcp_stop(&n);
        }
        if let Some(n) = remove {
            self.mcp.reg.remove(&n);
            self.save_mcp();
        }
        if run_manual {
            match serde_json::from_str::<serde_json::Value>(&self.mcp.manual_args) {
                Ok(args) => {
                    if let Some(client) = self.mcp.clients.get(&self.mcp.manual_server).cloned() {
                        let tool = self.mcp.manual_tool.clone();
                        self.mcp.calling = true;
                        self.spawn(move |tx| {
                            let _ = tx.send(Event::McpManual(client.call(&tool, args)));
                        });
                    }
                }
                Err(_) => {
                    self.mcp.manual_out = self
                        .t(
                            "Argumentele nu sunt JSON valid.",
                            "The arguments are not valid JSON.",
                        )
                        .to_string();
                }
            }
        }
        if !open {
            self.mcp.open = false;
        }
    }

    fn mcp_approval_window(&mut self, ctx: &egui::Context) {
        let Some(req) = self.mcp.approval.clone() else {
            return;
        };
        let p = self.p;
        let mut decision: Option<u8> = None; // 1 allow, 2 allow for this chat, 0 deny
        let mut open = true;
        egui::Window::new(self.t("Permiți această unealtă?", "Allow this tool?"))
            .open(&mut open)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .collapsible(false)
            .default_width(520.0)
            .show(ctx, |ui| {
                ui.label(
                    self.t(
                        "Modelul cere să ruleze:",
                        "The model asks to run:",
                    ),
                );
                ui.label(
                    RichText::new(format!("{}/{}", req.server, req.tool))
                        .strong()
                        .size(18.0),
                );
                egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                    ui.label(RichText::new(pretty(&req.arguments)).monospace().small());
                });
                muted(
                    ui,
                    &p,
                    self.t(
                        "Unealta poate citi sau modifica elemente de pe acest calculator, în funcție de modul în care a fost concepută. Verifică argumentele.",
                        "The tool can read or change things on this computer, depending on how it was made. Check the arguments.",
                    ),
                );
                ui.horizontal(|ui| {
                    if primary_button(ui, &p, self.t("Permite o dată", "Allow once")).clicked() {
                        decision = Some(1);
                    }
                    if ui
                        .button(self.t("Permite în această conversație", "Allow in this conversation"))
                        .clicked()
                    {
                        decision = Some(2);
                    }
                    if ui.button(self.t("Refuză", "Deny")).clicked() {
                        decision = Some(0);
                    }
                });
            });
        if !open {
            decision = Some(0);
        }
        match decision {
            Some(0) => {
                self.mcp.approval = None;
                self.mcp.rounds = 0;
                self.toast(self.t("Apel refuzat.", "Call denied."));
            }
            Some(n) => {
                self.mcp.approval = None;
                if n == 2 {
                    self.mcp
                        .trusted
                        .insert((req.server.clone(), req.tool.clone()));
                }
                self.mcp_run(req);
            }
            None => {}
        }
    }
}
