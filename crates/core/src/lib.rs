//! Engine-facing logic of Colibri Companion.
//!
//! This crate never links or copies colibri code. It only drives colibri through
//! its public, documented surface: the `coli` launcher (JSON output) and the
//! OpenAI-compatible HTTP API of `coli serve`. That keeps the two programs
//! independent, each under its own licence.

pub mod access;
pub mod advisor;
pub mod attach;
pub mod blocks;
pub mod chat;
pub mod coli;
pub mod connect;
pub mod diskbench;
pub mod duel;
pub mod embed;
pub mod fetch;
pub mod flow_eval;
pub mod flows;
pub mod gpu;
pub mod history;
pub mod http;
pub mod legatorocr;
pub mod light_models;
pub mod locate;
pub mod mcp;
pub mod memory;
pub mod meter;
pub mod models;
pub mod net;
pub mod npu;
pub mod paths;
pub mod platforms;
pub mod proc;
pub mod prompts;
pub mod rag;
pub mod resources;
pub mod sheets;
pub mod store;
pub mod studio;
pub mod update;

pub use chat::{
    ChatClient, ChatError, ChatMessage, ChatRequest, ErrorKind, Health, StreamEvent, StreamSummary,
    Thinking,
};
pub use coli::{Coli, Hardware, InstallState, ModelInfo, ServerInfo, StatusReport};
pub use locate::PythonCmd;
pub use store::{
    Backend, Conversation, ConversationStore, Lang, Settings, StoredMessage, Style, Theme,
};
pub mod autotune;
pub mod autotune_runner;
pub mod capture;
pub mod estimate;
pub mod mailbox;
