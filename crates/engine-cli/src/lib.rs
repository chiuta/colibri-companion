//! `corbul` — command-line front end for the companion engine.
//!
//! * [`template`] – a small, safe Jinja subset for chat templates, plus built-in fallbacks.
//! * [`http`]     – a strict, bounded HTTP/1.1 request reader and response writers.
//! * [`load`]     – loads a model folder or GGUF file together with its tokenizer and template.
//! * [`serve`]    – OpenAI-compatible and Ollama-style local server.
pub mod http;
pub mod load;
pub mod lorasrc;
pub mod auth;
pub mod anthropic;
pub mod constraint;
pub mod genloop;
pub mod metrics;
pub mod modelfile;
pub mod quotes;
pub mod args;
pub mod cmds;
pub mod runner;
pub mod registry;
pub mod shutdown;
pub mod serve;
pub mod template;
pub mod tools;
