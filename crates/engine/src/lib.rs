//! Clean-room groundwork for a universal local inference engine.
//!
//! Written from the public file-format descriptions only (GGUF, safetensors, Hugging Face
//! `config.json`); no code from colibri or any other engine was read or copied.
//!
//! Layers, bottom up:
//! * [`gguf`]  – reads GGUF headers, metadata and tensor tables without loading the weights.
//! * [`quant`] – block-quantisation formats: sizes, dequantisation and fused dot products.
//! * [`arch`]  – turns a Hugging Face `config.json` or GGUF metadata into one [`arch::Spec`],
//!   and says honestly how far each architecture family is supported.
//! * [`plan`]  – decides what stays in RAM and what is streamed from disk, for any RAM budget.
pub mod arch;
pub mod gguf;
pub mod backend;
pub mod family;
pub mod hub;
pub mod hw;
pub mod pool;
pub mod simd;
pub mod ssm;
pub mod loader;
pub mod model;
pub mod mla;
pub mod moe;
pub mod ops;
pub mod qmatmul;
pub mod plan;
pub mod quant;
pub mod tokenizer;
pub mod eval;
pub mod grammar;
pub mod gguf_write;
pub mod lora;
pub mod quantize;
pub mod sampling;
pub mod sched;
pub mod spec;
pub mod confidence;
pub mod governor;
pub mod kvpolicy;
pub mod lang;
pub mod mesh;
pub mod quotelock;
pub mod receipt;
pub mod snapshot;
pub mod stream_layers;
pub mod search;
pub mod selfspec;
pub mod autoquant;
pub mod steer;
pub mod rag;
pub mod merge;
pub mod watermark;
