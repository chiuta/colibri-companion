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
// Style lints that newer clippy releases added and that would force a rewrite of hot loops whose current shape
// is deliberate (index loops over several parallel arrays, fixed-size chunking that the compiler already unrolls).
#![allow(unknown_lints)]
#![allow(
    clippy::chunks_exact_to_as_chunks,
    clippy::needless_range_loop,
    clippy::needless_late_init,
    clippy::single_element_loop
)]
pub mod arch;
pub mod audio;
pub mod autoquant;
pub mod backend;
pub mod bert;
pub mod confidence;
pub mod eval;
pub mod family;
pub mod gguf;
pub mod gguf_write;
pub mod governor;
pub mod grammar;
pub mod hub;
pub mod hw;
pub mod hybrid;
pub mod imageio;
pub mod kvpolicy;
pub mod lang;
pub mod loader;
pub mod lora;
pub mod merge;
pub mod mesh;
pub mod mla;
pub mod model;
pub mod moe;
pub mod ops;
pub mod plan;
pub mod pool;
pub mod qmatmul;
pub mod quant;
pub mod quantize;
pub mod quotelock;
pub mod rag;
pub mod receipt;
pub mod sampling;
pub mod sched;
pub mod search;
pub mod selfspec;
pub mod simd;
pub mod snapshot;
pub mod spec;
pub mod ssm;
pub mod steer;
pub mod stream_layers;
pub mod tokenizer;
pub mod vision;
pub mod watermark;
pub mod wordpiece;
