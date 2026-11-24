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
pub mod loader;
pub mod model;
pub mod ops;
pub mod plan;
pub mod quant;
