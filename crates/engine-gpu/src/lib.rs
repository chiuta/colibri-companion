//! GPU backend for the engine.
//!
//! The crate is built so that everything except talking to a device can be checked on a machine
//! with no GPU:
//! * [`layout`] repacks GGML blocks into 32-bit-word layouts a shader can index;
//! * [`shaders`] holds the WGSL kernels (parsed and validated by `naga` in the tests);
//! * [`emu`] re-implements every kernel in plain Rust with the same blocking and the same
//!   reduction order, and the tests compare it with the engine's CPU kernels;
//! * [`plan`] has the dispatch-size and buffer-chunking arithmetic;
//! * [`emubackend`] is the device backend with those kernels run on the host, so the whole offload
//!   path (`Model::with_backend`) is testable with no GPU;
//! * [`caps`] describes what a device offers against what the kernels need, and picks an adapter;
//! * [`offload`] predicts how many layers fit a device, and how long a cache it can hold;
//! * [`timing`] wraps any backend with per-operation timing;
//! * [`report`] is the capability report for `doctor`, [`runtime`] opens the best backend and
//!   explains a CPU fallback.
//!
// Newer clippy wants as_chunks for fixed-size blocks; the block layouts here are written out on purpose.
#![allow(unknown_lints)]
#![allow(clippy::chunks_exact_to_as_chunks)]
//! The device runner (`GpuBackend`) is behind the `wgpu` cargo feature.
pub mod caps;
pub mod emu;
pub mod emubackend;
#[cfg(feature = "wgpu")]
pub mod gpu;
pub mod layout;
pub mod offload;
pub mod plan;
pub mod report;
pub mod runtime;
pub mod shaders;
pub mod timing;
