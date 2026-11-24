//! GPU backend for the engine.
//!
//! The crate is built so that everything except talking to a device can be checked on a machine
//! with no GPU:
//! * [`layout`] repacks GGML blocks into 32-bit-word layouts a shader can index;
//! * [`shaders`] holds the WGSL kernels (parsed and validated by `naga` in the tests);
//! * [`emu`] re-implements every kernel in plain Rust with the same blocking and the same
//!   reduction order, and the tests compare it with the engine's CPU kernels;
//! * [`plan`] has the dispatch-size and buffer-chunking arithmetic.
//!
//! The device runner (`GpuBackend`) is behind the `wgpu` cargo feature.
pub mod emu;
pub mod layout;
pub mod plan;
pub mod shaders;
#[cfg(feature = "wgpu")]
pub mod gpu;
