//! Clean-room groundwork for an expert-streaming inference engine.
//!
//! Written from public format descriptions (the safetensors header layout) and from first
//! principles about caching; no colibri source is used. It is a *stage one*: it can read a
//! model's weight files, find where every expert lives, and replay a routing trace against cache
//! policies to measure how much a smarter policy could ever gain. It does not run a model.

pub mod expert_index;
pub mod ioplan;
pub mod policy;
pub mod prefetch;
pub mod safetensors;
pub mod skew;
pub mod speculative;
pub mod trace;
