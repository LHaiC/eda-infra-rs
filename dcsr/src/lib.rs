//! # DCSR: Dynamic Compressed Sparse Row library
//!
//!
pub mod flatmem;
pub mod dcsr;
pub mod policy;

pub use flatmem::{FlatMem, FlatStorage};
pub use dcsr::DynamicCSR;
pub use policy::{MemPolicy, VanillaLogPolicy, PowerOfTwoSlabPolicy};
