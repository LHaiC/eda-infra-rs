//! # DCSR: Dynamic Compressed Sparse Row library
//!
//! 
pub mod flatmem;
pub mod dcsr;
pub mod policy;
pub mod delta_uvec;

pub use flatmem::FlatMem;
pub use dcsr::DynamicCSR;
pub use policy::{MemPolicy, VanillaLogPolicy, RelocationOp};
pub use delta_uvec::DeltaUVec;