//! `qubit_backend` is a module that incorporates the tools to simulate a quantum circuit based on
//! the `qubit` architecture. So superconducting qubits, cold atoms, trapped ions, etc...
pub mod circuit;
pub mod circuit_brute_force;
pub mod circuit_shuffle;
pub mod gates;

pub mod circuit_multithreading;

pub mod circuit_distributed;
pub mod circuit_distributed_cpu;

#[cfg(feature = "gpu")]
pub mod circuit_distributed_gpu;
#[cfg(feature = "gpu")]
pub mod circuit_gpu;
