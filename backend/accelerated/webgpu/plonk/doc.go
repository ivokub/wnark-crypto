//go:build js && wasm

// Package plonk provides the browser/WASM entry point for an experimental
// WebGPU-accelerated PLONK prover.
//
// The current scaffold is intentionally conservative: BN254 is wired end to
// end and delegates proving to gnark's native PLONK implementation. This keeps
// serialization, solving, transcript logic, and verification byte-compatible
// while giving us a local seam for replacing prover phases with WebGPU kernels.
package plonk
