//go:build js && wasm

// Package groth16 provides an experimental browser/WebGPU-accelerated Groth16
// prover surface for wasm targets.
//
// Scope of the current implementation:
//   - circuit compilation, setup, witness assignment, and solver stay in gnark
//   - Groth16 heavy MSMs are offloaded through a JS bridge to the browser
//     WebGPU runtime in this repository
//   - commitment hints are intentionally unsupported for now
//
// The package mirrors gnark's accelerated backend layout without modifying the
// gnark repository. Host applications are expected to load
// `backend/accelerated/webgpu/groth16/bridge.js` before invoking Prove so the
// wasm code can call into the browser runtime through `syscall/js`.
package groth16
