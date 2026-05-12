//go:build js && wasm

// Package groth16 provides an experimental browser/WebGPU-accelerated Groth16
// prover surface for wasm targets.
//
// Scope of the current implementation:
//   - circuit compilation, setup, witness assignment, and solver stay in gnark
//   - Groth16 heavy MSMs are offloaded through a JS bridge to the browser
//     WebGPU runtime in this repository
//   - BSB22 commitment hint, commitment MSM, and PoK MSM work is wired through
//     the same WebGPU bridge
//
// The package mirrors gnark's accelerated backend layout without modifying the
// gnark repository. Host applications are expected to load
// `backend/accelerated/webgpu/groth16/bridge.js` before invoking Prove so the
// wasm code can call into the browser runtime through `syscall/js`.
package groth16
