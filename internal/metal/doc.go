// Package metal contains the experimental native WebGPU backend used for
// development-time smoke tests and ad hoc benchmarks.
//
// This code is intentionally kept internal because it is not a supported
// product surface. On current macOS setups the underlying gogpu/wgpu backend
// is unstable and can fail during shader/module creation or later resource
// lifetime operations. The browser/WebGPU path is the primary supported
// runtime for this repository.
package metal
