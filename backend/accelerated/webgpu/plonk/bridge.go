//go:build js && wasm

package plonk

import (
	"syscall/js"

	webgpubridge "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/internal/bridge"
)

var bridgeClient = webgpubridge.NewClient("wnarkPlonkWebGPU", "webgpu plonk")

func jsUint8Array(src []byte) js.Value {
	return webgpubridge.JSUint8Array(src)
}

func jsObject() js.Value {
	return webgpubridge.JSObject()
}

func bridgeInit(curve string) error {
	return bridgeClient.Init(curve)
}

func bridgePrepareKey(curve string, payload js.Value) (string, error) {
	return bridgeClient.PrepareKey(curve, payload)
}

func bridgeMSMG1Slice(handle, vectorName string, start, count int, scalarsPacked []byte) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"msmG1",
		handle,
		vectorName,
		webgpubridge.JSUint8Array(scalarsPacked),
		start,
		count,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeMSMG1Batch(handle, vectorName string, start, termsPerInstance, instanceCount int, scalarsPacked []byte) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"msmG1Batch",
		handle,
		vectorName,
		webgpubridge.JSUint8Array(scalarsPacked),
		start,
		termsPerInstance,
		instanceCount,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeTransformQuotientCoset(curve string, valuesPacked, scalingPacked []byte, vectorCount, elementCount int) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"transformQuotientCoset",
		curve,
		webgpubridge.JSUint8Array(valuesPacked),
		webgpubridge.JSUint8Array(scalingPacked),
		vectorCount,
		elementCount,
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgePrewarmQuotientTransformDomain(curve string, elementCount int) error {
	_, err := bridgeClient.CallPromise("prewarmQuotientTransformDomain", curve, elementCount)
	return err
}
