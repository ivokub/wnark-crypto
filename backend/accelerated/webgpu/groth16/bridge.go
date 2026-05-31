//go:build js && wasm

package groth16

import (
	"syscall/js"

	webgpubridge "github.com/ivokub/wnark-crypto/backend/accelerated/webgpu/internal/bridge"
)

var bridgeClient = webgpubridge.NewClient("wnarkGroth16WebGPU", "webgpu groth16")

type bridgeMSMBatchResult struct {
	G1ABytes []byte
	G1BBytes []byte
	G1KBytes []byte
	G2BBytes []byte
}

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

func bridgeMSMG1(handle, vectorName string, scalarsPacked []byte) ([]byte, error) {
	value, err := bridgeClient.CallPromise("msmG1", handle, vectorName, webgpubridge.JSUint8Array(scalarsPacked))
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgeMSMBatch(handle string, g1A, g1B, g1K []byte) (bridgeMSMBatchResult, error) {
	payload := webgpubridge.JSObject()
	payload.Set("g1A", webgpubridge.JSUint8Array(g1A))
	payload.Set("g1B", webgpubridge.JSUint8Array(g1B))
	payload.Set("g1K", webgpubridge.JSUint8Array(g1K))
	value, err := bridgeClient.CallPromise("msmBatch", handle, payload)
	if err != nil {
		return bridgeMSMBatchResult{}, err
	}
	result := bridgeMSMBatchResult{}
	if result.G1ABytes, err = webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value.Get("g1A")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G1BBytes, err = webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value.Get("g1B")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G1KBytes, err = webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value.Get("g1K")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	if result.G2BBytes, err = webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value.Get("g2B")); err != nil {
		return bridgeMSMBatchResult{}, err
	}
	return result, nil
}

func bridgeComputeHZMSMG1(handle string, aPacked, bPacked, cPacked []byte) ([]byte, error) {
	value, err := bridgeClient.CallPromise(
		"computeHZMSMG1",
		handle,
		webgpubridge.JSUint8Array(aPacked),
		webgpubridge.JSUint8Array(bPacked),
		webgpubridge.JSUint8Array(cPacked),
	)
	if err != nil {
		return nil, err
	}
	return webgpubridge.GoBytes(bridgeClient.ErrorPrefix, value)
}

func bridgePrewarmQuotientDomain(curve string, domainSize int) error {
	_, err := bridgeClient.CallPromise("prewarmQuotientDomain", curve, domainSize)
	return err
}
